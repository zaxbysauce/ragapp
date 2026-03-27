"""Authentication, CSRF, and toggle utilities."""

import hashlib
import hmac
import logging
import secrets
import threading
import time
from typing import Callable, Dict

import redis
from fastapi import Depends, Header, HTTPException, Request, Response

from app.config import settings
from app.services.secret_manager import SecretManager

logger = logging.getLogger("security")
CSRF_COOKIE_NAME = "X-CSRF-Token"


class _InMemoryCSRFStore:
    """Thread-safe in-memory fallback for CSRF tokens when Redis is unavailable."""

    def __init__(self, ttl: int = 900) -> None:
        self.ttl = ttl
        self._store: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    def setex(self, key: str, ttl: int, value: str) -> None:
        with self._lock:
            self._store[key] = (value, time.time() + ttl)

    def get(self, key: str) -> str | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expiry = entry
            if time.time() > expiry:
                del self._store[key]
                return None
            return value

    def expire(self, key: str, ttl: int) -> None:
        with self._lock:
            if key in self._store:
                value, _ = self._store[key]
                self._store[key] = (value, time.time() + ttl)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def ping(self) -> bool:
        return True


class CSRFManager:
    def __init__(self, redis_url: str, ttl: int = 900) -> None:
        self.ttl = ttl
        self._redis: redis.Redis | None = None
        self._fallback_store: _InMemoryCSRFStore | None = None
        self._use_fallback = False
        self._lock = threading.Lock()

        try:
            self._redis = redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            logger.info("CSRFManager connected to Redis successfully")
        except redis.RedisError as exc:
            logger.warning("Redis unavailable for CSRF, using in-memory fallback: %s", exc)
            self._use_fallback = True
            self._fallback_store = _InMemoryCSRFStore(ttl=ttl)

    def _get_store(self):
        """Returns the active store (Redis or in-memory fallback)."""
        if self._use_fallback and self._fallback_store:
            return self._fallback_store
        if self._redis:
            return self._redis
        raise HTTPException(status_code=503, detail="CSRF storage unavailable")

    def _check_redis_available(self) -> bool:
        """Check if Redis is available, switch back from fallback if recovered."""
        with self._lock:
            if self._use_fallback and self._redis:
                try:
                    self._redis.ping()
                    logger.info("Redis recovered, switching from in-memory fallback to Redis")
                    self._use_fallback = False
                    self._fallback_store = None
                    return True
                except redis.RedisError:
                    pass
            return not self._use_fallback

    def generate_token(self) -> str:
        self._check_redis_available()
        store = self._get_store()
        token = secrets.token_urlsafe(16)
        key = f"csrf:{token}"
        try:
            store.setex(key, self.ttl, "1")
        except Exception as exc:
            logger.error("Storage error during token generation: %s", exc)
            raise HTTPException(status_code=503, detail="CSRF storage unavailable")
        return token

    def validate_token(self, token: str) -> bool:
        if not token:
            return False
        self._check_redis_available()
        store = self._get_store()
        key = f"csrf:{token}"
        try:
            exists = store.get(key)
        except Exception as exc:
            logger.error("Storage error during token validation: %s", exc)
            raise HTTPException(status_code=503, detail="CSRF storage unavailable")
        if exists:
            try:
                store.delete(key)
            except Exception:
                logger.warning("Failed to delete CSRF token after use")
            return True
        return False

    def revoke_token(self, token: str) -> None:
        store = self._get_store()
        try:
            store.delete(f"csrf:{token}")
        except Exception:
            logger.warning("Failed to revoke CSRF token (storage error)")


def get_csrf_manager(request: Request) -> CSRFManager:
    manager = getattr(request.app.state, "csrf_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="CSRF service unavailable")
    return manager


def require_scope(scope: str) -> Callable:
    def dependency(
        authorization: str = Header(None),
        x_scopes: str = Header(""),
    ) -> dict[str, str]:
        # Default token that must be changed before auth works
        DEFAULT_TOKEN = "admin-secret-token"
        
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header missing")
        if not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        parts = authorization.split(" ", 1)
        if len(parts) < 2 or not parts[1].strip():
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        token = parts[1].strip()
        scopes = [s.strip().lower() for s in x_scopes.split(",") if s.strip()]
        if scope.lower() not in scopes:
            raise HTTPException(status_code=403, detail="Missing required scope")

        # Encode to bytes for safe comparison (prevents Unicode TypeError)
        try:
            token_bytes = token.encode("utf-8")
        except (UnicodeEncodeError, AttributeError):
            raise HTTPException(status_code=400, detail="Invalid token encoding")
        admin_token_bytes = settings.admin_secret_token.encode("utf-8")
        default_bytes = DEFAULT_TOKEN.encode("utf-8") if isinstance(DEFAULT_TOKEN, str) else DEFAULT_TOKEN

        if secrets.compare_digest(token_bytes, default_bytes):
            raise HTTPException(status_code=403, detail="Unauthorized token")
        if not secrets.compare_digest(token_bytes, admin_token_bytes):
            raise HTTPException(status_code=403, detail="Unauthorized token")
        return {"user_id": token}

    return dependency


def csrf_protect(
    request: Request,
    x_csrf_token: str = Header(""),
) -> str:
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return x_csrf_token
    csrf_manager = get_csrf_manager(request)
    cookie = request.cookies.get(CSRF_COOKIE_NAME)
    if not cookie or not x_csrf_token or cookie != x_csrf_token:
        raise HTTPException(status_code=403, detail="CSRF token missing or mismatch")
    if not csrf_manager.validate_token(x_csrf_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    return x_csrf_token


def issue_csrf_token(response: Response, csrf_manager: CSRFManager) -> str:
    token = csrf_manager.generate_token()
    response.set_cookie(
        CSRF_COOKIE_NAME,
        token,
        max_age=settings.csrf_token_ttl,
        samesite="strict",
        secure=not getattr(settings, "debug", False),
        httponly=False,
    )
    return token


def require_auth(
    authorization: str = Header(None),
) -> dict:
    """Simple Bearer token auth. Validates Bearer token against admin_secret_token.
    
    SECURITY: Rejects the default token 'admin-secret-token' to prevent
    unauthorized access when admin hasn't configured a custom token.
    """
    # Default token that must be changed before auth works
    DEFAULT_TOKEN = "admin-secret-token"
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization scheme")
    parts = authorization.split(" ", 1)
    if len(parts) < 2 or not parts[1].strip():
        raise HTTPException(status_code=401, detail="Token missing")
    token = parts[1].strip()

    # Encode to bytes for safe comparison (prevents Unicode TypeError)
    try:
        token_bytes = token.encode("utf-8")
    except (UnicodeEncodeError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid token encoding")
    admin_token_bytes = settings.admin_secret_token.encode("utf-8")
    default_bytes = DEFAULT_TOKEN.encode("utf-8") if isinstance(DEFAULT_TOKEN, str) else DEFAULT_TOKEN

    if secrets.compare_digest(token_bytes, default_bytes):
        raise HTTPException(status_code=403, detail="Invalid credentials")
    if not secrets.compare_digest(token_bytes, admin_token_bytes):
        raise HTTPException(status_code=403, detail="Invalid credentials")
    return {"authenticated": True}


def log_action_digest(key: bytes, *parts: str) -> str:
    message = "|".join(parts)
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest()
