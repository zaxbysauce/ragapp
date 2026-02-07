"""Authentication, CSRF, and toggle utilities."""

import hashlib
import hmac
import logging
import secrets
from typing import Callable

import redis
from fastapi import Depends, Header, HTTPException, Request, Response

from app.config import settings
from app.services.secret_manager import SecretManager

logger = logging.getLogger("security")
CSRF_COOKIE_NAME = "X-CSRF-Token"


class CSRFManager:
    def __init__(self, redis_url: str, ttl: int = 900) -> None:
        self.ttl = ttl
        try:
            self.redis = redis.from_url(redis_url, decode_responses=True)
            self.redis.ping()
        except redis.RedisError as exc:
            logger.error("Redis unavailable for CSRF: %s", exc)
            raise HTTPException(status_code=503, detail="CSRF storage unavailable")

    def generate_token(self) -> str:
        token = secrets.token_urlsafe(16)
        key = f"csrf:{token}"
        self.redis.setex(key, self.ttl, "1")
        return token

    def validate_token(self, token: str) -> bool:
        if not token:
            return False
        key = f"csrf:{token}"
        try:
            exists = self.redis.get(key)
        except redis.RedisError:
            raise HTTPException(status_code=503, detail="CSRF storage unavailable")
        if exists:
            self.redis.expire(key, self.ttl)
            return True
        return False

    def revoke_token(self, token: str) -> None:
        try:
            self.redis.delete(f"csrf:{token}")
        except redis.RedisError:
            logger.warning("Failed to revoke CSRF token (Redis error)")


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
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header missing")
        if not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        token = authorization.split(" ", 1)[1]
        scopes = [s.strip().lower() for s in x_scopes.split(",") if s.strip()]
        if scope.lower() not in scopes:
            raise HTTPException(status_code=403, detail="Missing required scope")
        if token != settings.admin_secret_token:
            raise HTTPException(status_code=403, detail="Unauthorized token")
        return {"user_id": token}

    return dependency


def csrf_protect(
    request: Request,
    x_csrf_token: str = Header(""),
) -> str:
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
        secure=True,
        httponly=False,
    )
    return token


def log_action_digest(key: bytes, *parts: str) -> str:
    message = "|".join(parts)
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest()
