"""Authentication service with bcrypt and JWT."""

import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from passlib.context import CryptContext
import jwt
from app.config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)

# Constants
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 30
ALGORITHM = "HS256"


def hash_password(plain: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain, hashed)


def password_strength_check(plain_password: str) -> None:
    """
    Validate password meets strength requirements.

    Requirements:
    - Minimum 8 characters
    - At least 1 digit
    - At least 1 uppercase letter

    Raises ValueError with a clear message if requirements not met.
    """
    if not plain_password:
        raise ValueError("Password cannot be empty")

    if len(plain_password) < 8:
        raise ValueError("Password must be at least 8 characters long")

    if not any(char.isdigit() for char in plain_password):
        raise ValueError("Password must contain at least one digit")

    if not any(char.isupper() for char in plain_password):
        raise ValueError("Password must contain at least one uppercase letter")


def create_access_token(user_id: int, username: str, role: str) -> str:
    """Create a JWT access token."""
    expires = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": expires,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            return None
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def create_refresh_token() -> Tuple[str, str]:
    """
    Create a refresh token.
    Returns: (raw_token, sha256_hash)
    Store only the hash in the database.
    """
    raw = secrets.token_urlsafe(48)
    hashed = hashlib.sha256(raw.encode()).hexdigest()
    return raw, hashed
