"""Authentication routes."""
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.deps import get_current_active_user, get_db
from app.models.database import get_pool
from app.services.auth_service import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from app.limiter import limiter

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()

# Cookie settings for refresh tokens
REFRESH_TOKEN_COOKIE_NAME = "refresh_token"
REFRESH_TOKEN_MAX_AGE_DAYS = 30


@router.post("/register")
@limiter.limit("5/hour")
async def register(
    request: Request,
    username: str,
    password: str,
    full_name: str = "",
    db=Depends(get_db),
):
    """
    Register a new user. First user becomes superadmin.

    Rate limiting: Apply 5 requests per hour per IP.
    """
    # Validate input
    if not username or len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if not password or len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    # Check if username already exists
    cursor = db.execute("SELECT id FROM users WHERE username = ? COLLATE NOCASE", (username,))
    if cursor.fetchone():
        raise HTTPException(status_code=409, detail="Username already exists")

    # Check if this is the first user (becomes superadmin)
    cursor = db.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    role = "superadmin" if user_count == 0 else "member"

    # Hash password and create user
    hashed_pw = hash_password(password)

    try:
        cursor = db.execute(
            """
            INSERT INTO users (username, hashed_password, full_name, role, is_active)
            VALUES (?, ?, ?, ?, 1)
            """,
            (username, hashed_pw, full_name, role),
        )
        db.commit()
        user_id = cursor.lastrowid
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

    return {
        "id": user_id,
        "username": username,
        "full_name": full_name,
        "role": role,
        "is_active": True,
        "message": "User registered successfully",
    }


@router.post("/login")
@limiter.limit("10/minute")
async def login(
    request: Request,
    response: Response,
    username: str,
    password: str,
    db=Depends(get_db),
):
    """
    Login and receive access token + refresh cookie.

    Rate limiting: Apply 10 requests per minute per IP.
    """
    # Fetch user by username
    cursor = db.execute(
        "SELECT id, username, hashed_password, full_name, role, is_active FROM users WHERE username = ? COLLATE NOCASE",
        (username,),
    )
    row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    user_id, db_username, hashed_pw, full_name, role, is_active = row

    if not is_active:
        raise HTTPException(status_code=403, detail="User account is inactive")

    # Verify password
    if not verify_password(password, hashed_pw):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Create tokens
    access_token = create_access_token(user_id, db_username, role)
    refresh_token_raw, refresh_token_hash = create_refresh_token()

    # Store refresh token in database
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_MAX_AGE_DAYS)

    # Get client info
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    try:
        db.execute(
            """
            INSERT INTO user_sessions (user_id, refresh_token_hash, expires_at, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, refresh_token_hash, expires_at.isoformat(), ip_address, user_agent),
        )

        # Update last_login_at
        db.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), user_id),
        )
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

    # Set refresh token as httpOnly cookie
    response.set_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME,
        value=refresh_token_raw,
        httponly=True,
        secure=True,  # Only sent over HTTPS
        samesite="lax",
        max_age=REFRESH_TOKEN_MAX_AGE_DAYS * 24 * 60 * 60,  # 30 days in seconds
        path="/api/v1/auth/refresh",
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 15 * 60,  # 15 minutes in seconds
        "user": {
            "id": user_id,
            "username": db_username,
            "full_name": full_name,
            "role": role,
        },
    }


@router.post("/refresh")
@limiter.limit("30/minute")
async def refresh(
    request: Request,
    response: Response,
    refresh_token: Optional[str] = Cookie(None, alias=REFRESH_TOKEN_COOKIE_NAME),
    db=Depends(get_db),
):
    """
    Refresh access token using refresh token.

    Rate limiting: Apply 30 requests per minute per IP.
    """
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh token missing")

    import hashlib

    # Hash the provided refresh token to look it up
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()

    # Find valid session
    cursor = db.execute(
        """
        SELECT s.id, s.user_id, s.expires_at, u.username, u.role, u.is_active
        FROM user_sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.refresh_token_hash = ?
        """,
        (token_hash,),
    )
    row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    session_id, user_id, expires_at_str, username, role, is_active = row

    # Check if session is expired
    expires_at = datetime.fromisoformat(expires_at_str)
    if expires_at < datetime.now(timezone.utc):
        # Clean up expired session
        db.execute("DELETE FROM user_sessions WHERE id = ?", (session_id,))
        db.commit()
        raise HTTPException(status_code=401, detail="Refresh token expired")

    if not is_active:
        raise HTTPException(status_code=403, detail="User account is inactive")

    # Rotate refresh token (create new one, invalidate old)
    new_refresh_token_raw, new_refresh_token_hash = create_refresh_token()
    new_expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_MAX_AGE_DAYS)

    try:
        # Delete old session and create new one
        db.execute("DELETE FROM user_sessions WHERE id = ?", (session_id,))
        db.execute(
            """
            INSERT INTO user_sessions (user_id, refresh_token_hash, expires_at, last_used_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, new_refresh_token_hash, new_expires_at.isoformat(), datetime.now(timezone.utc).isoformat()),
        )
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to rotate session: {str(e)}")

    # Create new access token
    access_token = create_access_token(user_id, username, role)

    # Set new refresh token cookie
    response.set_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME,
        value=new_refresh_token_raw,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=REFRESH_TOKEN_MAX_AGE_DAYS * 24 * 60 * 60,
        path="/api/v1/auth/refresh",
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 15 * 60,
    }


@router.post("/logout")
async def logout(
    response: Response,
    refresh_token: Optional[str] = Cookie(None, alias=REFRESH_TOKEN_COOKIE_NAME),
    db=Depends(get_db),
):
    """
    Logout and revoke refresh token.

    Rate limiting: Apply 10 requests per minute per IP.
    """
    if refresh_token:
        import hashlib

        # Hash the token to find and delete the session
        token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
        db.execute("DELETE FROM user_sessions WHERE refresh_token_hash = ?", (token_hash,))
        db.commit()

    # Clear the refresh token cookie
    response.delete_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME,
        path="/api/v1/auth/refresh",
    )

    return {"message": "Logged out successfully"}


@router.get("/me")
async def get_me(user: dict = Depends(get_current_active_user)):
    """
    Get current user profile.

    Rate limiting: Apply 60 requests per minute per user.
    """
    return {
        "id": user["id"],
        "username": user["username"],
        "full_name": user.get("full_name", ""),
        "role": user["role"],
        "is_active": user["is_active"],
    }


@router.patch("/me")
async def update_me(
    full_name: Optional[str] = None,
    password: Optional[str] = None,
    user: dict = Depends(get_current_active_user),
    db=Depends(get_db),
):
    """
    Update current user profile.

    Rate limiting: Apply 10 requests per minute per user.
    """
    user_id = user["id"]
    updates = []
    params = []

    if full_name is not None:
        updates.append("full_name = ?")
        params.append(full_name)

    if password is not None:
        if len(password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        hashed_pw = hash_password(password)
        updates.append("hashed_password = ?")
        params.append(hashed_pw)

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    params.append(user_id)

    try:
        db.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE id = ?",
            tuple(params),
        )
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update user: {str(e)}")

    # Return updated user info
    cursor = db.execute(
        "SELECT id, username, full_name, role, is_active FROM users WHERE id = ?",
        (user_id,),
    )
    row = cursor.fetchone()

    return {
        "id": row[0],
        "username": row[1],
        "full_name": row[2],
        "role": row[3],
        "is_active": bool(row[4]),
        "message": "Profile updated successfully",
    }
