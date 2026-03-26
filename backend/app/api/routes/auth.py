"""Authentication routes."""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.api.deps import get_current_active_user, get_db
from app.models.database import get_pool
from app.services.auth_service import (
    create_access_token,
    create_refresh_token,
    hash_password,
    password_strength_check,
    verify_password,
)
from app.limiter import limiter

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()

# Cookie settings for refresh tokens
REFRESH_TOKEN_COOKIE_NAME = "refresh_token"
REFRESH_TOKEN_MAX_AGE_DAYS = 30


# Request models
class RegisterRequest(BaseModel):
    username: str = Field(max_length=255)
    password: str = Field(max_length=128)
    full_name: str = Field(default="", max_length=255)


class LoginRequest(BaseModel):
    username: str = Field(max_length=255)
    password: str = Field(max_length=128)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@router.post("/register")
@limiter.limit("5/hour")
async def register(
    request: Request,
    body: RegisterRequest,
    db=Depends(get_db),
):
    """
    Register a new user. First user becomes superadmin.

    Rate limiting: Apply 5 requests per hour per IP.
    """
    # Validate input
    if not body.username or len(body.username) < 3:
        raise HTTPException(
            status_code=400, detail="Username must be at least 3 characters"
        )
    if not body.password or len(body.password) < 8:
        raise HTTPException(
            status_code=400, detail="Password must be at least 8 characters"
        )

    # Check if username already exists
    cursor = db.execute(
        "SELECT id FROM users WHERE username = ? COLLATE NOCASE", (body.username,)
    )
    if cursor.fetchone():
        raise HTTPException(status_code=409, detail="Username already exists")

    # Check if this is the first user (becomes superadmin)
    cursor = db.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    role = "superadmin" if user_count == 0 else "member"

    # Hash password and create user
    hashed_pw = hash_password(body.password)

    try:
        cursor = db.execute(
            """
            INSERT INTO users (username, hashed_password, full_name, role, is_active)
            VALUES (?, ?, ?, ?, 1)
            """,
            (body.username, hashed_pw, body.full_name, role),
        )
        db.commit()
        user_id = cursor.lastrowid
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

    return {
        "id": user_id,
        "username": body.username,
        "full_name": body.full_name,
        "role": role,
        "is_active": True,
        "message": "User registered successfully",
    }


@router.post("/login")
@limiter.limit("10/minute")
async def login(
    request: Request,
    response: Response,
    body: LoginRequest,
    db=Depends(get_db),
):
    """
    Login and receive access token + refresh cookie.

    Rate limiting: Apply 10 requests per minute per IP.
    """
    # Fetch user by username including lockout fields
    cursor = db.execute(
        "SELECT id, username, hashed_password, full_name, role, is_active, failed_attempts, locked_until FROM users WHERE username = ? COLLATE NOCASE",
        (body.username,),
    )
    row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    (
        user_id,
        db_username,
        hashed_pw,
        full_name,
        role,
        is_active,
        failed_attempts,
        locked_until,
    ) = row

    # Check if account is locked
    if locked_until is not None:
        locked_until_dt = datetime.fromisoformat(locked_until)
        if locked_until_dt > datetime.now(timezone.utc):
            seconds_remaining = max(
                1, int((locked_until_dt - datetime.now(timezone.utc)).total_seconds())
            )
            raise HTTPException(
                status_code=423,
                detail=f"Account locked. Try again in {seconds_remaining} seconds.",
                headers={"Retry-After": str(seconds_remaining)},
            )

    if not is_active:
        raise HTTPException(status_code=403, detail="User account is inactive")

    # Verify password
    if not verify_password(body.password, hashed_pw):
        # Increment failed_attempts
        new_failed_attempts = failed_attempts + 1
        try:
            if new_failed_attempts >= 5:
                # Lock the account for 15 minutes
                lockout_time = datetime.now(timezone.utc) + timedelta(minutes=15)
                db.execute(
                    "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                    (new_failed_attempts, lockout_time.isoformat(), user_id),
                )
            else:
                db.execute(
                    "UPDATE users SET failed_attempts = ? WHERE id = ?",
                    (new_failed_attempts, user_id),
                )
            db.commit()
        except Exception:
            db.rollback()
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
            (
                user_id,
                refresh_token_hash,
                expires_at.isoformat(),
                ip_address,
                user_agent,
            ),
        )

        # Update last_login_at and reset failed_attempts/locked_until on successful login
        db.execute(
            "UPDATE users SET last_login_at = ?, failed_attempts = 0, locked_until = NULL WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), user_id),
        )
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to create session: {str(e)}"
        )

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
    new_expires_at = datetime.now(timezone.utc) + timedelta(
        days=REFRESH_TOKEN_MAX_AGE_DAYS
    )

    try:
        # Delete old session and create new one
        db.execute("DELETE FROM user_sessions WHERE id = ?", (session_id,))
        db.execute(
            """
            INSERT INTO user_sessions (user_id, refresh_token_hash, expires_at, last_used_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                user_id,
                new_refresh_token_hash,
                new_expires_at.isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to rotate session: {str(e)}"
        )

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
        db.execute(
            "DELETE FROM user_sessions WHERE refresh_token_hash = ?", (token_hash,)
        )
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
            raise HTTPException(
                status_code=400, detail="Password must be at least 8 characters"
            )
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


@router.post("/change-password")
async def change_password(
    request: Request,
    response: Response,
    body: ChangePasswordRequest,
    user: dict = Depends(get_current_active_user),
    db=Depends(get_db),
):
    """
    Change the current user's password.

    Validates current password, enforces strength policy on new password,
    revokes all existing sessions, and issues new tokens.
    """
    # 1. Fetch user from DB to get hashed_password
    cursor = db.execute("SELECT hashed_password FROM users WHERE id = ?", (user["id"],))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    hashed_pw = row[0]

    # 2. Verify current password
    if not verify_password(body.current_password, hashed_pw):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    # 3. Validate new password with password_strength_check
    try:
        password_strength_check(body.new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 4. Hash new password
    new_hashed = hash_password(body.new_password)

    # 5. Update password and set must_change_password=False
    db.execute(
        "UPDATE users SET hashed_password = ?, must_change_password = 0 WHERE id = ?",
        (new_hashed, user["id"]),
    )

    # 6. Revoke ALL existing sessions for this user (delete all user_sessions rows)
    db.execute("DELETE FROM user_sessions WHERE user_id = ?", (user["id"],))

    # 7. Create new tokens
    access_token = create_access_token(user["id"], user["username"], user["role"])
    refresh_token_raw, refresh_token_hash = create_refresh_token()
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_MAX_AGE_DAYS)

    db.execute(
        """INSERT INTO user_sessions (user_id, refresh_token_hash, expires_at, last_used_at)
           VALUES (?, ?, ?, ?)""",
        (
            user["id"],
            refresh_token_hash,
            expires_at.isoformat(),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    db.commit()

    # 8. Set new refresh token cookie
    response.set_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME,
        value=refresh_token_raw,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=REFRESH_TOKEN_MAX_AGE_DAYS * 24 * 60 * 60,
        path="/api/v1/auth/refresh",
    )

    return {
        "message": "Password changed successfully",
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.get("/sessions")
async def list_sessions(
    user: dict = Depends(get_current_active_user),
    db=Depends(get_db),
):
    """
    List all active sessions for the current user.

    Returns sessions with id, ip_address, user_agent, created_at, last_used_at.
    Does NOT return token hashes (security).
    """
    cursor = db.execute(
        """
        SELECT id, ip_address, user_agent, created_at, last_used_at
        FROM user_sessions
        WHERE user_id = ?
        """,
        (user["id"],),
    )
    rows = cursor.fetchall()

    sessions = []
    for row in rows:
        session_id, ip_address, user_agent, created_at, last_used_at = row
        sessions.append(
            {
                "id": session_id,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "created_at": created_at,
                "last_used_at": last_used_at if last_used_at else created_at,
            }
        )

    return sessions


@router.delete("/sessions/{session_id}")
async def revoke_session(
    session_id: int,
    user: dict = Depends(get_current_active_user),
    db=Depends(get_db),
):
    """
    Revoke a specific session by ID.

    User can only revoke their own sessions.
    """
    # Verify session belongs to user
    cursor = db.execute(
        "SELECT id FROM user_sessions WHERE id = ? AND user_id = ?",
        (session_id, user["id"]),
    )
    row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    # Delete the session
    db.execute("DELETE FROM user_sessions WHERE id = ?", (session_id,))
    db.commit()

    return Response(status_code=204)


@router.delete("/sessions")
async def revoke_all_sessions(
    response: Response,
    user: dict = Depends(get_current_active_user),
    db=Depends(get_db),
    refresh_token: Optional[str] = Cookie(None, alias=REFRESH_TOKEN_COOKIE_NAME),
):
    """
    Revoke all sessions for the current user except the current one.

    The current session is identified by the refresh token cookie.
    User stays logged in with a new session.
    """
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh token missing")

    import hashlib

    # Hash the current refresh token to find the current session
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()

    # Find the current session
    cursor = db.execute(
        "SELECT id FROM user_sessions WHERE refresh_token_hash = ? AND user_id = ?",
        (token_hash, user["id"]),
    )
    row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    current_session_id = row[0]

    # Delete all sessions for this user except the current one
    db.execute(
        "DELETE FROM user_sessions WHERE user_id = ? AND id != ?",
        (user["id"], current_session_id),
    )

    # Rotate the current session (create new refresh token for current session)
    new_refresh_token_raw, new_refresh_token_hash = create_refresh_token()
    new_expires_at = datetime.now(timezone.utc) + timedelta(
        days=REFRESH_TOKEN_MAX_AGE_DAYS
    )

    # Delete old session and create new one
    db.execute("DELETE FROM user_sessions WHERE id = ?", (current_session_id,))
    db.execute(
        """
        INSERT INTO user_sessions (user_id, refresh_token_hash, expires_at, last_used_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            user["id"],
            new_refresh_token_hash,
            new_expires_at.isoformat(),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    db.commit()

    # Set new refresh token cookie so user stays logged in
    response.set_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME,
        value=new_refresh_token_raw,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=REFRESH_TOKEN_MAX_AGE_DAYS * 24 * 60 * 60,
        path="/api/v1/auth/refresh",
    )

    return {"message": "All other sessions revoked"}
