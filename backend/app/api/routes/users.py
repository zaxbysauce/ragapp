"""User management routes (admin/superadmin only)."""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
from app.api.deps import get_current_active_user, require_role
from app.config import settings
from app.models.database import get_pool
from app.services.auth_service import hash_password, password_strength_check


class UserUpdateRequest(BaseModel):
    username: Optional[str] = None
    full_name: Optional[str] = None


class PasswordResetRequest(BaseModel):
    password: str


class UpdateUserGroupsRequest(BaseModel):
    group_ids: List[int]


router = APIRouter(prefix="/users", tags=["users"])


@router.get("/")
async def list_users(
    skip: int = 0, limit: int = 100, user: dict = Depends(require_role("admin"))
):
    """List all users (admin/superadmin only)."""
    pool = get_pool(str(settings.sqlite_path))
    conn = pool.get_connection()

    try:
        # Get total count of all users
        total_cursor = conn.execute("SELECT COUNT(*) FROM users")
        total = total_cursor.fetchone()[0]

        cursor = conn.execute(
            """SELECT id, username, full_name, role, is_active, created_at
               FROM users
               ORDER BY id
               LIMIT ? OFFSET ?""",
            (limit, skip),
        )
        rows = cursor.fetchall()

        if not rows:
            return {"users": [], "total": total}

        user_ids = [row[0] for row in rows]
        placeholders = ",".join("?" * len(user_ids))

        # Batch fetch all groups for users in one query
        group_cursor = conn.execute(
            f"""SELECT ug.user_id, g.name FROM groups g
               JOIN user_groups ug ON g.id = ug.group_id
               WHERE ug.user_id IN ({placeholders})""",
            user_ids,
        )
        user_groups = {}
        for user_id, group_name in group_cursor.fetchall():
            if user_id not in user_groups:
                user_groups[user_id] = []
            user_groups[user_id].append(group_name)

        # Batch fetch all last login times for users in one query
        session_cursor = conn.execute(
            f"""SELECT user_id, MAX(created_at) FROM user_sessions
               WHERE user_id IN ({placeholders})
               GROUP BY user_id""",
            user_ids,
        )
        user_last_logins = dict(session_cursor.fetchall())

        users = []
        for row in rows:
            user_id = row[0]
            users.append(
                {
                    "id": user_id,
                    "username": row[1],
                    "full_name": row[2] or "",
                    "role": row[3],
                    "is_active": bool(row[4]),
                    "created_at": row[5],
                    "groups": user_groups.get(user_id, []),
                    "last_login_at": user_last_logins.get(user_id),
                }
            )

        return {"users": users, "total": total}
    finally:
        pool.release_connection(conn)


@router.get("/{user_id}")
async def get_user(user_id: int, user: dict = Depends(require_role("admin"))):
    """Get user details (admin/superadmin only)."""
    pool = get_pool(str(settings.sqlite_path))
    conn = pool.get_connection()

    try:
        cursor = conn.execute(
            """SELECT id, username, full_name, role, is_active, created_at
               FROM users
               WHERE id = ?""",
            (user_id,),
        )
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "id": row[0],
            "username": row[1],
            "full_name": row[2] or "",
            "role": row[3],
            "is_active": bool(row[4]),
            "created_at": row[5],
        }
    finally:
        pool.release_connection(conn)


@router.patch("/{user_id}/role")
async def update_user_role(
    user_id: int, role: str, user: dict = Depends(require_role("superadmin"))
):
    """Update user role (superadmin only).

    Constraint: Cannot demote last superadmin.
    """
    valid_roles = ["superadmin", "admin", "member", "viewer"]
    if role not in valid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role. Must be one of: {', '.join(valid_roles)}",
        )

    pool = get_pool(str(settings.sqlite_path))
    conn = pool.get_connection()

    try:
        # Get current user details
        cursor = conn.execute("SELECT role FROM users WHERE id = ?", (user_id,))
        target_row = cursor.fetchone()

        if not target_row:
            raise HTTPException(status_code=404, detail="User not found")

        current_role = target_row[0]

        # Check if trying to demote from superadmin
        if current_role == "superadmin" and role != "superadmin":
            # Count superadmins
            cursor = conn.execute(
                "SELECT COUNT(*) FROM users WHERE role = 'superadmin' AND is_active = 1"
            )
            superadmin_count = cursor.fetchone()[0]

            if superadmin_count <= 1:
                raise HTTPException(
                    status_code=400, detail="Cannot demote the last superadmin"
                )

        # Update role
        cursor = conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "message": f"User role updated to {role}",
            "user_id": user_id,
            "role": role,
        }
    finally:
        pool.release_connection(conn)


@router.patch("/{user_id}/active")
async def update_user_active(
    user_id: int, is_active: bool, user: dict = Depends(require_role("admin"))
):
    """Activate/deactivate user (admin/superadmin only).

    Constraint: Cannot deactivate last superadmin.
    """
    pool = get_pool(str(settings.sqlite_path))
    conn = pool.get_connection()

    try:
        # Get current user details
        cursor = conn.execute(
            "SELECT role, is_active FROM users WHERE id = ?", (user_id,)
        )
        target_row = cursor.fetchone()

        if not target_row:
            raise HTTPException(status_code=404, detail="User not found")

        target_role = target_row[0]
        currently_active = bool(target_row[1])

        # Check if trying to deactivate a superadmin
        if target_role == "superadmin" and not is_active and currently_active:
            # Count active superadmins
            cursor = conn.execute(
                "SELECT COUNT(*) FROM users WHERE role = 'superadmin' AND is_active = 1"
            )
            superadmin_count = cursor.fetchone()[0]

            if superadmin_count <= 1:
                raise HTTPException(
                    status_code=400, detail="Cannot deactivate the last superadmin"
                )

        # Update active status
        cursor = conn.execute(
            "UPDATE users SET is_active = ? WHERE id = ?",
            (1 if is_active else 0, user_id),
        )
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")

        status_str = "activated" if is_active else "deactivated"
        return {
            "message": f"User {status_str}",
            "user_id": user_id,
            "is_active": is_active,
        }
    finally:
        pool.release_connection(conn)


@router.delete("/{user_id}")
async def delete_user(user_id: int, user: dict = Depends(require_role("superadmin"))):
    """Delete user (superadmin only).

    Constraint: Cannot delete last superadmin.
    """
    pool = get_pool(str(settings.sqlite_path))
    conn = pool.get_connection()

    try:
        # Get current user details
        cursor = conn.execute("SELECT role FROM users WHERE id = ?", (user_id,))
        target_row = cursor.fetchone()

        if not target_row:
            raise HTTPException(status_code=404, detail="User not found")

        target_role = target_row[0]

        # Check if trying to delete a superadmin
        if target_role == "superadmin":
            # Count superadmins
            cursor = conn.execute(
                "SELECT COUNT(*) FROM users WHERE role = 'superadmin' AND is_active = 1"
            )
            superadmin_count = cursor.fetchone()[0]

            if superadmin_count <= 1:
                raise HTTPException(
                    status_code=400, detail="Cannot delete the last superadmin"
                )

        # Cannot delete self
        if user_id == user.get("id"):
            raise HTTPException(
                status_code=400, detail="Cannot delete your own account"
            )

        # Delete user
        cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")

        return {"message": "User deleted", "user_id": user_id}
    finally:
        pool.release_connection(conn)


@router.patch("/{user_id}")
async def update_user(
    user_id: int,
    update: UserUpdateRequest,
    current_user: dict = Depends(require_role("admin")),
):
    """Update user username/full_name (admin/superadmin only)."""
    pool = get_pool(str(settings.sqlite_path))
    conn = pool.get_connection()

    try:
        # Check if user exists
        cursor = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="User not found")

        # Validate username uniqueness if provided
        if update.username is not None:
            cursor = conn.execute(
                "SELECT id FROM users WHERE username = ? AND id != ?",
                (update.username, user_id),
            )
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="Username already taken")

        # Build update fields using explicit whitelist
        allowed_fields = {"username", "full_name"}
        update_fields = []
        update_values = []

        # Only process fields that are explicitly in the whitelist
        if update.username is not None and "username" in allowed_fields:
            update_fields.append("username = ?")
            update_values.append(update.username)

        if update.full_name is not None and "full_name" in allowed_fields:
            update_fields.append("full_name = ?")
            update_values.append(update.full_name)

        if not update_fields:
            raise HTTPException(status_code=400, detail="No fields to update")

        update_values.append(user_id)

        # Execute update
        cursor = conn.execute(
            f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?", update_values
        )
        conn.commit()

        return {
            "message": "User updated",
            "user_id": user_id,
            "username": update.username,
            "full_name": update.full_name,
        }
    finally:
        pool.release_connection(conn)


@router.patch("/{user_id}/password")
async def reset_user_password(
    user_id: int,
    request: PasswordResetRequest,
    current_user: dict = Depends(require_role("admin")),
):
    """Admin password reset (admin/superadmin only)."""
    pool = get_pool(str(settings.sqlite_path))
    conn = pool.get_connection()

    try:
        # Check if user exists
        cursor = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="User not found")

        # Validate password strength
        try:
            password_strength_check(request.password)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Hash the new password
        hashed_password = hash_password(request.password)

        # Update password
        cursor = conn.execute(
            "UPDATE users SET hashed_password = ? WHERE id = ?",
            (hashed_password, user_id),
        )
        conn.commit()

        return {"message": "Password reset successfully", "user_id": user_id}
    finally:
        pool.release_connection(conn)


@router.get("/{user_id}/groups")
async def get_user_groups(
    user_id: int, current_user: dict = Depends(require_role("admin"))
):
    """Get user's groups (admin/superadmin only)."""
    pool = get_pool(str(settings.sqlite_path))
    conn = pool.get_connection()

    try:
        # Check if user exists
        cursor = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="User not found")

        # Get user's groups
        cursor = conn.execute(
            """SELECT g.id, g.name FROM groups g
               JOIN user_groups ug ON g.id = ug.group_id
               WHERE ug.user_id = ?""",
            (user_id,),
        )
        rows = cursor.fetchall()

        groups = [{"id": row[0], "name": row[1]} for row in rows]

        return {"groups": groups}
    finally:
        pool.release_connection(conn)


@router.put("/{user_id}/groups")
async def update_user_groups(
    user_id: int,
    request: UpdateUserGroupsRequest,
    current_user: dict = Depends(require_role("admin")),
):
    """Update user's groups (admin/superadmin only).

    Replaces user's group memberships with the provided group_ids.
    Uses explicit transaction for atomicity.
    """
    pool = get_pool(str(settings.sqlite_path))
    conn = pool.get_connection()

    try:
        # Start explicit transaction
        conn.execute("BEGIN")

        # Check if user exists
        cursor = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,))
        if not cursor.fetchone():
            conn.execute("ROLLBACK")
            raise HTTPException(status_code=404, detail="User not found")

        # Verify all group_ids exist
        if request.group_ids:
            placeholders = ",".join("?" * len(request.group_ids))
            cursor = conn.execute(
                f"SELECT id FROM groups WHERE id IN ({placeholders})", request.group_ids
            )
            existing_group_ids = {row[0] for row in cursor.fetchall()}

            invalid_groups = set(request.group_ids) - existing_group_ids
            if invalid_groups:
                conn.execute("ROLLBACK")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid group IDs: {sorted(invalid_groups)}",
                )

        # Delete existing group memberships
        cursor = conn.execute("DELETE FROM user_groups WHERE user_id = ?", (user_id,))

        # Insert new group memberships
        for group_id in request.group_ids:
            cursor = conn.execute(
                "INSERT INTO user_groups (user_id, group_id) VALUES (?, ?)",
                (user_id, group_id),
            )

        conn.execute("COMMIT")

        return {
            "message": "User groups updated",
            "user_id": user_id,
            "group_ids": request.group_ids,
        }
    except HTTPException:
        raise
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        pool.release_connection(conn)
