"""User management routes (admin/superadmin only)."""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from app.api.deps import get_current_active_user, require_role
from app.config import settings
from app.models.database import get_pool
from app.services.auth_service import hash_password

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/")
async def list_users(
    skip: int = 0,
    limit: int = 100,
    user: dict = Depends(require_role("admin"))
):
    """List all users (admin/superadmin only)."""
    pool = get_pool(str(settings.sqlite_path))
    conn = pool.get_connection()

    try:
        cursor = conn.execute(
            """SELECT id, username, full_name, role, is_active, created_at
               FROM users
               ORDER BY id
               LIMIT ? OFFSET ?""",
            (limit, skip)
        )
        rows = cursor.fetchall()
        
        users = []
        for row in rows:
            users.append({
                "id": row[0],
                "username": row[1],
                "full_name": row[2] or "",
                "role": row[3],
                "is_active": bool(row[4]),
                "created_at": row[5],
            })
        
        return {"users": users, "total": len(users)}
    finally:
        pool.release_connection(conn)


@router.get("/{user_id}")
async def get_user(
    user_id: int,
    user: dict = Depends(require_role("admin"))
):
    """Get user details (admin/superadmin only)."""
    pool = get_pool(str(settings.sqlite_path))
    conn = pool.get_connection()

    try:
        cursor = conn.execute(
            """SELECT id, username, full_name, role, is_active, created_at
               FROM users
               WHERE id = ?""",
            (user_id,)
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
    user_id: int,
    role: str,
    user: dict = Depends(require_role("superadmin"))
):
    """Update user role (superadmin only).
    
    Constraint: Cannot demote last superadmin.
    """
    valid_roles = ["superadmin", "admin", "member", "viewer"]
    if role not in valid_roles:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid role. Must be one of: {', '.join(valid_roles)}"
        )
    
    pool = get_pool(str(settings.sqlite_path))
    conn = pool.get_connection()

    try:
        # Get current user details
        cursor = conn.execute(
            "SELECT role FROM users WHERE id = ?",
            (user_id,)
        )
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
                    status_code=400,
                    detail="Cannot demote the last superadmin"
                )
        
        # Update role
        cursor = conn.execute(
            "UPDATE users SET role = ? WHERE id = ?",
            (role, user_id)
        )
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"message": f"User role updated to {role}", "user_id": user_id, "role": role}
    finally:
        pool.release_connection(conn)


@router.patch("/{user_id}/active")
async def update_user_active(
    user_id: int,
    is_active: bool,
    user: dict = Depends(require_role("admin"))
):
    """Activate/deactivate user (admin/superadmin only).
    
    Constraint: Cannot deactivate last superadmin.
    """
    pool = get_pool(str(settings.sqlite_path))
    conn = pool.get_connection()

    try:
        # Get current user details
        cursor = conn.execute(
            "SELECT role, is_active FROM users WHERE id = ?",
            (user_id,)
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
                    status_code=400,
                    detail="Cannot deactivate the last superadmin"
                )
        
        # Update active status
        cursor = conn.execute(
            "UPDATE users SET is_active = ? WHERE id = ?",
            (1 if is_active else 0, user_id)
        )
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        status_str = "activated" if is_active else "deactivated"
        return {"message": f"User {status_str}", "user_id": user_id, "is_active": is_active}
    finally:
        pool.release_connection(conn)


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    user: dict = Depends(require_role("superadmin"))
):
    """Delete user (superadmin only).
    
    Constraint: Cannot delete last superadmin.
    """
    pool = get_pool(str(settings.sqlite_path))
    conn = pool.get_connection()

    try:
        # Get current user details
        cursor = conn.execute(
            "SELECT role FROM users WHERE id = ?",
            (user_id,)
        )
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
                    status_code=400,
                    detail="Cannot delete the last superadmin"
                )
        
        # Cannot delete self
        if user_id == user.get("id"):
            raise HTTPException(
                status_code=400,
                detail="Cannot delete your own account"
            )
        
        # Delete user
        cursor = conn.execute(
            "DELETE FROM users WHERE id = ?",
            (user_id,)
        )
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"message": "User deleted", "user_id": user_id}
    finally:
        pool.release_connection(conn)
