"""Vault membership management routes."""
import asyncio
import sqlite3
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field, ConfigDict

from app.api.deps import (
    get_db,
    get_current_active_user,
    require_vault_permission,
)


router = APIRouter(prefix="/vaults/{vault_id}/members", tags=["vault-members"])


class VaultMemberCreateRequest(BaseModel):
    """Request model for adding a member to a vault."""
    member_user_id: int = Field(..., gt=0, description="User ID to add as member")
    permission: str = Field(..., pattern="^(read|write|admin)$", description="Permission level")


class VaultMemberUpdateRequest(BaseModel):
    """Request model for updating a vault member's permission."""
    permission: str = Field(..., pattern="^(read|write|admin)$", description="New permission level")


class VaultMemberResponse(BaseModel):
    """Response model for a vault member."""
    user_id: int
    username: str
    full_name: str
    permission: str
    granted_at: Optional[str] = None
    granted_by: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class VaultMemberListResponse(BaseModel):
    """Response model for listing vault members."""
    members: List[VaultMemberResponse]
    total: int


@router.get("/", response_model=VaultMemberListResponse)
async def list_vault_members(
    vault_id: int,
    user: dict = Depends(require_vault_permission("read")),
    db: sqlite3.Connection = Depends(get_db),
):
    """
    List all members of a vault.
    
    Returns a list of all users who have direct access to the vault,
    including their permission levels and grant information.
    """
    # Verify vault exists
    cursor = await asyncio.to_thread(
        db.execute,
        "SELECT id FROM vaults WHERE id = ?",
        (vault_id,)
    )
    row = await asyncio.to_thread(cursor.fetchone)
    if not row:
        raise HTTPException(status_code=404, detail=f"Vault with id {vault_id} not found")
    
    # Fetch all vault members with user details
    cursor = await asyncio.to_thread(
        db.execute,
        """
        SELECT 
            vm.user_id,
            u.username,
            u.full_name,
            vm.permission,
            vm.granted_at,
            vm.granted_by
        FROM vault_members vm
        JOIN users u ON vm.user_id = u.id
        WHERE vm.vault_id = ?
        ORDER BY u.username
        """,
        (vault_id,)
    )
    rows = await asyncio.to_thread(cursor.fetchall)
    
    members = [
        VaultMemberResponse(
            user_id=row[0],
            username=row[1],
            full_name=row[2] or "",
            permission=row[3],
            granted_at=row[4],
            granted_by=row[5],
        )
        for row in rows
    ]
    
    return VaultMemberListResponse(
        members=members,
        total=len(members)
    )


@router.post("/", response_model=VaultMemberResponse, status_code=201)
async def add_vault_member(
    vault_id: int,
    request: VaultMemberCreateRequest,
    user: dict = Depends(require_vault_permission("admin")),
    db: sqlite3.Connection = Depends(get_db),
):
    """
    Add a member to a vault.
    
    Grants a user access to the vault with the specified permission level.
    Returns 404 if vault or user not found.
    Returns 409 if user is already a member.
    """
    # Verify vault exists
    cursor = await asyncio.to_thread(
        db.execute,
        "SELECT id FROM vaults WHERE id = ?",
        (vault_id,)
    )
    row = await asyncio.to_thread(cursor.fetchone)
    if not row:
        raise HTTPException(status_code=404, detail=f"Vault with id {vault_id} not found")
    
    # Verify target user exists
    cursor = await asyncio.to_thread(
        db.execute,
        "SELECT id, username, full_name FROM users WHERE id = ? AND is_active = 1",
        (request.member_user_id,)
    )
    target_user = await asyncio.to_thread(cursor.fetchone)
    if not target_user:
        raise HTTPException(
            status_code=404,
            detail=f"User with id {request.member_user_id} not found or inactive"
        )
    
    # Check if user is already a member
    cursor = await asyncio.to_thread(
        db.execute,
        "SELECT user_id FROM vault_members WHERE vault_id = ? AND user_id = ?",
        (vault_id, request.member_user_id)
    )
    existing = await asyncio.to_thread(cursor.fetchone)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"User {request.member_user_id} is already a member of this vault"
        )
    
    # Add the member
    try:
        cursor = await asyncio.to_thread(
            db.execute,
            """
            INSERT INTO vault_members (vault_id, user_id, permission, granted_by)
            VALUES (?, ?, ?, ?)
            """,
            (vault_id, request.member_user_id, request.permission, user["id"])
        )
        await asyncio.to_thread(db.commit)
    except sqlite3.IntegrityError as e:
        raise HTTPException(
            status_code=409,
            detail=f"Failed to add member: {str(e)}"
        )
    
    # Fetch the newly created member record
    cursor = await asyncio.to_thread(
        db.execute,
        """
        SELECT 
            vm.user_id,
            u.username,
            u.full_name,
            vm.permission,
            vm.granted_at,
            vm.granted_by
        FROM vault_members vm
        JOIN users u ON vm.user_id = u.id
        WHERE vm.vault_id = ? AND vm.user_id = ?
        """,
        (vault_id, request.member_user_id)
    )
    row = await asyncio.to_thread(cursor.fetchone)
    
    return VaultMemberResponse(
        user_id=row[0],
        username=row[1],
        full_name=row[2] or "",
        permission=row[3],
        granted_at=row[4],
        granted_by=row[5],
    )


@router.patch("/{member_user_id}", response_model=VaultMemberResponse)
async def update_vault_member(
    vault_id: int,
    member_user_id: int,
    request: VaultMemberUpdateRequest,
    user: dict = Depends(require_vault_permission("admin")),
    db: sqlite3.Connection = Depends(get_db),
):
    """
    Update a vault member's permission.
    
    Changes the permission level for an existing vault member.
    Returns 404 if vault or member not found.
    """
    # Verify vault exists
    cursor = await asyncio.to_thread(
        db.execute,
        "SELECT id FROM vaults WHERE id = ?",
        (vault_id,)
    )
    row = await asyncio.to_thread(cursor.fetchone)
    if not row:
        raise HTTPException(status_code=404, detail=f"Vault with id {vault_id} not found")
    
    # Check if member exists
    cursor = await asyncio.to_thread(
        db.execute,
        "SELECT user_id FROM vault_members WHERE vault_id = ? AND user_id = ?",
        (vault_id, member_user_id)
    )
    existing = await asyncio.to_thread(cursor.fetchone)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail=f"User {member_user_id} is not a member of this vault"
        )
    
    # Update the permission
    await asyncio.to_thread(
        db.execute,
        """
        UPDATE vault_members
        SET permission = ?, granted_by = ?
        WHERE vault_id = ? AND user_id = ?
        """,
        (request.permission, user["id"], vault_id, member_user_id)
    )
    await asyncio.to_thread(db.commit)
    
    # Fetch updated member record
    cursor = await asyncio.to_thread(
        db.execute,
        """
        SELECT 
            vm.user_id,
            u.username,
            u.full_name,
            vm.permission,
            vm.granted_at,
            vm.granted_by
        FROM vault_members vm
        JOIN users u ON vm.user_id = u.id
        WHERE vm.vault_id = ? AND vm.user_id = ?
        """,
        (vault_id, member_user_id)
    )
    row = await asyncio.to_thread(cursor.fetchone)
    
    return VaultMemberResponse(
        user_id=row[0],
        username=row[1],
        full_name=row[2] or "",
        permission=row[3],
        granted_at=row[4],
        granted_by=row[5],
    )


@router.delete("/{member_user_id}", status_code=204)
async def remove_vault_member(
    vault_id: int,
    member_user_id: int,
    user: dict = Depends(require_vault_permission("admin")),
    db: sqlite3.Connection = Depends(get_db),
):
    """
    Remove a member from a vault.
    
    Revokes a user's access to the vault.
    Returns 404 if vault or member not found.
    """
    # Verify vault exists
    cursor = await asyncio.to_thread(
        db.execute,
        "SELECT id FROM vaults WHERE id = ?",
        (vault_id,)
    )
    row = await asyncio.to_thread(cursor.fetchone)
    if not row:
        raise HTTPException(status_code=404, detail=f"Vault with id {vault_id} not found")
    
    # Check if member exists
    cursor = await asyncio.to_thread(
        db.execute,
        "SELECT user_id FROM vault_members WHERE vault_id = ? AND user_id = ?",
        (vault_id, member_user_id)
    )
    existing = await asyncio.to_thread(cursor.fetchone)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail=f"User {member_user_id} is not a member of this vault"
        )
    
    # Delete the member
    await asyncio.to_thread(
        db.execute,
        "DELETE FROM vault_members WHERE vault_id = ? AND user_id = ?",
        (vault_id, member_user_id)
    )
    await asyncio.to_thread(db.commit)
    
    return None
