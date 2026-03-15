"""Organization management routes."""
import asyncio
import sqlite3
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field, ConfigDict, field_validator

from app.api.deps import get_db, require_role, get_current_active_user


router = APIRouter(prefix="/organizations", tags=["organizations"])


class OrganizationCreateRequest(BaseModel):
    """Request model for creating a new organization."""
    name: str = Field(..., min_length=1, max_length=255, description="Organization name")
    description: str = Field(default="", max_length=1000, description="Organization description")

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v


class OrganizationUpdateRequest(BaseModel):
    """Request model for updating an organization."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="New organization name")
    description: Optional[str] = Field(None, max_length=1000, description="New description")

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v


class OrganizationMemberResponse(BaseModel):
    """Response model for an organization member."""
    user_id: int
    username: str
    full_name: str
    role: str
    joined_at: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class OrganizationResponse(BaseModel):
    """Response model for an organization."""
    id: int
    name: str
    description: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    member_count: int = 0
    vault_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class OrganizationListResponse(BaseModel):
    """Response model for listing organizations."""
    organizations: List[OrganizationResponse]
    total: int


class OrganizationDetailResponse(OrganizationResponse):
    """Response model for detailed organization view including members."""
    members: List[OrganizationMemberResponse]


@router.get("/", response_model=OrganizationListResponse)
async def list_organizations(
    user: dict = Depends(require_role("member")),
    db: sqlite3.Connection = Depends(get_db),
):
    """
    List organizations the user belongs to.
    
    Returns all organizations where the current user is a member,
    including member counts and vault counts.
    """
    cursor = await asyncio.to_thread(
        db.execute,
        """
        SELECT 
            o.id,
            o.name,
            o.description,
            o.created_at,
            o.updated_at,
            COUNT(DISTINCT om.user_id) as member_count,
            COUNT(DISTINCT v.id) as vault_count
        FROM organizations o
        JOIN org_members om ON o.id = om.org_id
        LEFT JOIN org_members om2 ON o.id = om2.org_id
        LEFT JOIN vaults v ON v.org_id = o.id
        WHERE om.user_id = ?
        GROUP BY o.id
        ORDER BY o.name
        """,
        (user["id"],)
    )
    rows = await asyncio.to_thread(cursor.fetchall)
    
    organizations = [
        OrganizationResponse(
            id=row[0],
            name=row[1],
            description=row[2] or "",
            created_at=row[3],
            updated_at=row[4],
            member_count=row[5] or 0,
            vault_count=row[6] or 0,
        )
        for row in rows
    ]
    
    return OrganizationListResponse(
        organizations=organizations,
        total=len(organizations)
    )


@router.post("/", response_model=OrganizationResponse, status_code=201)
async def create_organization(
    request: OrganizationCreateRequest,
    user: dict = Depends(require_role("admin")),
    db: sqlite3.Connection = Depends(get_db),
):
    """
    Create a new organization (admin+).
    
    Creates a new organization and adds the creator as the first admin member.
    Returns 409 if organization name already exists.
    """
    try:
        # Start transaction
        await asyncio.to_thread(db.execute, "BEGIN TRANSACTION")
        
        # Create the organization
        cursor = await asyncio.to_thread(
            db.execute,
            "INSERT INTO organizations (name, description) VALUES (?, ?)",
            (request.name, request.description)
        )
        org_id = cursor.lastrowid
        
        if org_id is None:
            await asyncio.to_thread(db.rollback)
            raise HTTPException(status_code=500, detail="Failed to create organization")
        
        # Add creator as admin member
        await asyncio.to_thread(
            db.execute,
            "INSERT INTO org_members (org_id, user_id, role) VALUES (?, ?, ?)",
            (org_id, user["id"], "admin")
        )
        
        await asyncio.to_thread(db.commit)
        
    except sqlite3.IntegrityError:
        await asyncio.to_thread(db.rollback)
        raise HTTPException(
            status_code=409,
            detail=f"Organization with name '{request.name}' already exists"
        )
    except Exception:
        await asyncio.to_thread(db.rollback)
        raise
    
    # Fetch the created organization
    cursor = await asyncio.to_thread(
        db.execute,
        """
        SELECT 
            o.id,
            o.name,
            o.description,
            o.created_at,
            o.updated_at,
            1 as member_count,
            0 as vault_count
        FROM organizations o
        WHERE o.id = ?
        """,
        (org_id,)
    )
    row = await asyncio.to_thread(cursor.fetchone)
    
    return OrganizationResponse(
        id=row[0],
        name=row[1],
        description=row[2] or "",
        created_at=row[3],
        updated_at=row[4],
        member_count=row[5],
        vault_count=row[6],
    )


@router.get("/{org_id}", response_model=OrganizationDetailResponse)
async def get_organization(
    org_id: int,
    user: dict = Depends(require_role("member")),
    db: sqlite3.Connection = Depends(get_db),
):
    """
    Get organization details.
    
    Returns detailed information about an organization including its members.
    User must be a member of the organization.
    """
    # Check if user is a member of this organization
    cursor = await asyncio.to_thread(
        db.execute,
        "SELECT 1 FROM org_members WHERE org_id = ? AND user_id = ?",
        (org_id, user["id"])
    )
    is_member = await asyncio.to_thread(cursor.fetchone)
    if not is_member:
        raise HTTPException(
            status_code=403,
            detail="You are not a member of this organization"
        )
    
    # Fetch organization details
    cursor = await asyncio.to_thread(
        db.execute,
        """
        SELECT 
            o.id,
            o.name,
            o.description,
            o.created_at,
            o.updated_at,
            COUNT(DISTINCT om.user_id) as member_count,
            COUNT(DISTINCT v.id) as vault_count
        FROM organizations o
        LEFT JOIN org_members om ON o.id = om.org_id
        LEFT JOIN vaults v ON v.org_id = o.id
        WHERE o.id = ?
        GROUP BY o.id
        """,
        (org_id,)
    )
    row = await asyncio.to_thread(cursor.fetchone)
    
    if not row:
        raise HTTPException(status_code=404, detail=f"Organization with id {org_id} not found")
    
    # Fetch organization members
    cursor = await asyncio.to_thread(
        db.execute,
        """
        SELECT 
            u.id,
            u.username,
            u.full_name,
            om.role,
            om.joined_at
        FROM org_members om
        JOIN users u ON om.user_id = u.id
        WHERE om.org_id = ?
        ORDER BY om.role DESC, u.username
        """,
        (org_id,)
    )
    member_rows = await asyncio.to_thread(cursor.fetchall)
    
    members = [
        OrganizationMemberResponse(
            user_id=row[0],
            username=row[1],
            full_name=row[2] or "",
            role=row[3],
            joined_at=row[4],
        )
        for row in member_rows
    ]
    
    return OrganizationDetailResponse(
        id=row[0],
        name=row[1],
        description=row[2] or "",
        created_at=row[3],
        updated_at=row[4],
        member_count=row[5],
        vault_count=row[6],
        members=members,
    )


@router.patch("/{org_id}", response_model=OrganizationResponse)
async def update_organization(
    org_id: int,
    request: OrganizationUpdateRequest,
    user: dict = Depends(require_role("admin")),
    db: sqlite3.Connection = Depends(get_db),
):
    """
    Update organization (admin+).
    
    Updates the name and/or description of an organization.
    User must be an admin of the organization.
    Returns 404 if organization not found.
    Returns 409 if new name conflicts with existing organization.
    """
    # Check if user is an admin of this organization
    cursor = await asyncio.to_thread(
        db.execute,
        "SELECT role FROM org_members WHERE org_id = ? AND user_id = ?",
        (org_id, user["id"])
    )
    role_row = await asyncio.to_thread(cursor.fetchone)
    if not role_row or role_row[0] != "admin":
        raise HTTPException(
            status_code=403,
            detail="You must be an organization admin to update it"
        )
    
    # Check if organization exists
    cursor = await asyncio.to_thread(
        db.execute,
        "SELECT id FROM organizations WHERE id = ?",
        (org_id,)
    )
    row = await asyncio.to_thread(cursor.fetchone)
    if not row:
        raise HTTPException(status_code=404, detail=f"Organization with id {org_id} not found")
    
    # Build update query dynamically
    update_fields = []
    params = []
    
    if request.name is not None:
        update_fields.append("name = ?")
        params.append(request.name)
    if request.description is not None:
        update_fields.append("description = ?")
        params.append(request.description)
    
    if not update_fields:
        # No fields to update, fetch and return current record
        cursor = await asyncio.to_thread(
            db.execute,
            """
            SELECT 
                o.id,
                o.name,
                o.description,
                o.created_at,
                o.updated_at,
                COUNT(DISTINCT om.user_id) as member_count,
                COUNT(DISTINCT v.id) as vault_count
            FROM organizations o
            LEFT JOIN org_members om ON o.id = om.org_id
            LEFT JOIN vaults v ON v.org_id = o.id
            WHERE o.id = ?
            GROUP BY o.id
            """,
            (org_id,)
        )
        row = await asyncio.to_thread(cursor.fetchone)
        return OrganizationResponse(
            id=row[0],
            name=row[1],
            description=row[2] or "",
            created_at=row[3],
            updated_at=row[4],
            member_count=row[5],
            vault_count=row[6],
        )
    
    params.append(org_id)
    
    # Execute update
    try:
        await asyncio.to_thread(
            db.execute,
            f"""
            UPDATE organizations
            SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            params
        )
        await asyncio.to_thread(db.commit)
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail=f"Organization with name '{request.name}' already exists"
        )
    
    # Fetch updated record
    cursor = await asyncio.to_thread(
        db.execute,
        """
        SELECT 
            o.id,
            o.name,
            o.description,
            o.created_at,
            o.updated_at,
            COUNT(DISTINCT om.user_id) as member_count,
            COUNT(DISTINCT v.id) as vault_count
        FROM organizations o
        LEFT JOIN org_members om ON o.id = om.org_id
        LEFT JOIN vaults v ON v.org_id = o.id
        WHERE o.id = ?
        GROUP BY o.id
        """,
        (org_id,)
    )
    row = await asyncio.to_thread(cursor.fetchone)
    
    return OrganizationResponse(
        id=row[0],
        name=row[1],
        description=row[2] or "",
        created_at=row[3],
        updated_at=row[4],
        member_count=row[5],
        vault_count=row[6],
    )


@router.delete("/{org_id}", status_code=204)
async def delete_organization(
    org_id: int,
    user: dict = Depends(require_role("superadmin")),
    db: sqlite3.Connection = Depends(get_db),
):
    """
    Delete organization (superadmin only).
    
    Deletes an organization and all associated data (members, groups, vault access).
    This operation cascades to org_members, groups, and group_members.
    Returns 404 if organization not found.
    """
    # Check if organization exists
    cursor = await asyncio.to_thread(
        db.execute,
        "SELECT id, name FROM organizations WHERE id = ?",
        (org_id,)
    )
    row = await asyncio.to_thread(cursor.fetchone)
    if not row:
        raise HTTPException(status_code=404, detail=f"Organization with id {org_id} not found")
    
    org_name = row[1]
    
    # Delete the organization (cascades to org_members, groups, group_members via FK)
    await asyncio.to_thread(
        db.execute,
        "DELETE FROM organizations WHERE id = ?",
        (org_id,)
    )
    await asyncio.to_thread(db.commit)
    
    return None
