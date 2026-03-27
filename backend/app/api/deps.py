"""FastAPI dependency functions."""

import secrets
import sqlite3
from contextlib import contextmanager
from typing import List

from fastapi import Request, Depends, Header, HTTPException

from app.config import Settings, settings
from app.models.database import get_pool, SQLiteConnectionPool
from app.services.llm_client import LLMClient
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.memory_store import MemoryStore
from app.services.reranking import RerankingService
from app.services.rag_engine import RAGEngine
from app.services.secret_manager import SecretManager
from app.services.toggle_manager import ToggleManager
from app.services.background_tasks import BackgroundProcessor
from app.services.maintenance import MaintenanceService
from app.services.llm_health import LLMHealthChecker
from app.services.model_checker import ModelChecker
from app.services.email_service import EmailIngestionService
from app.services.auth_service import decode_access_token
from app.security import get_csrf_manager


def get_db():
    """Yield a database connection from the pool, releasing it when done."""
    pool = get_pool(str(settings.sqlite_path))
    conn = pool.get_connection()
    try:
        yield conn
    finally:
        pool.release_connection(conn)


def get_db_pool(request: Request) -> SQLiteConnectionPool:
    """Return the database pool from app state."""
    return request.app.state.db_pool


def get_settings() -> Settings:
    """Return the application settings."""
    return settings


def get_llm_client(request: Request) -> LLMClient:
    """Return the LLM client from app state."""
    return request.app.state.llm_client


def get_embedding_service(request: Request) -> EmbeddingService:
    """Return the embedding service from app state."""
    return request.app.state.embedding_service


def get_vector_store(request: Request) -> VectorStore:
    """Return the vector store from app state."""
    return request.app.state.vector_store


def get_memory_store(request: Request) -> MemoryStore:
    """Return the memory store from app state."""
    return request.app.state.memory_store


def get_reranking_service(request: Request) -> RerankingService | None:
    """Return the RerankingService from app state when initialized."""
    return getattr(request.app.state, "reranking_service", None)


def get_rag_engine(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    memory_store: MemoryStore = Depends(get_memory_store),
    llm_client: LLMClient = Depends(get_llm_client),
    reranking_service: RerankingService = Depends(get_reranking_service),
) -> RAGEngine:
    """Return a new RAGEngine initialized with dependencies."""
    return RAGEngine(
        embedding_service=embedding_service,
        vector_store=vector_store,
        memory_store=memory_store,
        llm_client=llm_client,
        reranking_service=reranking_service,
    )


def get_toggle_manager(request: Request) -> ToggleManager:
    """Return the toggle manager from app state."""
    return request.app.state.toggle_manager


def get_secret_manager(request: Request) -> SecretManager:
    """Return the secret manager from app state."""
    return request.app.state.secret_manager


def get_background_processor(request: Request) -> BackgroundProcessor:
    """Return the background processor from app state."""
    return request.app.state.background_processor


def get_maintenance_service(request: Request) -> MaintenanceService:
    """Return the maintenance service from app state."""
    return request.app.state.maintenance_service


def get_llm_health_checker(request: Request) -> LLMHealthChecker:
    """Return the LLM health checker from app state."""
    return request.app.state.llm_health_checker


def get_model_checker(request: Request) -> ModelChecker:
    """Return the model checker from app state."""
    return request.app.state.model_checker


def get_email_service(request: Request) -> EmailIngestionService:
    """Return the email ingestion service from app state."""
    return request.app.state.email_service


async def get_current_active_user(
    authorization: str = Header(None),
    db: sqlite3.Connection = Depends(get_db),
) -> dict:
    """
    FastAPI dependency to get the current authenticated user.

    When users_enabled=False: Validates against admin_secret_token
    When users_enabled=True: Validates JWT token and fetches user from database

    Args:
        authorization: Bearer token from Authorization header
        db: Database connection

    Returns:
        User dict with id, username, role, and is_active

    Raises:
        HTTPException: 401 if auth header missing or invalid
        HTTPException: 403 if token invalid or user inactive
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization scheme")

    parts = authorization.split(" ", 1)
    if len(parts) < 2 or not parts[1].strip():
        raise HTTPException(status_code=401, detail="Token missing")

    token = parts[1].strip()

    # SECURITY: Reject the default token unconditionally, before any scope or
    # mode check, so a caller holding the unchanged default value is denied
    # immediately regardless of users_enabled or any claimed role/scope.
    DEFAULT_TOKEN = "admin-secret-token"
    if secrets.compare_digest(token, DEFAULT_TOKEN):
        raise HTTPException(
            status_code=403,
            detail="Invalid credentials - change default admin token",
        )

    # When users are disabled, fall back to admin token authentication
    if not settings.users_enabled:
        if not secrets.compare_digest(token, settings.admin_secret_token):
            raise HTTPException(status_code=403, detail="Invalid credentials")

        # Return a pseudo-superadmin user for admin token auth
        return {
            "id": 0,
            "username": "admin",
            "full_name": "Admin",
            "role": "superadmin",
            "is_active": True,
            "must_change_password": False,
        }

    # User authentication enabled - validate JWT token
    payload = decode_access_token(token)

    if not payload:
        raise HTTPException(status_code=403, detail="Invalid or expired token")

    user_id = int(payload.get("sub", 0))
    if not user_id:
        raise HTTPException(status_code=403, detail="Invalid token payload")

    # Fetch user from database to verify active status
    cursor = db.execute(
        "SELECT id, username, full_name, role, is_active, must_change_password FROM users WHERE id = ?",
        (user_id,),
    )
    row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=403, detail="User not found")

    user = {
        "id": row[0],
        "username": row[1],
        "full_name": row[2] or "",
        "role": row[3],
        "is_active": bool(row[4]),
        "must_change_password": bool(row[5]) if row[5] is not None else False,
    }

    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="User account is inactive")

    return user


async def evaluate_policy(
    principal: dict,
    resource_type: str,
    resource_id: int | None,
    action: str,
) -> bool:
    """
    Centralized policy evaluation for RBAC.

    Resolution order (vault resources):
    1. superadmin → True for all actions
    2. admin → True for read/write, False for vault delete
    3. vault_members row → use permission column
    4. vault_group_access (user in group) → highest permission wins
    5. vault.visibility == 'public' AND action == 'read' → True
    6. Otherwise → False

    Args:
        principal: User dict with 'id', 'role', etc.
        resource_type: Type of resource ("vault", "org", "group", "system")
        resource_id: Resource identifier (None for system-level checks)
        action: Action being attempted ("read", "write", "delete", "admin")

    Returns:
        True if action is permitted, False otherwise
    """
    user_id = principal.get("id")
    user_role = principal.get("role", "")

    if not user_id:
        return False

    # Currently only vault resources are supported
    if resource_type != "vault":
        # For other resource types, only superadmin/admin have access
        return user_role == "superadmin"

    if resource_id is None:
        return False

    # 1. Superadmin check - grant all actions
    if user_role == "superadmin":
        return True

    # 2. Admin check - grant read/write, deny vault delete
    if user_role == "admin":
        if action in ("read", "write"):
            return True
        # Admin cannot delete vaults
        if action == "delete":
            return False
        # Admin cannot do admin-level actions on vaults
        if action == "admin":
            return False

    # Get database connection for remaining checks
    pool = get_pool(str(settings.sqlite_path))
    conn = pool.get_connection()

    try:
        # 3. Check vault_members for direct user permission
        cursor = conn.execute(
            """SELECT permission FROM vault_members 
               WHERE vault_id = ? AND user_id = ?""",
            (resource_id, user_id),
        )
        row = cursor.fetchone()

        if row:
            permission = row[0]
            # permission hierarchy: admin > write > read
            permission_levels = {"read": 1, "write": 2, "admin": 3}
            action_levels = {"read": 1, "write": 2, "delete": 3, "admin": 3}

            required_level = action_levels.get(action, 1)
            user_level = permission_levels.get(permission, 0)

            if user_level >= required_level:
                return True

        # 4. Check vault_group_access for group-based permissions
        cursor = conn.execute(
            """SELECT vga.permission FROM vault_group_access vga
               JOIN group_members gm ON vga.group_id = gm.group_id
               WHERE vga.vault_id = ? AND gm.user_id = ?""",
            (resource_id, user_id),
        )
        group_permissions = cursor.fetchall()

        if group_permissions:
            # Find highest permission level across all groups
            permission_levels = {"read": 1, "write": 2, "admin": 3}
            action_levels = {"read": 1, "write": 2, "delete": 3, "admin": 3}

            highest_level = max(
                permission_levels.get(p[0], 0) for p in group_permissions
            )
            required_level = action_levels.get(action, 1)

            if highest_level >= required_level:
                return True

        # 5. Check vault visibility for public read access
        if action == "read":
            cursor = conn.execute(
                "SELECT visibility FROM vaults WHERE id = ?", (resource_id,)
            )
            row = cursor.fetchone()

            if row and row[0] == "public":
                return True

        # 6. Otherwise deny
        return False

    finally:
        pool.release_connection(conn)


def require_vault_permission(*actions: str):
    """
    FastAPI dependency for vault permission checks.

    Creates a dependency that validates the current user has at least
    one of the specified permissions on the given vault.

    Args:
        *actions: One or more actions to check ("read", "write", "delete", "admin")

    Returns:
        Dependency function that checks permissions and returns user dict

    Example:
        @router.get("/vaults/{vault_id}")
        async def get_vault(
            vault_id: int,
            user: dict = Depends(require_vault_permission("read", "admin"))
        ):
            ...
    """

    async def _check(vault_id: int, user: dict = Depends(get_current_active_user)):
        for action in actions:
            if await evaluate_policy(user, "vault", vault_id, action):
                return user
        raise HTTPException(status_code=403, detail="Insufficient vault permissions")

    return _check


def require_role(role: str):
    """
    FastAPI dependency to require a specific role or higher.

    Role hierarchy (highest to lowest):
    - superadmin (highest, can do everything)
    - admin (can manage users except role changes)
    - member (regular user)
    - viewer (read-only access)

    Args:
        role: Minimum required role ("superadmin", "admin", "member", "viewer")

    Returns:
        Dependency function that checks role and returns user dict

    Raises:
        HTTPException: 403 if user lacks required role

    Example:
        @router.get("/admin-only")
        async def admin_endpoint(user: dict = Depends(require_role("admin"))):
            ...
    """
    role_hierarchy = {
        "superadmin": 4,
        "admin": 3,
        "member": 2,
        "viewer": 1,
    }

    required_level = role_hierarchy.get(role, 0)

    async def _check_role(user: dict = Depends(get_current_active_user)) -> dict:
        user_role = user.get("role", "viewer")
        user_level = role_hierarchy.get(user_role, 0)

        if user_level < required_level:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient privileges. Required role: {role}",
            )

        return user

    return _check_role


async def require_admin_role(user: dict = Depends(get_current_active_user)) -> dict:
    """
    Dependency that requires the user to have admin role.

    Validates that the user's role is 'superadmin' or 'admin'.
    Raises 403 if not authorized.

    Returns the user dict if authorized.
    """
    user_role = user.get("role", "")
    if user_role not in ("superadmin", "admin"):
        raise HTTPException(
            status_code=403,
            detail="Admin access required",
        )
    return user


async def get_user_accessible_vault_ids(user: dict, db) -> List[int]:  # NOTE: caller must ensure `user` was produced by an authenticated, authorized dependency (e.g. get_current_active_user) — this function does not re-validate identity.
    """
    Get all vault IDs that a user has access to.

    Returns list of vault IDs based on:
    - Direct vault_members permissions
    - vault_group_access via group membership
    - For superadmin/admin: returns empty list (means "all vaults")

    Returns list of vault IDs (may be empty for regular users with no vault access).
    """
    user_id = user["id"]
    user_role = user.get("role", "")

    # superadmin/admin can access all vaults
    if user_role in ("superadmin", "admin"):
        return []  # Empty list means "all vaults"

    vault_ids = set()

    # Direct vault_members access
    cursor = db.execute(
        "SELECT vault_id FROM vault_members WHERE user_id = ?", (user_id,)
    )
    for row in cursor.fetchall():
        vault_ids.add(row[0])

    # Group-based access
    cursor = db.execute(
        """SELECT DISTINCT vga.vault_id FROM vault_group_access vga
           JOIN group_members gm ON vga.group_id = gm.group_id
           WHERE gm.user_id = ?""",
        (user_id,),
    )
    for row in cursor.fetchall():
        vault_ids.add(row[0])

    return list(vault_ids)
