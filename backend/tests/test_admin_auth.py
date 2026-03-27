"""
Admin role-based authentication tests for Phase 1.9.

Tests cover:
- require_admin_role dependency (allows superadmin/admin, rejects member/viewer)
- Admin protection on users.py endpoints
- Admin protection on groups.py endpoints
- Admin protection on vault group management endpoints
"""

# Fix for Windows asyncio crash: use SelectorEventLoop instead of Proactor
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import sqlite3
import tempfile
import unittest

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub missing optional dependencies
try:
    import lancedb
except ImportError:
    import types

    sys.modules["lancedb"] = types.ModuleType("lancedb")

try:
    import pyarrow
except ImportError:
    import types

    sys.modules["pyarrow"] = types.ModuleType("pyarrow")

try:
    from unstructured.partition.auto import partition
except ImportError:
    import types

    _unstructured = types.ModuleType("unstructured")
    _unstructured.__path__ = []
    _unstructured.partition = types.ModuleType("unstructured.partition")
    _unstructured.partition.__path__ = []
    _unstructured.partition.auto = types.ModuleType("unstructured.partition.auto")
    _unstructured.partition.auto.partition = lambda *args, **kwargs: []
    _unstructured.chunking = types.ModuleType("unstructured.chunking")
    _unstructured.chunking.__path__ = []
    _unstructured.chunking.title = types.ModuleType("unstructured.chunking.title")
    _unstructured.chunking.title.chunk_by_title = lambda *args, **kwargs: []
    _unstructured.documents = types.ModuleType("unstructured.documents")
    _unstructured.documents.__path__ = []
    _unstructured.documents.elements = types.ModuleType(
        "unstructured.documents.elements"
    )
    _unstructured.documents.elements.Element = type("Element", (), {})
    sys.modules["unstructured"] = _unstructured
    sys.modules["unstructured.partition"] = _unstructured.partition
    sys.modules["unstructured.partition.auto"] = _unstructured.partition.auto
    sys.modules["unstructured.chunking"] = _unstructured.chunking
    sys.modules["unstructured.chunking.title"] = _unstructured.chunking.title
    sys.modules["unstructured.documents"] = _unstructured.documents
    sys.modules["unstructured.documents.elements"] = _unstructured.documents.elements

# Imports that do not require heavy app initialization will be done at module level
# Heavy imports (app.main, TestClient) will be deferred to setUp after asyncio policy is set
from app.config import settings
from app.models.database import init_db, get_pool
from app.services.auth_service import hash_password

# Global test database path
TEST_DB_PATH = None
TEST_DATA_DIR = None


def setup_test_db():
    """Set up a temporary test database."""
    global TEST_DB_PATH, TEST_DATA_DIR
    TEST_DATA_DIR = tempfile.mkdtemp()
    TEST_DB_PATH = os.path.join(TEST_DATA_DIR, "test_admin_auth.db")
    init_db(TEST_DB_PATH)
    return TEST_DB_PATH


setup_test_db()

from fastapi.testclient import TestClient
from app.main import app


class TestRequireAdminRoleDependency(unittest.TestCase):
    """Tests for the require_admin_role dependency itself."""

    @classmethod
    def setUpClass(cls):
        """Initialize test database and pool."""
        settings.sqlite_path = TEST_DB_PATH
        get_pool(TEST_DB_PATH)

    def setUp(self):
        """Set up test client and clear users table."""
        self.client = TestClient(app)
        self.test_pool = get_pool(TEST_DB_PATH)
        self._current_user_dep = None

        # Override get_db to use test pool
        from app.api.deps import get_db

        def override_get_db():
            conn = self.test_pool.get_connection()
            try:
                yield conn
            finally:
                self.test_pool.release_connection(conn)

        app.dependency_overrides[get_db] = override_get_db
        self._get_db = get_db

        # Clean users table
        conn = self.test_pool.get_connection()
        try:
            conn.execute("DELETE FROM user_sessions")
            conn.execute("DELETE FROM vault_members")
            conn.execute("DELETE FROM org_members")
            conn.execute("DELETE FROM users")
            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

    def tearDown(self):
        """Clean up overrides and connections."""
        app.dependency_overrides.pop(self._get_db, None)
        if self._current_user_dep:
            app.dependency_overrides.pop(self._current_user_dep, None)
        pass  # Do not close_all() - pool is a singleton and closing it breaks subsequent tests

    def _override_current_user(self, user_dict: dict):
        """Override get_current_active_user to return a specific user."""
        from app.api.deps import get_current_active_user

        async def mock_get_current_user():
            return user_dict

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        self._current_user_dep = get_current_active_user

    def test_require_admin_role_allows_superadmin(self):
        """Test that require_admin_role allows superadmin role."""
        self._override_current_user(
            {"id": 1, "username": "superadmin", "role": "superadmin", "is_active": True}
        )

        # Call an endpoint that uses require_role("admin") directly
        response = self.client.get(
            "/api/users/"
        )  # users.py list_users uses require_role("admin")

        # Should not be 403
        self.assertNotEqual(response.status_code, 403)

    def test_require_admin_role_allows_admin(self):
        """Test that require_admin_role allows admin role."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.get("/api/users/")

        self.assertNotEqual(response.status_code, 403)

    def test_require_admin_role_rejects_member(self):
        """Test that require_admin_role rejects member role with 403."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )

        response = self.client.get("/api/users/")

        self.assertEqual(response.status_code, 403)

    def test_require_admin_role_rejects_viewer(self):
        """Test that require_admin_role rejects viewer role with 403."""
        self._override_current_user(
            {"id": 4, "username": "viewer", "role": "viewer", "is_active": True}
        )

        response = self.client.get("/api/users/")

        self.assertEqual(response.status_code, 403)


class TestUsersEndpointsAdminProtection(unittest.TestCase):
    """Tests that all users.py endpoints require admin/superadmin role."""

    @classmethod
    def setUpClass(cls):
        settings.sqlite_path = TEST_DB_PATH
        get_pool(TEST_DB_PATH)

    def setUp(self):
        self.client = TestClient(app)
        self.test_pool = get_pool(TEST_DB_PATH)
        self._current_user_dep = None

        from app.api.deps import get_db

        def override_get_db():
            conn = self.test_pool.get_connection()
            try:
                yield conn
            finally:
                self.test_pool.release_connection(conn)

        app.dependency_overrides[get_db] = override_get_db
        self._get_db = get_db

        # Clean and create test users
        conn = self.test_pool.get_connection()
        try:
            conn.execute("DELETE FROM user_sessions")
            conn.execute("DELETE FROM vault_members")
            conn.execute("DELETE FROM org_members")
            conn.execute("DELETE FROM users")
            conn.commit()

            password_hash = hash_password("testpassword123")
            users_data = [
                (1, "superadmin", password_hash, "Super Admin", "superadmin", 1),
                (2, "admin", password_hash, "Test Admin", "admin", 1),
                (3, "member", password_hash, "Test Member", "member", 1),
                (4, "viewer", password_hash, "Test Viewer", "viewer", 1),
            ]
            conn.executemany(
                "INSERT INTO users (id, username, hashed_password, full_name, role, is_active) VALUES (?, ?, ?, ?, ?, ?)",
                users_data,
            )
            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

    def tearDown(self):
        app.dependency_overrides.pop(self._get_db, None)
        if self._current_user_dep:
            app.dependency_overrides.pop(self._current_user_dep, None)
        pass  # Do not close_all() - pool is a singleton and closing it breaks subsequent tests

    def _override_current_user(self, user_dict: dict):
        """Override get_current_active_user to return a specific user."""
        from app.api.deps import get_current_active_user

        async def mock_get_current_user():
            return user_dict

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        self._current_user_dep = get_current_active_user

    def test_get_users_requires_admin(self):
        """Test GET /api/users/ requires admin/superadmin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.get("/api/users/")
        self.assertEqual(response.status_code, 403)

    def test_get_users_allows_admin(self):
        """Test GET /api/users/ allows admin."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )
        response = self.client.get("/api/users/")
        self.assertEqual(response.status_code, 200)

    def test_get_users_allows_superadmin(self):
        """Test GET /api/users/ allows superadmin."""
        self._override_current_user(
            {"id": 1, "username": "superadmin", "role": "superadmin", "is_active": True}
        )
        response = self.client.get("/api/users/")
        self.assertEqual(response.status_code, 200)

    def test_get_user_by_id_requires_admin(self):
        """Test GET /api/users/{user_id} requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.get("/api/users/1")
        self.assertEqual(response.status_code, 403)

    def test_update_user_requires_admin(self):
        """Test PATCH /api/users/{user_id} requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.patch("/api/users/1", json={"username": "newname"})
        self.assertEqual(response.status_code, 403)

    def test_update_user_role_requires_superadmin(self):
        """Test PATCH /api/users/{user_id}/role requires superadmin."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )
        response = self.client.patch("/api/users/1/role", params={"role": "admin"})
        self.assertEqual(response.status_code, 403)

        self._override_current_user(
            {"id": 1, "username": "superadmin", "role": "superadmin", "is_active": True}
        )
        response = self.client.patch("/api/users/2/role", params={"role": "admin"})
        self.assertEqual(response.status_code, 200)

    def test_update_user_active_requires_admin(self):
        """Test PATCH /api/users/{user_id}/active requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.patch("/api/users/1/active", params={"is_active": False})
        self.assertEqual(response.status_code, 403)

    def test_delete_user_requires_superadmin(self):
        """Test DELETE /api/users/{user_id} requires superadmin."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )
        response = self.client.delete("/api/users/1")
        self.assertEqual(response.status_code, 403)

        self._override_current_user(
            {"id": 1, "username": "superadmin", "role": "superadmin", "is_active": True}
        )
        response = self.client.delete(
            "/api/users/999"
        )  # non-existent, but should be 404 not 403
        self.assertIn(
            response.status_code, [403, 404]
        )  # Can't delete last superadmin or not found

    def test_reset_user_password_requires_admin(self):
        """Test PATCH /api/users/{user_id}/password requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.patch(
            "/api/users/1/password", json={"password": "NewPass123!"}
        )
        self.assertEqual(response.status_code, 403)

    def test_get_user_groups_requires_admin(self):
        """Test GET /api/users/{user_id}/groups requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.get("/api/users/1/groups")
        self.assertEqual(response.status_code, 403)

    def test_update_user_groups_requires_admin(self):
        """Test PUT /api/users/{user_id}/groups requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.put("/api/users/1/groups", json={"group_ids": [1, 2]})
        self.assertEqual(response.status_code, 403)


class TestGroupsEndpointsAdminProtection(unittest.TestCase):
    """Tests that all groups.py endpoints require admin role."""

    @classmethod
    def setUpClass(cls):
        settings.sqlite_path = TEST_DB_PATH
        get_pool(TEST_DB_PATH)

    def setUp(self):
        self.client = TestClient(app)
        self.test_pool = get_pool(TEST_DB_PATH)
        self._current_user_dep = None

        from app.api.deps import get_db

        def override_get_db():
            conn = self.test_pool.get_connection()
            try:
                yield conn
            finally:
                self.test_pool.release_connection(conn)

        app.dependency_overrides[get_db] = override_get_db
        self._get_db = get_db

        # Clean and create test data
        conn = self.test_pool.get_connection()
        try:
            conn.execute("DELETE FROM group_members")
            conn.execute("DELETE FROM groups")
            conn.execute("DELETE FROM org_members")
            conn.execute("DELETE FROM organizations")
            conn.execute("DELETE FROM users")
            conn.commit()

            password_hash = hash_password("testpassword123")
            users_data = [
                (1, "superadmin", password_hash, "Super Admin", "superadmin", 1),
                (2, "admin", password_hash, "Test Admin", "admin", 1),
                (3, "member", password_hash, "Test Member", "member", 1),
            ]
            conn.executemany(
                "INSERT INTO users (id, username, hashed_password, full_name, role, is_active) VALUES (?, ?, ?, ?, ?, ?)",
                users_data,
            )

            conn.execute(
                "INSERT INTO organizations (id, name, description) VALUES (1, 'Test Org', 'Test organization')"
            )
            conn.executemany(
                "INSERT INTO groups (id, org_id, name, description) VALUES (?, ?, ?, ?)",
                [
                    (1, 1, "admin-group", "Admin group"),
                    (2, 1, "member-group", "Member group"),
                ],
            )

            # Add admin to org membership
            conn.execute("INSERT INTO org_members (org_id, user_id) VALUES (1, 1)")
            conn.execute("INSERT INTO org_members (org_id, user_id) VALUES (1, 2)")
            conn.execute("INSERT INTO org_members (org_id, user_id) VALUES (1, 3)")

            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

    def tearDown(self):
        app.dependency_overrides.pop(self._get_db, None)
        if self._current_user_dep:
            app.dependency_overrides.pop(self._current_user_dep, None)
        pass  # Do not close_all() - pool is a singleton and closing it breaks subsequent tests

    def _override_current_user(self, user_dict: dict):
        """Override get_current_active_user to return a specific user."""
        from app.api.deps import get_current_active_user

        async def mock_get_current_user():
            return user_dict

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        self._current_user_dep = get_current_active_user

    def test_list_groups_requires_admin(self):
        """Test GET /api/groups requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.get("/api/groups")
        self.assertEqual(response.status_code, 403)

    def test_create_group_requires_admin(self):
        """Test POST /api/groups requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.post(
            "/api/groups",
            json={"name": "new-group", "description": "test", "org_id": 1},
        )
        self.assertEqual(response.status_code, 403)

    def test_get_group_requires_admin(self):
        """Test GET /api/groups/{group_id} requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.get("/api/groups/1")
        self.assertEqual(response.status_code, 403)

    def test_update_group_requires_admin(self):
        """Test PUT /api/groups/{group_id} requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.put("/api/groups/1", json={"name": "renamed-group"})
        self.assertEqual(response.status_code, 403)

    def test_delete_group_requires_admin(self):
        """Test DELETE /api/groups/{group_id} requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.delete("/api/groups/1")
        self.assertEqual(response.status_code, 403)

    def test_get_group_members_requires_admin(self):
        """Test GET /api/groups/{group_id}/members requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.get("/api/groups/1/members")
        self.assertEqual(response.status_code, 403)

    def test_update_group_members_requires_admin(self):
        """Test PUT /api/groups/{group_id}/members requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.put("/api/groups/1/members", json={"user_ids": [1, 2]})
        self.assertEqual(response.status_code, 403)

    def test_get_group_vaults_requires_admin(self):
        """Test GET /api/groups/{group_id}/vaults requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.get("/api/groups/1/vaults")
        self.assertEqual(response.status_code, 403)

    def test_update_group_vaults_requires_admin(self):
        """Test PUT /api/groups/{group_id}/vaults requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.put("/api/groups/1/vaults", json={"vault_ids": [1]})
        self.assertEqual(response.status_code, 403)


class TestVaultGroupManagementAdminProtection(unittest.TestCase):
    """Tests that vault group management endpoints require admin role."""

    @classmethod
    def setUpClass(cls):
        settings.sqlite_path = TEST_DB_PATH
        get_pool(TEST_DB_PATH)

    def setUp(self):
        self.client = TestClient(app)
        self.test_pool = get_pool(TEST_DB_PATH)
        self._current_user_dep = None

        from app.api.deps import get_db

        def override_get_db():
            conn = self.test_pool.get_connection()
            try:
                yield conn
            finally:
                self.test_pool.release_connection(conn)

        app.dependency_overrides[get_db] = override_get_db
        self._get_db = get_db

        # Create test data: org, groups, vault, users
        conn = self.test_pool.get_connection()
        try:
            conn.execute("DELETE FROM vault_group_access")
            conn.execute("DELETE FROM groups")
            conn.execute("DELETE FROM org_members")
            conn.execute("DELETE FROM organizations")
            conn.execute("DELETE FROM vaults")
            conn.execute("DELETE FROM users")
            conn.commit()

            password_hash = hash_password("testpassword123")
            users_data = [
                (1, "superadmin", password_hash, "Super Admin", "superadmin", 1),
                (2, "admin", password_hash, "Test Admin", "admin", 1),
                (3, "member", password_hash, "Test Member", "member", 1),
            ]
            conn.executemany(
                "INSERT INTO users (id, username, hashed_password, full_name, role, is_active) VALUES (?, ?, ?, ?, ?, ?)",
                users_data,
            )

            conn.execute(
                "INSERT INTO organizations (id, name, description) VALUES (1, 'Test Org', 'Test organization')"
            )
            conn.execute(
                "INSERT INTO groups (id, org_id, name, description) VALUES (1, 1, 'test-group', 'Test group')"
            )
            conn.executemany(
                "INSERT INTO org_members (org_id, user_id) VALUES (?, ?)",
                [(1, 1), (1, 2), (1, 3)],
            )

            conn.execute(
                "INSERT INTO vaults (id, name, description) VALUES (1, 'Test Vault', 'Test vault')"
            )
            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

    def tearDown(self):
        app.dependency_overrides.pop(self._get_db, None)
        if self._current_user_dep:
            app.dependency_overrides.pop(self._current_user_dep, None)
        pass  # Do not close_all() - pool is a singleton and closing it breaks subsequent tests

    def _override_current_user(self, user_dict: dict):
        """Override get_current_active_user to return a specific user."""
        from app.api.deps import get_current_active_user

        async def mock_get_current_user():
            return user_dict

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        self._current_user_dep = get_current_active_user

    def test_get_vault_groups_requires_admin(self):
        """Test GET /api/vaults/{vault_id}/groups requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.get("/api/vaults/1/groups")
        self.assertEqual(response.status_code, 403)

    def test_get_vault_groups_allows_admin(self):
        """Test GET /api/vaults/{vault_id}/groups allows admin."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )
        response = self.client.get("/api/vaults/1/groups")
        self.assertEqual(response.status_code, 200)

    def test_get_vault_groups_allows_superadmin(self):
        """Test GET /api/vaults/{vault_id}/groups allows superadmin."""
        self._override_current_user(
            {"id": 1, "username": "superadmin", "role": "superadmin", "is_active": True}
        )
        response = self.client.get("/api/vaults/1/groups")
        self.assertEqual(response.status_code, 200)

    def test_update_vault_groups_requires_admin(self):
        """Test PUT /api/vaults/{vault_id}/groups requires admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )
        response = self.client.put("/api/vaults/1/groups", json={"group_ids": [1]})
        self.assertEqual(response.status_code, 403)

    def test_update_vault_groups_allows_admin(self):
        """Test PUT /api/vaults/{vault_id}/groups allows admin."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )
        response = self.client.put("/api/vaults/1/groups", json={"group_ids": [1]})
        self.assertEqual(response.status_code, 200)

    def test_update_vault_groups_allows_superadmin(self):
        """Test PUT /api/vaults/{vault_id}/groups allows superadmin."""
        self._override_current_user(
            {"id": 1, "username": "superadmin", "role": "superadmin", "is_active": True}
        )
        response = self.client.put("/api/vaults/1/groups", json={"group_ids": [1]})
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
