"""
User management endpoints tests using unittest and FastAPI TestClient.

Tests cover:
- GET /users/ (list users with groups and last_login_at)
- PATCH /users/{user_id} (update username/full_name)
- PATCH /users/{user_id}/password (admin password reset)
- GET /users/{user_id}/groups (get user's groups)
- PUT /users/{user_id}/groups (update user's groups)
- GET /groups (list all groups with pagination)

All tests include role-based access control verification and error handling.
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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

from fastapi.testclient import TestClient

from app.api.deps import get_llm_health_checker, get_model_checker

# Create a temporary database for testing
TEST_DB_PATH = None
TEST_DATA_DIR = None


def setup_test_db():
    """Set up a temporary test database."""
    global TEST_DB_PATH, TEST_DATA_DIR
    TEST_DATA_DIR = tempfile.mkdtemp()
    TEST_DB_PATH = Path(TEST_DATA_DIR) / "test.db"

    # Import and initialize the database
    from app.models.database import init_db

    init_db(str(TEST_DB_PATH))
    return str(TEST_DB_PATH)


def get_test_settings():
    """Get test settings with temporary database path."""
    from app.config import Settings

    settings = Settings()
    settings.data_dir = Path(TEST_DATA_DIR)
    return settings


# Set up test database before importing app
setup_test_db()

from app.main import app
from app.config import settings


class TestUserManagementEndpoints(unittest.TestCase):
    """Tests for user and group management endpoints."""

    def setUp(self):
        """Set up test client and database fixtures."""
        self.client = TestClient(app)

        # Create a connection pool for test data setup
        from app.models.database import SQLiteConnectionPool

        self.test_pool = SQLiteConnectionPool(str(TEST_DB_PATH), max_size=3)

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

        # Create test data: organizations, groups, users, and user_groups
        self._create_test_data()

    def tearDown(self):
        """Clean up test data and dependencies."""
        app.dependency_overrides.pop(self._get_db, None)
        self.test_pool.close_all()

        # Clean up test data
        conn = sqlite3.connect(str(TEST_DB_PATH))
        try:
            conn.execute("DELETE FROM user_groups")
            conn.execute("DELETE FROM groups")
            conn.execute("DELETE FROM org_members")
            conn.execute("DELETE FROM organizations")
            conn.execute("DELETE FROM users")
            conn.commit()
        finally:
            conn.close()

    def _create_test_data(self):
        """Create test organizations, groups, users, and user_sessions."""
        conn = sqlite3.connect(str(TEST_DB_PATH))
        try:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")

            # Create organization
            conn.execute(
                "INSERT INTO organizations (id, name, description) VALUES (1, 'Test Org', 'Test organization')"
            )

            # Create groups
            conn.executemany(
                "INSERT INTO groups (id, org_id, name, description) VALUES (?, ?, ?, ?)",
                [
                    (1, 1, "admin-group", "Admin group"),
                    (2, 1, "member-group", "Member group"),
                    (3, 1, "viewer-group", "Viewer group"),
                ],
            )

            # Create users with different roles
            from app.services.auth_service import hash_password

            password_hash = hash_password("testpassword123")

            users_data = [
                (1, "superadmin", password_hash, "Super Admin", "superadmin", 1),
                (2, "admin", password_hash, "Test Admin", "admin", 1),
                (3, "member", password_hash, "Test Member", "member", 1),
                (4, "viewer", password_hash, "Test Viewer", "viewer", 1),
                (5, "inactive", password_hash, "Inactive User", "member", 0),
            ]

            conn.executemany(
                "INSERT INTO users (id, username, hashed_password, full_name, role, is_active) VALUES (?, ?, ?, ?, ?, ?)",
                users_data,
            )

            # Create user_groups memberships
            user_groups = [
                (1, 1),  # superadmin in admin-group
                (2, 1),  # admin in admin-group
                (3, 2),  # member in member-group
                (4, 3),  # viewer in viewer-group
            ]

            conn.executemany(
                "INSERT INTO user_groups (user_id, group_id) VALUES (?, ?)", user_groups
            )

            # Create some user sessions for last_login_at testing
            conn.executemany(
                "INSERT INTO user_sessions (user_id, refresh_token_hash, expires_at) VALUES (?, ?, ?)",
                [
                    (1, "hash1", "2025-01-01 00:00:00"),
                    (2, "hash2", "2025-01-02 00:00:00"),
                    (3, "hash3", "2025-01-03 00:00:00"),
                ],
            )

            conn.commit()
        finally:
            conn.close()

    def _get_auth_headers(self, username: str) -> dict:
        """Helper to get authorization headers for a user."""
        # For testing, we'll use a mock token that the test client will accept
        # In the real app, this would be a valid JWT, but for tests we'll
        # bypass by mocking get_current_active_user when needed

        # We'll create a simple approach: for tests that need specific users,
        # we'll override get_current_active_user directly

        return {"Authorization": f"Bearer mock-token-{username}"}

    def _override_current_user(self, user_dict: dict):
        """Override get_current_active_user to return a specific user."""
        from app.api.deps import get_current_active_user

        async def mock_get_current_user():
            return user_dict

        app.dependency_overrides[get_current_active_user] = mock_get_current_user

    def test_list_users_requires_admin(self):
        """Test GET /users/ returns 403 for non-admin users."""
        # Override with a member user (not admin)
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )

        response = self.client.get("/users/")
        self.assertEqual(response.status_code, 403)
        self.assertIn("Insufficient privileges", response.json()["detail"])

    def test_list_users_returns_users_with_groups_and_last_login(self):
        """Test GET /users/ returns list with groups and last_login_at for admin."""
        # Override as admin
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.get("/users/")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify structure
        self.assertIn("users", data)
        self.assertIn("total", data)
        self.assertEqual(data["total"], 5)  # 5 total users

        users = data["users"]
        self.assertEqual(len(users), 5)  # Default limit is 100, so all 5

        # Find superadmin user
        superadmin = next(u for u in users if u["username"] == "superadmin")
        self.assertEqual(superadmin["role"], "superadmin")
        self.assertEqual(superadmin["groups"], ["admin-group"])
        self.assertIsNotNone(superadmin["last_login_at"])

        # Find member user
        member = next(u for u in users if u["username"] == "member")
        self.assertEqual(member["groups"], ["member-group"])

        # Find inactive user
        inactive = next(u for u in users if u["username"] == "inactive")
        self.assertEqual(inactive["is_active"], False)

    def test_list_users_with_pagination(self):
        """Test GET /users/ with skip and limit parameters."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.get("/users/?skip=2&limit=2")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(len(data["users"]), 2)
        self.assertEqual(data["total"], 5)

        # Users should be ordered by id (as per query ORDER BY id)
        usernames = [u["username"] for u in data["users"]]
        self.assertEqual(usernames, ["member", "viewer"])  # IDs 3,4

    def test_list_users_empty(self):
        """Test GET /users/ when no users exist."""
        # Delete all users first
        conn = sqlite3.connect(str(TEST_DB_PATH))
        try:
            conn.execute("DELETE FROM users")
            conn.commit()
        finally:
            conn.close()

        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.get("/users/")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["users"], [])
        self.assertEqual(data["total"], 0)

    def test_update_user_username_requires_admin(self):
        """Test PATCH /users/{user_id} returns 403 for non-admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )

        response = self.client.patch("/users/1", json={"username": "newname"})
        self.assertEqual(response.status_code, 403)

    def test_update_user_username_success(self):
        """Test PATCH /users/{user_id} updates username."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.patch("/users/3", json={"username": "updated_member"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["message"], "User updated")
        self.assertEqual(data["username"], "updated_member")
        self.assertEqual(data["user_id"], 3)

        # Verify in database
        conn = sqlite3.connect(str(TEST_DB_PATH))
        try:
            cursor = conn.execute("SELECT username FROM users WHERE id = 3")
            username = cursor.fetchone()[0]
            self.assertEqual(username, "updated_member")
        finally:
            conn.close()

    def test_update_user_username_duplicate(self):
        """Test PATCH /users/{user_id} rejects duplicate username."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.patch(
            "/users/3",
            json={"username": "superadmin"},  # Already exists
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Username already taken", response.json()["detail"])

    def test_update_user_full_name(self):
        """Test PATCH /users/{user_id} updates full_name."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.patch(
            "/users/4", json={"full_name": "Updated Viewer Name"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["full_name"], "Updated Viewer Name")

        # Verify in database
        conn = sqlite3.connect(str(TEST_DB_PATH))
        try:
            cursor = conn.execute("SELECT full_name FROM users WHERE id = 4")
            full_name = cursor.fetchone()[0]
            self.assertEqual(full_name, "Updated Viewer Name")
        finally:
            conn.close()

    def test_update_user_no_fields(self):
        """Test PATCH /users/{user_id} with no fields returns 400."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.patch("/users/3", json={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("No fields to update", response.json()["detail"])

    def test_update_user_not_found(self):
        """Test PATCH /users/{user_id} with non-existent ID returns 404."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.patch("/users/999", json={"username": "nonexistent"})

        self.assertEqual(response.status_code, 404)

    def test_reset_user_password_requires_admin(self):
        """Test PATCH /users/{user_id}/password returns 403 for non-admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )

        response = self.client.patch(
            "/users/1/password", json={"password": "NewPass123!"}
        )
        self.assertEqual(response.status_code, 403)

    def test_reset_user_password_weak_password(self):
        """Test PATCH /users/{user_id}/password rejects weak passwords."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.patch(
            "/users/3/password",
            json={"password": "weak"},  # Too short
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "Password must be at least 8 characters", response.json()["detail"]
        )

    def test_reset_user_password_success(self):
        """Test PATCH /users/{user_id}/password successfully resets password."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        new_password = "StrongPass123!"
        response = self.client.patch(
            "/users/3/password", json={"password": new_password}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["message"], "Password reset successfully")
        self.assertEqual(data["user_id"], 3)

        # Verify password was changed by trying to log in
        # We can't verify directly without hashing knowledge, but we can
        # ensure the user can still authenticate
        login_response = self.client.post(
            "/api/v1/auth/login", data={"username": "member", "password": new_password}
        )
        # If password didn't change, login would fail with old password
        # But we can't test old password since we didn't store it
        # Instead, we'll check the database hash changed
        conn = sqlite3.connect(str(TEST_DB_PATH))
        try:
            cursor = conn.execute("SELECT hashed_password FROM users WHERE id = 3")
            hash1 = cursor.fetchone()[0]

            # Hash should be different from initial
            from app.services.auth_service import hash_password

            old_hash = hash_password("testpassword123")
            self.assertNotEqual(hash1, old_hash)

            # Verify new hash is correct by testing known password
            import bcrypt

            self.assertTrue(
                bcrypt.checkpw(new_password.encode("utf-8"), hash1.encode("utf-8"))
            )
        finally:
            conn.close()

    def test_reset_user_password_user_not_found(self):
        """Test PATCH /users/{user_id}/password with non-existent user returns 404."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.patch(
            "/users/999/password", json={"password": "SomePass123!"}
        )

        self.assertEqual(response.status_code, 404)

    def test_get_user_groups_requires_admin(self):
        """Test GET /users/{user_id}/groups returns 403 for non-admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )

        response = self.client.get("/users/1/groups")
        self.assertEqual(response.status_code, 403)

    def test_get_user_groups_returns_groups(self):
        """Test GET /users/{user_id}/groups returns user's groups."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.get("/users/1/groups")  # superadmin

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("groups", data)
        groups = data["groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["id"], 1)
        self.assertEqual(groups[0]["name"], "admin-group")

    def test_get_user_groups_empty(self):
        """Test GET /users/{user_id}/groups for user with no groups."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.get("/users/5/groups")  # inactive user has no groups

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["groups"], [])

    def test_get_user_groups_user_not_found(self):
        """Test GET /users/{user_id}/groups with non-existent user returns 404."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.get("/users/999/groups")
        self.assertEqual(response.status_code, 404)

    def test_update_user_groups_requires_admin(self):
        """Test PUT /users/{user_id}/groups returns 403 for non-admin."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )

        response = self.client.put("/users/1/groups", json={"group_ids": [2, 3]})
        self.assertEqual(response.status_code, 403)

    def test_update_user_groups_valid(self):
        """Test PUT /users/{user_id}/groups updates group memberships."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        # member (user_id=3) initially has member-group (id=2)
        # Update to viewer-group (id=3) and admin-group (id=1)
        response = self.client.put("/users/3/groups", json={"group_ids": [1, 3]})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["message"], "User groups updated")
        self.assertEqual(data["user_id"], 3)
        self.assertEqual(sorted(data["group_ids"]), [1, 3])

        # Verify in database
        conn = sqlite3.connect(str(TEST_DB_PATH))
        try:
            cursor = conn.execute(
                "SELECT group_id FROM user_groups WHERE user_id = 3 ORDER BY group_id"
            )
            group_ids = [row[0] for row in cursor.fetchall()]
            self.assertEqual(group_ids, [1, 3])
        finally:
            conn.close()

    def test_update_user_groups_clear_all(self):
        """Test PUT /users/{user_id}/groups with empty list removes all groups."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        # Clear all groups for member
        response = self.client.put("/users/3/groups", json={"group_ids": []})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["group_ids"], [])

        # Verify in database
        conn = sqlite3.connect(str(TEST_DB_PATH))
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM user_groups WHERE user_id = 3")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 0)
        finally:
            conn.close()

    def test_update_user_groups_invalid_group_ids(self):
        """Test PUT /users/{user_id}/groups rejects invalid group IDs."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.put(
            "/users/3/groups",
            json={"group_ids": [1, 999]},  # 999 doesn't exist
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid group IDs", response.json()["detail"])
        self.assertIn("999", response.json()["detail"])

    def test_update_user_groups_user_not_found(self):
        """Test PUT /users/{user_id}/groups with non-existent user returns 404."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.put("/users/999/groups", json={"group_ids": [1, 2]})

        self.assertEqual(response.status_code, 404)

    def test_get_groups_requires_admin(self):
        """Test GET /groups returns 403 for non-admin users."""
        self._override_current_user(
            {"id": 3, "username": "member", "role": "member", "is_active": True}
        )

        response = self.client.get("/groups")
        self.assertEqual(response.status_code, 403)

    def test_get_groups_returns_groups_with_org_name(self):
        """Test GET /groups returns groups with organization details."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.get("/groups")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("groups", data)
        self.assertIn("total", data)
        self.assertEqual(data["total"], 3)

        groups = data["groups"]
        self.assertEqual(len(groups), 3)

        # Verify group structure
        group_names = {g["name"] for g in groups}
        self.assertEqual(group_names, {"admin-group", "member-group", "viewer-group"})

        for group in groups:
            self.assertIn("id", group)
            self.assertIn("org_id", group)
            self.assertIn("name", group)
            self.assertIn("description", group)
            self.assertIn("created_at", group)
            self.assertIn("organization_name", group)
            self.assertEqual(group["organization_name"], "Test Org")

    def test_get_groups_with_pagination(self):
        """Test GET /groups with skip and limit."""
        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.get("/groups?skip=1&limit=1")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(len(data["groups"]), 1)
        self.assertEqual(data["total"], 3)
        self.assertEqual(data["groups"][0]["name"], "member-group")

    def test_get_groups_empty(self):
        """Test GET /groups when no groups exist."""
        # Delete all groups
        conn = sqlite3.connect(str(TEST_DB_PATH))
        try:
            conn.execute("DELETE FROM user_groups")
            conn.execute("DELETE FROM groups")
            conn.commit()
        finally:
            conn.close()

        self._override_current_user(
            {"id": 2, "username": "admin", "role": "admin", "is_active": True}
        )

        response = self.client.get("/groups")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["groups"], [])
        self.assertEqual(data["total"], 0)


class TestAdversarialUserManagement(unittest.TestCase):
    """Adversarial/security-focused tests for user management endpoints."""

    def setUp(self):
        """Set up test client and basic data."""
        self.client = TestClient(app)

        from app.models.database import SQLiteConnectionPool

        self.test_pool = SQLiteConnectionPool(str(TEST_DB_PATH), max_size=3)

        from app.api.deps import get_db

        def override_get_db():
            conn = self.test_pool.get_connection()
            try:
                yield conn
            finally:
                self.test_pool.release_connection(conn)

        app.dependency_overrides[get_db] = override_get_db
        self._get_db = get_db

    def tearDown(self):
        """Clean up."""
        app.dependency_overrides.pop(self._get_db, None)
        self.test_pool.close_all()

    def test_update_user_username_oversized(self):
        """Test PATCH /users/{user_id} with extremely long username."""
        self._override_current_user = lambda x: (
            app.dependency_overrides.__setitem__("get_current_active_user", lambda: x)
            if hasattr(app.dependency_overrides, "__setitem__")
            else None
        )

        # Actually set override directly
        from app.api.deps import get_current_active_user

        async def mock_user():
            return {"id": 2, "username": "admin", "role": "admin", "is_active": True}

        app.dependency_overrides[get_current_active_user] = mock_user

        long_username = "a" * 1000
        response = self.client.patch("/users/3", json={"username": long_username})

        # Should either reject or truncate, but not crash
        self.assertIn(response.status_code, [200, 400, 422])
        if response.status_code == 200:
            # If successful, verify the username is reasonable length
            data = response.json()
            self.assertLess(len(data["username"]), 256)

    def test_reset_password_with_sql_injection(self):
        """Test PATCH /users/{user_id}/password rejects SQL injection in password."""
        from app.api.deps import get_current_active_user

        async def mock_user():
            return {"id": 2, "username": "admin", "role": "admin", "is_active": True}

        app.dependency_overrides[get_current_active_user] = mock_user

        # Try SQL injection patterns in password
        injection_passwords = [
            "'; DROP TABLE users; --",
            "admin' OR '1'='1",
            "\x00' OR 1=1 --",
        ]

        for pwd in injection_passwords:
            response = self.client.patch("/users/3/password", json={"password": pwd})
            # Should reject as weak password or process safely
            self.assertNotEqual(response.status_code, 500)
            if response.status_code == 400:
                # Weak password error is acceptable
                self.assertIn("password", response.json()["detail"].lower())

    def test_list_users_sql_injection_in_skip_limit(self):
        """Test GET /users/ with SQL injection in query params."""
        from app.api.deps import get_current_active_user

        async def mock_user():
            return {"id": 2, "username": "admin", "role": "admin", "is_active": True}

        app.dependency_overrides[get_current_active_user] = mock_user

        response = self.client.get("/users/?skip=0; DROP TABLE users; --")

        # Should handle safely
        self.assertIn(response.status_code, [200, 422, 400])
        # Verify users table still intact
        conn = sqlite3.connect(str(TEST_DB_PATH))
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM users")
            count = cursor.fetchone()[0]
            self.assertGreater(count, 0)
        finally:
            conn.close()

    def test_update_user_groups_with_duplicate_group_ids(self):
        """Test PUT /users/{user_id}/groups with duplicate group IDs."""
        from app.api.deps import get_current_active_user

        async def mock_user():
            return {"id": 2, "username": "admin", "role": "admin", "is_active": True}

        app.dependency_overrides[get_current_active_user] = mock_user

        response = self.client.put(
            "/users/3/groups",
            json={"group_ids": [1, 2, 2, 3]},  # duplicate 2
        )

        # Should either deduplicate or reject; at minimum should not crash
        self.assertIn(response.status_code, [200, 400])
        if response.status_code == 200:
            # Verify no duplicates in database
            conn = sqlite3.connect(str(TEST_DB_PATH))
            try:
                cursor = conn.execute(
                    "SELECT group_id FROM user_groups WHERE user_id = 3 ORDER BY group_id"
                )
                group_ids = [row[0] for row in cursor.fetchall()]
                self.assertEqual(group_ids, sorted(set([1, 2, 3])))
            finally:
                conn.close()

    def test_update_user_groups_with_negative_group_id(self):
        """Test PUT /users/{user_id}/groups with negative group ID."""
        from app.api.deps import get_current_active_user

        async def mock_user():
            return {"id": 2, "username": "admin", "role": "admin", "is_active": True}

        app.dependency_overrides[get_current_active_user] = mock_user

        response = self.client.put("/users/3/groups", json={"group_ids": [-1]})

        # Should reject as invalid group ID
        self.assertIn(response.status_code, [400, 404])

    def test_get_user_groups_with_nonexistent_user(self):
        """Test GET /users/{user_id}/groups with extremely large user_id."""
        from app.api.deps import get_current_active_user

        async def mock_user():
            return {"id": 2, "username": "admin", "role": "admin", "is_active": True}

        app.dependency_overrides[get_current_active_user] = mock_user

        response = self.client.get("/users/999999999/groups")
        self.assertEqual(response.status_code, 404)

    def test_reset_password_empty_body(self):
        """Test PATCH /users/{user_id}/password with empty JSON body."""
        from app.api.deps import get_current_active_user

        async def mock_user():
            return {"id": 2, "username": "admin", "role": "admin", "is_active": True}

        app.dependency_overrides[get_current_active_user] = mock_user

        response = self.client.patch("/users/3/password", json={})
        self.assertEqual(
            response.status_code, 422
        )  # Unprocessable Entity (validation error)

    def test_update_user_with_injectable_username(self):
        """Test PATCH /users/{user_id} with username containing special chars."""
        from app.api.deps import get_current_active_user

        async def mock_user():
            return {"id": 2, "username": "admin", "role": "admin", "is_active": True}

        app.dependency_overrides[get_current_active_user] = mock_user

        malicious_usernames = [
            "../../../etc/passwd",
            "<script>alert('xss')</script>",
            "admin\x00admin",
            "user\r\nadmin:true",
        ]

        for username in malicious_usernames:
            response = self.client.patch("/users/3", json={"username": username})
            # Should either accept (if allowed by validation) or reject, but not crash
            self.assertIn(response.status_code, [200, 400, 422])


if __name__ == "__main__":
    unittest.main()
