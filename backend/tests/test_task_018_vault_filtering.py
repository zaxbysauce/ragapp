"""
Tests for Task 1.8: Vault Filtering Implementation

Tests cover:
- get_user_accessible_vault_ids function
- GET /vaults/accessible endpoint
- chat/stream vault_id validation
- Authorization enforcement for regular users and admins
"""

import asyncio
import os
import sqlite3
import sys
import tempfile
import shutil
import unittest
from pathlib import Path
from queue import Empty, Queue
from unittest.mock import MagicMock, AsyncMock

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

from app.main import app
from app.models.database import init_db, get_pool
from app.api.deps import (
    get_db,
    get_rag_engine,
    get_current_active_user,
    get_user_accessible_vault_ids,
)


class SimpleConnectionPool:
    """Simple connection pool for testing."""

    def __init__(self, db_path):
        self.db_path = db_path
        self._pool = Queue(maxsize=5)
        self._lock = type(
            "Lock", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None}
        )()
        self._closed = False

    def get_connection(self):
        if self._closed:
            raise RuntimeError("Pool closed")
        try:
            return self._pool.get_nowait()
        except Empty:
            return self._create_connection()

    def _create_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def release_connection(self, conn):
        if not self._closed:
            try:
                self._pool.put_nowait(conn)
            except:
                conn.close()

    def close_all(self):
        self._closed = True
        while True:
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break


class TestGetUserAccessibleVaultIds(unittest.TestCase):
    """Test suite for get_user_accessible_vault_ids function."""

    @classmethod
    def setUpClass(cls):
        """Set up test database."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = str(Path(cls.temp_dir) / "test.db")
        init_db(cls.db_path)
        cls.pool = SimpleConnectionPool(cls.db_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up test database."""
        cls.pool.close_all()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """Set up fresh connection for each test."""
        self.conn = self.pool.get_connection()

    def tearDown(self):
        """Release connection and reset tables."""
        # Clear tables for fresh state
        self.conn.execute("DELETE FROM vault_members")
        self.conn.execute("DELETE FROM vault_group_access")
        self.conn.execute("DELETE FROM group_members")
        self.conn.execute("DELETE FROM groups")
        self.conn.execute("DELETE FROM organizations")
        self.conn.execute("DELETE FROM vaults")
        self.conn.execute("DELETE FROM users")
        self.conn.commit()
        self.pool.release_connection(self.conn)

    def _create_user(self, username, role="member"):
        """Helper to create a user."""
        cursor = self.conn.execute(
            "INSERT INTO users (username, hashed_password, role, is_active) VALUES (?, ?, ?, 1)",
            (username, "hash123", role),
        )
        self.conn.commit()
        return {
            "id": cursor.lastrowid,
            "username": username,
            "role": role,
            "is_active": True,
        }

    def _create_vault(self, name):
        """Helper to create a vault."""
        cursor = self.conn.execute(
            "INSERT INTO vaults (name, description) VALUES (?, '')", (name,)
        )
        self.conn.commit()
        return cursor.lastrowid

    def _add_vault_member(self, user_id, vault_id, permission="read"):
        """Helper to add user to vault_members."""
        self.conn.execute(
            "INSERT INTO vault_members (vault_id, user_id, permission) VALUES (?, ?, ?)",
            (vault_id, user_id, permission),
        )
        self.conn.commit()

    def _create_group(self, name):
        """Helper to create a group (creates an org if needed)."""
        # Ensure an org exists for the group
        self.conn.execute(
            "INSERT OR IGNORE INTO organizations (id, name) VALUES (1, 'Test Org')"
        )
        self.conn.commit()
        cursor = self.conn.execute(
            "INSERT INTO groups (org_id, name) VALUES (1, ?)", (name,)
        )
        self.conn.commit()
        return cursor.lastrowid

    def _add_to_group(self, user_id, group_id):
        """Helper to add user to group."""
        self.conn.execute(
            "INSERT INTO group_members (user_id, group_id) VALUES (?, ?)",
            (user_id, group_id),
        )
        self.conn.commit()

    def _add_group_vault_access(self, group_id, vault_id, permission="read"):
        """Helper to add group vault access."""
        self.conn.execute(
            "INSERT INTO vault_group_access (vault_id, group_id, permission) VALUES (?, ?, ?)",
            (vault_id, group_id, permission),
        )
        self.conn.commit()

    def test_user_with_no_vaults_returns_empty_list(self):
        """Test that a regular user with no vault access returns empty list."""
        user = self._create_user("regular_user", role="member")
        vault_ids = asyncio.run(get_user_accessible_vault_ids(user, self.conn))
        self.assertEqual(vault_ids, [])

    def test_user_with_direct_vault_access(self):
        """Test that a user with direct vault_members access gets those vault IDs."""
        user = self._create_user("user_with_vaults", role="member")
        vault1 = self._create_vault("Vault 1")
        vault2 = self._create_vault("Vault 2")

        self._add_vault_member(user["id"], vault1, "read")
        self._add_vault_member(user["id"], vault2, "write")

        vault_ids = asyncio.run(get_user_accessible_vault_ids(user, self.conn))
        self.assertEqual(set(vault_ids), {vault1, vault2})

    def test_admin_returns_empty_list(self):
        """Test that admin role returns empty list (means all vaults)."""
        user = self._create_user("admin_user", role="admin")
        vault_ids = asyncio.run(get_user_accessible_vault_ids(user, self.conn))
        # Empty list means "all vaults" for admin/superadmin
        self.assertEqual(vault_ids, [])

    def test_superadmin_returns_empty_list(self):
        """Test that superadmin role returns empty list (means all vaults)."""
        user = self._create_user("superadmin_user", role="superadmin")
        vault_ids = asyncio.run(get_user_accessible_vault_ids(user, self.conn))
        # Empty list means "all vaults" for admin/superadmin
        self.assertEqual(vault_ids, [])

    def test_user_with_group_vault_access(self):
        """Test that group-based vault access is included."""
        user = self._create_user("group_user", role="member")
        group = self._create_group("Test Group")
        vault = self._create_vault("Group Vault")

        self._add_to_group(user["id"], group)
        self._add_group_vault_access(group, vault, "read")

        vault_ids = asyncio.run(get_user_accessible_vault_ids(user, self.conn))
        self.assertEqual(vault_ids, [vault])

    def test_user_with_mixed_access(self):
        """Test that both direct and group access are combined."""
        user = self._create_user("mixed_user", role="member")
        group = self._create_group("Mixed Group")
        vault1 = self._create_vault("Direct Vault")
        vault2 = self._create_vault("Group Vault")

        self._add_vault_member(user["id"], vault1, "read")
        self._add_to_group(user["id"], group)
        self._add_group_vault_access(group, vault2, "write")

        vault_ids = asyncio.run(get_user_accessible_vault_ids(user, self.conn))
        self.assertEqual(set(vault_ids), {vault1, vault2})


class TestVaultsAccessibleEndpoint(unittest.TestCase):
    """Test suite for GET /vaults/accessible endpoint."""

    @classmethod
    def setUpClass(cls):
        """Set up test database."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = str(Path(cls.temp_dir) / "test.db")
        init_db(cls.db_path)
        cls.pool = SimpleConnectionPool(cls.db_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up test database."""
        cls.pool.close_all()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """Set up test client and dependency overrides."""
        self.client = TestClient(app)

        def override_get_db():
            conn = self.pool.get_connection()
            try:
                yield conn
            finally:
                self.pool.release_connection(conn)

        # Mock RAG engine
        self.mock_rag_engine = MagicMock()

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_rag_engine] = lambda: self.mock_rag_engine

    def tearDown(self):
        """Clean up dependency overrides and tables."""
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_rag_engine, None)

        conn = self.pool.get_connection()
        conn.execute("DELETE FROM vault_members")
        conn.execute("DELETE FROM vault_group_access")
        conn.execute("DELETE FROM group_members")
        conn.execute("DELETE FROM vaults")
        conn.execute("DELETE FROM users")
        conn.commit()
        self.pool.release_connection(conn)

    def _create_user(self, username, role="member"):
        """Helper to create a user."""
        conn = self.pool.get_connection()
        cursor = conn.execute(
            "INSERT INTO users (username, hashed_password, role, is_active) VALUES (?, ?, ?, 1)",
            (username, "hash123", role),
        )
        conn.commit()
        user_id = cursor.lastrowid
        self.pool.release_connection(conn)

        from app.services.auth_service import create_access_token

        token = create_access_token(user_id, username, role)
        return {"id": user_id, "username": username, "role": role, "token": token}

    def _create_vault(self, name):
        """Helper to create a vault."""
        conn = self.pool.get_connection()
        cursor = conn.execute(
            "INSERT INTO vaults (name, description) VALUES (?, '')", (name,)
        )
        conn.commit()
        vault_id = cursor.lastrowid
        self.pool.release_connection(conn)
        return vault_id

    def _add_vault_member(self, user_id, vault_id, permission="read"):
        """Helper to add user to vault_members."""
        conn = self.pool.get_connection()
        conn.execute(
            "INSERT INTO vault_members (vault_id, user_id, permission) VALUES (?, ?, ?)",
            (vault_id, user_id, permission),
        )
        conn.commit()
        self.pool.release_connection(conn)

    def test_accessible_returns_empty_for_user_with_no_vaults(self):
        """Test that /vaults/accessible returns empty list for user with no access."""
        user = self._create_user("no_access_user")
        response = self.client.get(
            "/api/vaults/accessible",
            headers={"Authorization": f"Bearer {user['token']}"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["vault_ids"], [])

    def test_accessible_returns_vault_ids_for_user_with_access(self):
        """Test that /vaults/accessible returns vault IDs for user with access."""
        user = self._create_user("has_access_user")
        vault1 = self._create_vault("Test Vault 1")
        vault2 = self._create_vault("Test Vault 2")

        self._add_vault_member(user["id"], vault1, "read")
        self._add_vault_member(user["id"], vault2, "write")

        response = self.client.get(
            "/api/vaults/accessible",
            headers={"Authorization": f"Bearer {user['token']}"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()["vault_ids"]), {vault1, vault2})

    def test_accessible_returns_empty_for_admin(self):
        """Test that /vaults/accessible returns empty list for admin (means all vaults)."""
        user = self._create_user("admin_user", role="admin")
        self._create_vault("Admin Vault")

        response = self.client.get(
            "/api/vaults/accessible",
            headers={"Authorization": f"Bearer {user['token']}"},
        )
        self.assertEqual(response.status_code, 200)
        # Empty list means "all vaults" for admin
        self.assertEqual(response.json()["vault_ids"], [])


class TestChatStreamVaultValidation(unittest.TestCase):
    """Test suite for chat/stream vault_id validation."""

    @classmethod
    def setUpClass(cls):
        """Set up test database."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = str(Path(cls.temp_dir) / "test.db")
        init_db(cls.db_path)
        cls.pool = SimpleConnectionPool(cls.db_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up test database."""
        cls.pool.close_all()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """Set up test client and dependency overrides."""
        self.client = TestClient(app)

        def override_get_db():
            conn = self.pool.get_connection()
            try:
                yield conn
            finally:
                self.pool.release_connection(conn)

        # Mock RAG engine with streaming
        self.mock_rag_engine = MagicMock()

        async def mock_query(*args, **kwargs):
            # Return a minimal done chunk
            yield {"type": "done", "sources": [], "memories_used": []}

        self.mock_rag_engine.query = mock_query

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_rag_engine] = lambda: self.mock_rag_engine

    def tearDown(self):
        """Clean up dependency overrides and tables."""
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_rag_engine, None)

        conn = self.pool.get_connection()
        conn.execute("DELETE FROM vault_members")
        conn.execute("DELETE FROM vault_group_access")
        conn.execute("DELETE FROM group_members")
        conn.execute("DELETE FROM vaults")
        conn.execute("DELETE FROM users")
        conn.commit()
        self.pool.release_connection(conn)

    def _create_user(self, username, role="member"):
        """Helper to create a user."""
        conn = self.pool.get_connection()
        cursor = conn.execute(
            "INSERT INTO users (username, hashed_password, role, is_active) VALUES (?, ?, ?, 1)",
            (username, "hash123", role),
        )
        conn.commit()
        user_id = cursor.lastrowid
        self.pool.release_connection(conn)

        from app.services.auth_service import create_access_token

        token = create_access_token(user_id, username, role)
        return {"id": user_id, "username": username, "role": role, "token": token}

    def _create_vault(self, name):
        """Helper to create a vault."""
        conn = self.pool.get_connection()
        cursor = conn.execute(
            "INSERT INTO vaults (name, description) VALUES (?, '')", (name,)
        )
        conn.commit()
        vault_id = cursor.lastrowid
        self.pool.release_connection(conn)
        return vault_id

    def _add_vault_member(self, user_id, vault_id, permission="read"):
        """Helper to add user to vault_members."""
        conn = self.pool.get_connection()
        conn.execute(
            "INSERT INTO vault_members (vault_id, user_id, permission) VALUES (?, ?, ?)",
            (vault_id, user_id, permission),
        )
        conn.commit()
        self.pool.release_connection(conn)

    def test_chat_stream_with_unauthorized_vault_returns_403(self):
        """Test that chat/stream returns 403 when user doesn't have access to vault."""
        user = self._create_user("restricted_user")
        unauthorized_vault = self._create_vault("Unauthorized Vault")

        response = self.client.post(
            "/api/chat/stream",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "vault_id": unauthorized_vault,
            },
            headers={"Authorization": f"Bearer {user['token']}"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("Vault access denied", response.json()["detail"])

    def test_chat_stream_with_authorized_vault_succeeds(self):
        """Test that chat/stream succeeds when user has access to vault."""
        user = self._create_user("authorized_user")
        authorized_vault = self._create_vault("Authorized Vault")
        self._add_vault_member(user["id"], authorized_vault, "read")

        response = self.client.post(
            "/api/chat/stream",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "vault_id": authorized_vault,
            },
            headers={"Authorization": f"Bearer {user['token']}"},
        )
        # Should not be 403
        self.assertNotEqual(response.status_code, 403)

    def test_admin_can_access_any_vault(self):
        """Test that admin can access any vault without explicit vault membership."""
        admin = self._create_user("admin_user", role="admin")
        any_vault = self._create_vault("Any Vault")

        response = self.client.post(
            "/api/chat/stream",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "vault_id": any_vault,
            },
            headers={"Authorization": f"Bearer {admin['token']}"},
        )
        # Admin should NOT get 403
        self.assertNotEqual(response.status_code, 403)

    def test_superadmin_can_access_any_vault(self):
        """Test that superadmin can access any vault without explicit vault membership."""
        superadmin = self._create_user("superadmin_user", role="superadmin")
        any_vault = self._create_vault("Any Vault")

        response = self.client.post(
            "/api/chat/stream",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "vault_id": any_vault,
            },
            headers={"Authorization": f"Bearer {superadmin['token']}"},
        )
        # Superadmin should NOT get 403
        self.assertNotEqual(response.status_code, 403)

    def test_regular_user_cannot_access_other_users_vault(self):
        """Test that a regular user cannot access vault they are not a member of."""
        user1 = self._create_user("user1")
        user2 = self._create_user("user2")
        user2_vault = self._create_vault("User2's Vault")
        self._add_vault_member(user2["id"], user2_vault, "read")

        # User1 tries to access User2's vault
        response = self.client.post(
            "/api/chat/stream",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "vault_id": user2_vault,
            },
            headers={"Authorization": f"Bearer {user1['token']}"},
        )
        self.assertEqual(response.status_code, 403)

    def test_chat_stream_default_vault_id(self):
        """Test that default vault_id works for authorized users."""
        user = self._create_user("default_vault_user")
        # Create a vault first, then give the user access
        vault_id = self._create_vault("Default Vault")
        self._add_vault_member(user["id"], vault_id, "read")

        response = self.client.post(
            "/api/chat/stream",
            json={"messages": [{"role": "user", "content": "Hello"}], "vault_id": vault_id},
            headers={"Authorization": f"Bearer {user['token']}"},
        )
        # Should succeed since the user has access to the vault
        self.assertNotEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
