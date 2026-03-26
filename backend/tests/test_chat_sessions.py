"""
Chat session management tests for Phase 1.10.

Tests cover:
- GET /api/v1/chat/sessions returns only current user's sessions
- POST /api/v1/chat/sessions creates a session with user_id set to current user
- GET /api/v1/chat/sessions/{id} returns 403 for sessions owned by other users
- DELETE /api/v1/chat/sessions/{id} returns 403 for sessions owned by other users
- PATCH /api/v1/chat/sessions/{id} updates title correctly
- POST /api/v1/chat/sessions/{id}/messages appends a message
- GET /api/v1/chat/sessions/{id}/messages returns messages in chronological order
"""

# Fix for Windows asyncio crash: use SelectorEventLoop instead of Proactor
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
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

from fastapi.testclient import TestClient
from app.main import app
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
    TEST_DB_PATH = os.path.join(TEST_DATA_DIR, "test_chat_sessions.db")
    init_db(TEST_DB_PATH)
    return TEST_DB_PATH


setup_test_db()


class TestChatSessionsEndpoints(unittest.TestCase):
    """Comprehensive test suite for chat session management endpoints."""

    @classmethod
    def setUpClass(cls):
        """Initialize test database and pool."""
        settings.sqlite_path = TEST_DB_PATH
        get_pool(TEST_DB_PATH)

    def setUp(self):
        """Set up test client, database, and create test users and vaults."""
        self.client = TestClient(app)
        self.test_pool = get_pool(TEST_DB_PATH)
        self._override_current_user = None

        from app.api.deps import get_db

        def override_get_db():
            conn = self.test_pool.get_connection()
            try:
                yield conn
            finally:
                self.test_pool.release_connection(conn)

        app.dependency_overrides[get_db] = override_get_db
        self._get_db = get_db

        # Create test users and vaults
        self._create_test_data()

    def tearDown(self):
        """Clean up overrides and connections."""
        app.dependency_overrides.pop(self._get_db, None)
        if self._override_current_user:
            app.dependency_overrides.pop(self._override_current_user, None)
        self.test_pool.close_all()

    def _create_test_data(self):
        """Create test users, vaults, and organization/group setup."""
        conn = self.test_pool.get_connection()
        try:
            # Clean tables
            conn.execute("DELETE FROM chat_messages")
            conn.execute("DELETE FROM chat_sessions")
            conn.execute("DELETE FROM vault_members")
            conn.execute("DELETE FROM user_groups")
            conn.execute("DELETE FROM groups")
            conn.execute("DELETE FROM org_members")
            conn.execute("DELETE FROM organizations")
            conn.execute("DELETE FROM vaults")
            conn.execute("DELETE FROM user_sessions")
            conn.execute("DELETE FROM users")
            conn.commit()

            password_hash = hash_password("testpassword123")
            users_data = [
                (1, "user1", password_hash, "User One", "member", 1),
                (2, "user2", password_hash, "User Two", "member", 1),
                (3, "admin", password_hash, "Admin User", "admin", 1),
                (4, "superadmin", password_hash, "Super Admin", "superadmin", 1),
            ]
            conn.executemany(
                "INSERT INTO users (id, username, hashed_password, full_name, role, is_active) VALUES (?, ?, ?, ?, ?, ?)",
                users_data,
            )

            # Create organization and groups
            conn.execute(
                "INSERT INTO organizations (id, name, description) VALUES (1, 'Test Org', 'Test organization')"
            )
            conn.execute(
                "INSERT INTO groups (id, org_id, name, description) VALUES (1, 1, 'test-group', 'Test group')"
            )
            conn.executemany(
                "INSERT INTO org_members (org_id, user_id) VALUES (?, ?)",
                [(1, 1), (1, 2), (1, 3), (1, 4)],
            )

            # Create vaults
            conn.executemany(
                "INSERT INTO vaults (id, name, description) VALUES (?, ?, ?)",
                [
                    (1, "User1 Vault", "Vault for user1"),
                    (2, "User2 Vault", "Vault for user2"),
                    (3, "Shared Vault", "Shared vault"),
                ],
            )

            # Give users access to vaults via vault_members
            vault_members = [
                (1, 1),  # user1 has admin access to vault1
                (2, 2),  # user2 has admin access to vault2
                (1, 3),  # user1 has admin access to vault3
                (2, 3),  # user2 has admin access to vault3
            ]
            conn.executemany(
                "INSERT INTO vault_members (vault_id, user_id, permission) VALUES (?, ?, ?)",
                [(v, u, "admin") for v, u in vault_members],
            )

            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

    def _override_current_user(self, user_dict: dict):
        """Override get_current_active_user to return a specific user."""
        from app.api.deps import get_current_active_user

        async def mock_get_current_user():
            return user_dict

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        self._override_current_user = get_current_active_user

    def test_list_sessions_returns_only_current_users_sessions(self):
        """Test GET /api/v1/chat/sessions returns only current user's sessions."""
        # Create sessions for both users via direct DB insert
        conn = self.test_pool.get_connection()
        try:
            conn.executemany(
                "INSERT INTO chat_sessions (id, title, vault_id, user_id) VALUES (?, ?, ?, ?)",
                [
                    (1, "User1 Session 1", 1, 1),
                    (2, "User1 Session 2", 1, 1),
                    (3, "User2 Session", 2, 2),
                ],
            )
            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

        # Authenticate as user1
        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        response = self.client.get("/api/v1/chat/sessions")
        self.assertEqual(response.status_code, 200)
        sessions = response.json()
        self.assertEqual(len(sessions), 2)
        titles = {s["title"] for s in sessions}
        self.assertEqual(titles, {"User1 Session 1", "User1 Session 2"})

    def test_create_session_assigns_current_user_id(self):
        """Test POST /api/v1/chat/sessions creates session with current user's ID."""
        self._override_current_user(
            {"id": 2, "username": "user2", "role": "member", "is_active": True}
        )

        response = self.client.post(
            "/api/v1/chat/sessions", json={"title": "New Session", "vault_id": 2}
        )
        self.assertEqual(response.status_code, 201)
        session = response.json()
        self.assertEqual(session["title"], "New Session")
        self.assertEqual(session["vault_id"], 2)
        self.assertEqual(session["user_id"], 2)  # Should be current user

        # Verify in database
        conn = sqlite3.connect(TEST_DB_PATH)
        try:
            cursor = conn.execute(
                "SELECT user_id, vault_id, title FROM chat_sessions WHERE id = ?",
                (session["id"],),
            )
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], 2)  # user_id
            self.assertEqual(row[1], 2)  # vault_id
            self.assertEqual(row[2], "New Session")
        finally:
            conn.close()

    def test_create_session_requires_vault_access(self):
        """Test POST /api/v1/chat/sessions rejects vaults user doesn't have access to."""
        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        # user1 does NOT have access to vault2 (only user2 does)
        response = self.client.post(
            "/api/v1/chat/sessions",
            json={"title": "Unauthorized Session", "vault_id": 2},
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("Access denied", response.json()["detail"])

    def test_get_session_owned_by_other_user_returns_403(self):
        """Test GET /api/v1/chat/sessions/{id} returns 403 for other user's session."""
        # Create two sessions via direct DB insert
        conn = self.test_pool.get_connection()
        try:
            conn.executemany(
                "INSERT INTO chat_sessions (id, title, vault_id, user_id) VALUES (?, ?, ?, ?)",
                [
                    (1, "User1 Session", 1, 1),
                    (2, "User2 Session", 2, 2),
                ],
            )
            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

        # Authenticate as user1, try to get user2's session
        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        response = self.client.get("/api/v1/chat/sessions/2")
        self.assertEqual(response.status_code, 403)
        self.assertIn("Access denied", response.json()["detail"])

    def test_delete_session_owned_by_other_user_returns_403(self):
        """Test DELETE /api/v1/chat/sessions/{id} returns 403 for other user's session."""
        conn = self.test_pool.get_connection()
        try:
            conn.executemany(
                "INSERT INTO chat_sessions (id, title, vault_id, user_id) VALUES (?, ?, ?, ?)",
                [
                    (1, "User1 Session", 1, 1),
                    (2, "User2 Session", 2, 2),
                ],
            )
            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        response = self.client.delete("/api/v1/chat/sessions/2")
        self.assertEqual(response.status_code, 403)
        self.assertIn("Access denied", response.json()["detail"])

    def test_patch_session_updates_title(self):
        """Test PATCH /api/v1/chat/sessions/{id} updates title correctly."""
        conn = self.test_pool.get_connection()
        try:
            conn.execute(
                "INSERT INTO chat_sessions (id, title, vault_id, user_id) VALUES (?, ?, ?, ?)",
                (1, "Original Title", 1, 1),
            )
            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        response = self.client.patch(
            "/api/v1/chat/sessions/1", json={"title": "Updated Title"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["title"], "Updated Title")
        self.assertEqual(data["id"], 1)

        # Verify in database
        conn = sqlite3.connect(TEST_DB_PATH)
        try:
            cursor = conn.execute("SELECT title FROM chat_sessions WHERE id = 1")
            title = cursor.fetchone()[0]
            self.assertEqual(title, "Updated Title")
        finally:
            conn.close()

    def test_append_message_to_session(self):
        """Test POST /api/v1/chat/sessions/{session_id}/messages appends a message."""
        # Create a session
        conn = self.test_pool.get_connection()
        try:
            conn.execute(
                "INSERT INTO chat_sessions (id, title, vault_id, user_id) VALUES (?, ?, ?, ?)",
                (1, "Test Session", 1, 1),
            )
            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        message_data = {
            "role": "user",
            "content": "Hello, this is a test message!",
            "sources": [{"filename": "doc1.txt", "score": 0.95}],
        }

        response = self.client.post(
            "/api/v1/chat/sessions/1/messages", json=message_data
        )
        self.assertEqual(response.status_code, 201)
        message = response.json()
        self.assertEqual(message["role"], "user")
        self.assertEqual(message["content"], "Hello, this is a test message!")
        self.assertEqual(message["sources"], [{"filename": "doc1.txt", "score": 0.95}])
        self.assertIn("id", message)
        self.assertIn("created_at", message)

    def test_append_message_updates_session_timestamp(self):
        """Test that appending a message updates the session's updated_at."""
        conn = self.test_pool.get_connection()
        try:
            conn.execute(
                "INSERT INTO chat_sessions (id, title, vault_id, user_id, updated_at) VALUES (?, ?, ?, ?, ?)",
                (1, "Test Session", 1, 1, "2020-01-01 00:00:00"),
            )
            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        self.client.post(
            "/api/v1/chat/sessions/1/messages",
            json={"role": "assistant", "content": "Response"},
        )

        # Check that updated_at changed
        conn = sqlite3.connect(TEST_DB_PATH)
        try:
            cursor = conn.execute("SELECT updated_at FROM chat_sessions WHERE id = 1")
            updated_at = cursor.fetchone()[0]
            self.assertIsNotNone(updated_at)
            # Should be more recent than 2020
            self.assertNotEqual(updated_at, "2020-01-01 00:00:00")
        finally:
            conn.close()

    def test_get_messages_returns_in_order(self):
        """Test GET /api/v1/chat/sessions/{session_id}/messages returns messages in chronological order."""
        conn = self.test_pool.get_connection()
        try:
            conn.execute(
                "INSERT INTO chat_sessions (id, title, vault_id, user_id) VALUES (?, ?, ?, ?)",
                (1, "Test Session", 1, 1),
            )
            # Insert messages with different created_at times
            conn.executemany(
                "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                [
                    (1, "user", "First message", "2025-01-01 10:00:00"),
                    (1, "assistant", "Second message", "2025-01-01 10:01:00"),
                    (1, "user", "Third message", "2025-01-01 10:02:00"),
                ],
            )
            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        response = self.client.get("/api/v1/chat/sessions/1/messages")
        self.assertEqual(response.status_code, 200)
        messages = response.json()
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["content"], "First message")
        self.assertEqual(messages[1]["content"], "Second message")
        self.assertEqual(messages[2]["content"], "Third message")
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[2]["role"], "user")

    def test_get_messages_for_other_users_session_returns_403(self):
        """Test GET /api/v1/chat/sessions/{session_id}/messages returns 403 for other user's session."""
        conn = self.test_pool.get_connection()
        try:
            conn.executemany(
                "INSERT INTO chat_sessions (id, title, vault_id, user_id) VALUES (?, ?, ?, ?)",
                [
                    (1, "User1 Session", 1, 1),
                    (2, "User2 Session", 2, 2),
                ],
            )
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
                (2, "user", "Test message"),
            )
            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        response = self.client.get("/api/v1/chat/sessions/2/messages")
        self.assertEqual(response.status_code, 403)

    def test_delete_session_removes_messages_via_cascade(self):
        """Test DELETE /api/v1/chat/sessions/{id} cascades to delete messages."""
        conn = self.test_pool.get_connection()
        try:
            conn.execute(
                "INSERT INTO chat_sessions (id, title, vault_id, user_id) VALUES (?, ?, ?, ?)",
                (1, "Test Session", 1, 1),
            )
            conn.executemany(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
                [
                    (1, "user", "Message 1"),
                    (1, "assistant", "Message 2"),
                    (1, "user", "Message 3"),
                ],
            )
            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        response = self.client.delete("/api/v1/chat/sessions/1")
        self.assertEqual(response.status_code, 204)

        # Verify session and messages are deleted
        conn = sqlite3.connect(TEST_DB_PATH)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM chat_sessions WHERE id = 1")
            session_count = cursor.fetchone()[0]
            self.assertEqual(session_count, 0)

            cursor = conn.execute(
                "SELECT COUNT(*) FROM chat_messages WHERE session_id = 1"
            )
            message_count = cursor.fetchone()[0]
            self.assertEqual(message_count, 0)
        finally:
            conn.close()

    def test_append_message_validates_role(self):
        """Test POST /api/v1/chat/sessions/{session_id}/messages rejects invalid role."""
        conn = self.test_pool.get_connection()
        try:
            conn.execute(
                "INSERT INTO chat_sessions (id, title, vault_id, user_id) VALUES (?, ?, ?, ?)",
                (1, "Test Session", 1, 1),
            )
            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        response = self.client.post(
            "/api/v1/chat/sessions/1/messages",
            json={"role": "invalid_role", "content": "Test"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid role", response.json()["detail"])

    def test_session_not_found_returns_404(self):
        """Test accessing non-existent session returns 404."""
        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        response = self.client.get("/api/v1/chat/sessions/999")
        self.assertEqual(response.status_code, 404)

        response = self.client.patch(
            "/api/v1/chat/sessions/999", json={"title": "Test"}
        )
        self.assertEqual(response.status_code, 404)

        response = self.client.delete("/api/v1/chat/sessions/999")
        self.assertEqual(response.status_code, 404)

        response = self.client.post(
            "/api/v1/chat/sessions/999/messages",
            json={"role": "user", "content": "Test"},
        )
        self.assertEqual(response.status_code, 404)

        response = self.client.get("/api/v1/chat/sessions/999/messages")
        self.assertEqual(response.status_code, 404)

    def test_patch_session_other_user_returns_403(self):
        """Test PATCH /api/v1/chat/sessions/{id} returns 403 for other user's session."""
        conn = self.test_pool.get_connection()
        try:
            conn.executemany(
                "INSERT INTO chat_sessions (id, title, vault_id, user_id) VALUES (?, ?, ?, ?)",
                [
                    (1, "User1 Session", 1, 1),
                    (2, "User2 Session", 2, 2),
                ],
            )
            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        response = self.client.patch(
            "/api/v1/chat/sessions/2", json={"title": "Hacked"}
        )
        self.assertEqual(response.status_code, 403)


class TestChatSessionsAdversarial(unittest.TestCase):
    """Adversarial/security-focused tests for chat session endpoints."""

    @classmethod
    def setUpClass(cls):
        settings.sqlite_path = TEST_DB_PATH
        get_pool(TEST_DB_PATH)

    def setUp(self):
        self.client = TestClient(app)
        self.test_pool = get_pool(TEST_DB_PATH)
        self._override_current_user = None

        from app.api.deps import get_db

        def override_get_db():
            conn = self.test_pool.get_connection()
            try:
                yield conn
            finally:
                self.test_pool.release_connection(conn)

        app.dependency_overrides[get_db] = override_get_db
        self._get_db = get_db

        self._create_test_data()

    def tearDown(self):
        app.dependency_overrides.pop(self._get_db, None)
        if self._override_current_user:
            app.dependency_overrides.pop(self._override_current_user, None)
        self.test_pool.close_all()

    def _create_test_data(self):
        """Create minimal test data."""
        conn = self.test_pool.get_connection()
        try:
            conn.execute("DELETE FROM chat_messages")
            conn.execute("DELETE FROM chat_sessions")
            conn.execute("DELETE FROM vault_members")
            conn.execute("DELETE FROM users")
            conn.execute("DELETE FROM vaults")
            conn.commit()

            password_hash = hash_password("testpassword123")
            conn.execute(
                "INSERT INTO users (id, username, hashed_password, full_name, role, is_active) VALUES (?, ?, ?, ?, ?, ?)",
                (1, "user1", password_hash, "User One", "member", 1),
            )
            conn.execute(
                "INSERT INTO vaults (id, name, description) VALUES (?, ?, ?)",
                (1, "Test Vault", "Test vault"),
            )
            conn.execute(
                "INSERT INTO vault_members (vault_id, user_id, permission) VALUES (?, ?, ?)",
                (1, 1, "admin"),
            )
            conn.execute(
                "INSERT INTO chat_sessions (id, title, vault_id, user_id) VALUES (?, ?, ?, ?)",
                (1, "Session 1", 1, 1),
            )
            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

    def _override_current_user(self, user_dict: dict):
        """Override get_current_active_user."""
        from app.api.deps import get_current_active_user

        async def mock_get_current_user():
            return user_dict

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        self._override_current_user = get_current_active_user

    def test_create_session_with_oversized_title(self):
        """Test POST /api/v1/chat/sessions with extremely long title."""
        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        long_title = "a" * 10000
        response = self.client.post(
            "/api/v1/chat/sessions", json={"title": long_title, "vault_id": 1}
        )
        # Should either truncate or reject
        self.assertIn(response.status_code, [200, 400, 422])
        if response.status_code == 200:
            data = response.json()
            self.assertLess(len(data["title"]), 10000)  # Either truncated or accepted

    def test_create_message_with_oversized_content(self):
        """Test POST message with extremely long content."""
        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        long_content = "a" * 100000
        response = self.client.post(
            "/api/v1/chat/sessions/1/messages",
            json={"role": "user", "content": long_content},
        )
        # Should handle gracefully
        self.assertIn(response.status_code, [201, 400, 413, 422])

    def test_create_message_with_null_bytes(self):
        """Test POST message with null bytes in content."""
        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        response = self.client.post(
            "/api/v1/chat/sessions/1/messages",
            json={"role": "user", "content": "test\x00with\x00nulls"},
        )
        # Should handle safely
        self.assertIn(response.status_code, [201, 400])

    def test_create_message_with_unicode(self):
        """Test POST message with unicode, emoji, RTL characters."""
        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        unicode_content = "Hello 世界 🌍 مرحبا"
        response = self.client.post(
            "/api/v1/chat/sessions/1/messages",
            json={"role": "user", "content": unicode_content},
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["content"], unicode_content)

    def test_create_message_with_empty_content(self):
        """Test POST message with empty content."""
        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        response = self.client.post(
            "/api/v1/chat/sessions/1/messages", json={"role": "user", "content": ""}
        )
        # Empty content might be allowed or rejected based on DB schema (NOT NULL constraint)
        # The chat_messages.content column is NOT NULL, so empty string "" is allowed (it's not NULL)
        # Let's see what happens
        if response.status_code == 400:
            self.assertIn("content", response.json()["detail"].lower())
        else:
            self.assertEqual(response.status_code, 201)

    def test_create_message_with_missing_role(self):
        """Test POST message without role field fails validation."""
        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        response = self.client.post(
            "/api/v1/chat/sessions/1/messages", json={"content": "Test message"}
        )
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity

    def test_patch_session_with_missing_title(self):
        """Test PATCH session without title fails validation."""
        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )

        response = self.client.patch("/api/v1/chat/sessions/1", json={})
        self.assertEqual(response.status_code, 422)

    def test_session_isolation_among_multiple_users(self):
        """Create sessions for multiple users and verify isolation."""
        conn = self.test_pool.get_connection()
        try:
            conn.executemany(
                "INSERT INTO chat_sessions (id, title, vault_id, user_id) VALUES (?, ?, ?, ?)",
                [
                    (1, "User1 Session A", 1, 1),
                    (2, "User1 Session B", 1, 1),
                    (3, "User2 Session A", 2, 2),
                    (4, "User2 Session B", 2, 2),
                    (5, "User2 Session C", 2, 2),
                ],
            )
            conn.commit()
        finally:
            self.test_pool.release_connection(conn)

        # User1 should see only 2 sessions
        self._override_current_user(
            {"id": 1, "username": "user1", "role": "member", "is_active": True}
        )
        response = self.client.get("/api/v1/chat/sessions")
        self.assertEqual(response.status_code, 200)
        sessions = response.json()
        self.assertEqual(len(sessions), 2)
        titles = {s["title"] for s in sessions}
        self.assertEqual(titles, {"User1 Session A", "User1 Session B"})

        # User2 should see only 3 sessions
        self._override_current_user(
            {"id": 2, "username": "user2", "role": "member", "is_active": True}
        )
        response = self.client.get("/api/v1/chat/sessions")
        self.assertEqual(response.status_code, 200)
        sessions = response.json()
        self.assertEqual(len(sessions), 3)
        titles = {s["title"] for s in sessions}
        self.assertEqual(
            titles, {"User2 Session A", "User2 Session B", "User2 Session C"}
        )


if __name__ == "__main__":
    unittest.main()
