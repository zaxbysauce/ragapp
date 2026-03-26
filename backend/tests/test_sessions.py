"""
Session management endpoint tests for Task 1.4.

Tests cover:
- GET /sessions - list active sessions
- DELETE /sessions/{session_id} - revoke specific session
- DELETE /sessions - revoke all except current
"""

import sys
import os
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub missing optional dependencies before importing app modules
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


import unittest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.models.database import init_db, get_pool
from app.services.auth_service import hash_password


class TestSessionManagement(unittest.TestCase):
    """Test suite for session management endpoints."""

    @classmethod
    def setUpClass(cls):
        """Set up test database."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.temp_dir, "test_sessions.db")
        settings.sqlite_path = cls.db_path  # type: ignore
        init_db(str(cls.db_path))
        get_pool(str(cls.db_path))

    @classmethod
    def tearDownClass(cls):
        """Clean up test database."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """Set up test client and clear tables."""
        self.client = TestClient(app)
        pool = get_pool(str(self.db_path))
        conn = pool.get_connection()
        conn.execute("DELETE FROM user_sessions")
        conn.execute("DELETE FROM users")
        conn.commit()
        conn.close()
        self._user_counter = getattr(self, "_user_counter", 0) + 1

    def _create_user(self, username=None, password="TestPassword123!"):
        """Helper to create a test user."""
        if username is None:
            self._user_counter += 1
            username = f"testuser_{self._user_counter}"

        pool = get_pool(str(self.db_path))
        conn = pool.get_connection()
        cursor = conn.execute(
            "INSERT INTO users (username, hashed_password, full_name, role, is_active) VALUES (?, ?, ?, ?, 1)",
            (username, hash_password(password), "Test User", "member"),
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return {"id": user_id, "username": username, "password": password}

    def _login(self, username, password):
        """Helper to login and return cookies."""
        resp = self.client.post(
            "/api/auth/login", json={"username": username, "password": password}
        )
        self.assertEqual(resp.status_code, 200, f"Login failed: {resp.text}")
        return resp.cookies

    # ===== GET /sessions tests =====

    def test_list_sessions_with_auth_returns_list(self):
        """GET /sessions with auth returns list of sessions."""
        user = self._create_user()
        cookies = self._login(user["username"], user["password"])

        resp = self.client.get("/api/v1/auth/sessions", cookies=cookies)

        self.assertEqual(resp.status_code, 200)
        sessions = resp.json()
        self.assertIsInstance(sessions, list)
        self.assertGreaterEqual(len(sessions), 1)

        # Verify session structure
        session = sessions[0]
        self.assertIn("id", session)
        self.assertIn("ip_address", session)
        self.assertIn("user_agent", session)
        self.assertIn("created_at", session)
        self.assertIn("last_used_at", session)

        # Verify no sensitive data
        self.assertNotIn("refresh_token_hash", session)
        self.assertNotIn("refresh_token", session)

    def test_list_sessions_without_auth_returns_401(self):
        """GET /sessions without auth returns 401."""
        resp = self.client.get("/api/v1/auth/sessions")
        self.assertEqual(resp.status_code, 401)

    # ===== DELETE /sessions/{session_id} tests =====

    def test_delete_session_with_valid_id_returns_204(self):
        """DELETE /sessions/{id} with valid id belonging to user returns 204."""
        user = self._create_user()
        cookies = self._login(user["username"], user["password"])

        # Get sessions
        resp = self.client.get("/api/v1/auth/sessions", cookies=cookies)
        sessions = resp.json()
        session_id = sessions[0]["id"]

        # Delete the session
        delete_resp = self.client.delete(
            f"/api/v1/auth/sessions/{session_id}", cookies=cookies
        )
        self.assertEqual(delete_resp.status_code, 204)

        # Verify session was deleted
        resp = self.client.get("/api/v1/auth/sessions", cookies=cookies)
        sessions_after = resp.json()
        self.assertEqual(len(sessions_after), 0)

    def test_delete_session_with_id_belonging_to_another_user_returns_404(self):
        """DELETE /sessions/{id} with id belonging to another user returns 404."""
        user1 = self._create_user()
        user2 = self._create_user()

        cookies1 = self._login(user1["username"], user1["password"])
        cookies2 = self._login(user2["username"], user2["password"])

        # Get user2's session ID
        resp = self.client.get("/api/v1/auth/sessions", cookies=cookies2)
        other_session_id = resp.json()[0]["id"]

        # Try to delete user2's session as user1
        delete_resp = self.client.delete(
            f"/api/v1/auth/sessions/{other_session_id}", cookies=cookies1
        )
        self.assertEqual(delete_resp.status_code, 404)

    def test_delete_session_with_nonexistent_id_returns_404(self):
        """DELETE /sessions/{id} with non-existent id returns 404."""
        user = self._create_user()
        cookies = self._login(user["username"], user["password"])

        # Try to delete a non-existent session
        delete_resp = self.client.delete(
            "/api/v1/auth/sessions/999999", cookies=cookies
        )
        self.assertEqual(delete_resp.status_code, 404)

    # ===== DELETE /sessions tests =====

    def test_delete_all_sessions_except_current(self):
        """DELETE /sessions deletes all except current, user stays logged in."""
        user = self._create_user()
        cookies = self._login(user["username"], user["password"])

        # Create additional sessions
        for _ in range(3):
            self._login(user["username"], user["password"])

        # Get sessions before deletion
        resp = self.client.get("/api/v1/auth/sessions", cookies=cookies)
        sessions_before = resp.json()
        initial_count = len(sessions_before)
        self.assertGreaterEqual(initial_count, 4)

        # Delete all except current
        delete_resp = self.client.delete("/api/v1/auth/sessions", cookies=cookies)
        self.assertEqual(delete_resp.status_code, 200)
        self.assertEqual(delete_resp.json()["message"], "All other sessions revoked")

        # User should still be logged in (can access /me)
        me_resp = self.client.get("/api/v1/auth/me", cookies=delete_resp.cookies)
        self.assertEqual(me_resp.status_code, 200)
        self.assertEqual(me_resp.json()["username"], user["username"])

        # Should have only 1 session left
        resp = self.client.get("/api/v1/auth/sessions", cookies=delete_resp.cookies)
        sessions_after = resp.json()
        self.assertEqual(len(sessions_after), 1)

        # Session ID should be different (rotated)
        new_session_id = sessions_after[0]["id"]
        original_ids = [s["id"] for s in sessions_before]
        self.assertNotIn(new_session_id, original_ids)


if __name__ == "__main__":
    unittest.main()
