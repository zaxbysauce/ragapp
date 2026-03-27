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
from app.api.deps import get_db, get_current_active_user


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
        # Ensure settings always point to the test DB (prevents pollution from other test modules)
        settings.sqlite_path = self.db_path  # type: ignore
        pool = get_pool(str(self.db_path))

        # Override get_db so all endpoints use the test pool
        def override_get_db():
            conn = pool.get_connection()
            try:
                yield conn
            finally:
                pool.release_connection(conn)

        app.dependency_overrides[get_db] = override_get_db
        # Defensively remove any leaked auth overrides from other test modules
        # (test_sessions tests require real token-based auth, not mocked auth)
        app.dependency_overrides.pop(get_current_active_user, None)

        self.client = TestClient(app)
        conn = pool.get_connection()
        conn.execute("DELETE FROM user_sessions")
        conn.execute("DELETE FROM users")
        conn.commit()
        pool.release_connection(conn)
        self._user_counter = getattr(self, "_user_counter", 0) + 1
        # Reset rate limiter to avoid 429 errors across tests
        from app.limiter import limiter
        try:
            limiter.reset()
        except Exception:
            pass

    def tearDown(self):
        """Clean up dependency overrides."""
        app.dependency_overrides.pop(get_db, None)

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
        pool.release_connection(conn)
        return {"id": user_id, "username": username, "password": password}

    def _login(self, username, password):
        """Helper to login and return (access_token, cookie_dict)."""
        # Reset rate limiter before each login to avoid 429 errors in test suites
        from app.limiter import limiter
        try:
            limiter.reset()
        except Exception:
            pass
        resp = self.client.post(
            "/api/auth/login", json={"username": username, "password": password}
        )
        self.assertEqual(resp.status_code, 200, f"Login failed: {resp.text}")
        # Extract raw cookie values from the response headers to bypass path restrictions
        cookie_dict = {}
        for cookie in resp.cookies.jar:
            cookie_dict[cookie.name] = cookie.value
        return resp.json()["access_token"], cookie_dict

    # ===== GET /sessions tests =====

    def test_list_sessions_with_auth_returns_list(self):
        """GET /sessions with auth returns list of sessions."""
        user = self._create_user()
        token, cookies = self._login(user["username"], user["password"])

        resp = self.client.get("/api/auth/sessions", headers={"Authorization": f"Bearer {token}"})

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
        resp = self.client.get("/api/auth/sessions")
        self.assertEqual(resp.status_code, 401)

    # ===== DELETE /sessions/{session_id} tests =====

    def test_delete_session_with_valid_id_returns_204(self):
        """DELETE /sessions/{id} with valid id belonging to user returns 204."""
        user = self._create_user()
        token, cookies = self._login(user["username"], user["password"])

        # Get sessions
        resp = self.client.get("/api/auth/sessions", headers={"Authorization": f"Bearer {token}"})
        sessions = resp.json()
        session_id = sessions[0]["id"]

        # Delete the session
        delete_resp = self.client.delete(
            f"/api/auth/sessions/{session_id}", headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(delete_resp.status_code, 204)

        # Verify session was deleted
        resp = self.client.get("/api/auth/sessions", headers={"Authorization": f"Bearer {token}"})
        sessions_after = resp.json()
        self.assertEqual(len(sessions_after), 0)

    def test_delete_session_with_id_belonging_to_another_user_returns_404(self):
        """DELETE /sessions/{id} with id belonging to another user returns 404."""
        user1 = self._create_user()
        user2 = self._create_user()

        token1, cookies1 = self._login(user1["username"], user1["password"])
        token2, cookies2 = self._login(user2["username"], user2["password"])

        # Get user2's session ID
        resp = self.client.get("/api/auth/sessions", headers={"Authorization": f"Bearer {token2}"})
        other_session_id = resp.json()[0]["id"]

        # Try to delete user2's session as user1
        delete_resp = self.client.delete(
            f"/api/auth/sessions/{other_session_id}", headers={"Authorization": f"Bearer {token1}"}
        )
        self.assertEqual(delete_resp.status_code, 404)

    def test_delete_session_with_nonexistent_id_returns_404(self):
        """DELETE /sessions/{id} with non-existent id returns 404."""
        user = self._create_user()
        token, cookies = self._login(user["username"], user["password"])

        # Try to delete a non-existent session
        delete_resp = self.client.delete(
            "/api/auth/sessions/999999", headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(delete_resp.status_code, 404)

    # ===== DELETE /sessions tests =====

    def test_delete_all_sessions_except_current(self):
        """DELETE /sessions deletes all except current, user stays logged in."""
        user = self._create_user()
        token, cookies = self._login(user["username"], user["password"])

        # Create additional sessions
        for _ in range(3):
            self._login(user["username"], user["password"])

        # Get sessions before deletion
        resp = self.client.get("/api/auth/sessions", headers={"Authorization": f"Bearer {token}"})
        sessions_before = resp.json()
        initial_count = len(sessions_before)
        self.assertGreaterEqual(initial_count, 4)

        # Delete all except current (needs both bearer token + refresh_token cookie)
        delete_resp = self.client.delete(
            "/api/auth/sessions",
            headers={"Authorization": f"Bearer {token}"},
            cookies=cookies,
        )
        self.assertEqual(delete_resp.status_code, 200)
        self.assertEqual(delete_resp.json()["message"], "All other sessions revoked")

        # User should still be logged in (can access /me) using original token
        new_cookies = delete_resp.cookies
        me_resp = self.client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
            cookies=new_cookies,
        )
        self.assertEqual(me_resp.status_code, 200)
        self.assertEqual(me_resp.json()["username"], user["username"])

        # Should have only 1 session left
        resp = self.client.get(
            "/api/auth/sessions",
            headers={"Authorization": f"Bearer {token}"},
            cookies=new_cookies,
        )
        sessions_after = resp.json()
        self.assertEqual(len(sessions_after), 1)

        # Session ID should be different (rotated)
        new_session_id = sessions_after[0]["id"]
        original_ids = [s["id"] for s in sessions_before]
        self.assertNotIn(new_session_id, original_ids)


if __name__ == "__main__":
    unittest.main()
