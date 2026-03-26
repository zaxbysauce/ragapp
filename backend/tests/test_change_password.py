"""
Tests for POST /auth/change-password endpoint (Task 1.3).

Tests cover:
- Change password with correct current + valid new password → 200
- Change password with wrong current password → 401
- Change password with weak new password → 400
- Session revocation after successful password change
- New tokens issued after successful change
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

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
from app.models.database import run_migrations, get_pool
from app.services.auth_service import hash_password, verify_password


class TestChangePassword(unittest.TestCase):
    """Test suite for /auth/change-password endpoint."""

    @classmethod
    def setUpClass(cls):
        """Set up test database."""
        cls.temp_dir = tempfile.mkdtemp()
        # Set data_dir to temp directory (sqlite_path is derived from data_dir)
        settings.data_dir = Path(cls.temp_dir)
        cls.db_path = str(settings.sqlite_path)
        run_migrations(cls.db_path)
        get_pool(cls.db_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up test database."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """Set up test client and clear tables."""
        self.client = TestClient(app)

        pool = get_pool(str(self.db_path))
        conn = pool.get_connection()
        try:
            conn.execute("DELETE FROM user_sessions")
            conn.execute("DELETE FROM vault_members")
            conn.execute("DELETE FROM org_members")
            conn.execute("DELETE FROM users")
            conn.commit()
        finally:
            pool.release_connection(conn)

    def _register_and_login(self, username, password, full_name="Test User"):
        """Helper: Register user and return login response."""
        self.client.post(
            "/api/auth/register",
            json={"username": username, "password": password, "full_name": full_name},
        )
        return self.client.post(
            "/api/auth/login", json={"username": username, "password": password}
        )

    def _get_session_count(self, user_id):
        """Helper: Get count of sessions for a user."""
        pool = get_pool(str(self.db_path))
        conn = pool.get_connection()
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM user_sessions WHERE user_id = ?", (user_id,)
            )
            return cursor.fetchone()[0]
        finally:
            pool.release_connection(conn)

    def _get_hashed_password(self, user_id):
        """Helper: Get hashed password for a user."""
        pool = get_pool(str(self.db_path))
        conn = pool.get_connection()
        try:
            cursor = conn.execute(
                "SELECT hashed_password FROM users WHERE id = ?", (user_id,)
            )
            return cursor.fetchone()[0]
        finally:
            pool.release_connection(conn)

    def test_change_password_success(self):
        """Test successful password change with correct current + valid new password."""
        # Register and login
        login_response = self._register_and_login("testuser", "OldPass123")
        self.assertEqual(login_response.status_code, 200)

        access_token = login_response.json()["access_token"]
        user_id = login_response.json()["user"]["id"]

        # Count sessions before change
        sessions_before = self._get_session_count(user_id)
        self.assertGreaterEqual(
            sessions_before, 1, "Should have at least 1 session before change"
        )

        # Change password
        response = self.client.post(
            "/api/auth/change-password",
            json={"current_password": "OldPass123", "new_password": "NewPass456"},
            headers={"Authorization": f"Bearer {access_token}"},
        )

        # Should succeed
        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200, got {response.status_code}: {response.json()}",
        )
        data = response.json()
        self.assertEqual(data["message"], "Password changed successfully")

        # Verify new password works
        new_hash = self._get_hashed_password(user_id)
        self.assertTrue(
            verify_password("NewPass456", new_hash), "New password should be verifiable"
        )
        self.assertFalse(
            verify_password("OldPass123", new_hash),
            "Old password should not be verifiable",
        )

    def test_change_password_wrong_current_password(self):
        """Test password change fails with wrong current password."""
        # Register and login
        login_response = self._register_and_login("testuser", "Correct123")
        self.assertEqual(login_response.status_code, 200)

        access_token = login_response.json()["access_token"]

        # Try to change with wrong current password
        response = self.client.post(
            "/api/auth/change-password",
            json={"current_password": "WrongPassword1", "new_password": "NewPass456"},
            headers={"Authorization": f"Bearer {access_token}"},
        )

        # Should fail with 401
        self.assertEqual(
            response.status_code,
            401,
            f"Expected 401, got {response.status_code}: {response.json()}",
        )
        data = response.json()
        self.assertIn(
            "incorrect",
            data["detail"].lower(),
            f"Expected 'incorrect' in error message, got: {data['detail']}",
        )

    def test_change_password_weak_password_too_short(self):
        """Test password change fails with too short new password."""
        # Register and login
        login_response = self._register_and_login("testuser", "OldPass123")
        self.assertEqual(login_response.status_code, 200)

        access_token = login_response.json()["access_token"]

        # Try to change with weak password (too short)
        response = self.client.post(
            "/api/auth/change-password",
            json={
                "current_password": "OldPass123",
                "new_password": "weak",  # Too short, no uppercase, no digit
            },
            headers={"Authorization": f"Bearer {access_token}"},
        )

        # Should fail with 400
        self.assertEqual(
            response.status_code,
            400,
            f"Expected 400, got {response.status_code}: {response.json()}",
        )
        data = response.json()
        self.assertIn(
            "at least 8 characters",
            data["detail"].lower(),
            f"Expected 'at least 8 characters', got: {data['detail']}",
        )

    def test_change_password_weak_password_no_uppercase(self):
        """Test password change fails with new password missing uppercase."""
        # Register and login
        login_response = self._register_and_login("testuser", "OldPass123")
        self.assertEqual(login_response.status_code, 200)

        access_token = login_response.json()["access_token"]

        # Try to change with password missing uppercase
        response = self.client.post(
            "/api/auth/change-password",
            json={
                "current_password": "OldPass123",
                "new_password": "newpassword123",  # No uppercase
            },
            headers={"Authorization": f"Bearer {access_token}"},
        )

        # Should fail with 400
        self.assertEqual(
            response.status_code,
            400,
            f"Expected 400, got {response.status_code}: {response.json()}",
        )
        data = response.json()
        self.assertIn(
            "uppercase",
            data["detail"].lower(),
            f"Expected 'uppercase', got: {data['detail']}",
        )

    def test_change_password_weak_password_no_digit(self):
        """Test password change fails with new password missing digit."""
        # Register and login
        login_response = self._register_and_login("testuser", "OldPass123")
        self.assertEqual(login_response.status_code, 200)

        access_token = login_response.json()["access_token"]

        # Try to change with password missing digit
        response = self.client.post(
            "/api/auth/change-password",
            json={
                "current_password": "OldPass123",
                "new_password": "NewPassword",  # No digit
            },
            headers={"Authorization": f"Bearer {access_token}"},
        )

        # Should fail with 400
        self.assertEqual(
            response.status_code,
            400,
            f"Expected 400, got {response.status_code}: {response.json()}",
        )
        data = response.json()
        self.assertIn(
            "digit", data["detail"].lower(), f"Expected 'digit', got: {data['detail']}"
        )

    def test_change_password_requires_authentication(self):
        """Test password change fails without authentication."""
        response = self.client.post(
            "/api/auth/change-password",
            json={"current_password": "OldPass123", "new_password": "NewPass456"},
        )

        # Should fail with 401 or 403
        self.assertIn(
            response.status_code,
            [401, 403],
            f"Expected 401/403, got {response.status_code}: {response.json()}",
        )

    def test_change_password_sessions_revoked(self):
        """Test that all old sessions are revoked after successful password change."""
        # Register and login
        login_response = self._register_and_login("testuser", "OldPass123")
        self.assertEqual(login_response.status_code, 200)

        access_token = login_response.json()["access_token"]
        user_id = login_response.json()["user"]["id"]

        # Count sessions before change (should be 1)
        sessions_before = self._get_session_count(user_id)
        self.assertEqual(
            sessions_before,
            1,
            f"Should have exactly 1 session before change, got {sessions_before}",
        )

        # Change password
        response = self.client.post(
            "/api/auth/change-password",
            json={"current_password": "OldPass123", "new_password": "NewPass456"},
            headers={"Authorization": f"Bearer {access_token}"},
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200, got {response.status_code}: {response.json()}",
        )

        # Should still have 1 session (new one created after old ones revoked)
        sessions_after = self._get_session_count(user_id)
        self.assertEqual(
            sessions_after,
            1,
            f"Should have exactly 1 session after change, got {sessions_after}",
        )

    def test_change_password_new_token_issued(self):
        """Test that new tokens are issued after successful password change."""
        # Register and login
        login_response = self._register_and_login("testuser", "OldPass123")
        self.assertEqual(login_response.status_code, 200)

        old_access_token = login_response.json()["access_token"]

        # Change password
        response = self.client.post(
            "/api/auth/change-password",
            json={"current_password": "OldPass123", "new_password": "NewPass456"},
            headers={"Authorization": f"Bearer {old_access_token}"},
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200, got {response.status_code}: {response.json()}",
        )

        # Should have a new refresh token cookie
        refresh_cookie = response.cookies.get("refresh_token")
        self.assertIsNotNone(refresh_cookie, "Should have new refresh_token cookie")

    def test_change_password_without_auth_returns_401(self):
        """Test password change without auth token returns 401."""
        response = self.client.post(
            "/api/auth/change-password",
            json={"current_password": "OldPass123", "new_password": "NewPass456"},
        )

        self.assertEqual(
            response.status_code,
            401,
            f"Expected 401, got {response.status_code}: {response.json()}",
        )

    def test_change_password_with_invalid_token_returns_403(self):
        """Test password change with invalid token returns 403."""
        response = self.client.post(
            "/api/auth/change-password",
            json={"current_password": "OldPass123", "new_password": "NewPass456"},
            headers={"Authorization": "Bearer invalid_token_here"},
        )

        self.assertEqual(
            response.status_code,
            403,
            f"Expected 403, got {response.status_code}: {response.json()}",
        )


if __name__ == "__main__":
    unittest.main()
