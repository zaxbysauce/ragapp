"""
Account lockout tests for auth.py login endpoint.

Tests cover:
- Login with locked account returns 423
- 5 wrong passwords lock the account
- Correct password resets the counter
- After lockout expires, login works again
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
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.models.database import init_db, get_pool


class TestAccountLockout(unittest.TestCase):
    """Test suite for account lockout functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test database."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.temp_dir, "test_lockout.db")

        # Override settings for testing
        settings.sqlite_path = cls.db_path

        # Initialize database with schema
        init_db(str(cls.db_path))

        # Initialize the connection pool
        get_pool(str(cls.db_path))

    @classmethod
    def tearDownClass(cls):
        """Clean up test database."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """Set up test client and clear users table."""
        self.client = TestClient(app)

        # Clear users and sessions tables
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

    def _register_user(self, username, password):
        """Helper to register a test user."""
        return self.client.post(
            "/api/auth/register",
            params={
                "username": username,
                "password": password,
                "full_name": f"Test {username}",
            },
        )

    def _get_user_lockout_info(self, username):
        """Helper to get lockout info for a user."""
        pool = get_pool(str(self.db_path))
        conn = pool.get_connection()
        try:
            cursor = conn.execute(
                "SELECT failed_attempts, locked_until FROM users WHERE username = ?",
                (username,),
            )
            row = cursor.fetchone()
            return row if row else (None, None)
        finally:
            pool.release_connection(conn)

    def test_login_with_locked_account_returns_423(self):
        """Test that login with locked account returns 423 status."""
        # Register user
        self._register_user("lockeduser", "password123")

        # Manually lock the account
        pool = get_pool(str(self.db_path))
        conn = pool.get_connection()
        try:
            lockout_time = datetime.now(timezone.utc) + timedelta(minutes=15)
            conn.execute(
                "UPDATE users SET failed_attempts = 5, locked_until = ? WHERE username = ?",
                (lockout_time.isoformat(), "lockeduser"),
            )
            conn.commit()
        finally:
            pool.release_connection(conn)

        # Try to login
        response = self.client.post(
            "/api/auth/login",
            params={"username": "lockeduser", "password": "password123"},
        )

        self.assertEqual(response.status_code, 423)
        data = response.json()
        self.assertIn("locked", data["detail"].lower())
        # Verify Retry-After header is present
        self.assertIn("Retry-After", response.headers)
        # Verify seconds remaining is approximately 900 (15 minutes)
        retry_after = int(response.headers["Retry-After"])
        self.assertGreaterEqual(retry_after, 899)  # Allow 1 second tolerance
        self.assertLessEqual(retry_after, 901)

    def test_5_wrong_passwords_locks_account(self):
        """Test that 5 wrong passwords lock the account."""
        # Register user
        self._register_user("locktest", "correctpassword")

        # Attempt 5 wrong passwords
        for i in range(4):
            response = self.client.post(
                "/api/auth/login",
                params={"username": "locktest", "password": f"wrongpassword{i}"},
            )
            self.assertEqual(response.status_code, 401)

        # 5th wrong password should lock the account
        response = self.client.post(
            "/api/auth/login",
            params={"username": "locktest", "password": "wrongpassword5"},
        )
        self.assertEqual(response.status_code, 423)
        data = response.json()
        self.assertIn("locked", data["detail"].lower())

        # Verify lockout info in database
        failed_attempts, locked_until = self._get_user_lockout_info("locktest")
        self.assertEqual(failed_attempts, 5)
        self.assertIsNotNone(locked_until)

        # Verify locked_until is approximately 15 minutes in the future
        locked_dt = datetime.fromisoformat(locked_until)
        now = datetime.now(timezone.utc)
        diff = locked_dt - now
        self.assertGreaterEqual(diff.total_seconds(), 899)  # ~15 minutes
        self.assertLessEqual(diff.total_seconds(), 901)

    def test_correct_password_resets_counter(self):
        """Test that correct password resets failed_attempts and locked_until."""
        # Register user
        self._register_user("resettest", "correctpassword")

        # First, do 3 wrong passwords to increment counter
        for i in range(3):
            self.client.post(
                "/api/auth/login",
                params={"username": "resettest", "password": f"wrong{i}"},
            )

        # Check failed_attempts is now 3
        failed_attempts_before, _ = self._get_user_lockout_info("resettest")
        self.assertEqual(failed_attempts_before, 3)

        # Now login with correct password
        response = self.client.post(
            "/api/auth/login",
            params={"username": "resettest", "password": "correctpassword"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("access_token", data)

        # Verify counter is reset
        failed_attempts_after, locked_until_after = self._get_user_lockout_info(
            "resettest"
        )
        self.assertEqual(failed_attempts_after, 0)
        self.assertIsNone(locked_until_after)

    def test_login_after_lockout_expires(self):
        """Test that login works after 15 minute lockout expires."""
        # Register user
        self._register_user("expirytest", "correctpassword")

        # Manually set lockout to expire in the past
        pool = get_pool(str(self.db_path))
        conn = pool.get_connection()
        try:
            past_time = datetime.now(timezone.utc) - timedelta(minutes=1)
            conn.execute(
                "UPDATE users SET failed_attempts = 5, locked_until = ? WHERE username = ?",
                (past_time.isoformat(), "expirytest"),
            )
            conn.commit()
        finally:
            pool.release_connection(conn)

        # Login should now work
        response = self.client.post(
            "/api/auth/login",
            params={"username": "expirytest", "password": "correctpassword"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("access_token", data)

        # Verify counter is reset after successful login
        failed_attempts, locked_until = self._get_user_lockout_info("expirytest")
        self.assertEqual(failed_attempts, 0)
        self.assertIsNone(locked_until)

    def test_failed_attempts_increment_on_wrong_password(self):
        """Test that failed_attempts increments with each wrong password."""
        # Register user
        self._register_user("incrementtest", "correctpassword")

        # Check initial failed_attempts is 0
        failed_attempts, _ = self._get_user_lockout_info("incrementtest")
        self.assertEqual(failed_attempts, 0)

        # 3 wrong passwords
        for i in range(3):
            self.client.post(
                "/api/auth/login",
                params={"username": "incrementtest", "password": f"wrong{i}"},
            )
            failed_attempts, _ = self._get_user_lockout_info("incrementtest")
            self.assertEqual(failed_attempts, i + 1)

    def test_locked_account_with_correct_password_still_fails(self):
        """Test that locked account rejects correct password (no auth bypass)."""
        # Register user
        self._register_user("authtest", "correctpassword")

        # Manually lock the account
        pool = get_pool(str(self.db_path))
        conn = pool.get_connection()
        try:
            lockout_time = datetime.now(timezone.utc) + timedelta(minutes=15)
            conn.execute(
                "UPDATE users SET failed_attempts = 5, locked_until = ? WHERE username = ?",
                (lockout_time.isoformat(), "authtest"),
            )
            conn.commit()
        finally:
            pool.release_connection(conn)

        # Try correct password - should still be locked
        response = self.client.post(
            "/api/auth/login",
            params={"username": "authtest", "password": "correctpassword"},
        )
        self.assertEqual(response.status_code, 423)
        self.assertIn("locked", response.json()["detail"].lower())

    def test_lockout_check_before_password_verify(self):
        """Test that lockout is checked before password verification."""
        # Register user
        self._register_user("ordertest", "correctpassword")

        # Manually lock the account
        pool = get_pool(str(self.db_path))
        conn = pool.get_connection()
        try:
            lockout_time = datetime.now(timezone.utc) + timedelta(minutes=15)
            conn.execute(
                "UPDATE users SET failed_attempts = 5, locked_until = ? WHERE username = ?",
                (lockout_time.isoformat(), "ordertest"),
            )
            conn.commit()
        finally:
            pool.release_connection(conn)

        # Wrong password should give 423 (not 401)
        response = self.client.post(
            "/api/auth/login",
            params={"username": "ordertest", "password": "wrongpassword"},
        )
        self.assertEqual(response.status_code, 423)
        self.assertIn("locked", response.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
