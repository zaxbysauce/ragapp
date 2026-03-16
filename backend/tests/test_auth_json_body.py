"""
Tests for auth endpoints JSON body acceptance.

Verifies that register and login endpoints correctly accept and process
JSON request bodies as defined by their Pydantic models.
"""
import sys
import os
import tempfile
import shutil
from unittest.mock import MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# =============================================================================
# CRITICAL: Set up stubs BEFORE any app imports
# =============================================================================

# Stub missing optional dependencies
try:
    import lancedb
except ImportError:
    import types
    lancedb_stub = types.ModuleType('lancedb')
    sys.modules['lancedb'] = lancedb_stub

try:
    import pyarrow
except ImportError:
    import types
    pyarrow_stub = types.ModuleType('pyarrow')
    sys.modules['pyarrow'] = pyarrow_stub

# Stub unstructured before anything else tries to import it
import types
_unstructured = types.ModuleType('unstructured')
_unstructured.__path__ = []
_unstructured.partition = types.ModuleType('unstructured.partition')
_unstructured.partition.__path__ = []
_unstructured.partition.auto = types.ModuleType('unstructured.partition.auto')
_unstructured.partition.auto.partition = lambda *args, **kwargs: []
_unstructured.chunking = types.ModuleType('unstructured.chunking')
_unstructured.chunking.__path__ = []
_unstructured.chunking.title = types.ModuleType('unstructured.chunking.title')
_unstructured.chunking.title.chunk_by_title = lambda *args, **kwargs: []
_unstructured.documents = types.ModuleType('unstructured.documents')
_unstructured.documents.__path__ = []
_unstructured.documents.elements = types.ModuleType('unstructured.documents.elements')
_unstructured.documents.elements.Element = type('Element', (), {})
sys.modules['unstructured'] = _unstructured
sys.modules['unstructured.partition'] = _unstructured.partition
sys.modules['unstructured.partition.auto'] = _unstructured.partition.auto
sys.modules['unstructured.chunking'] = _unstructured.chunking
sys.modules['unstructured.chunking.title'] = _unstructured.chunking.title
sys.modules['unstructured.documents'] = _unstructured.documents
sys.modules['unstructured.documents.elements'] = _unstructured.documents.elements


# =============================================================================
# Test Helper Classes
# =============================================================================

class NoOpLimiter:
    """A limiter that doesn't actually rate limit (for testing)."""
    enabled = False
    
    def limit(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator


import unittest


class TestAuthJsonBody(unittest.TestCase):
    """Test suite for JSON body acceptance in auth endpoints."""

    @classmethod
    def setUpClass(cls):
        """Set up test database."""
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up test database."""
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def tearDown(self):
        """Clean up after each test."""
        import shutil
        if hasattr(self, 'test_data_dir') and self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir, ignore_errors=True)

    def setUp(self):
        """Set up test fixtures."""
        # Import inside setUp to avoid collection issues
        from app.config import settings
        from app.models.database import init_db, get_pool
        from pathlib import Path
        
        # Create temp directory for test database
        self.test_data_dir = Path(tempfile.mkdtemp(prefix="knowledgevault_test_"))
        
        # Override settings for testing - use data_dir, not sqlite_path
        settings.data_dir = self.test_data_dir
        
        # Initialize database with schema
        init_db(str(settings.sqlite_path))
        
        # Store db_path for later use
        self.db_path = str(settings.sqlite_path)
        
        # Initialize the connection pool
        get_pool(str(self.db_path))

    def _create_test_client(self):
        """Create a TestClient with proper app state."""
        import shutil
        from fastapi.testclient import TestClient
        from app.main import app
        from app.models.database import get_pool
        from pathlib import Path
        
        # Clean up previous test data dirs
        for old_dir in getattr(self, '_test_data_dirs', []):
            shutil.rmtree(old_dir, ignore_errors=True)
        self._test_data_dirs = []
        
        # Create a fresh temp directory for this test
        test_data_dir = Path(tempfile.mkdtemp(prefix="knowledgevault_test_"))
        self._test_data_dirs.append(test_data_dir)
        
        # Import settings and configure
        from app.config import settings
        settings.data_dir = test_data_dir
        
        # Initialize database
        from app.models.database import init_db
        init_db(str(settings.sqlite_path))
        
        # Get pool
        pool = get_pool(str(settings.sqlite_path))
        
        # CRITICAL: Patch the limiter module to disable rate limiting
        # This must be done AFTER the app is imported but BEFORE requests are made
        import app.api.routes.auth as auth_module
        auth_module.limiter.enabled = False
        
        # Also patch at app state level
        app.state.limiter = NoOpLimiter()
        
        # Set up app state
        app.state.db_pool = pool
        
        # Mock other required state
        mock_maintenance = MagicMock()
        mock_maintenance.get_flag.return_value = MagicMock(
            enabled=False, reason="", version=0, updated_at=None
        )
        app.state.maintenance_service = mock_maintenance
        
        return TestClient(app)

    def _clear_users(self):
        """Clear users and sessions tables."""
        from app.models.database import get_pool
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

    # ========== REGISTER ENDPOINT JSON BODY TESTS ==========

    def test_register_accepts_json_body(self):
        """Test that register endpoint accepts JSON body correctly."""
        client = self._create_test_client()
        self._clear_users()
        
        response = client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "password123",
                "full_name": "Test User"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["username"], "testuser")
        self.assertEqual(data["full_name"], "Test User")
        self.assertEqual(data["role"], "superadmin")  # First user
        self.assertTrue(data["is_active"])

    def test_register_json_body_minimal(self):
        """Test register with only required fields in JSON body."""
        client = self._create_test_client()
        self._clear_users()
        
        response = client.post(
            "/api/auth/register",
            json={
                "username": "minimaluser",
                "password": "password123"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["username"], "minimaluser")
        self.assertEqual(data["full_name"], "")  # Default empty string

    def test_register_json_body_with_special_chars_in_username(self):
        """Test register with special characters in username via JSON."""
        client = self._create_test_client()
        self._clear_users()
        
        response = client.post(
            "/api/auth/register",
            json={
                "username": "user_name-123",
                "password": "password123",
                "full_name": "Test User"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["username"], "user_name-123")

    def test_register_json_body_with_unicode_full_name(self):
        """Test register with Unicode characters in full_name via JSON."""
        client = self._create_test_client()
        self._clear_users()
        
        response = client.post(
            "/api/auth/register",
            json={
                "username": "unicodeuser",
                "password": "password123",
                "full_name": "Пользователь Тест"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["full_name"], "Пользователь Тест")

    def test_register_json_body_rejects_empty_username(self):
        """Test register rejects empty username in JSON body."""
        client = self._create_test_client()
        self._clear_users()
        
        response = client.post(
            "/api/auth/register",
            json={
                "username": "",
                "password": "password123"
            }
        )
        
        self.assertEqual(response.status_code, 400)

    def test_register_json_body_rejects_short_password(self):
        """Test register rejects short password in JSON body."""
        client = self._create_test_client()
        self._clear_users()
        
        response = client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "short"
            }
        )
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("at least 8 characters", data["detail"].lower())

    def test_register_json_body_rejects_missing_password(self):
        """Test register rejects missing password in JSON body."""
        client = self._create_test_client()
        self._clear_users()
        
        response = client.post(
            "/api/auth/register",
            json={
                "username": "testuser"
            }
        )
        
        # Pydantic validation error
        self.assertEqual(response.status_code, 422)

    def test_register_json_body_rejects_missing_username(self):
        """Test register rejects missing username in JSON body."""
        client = self._create_test_client()
        self._clear_users()
        
        response = client.post(
            "/api/auth/register",
            json={
                "password": "password123"
            }
        )
        
        # Pydantic validation error
        self.assertEqual(response.status_code, 422)

    def test_register_json_body_rejects_extra_fields(self):
        """Test register accepts extra fields (Pydantic ignores them by default)."""
        client = self._create_test_client()
        self._clear_users()
        
        response = client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "password123",
                "full_name": "Test",
                "unknown_field": "not_allowed"  # Extra field - ignored by default
            }
        )
        
        # Pydantic ignores extra fields by default (no ConfigDict(extra='forbid'))
        # So this should succeed
        self.assertEqual(response.status_code, 200)

    # ========== LOGIN ENDPOINT JSON BODY TESTS ==========

    def test_login_accepts_json_body(self):
        """Test that login endpoint accepts JSON body correctly."""
        client = self._create_test_client()
        self._clear_users()
        
        # First register a user
        client.post(
            "/api/auth/register",
            json={
                "username": "loginuser",
                "password": "password123",
                "full_name": "Login User"
            }
        )
        
        # Now login with JSON body
        response = client.post(
            "/api/auth/login",
            json={
                "username": "loginuser",
                "password": "password123"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["token_type"], "bearer")
        self.assertEqual(data["expires_in"], 15 * 60)  # 15 minutes
        self.assertEqual(data["user"]["username"], "loginuser")
        self.assertIn("role", data["user"])

    def test_login_json_body_returns_refresh_cookie(self):
        """Test login with JSON body returns refresh token cookie."""
        client = self._create_test_client()
        self._clear_users()
        
        # Register user
        client.post(
            "/api/auth/register",
            json={
                "username": "cookieuser",
                "password": "password123"
            }
        )
        
        # Login
        response = client.post(
            "/api/auth/login",
            json={
                "username": "cookieuser",
                "password": "password123"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        # Check refresh token cookie is set
        self.assertIn("refresh_token", response.cookies)

    def test_login_json_body_rejects_wrong_password(self):
        """Test login with JSON body rejects wrong password."""
        client = self._create_test_client()
        self._clear_users()
        
        # Register user
        client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "correctpassword"
            }
        )
        
        # Login with wrong password
        response = client.post(
            "/api/auth/login",
            json={
                "username": "testuser",
                "password": "wrongpassword"
            }
        )
        
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertIn("invalid", data["detail"].lower())

    def test_login_json_body_rejects_nonexistent_user(self):
        """Test login with JSON body rejects non-existent user."""
        client = self._create_test_client()
        self._clear_users()
        
        response = client.post(
            "/api/auth/login",
            json={
                "username": "nonexistent",
                "password": "password123"
            }
        )
        
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertIn("invalid", data["detail"].lower())

    def test_login_json_body_rejects_missing_username(self):
        """Test login with JSON body rejects missing username."""
        client = self._create_test_client()
        self._clear_users()
        
        response = client.post(
            "/api/auth/login",
            json={
                "password": "password123"
            }
        )
        
        # Pydantic validation error
        self.assertEqual(response.status_code, 422)

    def test_login_json_body_rejects_missing_password(self):
        """Test login with JSON body rejects missing password."""
        client = self._create_test_client()
        self._clear_users()
        
        response = client.post(
            "/api/auth/login",
            json={
                "username": "testuser"
            }
        )
        
        # Pydantic validation error
        self.assertEqual(response.status_code, 422)

    def test_login_json_body_rejects_empty_body(self):
        """Test login with JSON body rejects empty body."""
        client = self._create_test_client()
        self._clear_users()
        
        response = client.post(
            "/api/auth/login",
            json={}
        )
        
        # Pydantic validation error
        self.assertEqual(response.status_code, 422)

    # ========== EDGE CASES ==========

    def test_register_json_body_with_very_long_full_name(self):
        """Test register with very long full_name in JSON body."""
        client = self._create_test_client()
        self._clear_users()
        
        long_name = "A" * 255  # Max length per field definition
        
        response = client.post(
            "/api/auth/register",
            json={
                "username": "longnameuser",
                "password": "password123",
                "full_name": long_name
            }
        )
        
        self.assertEqual(response.status_code, 200)

    def test_register_json_body_max_username_length(self):
        """Test register with max length username in JSON body."""
        client = self._create_test_client()
        self._clear_users()
        
        max_username = "A" * 255  # Max length per field definition
        
        response = client.post(
            "/api/auth/register",
            json={
                "username": max_username,
                "password": "password123"
            }
        )
        
        self.assertEqual(response.status_code, 200)

    def test_login_json_body_with_json_content_type(self):
        """Test login explicitly sets JSON content type."""
        client = self._create_test_client()
        self._clear_users()
        
        # Register user first
        client.post(
            "/api/auth/register",
            json={
                "username": "jsonctype",
                "password": "password123"
            }
        )
        
        # Login with explicit JSON content type
        response = client.post(
            "/api/auth/login",
            json={
                "username": "jsonctype",
                "password": "password123"
            },
            headers={"Content-Type": "application/json"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("access_token", data)


if __name__ == "__main__":
    unittest.main()
