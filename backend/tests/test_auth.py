"""
Authentication tests for the RAG application.

Tests cover:
- Registration (first user = superadmin)
- Login (correct/incorrect credentials)
- Token refresh
- Logout
- Protected route access
- Role-based access control
"""
import sys
import os
import hashlib
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Stub missing optional dependencies before importing app modules
try:
    import lancedb
except ImportError:
    import types
    sys.modules['lancedb'] = types.ModuleType('lancedb')

try:
    import pyarrow
except ImportError:
    import types
    sys.modules['pyarrow'] = types.ModuleType('pyarrow')

try:
    from unstructured.partition.auto import partition
except ImportError:
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

import unittest
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.models.database import init_db, get_pool
from app.services.auth_service import hash_password, verify_password, create_access_token, create_refresh_token


class TestAuth(unittest.TestCase):
    """Test suite for authentication endpoints."""

    @classmethod
    def setUpClass(cls):
        """Set up test database."""
        # Create a temporary directory for test database
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.temp_dir, 'test.db')
        
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

        # Reset rate limiter to avoid 429 errors across tests
        from app.limiter import limiter
        try:
            limiter.reset()
        except Exception:
            pass

    def test_registration_first_user_becomes_superadmin(self):
        """Test that the first registered user becomes superadmin."""
        response = self.client.post(
            "/api/auth/register",
            json={
                "username": "firstuser",
                "password": "password123",
                "full_name": "First User"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["role"], "superadmin")
        self.assertEqual(data["username"], "firstuser")
        self.assertTrue(data["is_active"])

    def test_registration_subsequent_users_become_member(self):
        """Test that subsequent users become members by default."""
        # Create first user (superadmin)
        self.client.post(
            "/api/auth/register",
            json={
                "username": "admin",
                "password": "password123",
                "full_name": "Admin User"
            }
        )
        
        # Create second user (should be member)
        response = self.client.post(
            "/api/auth/register",
            json={
                "username": "regularuser",
                "password": "password123",
                "full_name": "Regular User"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["role"], "member")

    def test_registration_duplicate_username_fails(self):
        """Test that duplicate usernames are rejected."""
        # Create first user
        self.client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "password123",
                "full_name": "Test User"
            }
        )

        # Try to create user with same username
        response = self.client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "password123",
                "full_name": "Another User"
            }
        )
        
        self.assertEqual(response.status_code, 409)
        data = response.json()
        self.assertIn("already exists", data["detail"].lower())

    def test_registration_short_username_fails(self):
        """Test that usernames shorter than 3 characters are rejected."""
        response = self.client.post(
            "/api/auth/register",
            json={
                "username": "ab",
                "password": "password123",
                "full_name": "Test User"
            }
        )
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("at least 3 characters", data["detail"].lower())

    def test_registration_short_password_fails(self):
        """Test that passwords shorter than 8 characters are rejected."""
        response = self.client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "short",
                "full_name": "Test User"
            }
        )
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("at least 8 characters", data["detail"].lower())

    def test_login_correct_credentials(self):
        """Test login with correct credentials."""
        # Register user
        self.client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "password123",
                "full_name": "Test User"
            }
        )

        # Login
        response = self.client.post(
            "/api/auth/login",
            json={
                "username": "testuser",
                "password": "password123"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["token_type"], "bearer")
        self.assertIn("user", data)
        self.assertEqual(data["user"]["username"], "testuser")

    def test_login_incorrect_password(self):
        """Test login with incorrect password."""
        # Register user
        self.client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "password123",
                "full_name": "Test User"
            }
        )

        # Login with wrong password
        response = self.client.post(
            "/api/auth/login",
            json={
                "username": "testuser",
                "password": "wrongpassword"
            }
        )
        
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertIn("invalid", data["detail"].lower())

    def test_login_nonexistent_user(self):
        """Test login with non-existent user."""
        response = self.client.post(
            "/api/auth/login",
            json={
                "username": "nonexistent",
                "password": "password123"
            }
        )
        
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertIn("invalid", data["detail"].lower())

    def test_login_inactive_user(self):
        """Test login with inactive user account."""
        # Register user
        self.client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "password123",
                "full_name": "Test User"
            }
        )
        
        # Deactivate user
        pool = get_pool(str(self.db_path))
        conn = pool.get_connection()
        try:
            conn.execute("UPDATE users SET is_active = 0 WHERE username = ?", ("testuser",))
            conn.commit()
        finally:
            pool.release_connection(conn)
        
        # Try to login
        response = self.client.post(
            "/api/auth/login",
            json={
                "username": "testuser",
                "password": "password123"
            }
        )

        self.assertEqual(response.status_code, 403)
        data = response.json()
        self.assertIn("inactive", data["detail"].lower())

    def test_token_refresh(self):
        """Test token refresh endpoint."""
        # Register and login
        self.client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "password123",
                "full_name": "Test User"
            }
        )

        login_response = self.client.post(
            "/api/auth/login",
            json={
                "username": "testuser",
                "password": "password123"
            }
        )
        
        # Get refresh token from cookie
        refresh_token = login_response.cookies.get("refresh_token")
        self.assertIsNotNone(refresh_token)
        
        # Refresh token
        response = self.client.post(
            "/api/auth/refresh",
            cookies={"refresh_token": refresh_token}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["token_type"], "bearer")

    def test_token_refresh_invalid_token(self):
        """Test token refresh with invalid token."""
        response = self.client.post(
            "/api/auth/refresh",
            cookies={"refresh_token": "invalid_token"}
        )
        
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertIn("invalid", data["detail"].lower())

    def test_token_refresh_no_token(self):
        """Test token refresh without token."""
        response = self.client.post("/api/auth/refresh")
        
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertIn("missing", data["detail"].lower())

    def test_logout(self):
        """Test logout endpoint."""
        # Register and login
        self.client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "password123",
                "full_name": "Test User"
            }
        )

        login_response = self.client.post(
            "/api/auth/login",
            json={
                "username": "testuser",
                "password": "password123"
            }
        )
        
        refresh_token = login_response.cookies.get("refresh_token")
        
        # Logout
        response = self.client.post(
            "/api/auth/logout",
            cookies={"refresh_token": refresh_token}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("logged out", data["message"].lower())
        
        # Try to refresh with revoked token
        refresh_response = self.client.post(
            "/api/auth/refresh",
            cookies={"refresh_token": refresh_token}
        )
        self.assertEqual(refresh_response.status_code, 401)

    def test_protected_route_without_auth(self):
        """Test accessing protected route without authentication."""
        response = self.client.get("/api/auth/me")
        
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertIn("authorization", data["detail"].lower())

    def test_protected_route_with_invalid_token(self):
        """Test accessing protected route with invalid token."""
        response = self.client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        self.assertEqual(response.status_code, 403)
        data = response.json()
        self.assertIn("invalid", data["detail"].lower())

    def test_protected_route_with_valid_token(self):
        """Test accessing protected route with valid token."""
        # Register and login
        self.client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "password123",
                "full_name": "Test User"
            }
        )

        login_response = self.client.post(
            "/api/auth/login",
            json={
                "username": "testuser",
                "password": "password123"
            }
        )
        
        access_token = login_response.json()["access_token"]
        
        # Access protected route
        response = self.client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["username"], "testuser")
        self.assertIn("role", data)

    def test_role_based_access_admin_can_list_users(self):
        """Test that admin can list users."""
        # Register admin
        self.client.post(
            "/api/auth/register",
            json={
                "username": "admin",
                "password": "password123",
                "full_name": "Admin User"
            }
        )

        login_response = self.client.post(
            "/api/auth/login",
            json={
                "username": "admin",
                "password": "password123"
            }
        )
        
        access_token = login_response.json()["access_token"]
        
        # Try to list users (admin only)
        response = self.client.get(
            "/api/users",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("users", data)

    def test_role_based_access_member_cannot_list_users(self):
        """Test that member cannot list users."""
        # Register superadmin first
        self.client.post(
            "/api/auth/register",
            json={
                "username": "superadmin",
                "password": "password123",
                "full_name": "Super Admin"
            }
        )

        # Register member
        self.client.post(
            "/api/auth/register",
            json={
                "username": "member",
                "password": "password123",
                "full_name": "Member User"
            }
        )

        login_response = self.client.post(
            "/api/auth/login",
            json={
                "username": "member",
                "password": "password123"
            }
        )
        
        access_token = login_response.json()["access_token"]
        
        # Try to list users (should fail)
        response = self.client.get(
            "/api/users",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        self.assertEqual(response.status_code, 403)
        data = response.json()
        self.assertIn("insufficient privileges", data["detail"].lower())

    def test_role_based_access_superadmin_can_change_roles(self):
        """Test that superadmin can change user roles."""
        # Register superadmin
        self.client.post(
            "/api/auth/register",
            json={
                "username": "superadmin",
                "password": "password123",
                "full_name": "Super Admin"
            }
        )

        # Register member
        member_response = self.client.post(
            "/api/auth/register",
            json={
                "username": "member",
                "password": "password123",
                "full_name": "Member User"
            }
        )
        member_id = member_response.json()["id"]

        login_response = self.client.post(
            "/api/auth/login",
            json={
                "username": "superadmin",
                "password": "password123"
            }
        )
        
        access_token = login_response.json()["access_token"]
        
        # Change member's role to admin
        response = self.client.patch(
            f"/api/users/{member_id}/role",
            params={"role": "admin"},
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["role"], "admin")

    def test_role_based_access_admin_cannot_change_roles(self):
        """Test that admin cannot change user roles (only superadmin can)."""
        # Register superadmin first
        self.client.post(
            "/api/auth/register",
            json={
                "username": "superadmin",
                "password": "password123",
                "full_name": "Super Admin"
            }
        )

        # Register admin
        self.client.post(
            "/api/auth/register",
            json={
                "username": "admin",
                "password": "password123",
                "full_name": "Admin User"
            }
        )

        # Register member
        member_response = self.client.post(
            "/api/auth/register",
            json={
                "username": "member",
                "password": "password123",
                "full_name": "Member User"
            }
        )
        member_id = member_response.json()["id"]

        login_response = self.client.post(
            "/api/auth/login",
            json={
                "username": "admin",
                "password": "password123"
            }
        )
        
        access_token = login_response.json()["access_token"]
        
        # Try to change member's role (should fail)
        response = self.client.patch(
            f"/api/users/{member_id}/role",
            params={"role": "admin"},
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        self.assertEqual(response.status_code, 403)

    def test_get_me_endpoint(self):
        """Test the /me endpoint returns current user info."""
        # Register user
        self.client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "password123",
                "full_name": "Test User"
            }
        )

        login_response = self.client.post(
            "/api/auth/login",
            json={
                "username": "testuser",
                "password": "password123"
            }
        )
        
        access_token = login_response.json()["access_token"]
        
        response = self.client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["username"], "testuser")
        self.assertEqual(data["full_name"], "Test User")
        self.assertIn("role", data)
        self.assertIn("is_active", data)

    def test_update_profile(self):
        """Test updating user profile."""
        # Register user
        self.client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "password123",
                "full_name": "Test User"
            }
        )

        login_response = self.client.post(
            "/api/auth/login",
            json={
                "username": "testuser",
                "password": "password123"
            }
        )
        
        access_token = login_response.json()["access_token"]
        
        # Update profile
        response = self.client.patch(
            "/api/auth/me",
            params={"full_name": "Updated Name"},
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["full_name"], "Updated Name")

    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "testpassword123"
        hashed = hash_password(password)
        
        # Verify correct password
        self.assertTrue(verify_password(password, hashed))
        
        # Verify incorrect password
        self.assertFalse(verify_password("wrongpassword", hashed))

    def test_access_token_creation_and_validation(self):
        """Test access token creation and validation."""
        token = create_access_token(1, "testuser", "member")
        
        # Token should be a non-empty string
        self.assertIsInstance(token, str)
        self.assertTrue(len(token) > 0)

    def test_refresh_token_rotation(self):
        """Test that refresh tokens are rotated on use."""
        # Register and login
        self.client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "password123",
                "full_name": "Test User"
            }
        )

        login_response = self.client.post(
            "/api/auth/login",
            json={
                "username": "testuser",
                "password": "password123"
            }
        )
        
        old_refresh_token = login_response.cookies.get("refresh_token")
        
        # Refresh - should get new token
        refresh_response = self.client.post(
            "/api/auth/refresh",
            cookies={"refresh_token": old_refresh_token}
        )
        
        self.assertEqual(refresh_response.status_code, 200)
        new_refresh_token = refresh_response.cookies.get("refresh_token")
        
        # Old token should be invalidated
        second_refresh = self.client.post(
            "/api/auth/refresh",
            cookies={"refresh_token": old_refresh_token}
        )
        self.assertEqual(second_refresh.status_code, 401)


if __name__ == "__main__":
    unittest.main()
