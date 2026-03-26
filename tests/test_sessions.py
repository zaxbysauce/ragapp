"""Tests for session management endpoints (Task 1.4)."""

import hashlib
import pytest
from fastapi.testclient import TestClient


# Test configuration
BASE_URL = "http://testserver/api/v1/auth"
COOKIE_NAME = "refresh_token"


class TestListSessions:
    """Test GET /sessions endpoint."""

    def test_list_sessions_with_auth_returns_list(self, client, test_user, db):
        """GET /sessions with auth returns list (may be empty)."""
        # Login to get a valid session
        login_resp = client.post(
            f"{BASE_URL}/login",
            json={"username": test_user["username"], "password": test_user["password"]},
        )
        assert login_resp.status_code == 200
        cookies = login_resp.cookies

        # Get sessions
        resp = client.get(f"{BASE_URL}/sessions", cookies=cookies)

        assert resp.status_code == 200
        sessions = resp.json()
        assert isinstance(sessions, list)
        # Should have at least the session we just created
        assert len(sessions) >= 1

        # Verify session structure (no token hashes)
        session = sessions[0]
        assert "id" in session
        assert "ip_address" in session
        assert "user_agent" in session
        assert "created_at" in session
        assert "last_used_at" in session
        # Verify no sensitive data
        assert "refresh_token_hash" not in session
        assert "refresh_token" not in session

    def test_list_sessions_without_auth_returns_401(self, client):
        """GET /sessions without auth returns 401."""
        resp = client.get(f"{BASE_URL}/sessions")
        assert resp.status_code == 401


class TestRevokeSession:
    """Test DELETE /sessions/{session_id} endpoint."""

    def test_delete_session_with_valid_id_returns_204(self, client, test_user, db):
        """DELETE /sessions/{id} with valid id belonging to user returns 204."""
        # Login to get a valid session
        login_resp = client.post(
            f"{BASE_URL}/login",
            json={"username": test_user["username"], "password": test_user["password"]},
        )
        assert login_resp.status_code == 200
        cookies = login_resp.cookies

        # Get current sessions
        resp = client.get(f"{BASE_URL}/sessions", cookies=cookies)
        sessions_before = resp.json()
        initial_count = len(sessions_before)

        # Delete the session (use the first session id)
        session_id = sessions_before[0]["id"]
        delete_resp = client.delete(
            f"{BASE_URL}/sessions/{session_id}", cookies=cookies
        )

        assert delete_resp.status_code == 204

        # Verify session was deleted
        resp = client.get(f"{BASE_URL}/sessions", cookies=cookies)
        sessions_after = resp.json()
        assert len(sessions_after) == initial_count - 1

    def test_delete_session_with_id_belonging_to_another_user_returns_404(
        self, client, test_user, other_user, db
    ):
        """DELETE /sessions/{id} with id belonging to another user returns 404."""
        # Login as test_user
        login_resp = client.post(
            f"{BASE_URL}/login",
            json={"username": test_user["username"], "password": test_user["password"]},
        )
        test_user_cookies = login_resp.cookies

        # Login as other_user
        login_resp = client.post(
            f"{BASE_URL}/login",
            json={
                "username": other_user["username"],
                "password": other_user["password"],
            },
        )
        other_user_cookies = login_resp.cookies

        # Get other_user's session ID
        resp = client.get(f"{BASE_URL}/sessions", cookies=other_user_cookies)
        other_user_session_id = resp.json()[0]["id"]

        # Try to delete other_user's session as test_user
        delete_resp = client.delete(
            f"{BASE_URL}/sessions/{other_user_session_id}", cookies=test_user_cookies
        )

        assert delete_resp.status_code == 404

    def test_delete_session_with_nonexistent_id_returns_404(
        self, client, test_user, db
    ):
        """DELETE /sessions/{id} with non-existent id returns 404."""
        # Login
        login_resp = client.post(
            f"{BASE_URL}/login",
            json={"username": test_user["username"], "password": test_user["password"]},
        )
        cookies = login_resp.cookies

        # Try to delete a non-existent session ID (use a large number that won't exist)
        delete_resp = client.delete(f"{BASE_URL}/sessions/999999", cookies=cookies)

        assert delete_resp.status_code == 404


class TestRevokeAllSessions:
    """Test DELETE /sessions endpoint."""

    def test_delete_all_sessions_except_current(self, client, test_user, db):
        """DELETE /sessions deletes all except current, user stays logged in."""
        # Login
        login_resp = client.post(
            f"{BASE_URL}/login",
            json={"username": test_user["username"], "password": test_user["password"]},
        )
        cookies = login_resp.cookies

        # Create additional sessions by logging in again
        for _ in range(3):
            login_resp = client.post(
                f"{BASE_URL}/login",
                json={
                    "username": test_user["username"],
                    "password": test_user["password"],
                },
            )

        # Get sessions before deletion
        resp = client.get(f"{BASE_URL}/sessions", cookies=cookies)
        sessions_before = resp.json()
        initial_count = len(sessions_before)
        assert initial_count >= 4

        # Delete all except current
        delete_resp = client.delete(f"{BASE_URL}/sessions", cookies=cookies)

        assert delete_resp.status_code == 200
        assert delete_resp.json()["message"] == "All other sessions revoked"

        # User should still be logged in
        resp = client.get(f"{BASE_URL}/me", cookies=delete_resp.cookies)
        assert resp.status_code == 200
        assert resp.json()["username"] == test_user["username"]

        # Should have only 1 session left (the current one)
        resp = client.get(f"{BASE_URL}/sessions", cookies=delete_resp.cookies)
        sessions_after = resp.json()
        assert len(sessions_after) == 1

        # The session ID should be different (rotated)
        current_session_id = sessions_after[0]["id"]
        assert current_session_id not in [s["id"] for s in sessions_before]


# Fixtures
@pytest.fixture
def client():
    """Create test client."""
    from main import app

    return TestClient(app)


@pytest.fixture
def db():
    """Get database connection."""
    from app.models.database import get_pool

    pool = get_pool()
    conn = pool.get_connection()
    yield conn
    conn.close()


@pytest.fixture
def test_user(db):
    """Create a test user for authentication."""
    from app.services.auth_service import hash_password
    import secrets

    username = f"testuser_{secrets.token_hex(4)}"
    password = "TestPassword123!"

    cursor = db.execute(
        "INSERT INTO users (username, hashed_password, full_name, role, is_active) VALUES (?, ?, ?, ?, 1)",
        (username, hash_password(password), "Test User", "member"),
    )
    db.commit()
    user_id = cursor.lastrowid

    yield {"id": user_id, "username": username, "password": password}

    # Cleanup
    db.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    db.commit()


@pytest.fixture
def other_user(db):
    """Create another test user for cross-user testing."""
    from app.services.auth_service import hash_password
    import secrets

    username = f"otheruser_{secrets.token_hex(4)}"
    password = "OtherPassword123!"

    cursor = db.execute(
        "INSERT INTO users (username, hashed_password, full_name, role, is_active) VALUES (?, ?, ?, ?, 1)",
        (username, hash_password(password), "Other User", "member"),
    )
    db.commit()
    user_id = cursor.lastrowid

    yield {"id": user_id, "username": username, "password": password}

    # Cleanup
    db.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    db.commit()
