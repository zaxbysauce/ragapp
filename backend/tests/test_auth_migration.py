"""
Tests for auth migration functions (Phase 1 Task 1.1).

Tests the migrate_add_auth_extensions and rollback_auth_extensions functions
for correct schema changes, idempotency, and proper cleanup.
"""

import sqlite3
import tempfile
import os
from pathlib import Path

import pytest

from app.models.database import (
    init_db,
    migrate_add_auth_extensions,
    rollback_auth_extensions,
)


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """Get set of column names from a table."""
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


def get_index_names(
    conn: sqlite3.Connection, table_name: str | None = None
) -> set[str]:
    """Get set of index names, optionally filtered by table."""
    if table_name:
        # SQLite PRAGMA doesn't support parameterized queries - use f-string
        cursor = conn.execute(f"PRAGMA index_list({table_name})")
    else:
        cursor = conn.execute("PRAGMA index_list(user_sessions)")
    return {row[1] for row in cursor.fetchall()}


@pytest.fixture
def temp_db():
    """Create a temporary test database with base schema."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    # Initialize with base schema
    init_db(path)

    yield path

    # Cleanup
    try:
        os.unlink(path)
    except OSError:
        pass


class TestMigrateAddAuthExtensions:
    """Test suite for migrate_add_auth_extensions function."""

    def test_adds_must_change_password_column(self, temp_db):
        """Verify must_change_password column is added to users table."""
        migrate_add_auth_extensions(temp_db)

        conn = sqlite3.connect(temp_db)
        try:
            columns = get_table_columns(conn, "users")
            assert "must_change_password" in columns, (
                "must_change_password column not found in users table"
            )
        finally:
            conn.close()

    def test_adds_failed_attempts_column(self, temp_db):
        """Verify failed_attempts column is added to users table."""
        migrate_add_auth_extensions(temp_db)

        conn = sqlite3.connect(temp_db)
        try:
            columns = get_table_columns(conn, "users")
            assert "failed_attempts" in columns, (
                "failed_attempts column not found in users table"
            )
        finally:
            conn.close()

    def test_adds_locked_until_column(self, temp_db):
        """Verify locked_until column is added to users table."""
        migrate_add_auth_extensions(temp_db)

        conn = sqlite3.connect(temp_db)
        try:
            columns = get_table_columns(conn, "users")
            assert "locked_until" in columns, (
                "locked_until column not found in users table"
            )
        finally:
            conn.close()

    def test_adds_sessions_expires_index(self, temp_db):
        """Verify idx_user_sessions_expires index is created on user_sessions."""
        migrate_add_auth_extensions(temp_db)

        conn = sqlite3.connect(temp_db)
        try:
            indexes = get_index_names(conn, "user_sessions")
            assert "idx_user_sessions_expires" in indexes, (
                "idx_user_sessions_expires index not found on user_sessions"
            )
        finally:
            conn.close()

    def test_adds_locked_until_index(self, temp_db):
        """Verify idx_users_locked_until index is created on users."""
        migrate_add_auth_extensions(temp_db)

        conn = sqlite3.connect(temp_db)
        try:
            indexes = get_index_names(conn, "users")
            assert "idx_users_locked_until" in indexes, (
                "idx_users_locked_until index not found on users"
            )
        finally:
            conn.close()

    def test_is_idempotent(self, temp_db):
        """Verify migration can be run multiple times without errors."""
        # Run migration twice
        migrate_add_auth_extensions(temp_db)
        migrate_add_auth_extensions(temp_db)  # Should not raise

        # Verify all columns still exist exactly once
        conn = sqlite3.connect(temp_db)
        try:
            columns = get_table_columns(conn, "users")
            expected_columns = {
                "must_change_password",
                "failed_attempts",
                "locked_until",
            }
            for col in expected_columns:
                assert col in columns, f"Column {col} missing after idempotency check"
        finally:
            conn.close()

    def test_preserves_existing_data(self, temp_db):
        """Verify migration does not affect existing data in users table."""
        conn = sqlite3.connect(temp_db)
        try:
            # Insert test user before migration
            conn.execute(
                "INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)",
                ("testuser", "hash123", "member"),
            )
            conn.commit()

            user_id = conn.execute(
                "SELECT id FROM users WHERE username = ?", ("testuser",)
            ).fetchone()[0]
        finally:
            conn.close()

        # Run migration
        migrate_add_auth_extensions(temp_db)

        # Verify user data is preserved
        conn = sqlite3.connect(temp_db)
        conn.row_factory = sqlite3.Row
        try:
            user = conn.execute(
                "SELECT id, username, hashed_password, role FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            assert user is not None, "User data was lost during migration"
            assert user["username"] == "testuser"
            assert user["hashed_password"] == "hash123"
            assert user["role"] == "member"
        finally:
            conn.close()

    def test_new_columns_have_correct_defaults(self, temp_db):
        """Verify new columns have correct default values."""
        migrate_add_auth_extensions(temp_db)

        conn = sqlite3.connect(temp_db)
        conn.row_factory = sqlite3.Row
        try:
            # Insert user after migration
            conn.execute(
                "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
                ("newuser", "hash456"),
            )
            conn.commit()

            user = conn.execute(
                "SELECT must_change_password, failed_attempts, locked_until FROM users WHERE username = ?",
                ("newuser",),
            ).fetchone()

            assert user["must_change_password"] == 0, (
                "must_change_password should default to 0"
            )
            assert user["failed_attempts"] == 0, "failed_attempts should default to 0"
            assert user["locked_until"] is None, "locked_until should default to NULL"
        finally:
            conn.close()


class TestRollbackAuthExtensions:
    """Test suite for rollback_auth_extensions function."""

    def test_removes_must_change_password_column(self, temp_db):
        """Verify must_change_password column is removed by rollback."""
        migrate_add_auth_extensions(temp_db)
        rollback_auth_extensions(temp_db)

        conn = sqlite3.connect(temp_db)
        try:
            columns = get_table_columns(conn, "users")
            assert "must_change_password" not in columns, (
                "must_change_password column still exists after rollback"
            )
        finally:
            conn.close()

    def test_removes_failed_attempts_column(self, temp_db):
        """Verify failed_attempts column is removed by rollback."""
        migrate_add_auth_extensions(temp_db)
        rollback_auth_extensions(temp_db)

        conn = sqlite3.connect(temp_db)
        try:
            columns = get_table_columns(conn, "users")
            assert "failed_attempts" not in columns, (
                "failed_attempts column still exists after rollback"
            )
        finally:
            conn.close()

    def test_removes_locked_until_column(self, temp_db):
        """Verify locked_until column is removed by rollback."""
        migrate_add_auth_extensions(temp_db)
        rollback_auth_extensions(temp_db)

        conn = sqlite3.connect(temp_db)
        try:
            columns = get_table_columns(conn, "users")
            assert "locked_until" not in columns, (
                "locked_until column still exists after rollback"
            )
        finally:
            conn.close()

    def test_removes_sessions_expires_index(self, temp_db):
        """Verify idx_user_sessions_expires index is removed by rollback."""
        migrate_add_auth_extensions(temp_db)
        rollback_auth_extensions(temp_db)

        conn = sqlite3.connect(temp_db)
        try:
            indexes = get_index_names(conn, "user_sessions")
            assert "idx_user_sessions_expires" not in indexes, (
                "idx_user_sessions_expires index still exists after rollback"
            )
        finally:
            conn.close()

    def test_removes_locked_until_index(self, temp_db):
        """Verify idx_users_locked_until index is removed by rollback."""
        migrate_add_auth_extensions(temp_db)
        rollback_auth_extensions(temp_db)

        conn = sqlite3.connect(temp_db)
        try:
            indexes = get_index_names(conn, "users")
            assert "idx_users_locked_until" not in indexes, (
                "idx_users_locked_until index still exists after rollback"
            )
        finally:
            conn.close()

    def test_is_idempotent(self, temp_db):
        """Verify rollback can be run multiple times without errors."""
        migrate_add_auth_extensions(temp_db)

        # Run rollback twice
        rollback_auth_extensions(temp_db)
        rollback_auth_extensions(temp_db)  # Should not raise

        # Verify all columns are removed
        conn = sqlite3.connect(temp_db)
        try:
            columns = get_table_columns(conn, "users")
            unexpected_columns = {
                "must_change_password",
                "failed_attempts",
                "locked_until",
            }
            for col in unexpected_columns:
                assert col not in columns, (
                    f"Column {col} still exists after idempotent rollback"
                )
        finally:
            conn.close()

    def test_preserves_existing_base_columns(self, temp_db):
        """Verify rollback preserves original users table columns."""
        # Get original columns before migration
        conn = sqlite3.connect(temp_db)
        original_columns = get_table_columns(conn, "users")
        conn.close()

        # Run full migration then rollback
        migrate_add_auth_extensions(temp_db)
        rollback_auth_extensions(temp_db)

        # Verify original columns are preserved
        conn = sqlite3.connect(temp_db)
        try:
            final_columns = get_table_columns(conn, "users")
            for col in original_columns:
                assert col in final_columns, (
                    f"Original column {col} was lost during rollback"
                )
        finally:
            conn.close()

    def test_full_migrate_rollback_cycle(self, temp_db):
        """Verify complete migrate-then-rollback cycle leaves DB in original state."""
        # Get original state
        conn = sqlite3.connect(temp_db)
        original_user_columns = get_table_columns(conn, "users")
        original_sessions_indexes = get_index_names(conn, "user_sessions")
        conn.close()

        # Run full cycle
        migrate_add_auth_extensions(temp_db)
        rollback_auth_extensions(temp_db)

        # Verify state matches original
        conn = sqlite3.connect(temp_db)
        try:
            final_user_columns = get_table_columns(conn, "users")
            final_sessions_indexes = get_index_names(conn, "user_sessions")

            # User columns should match
            assert original_user_columns == final_user_columns, (
                "Users table columns changed after full migrate/rollback cycle"
            )

            # Sessions indexes should match (migration adds one new index)
            assert final_sessions_indexes == original_sessions_indexes, (
                "User_sessions indexes changed after full migrate/rollback cycle"
            )
        finally:
            conn.close()


class TestMigrationEdgeCases:
    """Test edge cases and error conditions."""

    def test_rollback_without_migrate(self, temp_db):
        """Verify rollback is safe to run without prior migration."""
        # Should not raise
        rollback_auth_extensions(temp_db)

        # Users table should still have its base columns
        conn = sqlite3.connect(temp_db)
        try:
            columns = get_table_columns(conn, "users")
            assert "username" in columns
            assert "hashed_password" in columns
        finally:
            conn.close()

    def test_migrate_on_empty_db_without_users(self, temp_db):
        """Verify migration works when users table exists but is empty."""
        migrate_add_auth_extensions(temp_db)

        conn = sqlite3.connect(temp_db)
        try:
            columns = get_table_columns(conn, "users")
            assert "must_change_password" in columns
            assert "failed_attempts" in columns
            assert "locked_until" in columns
        finally:
            conn.close()
