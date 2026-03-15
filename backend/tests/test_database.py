"""Unit tests for database schema initialization."""

import os
import sqlite3
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.models.database import get_db_connection, init_db, run_migrations


class TestDatabaseSchema(unittest.TestCase):
    """Test cases for database schema initialization."""

    def setUp(self):
        """Create a temporary database file for each test."""
        self.temp_fd, self.temp_db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)

    def tearDown(self):
        """Clean up the temporary database file."""
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_init_db_creates_required_tables(self):
        """Test that init_db creates all required tables and FTS virtual table."""
        # Initialize the database
        init_db(self.temp_db_path)

        # Connect and query sqlite_master for tables
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        # Get all tables and virtual tables
        cursor.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'virtual table')"
        )
        results = cursor.fetchall()
        conn.close()

        # Extract table names
        table_names = {name for name, _ in results}

        # Assert all required tables exist
        required_tables = {
            'files',
            'memories',
            'memories_fts',
            'chat_sessions',
            'chat_messages'
        }
        
        for table in required_tables:
            self.assertIn(
                table,
                table_names,
                f"Required table '{table}' was not created by init_db()"
            )

    def test_init_db_is_idempotent(self):
        """Test that init_db can be called multiple times without error."""
        # Initialize twice
        init_db(self.temp_db_path)
        init_db(self.temp_db_path)

        # Verify tables still exist (query both table and virtual table types)
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'virtual table')"
        )
        results = cursor.fetchall()
        conn.close()

        # Extract table names
        table_names = {name for name, _ in results}

        # Assert all required tables exist
        required_tables = {
            'files',
            'memories',
            'memories_fts',
            'chat_sessions',
            'chat_messages'
        }

        for table in required_tables:
            self.assertIn(
                table,
                table_names,
                f"Required table '{table}' was not found after idempotent init_db() calls"
            )

    def test_users_table_schema(self):
        """Test that users table has correct schema (3.0.1)."""
        init_db(self.temp_db_path)
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(users)")
        columns = {row[1]: row for row in cursor.fetchall()}
        conn.close()

        required_columns = [
            'id', 'username', 'hashed_password', 'full_name',
            'role', 'is_active', 'created_at', 'last_login_at'
        ]
        for col in required_columns:
            self.assertIn(col, columns, f"Column '{col}' missing from users table")

        # Check username is UNIQUE and COLLATE NOCASE
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='users'"
        )
        create_sql = cursor.fetchone()[0]
        conn.close()
        self.assertIn('UNIQUE', create_sql)
        self.assertIn('NOCASE', create_sql)

    def test_organizations_table_schema(self):
        """Test that organizations and org_members tables exist (3.0.2)."""
        init_db(self.temp_db_path)
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        self.assertIn('organizations', tables)
        self.assertIn('org_members', tables)

    def test_groups_table_schema(self):
        """Test that groups and group_members tables exist (3.0.3)."""
        init_db(self.temp_db_path)
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        self.assertIn('groups', tables)
        self.assertIn('group_members', tables)

    def test_vault_permissions_tables_schema(self):
        """Test that vault permission tables exist (3.0.4)."""
        init_db(self.temp_db_path)
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        self.assertIn('vault_members', tables)
        self.assertIn('vault_group_access', tables)

    def test_user_sessions_table_schema(self):
        """Test that user_sessions table exists (3.0.5)."""
        init_db(self.temp_db_path)
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        self.assertIn('user_sessions', tables)

    def test_vault_permission_columns(self):
        """Test that vault columns exist after migration (3.0.6)."""
        run_migrations(self.temp_db_path)
        conn = sqlite3.connect(self.temp_db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(vaults)")
            columns = {row[1] for row in cursor.fetchall()}
        finally:
            conn.close()

        self.assertIn('owner_id', columns)
        self.assertIn('org_id', columns)
        self.assertIn('visibility', columns)

    def test_foreign_key_pragma_applied(self):
        """Test that foreign keys pragma is applied in connections (3.0.7)."""
        init_db(self.temp_db_path)
        conn = get_db_connection(self.temp_db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys")
            fk_enabled = cursor.fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(fk_enabled, 1, "Foreign keys should be enabled")

    def test_schema_integrity(self):
        """Test overall schema integrity (3.0.8)."""
        run_migrations(self.temp_db_path)
        conn = sqlite3.connect(self.temp_db_path)
        try:
            cursor = conn.cursor()

            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            self.assertEqual(result, 'ok', "Database integrity check failed")

            # Run foreign key check
            cursor.execute("PRAGMA foreign_key_check")
            fk_violations = cursor.fetchall()
        finally:
            conn.close()

        self.assertEqual(len(fk_violations), 0, f"Foreign key violations found: {fk_violations}")

    def test_all_phase3_tables_exist(self):
        """Test that all Phase 3 tables are created by run_migrations (3.0.9)."""
        run_migrations(self.temp_db_path)
        conn = sqlite3.connect(self.temp_db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}
        finally:
            conn.close()

        phase3_tables = [
            'users', 'organizations', 'org_members', 'groups',
            'group_members', 'vault_members', 'vault_group_access',
            'user_sessions'
        ]
        for table in phase3_tables:
            self.assertIn(table, tables, f"Phase 3 table '{table}' not found")


if __name__ == '__main__':
    unittest.main()
