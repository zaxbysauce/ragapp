"""
SQLite database initialization and schema for RAGAPPv2.

This module provides the database schema and initialization helper for the application.
"""

import logging
import sqlite3
import threading
from pathlib import Path
from queue import Queue, Empty, Full
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Database schema definition
SCHEMA = """
-- Vaults table: stores document collection vaults
CREATE TABLE IF NOT EXISTS vaults (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Files table: stores uploaded file metadata
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vault_id INTEGER NOT NULL DEFAULT 1,
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_hash TEXT,
    file_size INTEGER NOT NULL,
    file_type TEXT,
    chunk_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'indexed', 'error')),
    error_message TEXT,
    source TEXT DEFAULT 'upload',
    email_subject TEXT,
    email_sender TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (vault_id) REFERENCES vaults(id)
);

-- Memories table: stores processed document chunks with embeddings
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vault_id INTEGER,
    content TEXT NOT NULL,
    category TEXT,
    tags TEXT,       -- JSON array of tags
    source TEXT,     -- Source reference
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Full-text search virtual table for memories content
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    category,
    content='memories',
    content_rowid='id'
);

-- Trigger to keep FTS index in sync with memories table (insert)
CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, category) VALUES (new.id, new.content, new.category);
END;

-- Trigger to keep FTS index in sync with memories table (delete)
CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, category) VALUES ('delete', old.id, old.content, old.category);
END;

-- Trigger to keep FTS index in sync with memories table (update)
CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, category) VALUES ('delete', old.id, old.content, old.category);
    INSERT INTO memories_fts(rowid, content, category) VALUES (new.id, new.content, new.category);
END;

-- Chat sessions table: stores conversation sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vault_id INTEGER NOT NULL DEFAULT 1,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (vault_id) REFERENCES vaults(id)
);

-- Chat messages table: stores individual messages within sessions
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    sources TEXT,    -- JSON array of source references
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);

-- Document actions for auditing admin operations
CREATE TABLE IF NOT EXISTS document_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    user_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hmac_sha256 TEXT NOT NULL
);

-- Admin feature toggles
CREATE TABLE IF NOT EXISTS admin_toggles (
    feature TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit log for toggle changes
CREATE TABLE IF NOT EXISTS audit_toggle_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    user_id TEXT,
    ip TEXT,
    timestamp TEXT NOT NULL,
    key_version TEXT,
    hmac_sha256 TEXT NOT NULL
);

-- Secret key metadata for audit hashing
CREATE TABLE IF NOT EXISTS secret_keys (
    version TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- System flags for maintenance mode and feature toggles
CREATE TABLE IF NOT EXISTS system_flags (
    name TEXT PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 0,
    reason TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Settings key-value store for persistence across restarts
CREATE TABLE IF NOT EXISTS settings_kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Users table: stores user accounts for authentication
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    hashed_password TEXT NOT NULL,
    full_name TEXT DEFAULT '',
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('superadmin','admin','member','viewer')),
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP
);

-- Organizations table: stores organization entities
CREATE TABLE IF NOT EXISTS organizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Organization members table: links users to organizations with roles
CREATE TABLE IF NOT EXISTS org_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('admin','member')),
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(org_id, user_id)
);

-- Groups table: stores permission groups within organizations
CREATE TABLE IF NOT EXISTS groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE,
    UNIQUE(org_id, name)
);

-- Group members table: links users to groups
CREATE TABLE IF NOT EXISTS group_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(group_id, user_id)
);

-- Vault members table: direct user access to vaults
CREATE TABLE IF NOT EXISTS vault_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vault_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    permission TEXT NOT NULL DEFAULT 'read' CHECK (permission IN ('read','write','admin')),
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    granted_by INTEGER,
    FOREIGN KEY (vault_id) REFERENCES vaults(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (granted_by) REFERENCES users(id) ON DELETE SET NULL,
    UNIQUE(vault_id, user_id)
);

-- Vault group access table: group-based access to vaults
CREATE TABLE IF NOT EXISTS vault_group_access (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vault_id INTEGER NOT NULL,
    group_id INTEGER NOT NULL,
    permission TEXT NOT NULL DEFAULT 'read' CHECK (permission IN ('read','write','admin')),
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    granted_by INTEGER,
    FOREIGN KEY (vault_id) REFERENCES vaults(id) ON DELETE CASCADE,
    FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE,
    FOREIGN KEY (granted_by) REFERENCES users(id) ON DELETE SET NULL,
    UNIQUE(vault_id, group_id)
);

-- User sessions table: stores refresh tokens for authentication
CREATE TABLE IF NOT EXISTS user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    refresh_token_hash TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP,
    ip_address TEXT,
    user_agent TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Indexes for new tables
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_org_members_org_id ON org_members(org_id);
CREATE INDEX IF NOT EXISTS idx_org_members_user_id ON org_members(user_id);
CREATE INDEX IF NOT EXISTS idx_group_members_group_id ON group_members(group_id);
CREATE INDEX IF NOT EXISTS idx_group_members_user_id ON group_members(user_id);
CREATE INDEX IF NOT EXISTS idx_vault_members_vault_id ON vault_members(vault_id);
CREATE INDEX IF NOT EXISTS idx_vault_members_user_id ON vault_members(user_id);
CREATE INDEX IF NOT EXISTS idx_vault_group_access_vault_id ON vault_group_access(vault_id);
CREATE INDEX IF NOT EXISTS idx_vault_group_access_group_id ON vault_group_access(group_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_refresh_hash ON user_sessions(refresh_token_hash);
"""


def init_db(sqlite_path: str) -> None:
    """
    Initialize the SQLite database with the schema.

    Args:
        sqlite_path: Path to the SQLite database file.

    Raises:
        sqlite3.Error: If database initialization fails.
    """
    # Ensure parent directory exists
    db_path = Path(sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Connect and execute schema
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SCHEMA)
        # Ensure default vault exists
        conn.execute(
            "INSERT OR IGNORE INTO vaults (id, name, description) VALUES (1, 'Default', 'Default vault')"
        )
        conn.commit()
    finally:
        conn.close()


def run_migrations(sqlite_path: str) -> None:
    """
    Run database migrations to initialize the schema.

    This function calls init_db to create the database and apply the schema.
    It is intended to be called during application startup to ensure the
    database is properly initialized.

    Phase 3 migrations included:
    - Users table (3.0.1)
    - Organizations table (3.0.2)
    - Groups table (3.0.3)
    - Vault permissions (3.0.4)
    - User sessions (3.0.5)
    - Vault columns (3.0.6)

    Args:
        sqlite_path: Path to the SQLite database file.

    Returns:
        None
    """
    init_db(sqlite_path)
    migrate_add_vaults(sqlite_path)
    migrate_add_email_columns(sqlite_path)
    migrate_add_versioning_fields(sqlite_path)

    # Phase 3: RBAC and permission system migrations
    migrate_add_user_org_tables(sqlite_path)
    migrate_add_vault_permission_columns(sqlite_path)

    # Phase 1: Auth extensions (password policy, account lockout)
    migrate_add_auth_extensions(sqlite_path)

    # Phase 1: Chat session persistence - add user_id to chat_sessions
    migrate_add_chat_sessions_user_id(sqlite_path)

    # Vault path migration: name-based → ID-based directories
    try:
        from app.services.upload_path import migrate_vault_paths
        from app.config import settings
        import asyncio

        migrate_vault_paths(sqlite_path, settings.data_dir)
    except Exception as e:
        logger.warning(f"Vault path migration failed (continuing): {e}")


def migrate_add_vaults(sqlite_path: str) -> None:
    """
    Migration: Add vaults table and vault_id columns to existing databases.

    This migration is idempotent — safe to run multiple times.
    It creates the vaults table, inserts a default vault, adds vault_id
    columns to files/memories/chat_sessions if missing, and backfills
    existing rows with the default vault.

    Args:
        sqlite_path: Path to the SQLite database file.
    """
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")

        # 1. Create vaults table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vaults (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. Insert default vault
        conn.execute(
            "INSERT OR IGNORE INTO vaults (id, name, description) VALUES (1, 'Default', 'Default vault')"
        )

        # 3. Add vault_id columns if missing (SQLite doesn't support IF NOT EXISTS for columns)
        # Check files table for vault_id column
        cursor = conn.execute("PRAGMA table_info(files)")
        has_files_vault_id = any(row[1] == "vault_id" for row in cursor.fetchall())
        if not has_files_vault_id:
            conn.execute(
                "ALTER TABLE files ADD COLUMN vault_id INTEGER NOT NULL DEFAULT 1"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_vault_id ON files(vault_id)"
            )

        # Check memories table for vault_id column
        cursor = conn.execute("PRAGMA table_info(memories)")
        has_memories_vault_id = any(row[1] == "vault_id" for row in cursor.fetchall())
        if not has_memories_vault_id:
            conn.execute("ALTER TABLE memories ADD COLUMN vault_id INTEGER")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_vault_id ON memories(vault_id)"
            )

        # Check chat_sessions table for vault_id column
        cursor = conn.execute("PRAGMA table_info(chat_sessions)")
        has_chat_sessions_vault_id = any(
            row[1] == "vault_id" for row in cursor.fetchall()
        )
        if not has_chat_sessions_vault_id:
            conn.execute(
                "ALTER TABLE chat_sessions ADD COLUMN vault_id INTEGER NOT NULL DEFAULT 1"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_sessions_vault_id ON chat_sessions(vault_id)"
            )

        # 4. Backfill existing rows with default vault
        conn.execute("UPDATE files SET vault_id = 1 WHERE vault_id IS NULL")
        conn.execute("UPDATE chat_sessions SET vault_id = 1 WHERE vault_id IS NULL")
        # memories: NULL vault_id is intentional (global), no backfill needed

        conn.commit()
    finally:
        conn.close()


def migrate_add_email_columns(sqlite_path: str) -> None:
    """
    Migration: Add email tracking columns to files table.

    Adds source, email_subject, and email_sender columns to track
    documents ingested via email.

    Args:
        sqlite_path: Path to the SQLite database file.
    """
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")

        # Add source column (track upload, scan, email)
        cursor = conn.execute("PRAGMA table_info(files)")
        has_source = any(row[1] == "source" for row in cursor.fetchall())
        if not has_source:
            conn.execute(
                "ALTER TABLE files ADD COLUMN source TEXT NOT NULL DEFAULT 'upload'"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_source ON files(source)")

        # Add email_subject column (nullable)
        cursor = conn.execute("PRAGMA table_info(files)")
        has_email_subject = any(row[1] == "email_subject" for row in cursor.fetchall())
        if not has_email_subject:
            conn.execute("ALTER TABLE files ADD COLUMN email_subject TEXT")

        # Add email_sender column (nullable)
        cursor = conn.execute("PRAGMA table_info(files)")
        has_email_sender = any(row[1] == "email_sender" for row in cursor.fetchall())
        if not has_email_sender:
            conn.execute("ALTER TABLE files ADD COLUMN email_sender TEXT")

        conn.commit()
    finally:
        conn.close()


def migrate_add_versioning_fields(sqlite_path: str) -> None:
    """
    Migration: Add document versioning columns to the files table.

    Adds document_date, supersedes_file_id, and ingestion_version columns.
    This migration is idempotent — safe to run multiple times.

    Args:
        sqlite_path: Path to the SQLite database file.
    """
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")

        cursor = conn.execute("PRAGMA table_info(files)")
        columns = [row[1] for row in cursor.fetchall()]

        if "document_date" not in columns:
            conn.execute("ALTER TABLE files ADD COLUMN document_date TEXT")

        if "supersedes_file_id" not in columns:
            conn.execute("ALTER TABLE files ADD COLUMN supersedes_file_id INTEGER")

        if "ingestion_version" not in columns:
            conn.execute(
                "ALTER TABLE files ADD COLUMN ingestion_version INTEGER DEFAULT 1"
            )
            conn.execute(
                "UPDATE files SET ingestion_version = 1 WHERE ingestion_version IS NULL"
            )

        conn.commit()
    finally:
        conn.close()


def migrate_add_vault_permission_columns(sqlite_path: str) -> None:
    """
    Migration: Add permission columns to vaults table.

    Adds owner_id, org_id, and visibility columns to support
    the new RBAC permission system.

    Args:
        sqlite_path: Path to the SQLite database file.
    """
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")

        cursor = conn.execute("PRAGMA table_info(vaults)")
        columns = [row[1] for row in cursor.fetchall()]

        # Add owner_id column
        if "owner_id" not in columns:
            conn.execute("ALTER TABLE vaults ADD COLUMN owner_id INTEGER")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vaults_owner_id ON vaults(owner_id)"
            )

        # Add org_id column
        if "org_id" not in columns:
            conn.execute("ALTER TABLE vaults ADD COLUMN org_id INTEGER")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vaults_org_id ON vaults(org_id)"
            )

        # Add visibility column
        if "visibility" not in columns:
            conn.execute(
                "ALTER TABLE vaults ADD COLUMN visibility TEXT DEFAULT 'private' "
                "CHECK (visibility IN ('private', 'org', 'public'))"
            )
            # Set default visibility for existing vaults
            conn.execute(
                "UPDATE vaults SET visibility = 'private' WHERE visibility IS NULL"
            )

        conn.commit()
    finally:
        conn.close()


def migrate_add_user_org_tables(sqlite_path: str) -> None:
    """
    Migration: Ensure user and organization tables exist.

    This migration runs the full schema which includes users, organizations,
    groups, and permission tables. It is idempotent — safe to run multiple times.

    Args:
        sqlite_path: Path to the SQLite database file.
    """
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def migrate_add_auth_extensions(sqlite_path: str) -> None:
    """
    Migration: Add password policy and account lockout columns to users table.

    Phase 1 auth migration - adds must_change_password, failed_attempts,
    and locked_until columns to the users table. Also creates indexes for
    session cleanup and lockout lookup. This migration is idempotent.

    Args:
        sqlite_path: Path to the SQLite database file.
    """
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")

        # Get existing columns from users table
        cursor = conn.execute("PRAGMA table_info(users)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Add must_change_password column if not exists
        if "must_change_password" not in existing_columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0"
            )

        # Add failed_attempts column if not exists
        if "failed_attempts" not in existing_columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN failed_attempts INTEGER NOT NULL DEFAULT 0"
            )

        # Add locked_until column if not exists (nullable TIMESTAMP)
        if "locked_until" not in existing_columns:
            conn.execute("ALTER TABLE users ADD COLUMN locked_until TIMESTAMP")

        # Create index for session cleanup (expires_at on user_sessions)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON user_sessions(expires_at)"
        )

        # Create index for lockout lookup (locked_until on users)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_locked_until ON users(locked_until)"
        )

        conn.commit()
    finally:
        conn.close()


def rollback_auth_extensions(sqlite_path: str) -> None:
    """
    Down-migration: Remove password policy and account lockout columns.

    Reverses the changes made by migrate_add_auth_extensions by dropping
    the added columns and indexes. Safe to run multiple times (idempotent).
    Requires SQLite 3.35.0+ for DROP COLUMN support.

    Args:
        sqlite_path: Path to the SQLite database file.
    """
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")

        # Drop indexes if they exist
        conn.execute("DROP INDEX IF EXISTS idx_user_sessions_expires")
        conn.execute("DROP INDEX IF EXISTS idx_users_locked_until")

        # Get existing columns from users table
        cursor = conn.execute("PRAGMA table_info(users)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Drop columns if they exist (SQLite 3.35.0+ supports DROP COLUMN)
        if "locked_until" in existing_columns:
            conn.execute("ALTER TABLE users DROP COLUMN locked_until")

        if "failed_attempts" in existing_columns:
            conn.execute("ALTER TABLE users DROP COLUMN failed_attempts")

        if "must_change_password" in existing_columns:
            conn.execute("ALTER TABLE users DROP COLUMN must_change_password")

        conn.commit()
    finally:
        conn.close()


def migrate_add_chat_sessions_user_id(sqlite_path: str) -> None:
    """
    Migration: Add user_id column to chat_sessions table for session persistence.

    Phase 1 migration - adds user_id column and index for user-scoped
    chat session access. This migration is idempotent.

    Args:
        sqlite_path: Path to the SQLite database file.
    """
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")

        # Check if user_id column already exists in chat_sessions
        cursor = conn.execute("PRAGMA table_info(chat_sessions)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Add user_id column if not exists
        if "user_id" not in existing_columns:
            conn.execute("ALTER TABLE chat_sessions ADD COLUMN user_id INTEGER")

        # Create index on user_id for faster lookups
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id)"
        )

        conn.commit()
    finally:
        conn.close()


def rollback_chat_sessions_user_id(sqlite_path: str) -> None:
    """
    Down-migration: Remove user_id column from chat_sessions table.

    Reverses the changes made by migrate_add_chat_sessions_user_id.
    Safe to run multiple times (idempotent).
    Requires SQLite 3.35.0+ for DROP COLUMN support.

    Args:
        sqlite_path: Path to the SQLite database file.
    """
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")

        # Drop index if exists
        conn.execute("DROP INDEX IF EXISTS idx_chat_sessions_user_id")

        # Get existing columns from chat_sessions table
        cursor = conn.execute("PRAGMA table_info(chat_sessions)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Drop user_id column if exists (SQLite 3.35.0+ supports DROP COLUMN)
        if "user_id" in existing_columns:
            conn.execute("ALTER TABLE chat_sessions DROP COLUMN user_id")

        conn.commit()
    finally:
        conn.close()


def get_db_connection(sqlite_path: str) -> sqlite3.Connection:
    """
    Get a connection to the SQLite database.

    Args:
        sqlite_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection: Database connection with row factory set.
    """
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


class SQLiteConnectionPool:
    """
    A connection pool for SQLite databases.

    Manages a pool of reusable SQLite connections to improve performance
    in multi-threaded environments.
    """

    def __init__(self, sqlite_path: str, max_size: int = 5):
        """
        Initialize the connection pool.

        Args:
            sqlite_path: Path to the SQLite database file.
            max_size: Maximum number of connections in the pool.
        """
        self.sqlite_path = sqlite_path
        self.max_size = max_size
        self._pool = Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._created_count = 0
        self._closed = False

    def _create_connection(self) -> sqlite3.Connection:
        """
        Create a new SQLite connection with proper settings.

        Returns:
            sqlite3.Connection: A new database connection.

        Raises:
            sqlite3.Error: If connection creation fails.
        """
        try:
            conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            conn.execute("PRAGMA foreign_keys = ON;")
            return conn
        except Exception:
            # Decrement created count on any failure
            with self._lock:
                self._created_count -= 1
            raise

    def _validate_connection(self, conn: sqlite3.Connection) -> bool:
        """
        Validate that a connection is still alive and usable.

        Re-applies PRAGMA foreign_keys = ON after validation to ensure
        foreign key constraints are enforced.

        Args:
            conn: The connection to validate.

        Returns:
            bool: True if the connection is valid, False otherwise.
        """
        try:
            conn.execute("SELECT 1")
            # Re-apply foreign keys pragma after validation
            conn.execute("PRAGMA foreign_keys = ON")
            return True
        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            # Rollback any failed transaction state
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            return False
        except sqlite3.Error:
            return False

    def get_connection(self, max_wait_attempts: int = 3) -> sqlite3.Connection:
        """
        Get a connection from the pool.

        If the pool has available connections, returns one from the pool.
        Otherwise, creates a new connection if under max_size limit.
        Validates connections before returning them.

        Args:
            max_wait_attempts: Maximum number of wait attempts when pool is at capacity.

        Returns:
            sqlite3.Connection: A database connection.

        Raises:
            RuntimeError: If the pool has been closed or max wait attempts exhausted.
        """
        if self._closed:
            raise RuntimeError("Connection pool has been closed")

        attempts = 0
        while attempts < max_wait_attempts:
            # Try to get an existing connection from the pool
            try:
                conn = self._pool.get_nowait()
                # Validate the connection before returning it
                if self._validate_connection(conn):
                    return conn
                else:
                    # Connection is invalid, decrement count and try again
                    with self._lock:
                        self._created_count -= 1
                    try:
                        conn.close()
                    except sqlite3.Error:
                        pass
                    continue
            except Empty:
                pass

            # No available connections, try to create a new one if under limit
            with self._lock:
                if self._created_count < self.max_size:
                    self._created_count += 1
                    try:
                        return self._create_connection()
                    except sqlite3.Error:
                        self._created_count -= 1
                        raise

            # If at max capacity, block until a connection is available
            try:
                conn = self._pool.get(timeout=5)
                # Validate the connection before returning it
                if self._validate_connection(conn):
                    return conn
                else:
                    # Connection is invalid, decrement count and try again
                    with self._lock:
                        self._created_count -= 1
                    try:
                        conn.close()
                    except sqlite3.Error:
                        pass
                    continue
            except Empty:
                # Timeout occurred, increment attempts and retry
                attempts += 1
                continue

        # Max attempts exhausted
        raise RuntimeError(
            f"Could not obtain a connection from the pool after {max_wait_attempts} attempts"
        )

    def release_connection(self, conn: sqlite3.Connection) -> None:
        """
        Release a connection back to the pool.

        Args:
            conn: The connection to release back to the pool.

        Raises:
            RuntimeError: If the pool has been closed.
        """
        if self._closed:
            raise RuntimeError("Connection pool has been closed")

        try:
            self._pool.put_nowait(conn)
        except Full:
            # Pool is full, close the connection
            conn.close()

    def close_all(self) -> None:
        """
        Close all connections in the pool.

        This closes all pooled connections and prevents further use of the pool.
        """
        with self._lock:
            self._closed = True
            # Close all connections in the pool using while True/except Empty pattern
            while True:
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                except Empty:
                    break

    @contextmanager
    def connection(self):
        """
        Context manager for getting and releasing a connection.

        Automatically releases the connection back to the pool when done.

        Example:
            with pool.connection() as conn:
                cursor = conn.execute("SELECT * FROM table")
                results = cursor.fetchall()
        """
        conn = None
        try:
            conn = self.get_connection()
            yield conn
        finally:
            if conn is not None:
                try:
                    self.release_connection(conn)
                except Exception:
                    # Ignore release errors to avoid masking original exception
                    pass


# Global pool cache for singleton pattern
_pool_cache: dict[str, SQLiteConnectionPool] = {}
_pool_cache_lock = threading.Lock()


def get_pool(sqlite_path: str, max_size: int = 5) -> SQLiteConnectionPool:
    """
    Get or create a connection pool for the given SQLite path.

    This function implements a singleton pattern, returning the same
    pool instance for the same sqlite_path.

    Args:
        sqlite_path: Path to the SQLite database file.
        max_size: Maximum number of connections in the pool.

    Returns:
        SQLiteConnectionPool: A connection pool instance.
    """
    global _pool_cache

    with _pool_cache_lock:
        if sqlite_path not in _pool_cache:
            _pool_cache[sqlite_path] = SQLiteConnectionPool(sqlite_path, max_size)
        return _pool_cache[sqlite_path]
