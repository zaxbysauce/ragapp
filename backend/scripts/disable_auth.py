#!/usr/bin/env python3
"""
Auth Disable Script

This script disables user authentication and reverts the system to token-only mode.
Useful for:
- Emergency access when auth system is broken
- Migrating back to simple token-based auth
- Troubleshooting auth-related issues

Usage:
    python disable_auth.py [--restore]

Options:
    --restore    Restore user authentication (undo the disable)

WARNING: This script modifies database tables and should be used with caution.
Always backup your database before running this script.
"""

import argparse
import os
import sys
import sqlite3
from pathlib import Path

# Whitelist of valid auth tables
AUTH_TABLES = {
    'users', 'user_sessions', 'organizations', 'org_members',
    'groups', 'group_members', 'vault_members', 'vault_group_access',
    'users_disabled', 'user_sessions_disabled', 'org_members_disabled',
    'vault_members_disabled'
}

def validate_table_name(table: str) -> None:
    """Validate table name against whitelist."""
    if table not in AUTH_TABLES:
        raise ValueError(f"Invalid table name: {table}")


def get_database_path():
    """Get the database path from environment or default location."""
    # Try to import settings from the app
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from app.config import settings
        return str(settings.sqlite_path)
    except ImportError:
        # Fallback to default location
        default_path = Path(__file__).parent.parent / "data" / "kv_store.db"
        return str(default_path)


def check_backup_exists(db_path):
    """Check if a backup of the auth tables exists."""
    backup_dir = Path(db_path).parent / "auth_backup"
    return backup_dir.exists() and any(backup_dir.iterdir())


def backup_auth_tables(conn, backup_dir):
    """Backup auth-related tables to SQL files."""
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(exist_ok=True)
    
    tables_to_backup = [
        'users',
        'user_sessions',
        'org_members',
        'vault_members',
    ]
    
    cursor = conn.cursor()
    
    for table in tables_to_backup:
        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        )
        if not cursor.fetchone():
            print(f"  Table '{table}' does not exist, skipping...")
            continue
        
        backup_file = backup_dir / f"{table}.sql"
        
        # Get table schema
        validate_table_name(table)
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
        schema = cursor.fetchone()[0]
        
        # Get table data
        validate_table_name(table)
        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        
        # Get column names
        validate_table_name(table)
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Write backup
        with open(backup_file, 'w') as f:
            f.write(f"-- Backup of {table} table\n")
            f.write(f"-- Generated: {__import__('datetime').datetime.now().isoformat()}\n\n")
            f.write(f"{schema};\n\n")
            
            if rows:
                placeholders = ', '.join(['?' for _ in columns])
                f.write(f"INSERT INTO {table} ({', '.join(columns)}) VALUES\n")
                
                for i, row in enumerate(rows):
                    values = []
                    for val in row:
                        if val is None:
                            values.append('NULL')
                        elif isinstance(val, str):
                            # Escape single quotes
                            escaped = val.replace("'", "''")
                            values.append(f"'{escaped}'")
                        elif isinstance(val, int):
                            values.append(str(val))
                        else:
                            values.append(f"'{str(val)}'")
                    
                    suffix = ',' if i < len(rows) - 1 else ';'
                    f.write(f"  ({', '.join(values)}){suffix}\n")
        
        print(f"  Backed up {len(rows)} rows from '{table}' to {backup_file}")


def restore_auth_tables(conn, backup_dir):
    """Restore auth-related tables from SQL files."""
    backup_dir = Path(backup_dir)
    
    if not backup_dir.exists():
        print("Error: No backup directory found!")
        return False
    
    cursor = conn.cursor()
    
    # Order matters for foreign key constraints
    tables_to_restore = [
        'users',
        'user_sessions',
        'org_members',
        'vault_members',
    ]
    
    for table in tables_to_restore:
        backup_file = backup_dir / f"{table}.sql"
        
        if not backup_file.exists():
            print(f"  Backup for '{table}' not found, skipping...")
            continue
        
        # Read and execute the SQL
        with open(backup_file, 'r') as f:
            sql = f.read()
        
        # Split into individual statements (simple approach)
        statements = [s.strip() for s in sql.split(';') if s.strip()]
        
        for statement in statements:
            try:
                cursor.execute(statement)
            except sqlite3.Error as e:
                print(f"  Warning: Could not execute statement: {e}")
        
        print(f"  Restored table '{table}'")
    
    conn.commit()
    return True


def disable_auth(db_path):
    """Disable user authentication and revert to token-only mode."""
    print(f"\nDisabling authentication for database: {db_path}")
    print("=" * 60)
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create backup directory
    backup_dir = Path(db_path).parent / "auth_backup"
    
    # Check if already disabled
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='users_disabled'"
    )
    if cursor.fetchone():
        print("Authentication is already disabled.")
        conn.close()
        return True
    
    print("\n1. Creating backup of auth tables...")
    backup_auth_tables(conn, backup_dir)
    
    print("\n2. Renaming auth tables (disabling)...")
    
    # Rename tables to disable them
    tables_to_disable = [
        ('users', 'users_disabled'),
        ('user_sessions', 'user_sessions_disabled'),
        ('org_members', 'org_members_disabled'),
        ('vault_members', 'vault_members_disabled'),
    ]
    
    for old_name, new_name in tables_to_disable:
        try:
            cursor.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{old_name}'"
            )
            if cursor.fetchone():
                validate_table_name(old_name)
                cursor.execute(f"ALTER TABLE {old_name} RENAME TO {new_name}")
                print(f"  Renamed '{old_name}' -> '{new_name}'")
        except sqlite3.Error as e:
            print(f"  Warning: Could not rename '{old_name}': {e}")
    
    print("\n3. Creating minimal users table for token-only mode...")
    
    # Create a minimal users table that accepts any token
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT,
            full_name TEXT,
            role TEXT DEFAULT 'superadmin',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login_at TIMESTAMP
        )
    ''')
    
    # Insert a default superadmin user
    cursor.execute('''
        INSERT OR IGNORE INTO users (id, username, full_name, role, is_active)
        VALUES (1, 'admin', 'System Administrator', 'superadmin', 1)
    ''')
    
    print("  Created minimal users table with default admin user")
    
    # Create minimal sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER REFERENCES users(id),
            refresh_token_hash TEXT UNIQUE,
            expires_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used_at TIMESTAMP,
            ip_address TEXT,
            user_agent TEXT
        )
    ''')
    
    print("  Created minimal sessions table")
    
    # Create minimal org_members and vault_members for compatibility
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS org_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            org_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT DEFAULT 'admin',
            joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vault_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vault_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            permission TEXT DEFAULT 'admin',
            granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            granted_by INTEGER
        )
    ''')
    
    print("  Created minimal member tables")
    
    conn.commit()
    conn.close()
    
    print("\n" + "=" * 60)
    print("Authentication has been DISABLED.")
    print("The system is now in token-only mode.")
    print(f"\nBackup location: {backup_dir}")
    print("\nTo restore authentication, run:")
    print(f"  python {__file__} --restore")
    print("=" * 60 + "\n")
    
    return True


def restore_auth(db_path):
    """Restore user authentication from backup."""
    print(f"\nRestoring authentication for database: {db_path}")
    print("=" * 60)
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return False
    
    backup_dir = Path(db_path).parent / "auth_backup"
    
    if not backup_dir.exists():
        print(f"Error: No backup found at {backup_dir}")
        print("Cannot restore - no backup exists!")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if auth is currently disabled
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='users_disabled'"
    )
    if not cursor.fetchone():
        print("Warning: Authentication does not appear to be disabled.")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            conn.close()
            return False
    
    print("\n1. Dropping temporary tables...")
    
    temp_tables = ['users', 'user_sessions', 'org_members', 'vault_members']
    for table in temp_tables:
        try:
            validate_table_name(table)
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            print(f"  Dropped '{table}'")
        except sqlite3.Error as e:
            print(f"  Warning: Could not drop '{table}': {e}")
    
    print("\n2. Restoring original tables from backup...")
    
    # Restore from backup
    restore_auth_tables(conn, backup_dir)
    
    print("\n3. Dropping disabled tables...")
    
    disabled_tables = [
        'users_disabled',
        'user_sessions_disabled',
        'org_members_disabled',
        'vault_members_disabled',
    ]
    
    for table in disabled_tables:
        try:
            validate_table_name(table)
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            print(f"  Dropped '{table}'")
        except sqlite3.Error as e:
            print(f"  Warning: Could not drop '{table}': {e}")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("Authentication has been RESTORED.")
    print("User authentication is now enabled.")
    print("=" * 60 + "\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Disable/restore user authentication',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Disable authentication (token-only mode)
    python disable_auth.py
    
    # Restore authentication
    python disable_auth.py --restore
    
    # Specify custom database path
    python disable_auth.py --db /path/to/database.db
        """
    )
    
    parser.add_argument(
        '--restore',
        action='store_true',
        help='Restore user authentication (undo disable)'
    )
    
    parser.add_argument(
        '--db',
        type=str,
        help='Path to SQLite database (default: auto-detect)'
    )
    
    parser.add_argument(
        '--yes',
        '-y',
        action='store_true',
        help='Skip confirmation prompts'
    )
    
    args = parser.parse_args()
    
    # Get database path
    db_path = args.db or get_database_path()
    
    print("\n" + "=" * 60)
    print("RAG Application Auth Management Tool")
    print("=" * 60)
    
    if args.restore:
        print("\nMODE: Restore Authentication")
        if not args.yes:
            response = input(f"\nThis will restore auth tables from backup.\n"
                           f"Database: {db_path}\n"
                           f"Continue? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted.")
                return
        
        success = restore_auth(db_path)
    else:
        print("\nMODE: Disable Authentication")
        print("\nWARNING: This will disable user authentication and revert")
        print("         the system to token-only mode.")
        print("\nA backup of your auth tables will be created.")
        
        if not args.yes:
            response = input(f"\nDatabase: {db_path}\n"
                           f"Continue? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted.")
                return
        
        success = disable_auth(db_path)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
