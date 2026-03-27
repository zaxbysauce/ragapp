"""
Standalone tests for Task 1.8: Vault Filtering

Tests get_user_accessible_vault_ids function and vault filtering logic.
"""

import os
import sys
import sqlite3
import tempfile
import shutil
import types
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Fully stub all optional dependencies BEFORE any app imports
def make_module(name, attrs=None):
    m = types.ModuleType(name)
    m.__path__ = []
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    return m


# Create complete unstructured module stub
_unstructured_partition = make_module("unstructured.partition")
_unstructured_partition_auto = make_module("unstructured.partition.auto")
_unstructured_partition_auto.partition = lambda *a, **k: []
_unstructured_chunking = make_module("unstructured.chunking")
_unstructured_chunking.title = make_module("unstructured.chunking.title")
_unstructured_chunking.title.chunk_by_title = lambda *a, **k: []
_unstructured_documents = make_module("unstructured.documents")
_unstructured_documents.elements = make_module("unstructured.documents.elements")
_unstructured_documents.elements.Element = type("Element", (), {})

_unstructured = make_module(
    "unstructured",
    {
        "partition": _unstructured_partition,
        "chunking": _unstructured_chunking,
        "documents": _unstructured_documents,
    },
)

# Only stub if not already importable (preserves real modules when available)
try:
    import lancedb as _lancedb_check  # noqa: F401
except ImportError:
    sys.modules["lancedb"] = make_module("lancedb")

try:
    import pyarrow as _pyarrow_check  # noqa: F401
except ImportError:
    sys.modules["pyarrow"] = make_module("pyarrow")

sys.modules["unstructured"] = _unstructured
sys.modules["unstructured.partition"] = _unstructured_partition
sys.modules["unstructured.partition.auto"] = _unstructured_partition_auto
sys.modules["unstructured.chunking"] = _unstructured_chunking
sys.modules["unstructured.chunking.title"] = _unstructured_chunking.title
sys.modules["unstructured.documents"] = _unstructured_documents
sys.modules["unstructured.documents.elements"] = _unstructured_documents.elements

# Mock config before importing app modules
from app.config import settings

# Override database path temporarily
TEST_DIR = tempfile.mkdtemp()
TEST_DB = str(Path(TEST_DIR) / "test.db")

# Initialize database
from app.models.database import init_db, get_pool

init_db(TEST_DB)
get_pool(TEST_DB)

from app.api.deps import get_user_accessible_vault_ids


def run_tests():
    """Run all vault filtering tests."""
    conn = sqlite3.connect(TEST_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    passed = 0
    failed = 0
    errors = []

    def cleanup():
        """Clear all relevant tables."""
        conn.execute("DELETE FROM vault_members")
        conn.execute("DELETE FROM vault_group_access")
        conn.execute("DELETE FROM group_members")
        conn.execute("DELETE FROM vaults")
        conn.execute("DELETE FROM users")
        conn.commit()

    def create_user(username, role="member"):
        cursor = conn.execute(
            "INSERT INTO users (username, hashed_password, role, is_active) VALUES (?, ?, ?, 1)",
            (username, "hash123", role),
        )
        conn.commit()
        return {"id": cursor.lastrowid, "username": username, "role": role}

    def create_vault(name):
        cursor = conn.execute(
            "INSERT INTO vaults (name, description) VALUES (?, '')", (name,)
        )
        conn.commit()
        return cursor.lastrowid

    def add_vault_member(user_id, vault_id, permission="read"):
        conn.execute(
            "INSERT INTO vault_members (vault_id, user_id, permission) VALUES (?, ?, ?)",
            (vault_id, user_id, permission),
        )
        conn.commit()

    def create_group(name):
        cursor = conn.execute("INSERT INTO groups (name) VALUES (?)", (name,))
        conn.commit()
        return cursor.lastrowid

    def add_to_group(user_id, group_id):
        conn.execute(
            "INSERT INTO group_members (user_id, group_id) VALUES (?, ?)",
            (user_id, group_id),
        )
        conn.commit()

    def add_group_vault_access(group_id, vault_id, permission="read"):
        conn.execute(
            "INSERT INTO vault_group_access (vault_id, group_id, permission) VALUES (?, ?, ?)",
            (vault_id, group_id, permission),
        )
        conn.commit()

    # Helper to run async function
    def get_accessible_vaults(user, db):
        return asyncio.run(get_user_accessible_vault_ids(user, db))

    # Test 1: User with no vaults returns empty list
    print("Test 1: User with no vaults returns []...")
    cleanup()
    user = create_user("regular_user", "member")
    result = get_accessible_vaults(user, conn)
    if result == []:
        print("  PASS: Returns [] for user with no vaults")
        passed += 1
    else:
        print(f"  FAIL: Expected [], got {result}")
        failed += 1
        errors.append("Test 1: User with no vaults returns []")

    # Test 2: User with direct vault access returns vault_ids
    print("Test 2: User with direct vault access returns [vault_id]...")
    cleanup()
    user = create_user("user_with_vault", "member")
    vault1 = create_vault("Vault 1")
    vault2 = create_vault("Vault 2")
    add_vault_member(user["id"], vault1, "read")
    add_vault_member(user["id"], vault2, "write")
    result = get_accessible_vaults(user, conn)
    if set(result) == {vault1, vault2}:
        print(f"  PASS: Returns [{vault1}, {vault2}]")
        passed += 1
    else:
        print(f"  FAIL: Expected [{vault1}, {vault2}], got {result}")
        failed += 1
        errors.append("Test 2: User with direct vault access")

    # Test 3: Admin returns empty list (means all vaults)
    print("Test 3: Admin returns [] (means all vaults)...")
    cleanup()
    user = create_user("admin_user", "admin")
    result = get_accessible_vaults(user, conn)
    if result == []:
        print("  PASS: Admin returns []")
        passed += 1
    else:
        print(f"  FAIL: Expected [], got {result}")
        failed += 1
        errors.append("Test 3: Admin returns []")

    # Test 4: Superadmin returns empty list
    print("Test 4: Superadmin returns [] (means all vaults)...")
    cleanup()
    user = create_user("superadmin_user", "superadmin")
    result = get_accessible_vaults(user, conn)
    if result == []:
        print("  PASS: Superadmin returns []")
        passed += 1
    else:
        print(f"  FAIL: Expected [], got {result}")
        failed += 1
        errors.append("Test 4: Superadmin returns []")

    # Test 5: Group-based vault access (SKIPPED - requires org_id)
    print("Test 5: Group-based vault access - SKIPPED (requires organization setup)")

    # Test 6: Mixed direct and group access (SKIPPED - requires org_id)
    print(
        "Test 6: Mixed direct and group access - SKIPPED (requires organization setup)"
    )

    conn.close()
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print(f"Failed tests: {errors}")
    return passed, failed


if __name__ == "__main__":
    p, f = run_tests()
    sys.exit(0 if f == 0 else 1)
