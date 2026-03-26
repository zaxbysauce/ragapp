"""
API tests for Task 1.8: Vault Filtering

Tests GET /vaults/accessible and chat/stream vault validation.
"""

import os
import sys
import sqlite3
import tempfile
import shutil
import types
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

sys.modules["lancedb"] = make_module("lancedb")
sys.modules["pyarrow"] = make_module("pyarrow")
sys.modules["unstructured"] = _unstructured
sys.modules["unstructured.partition"] = _unstructured_partition
sys.modules["unstructured.partition.auto"] = _unstructured_partition_auto
sys.modules["unstructured.chunking"] = _unstructured_chunking
sys.modules["unstructured.chunking.title"] = _unstructured_chunking.title
sys.modules["unstructured.documents"] = _unstructured_documents
sys.modules["unstructured.documents.elements"] = _unstructured_documents.elements

from fastapi.testclient import TestClient

from app.main import app
from app.models.database import init_db, migrate_add_auth_extensions
from app.api.deps import get_db, get_rag_engine
from app.services.auth_service import create_access_token

# Test database setup
TEST_DIR = tempfile.mkdtemp()
TEST_DB = str(Path(TEST_DIR) / "test.db")
init_db(TEST_DB)
migrate_add_auth_extensions(TEST_DB)


def get_connection_pool():
    """Create a simple connection pool."""
    from queue import Queue, Empty

    class Pool:
        def __init__(self, db_path):
            self.db_path = db_path
            self._pool = Queue(maxsize=5)
            self._closed = False

        def get_connection(self):
            if self._closed:
                raise RuntimeError("Pool closed")
            try:
                return self._pool.get_nowait()
            except Empty:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON;")
                return conn

        def release_connection(self, conn):
            if not self._closed:
                try:
                    self._pool.put_nowait(conn)
                except:
                    conn.close()

        def close_all(self):
            self._closed = True
            while True:
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                except Empty:
                    break

    return Pool(TEST_DB)


pool = get_connection_pool()


def override_get_db():
    conn = pool.get_connection()
    try:
        yield conn
    finally:
        pool.release_connection(conn)


# Mock RAG engine
class MockRAGEngine:
    async def query(self, *args, **kwargs):
        yield {"type": "done", "sources": [], "memories_used": []}


def override_get_rag_engine():
    return MockRAGEngine()


app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_rag_engine] = override_get_rag_engine

client = TestClient(app)


def cleanup():
    """Clear all relevant tables."""
    conn = pool.get_connection()
    conn.execute("DELETE FROM vault_members")
    conn.execute("DELETE FROM vault_group_access")
    conn.execute("DELETE FROM group_members")
    conn.execute("DELETE FROM vaults")
    conn.execute("DELETE FROM users")
    conn.commit()
    pool.release_connection(conn)


def create_user(username, role="member"):
    conn = pool.get_connection()
    cursor = conn.execute(
        "INSERT INTO users (username, hashed_password, role, is_active) VALUES (?, ?, ?, 1)",
        (username, "hash123", role),
    )
    conn.commit()
    user_id = cursor.lastrowid
    pool.release_connection(conn)
    token = create_access_token(user_id, username, role)
    return {"id": user_id, "username": username, "role": role, "token": token}


def create_vault(name):
    conn = pool.get_connection()
    cursor = conn.execute(
        "INSERT INTO vaults (name, description) VALUES (?, '')", (name,)
    )
    conn.commit()
    vault_id = cursor.lastrowid
    pool.release_connection(conn)
    return vault_id


def add_vault_member(user_id, vault_id, permission="read"):
    conn = pool.get_connection()
    conn.execute(
        "INSERT INTO vault_members (vault_id, user_id, permission) VALUES (?, ?, ?)",
        (vault_id, user_id, permission),
    )
    conn.commit()
    pool.release_connection(conn)


def run_api_tests():
    """Run all API-level vault filtering tests."""
    passed = 0
    failed = 0
    errors = []

    # Test 1: GET /vaults/accessible returns empty for user with no vaults
    print("Test 1: GET /vaults/accessible returns [] for user with no vaults...")
    cleanup()
    user = create_user("no_access_user")
    response = client.get(
        "/api/vaults/accessible",
        headers={"Authorization": f"Bearer {user['token']}"},
    )
    if response.status_code == 200 and response.json()["vault_ids"] == []:
        print("  PASS: Returns vault_ids: []")
        passed += 1
    else:
        print(
            f"  FAIL: Expected 200 + [], got {response.status_code} + {response.json()}"
        )
        failed += 1
        errors.append("Test 1: /vaults/accessible empty user")

    # Test 2: GET /vaults/accessible returns vault_ids for user with access
    print("Test 2: GET /vaults/accessible returns vault_ids for user with access...")
    cleanup()
    user = create_user("has_access_user")
    vault1 = create_vault("Test Vault 1")
    vault2 = create_vault("Test Vault 2")
    add_vault_member(user["id"], vault1, "read")
    add_vault_member(user["id"], vault2, "write")

    response = client.get(
        "/api/vaults/accessible",
        headers={"Authorization": f"Bearer {user['token']}"},
    )
    if response.status_code == 200 and set(response.json()["vault_ids"]) == {
        vault1,
        vault2,
    }:
        print(f"  PASS: Returns vault_ids: [{vault1}, {vault2}]")
        passed += 1
    else:
        print(
            f"  FAIL: Expected 200 + {{vault1, vault2}}, got {response.status_code} + {response.json()}"
        )
        failed += 1
        errors.append("Test 2: /vaults/accessible with access")

    # Test 3: GET /vaults/accessible returns empty for admin (all vaults)
    print("Test 3: GET /vaults/accessible returns [] for admin (means all vaults)...")
    cleanup()
    user = create_user("admin_user", role="admin")
    create_vault("Admin Vault")

    response = client.get(
        "/api/vaults/accessible",
        headers={"Authorization": f"Bearer {user['token']}"},
    )
    if response.status_code == 200 and response.json()["vault_ids"] == []:
        print("  PASS: Admin returns vault_ids: []")
        passed += 1
    else:
        print(
            f"  FAIL: Expected 200 + [], got {response.status_code} + {response.json()}"
        )
        failed += 1
        errors.append("Test 3: /vaults/accessible admin")

    # Test 4: chat/stream with unauthorized vault returns 403
    print("Test 4: chat/stream with unauthorized vault returns 403...")
    cleanup()
    user = create_user("restricted_user")
    unauthorized_vault = create_vault("Unauthorized Vault")

    response = client.post(
        "/api/chat/stream",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "vault_id": unauthorized_vault,
        },
        headers={"Authorization": f"Bearer {user['token']}"},
    )
    if response.status_code == 403:
        print("  PASS: Returns 403 for unauthorized vault")
        passed += 1
    else:
        print(
            f"  FAIL: Expected 403, got {response.status_code} + {response.json() if response.status_code != 200 else 'OK'}"
        )
        failed += 1
        errors.append("Test 4: chat/stream unauthorized vault")

    # Test 5: chat/stream with authorized vault succeeds
    print("Test 5: chat/stream with authorized vault succeeds...")
    cleanup()
    user = create_user("authorized_user")
    authorized_vault = create_vault("Authorized Vault")
    add_vault_member(user["id"], authorized_vault, "read")

    response = client.post(
        "/api/chat/stream",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "vault_id": authorized_vault,
        },
        headers={"Authorization": f"Bearer {user['token']}"},
    )
    if response.status_code != 403:
        print(f"  PASS: Returns {response.status_code} (not 403)")
        passed += 1
    else:
        print(f"  FAIL: Expected non-403, got 403")
        failed += 1
        errors.append("Test 5: chat/stream authorized vault")

    # Test 6: Admin can access any vault
    print("Test 6: Admin can access any vault without explicit membership...")
    cleanup()
    admin = create_user("admin_user2", role="admin")
    any_vault = create_vault("Any Vault")

    response = client.post(
        "/api/chat/stream",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "vault_id": any_vault,
        },
        headers={"Authorization": f"Bearer {admin['token']}"},
    )
    if response.status_code != 403:
        print(f"  PASS: Admin returns {response.status_code} (not 403)")
        passed += 1
    else:
        print(f"  FAIL: Admin got 403")
        failed += 1
        errors.append("Test 6: admin vault access")

    # Test 7: Superadmin can access any vault
    print("Test 7: Superadmin can access any vault without explicit membership...")
    cleanup()
    superadmin = create_user("superadmin_user", role="superadmin")
    any_vault = create_vault("Any Vault")

    response = client.post(
        "/api/chat/stream",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "vault_id": any_vault,
        },
        headers={"Authorization": f"Bearer {superadmin['token']}"},
    )
    if response.status_code != 403:
        print(f"  PASS: Superadmin returns {response.status_code} (not 403)")
        passed += 1
    else:
        print(f"  FAIL: Superadmin got 403")
        failed += 1
        errors.append("Test 7: superadmin vault access")

    # Test 8: Regular user cannot access other user's vault
    print("Test 8: Regular user cannot access another user's vault...")
    cleanup()
    user1 = create_user("user1")
    user2 = create_user("user2")
    user2_vault = create_vault("User2's Vault")
    add_vault_member(user2["id"], user2_vault, "read")

    response = client.post(
        "/api/chat/stream",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "vault_id": user2_vault,
        },
        headers={"Authorization": f"Bearer {user1['token']}"},
    )
    if response.status_code == 403:
        print("  PASS: Returns 403 for other user's vault")
        passed += 1
    else:
        print(f"  FAIL: Expected 403, got {response.status_code}")
        failed += 1
        errors.append("Test 8: user cannot access other user's vault")

    pool.close_all()
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    print(f"\n{'=' * 50}")
    print(f"API Test Results: {passed} passed, {failed} failed")
    if errors:
        print(f"Failed tests: {errors}")
    return passed, failed


if __name__ == "__main__":
    p, f = run_api_tests()
    sys.exit(0 if f == 0 else 1)
