"""
Backend API route tests using unittest and FastAPI TestClient.

Tests cover health, settings, memories, documents, and chat endpoints.
All external services (LLM, vector store) are mocked for deterministic tests.
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Stub missing optional dependencies
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

from fastapi.testclient import TestClient


# Create a temporary database for testing
TEST_DB_PATH = None
TEST_DATA_DIR = None


def setup_test_db():
    """Set up a temporary test database."""
    global TEST_DB_PATH, TEST_DATA_DIR
    TEST_DATA_DIR = tempfile.mkdtemp()
    TEST_DB_PATH = Path(TEST_DATA_DIR) / "test.db"
    
    # Import and initialize the database
    from app.models.database import init_db
    init_db(str(TEST_DB_PATH))
    return str(TEST_DB_PATH)


def get_test_settings():
    """Get test settings with temporary database path."""
    from app.config import Settings
    
    settings = Settings()
    settings.data_dir = Path(TEST_DATA_DIR)
    return settings


# Set up test database before importing app
setup_test_db()

from app.main import app
from app.config import settings


class TestHealthEndpoint(unittest.TestCase):
    """Tests for the /api/health endpoint."""
    
    def setUp(self):
        self.client = TestClient(app)
    
    @patch("app.api.routes.health.LLMHealthChecker")
    @patch("app.api.routes.health.ModelChecker")
    def test_health_check_success(self, mock_model_checker_class, mock_llm_checker_class):
        """Test health check returns ok status with mocked services."""
        # Mock LLMHealthChecker
        mock_llm_checker = MagicMock()
        mock_llm_checker.check_all = AsyncMock(return_value={
            "ok": True,
            "embeddings": {"ok": True, "error": None},
            "chat": {"ok": True, "error": None},
            "error": None
        })
        mock_llm_checker_class.return_value = mock_llm_checker

        # Mock ModelChecker
        mock_model_checker = MagicMock()
        mock_model_checker.check_models = AsyncMock(return_value={
            "embedding_model": {"available": True, "error": None},
            "chat_model": {"available": True, "error": None}
        })
        mock_model_checker_class.return_value = mock_model_checker

        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("llm", data)
        self.assertIn("models", data)
        self.assertTrue(data["llm"]["ok"])

    @patch("app.api.routes.health.LLMHealthChecker")
    @patch("app.api.routes.health.ModelChecker")
    def test_health_check_llm_failure(self, mock_model_checker_class, mock_llm_checker_class):
        """Test health check handles LLM service failure."""
        mock_llm_checker = MagicMock()
        mock_llm_checker.check_all = AsyncMock(return_value={
            "ok": False,
            "embeddings": {"ok": False, "error": "Embedding service error"},
            "chat": {"ok": True, "error": None},
            "error": "Embedding service error"
        })
        mock_llm_checker_class.return_value = mock_llm_checker

        mock_model_checker = MagicMock()
        mock_model_checker.check_models = AsyncMock(return_value={
            "embedding_model": {"available": True, "error": None},
            "chat_model": {"available": True, "error": None}
        })
        mock_model_checker_class.return_value = mock_model_checker

        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertFalse(data["llm"]["ok"])


class TestSettingsEndpoints(unittest.TestCase):
    """Tests for the /api/settings GET and POST endpoints."""
    
    def setUp(self):
        self.client = TestClient(app)
        # Store original values
        self._original_chunk_size = settings.chunk_size
        self._original_rag_threshold = settings.rag_relevance_threshold
    
    def tearDown(self):
        # Restore original values
        settings.chunk_size = self._original_chunk_size
        settings.rag_relevance_threshold = self._original_rag_threshold
    
    def test_get_settings(self):
        """Test GET /api/settings returns current settings."""
        response = self.client.get("/api/settings")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("chunk_size", data)
        self.assertIn("rag_relevance_threshold", data)
    
    def test_post_settings_valid(self):
        """Test POST /api/settings with valid settings updates values."""
        payload = {
            "chunk_size": 1024,
            "rag_relevance_threshold": 0.5
        }
        
        response = self.client.post("/api/settings", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["chunk_size"], 1024)
        self.assertEqual(data["rag_relevance_threshold"], 0.5)
    
    def test_post_settings_invalid_chunk_size(self):
        """Test POST /api/settings with invalid chunk_size returns 422."""
        payload = {"chunk_size": 0}
        
        response = self.client.post("/api/settings", json=payload)
        
        self.assertEqual(response.status_code, 422)
    
    def test_post_settings_invalid_rag_threshold_low(self):
        """Test POST /api/settings with rag_relevance_threshold < 0 returns 422."""
        payload = {"rag_relevance_threshold": -0.1}
        
        response = self.client.post("/api/settings", json=payload)
        
        self.assertEqual(response.status_code, 422)
    
    def test_post_settings_invalid_rag_threshold_high(self):
        """Test POST /api/settings with rag_relevance_threshold > 1 returns 422."""
        payload = {"rag_relevance_threshold": 1.5}
        
        response = self.client.post("/api/settings", json=payload)
        
        self.assertEqual(response.status_code, 422)
    
    def test_post_settings_no_valid_fields(self):
        """Test POST /api/settings with no valid fields returns 400."""
        payload = {}
        
        response = self.client.post("/api/settings", json=payload)
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("No valid fields provided", data["detail"])


class TestMemoriesEndpoints(unittest.TestCase):
    """Tests for the /api/memories CRUD endpoints."""
    
    def setUp(self):
        self.client = TestClient(app)
        # Create temp directory and set settings.data_dir so sqlite_path resolves correctly
        self._temp_dir = tempfile.mkdtemp()
        self._original_data_dir = settings.data_dir
        settings.data_dir = Path(self._temp_dir)
        # Ensure temp db directory exists
        Path(self._temp_dir).mkdir(parents=True, exist_ok=True)
        # Patch settings for routes
        self.settings_patcher = patch("app.api.routes.memories.settings")
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.data_dir = Path(self._temp_dir)
        self.mock_settings.sqlite_path = Path(self._temp_dir) / "test.db"
        # Ensure MemoryStore uses the same path - patch __init__ to set sqlite_path
        from app.services.memory_store import MemoryStore
        self._original_memory_store_init = MemoryStore.__init__
        def patched_init(instance):
            instance.sqlite_path = str(Path(self._temp_dir) / "test.db")
        MemoryStore.__init__ = patched_init
        self._MemoryStore = MemoryStore
        # Clear/init memories table
        from app.models.database import init_db, get_db_connection
        db_path = str(Path(self._temp_dir) / "test.db")
        init_db(db_path)
        conn = get_db_connection(db_path)
        try:
            conn.execute("DELETE FROM memories")
            conn.commit()
        finally:
            conn.close()
    
    def tearDown(self):
        self.settings_patcher.stop()
        settings.data_dir = self._original_data_dir
        # Restore MemoryStore.__init__
        self._MemoryStore.__init__ = self._original_memory_store_init
        # Cleanup temp directory
        import shutil
        shutil.rmtree(self._temp_dir, ignore_errors=True)
    
    def test_list_memories_empty(self):
        """Test GET /api/memories returns empty list when no memories."""
        response = self.client.get("/api/memories")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["memories"], [])
    
    @patch("app.api.routes.memories.MemoryStore")
    def test_create_memory(self, mock_store_class):
        """Test POST /api/memories creates a new memory."""
        from app.services.memory_store import MemoryRecord
        
        mock_record = MemoryRecord(
            id=1,
            content="Test memory content",
            category="test",
            tags="[\"tag1\", \"tag2\"]",
            source="test_source",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00"
        )
        
        mock_store = MagicMock()
        mock_store.add_memory.return_value = mock_record
        mock_store_class.return_value = mock_store
        
        payload = {
            "content": "Test memory content",
            "category": "test",
            "tags": "[\"tag1\", \"tag2\"]",
            "source": "test_source"
        }
        
        response = self.client.post("/api/memories", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["content"], "Test memory content")
        self.assertEqual(data["category"], "test")
        self.assertEqual(data["id"], 1)
    
    @patch("app.api.routes.memories.MemoryStore")
    def test_create_memory_invalid_empty_content(self, mock_store_class):
        """Test POST /api/memories with empty content returns 422."""
        payload = {
            "content": "",
            "category": "test"
        }
        
        response = self.client.post("/api/memories", json=payload)
        
        self.assertEqual(response.status_code, 422)
    
    def test_create_and_list_memory(self):
        """Test creating a memory and then listing it."""
        # Create a memory
        create_payload = {
            "content": "Integration test memory",
            "category": "integration",
            "tags": "[\"test\"]"
        }
        
        create_response = self.client.post("/api/memories", json=create_payload)
        self.assertEqual(create_response.status_code, 200)
        created = create_response.json()
        self.assertEqual(created["content"], "Integration test memory")
        
        # List memories
        list_response = self.client.get("/api/memories")
        self.assertEqual(list_response.status_code, 200)
        data = list_response.json()
        self.assertEqual(len(data["memories"]), 1)
        self.assertEqual(data["memories"][0]["content"], "Integration test memory")
    
    def test_update_memory(self):
        """Test PUT /api/memories/{id} updates a memory."""
        # Create a memory first
        create_payload = {
            "content": "Original content",
            "category": "original"
        }
        create_response = self.client.post("/api/memories", json=create_payload)
        self.assertEqual(create_response.status_code, 200)
        memory_id = create_response.json()["id"]
        
        # Update the memory
        update_payload = {
            "content": "Updated content",
            "category": "updated"
        }
        update_response = self.client.put(f"/api/memories/{memory_id}", json=update_payload)
        
        self.assertEqual(update_response.status_code, 200)
        data = update_response.json()
        self.assertEqual(data["content"], "Updated content")
        self.assertEqual(data["category"], "updated")
    
    def test_update_memory_not_found(self):
        """Test PUT /api/memories/{id} returns 404 for non-existent memory."""
        update_payload = {"content": "Updated content"}
        
        response = self.client.put("/api/memories/99999", json=update_payload)
        
        self.assertEqual(response.status_code, 404)
    
    def test_delete_memory(self):
        """Test DELETE /api/memories/{id} deletes a memory."""
        # Create a memory first
        create_payload = {"content": "Memory to delete"}
        create_response = self.client.post("/api/memories", json=create_payload)
        self.assertEqual(create_response.status_code, 200)
        memory_id = create_response.json()["id"]
        
        # Delete the memory
        delete_response = self.client.delete(f"/api/memories/{memory_id}")
        
        self.assertEqual(delete_response.status_code, 200)
        data = delete_response.json()
        self.assertIn("deleted successfully", data["message"])
        
        # Verify it's gone
        list_response = self.client.get("/api/memories")
        self.assertEqual(len(list_response.json()["memories"]), 0)
    
    def test_delete_memory_not_found(self):
        """Test DELETE /api/memories/{id} returns 404 for non-existent memory."""
        response = self.client.delete("/api/memories/99999")
        
        self.assertEqual(response.status_code, 404)


class TestDocumentsEndpoints(unittest.TestCase):
    """Tests for the /api/documents endpoints."""
    
    def setUp(self):
        self.client = TestClient(app)
        # Patch settings for this test
        self.settings_patcher = patch("app.api.routes.documents.settings")
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.sqlite_path = TEST_DB_PATH
        self.mock_settings.data_dir = Path(TEST_DATA_DIR)
        # Clear files table before each test
        from app.models.database import get_db_connection
        conn = get_db_connection(str(TEST_DB_PATH))
        try:
            conn.execute("DELETE FROM files")
            conn.commit()
        finally:
            conn.close()
    
    def tearDown(self):
        self.settings_patcher.stop()
    
    def test_get_document_stats_empty(self):
        """Test GET /api/documents/stats returns success with zero counts."""
        response = self.client.get("/api/documents/stats")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_files"], 0)
        self.assertEqual(data["total_chunks"], 0)
        self.assertEqual(data["status"], "success")
    
    def test_list_documents_empty(self):
        """Test GET /api/documents returns empty list when no documents."""
        response = self.client.get("/api/documents/")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["documents"], [])


class TestChatEndpoint(unittest.TestCase):
    """Tests for the /api/chat endpoint with mocked RAGEngine."""
    
    def setUp(self):
        self.client = TestClient(app)
    
    @patch("app.api.routes.chat.RAGEngine")
    def test_chat_non_streaming(self, mock_rag_engine_class):
        """Test POST /api/chat non-streaming returns content and sources."""

        async def mock_query(user_input, chat_history, stream=False):
            """Mock async generator for RAG query."""
            yield {"type": "content", "content": "This is a test response."}
            yield {
                "type": "done",
                "sources": [
                    {"file_id": "1", "score": 0.95, "metadata": {"source_file": "test.txt"}},
                    {"file_id": "2", "score": 0.85, "metadata": {"source_file": "test2.txt"}}
                ],
                "memories_used": ["Memory 1"]
            }

        mock_rag_engine = MagicMock()
        mock_rag_engine.query = mock_query
        mock_rag_engine_class.return_value = mock_rag_engine

        payload = {
            "message": "What is the test?",
            "history": [],
            "stream": False
        }

        response = self.client.post("/api/chat", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["content"], "This is a test response.")
        self.assertEqual(len(data["sources"]), 2)
        self.assertEqual(data["sources"][0]["file_id"], "1")
        self.assertEqual(len(data["memories_used"]), 1)
    
    @patch("app.api.routes.chat.RAGEngine")
    def test_chat_non_streaming_with_history(self, mock_rag_engine_class):
        """Test POST /api/chat with chat history."""
        
        async def mock_query(user_input, chat_history, stream=False):
            """Mock async generator that verifies history is passed."""
            # Verify history is passed
            self.assertEqual(len(chat_history), 1)
            self.assertEqual(chat_history[0]["role"], "user")
            self.assertEqual(chat_history[0]["content"], "Previous message")
            
            yield {"type": "content", "content": "Response with history."}
            yield {
                "type": "done",
                "sources": [],
                "memories_used": []
            }
        
        mock_rag_engine = MagicMock()
        mock_rag_engine.query = mock_query
        mock_rag_engine_class.return_value = mock_rag_engine
        
        payload = {
            "message": "Follow-up question",
            "history": [{"role": "user", "content": "Previous message"}],
            "stream": False
        }
        
        response = self.client.post("/api/chat", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["content"], "Response with history.")
    
    @patch("app.api.routes.chat.RAGEngine")
    def test_chat_empty_sources(self, mock_rag_engine_class):
        """Test POST /api/chat handles empty sources gracefully."""
        
        async def mock_query(user_input, chat_history, stream=False):
            yield {"type": "content", "content": "No relevant sources found."}
            yield {
                "type": "done",
                "sources": [],
                "memories_used": []
            }
        
        mock_rag_engine = MagicMock()
        mock_rag_engine.query = mock_query
        mock_rag_engine_class.return_value = mock_rag_engine
        
        payload = {
            "message": "Unknown query",
            "history": [],
            "stream": False
        }
        
        response = self.client.post("/api/chat", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["content"], "No relevant sources found.")
        self.assertEqual(data["sources"], [])
        self.assertEqual(data["memories_used"], [])


class TestBasicHealthEndpoint(unittest.TestCase):
    """Tests for the basic /health endpoint."""
    
    def setUp(self):
        self.client = TestClient(app)
    
    def test_basic_health_check(self):
        """Test GET /health returns ok status."""
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")


if __name__ == "__main__":
    unittest.main()
