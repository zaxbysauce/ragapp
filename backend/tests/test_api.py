"""
Basic API integration tests using unittest and FastAPI TestClient.
"""
import sys
import os

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

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app


class TestAPI(unittest.TestCase):
    """Test suite for API endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_get_health_returns_status_ok(self):
        """Test GET /health returns status ok."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")

    @patch("app.api.routes.health.LLMHealthChecker")
    @patch("app.api.routes.health.ModelChecker")
    def test_get_api_health_returns_status_ok_with_llm_models(
        self, mock_model_checker, mock_llm_checker
    ):
        """Test GET /api/health returns status ok with llm and models data."""
        # Configure mocks to return deterministic values matching actual service structure
        mock_llm_instance = mock_llm_checker.return_value
        mock_llm_instance.check_all = AsyncMock(return_value={
            "ok": True,
            "embeddings": {"ok": True, "error": None},
            "chat": {"ok": True, "error": None},
            "error": None
        })

        mock_model_instance = mock_model_checker.return_value
        mock_model_instance.check_models = AsyncMock(return_value={
            "embedding_model": {"available": True, "error": None},
            "chat_model": {"available": True, "error": None}
        })

        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("llm", data)
        self.assertIn("models", data)
        self.assertEqual(data["llm"]["ok"], True)
        self.assertEqual(data["llm"]["embeddings"]["ok"], True)
        self.assertEqual(data["llm"]["chat"]["ok"], True)
        self.assertEqual(data["models"]["embedding_model"]["available"], True)
        self.assertEqual(data["models"]["chat_model"]["available"], True)

    def test_get_api_settings_returns_expected_keys(self):
        """Test GET /api/settings returns expected configuration keys."""
        response = self.client.get("/api/settings")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        expected_keys = [
            "chunk_size",
            "chunk_overlap",
            "max_context_chunks",
            "rag_relevance_threshold"
        ]
        for key in expected_keys:
            self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
