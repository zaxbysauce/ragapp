"""Tests for Task 1.8 - score_type field in done event.

Verification Points:
1. When reranking is applied, score_type = "rerank_score"
2. When reranking is not applied but hybrid search is enabled, score_type = "hybrid_rrf"
3. When neither reranking nor hybrid search, score_type = "dense_distance"
"""

import os
import sys
import unittest
from typing import Dict, List, Optional, cast

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
    _unstructured.partition = types.ModuleType('unstructured.partition')
    _unstructured.partition.auto = types.ModuleType('unstructured.partition.auto')
    _unstructured.partition.auto.partition = lambda *args, **kwargs: []
    _unstructured.chunking = types.ModuleType('unstructured.chunking')
    _unstructured.chunking.title = types.ModuleType('unstructured.chunking.title')
    _unstructured.chunking.title.chunk_by_title = lambda *args, **kwargs: []
    _unstructured.documents = types.ModuleType('unstructured.documents')
    _unstructured.documents.elements = types.ModuleType('unstructured.documents.elements')
    _unstructured.documents.elements.Element = type('Element', (), {})
    sys.modules['unstructured'] = _unstructured
    sys.modules['unstructured.partition'] = _unstructured.partition
    sys.modules['unstructured.partition.auto'] = _unstructured.partition.auto
    sys.modules['unstructured.chunking'] = _unstructured.chunking
    sys.modules['unstructured.chunking.title'] = _unstructured.chunking.title
    sys.modules['unstructured.documents'] = _unstructured.documents
    sys.modules['unstructured.documents.elements'] = _unstructured.documents.elements

from app.services.embeddings import EmbeddingService
from app.services.llm_client import LLMClient
from app.services.memory_store import MemoryRecord, MemoryStore
from app.services.rag_engine import RAGEngine
from app.services.vector_store import VectorStore


class FakeEmbeddingService:
    def __init__(self, embedding: List[float]):
        self.embedding = embedding

    async def embed_single(self, text: str) -> List[float]:
        return self.embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embedding for _ in texts]


class FakeVectorStore:
    def __init__(self, results: List[Dict]):
        self._results = results
        self._is_connected = True

    def search(
        self,
        embedding: List[float],
        limit: int = 10,
        filter_expr=None,
        vault_id=None,
        query_text: str = "",
        hybrid: bool = False,
        hybrid_alpha: float = 0.5,
        query_sparse=None,
    ):
        return self._results[:limit]

    def get_chunks_by_uid(self, chunk_uids: List[str]):
        return []


class FakeMemoryStore:
    def __init__(self, memories: Optional[List[MemoryRecord]] = None):
        self._memories = memories or []

    def detect_memory_intent(self, text: str):
        return None

    def add_memory(self, content: str, category=None, tags=None, source=None, vault_id=None):
        return MemoryRecord(id=1, content=content, category=category, tags=tags, source=source, created_at=None, updated_at=None)

    def search_memories(self, query: str, limit: int = 5, vault_id=None):
        return self._memories[:limit]


class FakeLLMClient:
    def __init__(self, response: str = "test response"):
        self._response = response

    async def chat_completion(self, messages, **kwargs):
        return self._response

    async def chat_completion_stream(self, messages, **kwargs):
        yield "chunk"


class FakeRerankingService:
    """Fake reranking service that simulates reranking."""
    
    def __init__(self, reranked_results: Optional[List[Dict]] = None):
        self._reranked_results = reranked_results or []
        self.was_called = False
    
    async def rerank(self, query: str, chunks: List[Dict], top_n: int = 10):
        self.was_called = True
        return self._reranked_results if self._reranked_results else chunks[:top_n]


class ScoreTypeTests(unittest.IsolatedAsyncioTestCase):
    """Test suite for score_type field in done event."""

    async def test_reranking_applied_score_type_is_rerank_score(self):
        """Verification Point 1: When reranking is applied, score_type = 'rerank_score'"""
        # Setup: reranking enabled, hybrid disabled
        vector_results = [
            {"text": "chunk 1", "file_id": "file1", "metadata": {"source_file": "doc.md"}, "_distance": 0.3},
            {"text": "chunk 2", "file_id": "file2", "metadata": {"source_file": "doc2.md"}, "_distance": 0.4},
        ]
        
        # Fake reranking service that will be called
        reranked_results = [
            {"text": "chunk 1", "file_id": "file1", "metadata": {"source_file": "doc.md"}, "_distance": 0.3},
        ]
        
        engine = RAGEngine()
        engine.embedding_service = cast(EmbeddingService, FakeEmbeddingService([0.1, 0.2]))
        engine.vector_store = cast(VectorStore, FakeVectorStore(vector_results))
        engine.memory_store = cast(MemoryStore, FakeMemoryStore())
        engine.llm_client = cast(LLMClient, FakeLLMClient(response="answer"))
        engine.reranking_service = FakeRerankingService(reranked_results)
        
        # Enable reranking, disable hybrid search
        engine.reranking_enabled = True
        engine.hybrid_search_enabled = False
        
        results = [msg async for msg in engine.query("test query", [], stream=False)]
        
        # Find the done event
        done_events = [r for r in results if r.get("type") == "done"]
        self.assertEqual(1, len(done_events), "Should have exactly one done event")
        
        done = done_events[0]
        self.assertIn("score_type", done, "Done event must include score_type field")
        
        # Verify score_type is "rerank_score" when reranking is applied
        self.assertEqual(
            "rerank_score",
            done["score_type"],
            "When reranking is applied, score_type should be 'rerank_score'"
        )
        
        # Verify reranking service was actually called
        self.assertTrue(
            engine.reranking_service.was_called,
            "Reranking service should have been called"
        )

    async def test_hybrid_enabled_no_reranking_score_type_is_hybrid_rrf(self):
        """Verification Point 2: When reranking is not applied but hybrid search is enabled, score_type = 'hybrid_rrf'"""
        # Setup: hybrid enabled, reranking disabled
        vector_results = [
            {"text": "chunk 1", "file_id": "file1", "metadata": {"source_file": "doc.md"}, "_distance": 0.3},
            {"text": "chunk 2", "file_id": "file2", "metadata": {"source_file": "doc2.md"}, "_distance": 0.4},
        ]
        
        engine = RAGEngine()
        engine.embedding_service = cast(EmbeddingService, FakeEmbeddingService([0.1, 0.2]))
        engine.vector_store = cast(VectorStore, FakeVectorStore(vector_results))
        engine.memory_store = cast(MemoryStore, FakeMemoryStore())
        engine.llm_client = cast(LLMClient, FakeLLMClient(response="answer"))
        
        # Disable reranking, enable hybrid search
        engine.reranking_enabled = False
        engine.hybrid_search_enabled = True
        
        results = [msg async for msg in engine.query("test query", [], stream=False)]
        
        # Find the done event
        done_events = [r for r in results if r.get("type") == "done"]
        self.assertEqual(1, len(done_events), "Should have exactly one done event")
        
        done = done_events[0]
        self.assertIn("score_type", done, "Done event must include score_type field")
        
        # Verify score_type is "hybrid_rrf" when hybrid is enabled but no reranking
        self.assertEqual(
            "hybrid_rrf",
            done["score_type"],
            "When hybrid search is enabled without reranking, score_type should be 'hybrid_rrf'"
        )

    async def test_neither_reranking_nor_hybrid_score_type_is_dense_distance(self):
        """Verification Point 3: When neither reranking nor hybrid search, score_type = 'dense_distance'"""
        # Setup: both reranking and hybrid disabled
        vector_results = [
            {"text": "chunk 1", "file_id": "file1", "metadata": {"source_file": "doc.md"}, "_distance": 0.3},
            {"text": "chunk 2", "file_id": "file2", "metadata": {"source_file": "doc2.md"}, "_distance": 0.4},
        ]
        
        engine = RAGEngine()
        engine.embedding_service = cast(EmbeddingService, FakeEmbeddingService([0.1, 0.2]))
        engine.vector_store = cast(VectorStore, FakeVectorStore(vector_results))
        engine.memory_store = cast(MemoryStore, FakeMemoryStore())
        engine.llm_client = cast(LLMClient, FakeLLMClient(response="answer"))
        
        # Disable both reranking and hybrid search
        engine.reranking_enabled = False
        engine.hybrid_search_enabled = False
        
        results = [msg async for msg in engine.query("test query", [], stream=False)]
        
        # Find the done event
        done_events = [r for r in results if r.get("type") == "done"]
        self.assertEqual(1, len(done_events), "Should have exactly one done event")
        
        done = done_events[0]
        self.assertIn("score_type", done, "Done event must include score_type field")
        
        # Verify score_type is "dense_distance" when neither reranking nor hybrid
        self.assertEqual(
            "dense_distance",
            done["score_type"],
            "When neither reranking nor hybrid search, score_type should be 'dense_distance'"
        )

    async def test_score_type_field_present_in_done_event(self):
        """Sanity check: Ensure score_type field is always present in done event."""
        vector_results = [
            {"text": "chunk 1", "file_id": "file1", "metadata": {"source_file": "doc.md"}, "_distance": 0.3},
        ]
        
        engine = RAGEngine()
        engine.embedding_service = cast(EmbeddingService, FakeEmbeddingService([0.1, 0.2]))
        engine.vector_store = cast(VectorStore, FakeVectorStore(vector_results))
        engine.memory_store = cast(MemoryStore, FakeMemoryStore())
        engine.llm_client = cast(LLMClient, FakeLLMClient(response="answer"))
        
        # Test all combinations
        for reranking_enabled in [True, False]:
            for hybrid_enabled in [True, False]:
                engine.reranking_enabled = reranking_enabled
                engine.hybrid_search_enabled = hybrid_enabled
                
                results = [msg async for msg in engine.query("test query", [], stream=False)]
                done_events = [r for r in results if r.get("type") == "done"]
                
                self.assertEqual(1, len(done_events))
                done = done_events[0]
                
                # Score type must be present and be one of the valid values
                self.assertIn("score_type", done)
                self.assertIn(
                    done["score_type"],
                    ["rerank_score", "hybrid_rrf", "dense_distance"],
                    f"score_type must be one of the valid values, got: {done['score_type']}"
                )


if __name__ == "__main__":
    unittest.main()
