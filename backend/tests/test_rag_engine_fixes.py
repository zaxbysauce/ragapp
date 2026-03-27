"""Integration tests for RAG engine Phase 1 enhancements.

Tests for:
- EmbeddingSemanticChunker with cosine similarity breakpoints
- Token-aware context packing
- Inline citation grounding
- Integration with RAG pipeline
"""

import os
import sys
import asyncio
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
from app.services.rag_engine import RAGEngine, RAGSource
from app.services.vector_store import VectorStore
from app.services.chunking import (
    SemanticChunker,
    EmbeddingSemanticChunker,
    ThresholdType,
    ProcessedChunk,
)


class FakeEmbeddingService:
    """Fake embedding service for testing."""
    
    def __init__(self, embedding: List[float], embed_dim: int = 768):
        self.embedding = embedding
        self.embed_dim = embed_dim

    async def embed_single(self, text: str) -> List[float]:
        # Return deterministic embeddings based on text content
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Generate a deterministic vector
        vec = [float((hash_val + i) % 1000) / 1000.0 for i in range(self.embed_dim)]
        # Normalize
        import math
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed_single(text) for text in texts]


class FakeVectorStore:
    """Fake vector store for testing."""
    
    def __init__(self, results: List[Dict]):
        self._results = results

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
    """Fake memory store for testing."""
    
    def __init__(self, intent: Optional[str] = None, memories: Optional[List[MemoryRecord]] = None):
        self.intent = intent
        self._memories = memories or []
        self.added: List[str] = []

    def detect_memory_intent(self, text: str):
        return self.intent

    def add_memory(self, content: str, category=None, tags=None, source=None, vault_id=None):
        self.added.append(content)
        return MemoryRecord(id=1, content=content, category=category, tags=tags, source=source, created_at=None, updated_at=None)

    def search_memories(self, query: str, limit: int = 5, vault_id=None):
        return self._memories[:limit]


class FakeLLMClient:
    """Fake LLM client for testing."""
    
    def __init__(self, response: str, stream_chunks: Optional[List[str]] = None):
        self._response = response
        self._stream_chunks = stream_chunks or []

    async def chat_completion(self, messages):
        return self._response

    async def chat_completion_stream(self, messages):
        for chunk in self._stream_chunks:
            yield chunk


class TestEmbeddingSemanticChunker(unittest.IsolatedAsyncioTestCase):
    """Tests for EmbeddingSemanticChunker."""
    
    async def asyncSetUp(self):
        self.embedding_service = FakeEmbeddingService([0.1] * 768)
    
    async def test_chunker_initialization(self):
        """Test that chunker initializes with correct parameters."""
        chunker = EmbeddingSemanticChunker(
            embedding_service=self.embedding_service,
            threshold_type=ThresholdType.PERCENTILE,
            threshold_value=50.0,
            min_chunk_size=100,
            max_chunk_size=1000,
        )
        self.assertEqual(chunker.threshold_type, ThresholdType.PERCENTILE)
        self.assertEqual(chunker.threshold_value, 50.0)
        self.assertEqual(chunker.min_chunk_size, 100)
        self.assertEqual(chunker.max_chunk_size, 1000)
    
    async def test_chunker_with_percentile_threshold(self):
        """Test chunking with percentile threshold type."""
        chunker = EmbeddingSemanticChunker(
            embedding_service=self.embedding_service,
            threshold_type=ThresholdType.PERCENTILE,
            threshold_value=50.0,
            min_chunk_size=50,
            max_chunk_size=2000,
        )
        
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        chunks = await chunker.chunk_text(text)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, ProcessedChunk)
            self.assertGreater(len(chunk.text), 0)
    
    async def test_chunker_with_stddev_threshold(self):
        """Test chunking with stddev threshold type."""
        chunker = EmbeddingSemanticChunker(
            embedding_service=self.embedding_service,
            threshold_type=ThresholdType.STDDEV,
            threshold_value=1.0,
            min_chunk_size=50,
            max_chunk_size=2000,
        )
        
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = await chunker.chunk_text(text)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
    
    async def test_chunker_with_gradient_threshold(self):
        """Test chunking with gradient threshold type."""
        chunker = EmbeddingSemanticChunker(
            embedding_service=self.embedding_service,
            threshold_type=ThresholdType.GRADIENT,
            threshold_value=0.3,
            min_chunk_size=50,
            max_chunk_size=2000,
        )
        
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = await chunker.chunk_text(text)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
    
    async def test_chunker_fallback_on_embedding_error(self):
        """Test that chunker falls back to title-based on embedding errors."""
        
        class FailingEmbeddingService:
            async def embed_single(self, text: str):
                raise RuntimeError("Embedding service failed")
        
        chunker = EmbeddingSemanticChunker(
            embedding_service=FailingEmbeddingService(),
            threshold_type=ThresholdType.PERCENTILE,
            threshold_value=50.0,
        )
        
        text = "This is a test sentence. This is another sentence."
        chunks = await chunker.chunk_text(text)
        
        # Should still return chunks via fallback
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
    
    async def test_chunker_respects_min_max_size(self):
        """Test that chunker respects min and max chunk sizes."""
        chunker = EmbeddingSemanticChunker(
            embedding_service=self.embedding_service,
            threshold_type=ThresholdType.PERCENTILE,
            threshold_value=50.0,
            min_chunk_size=50,
            max_chunk_size=200,
        )
        
        # Long text that should be split
        text = " ".join([f"This is sentence number {i} with enough words to make it substantial." for i in range(20)])
        chunks = await chunker.chunk_text(text)
        
        for chunk in chunks:
            self.assertGreaterEqual(len(chunk.text), 50)
            self.assertLessEqual(len(chunk.text), 200 * 2)  # Allow some flexibility due to sentence boundaries
    
    async def test_chunker_single_sentence(self):
        """Test chunker with single sentence."""
        chunker = EmbeddingSemanticChunker(
            embedding_service=self.embedding_service,
            threshold_type=ThresholdType.PERCENTILE,
        )
        
        text = "Only one sentence."
        chunks = await chunker.chunk_text(text)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)
    
    async def test_chunker_empty_text(self):
        """Test chunker with empty text."""
        chunker = EmbeddingSemanticChunker(
            embedding_service=self.embedding_service,
            threshold_type=ThresholdType.PERCENTILE,
        )
        
        chunks = await chunker.chunk_text("")
        self.assertEqual(len(chunks), 0)
    
    async def test_chunker_with_section_title(self):
        """Test chunker with section title metadata."""
        chunker = EmbeddingSemanticChunker(
            embedding_service=self.embedding_service,
            threshold_type=ThresholdType.PERCENTILE,
        )
        
        text = "First sentence. Second sentence."
        chunks = await chunker.chunk_text(text, section_title="Test Section")
        
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertEqual(chunk.metadata.get("section_title"), "Test Section")


class TestRAGEngineTokenPacking(unittest.TestCase):
    """Tests for token-aware context packing in RAGEngine."""
    
    def test_pack_context_by_token_budget_basic(self):
        """Test basic token budget packing."""
        engine = RAGEngine()
        
        chunks = [
            RAGSource(text="A" * 100, file_id="f1", score=0.9, metadata={}),  # ~25 tokens
            RAGSource(text="B" * 100, file_id="f2", score=0.8, metadata={}),  # ~25 tokens
            RAGSource(text="C" * 100, file_id="f3", score=0.7, metadata={}),  # ~25 tokens
        ]
        
        # Pack with 60 token budget (should fit 2 chunks)
        packed = engine._pack_context_by_token_budget(chunks, max_tokens=60)
        
        self.assertEqual(len(packed), 2)
        self.assertEqual(packed[0].file_id, "f1")
        self.assertEqual(packed[1].file_id, "f2")
    
    def test_pack_context_by_token_budget_exact_fit(self):
        """Test packing when chunks fit exactly."""
        engine = RAGEngine()
        
        chunks = [
            RAGSource(text="A" * 40, file_id="f1", score=0.9, metadata={}),  # ~10 tokens
            RAGSource(text="B" * 40, file_id="f2", score=0.8, metadata={}),  # ~10 tokens
        ]
        
        # Pack with 20 token budget (should fit both)
        packed = engine._pack_context_by_token_budget(chunks, max_tokens=20)
        
        self.assertEqual(len(packed), 2)
    
    def test_pack_context_by_token_budget_empty(self):
        """Test packing with empty chunks list."""
        engine = RAGEngine()
        
        packed = engine._pack_context_by_token_budget([], max_tokens=100)
        self.assertEqual(len(packed), 0)
    
    def test_pack_context_by_token_budget_always_includes_first(self):
        """Test that at least first chunk is always included if possible."""
        engine = RAGEngine()
        
        # First chunk alone exceeds budget
        chunks = [
            RAGSource(text="A" * 1000, file_id="f1", score=0.9, metadata={}),  # ~250 tokens
        ]
        
        # Pack with 100 token budget - first chunk still included
        packed = engine._pack_context_by_token_budget(chunks, max_tokens=100)
        
        # First chunk is included even if it exceeds budget (since packed is empty)
        self.assertEqual(len(packed), 1)
    
    def test_pack_context_respects_order(self):
        """Test that packing respects input order."""
        engine = RAGEngine()
        
        chunks = [
            RAGSource(text="A" * 100, file_id="f1", score=0.5, metadata={}),
            RAGSource(text="B" * 100, file_id="f2", score=0.9, metadata={}),
            RAGSource(text="C" * 100, file_id="f3", score=0.8, metadata={}),
        ]
        
        packed = engine._pack_context_by_token_budget(chunks, max_tokens=75)
        
        # Should be in original order
        self.assertEqual(packed[0].file_id, "f1")
        self.assertEqual(packed[1].file_id, "f2")


class TestRAGEngineCitationGrounding(unittest.TestCase):
    """Tests for inline citation grounding in RAGEngine."""
    
    def test_system_prompt_contains_citation_instruction(self):
        """Test that system prompt includes CITATION_INSTRUCTION."""
        engine = RAGEngine()
        prompt = engine._build_system_prompt()
        
        self.assertIn("cite sources", prompt.lower())
        self.assertIn("[Source:", prompt)
    
    def test_citation_instruction_content(self):
        """Test that citation instruction has required content."""
        engine = RAGEngine()
        prompt = engine._build_system_prompt()
        
        # Check for key elements of citation instruction
        self.assertIn("cite sources inline", prompt.lower())
        self.assertIn("every substantive claim", prompt.lower())
        self.assertIn("cannot find a source", prompt.lower())


class TestRAGEngineIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for RAGEngine enhancements."""
    
    async def asyncSetUp(self):
        self.embedding_service = FakeEmbeddingService([0.1] * 768)
        self.vector_results = [
            {"text": "chunk one", "file_id": "file1", "metadata": {"source_file": "doc.md"}, "score": 0.9},
            {"text": "chunk two", "file_id": "file2", "metadata": {"source_file": "doc2.md"}, "score": 0.8},
            {"text": "chunk three", "file_id": "file3", "metadata": {"source_file": "doc3.md"}, "score": 0.7},
        ]
    
    async def test_rag_engine_uses_token_packing(self):
        """Test that RAG engine can use token packing."""
        engine = RAGEngine()
        engine.embedding_service = cast(EmbeddingService, self.embedding_service)
        engine.vector_store = cast(VectorStore, FakeVectorStore(self.vector_results))
        engine.memory_store = cast(MemoryStore, FakeMemoryStore())
        engine.llm_client = cast(LLMClient, FakeLLMClient(response="answer"))
        
        # Create RAGSource chunks
        chunks = [
            RAGSource(text="A" * 1000, file_id="f1", score=0.9, metadata={}),
            RAGSource(text="B" * 1000, file_id="f2", score=0.8, metadata={}),
            RAGSource(text="C" * 1000, file_id="f3", score=0.7, metadata={}),
        ]
        
        # Test token packing
        packed = engine._pack_context_by_token_budget(chunks, max_tokens=600)
        self.assertGreater(len(packed), 0)
        self.assertLessEqual(len(packed), len(chunks))
    
    async def test_rag_engine_citation_in_prompt(self):
        """Test that RAG engine includes citation in system prompt."""
        engine = RAGEngine()
        
        messages = engine._build_messages(
            user_input="test query",
            chat_history=[],
            chunks=[],
            memories=[],
        )
        
        system_message = messages[0]
        self.assertEqual(system_message["role"], "system")
        self.assertIn("cite", system_message["content"].lower())


class TestDefaultRetrievalProfile(unittest.TestCase):
    """Tests for default retrieval profile settings."""
    
    def test_default_retrieval_settings(self):
        """Test that default retrieval settings are correctly configured."""
        from app.config import settings
        
        # Check that all Phase 1 default settings are enabled
        self.assertTrue(settings.reranking_enabled)
        self.assertTrue(settings.query_transformation_enabled)
        self.assertTrue(settings.hyde_enabled)
        self.assertTrue(settings.retrieval_evaluation_enabled)
        self.assertTrue(settings.context_distillation_enabled)
        
        # Check numeric defaults
        self.assertEqual(settings.hybrid_alpha, 0.6)
        self.assertEqual(settings.initial_retrieval_top_k, 25)
        self.assertEqual(settings.reranker_top_n, 7)


class TestRAGASEndpoint(unittest.IsolatedAsyncioTestCase):
    """Tests for the RAGAS evaluation endpoint."""

    def setUp(self):
        """Set up test fixtures."""
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        from app.api.deps import get_embedding_service
        from app.security import require_auth

        # Stub embedding service so the endpoint never touches app.state
        class _FakeEmbeddingService:
            async def embed_single(self, text: str):
                return [0.1, 0.2, 0.3]

            async def embed_batch(self, texts):
                return [[0.1, 0.2, 0.3] for _ in texts]

        def _override_embedding():
            return _FakeEmbeddingService()

        def _override_auth(authorization: str = None):
            return {"role": "admin", "sub": "test-user"}

        app.dependency_overrides[get_embedding_service] = _override_embedding
        app.dependency_overrides[require_auth] = _override_auth
        self._app = app
        self.client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self):
        """Clean up after tests."""
        await self.client.aclose()
        # Remove overrides to avoid contaminating other test suites
        from app.api.deps import get_embedding_service
        from app.security import require_auth
        self._app.dependency_overrides.pop(get_embedding_service, None)
        self._app.dependency_overrides.pop(require_auth, None)

    async def test_ragas_endpoint_basic(self):
        """Test RAGAS evaluation endpoint with valid request."""
        request_data = {
            "query": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
            "contexts": ["Paris is the capital city of France.", "France is a country in Europe."],
            "ground_truth": "Paris"
        }
        
        response = await self.client.post("/api/eval/ragas", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check response structure
        self.assertIn("metrics", data)
        self.assertIn("evaluation_time_ms", data)
        
        # Check metrics
        metrics = data["metrics"]
        self.assertIn("faithfulness", metrics)
        self.assertIn("answer_relevancy", metrics)
        self.assertIn("context_precision", metrics)
        self.assertIn("context_recall", metrics)
        self.assertIn("context_relevancy", metrics)
        
        # Check metric ranges
        self.assertGreaterEqual(metrics["faithfulness"], 0.0)
        self.assertLessEqual(metrics["faithfulness"], 1.0)

    async def test_ragas_endpoint_without_ground_truth(self):
        """Test RAGAS endpoint without optional ground truth."""
        request_data = {
            "query": "What is Python?",
            "answer": "Python is a programming language.",
            "contexts": ["Python is a popular programming language."]
        }
        
        response = await self.client.post("/api/eval/ragas", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("metrics", data)
        self.assertIsNone(data["metrics"].get("answer_similarity"))

    async def test_ragas_endpoint_validation_error(self):
        """Test RAGAS endpoint with invalid request."""
        request_data = {
            "query": "",  # Empty query should fail validation
            "answer": "Some answer",
            "contexts": ["Some context"]
        }
        
        response = await self.client.post("/api/eval/ragas", json=request_data)
        
        self.assertEqual(response.status_code, 422)  # Validation error

    async def test_ragas_endpoint_missing_contexts(self):
        """Test RAGAS endpoint with missing contexts."""
        request_data = {
            "query": "What is test?",
            "answer": "This is a test."
            # Missing contexts field
        }
        
        response = await self.client.post("/api/eval/ragas", json=request_data)
        
        self.assertEqual(response.status_code, 422)  # Validation error

    async def test_ragas_metrics_calculation(self):
        """Test RAGAS metrics calculation logic."""
        from app.api.routes.eval import (
            _calculate_faithfulness,
            _calculate_answer_relevancy,
            _calculate_context_precision,
            _calculate_context_recall,
            _calculate_context_relevancy,
        )
        
        # Test faithfulness - answer grounded in context
        faithfulness = _calculate_faithfulness(
            "Paris is the capital of France.",
            ["Paris is the capital city of France."]
        )
        self.assertGreater(faithfulness, 0.5)  # Should be high
        
        # Test answer relevancy
        relevancy = _calculate_answer_relevancy(
            "What is the capital of France?",
            "The capital of France is Paris."
        )
        self.assertGreaterEqual(relevancy, 0.0)
        self.assertLessEqual(relevancy, 1.0)
        
        # Test context precision
        precision = _calculate_context_precision(
            ["Python is a programming language.", "Java is also a language."],
            "What is Python?"
        )
        self.assertGreaterEqual(precision, 0.0)
        self.assertLessEqual(precision, 1.0)
        
        # Test context recall with ground truth
        recall = _calculate_context_recall(
            ["Paris is the capital city of France."],
            "The capital of France is Paris."
        )
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)
        
        # Test context relevancy
        ctx_relevancy = _calculate_context_relevancy(
            ["Paris is the capital city of France."],
            "What is the capital of France?"
        )
        self.assertGreaterEqual(ctx_relevancy, 0.0)
        self.assertLessEqual(ctx_relevancy, 1.0)


if __name__ == "__main__":
    unittest.main()
