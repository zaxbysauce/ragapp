"""Unit tests for the RAG pipeline."""

import os
import sys
import asyncio
import unittest
from typing import Any, AsyncIterator, Dict, List, Optional

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

from app.services.rag_engine import RAGEngine, RAGSource


class FakeEmbeddingService:
    """Deterministic fake embedding service for testing."""

    def __init__(self, embedding: Optional[List[float]] = None):
        # Default to a 3-dimensional embedding for predictable tests
        self.embedding = embedding if embedding is not None else [0.1, 0.2, 0.3]

    async def embed_single(self, text: str) -> List[float]:
        return self.embedding.copy()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embedding.copy() for _ in texts]


class FakeVectorStore:
    """Deterministic fake vector store for testing."""

    def __init__(self, results: Optional[List[Dict[str, Any]]] = None):
        self._results = results if results is not None else []

    def search(
        self,
        embedding: List[float],
        limit: int = 10,
        filter_expr: Optional[str] = None,
        vault_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return self._results[:limit]

    def get_chunks_by_uid(self, chunk_uids: List[str]) -> List[Dict[str, Any]]:
        # Return empty list for fake - real implementation would fetch from DB
        return []


class FakeMemoryRecord:
    """Simple fake memory record for testing."""

    def __init__(
        self,
        id: int = 1,
        content: str = "",
        category: Optional[str] = None,
        tags: Optional[str] = None,
        source: Optional[str] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None
    ):
        self.id = id
        self.content = content
        self.category = category
        self.tags = tags
        self.source = source
        self.created_at = created_at
        self.updated_at = updated_at


class FakeMemoryStore:
    """Deterministic fake memory store for testing."""

    def __init__(
        self,
        intent: Optional[str] = None,
        memories: Optional[List[FakeMemoryRecord]] = None
    ):
        self.intent = intent
        self._memories = memories if memories is not None else []
        self.added_memories: List[Dict[str, Any]] = []

    def detect_memory_intent(self, text: str) -> Optional[str]:
        return self.intent

    def add_memory(
        self,
        content: str,
        category: Optional[str] = None,
        tags: Optional[str] = None,
        source: Optional[str] = None,
        vault_id: Optional[int] = None
    ) -> FakeMemoryRecord:
        self.added_memories.append({
            "content": content,
            "category": category,
            "tags": tags,
            "source": source
        })
        return FakeMemoryRecord(
            id=len(self.added_memories),
            content=content,
            category=category,
            tags=tags,
            source=source
        )

    def search_memories(self, query: str, limit: int = 5, vault_id: Optional[int] = None) -> List[FakeMemoryRecord]:
        return self._memories[:limit]


class FakeLLMClient:
    """Deterministic fake LLM client for testing."""

    def __init__(
        self,
        response: str = "",
        stream_chunks: Optional[List[str]] = None
    ):
        self._response = response
        self._stream_chunks = stream_chunks if stream_chunks is not None else []
        self.last_messages: Optional[List[Dict[str, str]]] = None

    async def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        self.last_messages = messages
        return self._response

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]]
    ) -> AsyncIterator[str]:
        self.last_messages = messages
        for chunk in self._stream_chunks:
            yield chunk


class TestRAGPipeline(unittest.IsolatedAsyncioTestCase):
    """Test suite for RAG pipeline functionality."""

    async def test_memory_intent_path_returns_memory_stored_message(self):
        """Test that memory intent detection returns 'Memory stored' message."""
        memory_content = "my birthday is on January 1st"
        fake_memory_store = FakeMemoryStore(intent=memory_content)
        fake_embedding = FakeEmbeddingService()
        fake_vector = FakeVectorStore()
        fake_llm = FakeLLMClient()

        engine = RAGEngine(
            embedding_service=fake_embedding,
            vector_store=fake_vector,
            memory_store=fake_memory_store,
            llm_client=fake_llm
        )

        results = []
        async for msg in engine.query("remember that my birthday is on January 1st", []):
            results.append(msg)

        # Should return exactly one message
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["type"], "content")
        self.assertIn("Memory stored", results[0]["content"])
        self.assertIn(memory_content, results[0]["content"])

        # Verify memory was added
        self.assertEqual(len(fake_memory_store.added_memories), 1)
        self.assertEqual(fake_memory_store.added_memories[0]["content"], memory_content)
        self.assertEqual(fake_memory_store.added_memories[0]["source"], "chat")

    async def test_query_path_returns_content_and_done(self):
        """Test that normal query returns content and done message with sources."""
        vector_results = [
            {
                "text": "Python is a programming language",
                "file_id": "doc1",
                "score": 0.85,
                "metadata": {"source_file": "python_guide.md"}
            },
            {
                "text": "Python supports multiple paradigms",
                "file_id": "doc2",
                "score": 0.75,
                "metadata": {"source_file": "features.txt"}
            }
        ]
        memories = [
            FakeMemoryRecord(id=1, content="User prefers Python over Java"),
            FakeMemoryRecord(id=2, content="User likes clean code")
        ]

        fake_memory_store = FakeMemoryStore(memories=memories)
        fake_embedding = FakeEmbeddingService()
        fake_vector = FakeVectorStore(results=vector_results)
        fake_llm = FakeLLMClient(response="Python is indeed a versatile language.")

        engine = RAGEngine(
            embedding_service=fake_embedding,
            vector_store=fake_vector,
            memory_store=fake_memory_store,
            llm_client=fake_llm
        )

        results = []
        async for msg in engine.query("Tell me about Python", []):
            results.append(msg)

        # Should have content message and done message
        self.assertEqual(len(results), 2)

        # First message should be content
        self.assertEqual(results[0]["type"], "content")
        self.assertEqual(results[0]["content"], "Python is indeed a versatile language.")

        # Last message should be done with sources and memories
        done_msg = results[-1]
        self.assertEqual(done_msg["type"], "done")
        self.assertIn("sources", done_msg)
        self.assertIn("memories_used", done_msg)

        # Verify sources
        self.assertEqual(len(done_msg["sources"]), 2)
        self.assertEqual(done_msg["sources"][0]["file_id"], "doc1")
        self.assertEqual(done_msg["sources"][0]["score"], 0.85)
        self.assertEqual(done_msg["sources"][1]["file_id"], "doc2")
        self.assertEqual(done_msg["sources"][1]["score"], 0.75)

        # Verify memories used
        self.assertEqual(len(done_msg["memories_used"]), 2)
        self.assertEqual(done_msg["memories_used"][0], "User prefers Python over Java")
        self.assertEqual(done_msg["memories_used"][1], "User likes clean code")

    async def test_relevance_threshold_filters_low_scores(self):
        """Test that relevance threshold filters out low-scoring results."""
        vector_results = [
            {"text": "High relevance", "file_id": "doc1", "score": 0.9, "metadata": {}},
            {"text": "Medium relevance", "file_id": "doc2", "score": 0.5, "metadata": {}},
            {"text": "Low relevance", "file_id": "doc3", "score": 0.15, "metadata": {}},
            {"text": "Below threshold", "file_id": "doc4", "score": 0.05, "metadata": {}},
        ]

        fake_memory_store = FakeMemoryStore()
        fake_embedding = FakeEmbeddingService()
        fake_vector = FakeVectorStore(results=vector_results)
        fake_llm = FakeLLMClient(response="Answer based on relevant docs.")

        engine = RAGEngine(
            embedding_service=fake_embedding,
            vector_store=fake_vector,
            memory_store=fake_memory_store,
            llm_client=fake_llm
        )

        # Set relevance threshold to 0.2
        engine.relevance_threshold = 0.2

        results = []
        async for msg in engine.query("test query", []):
            results.append(msg)

        done_msg = results[-1]
        sources = done_msg["sources"]

        # Only results with score >= 0.2 should be included (scores equal to threshold are included)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["file_id"], "doc1")
        self.assertEqual(sources[1]["file_id"], "doc2")

    async def test_relevance_threshold_zero_includes_all(self):
        """Test that threshold of 0 includes all results."""
        vector_results = [
            {"text": "A", "file_id": "doc1", "score": 0.01, "metadata": {}},
            {"text": "B", "file_id": "doc2", "score": 0.0, "metadata": {}},
        ]

        fake_memory_store = FakeMemoryStore()
        fake_embedding = FakeEmbeddingService()
        fake_vector = FakeVectorStore(results=vector_results)
        fake_llm = FakeLLMClient(response="Answer.")

        engine = RAGEngine(
            embedding_service=fake_embedding,
            vector_store=fake_vector,
            memory_store=fake_memory_store,
            llm_client=fake_llm
        )

        engine.relevance_threshold = 0.0

        results = []
        async for msg in engine.query("test", []):
            results.append(msg)

        done_msg = results[-1]
        self.assertEqual(len(done_msg["sources"]), 2)

    async def test_relevance_threshold_high_excludes_most(self):
        """Test that high threshold excludes most results."""
        vector_results = [
            {"text": "A", "file_id": "doc1", "score": 0.95, "metadata": {}},
            {"text": "B", "file_id": "doc2", "score": 0.8, "metadata": {}},
            {"text": "C", "file_id": "doc3", "score": 0.79, "metadata": {}},
        ]

        fake_memory_store = FakeMemoryStore()
        fake_embedding = FakeEmbeddingService()
        fake_vector = FakeVectorStore(results=vector_results)
        fake_llm = FakeLLMClient(response="Answer.")

        engine = RAGEngine(
            embedding_service=fake_embedding,
            vector_store=fake_vector,
            memory_store=fake_memory_store,
            llm_client=fake_llm
        )

        engine.relevance_threshold = 0.8

        results = []
        async for msg in engine.query("test", []):
            results.append(msg)

        done_msg = results[-1]
        sources = done_msg["sources"]

        # Only scores >= 0.8 should be included (scores equal to threshold are included)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["file_id"], "doc1")
        self.assertEqual(sources[1]["file_id"], "doc2")
        self.assertEqual(sources[0]["score"], 0.95)
        self.assertEqual(sources[1]["score"], 0.8)

    async def test_streaming_query_yields_chunks(self):
        """Test that streaming query yields content chunks and done."""
        stream_chunks = ["Hello, ", "this ", "is ", "a ", "streamed ", "response."]

        fake_memory_store = FakeMemoryStore()
        fake_embedding = FakeEmbeddingService()
        fake_vector = FakeVectorStore(results=[])
        fake_llm = FakeLLMClient(stream_chunks=stream_chunks)

        engine = RAGEngine(
            embedding_service=fake_embedding,
            vector_store=fake_vector,
            memory_store=fake_memory_store,
            llm_client=fake_llm
        )

        results = []
        async for msg in engine.query("test", [], stream=True):
            results.append(msg)

        # Should have 6 content chunks + 1 done message
        self.assertEqual(len(results), 7)

        # Check content chunks
        for i, chunk in enumerate(stream_chunks):
            self.assertEqual(results[i]["type"], "content")
            self.assertEqual(results[i]["content"], chunk)

        # Last message should be done
        self.assertEqual(results[-1]["type"], "done")
        self.assertIn("sources", results[-1])
        self.assertIn("memories_used", results[-1])

    async def test_empty_vector_results_returns_no_sources(self):
        """Test that empty vector results yields no sources."""
        fake_memory_store = FakeMemoryStore()
        fake_embedding = FakeEmbeddingService()
        fake_vector = FakeVectorStore(results=[])
        fake_llm = FakeLLMClient(response="No relevant documents found.")

        engine = RAGEngine(
            embedding_service=fake_embedding,
            vector_store=fake_vector,
            memory_store=fake_memory_store,
            llm_client=fake_llm
        )

        results = []
        async for msg in engine.query("obscure topic", []):
            results.append(msg)

        done_msg = results[-1]
        self.assertEqual(done_msg["type"], "done")
        self.assertEqual(len(done_msg["sources"]), 0)

    async def test_no_memory_intent_proceeds_with_rag(self):
        """Test that when no memory intent is detected, RAG proceeds normally."""
        fake_memory_store = FakeMemoryStore(intent=None)
        fake_embedding = FakeEmbeddingService()
        fake_vector = FakeVectorStore(results=[])
        fake_llm = FakeLLMClient(response="Regular RAG response.")

        engine = RAGEngine(
            embedding_service=fake_embedding,
            vector_store=fake_vector,
            memory_store=fake_memory_store,
            llm_client=fake_llm
        )

        results = []
        async for msg in engine.query("What is the weather?", []):
            results.append(msg)

        # Should not have added any memories
        self.assertEqual(len(fake_memory_store.added_memories), 0)

        # Should have content and done
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["content"], "Regular RAG response.")

    async def test_chat_history_passed_to_llm(self):
        """Test that chat history is passed to LLM client."""
        chat_history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]

        fake_memory_store = FakeMemoryStore()
        fake_embedding = FakeEmbeddingService()
        fake_vector = FakeVectorStore(results=[])
        fake_llm = FakeLLMClient(response="Response.")

        engine = RAGEngine(
            embedding_service=fake_embedding,
            vector_store=fake_vector,
            memory_store=fake_memory_store,
            llm_client=fake_llm
        )

        async for _ in engine.query("Current question", chat_history):
            pass

        # Verify messages were passed to LLM
        self.assertIsNotNone(fake_llm.last_messages)
        self.assertEqual(len(fake_llm.last_messages), 4)  # system + 2 history + user
        self.assertEqual(fake_llm.last_messages[1], chat_history[0])
        self.assertEqual(fake_llm.last_messages[2], chat_history[1])


if __name__ == "__main__":
    unittest.main()
