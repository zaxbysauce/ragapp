"""Unit tests for the RAG pipeline."""

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
from app.services.rag_engine import RAGEngine
from app.services.vector_store import VectorStore


class FakeEmbeddingService:
    def __init__(self, embedding: List[float]):
        self.embedding = embedding

    async def embed_single(self, text: str) -> List[float]:
        return self.embedding


class FakeVectorStore:
    def __init__(self, results: List[Dict]):
        self._results = results

    def search(self, embedding: List[float], limit: int = 10, filter_expr=None, vault_id=None):
        return self._results[:limit]


class FakeMemoryStore:
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
    def __init__(self, response: str, stream_chunks: Optional[List[str]] = None):
        self._response = response
        self._stream_chunks = stream_chunks or []

    async def chat_completion(self, messages):
        return self._response

    async def chat_completion_stream(self, messages):
        for chunk in self._stream_chunks:
            yield chunk


class RAGEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_detects_memory_intent_and_returns_confirmation(self):
        memory_store = FakeMemoryStore(intent="remember that you are helpful")
        engine = RAGEngine()
        engine.embedding_service = cast(EmbeddingService, FakeEmbeddingService([0.1, 0.2]))
        engine.vector_store = cast(VectorStore, FakeVectorStore([]))
        engine.memory_store = cast(MemoryStore, memory_store)
        engine.llm_client = cast(LLMClient, FakeLLMClient(response=""))
        results = [msg async for msg in engine.query("remember that foo", [], stream=False)]
        self.assertEqual(1, len(results))
        self.assertEqual("content", results[0]["type"])
        self.assertIn("Memory stored", results[0]["content"])
        self.assertEqual(["remember that you are helpful"], memory_store.added)

    async def test_query_returns_sources_and_memories(self):
        memory = MemoryRecord(id=1, content="Important fact", category=None, tags=None, source="test", created_at=None, updated_at=None)
        vector_results = [
            {"text": "chunk one", "file_id": "file1", "metadata": {"source_file": "doc.md"}, "score": 0.9},
        ]
        engine = RAGEngine()
        engine.embedding_service = cast(EmbeddingService, FakeEmbeddingService([0.1, 0.2]))
        engine.vector_store = cast(VectorStore, FakeVectorStore(vector_results))
        engine.memory_store = cast(MemoryStore, FakeMemoryStore(memories=[memory]))
        engine.llm_client = cast(LLMClient, FakeLLMClient(response="answer"))
        results = [msg async for msg in engine.query("query", [], stream=False)]
        self.assertEqual("content", results[0]["type"])
        self.assertEqual("answer", results[0]["content"])
        done = results[-1]
        self.assertEqual("done", done["type"])
        self.assertEqual(1, len(done["sources"]))
        self.assertEqual([memory.content], done["memories_used"])

    async def test_streaming_response_yields_chunks(self):
        engine = RAGEngine()
        engine.embedding_service = cast(EmbeddingService, FakeEmbeddingService([0.1, 0.2]))
        engine.vector_store = cast(
            VectorStore,
            FakeVectorStore([
                {"text": "chunk", "file_id": "file1", "metadata": {}, "score": 0.5}
            ]),
        )
        engine.memory_store = cast(MemoryStore, FakeMemoryStore())
        engine.llm_client = cast(LLMClient, FakeLLMClient(response="", stream_chunks=["part1", "part2"]))
        stream = [msg async for msg in engine.query("query", [], stream=True)]
        self.assertEqual("content", stream[0]["type"])
        self.assertEqual("part1", stream[0]["content"])
        self.assertEqual("done", stream[-1]["type"])

    def test_filter_relevant_filters_scores_below_threshold(self):
        engine = RAGEngine()
        engine.relevance_threshold = 0.3
        results = [
            {"text": "low", "file_id": "f1", "metadata": {}, "score": 0.2},
            {"text": "medium", "file_id": "f2", "metadata": {}, "score": 0.3},
            {"text": "high", "file_id": "f3", "metadata": {}, "score": 0.5},
        ]
        filtered = engine._filter_relevant(results)
        self.assertEqual(2, len(filtered))
        self.assertEqual("medium", filtered[0].text)
        self.assertEqual("high", filtered[1].text)

    def test_filter_relevant_includes_scores_equal_to_threshold(self):
        engine = RAGEngine()
        engine.relevance_threshold = 0.3
        results = [
            {"text": "equal", "file_id": "f1", "metadata": {}, "score": 0.3},
            {"text": "above", "file_id": "f2", "metadata": {}, "score": 0.31},
        ]
        filtered = engine._filter_relevant(results)
        # Scores equal to threshold should be included
        self.assertEqual(2, len(filtered))
        self.assertEqual("equal", filtered[0].text)
        self.assertEqual("above", filtered[1].text)

    def test_filter_relevant_handles_none_score_as_default(self):
        engine = RAGEngine()
        engine.relevance_threshold = 0.5
        results = [
            {"text": "none_score", "file_id": "f1", "metadata": {}, "score": None},
            {"text": "low", "file_id": "f2", "metadata": {}, "score": 0.4},
        ]
        filtered = engine._filter_relevant(results)
        self.assertEqual(1, len(filtered))
        self.assertEqual("none_score", filtered[0].text)
        self.assertEqual(1.0, filtered[0].score)

    def test_filter_relevant_with_mixed_scores(self):
        engine = RAGEngine()
        engine.relevance_threshold = 0.3
        results = [
            {"text": "a", "file_id": "f1", "metadata": {"s": 1}, "score": 0.2},
            {"text": "b", "file_id": "f2", "metadata": {"s": 2}, "score": 0.3},
            {"text": "c", "file_id": "f3", "metadata": {"s": 3}, "score": 0.4},
            {"text": "d", "file_id": "f4", "metadata": {"s": 4}, "score": 0.1},
            {"text": "e", "file_id": "f5", "metadata": {"s": 5}, "score": 0.5},
        ]
        filtered = engine._filter_relevant(results)
        # Scores >= 0.3: b (0.3), c (0.4), e (0.5) = 3 results
        self.assertEqual(3, len(filtered))
        self.assertEqual("b", filtered[0].text)
        self.assertEqual(0.3, filtered[0].score)
        self.assertEqual({"s": 2}, filtered[0].metadata)
        self.assertEqual("c", filtered[1].text)
        self.assertEqual(0.4, filtered[1].score)
        self.assertEqual({"s": 3}, filtered[1].metadata)
        self.assertEqual("e", filtered[2].text)
        self.assertEqual(0.5, filtered[2].score)
        self.assertEqual({"s": 5}, filtered[2].metadata)

    def test_build_messages_with_empty_context(self):
        engine = RAGEngine()
        messages = engine._build_messages("my question", [], [], [])
        self.assertEqual(2, len(messages))
        self.assertEqual("system", messages[0]["role"])
        self.assertEqual("user", messages[1]["role"])
        self.assertEqual("Question: my question", messages[1]["content"])

    def test_build_system_prompt_contains_knowledgevault_and_cite_sources(self):
        engine = RAGEngine()
        prompt = engine._build_system_prompt()
        self.assertIn("KnowledgeVault", prompt)
        self.assertIn("cite", prompt.lower())

    def test_format_chunk_defaults_to_document_when_metadata_missing(self):
        engine = RAGEngine()
        from app.services.rag_engine import RAGSource
        chunk = RAGSource(text="some text", file_id="f1", score=0.8, metadata={})
        formatted = engine._format_chunk(chunk)
        self.assertIn("document", formatted)
        self.assertIn("some text", formatted)


if __name__ == "__main__":
    unittest.main()
