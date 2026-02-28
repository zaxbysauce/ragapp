"""Unit tests for ContextualChunker service."""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock unstructured module before importing our modules
sys.modules['unstructured'] = MagicMock()
sys.modules['unstructured.chunking'] = MagicMock()
sys.modules['unstructured.chunking.title'] = MagicMock()
sys.modules['unstructured.documents'] = MagicMock()
sys.modules['unstructured.documents.elements'] = MagicMock()
sys.modules['unstructured.partition'] = MagicMock()
sys.modules['unstructured.partition.text'] = MagicMock()
sys.modules['unstructured.partition.auto'] = MagicMock()

# Define ProcessedChunk locally to avoid unstructured import
@dataclass
class ProcessedChunk:
    """Mock ProcessedChunk for testing."""
    text: str
    metadata: dict
    chunk_index: int = 0
    chunk_uid: Optional[str] = None
    original_indices: List[int] = field(default_factory=list)

from app.services.contextual_chunking import ContextualChunker


class TestContextualChunkerInit(unittest.TestCase):
    """Test ContextualChunker initialization."""

    def test_init_with_llm_client(self):
        """Test initialization with LLMClient dependency injection."""
        mock_llm_client = MagicMock()
        chunker = ContextualChunker(llm_client=mock_llm_client)
        self.assertIs(chunker._llm_client, mock_llm_client)
        self.assertIsNotNone(chunker._semaphore)

    def test_init_semaphore_default_concurrency(self):
        """Test that semaphore is created with default concurrency when not in settings."""
        mock_llm_client = MagicMock()
        chunker = ContextualChunker(llm_client=mock_llm_client)
        # Default concurrency is 5
        self.assertIsInstance(chunker._semaphore, asyncio.Semaphore)


class TestTruncateDocument(unittest.TestCase):
    """Test _truncate_document method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_client = MagicMock()
        self.chunker = ContextualChunker(llm_client=self.mock_llm_client)

    def test_no_truncation_for_short_document(self):
        """Test that short documents are not truncated."""
        short_doc = "This is a short document."
        result = self.chunker._truncate_document(short_doc)
        self.assertEqual(result, short_doc)

    def test_no_truncation_at_exact_limit(self):
        """Test that document at exact limit is not truncated."""
        exact_doc = "x" * ContextualChunker._MAX_DOCUMENT_LENGTH
        result = self.chunker._truncate_document(exact_doc)
        self.assertEqual(len(result), ContextualChunker._MAX_DOCUMENT_LENGTH)
        self.assertEqual(result, exact_doc)

    def test_truncation_for_long_document(self):
        """Test that long documents are truncated with [...truncated...] marker."""
        # Create a document that exceeds the limit
        long_doc = "a" * ContextualChunker._MAX_DOCUMENT_LENGTH + "b" * 1000
        result = self.chunker._truncate_document(long_doc)
        
        # Should be truncated
        self.assertIn("[...truncated...]", result)
        self.assertLess(len(result), len(long_doc))
        
        # Should contain first _TRUNCATE_CHARS and last _TRUNCATE_CHARS
        self.assertTrue(result.startswith("a" * 50))
        self.assertTrue(result.endswith("b" * 50))

    def test_truncation_preserves_content_at_boundaries(self):
        """Test that truncation keeps content from start and end."""
        # Create a document with identifiable start and end
        start_marker = "START_MARKER_CONTENT"
        end_marker = "END_MARKER_CONTENT"
        
        # Build a long document
        doc = start_marker + "x" * (ContextualChunker._MAX_DOCUMENT_LENGTH + 50000) + end_marker
        
        result = self.chunker._truncate_document(doc)
        self.assertIn(start_marker, result)
        self.assertIn(end_marker, result)


class TestBuildPrompt(unittest.TestCase):
    """Test _build_prompt method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_client = MagicMock()
        self.chunker = ContextualChunker(llm_client=self.mock_llm_client)

    def test_build_prompt_structure(self):
        """Test that _build_prompt returns correct message structure."""
        doc_text = "Full document text"
        chunk_text = "Chunk text"
        
        messages = self.chunker._build_prompt(
            document_text=doc_text,
            chunk_text=chunk_text,
            chunk_index=0,
            total_chunks=5,
            source_filename="test.txt"
        )
        
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

    def test_build_prompt_includes_chunk_info(self):
        """Test that prompt includes chunk index and total."""
        messages = self.chunker._build_prompt(
            document_text="doc",
            chunk_text="chunk",
            chunk_index=2,
            total_chunks=10,
            source_filename="document.pdf"
        )
        
        user_content = messages[1]["content"]
        self.assertIn("Chunk 3 of 10", user_content)  # chunk_index is 0-based
        self.assertIn("document.pdf", user_content)

    def test_build_prompt_includes_document_and_chunk(self):
        """Test that prompt includes document text and chunk text."""
        messages = self.chunker._build_prompt(
            document_text="Full document content here",
            chunk_text="This is the specific chunk",
            chunk_index=0,
            total_chunks=1,
            source_filename="file.md"
        )
        
        user_content = messages[1]["content"]
        self.assertIn("Full document content here", user_content)
        self.assertIn("This is the specific chunk", user_content)


class TestContextualizeChunks(unittest.TestCase):
    """Test contextualize_chunks method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_client = MagicMock()
        self.mock_llm_client.chat_completion = AsyncMock(return_value="Generated context")
        self.chunker = ContextualChunker(llm_client=self.mock_llm_client)

    def test_empty_chunks_returns_unchanged(self):
        """Test that empty chunks list returns unchanged."""
        result = asyncio.run(self.chunker.contextualize_chunks(
            document_text="doc",
            chunks=[],
            source_filename="test.txt"
        ))
        self.assertEqual(result, [])

    def test_chunks_are_contextualized(self):
        """Test that chunks are contextualized with LLM-generated context."""
        chunks = [
            ProcessedChunk(text="First chunk content", metadata={"page": 1}),
            ProcessedChunk(text="Second chunk content", metadata={"page": 2}),
        ]
        
        result = asyncio.run(self.chunker.contextualize_chunks(
            document_text="Full document text",
            chunks=chunks,
            source_filename="test.txt"
        ))
        
        # Check that context was prepended
        self.assertIn("Generated context", result[0].text)
        self.assertIn("First chunk content", result[0].text)
        self.assertTrue(result[0].metadata.get('contextualized'))

    def test_handles_llm_error_gracefully(self):
        """Test that LLM errors are handled gracefully."""
        from app.services.llm_client import LLMError
        
        self.mock_llm_client.chat_completion = AsyncMock(
            side_effect=LLMError("LLM service unavailable")
        )
        
        chunks = [ProcessedChunk(text="Chunk content", metadata={})]
        
        # Should not raise exception
        result = asyncio.run(self.chunker.contextualize_chunks(
            document_text="Document text",
            chunks=chunks,
            source_filename="test.txt"
        ))
        
        # Chunk should still be marked as contextualized
        self.assertTrue(result[0].metadata.get('contextualized'))
        # Original text should remain unchanged (no context prepended)
        self.assertEqual(result[0].text, "Chunk content")

    def test_handles_empty_llm_response(self):
        """Test that empty LLM responses are handled."""
        self.mock_llm_client.chat_completion = AsyncMock(return_value="   ")  # Whitespace only
        
        chunks = [ProcessedChunk(text="Chunk content", metadata={})]
        
        result = asyncio.run(self.chunker.contextualize_chunks(
            document_text="Document text",
            chunks=chunks,
            source_filename="test.txt"
        ))
        
        # Should be marked as contextualized but text unchanged
        self.assertTrue(result[0].metadata.get('contextualized'))
        self.assertEqual(result[0].text, "Chunk content")


class TestContextualizeSingleChunk(unittest.TestCase):
    """Test _contextualize_single_chunk method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_client = MagicMock()
        self.mock_llm_client.chat_completion = AsyncMock(return_value="Context for chunk")
        self.chunker = ContextualChunker(llm_client=self.mock_llm_client)

    def test_context_prepended_to_chunk_text(self):
        """Test that generated context is prepended to chunk text."""
        chunk = ProcessedChunk(text="Original chunk text", metadata={})
        
        asyncio.run(self.chunker._contextualize_single_chunk(
            chunk=chunk,
            document_text="Full document",
            chunk_index=0,
            total_chunks=1,
            source_filename="test.txt"
        ))
        
        self.assertEqual(chunk.text, "Context for chunk\n\nOriginal chunk text")
        self.assertTrue(chunk.metadata['contextualized'])

    def test_metadata_contextualized_always_set(self):
        """Test that contextualized is always set to True in metadata."""
        from app.services.llm_client import LLMError
        
        self.mock_llm_client.chat_completion = AsyncMock(
            side_effect=LLMError("Error")
        )
        
        chunk = ProcessedChunk(text="Text", metadata={})
        
        asyncio.run(self.chunker._contextualize_single_chunk(
            chunk=chunk,
            document_text="Doc",
            chunk_index=0,
            total_chunks=1,
            source_filename="test.txt"
        ))
        
        self.assertTrue(chunk.metadata['contextualized'])


class TestDependencyInjection(unittest.TestCase):
    """Test that ContextualChunker properly uses dependency injection."""

    def test_uses_injected_llm_client(self):
        """Test that the chunker uses the injected LLMClient, not a global."""
        mock_llm_client = MagicMock()
        mock_llm_client.chat_completion = AsyncMock(return_value="Context")
        
        chunker = ContextualChunker(llm_client=mock_llm_client)
        
        # Verify it's using the injected client
        self.assertIs(chunker._llm_client, mock_llm_client)

    def test_different_instances_use_different_clients(self):
        """Test that different ContextualChunker instances can use different LLMClients."""
        client1 = MagicMock()
        client2 = MagicMock()
        
        chunker1 = ContextualChunker(llm_client=client1)
        chunker2 = ContextualChunker(llm_client=client2)
        
        self.assertIs(chunker1._llm_client, client1)
        self.assertIs(chunker2._llm_client, client2)
        self.assertIsNot(chunker1._llm_client, chunker2._llm_client)


if __name__ == "__main__":
    unittest.main()
