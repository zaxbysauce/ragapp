"""
Chat streaming endpoint tests using unittest and TestClient.

Tests SSE format, content accumulation, and done event structure.
"""
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

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

from app.main import app


class TestChatStreaming(unittest.TestCase):
    """Test suite for chat streaming endpoint."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def _parse_sse_events(self, response_text: str) -> list:
        """Parse SSE response text into list of event data.
        
        Handles multi-line events, data: with/without space,
        event: field, and retry: field (ignored).
        """
        events = []
        for block in response_text.strip().split('\n\n'):
            if not block:
                continue
            event_data = {}
            data_lines = []
            for line in block.split('\n'):
                if line.startswith('data:'):
                    # Handle 'data: ' and 'data:' (with or without space)
                    prefix_len = 6 if line.startswith('data: ') else 5
                    data_lines.append(line[prefix_len:])
                elif line.startswith('event:'):
                    # Handle 'event: ' and 'event:' (with or without space)
                    prefix_len = 7 if line.startswith('event: ') else 6
                    event_data['event_type'] = line[prefix_len:]
                elif line.startswith('retry:'):
                    # Retry field is parsed but ignored per spec
                    pass
            if data_lines:
                # Join multi-line data with newlines
                full_data = '\n'.join(data_lines)
                event_data['data'] = json.loads(full_data)
                events.append(event_data)
        return events

    @patch("app.api.routes.chat.RAGEngine")
    def test_stream_chat_returns_sse_format(self, mock_rag_engine_class):
        """Test streaming chat returns SSE format with data: lines."""
        # Mock RAGEngine to yield deterministic chunks
        mock_engine = mock_rag_engine_class.return_value
        
        async def mock_query(*args, **kwargs):
            yield {"type": "content", "content": "Hello"}
            yield {"type": "content", "content": " world"}
            yield {"type": "done", "sources": [], "memories_used": []}
        
        mock_engine.query = mock_query

        response = self.client.post(
            "/api/chat",
            json={"message": "test", "stream": True}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        
        # Verify SSE format: each line starts with "data: "
        text = response.text
        for line in text.strip().split('\n\n'):
            self.assertTrue(line.startswith("data: "), f"Line does not start with 'data: ': {line}")

    @patch("app.api.routes.chat.RAGEngine")
    def test_stream_chat_accumulates_content(self, mock_rag_engine_class):
        """Test streaming chat accumulates content chunks correctly."""
        mock_engine = mock_rag_engine_class.return_value
        
        async def mock_query(*args, **kwargs):
            yield {"type": "content", "content": "First"}
            yield {"type": "content", "content": " second"}
            yield {"type": "content", "content": " third"}
            yield {"type": "done", "sources": [], "memories_used": []}
        
        mock_engine.query = mock_query

        response = self.client.post(
            "/api/chat",
            json={"message": "test", "stream": True}
        )

        events = self._parse_sse_events(response.text)
        
        # Filter content events
        content_events = [e['data'] for e in events if e.get('data', {}).get("type") == "content"]
        self.assertEqual(len(content_events), 3)
        
        # Verify content accumulation
        full_content = "".join(e.get("content", "") for e in content_events)
        self.assertEqual(full_content, "First second third")

    @patch("app.api.routes.chat.RAGEngine")
    def test_stream_chat_done_event_has_sources(self, mock_rag_engine_class):
        """Test done event includes sources array."""
        mock_engine = mock_rag_engine_class.return_value
        
        expected_sources = [
            {"file_id": "doc1.txt", "score": 0.95, "metadata": {"source_file": "doc1.txt"}},
            {"file_id": "doc2.txt", "score": 0.87, "metadata": {"source_file": "doc2.txt"}}
        ]
        
        async def mock_query(*args, **kwargs):
            yield {"type": "content", "content": "Response"}
            yield {"type": "done", "sources": expected_sources, "memories_used": []}
        
        mock_engine.query = mock_query

        response = self.client.post(
            "/api/chat",
            json={"message": "test", "stream": True}
        )

        events = self._parse_sse_events(response.text)
        done_events = [e['data'] for e in events if e.get('data', {}).get("type") == "done"]
        
        self.assertEqual(len(done_events), 1)
        done_event = done_events[0]
        self.assertIn("sources", done_event)
        self.assertEqual(done_event["sources"], expected_sources)

    @patch("app.api.routes.chat.RAGEngine")
    def test_stream_chat_done_event_has_memories_used(self, mock_rag_engine_class):
        """Test done event includes memories_used array."""
        mock_engine = mock_rag_engine_class.return_value
        
        expected_memories = ["User likes Python", "User prefers dark mode"]
        
        async def mock_query(*args, **kwargs):
            yield {"type": "content", "content": "Response"}
            yield {"type": "done", "sources": [], "memories_used": expected_memories}
        
        mock_engine.query = mock_query

        response = self.client.post(
            "/api/chat",
            json={"message": "test", "stream": True}
        )

        events = self._parse_sse_events(response.text)
        done_events = [e['data'] for e in events if e.get('data', {}).get("type") == "done"]
        
        self.assertEqual(len(done_events), 1)
        done_event = done_events[0]
        self.assertIn("memories_used", done_event)
        self.assertEqual(done_event["memories_used"], expected_memories)

    @patch("app.api.routes.chat.RAGEngine")
    def test_stream_chat_with_history(self, mock_rag_engine_class):
        """Test streaming chat accepts history parameter."""
        mock_engine = mock_rag_engine_class.return_value
        captured_history = None
        
        async def mock_query(message, history, stream=False):
            nonlocal captured_history
            captured_history = history
            yield {"type": "content", "content": "Response"}
            yield {"type": "done", "sources": [], "memories_used": []}
        
        mock_engine.query = mock_query

        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]

        response = self.client.post(
            "/api/chat",
            json={"message": "test", "stream": True, "history": history}
        )

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(captured_history)
        self.assertEqual(len(captured_history), 2)
        
        # Assert RAGEngine was instantiated
        mock_rag_engine_class.assert_called_once()
        
        # Assert history content is passed correctly
        self.assertEqual(captured_history[0]["role"], "user")
        self.assertEqual(captured_history[0]["content"], "Previous question")
        self.assertEqual(captured_history[1]["role"], "assistant")
        self.assertEqual(captured_history[1]["content"], "Previous answer")

    @patch("app.api.routes.chat.RAGEngine")
    def test_stream_chat_empty_content_chunks(self, mock_rag_engine_class):
        """Test streaming handles empty content chunks gracefully."""
        mock_engine = mock_rag_engine_class.return_value
        
        async def mock_query(*args, **kwargs):
            yield {"type": "content", "content": ""}
            yield {"type": "content", "content": "Actual content"}
            yield {"type": "done", "sources": [], "memories_used": []}
        
        mock_engine.query = mock_query

        response = self.client.post(
            "/api/chat",
            json={"message": "test", "stream": True}
        )

        events = self._parse_sse_events(response.text)
        content_events = [e['data'] for e in events if e.get('data', {}).get("type") == "content"]
        
        # Should include empty content chunk
        self.assertEqual(len(content_events), 2)
        self.assertEqual(content_events[0].get("content"), "")
        self.assertEqual(content_events[1].get("content"), "Actual content")

    @patch("app.api.routes.chat.RAGEngine")
    def test_stream_chat_single_chunk_response(self, mock_rag_engine_class):
        """Test streaming with single content chunk and done event."""
        mock_engine = mock_rag_engine_class.return_value
        
        async def mock_query(*args, **kwargs):
            yield {"type": "content", "content": "Complete response"}
            yield {"type": "done", "sources": [], "memories_used": []}
        
        mock_engine.query = mock_query

        response = self.client.post(
            "/api/chat",
            json={"message": "test", "stream": True}
        )

        events = self._parse_sse_events(response.text)
        
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['data'].get("type"), "content")
        self.assertEqual(events[0]['data'].get("content"), "Complete response")
        self.assertEqual(events[1]['data'].get("type"), "done")

    def test_sse_parser_handles_multiline_data(self):
        """Test SSE parser handles multi-line data fields."""
        # Simulate SSE with multi-line data - newlines must be escaped in JSON
        sse_text = """data: {"type": "content", "content": "Line 1\\nLine 2"}

"""
        events = self._parse_sse_events(sse_text)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['data']['content'], "Line 1\nLine 2")

    def test_sse_parser_handles_data_without_space(self):
        """Test SSE parser handles 'data:' without space after colon."""
        sse_text = """data:{"type": "content", "content": "test"}

"""
        events = self._parse_sse_events(sse_text)
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['data']['content'], "test")

    def test_sse_parser_handles_event_field(self):
        """Test SSE parser captures event type field."""
        sse_text = """event: message
data: {"type": "content", "content": "test"}

"""
        events = self._parse_sse_events(sse_text)
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event_type'], "message")
        self.assertEqual(events[0]['data']['content'], "test")

    def test_sse_parser_ignores_retry_field(self):
        """Test SSE parser ignores retry field as per spec."""
        sse_text = """retry: 5000
data: {"type": "content", "content": "test"}

"""
        events = self._parse_sse_events(sse_text)
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['data']['content'], "test")
        # Retry field should not appear in parsed event
        self.assertNotIn('retry', events[0])

    @patch("app.api.routes.chat.RAGEngine")
    def test_stream_chat_newline_encoding_in_data(self, mock_rag_engine_class):
        """Test streaming handles newline characters in content data."""
        mock_engine = mock_rag_engine_class.return_value
        
        async def mock_query(*args, **kwargs):
            yield {"type": "content", "content": "Line 1\nLine 2\nLine 3"}
            yield {"type": "done", "sources": [], "memories_used": []}
        
        mock_engine.query = mock_query

        response = self.client.post(
            "/api/chat",
            json={"message": "test", "stream": True}
        )

        events = self._parse_sse_events(response.text)
        content_events = [e['data'] for e in events if e.get('data', {}).get("type") == "content"]
        
        self.assertEqual(len(content_events), 1)
        self.assertEqual(content_events[0].get("content"), "Line 1\nLine 2\nLine 3")
        
        # Assert RAGEngine was called
        mock_rag_engine_class.assert_called_once()


if __name__ == "__main__":
    unittest.main()
