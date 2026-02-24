"""
Tests for embedding service with adaptive batching.
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

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import HTTPError

from app.services.embeddings import EmbeddingService, EmbeddingError


@pytest.mark.asyncio
class TestEmbeddingBatching:
    """Test suite for adaptive embedding batching."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.mock_settings_patcher = patch('app.services.embeddings.settings')
        self.mock_settings = self.mock_settings_patcher.start()
        
        # Configure mock settings
        self.mock_settings.ollama_embedding_url = "http://localhost:11434/api/embeddings"
        self.mock_settings.embedding_model = "nomic-embed-text"
        self.mock_settings.embedding_doc_prefix = ""
        self.mock_settings.embedding_query_prefix = ""
        self.mock_settings.embedding_batch_size = 512
        self.mock_settings.embedding_batch_max_retries = 3
        self.mock_settings.embedding_batch_min_sub_size = 1
        
        # Mock the settings for chunk_size_chars
        self.mock_settings.chunk_size_chars = 1200
        self.mock_settings.chunk_overlap_chars = 120

    @pytest.fixture(autouse=True)
    def teardown(self):
        """Tear down test fixtures."""
        yield
        self.mock_settings_patcher.stop()

    def _create_mock_response(self, embeddings):
        """Create a mock HTTP response with embeddings (Ollama format)."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "embeddings": embeddings
        }
        return response

    def _create_overflow_error(self, message="input (4096 tokens) is too large to process, current batch size: 1024"):
        """Create an HTTPError simulating llama.cpp token overflow."""
        error = HTTPError(message)
        return error

    async def test_multi_item_batch_overflow_split_and_retry(self):
        """Test that overflow in multi-item batch triggers split/halving and succeeds."""
        # Create service
        service = EmbeddingService()
        
        # Mock texts - 20 items that will trigger overflow
        texts = [f"test text {i}" for i in range(20)]
        
        # First call (batch of 20) will overflow (retry_count=0)
        # Second call (batch of 10) will overflow (retry_count=1)
        # Third call (batch of 5) will succeed (retry_count=2)
        # The split creates: 20 -> 10+10 -> 5+5+5+5
        # So we need: 1 overflow for 20, 2 overflows for 10s, 4 successes for 5s
        mock_client = MagicMock()
        
        responses = [
            self._create_overflow_error(),  # Batch of 20 overflows
            self._create_overflow_error(),  # First batch of 10 overflows
            self._create_mock_response([[0.1] * 768] * 5),  # First sub-batch (5) succeeds
            self._create_mock_response([[0.1] * 768] * 5),  # Second sub-batch (5) succeeds
            self._create_overflow_error(),  # Second batch of 10 overflows
            self._create_mock_response([[0.2] * 768] * 5),  # Third sub-batch (5) succeeds
            self._create_mock_response([[0.2] * 768] * 5),  # Fourth sub-batch (5) succeeds
        ]
        
        async def mock_post(*args, **kwargs):
            resp = responses.pop(0)
            if isinstance(resp, HTTPError):
                raise resp
            return resp
        
        with patch('app.services.embeddings.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post = mock_post
            
            # This should succeed after splitting
            embeddings = await service.embed_batch(texts, batch_size=20)
            
            # Verify we got 20 embeddings (one per input)
            assert len(embeddings) == 20
            
            # Verify order is preserved (first 5 from first sub-batch, next 5 from second, etc.)
            for i in range(5):
                assert embeddings[i] == [0.1] * 768, f"Index {i} should be [0.1]*768"
            for i in range(5, 10):
                assert embeddings[i] == [0.1] * 768, f"Index {i} should be [0.1]*768"
            for i in range(10, 15):
                assert embeddings[i] == [0.2] * 768, f"Index {i} should be [0.2]*768"
            for i in range(15, 20):
                assert embeddings[i] == [0.2] * 768, f"Index {i} should be [0.2]*768"

    async def test_single_item_batch_overflow_raises_actionable_error(self):
        """Test that overflow with single-item batch raises actionable EmbeddingError."""
        # Create service
        service = EmbeddingService()
        
        # Single text that will overflow
        texts = ["very long test text that will overflow the token limit"]
        
        mock_client = MagicMock()
        
        # Always return overflow error
        overflow_error = self._create_overflow_error()
        
        with patch('app.services.embeddings.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=overflow_error)
            
            # Should raise EmbeddingError with actionable message
            with pytest.raises(EmbeddingError) as context:
                await service.embed_batch(texts, batch_size=1)
            
            # Verify error message is actionable
            error_msg = str(context.value)
            assert "single input" in error_msg.lower()
            assert "chunk_size_chars" in error_msg.lower()
            assert "server batch" in error_msg.lower()

    async def test_batch_preserves_order_after_split(self):
        """Test that batch order is preserved after adaptive splitting."""
        service = EmbeddingService()
        
        # 15 items that will split into batches of 7 and 8 (midpoint split)
        texts = [f"text_{i}" for i in range(15)]
        
        mock_client = MagicMock()
        
        # First batch of 15 overflows, splits to 7 and 8
        # Both sub-batches succeed
        responses = [
            self._create_overflow_error(),  # Initial batch overflows
            self._create_mock_response([[float(i)] * 768 for i in range(7)]),  # First 7
            self._create_mock_response([[float(i + 7)] * 768 for i in range(8)]),  # Next 8
        ]
        
        async def mock_post(*args, **kwargs):
            resp = responses.pop(0)
            if isinstance(resp, HTTPError):
                raise resp
            return resp
        
        with patch('app.services.embeddings.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post = mock_post
            
            embeddings = await service.embed_batch(texts, batch_size=15)
            
            # Verify order is preserved
            assert len(embeddings) == 15
            for i in range(7):
                assert embeddings[i] == [float(i)] * 768
            for i in range(8):
                assert embeddings[7 + i] == [float(7 + i)] * 768

    async def test_no_overflow_returns_immediately(self):
        """Test that successful batch returns without retry."""
        service = EmbeddingService()
        
        texts = ["text1", "text2", "text3"]
        
        mock_client = MagicMock()
        
        # Success on first try
        mock_client.post = AsyncMock(return_value=self._create_mock_response(
            [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        ))
        
        with patch('app.services.embeddings.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            embeddings = await service.embed_batch(texts, batch_size=5)
            
            # Should return immediately without retry
            assert len(embeddings) == 3
            assert embeddings[0] == [0.1] * 768
            assert embeddings[1] == [0.2] * 768
            assert embeddings[2] == [0.3] * 768

    async def test_non_overflow_error_raises_immediately(self):
        """Test that non-token-overflow errors raise immediately."""
        service = EmbeddingService()
        
        texts = ["text1"]
        
        mock_client = MagicMock()
        
        # Non-overflow error (e.g., 500 internal server error)
        error = HTTPError("Internal server error")
        
        with patch('app.services.embeddings.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=error)
            
            with pytest.raises(EmbeddingError) as context:
                await service.embed_batch(texts, batch_size=5)
            
            # Error should not mention token overflow
            assert "single input" not in str(context.value).lower()

    async def test_ollama_mode_overflow_handling(self):
        """Test that Ollama mode also handles overflow correctly."""
        # Configure for Ollama mode
        self.mock_settings.ollama_embedding_url = "http://localhost:11434/api/embeddings"
        service = EmbeddingService()
        
        assert service.provider_mode == 'ollama'
        
        texts = ["text1", "text2"]
        
        mock_client = MagicMock()
        
        # Ollama overflow error format
        overflow_error = HTTPError("input (8192 tokens) is too large to process")
        
        with patch('app.services.embeddings.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=overflow_error)
            
            with pytest.raises(EmbeddingError) as context:
                await service.embed_batch(texts, batch_size=5)
            
            # Should raise actionable error
            assert "single input" in str(context.value).lower()


class TestIsTokenOverflowError:
    """Test the _is_token_overflow_error detection method."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.mock_settings_patcher = patch('app.services.embeddings.settings')
        self.mock_settings = self.mock_settings_patcher.start()
        self.mock_settings.ollama_embedding_url = "http://localhost:11434/api/embeddings"
        self.mock_settings.embedding_model = "nomic-embed-text"
        self.mock_settings.embedding_doc_prefix = ""
        self.mock_settings.embedding_query_prefix = ""
        self.mock_settings.embedding_batch_size = 512
        self.mock_settings.embedding_batch_max_retries = 3
        self.mock_settings.embedding_batch_min_sub_size = 1
        self.mock_settings.chunk_size_chars = 1200
        self.mock_settings.chunk_overlap_chars = 120

    @pytest.fixture(autouse=True)
    def teardown(self):
        yield
        self.mock_settings_patcher.stop()

    def _create_service(self):
        return EmbeddingService()

    def test_llama_cpp_overflow_pattern(self):
        """Test detection of llama.cpp overflow pattern."""
        service = self._create_service()
        error_msg = "input (4096 tokens) is too large to process, current batch size: 1024"
        assert service._is_token_overflow_error(error_msg) is True

    def test_openai_overflow_pattern(self):
        """Test detection of OpenAI overflow pattern."""
        service = self._create_service()
        error_msg = "too large to process, current batch size: 1024"
        assert service._is_token_overflow_error(error_msg) is True

    def test_token_limit_exceeded_pattern(self):
        """Test detection of token limit exceeded pattern."""
        service = self._create_service()
        error_msg = "token limit exceeded for request"
        assert service._is_token_overflow_error(error_msg) is True

    def test_batch_size_too_small_pattern(self):
        """Test detection of batch size too small pattern."""
        service = self._create_service()
        error_msg = "batch size too small for processing"
        assert service._is_token_overflow_error(error_msg) is True

    def test_non_overflow_error(self):
        """Test that non-overflow errors are not detected."""
        service = self._create_service()
        error_msg = "Internal server error"
        assert service._is_token_overflow_error(error_msg) is False

    def test_case_insensitive_detection(self):
        """Test that detection is case-insensitive."""
        service = self._create_service()
        error_msg = "INPUT (4096 TOKENS) IS TOO LARGE TO PROCESS"
        assert service._is_token_overflow_error(error_msg) is True
