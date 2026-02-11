"""
Dual-provider embedding client service supporting Ollama and OpenAI-compatible APIs.
"""
import asyncio
import httpx
from typing import List
from urllib.parse import urlparse
from app.config import settings


class EmbeddingError(Exception):
    """Exception raised for embedding service errors."""
    pass


class EmbeddingService:
    """Service for generating text embeddings via Ollama or OpenAI-compatible APIs."""
    
    def __init__(self):
        """Initialize the embedding service with HTTP client and provider detection."""
        base_url = settings.ollama_embedding_url

        # Validate base_url
        if not base_url:
            raise EmbeddingError("OLLAMA_EMBEDDING_URL is not configured")
        if not base_url.startswith(('http://', 'https://')):
            raise EmbeddingError(f"Invalid OLLAMA_EMBEDDING_URL: {base_url}. Must start with http:// or https://")

        # Detect provider mode based on URL path
        self.provider_mode, self.embeddings_url = self._detect_provider_mode(base_url)
        self.timeout = 30.0
    
    def _detect_provider_mode(self, base_url: str) -> tuple:
        """
        Detect which embedding provider mode to use based on URL path.
        
        Detection strategy:
        - If URL path includes '/api/embeddings' -> Ollama mode
        - If URL path includes '/v1/embeddings' -> OpenAI mode
        - If no explicit embeddings path:
          - Port 1234 -> OpenAI mode (LM Studio default)
          - Otherwise -> Ollama mode
        
        Args:
            base_url: The configured embedding URL
            
        Returns:
            Tuple of (provider_mode, embeddings_url)
        """
        parsed = urlparse(base_url)
        path = parsed.path
        
        # Check for explicit paths
        if '/api/embeddings' in path:
            # Already has Ollama path, use as-is
            return ('ollama', base_url)
        elif '/v1/embeddings' in path:
            # Already has OpenAI path, use as-is
            return ('openai', base_url)
        
        # No explicit path - determine by port
        port = parsed.port
        if port == 1234:
            # LM Studio default port - use OpenAI mode
            base_url = base_url.rstrip('/') + '/v1/embeddings'
            return ('openai', base_url)
        else:
            # Default to Ollama mode
            base_url = base_url.rstrip('/') + '/api/embeddings'
            return ('ollama', base_url)
    
    def _build_payload(self, text: str) -> dict:
        """
        Build the API request payload based on provider mode.
        
        Args:
            text: The text to embed
            
        Returns:
            Dictionary payload for the API request
        """
        if self.provider_mode == 'openai':
            return {
                "model": settings.embedding_model,
                "input": text
            }
        else:  # ollama mode
            return {
                "model": settings.embedding_model,
                "prompt": text
            }
    
    def _extract_embedding(self, data: dict) -> List[float]:
        """
        Extract embedding vector from API response based on provider mode.
        
        Args:
            data: Parsed JSON response from the API
            
        Returns:
            List of float values representing the embedding vector
            
        Raises:
            EmbeddingError: If embedding cannot be extracted
        """
        if self.provider_mode == 'openai':
            # OpenAI format: data[0].embedding
            if "data" not in data:
                raise EmbeddingError(f"[OpenAI mode] Embedding API response missing 'data' field")
            if not isinstance(data["data"], list) or len(data["data"]) == 0:
                raise EmbeddingError(f"[OpenAI mode] Embedding API response 'data' field is empty or invalid")
            embedding = data["data"][0].get("embedding")
            if embedding is None:
                raise EmbeddingError(f"[OpenAI mode] Embedding API response missing 'data[0].embedding' field")
        else:  # ollama mode
            # Ollama format: embedding
            embedding = data.get("embedding")
            if embedding is None:
                raise EmbeddingError(f"[Ollama mode] Embedding API response missing 'embedding' field")
        
        return embedding
    
    async def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            List of float values representing the embedding vector.

        Raises:
            EmbeddingError: If the API request fails or returns non-200 status.
        """
        # Validate text input
        if text is None:
            raise EmbeddingError("Text cannot be None")
        if not text.strip():
            raise EmbeddingError("Text cannot be empty or whitespace only")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    self.embeddings_url,
                    json=self._build_payload(text)
                )
                
                if response.status_code != 200:
                    raise EmbeddingError(
                        f"[{self.provider_mode.upper()} mode] Embedding API returned status {response.status_code}: {response.text}"
                    )

                try:
                    data = response.json()
                except ValueError as e:
                    raise EmbeddingError(f"[{self.provider_mode.upper()} mode] Invalid JSON response from embedding API: {e}")

                return self._extract_embedding(data)
                
            except httpx.TimeoutException as e:
                raise EmbeddingError(f"[{self.provider_mode.upper()} mode] Embedding request timed out: {e}")
            except httpx.HTTPError as e:
                raise EmbeddingError(f"[{self.provider_mode.upper()} mode] Embedding HTTP error: {e}")
    
    async def validate_embedding_dimension(self, expected_dim: int) -> bool:
        """
        Validate that the embedding dimension matches the expected value.

        Args:
            expected_dim: The expected dimension of the embedding vector.
                Must be a positive integer.

        Returns:
            True if the dimension matches.

        Raises:
            EmbeddingError: If expected_dim is invalid or if the dimension
                does not match the expected value.
        """
        # Validate expected_dim input
        if expected_dim is None:
            raise EmbeddingError("expected_dim cannot be None")
        if not isinstance(expected_dim, int) or expected_dim <= 0:
            raise EmbeddingError(f"expected_dim must be a positive integer, got {expected_dim}")

        embedding = await self.embed_single('dimension_check')
        actual_dim = len(embedding)
        if actual_dim != expected_dim:
            raise EmbeddingError(
                f"[{self.provider_mode.upper()} mode] Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}"
            )
        return True

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts concurrently.

        Uses asyncio.gather with a semaphore to limit concurrency.
        Processes texts in sub-batches of up to 64, with up to 10
        concurrent requests per sub-batch.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors, one for each input text, in order.

        Raises:
            EmbeddingError: If any API request fails.
        """
        if not texts:
            return []
        
        semaphore = asyncio.Semaphore(10)
        
        async def _embed_with_limit(text: str) -> List[float]:
            async with semaphore:
                return await self.embed_single(text)
        
        # Process in sub-batches of 64 to avoid overwhelming the server
        all_embeddings: List[List[float]] = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            sub_batch = texts[i:i + batch_size]
            results = await asyncio.gather(*[_embed_with_limit(t) for t in sub_batch])
            all_embeddings.extend(results)
        
        return all_embeddings



