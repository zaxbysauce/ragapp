"""
Ollama embedding client service.
"""
import httpx
from typing import List
from app.config import settings


class EmbeddingError(Exception):
    """Exception raised for embedding service errors."""
    pass


class EmbeddingService:
    """Service for generating text embeddings via Ollama API."""
    
    def __init__(self):
        """Initialize the embedding service with HTTP client."""
        base_url = settings.ollama_embedding_url

        # Validate base_url
        if not base_url:
            raise EmbeddingError("OLLAMA_EMBEDDING_URL is not configured")
        if not base_url.startswith(('http://', 'https://')):
            raise EmbeddingError(f"Invalid OLLAMA_EMBEDDING_URL: {base_url}. Must start with http:// or https://")

        # Ensure URL ends with /api/embeddings
        if not base_url.endswith('/api/embeddings'):
            base_url = base_url.rstrip('/') + '/api/embeddings'
        self.embeddings_url = base_url
        self.timeout = 30.0
    
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
                    json={
                        "model": settings.embedding_model,
                        "prompt": text
                    }
                )
                
                if response.status_code != 200:
                    raise EmbeddingError(
                        f"Embedding API returned status {response.status_code}: {response.text}"
                    )

                try:
                    data = response.json()
                except ValueError as e:
                    raise EmbeddingError(f"Invalid JSON response from embedding API: {e}")

                embedding = data.get("embedding")
                
                if embedding is None:
                    raise EmbeddingError("Embedding API response missing 'embedding' field")
                
                return embedding
                
            except httpx.TimeoutException as e:
                raise EmbeddingError(f"Embedding request timed out: {e}")
            except httpx.HTTPError as e:
                raise EmbeddingError(f"Embedding HTTP error: {e}")
    
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
                f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}"
            )
        return True

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts sequentially.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors, one for each input text.

        Raises:
            EmbeddingError: If any API request fails.
        """
        # Process sequentially - no concurrency as per spec
        embeddings = []
        for text in texts:
            embedding = await self.embed_single(text)
            embeddings.append(embedding)
        return embeddings



