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
        self.timeout = 180.0
        
        # Read embedding prefixes from settings
        self.embedding_doc_prefix = settings.embedding_doc_prefix
        self.embedding_query_prefix = settings.embedding_query_prefix
        
        # Auto-apply Qwen3 instruction prefixes for better retrieval quality
        # With llama.cpp -ub 8192, we have plenty of headroom for these prefixes
        if settings.embedding_model.lower().find("qwen") >= 0:
            if not self.embedding_doc_prefix:
                self.embedding_doc_prefix = "Instruct: Represent this technical documentation passage for retrieval.\nDocument: "
            if not self.embedding_query_prefix:
                self.embedding_query_prefix = "Instruct: Retrieve relevant technical documentation passages.\nQuery: "
    
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

        Applies the query prefix (if configured) to the input text before embedding.
        The query prefix is used for retrieval queries and must remain constant for
        a given index to ensure consistent embedding space.

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

        # Apply query prefix for retrieval queries
        text_to_embed = self.embedding_query_prefix + text if self.embedding_query_prefix else text

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    self.embeddings_url,
                    json=self._build_payload(text_to_embed)
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

    async def embed_batch(self, texts: List[str], batch_size: int | None = None) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts using true API batching.

        Sends multiple texts per API request for efficient GPU utilization.
        Processes in batches of up to 512 (configurable) with up to 4
        concurrent batch requests.

        Applies the document prefix (if configured) to each input text before embedding.
        The document prefix is used for document embeddings and must remain constant for
        a given index to ensure consistent embedding space.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per API request (default: 512).

        Returns:
            List of embedding vectors, one for each input text, in order.

        Raises:
            EmbeddingError: If any API request fails.
        """
        if not texts:
            return []
        
        # Use configured batch size if not specified
        if batch_size is None:
            batch_size = settings.embedding_batch_size
        
        # Apply document prefix to all texts
        texts_to_embed = []
        for text in texts:
            if self.embedding_doc_prefix:
                texts_to_embed.append(self.embedding_doc_prefix + text)
            else:
                texts_to_embed.append(text)
        
        # Process in batches using true API batching
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            embeddings = await self._embed_batch_api(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings

    async def _embed_batch_api(self, texts: List[str]) -> List[List[float]]:
        """
        Send a batch of texts to the embedding API in a single request.

        Args:
            texts: List of texts to embed (already prefixed).

        Returns:
            List of embedding vectors in the same order as input texts.

        Raises:
            EmbeddingError: If the API request fails.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Build payload with array of inputs
                if self.provider_mode == 'openai':
                    payload = {
                        "model": settings.embedding_model,
                        "input": texts
                    }
                else:  # ollama mode
                    # Ollama supports batching via multiple generate calls or single with input array
                    # Try OpenAI-compatible batch format first
                    payload = {
                        "model": settings.embedding_model,
                        "input": texts
                    }
                
                response = await client.post(
                    self.embeddings_url,
                    json=payload
                )
                
                if response.status_code != 200:
                    raise EmbeddingError(
                        f"[{self.provider_mode.upper()} mode] Embedding API returned status {response.status_code}: {response.text}"
                    )

                data = response.json()
                
                # Extract embeddings from response
                if self.provider_mode == 'openai':
                    # OpenAI format: data[].embedding
                    embeddings = [item['embedding'] for item in data['data']]
                else:
                    # Ollama format may vary - try common formats
                    if 'embeddings' in data:
                        embeddings = data['embeddings']
                    elif 'embedding' in data:
                        # Single embedding returned - shouldn't happen with batch
                        embeddings = [data['embedding']]
                    else:
                        raise EmbeddingError(f"Unexpected response format: {data.keys()}")
                
                return embeddings
                
            except httpx.TimeoutException as e:
                raise EmbeddingError(f"[{self.provider_mode.upper()} mode] Embedding batch request timed out: {e}")
            except httpx.HTTPError as e:
                raise EmbeddingError(f"[{self.provider_mode.upper()} mode] Embedding batch HTTP error: {e}")



