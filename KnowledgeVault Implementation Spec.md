# KnowledgeVault Implementation Spec
## BGE-M3 + BGE-Reranker-v2-M3 · Hybrid Search · Three-Endpoint Config · Bug Fixes

**Target repo:** `zaxbysauce-ragapp` / KnowledgeVault  
**Implementing agent instructions:** Apply every change below in the order listed. Each section specifies the exact file, what to change, and provides the replacement code. Do not skip sections. Run the test suite after Phase 1 before proceeding.

***

## Context & Architecture Note

The existing retrieval pipeline is: embed query → dense vector search → threshold filter (currently broken) → build prompt → stream LLM. After these changes it becomes: embed query → **hybrid BM25 + dense search + RRF** → **cross-encoder rerank** → build prompt → stream LLM. The embedding model is **BGE-M3** (dense via Ollama). The reranker is **BGE-Reranker-v2-M3** served via a configurable endpoint (TEI-compatible) or loaded locally via `sentence-transformers` as fallback.

***

## Phase 1 — Bug Fixes (implement and test first)

### 1.1 Fix `_distance` vs `score` field in `rag_engine.py`

**Problem:** `_filter_relevant()` reads `record.get("score", 1.0)`. LanceDB returns `_distance`, not `score`. The default of `1.0` means every chunk passes regardless of relevance — the threshold setting is a no-op.

**File:** `backend/app/services/rag_engine.py`

Find the `_filter_relevant` method (or wherever relevance filtering occurs) and replace the score logic:

```python
# REMOVE this pattern wherever it appears:
score = record.get("score", 1.0)
if score < self.relevance_threshold:
    continue

# REPLACE WITH:
distance = record.get("_distance")
if distance is None:
    # _distance missing means LanceDB API changed; log and include
    logger.warning("Record missing _distance field; including without threshold filter")
elif self.max_distance_threshold is not None and self.max_distance_threshold > 0:
    if distance > self.max_distance_threshold:
        continue
```

Also update any `RAGEngine.__init__` that reads `self.relevance_threshold` to instead read `self.max_distance_threshold`:

```python
# In RAGEngine.__init__:
self.max_distance_threshold: Optional[float] = getattr(settings, "max_distance_threshold", None)
```

***

### 1.2 Fix `_load_persisted_settings` in `main.py`

**Problem:** `_load_persisted_settings` loads legacy keys (`chunk_size`, `rag_relevance_threshold`, `vector_top_k`) from the `settings_kv` table but does NOT load the new keys (`chunk_size_chars`, `chunk_overlap_chars`, `retrieval_top_k`, `max_distance_threshold`). On restart, all new settings revert to defaults.

**File:** `backend/app/main.py`

In the `_load_persisted_settings` function, extend the key list to include all new fields. Add after the existing legacy-key loading block:

```python
# New fields — load directly without legacy conversion
NEW_DIRECT_KEYS = [
    "chunk_size_chars",
    "chunk_overlap_chars",
    "retrieval_top_k",
    "retrieval_window",
    "max_distance_threshold",
    "vector_metric",
    "embedding_doc_prefix",
    "embedding_query_prefix",
    "embedding_batch_size",
    "reranking_enabled",
    "reranker_top_n",
    "initial_retrieval_top_k",
    "hybrid_search_enabled",
    "hybrid_alpha",
    "reranker_url",
    "reranker_model",
]
for key in NEW_DIRECT_KEYS:
    if key in persisted:
        try:
            expected_type = type(getattr(settings, key))
            raw = persisted[key]
            if expected_type == bool:
                setattr(settings, key, str(raw).lower() in ("true", "1", "yes"))
            elif expected_type == int:
                setattr(settings, key, int(raw))
            elif expected_type == float:
                setattr(settings, key, float(raw))
            else:
                setattr(settings, key, raw)
        except Exception as e:
            logger.warning(f"Failed to restore persisted setting {key}: {e}")
```

***

### 1.3 Fix `retrieval_top_k` priority in `RAGEngine.query()`

**Problem:** `RAGEngine.query()` uses `self.top_k if self.top_k is not None else self.retrieval_top_k`. If `vector_top_k` was ever set, `retrieval_top_k` is ignored, and the UI "Max Context Chunks" control has no effect.

**File:** `backend/app/services/rag_engine.py`

Find the top_k selection logic and replace:

```python
# REMOVE:
top_k_value = self.top_k if self.top_k is not None else self.retrieval_top_k

# REPLACE WITH:
# Always use retrieval_top_k (the canonical field). For two-stage retrieval,
# initial_retrieval_top_k is used before reranking; retrieval_top_k is the
# final count passed to the LLM context.
initial_top_k = getattr(self, "initial_retrieval_top_k", self.retrieval_top_k * 3)
final_top_k = self.retrieval_top_k
```

***

### 1.4 Fix `SettingsResponse` Pydantic validation error for legacy Optional fields

**Problem:** Tests show `pydantic_core.ValidationError: 4 validation errors for SettingsResponse` when `chunk_size`, `chunk_overlap`, `rag_relevance_threshold`, `vector_top_k` are `None`. The `SettingsResponse` model declares them as `int`/`float` but they are now `Optional`.

**File:** `backend/app/api/routes/settings.py`

In `SettingsResponse`, change the four legacy fields to `Optional`:

```python
# CHANGE these four field declarations:
chunk_size: Optional[int] = None       # was: int
chunk_overlap: Optional[int] = None    # was: int
rag_relevance_threshold: Optional[float] = None  # was: float
vector_top_k: Optional[int] = None    # was: int
```

Also update both `get_settings()` and `apply_settings_update()` to not include these in the dict if they are `None`, to avoid validation cascade.

***

## Phase 2 — New Configuration Fields

### 2.1 Add reranker and hybrid search fields to `config.py`

**File:** `backend/app/config.py`

Add the following fields to the `Settings` class. Place them after the existing `embedding_batch_size` field:

```python
# ── Reranker configuration ────────────────────────────────────────────────
reranker_url: str = ""
# Empty string = use sentence-transformers locally (no external endpoint needed).
# Set to a TEI-compatible endpoint, e.g. http://host.docker.internal:8082
# TEI rerank endpoint: POST {reranker_url}/rerank
# Expected request: {"query": str, "texts": [str, ...], "top_n": int, "truncate": true}
# Expected response: [{"index": int, "score": float}, ...]

reranker_model: str = "BAAI/bge-reranker-v2-m3"
# Used as the HuggingFace model ID for local sentence-transformers loading,
# OR sent as the model name if the TEI endpoint requires it.

reranking_enabled: bool = False
# Set to true to activate cross-encoder reranking after vector retrieval.

reranker_top_n: int = 5
# Number of chunks to keep after reranking. Must be <= initial_retrieval_top_k.

initial_retrieval_top_k: int = 20
# Chunks fetched from the vector store BEFORE reranking.
# If reranking_enabled=false, this is unused and retrieval_top_k is used directly.

# ── Hybrid search configuration ───────────────────────────────────────────
hybrid_search_enabled: bool = True
# Combine dense vector search + BM25 full-text search via RRF fusion.
# Requires LanceDB >= 0.10.0 and a full-text index on the 'text' column.

hybrid_alpha: float = 0.5
# Weight for dense vs sparse scores in RRF.
# 0.0 = pure BM25, 1.0 = pure dense. 0.5 = equal weight.
```

***

### 2.2 Update `.env.example`

**File:** `.env.example`

Add after the existing `OLLAMA_CHAT_URL` block:

```bash
# Reranker Configuration
# The 8GB GPU runs BGE-M3 and the reranker ONLY (chat model is on a separate host).
# Leave RERANKER_URL empty to load BGE-Reranker-v2-M3 locally via sentence-transformers
# directly on the same GPU as BGE-M3. With ~2.4GB total for both models, there is
# ample headroom on 8GB and reranking is enabled by default.
# Set to a TEI endpoint URL to use an external reranker service instead.
RERANKER_URL=
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKING_ENABLED=true
RERANKER_TOP_N=5
INITIAL_RETRIEVAL_TOP_K=20

# Hybrid Search Configuration
HYBRID_SEARCH_ENABLED=true
HYBRID_ALPHA=0.5
```

Also **update** the following existing defaults:

```bash
# Model Configuration
EMBEDDING_MODEL=bge-m3            # changed from nomic-embed-text / qwen3-embedding:4b
CHAT_MODEL=qwen3:8b               # update to reflect current recommended model
```

***

### 2.3 Update `docker-compose.yml`

**File:** `docker-compose.yml`

In the `knowledgevault` service `environment` block, add the new variables and update model defaults:

```yaml
environment:
  - DATA_DIR=/data/knowledgevault
  - OLLAMA_EMBEDDING_URL=${OLLAMA_EMBEDDING_URL:-http://host.docker.internal:11434}
  - OLLAMA_CHAT_URL=${OLLAMA_CHAT_URL:-http://host.docker.internal:11434}
  - EMBEDDING_MODEL=${EMBEDDING_MODEL:-bge-m3}
  - CHAT_MODEL=${CHAT_MODEL:-qwen3:8b}
  - RERANKER_URL=${RERANKER_URL:-}
  - RERANKER_MODEL=${RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}
  - RERANKING_ENABLED=${RERANKING_ENABLED:-true}
  - RERANKER_TOP_N=${RERANKER_TOP_N:-5}
  - INITIAL_RETRIEVAL_TOP_K=${INITIAL_RETRIEVAL_TOP_K:-20}
  - HYBRID_SEARCH_ENABLED=${HYBRID_SEARCH_ENABLED:-true}
  - HYBRID_ALPHA=${HYBRID_ALPHA:-0.5}
  - LOG_LEVEL=${LOG_LEVEL:-INFO}
```

Also add a Redis service for persistent CSRF tokens (add as a new service alongside `knowledgevault`):

```yaml
  redis:
    image: redis:7-alpine
    container_name: knowledgevault_redis
    restart: unless-stopped
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

volumes:
  redis_data:
```

Add `REDIS_URL` to the `knowledgevault` environment block:

```yaml
  - REDIS_URL=${REDIS_URL:-redis://redis:6379/0}
```

Add `depends_on` to `knowledgevault`:

```yaml
    depends_on:
      redis:
        condition: service_healthy
```

***

### 2.4 Update `SettingsResponse` and `SettingsUpdate` in `settings.py`

**File:** `backend/app/api/routes/settings.py`

Add the following fields to **both** `SettingsResponse` and `SettingsUpdate` Pydantic models:

```python
# Add to SettingsResponse (all required with defaults):
reranker_url: str = ""
reranker_model: str = "BAAI/bge-reranker-v2-m3"
reranking_enabled: bool = False
reranker_top_n: int = 5
initial_retrieval_top_k: int = 20
hybrid_search_enabled: bool = True
hybrid_alpha: float = 0.5

# Add to SettingsUpdate (all Optional):
reranker_url: Optional[str] = None
reranker_model: Optional[str] = None
reranking_enabled: Optional[bool] = None
reranker_top_n: Optional[int] = None
initial_retrieval_top_k: Optional[int] = None
hybrid_search_enabled: Optional[bool] = None
hybrid_alpha: Optional[float] = None
```

Update the `ALLOWED_FIELDS` list (or equivalent) to include all seven new fields.

Update the `settings_dict` in both `get_settings()` and `apply_settings_update()` to include:

```python
"reranker_url": settings.reranker_url,
"reranker_model": settings.reranker_model,
"reranking_enabled": settings.reranking_enabled,
"reranker_top_n": settings.reranker_top_n,
"initial_retrieval_top_k": settings.initial_retrieval_top_k,
"hybrid_search_enabled": settings.hybrid_search_enabled,
"hybrid_alpha": settings.hybrid_alpha,
```

***

### 2.5 Update `/api/settings/connection` endpoint

**File:** `backend/app/api/routes/settings.py`

The existing `test_connection` endpoint checks embedding and chat URLs. Extend it to also test the reranker URL when configured:

```python
@router.get("/settings/connection")
async def test_connection():
    targets = {
        "embeddings": settings.ollama_embedding_url,
        "chat": settings.ollama_chat_url,
    }
    if settings.reranker_url:
        targets["reranker"] = settings.reranker_url

    async with httpx.AsyncClient(timeout=5.0) as client:
        results = {}
        for name, url in targets.items():
            try:
                response = await client.get(url)
                results[name] = {"url": url, "status": response.status_code, "ok": response.status_code < 300}
            except Exception as exc:
                results[name] = {"url": url, "status": None, "ok": False, "error": str(exc)}
        
        # If no reranker_url, report local mode
        if not settings.reranker_url:
            results["reranker"] = {
                "url": "local (sentence-transformers)",
                "ok": True,
                "status": "local",
                "model": settings.reranker_model,
            }
    return results
```

***

## Phase 3 — New `RerankingService`

### 3.1 Create `backend/app/services/reranking.py`

Create this file from scratch:

```python
"""
Cross-encoder reranking service for KnowledgeVault.

Supports two backends:
  1. TEI endpoint (if reranker_url is set): POST {url}/rerank
     Expected request:  {"query": str, "texts": [str], "top_n": int, "truncate": true}
     Expected response: [{"index": int, "score": float}, ...]
  2. Local sentence-transformers CrossEncoder (if reranker_url is empty).
     Model is loaded lazily on first use and cached.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_local_model = None  # lazy-loaded CrossEncoder instance


def _get_local_model(model_id: str):
    """Lazy-load and cache a sentence-transformers CrossEncoder."""
    global _local_model
    if _local_model is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading local CrossEncoder reranker: {model_id}")
            _local_model = CrossEncoder(model_id)
            logger.info("Local reranker loaded successfully")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Either set RERANKER_URL to a TEI endpoint, or add "
                "'sentence-transformers>=2.7.0' to requirements.txt."
            )
    return _local_model


class RerankingService:
    """
    Reranks a list of (chunk_dict) results given a query string.

    chunk_dict must have at minimum a 'text' key.
    Returns the top_n highest-scoring chunks, ordered by relevance descending.
    """

    def __init__(self, reranker_url: str, reranker_model: str, top_n: int = 5):
        self.reranker_url = reranker_url.rstrip("/") if reranker_url else ""
        self.reranker_model = reranker_model
        self.top_n = top_n

    async def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks given a query. Returns top_n chunks sorted by score desc.

        Args:
            query: The user's query string.
            chunks: List of chunk dicts (must contain 'text' key).
            top_n: Override instance top_n for this call.

        Returns:
            Reranked and trimmed list of chunk dicts with '_rerank_score' added.
        """
        n = top_n or self.top_n
        if not chunks:
            return []
        if len(chunks) <= 1:
            return chunks

        texts = [c.get("text", "") for c in chunks]

        try:
            if self.reranker_url:
                scored = await self._rerank_via_endpoint(query, texts, n)
            else:
                scored = await self._rerank_local(query, texts, n)
        except Exception as e:
            logger.error(f"Reranking failed, returning original order: {e}")
            return chunks[:n]

        # Attach score and return top_n
        result = []
        for idx, score in scored:
            chunk = dict(chunks[idx])
            chunk["_rerank_score"] = score
            result.append(chunk)
        return result

    async def _rerank_via_endpoint(
        self, query: str, texts: List[str], top_n: int
    ) -> List[Tuple[int, float]]:
        """
        Call a TEI-compatible rerank endpoint.

        TEI format:
          POST /rerank
          Body: {"query": str, "texts": [str], "top_n": int, "truncate": true}
          Response: [{"index": int, "score": float}, ...]
        """
        url = f"{self.reranker_url}/rerank"
        payload = {
            "query": query,
            "texts": texts,
            "top_n": top_n,
            "truncate": True,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        # data is list of {"index": int, "score": float}
        return [(item["index"], item["score"]) for item in data[:top_n]]

    async def _rerank_local(
        self, query: str, texts: List[str], top_n: int
    ) -> List[Tuple[int, float]]:
        """
        Rerank using a locally loaded sentence-transformers CrossEncoder.
        Runs in a thread to avoid blocking the event loop.
        """
        import asyncio

        def _score():
            model = _get_local_model(self.reranker_model)
            pairs = [(query, text) for text in texts]
            scores = model.predict(pairs)
            indexed = sorted(enumerate(scores), key=lambda x: x[^1], reverse=True)
            return indexed[:top_n]

        return await asyncio.to_thread(_score)
```

***

### 3.2 Add `sentence-transformers` to `requirements.txt`

**File:** `backend/requirements.txt`

Add:

```
sentence-transformers>=2.7.0
```

This is only used when `RERANKER_URL` is empty (local mode). It will not be imported unless reranking is enabled and no endpoint is configured.

***

### 3.3 Register `RerankingService` in app state

**File:** `backend/app/main.py`

In the app startup (`lifespan` or `startup` event), after existing service initialization:

```python
from app.services.reranking import RerankingService

# After existing service setup:
app.state.reranking_service = RerankingService(
    reranker_url=settings.reranker_url,
    reranker_model=settings.reranker_model,
    top_n=settings.reranker_top_n,
)
```

***

### 3.4 Add `get_reranking_service` dependency

**File:** `backend/app/api/deps.py`

```python
from app.services.reranking import RerankingService

def get_reranking_service(request: Request) -> RerankingService:
    """Return the RerankingService from app state."""
    return request.app.state.reranking_service
```

***

### 3.5 Update `get_rag_engine` to accept `RerankingService`

**File:** `backend/app/api/deps.py`

Update the `get_rag_engine` dependency to inject `RerankingService`:

```python
def get_rag_engine(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    memory_store: MemoryStore = Depends(get_memory_store),
    llm_client: LLMClient = Depends(get_llm_client),
    reranking_service: RerankingService = Depends(get_reranking_service),
) -> RAGEngine:
    return RAGEngine(
        embedding_service=embedding_service,
        vector_store=vector_store,
        memory_store=memory_store,
        llm_client=llm_client,
        reranking_service=reranking_service,
    )
```

***

## Phase 4 — Hybrid Search in `VectorStore`

### 4.1 Upgrade LanceDB

**File:** `backend/requirements.txt`

Change:

```
# FROM:
lancedb>=0.4.0

# TO:
lancedb>=0.10.0
```

***

### 4.2 Add FTS index creation to `VectorStore.init_table()`

**File:** `backend/app/services/vector_store.py`

In `init_table()`, after the table is created or opened, add a full-text search index on the `text` column:

```python
# After table creation/opening:
try:
    table.create_fts_index("text", replace=True)
    logger.info("Full-text search index created on 'text' column")
except Exception as e:
    logger.warning(f"FTS index creation failed (hybrid search will be unavailable): {e}")
```

Also store the vector metric explicitly at creation time:

```python
# When creating a NEW table (not opening existing):
table.create_index(
    metric=settings.vector_metric,  # "cosine" or "l2"
    num_partitions=256,
    num_sub_vectors=96,
    replace=True,
)
logger.info(f"Vector index created with metric={settings.vector_metric}")
```

***

### 4.3 Implement hybrid search with RRF in `VectorStore.search()`

**File:** `backend/app/services/vector_store.py`

Replace the existing `search()` method with the hybrid implementation:

```python
def search(
    self,
    query_vector: list,
    query_text: str,
    top_k: int = 10,
    vault_id: Optional[int] = None,
    hybrid: bool = True,
    hybrid_alpha: float = 0.5,
) -> list:
    """
    Search the vector store.

    If hybrid=True (and FTS index exists), runs both dense vector search
    and BM25 full-text search, then fuses results with Reciprocal Rank Fusion.

    Args:
        query_vector: Dense embedding of the query.
        query_text:   Raw query string for BM25 search.
        top_k:        Number of results to return.
        vault_id:     Optional vault filter.
        hybrid:       Enable hybrid search (dense + BM25).
        hybrid_alpha: Not used in pure RRF but reserved for weighted fusion.

    Returns:
        List of result dicts with _distance and optionally _rrf_score.
    """
    table = self._get_table()
    if table is None:
        return []

    where_clause = f"vault_id = {vault_id}" if vault_id is not None else None
    fetch_k = top_k * 3  # fetch more candidates for fusion

    # ── Dense search ──────────────────────────────────────────────────────
    dense_query = table.search(query_vector, vector_column_name="embedding")
    if where_clause:
        dense_query = dense_query.where(where_clause)
    try:
        dense_results = dense_query.limit(fetch_k).to_list()
    except Exception as e:
        logger.error(f"Dense search failed: {e}")
        dense_results = []

    if not hybrid:
        return dense_results[:top_k]

    # ── BM25 / FTS search ─────────────────────────────────────────────────
    try:
        fts_query = table.search(query_text)  # LanceDB FTS query
        if where_clause:
            fts_query = fts_query.where(where_clause)
        fts_results = fts_query.limit(fetch_k).to_list()
    except Exception as e:
        logger.warning(f"FTS search failed (falling back to dense-only): {e}")
        return dense_results[:top_k]

    # ── RRF Fusion ────────────────────────────────────────────────────────
    # RRF score: sum(1 / (k + rank)) across all rankers. k=60 is standard.
    k_rrf = 60
    rrf_scores: dict[str, float] = {}
    id_to_record: dict[str, dict] = {}

    for rank, record in enumerate(dense_results):
        uid = record.get("id", f"dense_{rank}")
        rrf_scores[uid] = rrf_scores.get(uid, 0.0) + 1.0 / (k_rrf + rank + 1)
        id_to_record[uid] = record

    for rank, record in enumerate(fts_results):
        uid = record.get("id", f"fts_{rank}")
        rrf_scores[uid] = rrf_scores.get(uid, 0.0) + 1.0 / (k_rrf + rank + 1)
        if uid not in id_to_record:
            id_to_record[uid] = record

    # Sort by RRF score descending, return top_k
    sorted_uids = sorted(rrf_scores, key=lambda u: rrf_scores[u], reverse=True)
    fused = []
    for uid in sorted_uids[:top_k]:
        record = dict(id_to_record[uid])
        record["_rrf_score"] = rrf_scores[uid]
        fused.append(record)

    logger.debug(
        f"Hybrid search: dense={len(dense_results)}, "
        f"fts={len(fts_results)}, fused={len(fused)}"
    )
    return fused
```

***

## Phase 5 — Update `RAGEngine.query()` Pipeline

### 5.1 Update `RAGEngine.__init__`

**File:** `backend/app/services/rag_engine.py`

Add `reranking_service` parameter and new config fields:

```python
def __init__(
    self,
    embedding_service,
    vector_store,
    memory_store,
    llm_client,
    reranking_service=None,  # NEW
):
    self.embedding_service = embedding_service
    self.vector_store = vector_store
    self.memory_store = memory_store
    self.llm_client = llm_client
    self.reranking_service = reranking_service  # NEW

    # Config — read from settings at init time
    self.retrieval_top_k: int = getattr(settings, "retrieval_top_k", 10)
    self.initial_retrieval_top_k: int = getattr(settings, "initial_retrieval_top_k", 20)
    self.retrieval_window: int = getattr(settings, "retrieval_window", 1)
    self.max_distance_threshold: Optional[float] = getattr(settings, "max_distance_threshold", None)
    self.reranking_enabled: bool = getattr(settings, "reranking_enabled", False)
    self.reranker_top_n: int = getattr(settings, "reranker_top_n", 5)
    self.hybrid_search_enabled: bool = getattr(settings, "hybrid_search_enabled", True)
    self.hybrid_alpha: float = getattr(settings, "hybrid_alpha", 0.5)
    self.max_context_chunks: int = getattr(settings, "retrieval_top_k", 10)
```

***

### 5.2 Update `RAGEngine.query()` to use the two-stage pipeline

**File:** `backend/app/services/rag_engine.py`

Replace the vector search + filter block inside `query()` with:

```python
# ── Stage 1: Retrieval ────────────────────────────────────────────────────
fetch_k = self.initial_retrieval_top_k if self.reranking_enabled else self.retrieval_top_k

query_embedding = await self.embedding_service.embed_query(query)

raw_chunks = await asyncio.to_thread(
    self.vector_store.search,
    query_embedding,
    query,                          # pass raw text for BM25
    top_k=fetch_k,
    vault_id=vault_id,
    hybrid=self.hybrid_search_enabled,
    hybrid_alpha=self.hybrid_alpha,
)

# ── Distance threshold filter ─────────────────────────────────────────────
filtered_chunks = []
for record in raw_chunks:
    distance = record.get("_distance")
    if (
        distance is not None
        and self.max_distance_threshold is not None
        and self.max_distance_threshold > 0
        and distance > self.max_distance_threshold
    ):
        continue
    filtered_chunks.append(record)

# ── Stage 2: Reranking (optional) ────────────────────────────────────────
if self.reranking_enabled and self.reranking_service and filtered_chunks:
    reranked = await self.reranking_service.rerank(
        query=query,
        chunks=filtered_chunks,
        top_n=self.reranker_top_n,
    )
    context_chunks = reranked
else:
    context_chunks = filtered_chunks[:self.retrieval_top_k]

# ── Adjacent-window expansion (existing logic — keep as-is) ───────────────
# ... existing windowing code continues here unchanged ...

# ── Retrieval debug payload ───────────────────────────────────────────────
retrieval_debug = {
    "retrieval_top_k": self.retrieval_top_k,
    "initial_retrieval_top_k": fetch_k,
    "retrieval_window": self.retrieval_window,
    "hybrid_search": self.hybrid_search_enabled,
    "reranking": self.reranking_enabled,
    "threshold": self.max_distance_threshold,
    "raw_hits": len(raw_chunks),
    "after_threshold": len(filtered_chunks),
    "final_context_chunks": len(context_chunks),
    "distance_stats": {
        "min": min((r.get("_distance", 0) for r in raw_chunks), default=None),
        "max": max((r.get("_distance", 0) for r in raw_chunks), default=None),
    },
}
```

***

### 5.3 Update `EmbeddingService` to expose an `embed_query` method

**File:** `backend/app/services/embeddings.py`

The existing service likely has `embed_texts()` or `embed_batch()`. Add a dedicated `embed_query` method that applies the query prefix:

```python
async def embed_query(self, query: str) -> list:
    """
    Embed a single query string with the configured query prefix.
    Uses embedding_query_prefix from settings if set.
    """
    prefix = getattr(settings, "embedding_query_prefix", "")
    prefixed = f"{prefix}{query}" if prefix else query
    results = await self.embed_texts([prefixed])
    return results
```

If the existing code already handles prefixes, ensure `embed_query` uses the query-side prefix (not the document prefix).

***

## Phase 6 — Embedding Model: BGE-M3 Notes

### 6.1 Instruction prefix defaults for BGE-M3

**BGE-M3 does not use instruction prefixes.** Ensure the default values in `config.py` for `embedding_doc_prefix` and `embedding_query_prefix` are empty strings when the model is BGE-M3:

```python
# In config.py, the auto-prefix logic (if any) should only activate for Qwen models:
@property
def effective_embedding_doc_prefix(self) -> str:
    if self.embedding_doc_prefix:
        return self.embedding_doc_prefix
    # Auto-apply Qwen3 instruction prefix only for Qwen models
    if "qwen" in self.embedding_model.lower():
        return "Instruct: Represent this technical documentation passage for retrieval.\nDocument: "
    return ""

@property
def effective_embedding_query_prefix(self) -> str:
    if self.embedding_query_prefix:
        return self.embedding_query_prefix
    if "qwen" in self.embedding_model.lower():
        return "Instruct: Retrieve relevant technical documentation passages.\nQuery: "
    return ""
```

Update all embedding calls to use `settings.effective_embedding_doc_prefix` and `settings.effective_embedding_query_prefix` instead of reading the raw fields directly.

***

### 6.2 Pull BGE-M3 via Ollama

In `README.md` and `docs/admin-guide.md`, replace all references to `nomic-embed-text` and `qwen3-embedding:4b` with:

```bash
# Required: Embedding model
ollama pull bge-m3

# Optional: Reranker (only needed if RERANKER_URL is empty and RERANKING_ENABLED=true)
# BGE-Reranker-v2-M3 is loaded automatically via sentence-transformers from HuggingFace.
# No Ollama pull required for the reranker.
```

***

## Phase 7 — Vector Metric Lock

### 7.1 Lock cosine metric at LanceDB table creation

**File:** `backend/app/services/vector_store.py`

In `init_table()`, when creating a NEW table, set the metric explicitly:

```python
# After table.add(initial_data) or when creating index:
try:
    table.create_index(
        metric=settings.vector_metric,  # default "cosine"
        num_partitions=256,
        num_sub_vectors=96,
        replace=True,
    )
    logger.info(f"ANN index created: metric={settings.vector_metric}, dim={embedding_dim}")
except Exception as e:
    logger.warning(f"ANN index creation failed (search will use brute force): {e}")
```

Also store the metric in the table metadata so schema validation can detect mismatches:

```python
# Store in a separate metadata table or in the DB:
conn.execute(
    "INSERT OR REPLACE INTO settings_kv (key, value) VALUES (?, ?)",
    ("lancedb_vector_metric", settings.vector_metric)
)
```

At startup, validate that the stored metric matches the current config:

```python
stored_metric = _read_kv("lancedb_vector_metric")
if stored_metric and stored_metric != settings.vector_metric:
    logger.error(
        f"Vector metric mismatch: index was built with '{stored_metric}', "
        f"config says '{settings.vector_metric}'. Reindex required."
    )
    # Block search until resolved
    self._metric_mismatch = True
```

***

## Phase 8 — Frontend Settings UI

### 8.1 Add reranker fields to `SettingsPage.tsx`

**File:** `frontend/src/pages/SettingsPage.tsx`

Add a new "Retrieval" section to the settings form. The three endpoint fields should be grouped:

```tsx
{/* Endpoint Configuration */}
<section>
  <h3>Endpoints</h3>
  <SettingField
    label="Embedding URL"
    description="Ollama endpoint serving the embedding model."
    value={settings.ollama_embedding_url}
    onChange={(v) => updateSetting("ollama_embedding_url", v)}
  />
  <SettingField
    label="Chat URL"
    description="Ollama endpoint serving the chat model."
    value={settings.ollama_chat_url}
    onChange={(v) => updateSetting("ollama_chat_url", v)}
  />
  <SettingField
    label="Reranker URL"
    description="TEI-compatible reranker endpoint. Leave empty to use local sentence-transformers."
    value={settings.reranker_url}
    onChange={(v) => updateSetting("reranker_url", v)}
    placeholder="http://host.docker.internal:8082  (or leave empty for local)"
  />
  <SettingField
    label="Reranker Model"
    description="HuggingFace model ID for local reranking, or model name sent to TEI endpoint."
    value={settings.reranker_model}
    onChange={(v) => updateSetting("reranker_model", v)}
  />
</section>

{/* Reranking */}
<section>
  <h3>Reranking</h3>
  <ToggleField
    label="Enable Reranking"
    description="Re-score retrieved chunks with a cross-encoder before sending to the LLM."
    value={settings.reranking_enabled}
    onChange={(v) => updateSetting("reranking_enabled", v)}
  />
  <NumberField
    label="Initial Retrieval Top-K"
    description="Chunks fetched before reranking. Should be 3–4× Reranker Top-N."
    value={settings.initial_retrieval_top_k}
    onChange={(v) => updateSetting("initial_retrieval_top_k", v)}
    min={5} max={100}
  />
  <NumberField
    label="Reranker Top-N"
    description="Chunks kept after reranking and passed to the LLM."
    value={settings.reranker_top_n}
    onChange={(v) => updateSetting("reranker_top_n", v)}
    min={1} max={20}
  />
</section>

{/* Hybrid Search */}
<section>
  <h3>Hybrid Search</h3>
  <ToggleField
    label="Enable Hybrid Search"
    description="Combine BM25 keyword search with dense vector search using RRF fusion."
    value={settings.hybrid_search_enabled}
    onChange={(v) => updateSetting("hybrid_search_enabled", v)}
  />
</section>
```

Ensure `useSettingsStore.ts` includes all new fields in the settings type definition.

***

## Phase 9 — Tests

### 9.1 Unit tests to add/update

**File:** `backend/tests/test_rag_pipeline.py`

Add these test cases:

```python
def test_distance_threshold_filters_correctly():
    """_distance field must be used for threshold filtering, not 'score'."""
    records = [
        {"id": "a", "_distance": 0.1, "text": "close"},
        {"id": "b", "_distance": 0.9, "text": "far"},
    ]
    engine = build_test_rag_engine(max_distance_threshold=0.5)
    filtered = engine._apply_threshold(records)
    assert len(filtered) == 1
    assert filtered["id"] == "a"


def test_no_threshold_includes_all():
    """When max_distance_threshold is None, no records are filtered."""
    records = [{"id": "a", "_distance": 0.99}, {"id": "b", "_distance": 0.01}]
    engine = build_test_rag_engine(max_distance_threshold=None)
    filtered = engine._apply_threshold(records)
    assert len(filtered) == 2


def test_rrf_fusion_deduplicates():
    """RRF fusion must deduplicate records appearing in both dense and BM25 results."""
    dense = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    sparse = [{"id": "b"}, {"id": "d"}, {"id": "a"}]
    fused = rrf_fuse(dense, sparse, k=60)
    ids = [r["id"] for r in fused]
    assert len(ids) == len(set(ids)), "Duplicates found in RRF output"
    # 'a' and 'b' appear in both, should rank highest
    assert ids in ("a", "b")
    assert ids[^1] in ("a", "b")


def test_reranking_service_local_fallback(monkeypatch):
    """RerankingService falls back gracefully when endpoint is unavailable."""
    service = RerankingService(reranker_url="", reranker_model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=2)
    chunks = [{"text": "Paris is the capital of France"}, {"text": "Dogs are mammals"}, {"text": "France hosts the Eiffel Tower"}]
    import asyncio
    result = asyncio.run(service.rerank("What is the capital of France?", chunks, top_n=2))
    assert len(result) == 2
    # Top result should be about Paris/France
    assert "France" in result["text"] or "Paris" in result["text"]
```

**File:** `backend/tests/test_settings.py`

Add:

```python
def test_settings_response_handles_none_legacy_fields():
    """SettingsResponse must not raise ValidationError when legacy Optional fields are None."""
    data = {**BASE_SETTINGS_DICT, "chunk_size": None, "chunk_overlap": None,
            "rag_relevance_threshold": None, "vector_top_k": None}
    response = SettingsResponse.model_validate(data)
    assert response.chunk_size is None


def test_new_reranker_fields_in_settings_response():
    """SettingsResponse must include reranker_url, reranker_model, reranking_enabled."""
    data = {**BASE_SETTINGS_DICT}
    response = SettingsResponse.model_validate(data)
    assert hasattr(response, "reranker_url")
    assert hasattr(response, "reranking_enabled")
    assert hasattr(response, "hybrid_search_enabled")
```

***

## Summary of All Changed Files

| File | Change Type | Phase |
|------|-------------|-------|
| `backend/app/services/rag_engine.py` | Fix `_distance` bug, add two-stage pipeline | 1, 5 |
| `backend/app/main.py` | Fix `_load_persisted_settings`, register RerankingService | 1, 3 |
| `backend/app/api/routes/settings.py` | Fix Optional fields, add new fields + endpoints | 1, 2 |
| `backend/app/config.py` | Add reranker + hybrid search config fields, auto-prefix property | 2, 6 |
| `.env.example` | Add RERANKER_URL/MODEL/ENABLED, HYBRID_*, update model defaults | 2 |
| `docker-compose.yml` | Add new env vars, Redis service | 2 |
| `backend/app/services/reranking.py` | **Create new file** | 3 |
| `backend/requirements.txt` | Add `sentence-transformers>=2.7.0`, upgrade `lancedb>=0.10.0` | 3, 4 |
| `backend/app/api/deps.py` | Add `get_reranking_service`, update `get_rag_engine` | 3 |
| `backend/app/services/vector_store.py` | FTS index, hybrid search, metric lock | 4, 7 |
| `backend/app/services/embeddings.py` | Add `embed_query()` with query-side prefix | 5 |
| `frontend/src/pages/SettingsPage.tsx` | Add reranker + hybrid search fields, three endpoint fields | 8 |
| `frontend/src/stores/useSettingsStore.ts` | Add new field types | 8 |
| `backend/tests/test_rag_pipeline.py` | New threshold + RRF + reranking tests | 9 |
| `backend/tests/test_settings.py` | New settings validation tests | 9 |
| `README.md` / `docs/admin-guide.md` | Update Ollama pull instructions for BGE-M3 | 6 |

---

## References

1. [zaxbysauce-ragapp-8a5edab282632443-4.txt](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/75010107/afb3033a-33bd-4ce8-9fab-22acbdd5b3b8/zaxbysauce-ragapp-8a5edab282632443-4.txt?AWSAccessKeyId=ASIA2F3EMEYEUHOJNZIN&Signature=H0QRdxngYNLqR6YSlLOmDQR%2FqiI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEDUaCXVzLWVhc3QtMSJHMEUCIQDwX5kfgebhw4dj%2B5v7DfznipB0Wby2hOld94VJI9wngAIgV2YsiNrEGg1lVtZ8csuClpGKItnK6OIgGFhl8XO01f0q%2FAQI%2Ff%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDGl9EYVHWfHJtydKJirQBBqZt06aGiH%2BaIJEaixMxhzi%2FEPTxCwten%2B7cAimjE4%2FbVmtpNFrfi%2Bne%2BKDVUNv9FkfMeUHrUjFRnFckx9zVpb%2FF8W9yQJ7DwUYstLqOm%2BenqA9B1KARU7DqqNyEF5EFy8ZEV1W3Ow1x1DntIPSikEhwXdTs2raW3Ejih7wtVxJ4pshr7A4xE0IYfwIHQtO0Q%2FVr38eVs%2F22QXgSYQmvELU73WhrZHEW4%2BMxcFYTF%2BNZUB3GZM0%2BYEhu%2FoFSn3cw6fwW%2BHuXOGdkfJeOIIxK7LXY7oHq8DVx%2FKoHhG5ZR1dna0TUaddAtgwxAfQ4BvK8U0ANOZKzoKXsZ%2B2cgQDtpUG1xtX35Dsn7fvR5VRpwrmOXIzD3DQg8BawfaBJLh0ELaRZVnogOupuITzDaUQAty0En%2FIMd7aT7Z%2ByGkwKfzRX3PAhaA0TIJYNgfwlJmy1S%2BzaV6QEwjxtrsuiLNJ4INC%2FJM%2BglW8BOBHxeJuLNimi31wlaTHHcN5yfuCracXC6CkWi3mDCFTZQVkB1AgS4K%2B7lj8XYdwKEjgZ81piqe8Yxwjo%2Fvvsbbmsiqdj5HohXA64LklWfVmpiT%2FnqprAGK13CXVCqAnJiU%2BFDv%2FCmhUFXZCevvpHbNh3s9n6h%2BcYq2a1Dlp9j2EBPOisxpMP8ZkE0ZNUvNTfh9pqwtDW38Oo%2FVz%2FTxa%2FFP7UmJ5RsH87nHkDYafUfGfRcZQFTOJFWZcftdah2Tvqrtv4uFkFRe2I3rU%2BcuI%2BjISqNpWFbh3CjGYqbuI5wvO6tbQnW4neTgwxor4zAY6mAHq%2FzaaoKjb%2FvhINwf8%2FZHTg0oyup0tQKOp4xeH7PW%2FNo%2BVgXN8FQnhMXmJLT8jnAAIBIxJrKGW6qJ7yVueGcFTCXYkcjw0NCcP5tX1sF1Sr9WuM136y3aM8R2EBcnAqwbz6lp6ZFCQH4hxzzZnu%2FJqELlFCuWdXd%2B6fMtzUrJktnim7ZCPeUx05xQmADLp5gM1XYK2Gt2wqA%3D%3D&Expires=1771967775) - def requireauth authorization str HeaderNone, - dict Simple Bearer token auth. Skips if adminsecrett...

