# KnowledgeVault — Bleeding-Edge RAG Improvements

**Swarm:** default
**Baseline:** Current main branch
**Goal:** Implement bleeding-edge RAG techniques to improve retrieval accuracy, reduce latency, and unlock unused capabilities in the existing BGE-M3 + BGE-Reranker stack

---

## Reference: Current Architecture

All tasks in this plan target the `backend/app/` directory tree of the KnowledgeVault RAG application. The application is Python/FastAPI with a React/TypeScript frontend, deployed via Docker.

| Component | Current State | Notes |
|-----------|--------------|-------|
| Embedding | BGE-M3 via OpenAI-compat endpoint (port 18080) | Dense vectors only; sparse + ColBERT unused |
| Reranker | BGE-Reranker-v2-M3 via TEI (port 8081) | Enabled, working |
| Vector DB | LanceDB with hybrid search (dense + BM25 FTS + RRF) | BM25 FTS is weaker than BGE-M3's learned sparse |
| Chunking | `unstructured` chunk_by_title, 1024 chars, 100 overlap | No document-level context prepended |
| Query | Single query → embed → search → rerank → LLM | No query transformation |
| Chat embedding | `embed_single()` creates new httpx client per call | No connection pooling |
| Hardware | 8GB VRAM GPU for embedding+reranking; separate GPU for chat LLM | Multiple GPUs available |

---

## Phase 1: Connection Pooling & Embedding Service Hardening [SMALL]

### Task 1.1 — Add persistent httpx client to EmbeddingService `SMALL`

**FILE:** `backend/app/services/embeddings.py`

**TASK:** Replace the per-call `httpx.AsyncClient` creation in `embed_single()` (and `embed_batch()` if applicable) with a persistent `httpx.AsyncClient` stored as an instance attribute on `EmbeddingService`. The client should be created in `__init__()` with the existing `self.timeout` value and connection pool limits (`max_connections=20`, `max_keepalive_connections=10`).

**CONSTRAINT:**
- Do NOT change the `_build_payload()` or `_extract_embedding()` methods
- Do NOT change any timeout values (keep 60.0s)
- Do NOT change provider detection logic
- The persistent client must use `httpx.AsyncClient(timeout=self.timeout, limits=httpx.Limits(max_connections=20, max_keepalive_connections=10))`
- Remove ALL `async with httpx.AsyncClient(...)` context managers in embedding methods and replace with `self._client`
- Add an `async def close(self)` method that calls `await self._client.aclose()`

**ACCEPTANCE:**
- `embed_single()` no longer creates a new httpx client per call
- `embed_batch()` (if it exists) no longer creates a new httpx client per call
- All existing embedding tests pass
- New test: `EmbeddingService` instance has a `_client` attribute that is an `httpx.AsyncClient`
- New test: `EmbeddingService.close()` method exists and is async

---

### Task 1.2 — Wire EmbeddingService lifecycle into FastAPI app startup/shutdown `SMALL`

**FILE:** `backend/app/main.py` (or wherever the FastAPI app lifecycle is defined)

**TASK:** Ensure the shared `EmbeddingService` instance created during app startup is properly closed during app shutdown by calling `await embedding_service.close()` in the shutdown handler.

**CONSTRAINT:**
- Do NOT change how the EmbeddingService is instantiated or stored on `app.state`
- Only add the shutdown cleanup call
- If a shutdown handler already exists, append to it; do not replace it

**ACCEPTANCE:**
- App shutdown handler calls `embedding_service.close()`
- Application starts and stops cleanly without resource warnings
- All existing tests pass

---

## Phase 2: Contextual Chunking (Anthropic-style Context Prepending) [MEDIUM]

### Task 2.1 — Create ContextualChunker service `MEDIUM`

**FILE:** `backend/app/services/contextual_chunking.py` (NEW FILE)

**TASK:** Create a new `ContextualChunker` class that takes a full document text and a list of `ProcessedChunk` objects (from `chunking.py`), then uses the chat LLM to generate a short context prefix for each chunk. The prefix situates the chunk within the overall document for improved retrieval.

The class should:
1. Accept an `LLMClient` instance (from `llm_client.py`) for making LLM calls
2. Implement `async def contextualize_chunks(self, document_text: str, chunks: List[ProcessedChunk], source_filename: str) -> List[ProcessedChunk]`
3. For each chunk, call the LLM with a prompt that includes the full document text and the chunk text, asking for a 1-2 sentence context prefix
4. Prepend the generated context to each chunk's `text` field, separated by `\n\n`
5. Add `contextualized: true` to each chunk's metadata dict
6. Process chunks concurrently using `asyncio.gather()` with a configurable concurrency semaphore (default 5)
7. If context generation fails for a chunk, log a warning and return the chunk unchanged (no context prefix)

**CONSTRAINT:**
- The LLM prompt must follow this template:
  ```
  <document>
  {DOCUMENT_TEXT}
  </document>
  Here is a chunk from the document "{SOURCE_FILENAME}":
  <chunk>
  {CHUNK_TEXT}
  </chunk>
  Give a short succinct context (1-2 sentences) to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
  ```
- Do NOT modify `chunking.py` or `ProcessedChunk` dataclass
- Import `ProcessedChunk` from `app.services.chunking`
- If the document text exceeds 100,000 characters, truncate to the first and last 50,000 characters with a `[...truncated...]` separator
- The class must be stateless (no instance caching of LLM responses)

**ACCEPTANCE:**
- New file `backend/app/services/contextual_chunking.py` exists
- `ContextualChunker` class instantiates with an `LLMClient`
- `contextualize_chunks()` returns chunks with context prepended to `.text`
- Chunks that fail context generation are returned unchanged with a log warning
- Concurrency is bounded by semaphore
- Unit test: mock LLM returns a context string → chunk.text starts with that context
- Unit test: LLM failure for one chunk does not crash the batch

---

### Task 2.2 — Add CONTEXTUAL_CHUNKING_ENABLED config option `SMALL`

**FILE:** `backend/app/config.py`

**TASK:** Add contextual chunking configuration fields to the `Settings` class.

```python
# ── Contextual chunking configuration ────────────────────────────────────
contextual_chunking_enabled: bool = False
"""Enable LLM-based contextual chunking (prepends document context to each chunk)."""
contextual_chunking_concurrency: int = 5
"""Maximum concurrent LLM calls for contextual chunking."""
```

**CONSTRAINT:**
- Add immediately after the existing `## Hybrid search configuration` section
- Follow the exact comment style used by other config sections (see `reranker_url`, `hybrid_search_enabled` for pattern)
- Environment variable names will be auto-derived: `CONTEXTUAL_CHUNKING_ENABLED`, `CONTEXTUAL_CHUNKING_CONCURRENCY`

**ACCEPTANCE:**
- `settings.contextual_chunking_enabled` defaults to `False`
- `settings.contextual_chunking_concurrency` defaults to `5`
- Both are overridable via environment variables
- All existing config tests pass

---

### Task 2.3 — Integrate ContextualChunker into document processing pipeline `MEDIUM`

**FILE:** `backend/app/services/document_processor.py`

**TASK:** After the existing chunking step (where `SemanticChunker.chunk_elements()` is called) and before the embedding step, insert a conditional call to `ContextualChunker.contextualize_chunks()` when `settings.contextual_chunking_enabled` is `True`.

**CONSTRAINT:**
- The `ContextualChunker` must be instantiated using the existing `LLMClient` instance (either from the processor's dependencies or by creating one)
- Guard with `if settings.contextual_chunking_enabled:` — when disabled, the pipeline must behave identically to before
- Pass `settings.contextual_chunking_concurrency` as the semaphore limit
- Log `"Contextual chunking: processing %d chunks for %s"` at INFO level before starting
- Log `"Contextual chunking: completed for %s (%d chunks contextualized)"` at INFO level after finishing
- Do NOT change the chunking step itself or the embedding step
- Do NOT change method signatures of existing public methods

**ACCEPTANCE:**
- With `CONTEXTUAL_CHUNKING_ENABLED=false`: pipeline behavior is identical to before (no LLM calls for context)
- With `CONTEXTUAL_CHUNKING_ENABLED=true`: chunks are contextualized before embedding
- Document upload with contextual chunking enabled produces chunks whose text starts with a context prefix
- All existing document processing tests pass
- New test: mock the ContextualChunker and verify it is called when enabled, not called when disabled

---

## Phase 3: Multi-Scale Chunk Indexing with RRF Aggregation [MEDIUM]

### Task 3.1 — Add multi-scale chunking config options `SMALL`

**FILE:** `backend/app/config.py`

**TASK:** Add multi-scale indexing configuration fields to the `Settings` class.

```python
# ── Multi-scale indexing configuration ───────────────────────────────────
multi_scale_indexing_enabled: bool = False
"""Index documents at multiple chunk sizes and aggregate results with RRF."""
multi_scale_chunk_sizes: str = "512,1024,2048"
"""Comma-separated list of character-based chunk sizes for multi-scale indexing."""
multi_scale_overlap_ratio: float = 0.1
"""Overlap ratio (fraction of chunk size) for each scale."""
```

**CONSTRAINT:**
- `multi_scale_chunk_sizes` is stored as a comma-separated string (Pydantic Settings reads env vars as strings)
- Add a `@field_validator` for `multi_scale_chunk_sizes` that validates each value is a positive integer when split by comma
- Add immediately after the contextual chunking config section
- Environment variables: `MULTI_SCALE_INDEXING_ENABLED`, `MULTI_SCALE_CHUNK_SIZES`, `MULTI_SCALE_OVERLAP_RATIO`

**ACCEPTANCE:**
- `settings.multi_scale_indexing_enabled` defaults to `False`
- `settings.multi_scale_chunk_sizes` defaults to `"512,1024,2048"`
- Validator rejects `"512,abc,2048"` with a clear error
- All existing config tests pass

---

### Task 3.2 — Add `chunk_scale` metadata field to vector store schema `SMALL`

**FILE:** `backend/app/services/vector_store.py`

**TASK:** Add a `chunk_scale` column (string type) to the LanceDB `chunks` table schema. This field stores the character-based chunk size used to create each chunk (e.g., `"1024"`), enabling scale-aware filtering at query time.

**CONSTRAINT:**
- Add `("chunk_scale", pa.string())` to the `pa.schema([...])` in `init_table()`
- Default value for `chunk_scale` in `add_chunks()` should be `record.get("chunk_scale", "default")`
- Do NOT change the `search()` method in this task (that's Task 3.4)
- Handle migration: if the `chunks` table already exists without `chunk_scale`, the existing data should still load (LanceDB handles missing columns gracefully for reads, but document this as a known migration consideration in a code comment)

**ACCEPTANCE:**
- New records include `chunk_scale` field
- Existing records without `chunk_scale` do not cause errors on read
- All existing vector store tests pass
- New test: `add_chunks()` with records containing `chunk_scale` stores and retrieves the value

---

### Task 3.3 — Implement multi-scale chunking in document processor `MEDIUM`

**FILE:** `backend/app/services/document_processor.py`

**TASK:** When `settings.multi_scale_indexing_enabled` is `True`, instead of chunking at a single size, chunk the document at each size specified in `settings.multi_scale_chunk_sizes`. Each chunk record must include a `chunk_scale` metadata field indicating which scale produced it. Chunk IDs must be unique across scales (e.g., `{file_id}_{scale}_{chunk_index}`).

**CONSTRAINT:**
- Parse `settings.multi_scale_chunk_sizes` by splitting on comma and converting to int list
- For each scale, create a new `SemanticChunker(chunk_size_chars=size, chunk_overlap_chars=int(size * settings.multi_scale_overlap_ratio))`
- Concatenate all chunks from all scales before passing to the embedding step
- Each chunk's `id` field must include the scale: `f"{file_id}_{scale}_{chunk_index}"`
- Each chunk's metadata must include `"chunk_scale": str(scale)`
- When multi-scale is disabled, behavior is identical to before (single-scale chunking)
- Log `"Multi-scale indexing: chunking at scales %s for %s"` at INFO level
- If contextual chunking is also enabled, apply contextual chunking to ALL scales' chunks

**ACCEPTANCE:**
- With `MULTI_SCALE_INDEXING_ENABLED=false`: single-scale chunking (unchanged behavior)
- With `MULTI_SCALE_INDEXING_ENABLED=true`: document produces chunks at each configured scale
- Chunk IDs are unique across scales
- Contextual chunking (if enabled) applies to multi-scale chunks
- All existing tests pass
- New test: mock chunker and verify it's called once per scale with correct parameters

---

### Task 3.4 — Implement multi-scale RRF aggregation at query time `MEDIUM`

**FILE:** `backend/app/services/vector_store.py`

**TASK:** When `settings.multi_scale_indexing_enabled` is `True`, modify the `search()` method to aggregate results across scales using RRF. Instead of a single search, run the same vector search but treat results from different `chunk_scale` values as separate ranked lists, then fuse with RRF.

**CONSTRAINT:**
- Do NOT change the search behavior when `multi_scale_indexing_enabled` is `False`
- The RRF aggregation must use the same `k_rrf = 60` constant already used for hybrid search fusion
- If multi-scale AND hybrid search are both enabled, the fusion order is: (1) for each scale, fuse dense + FTS via RRF; (2) then fuse across scales via RRF. This produces a two-level RRF.
- Add `multi_scale_indexing_enabled` as a parameter to `search()` (default `False`) so it can be passed from the RAG engine
- After RRF, deduplicate chunks that overlap significantly (same `file_id` and overlapping `chunk_index` ranges) — keep the highest-scoring variant
- Return the standard `limit` number of results after dedup

**ACCEPTANCE:**
- With multi-scale disabled: search behavior unchanged
- With multi-scale enabled: results are aggregated across scales
- Deduplication removes near-duplicate chunks from different scales
- All existing search tests pass
- New test: insert chunks at two scales, search, verify results contain chunks from both scales fused by RRF

---

## Phase 4: Query Transformation Pipeline [MEDIUM]

### Task 4.1 — Create QueryTransformer service `MEDIUM`

**FILE:** `backend/app/services/query_transformer.py` (NEW FILE)

**TASK:** Create a `QueryTransformer` class that implements step-back prompting and query expansion. Given a user query, it generates one or more reformulated queries to improve retrieval coverage.

The class should:
1. Accept an `LLMClient` instance
2. Implement `async def transform(self, query: str) -> List[str]` that returns a list of queries (always including the original)
3. Use the LLM to generate a "step-back" query — a broader, more general version of the input
4. Return `[original_query, stepback_query]` — always exactly 2 queries
5. If LLM call fails, log a warning and return `[original_query]` (graceful fallback)

**CONSTRAINT:**
- Step-back prompt template:
  ```
  You are an expert at reformulating search queries for better retrieval.
  Given the user's question, generate a single broader "step-back" version that captures the high-level topic.
  
  User question: {QUERY}
  
  Respond with ONLY the step-back question, nothing else.
  ```
- Do NOT implement query decomposition, HyDE, or RAG-Fusion in this task — only step-back
- The class must be stateless
- Maximum LLM response length: 200 tokens
- Timeout: use the LLMClient's existing timeout

**ACCEPTANCE:**
- New file `backend/app/services/query_transformer.py` exists
- `transform("What is the refresh rate of iPhone 13 Pro Max?")` returns a list with 2 items: the original query and a broader step-back query
- LLM failure returns `[original_query]` with a log warning
- Unit test: mock LLM → verify 2-item list with original + step-back
- Unit test: LLM failure → verify 1-item list with original only

---

### Task 4.2 — Add query transformation config option `SMALL`

**FILE:** `backend/app/config.py`

**TASK:** Add query transformation configuration to the `Settings` class.

```python
# ── Query transformation configuration ───────────────────────────────────
query_transformation_enabled: bool = False
"""Enable LLM-based step-back query transformation for improved retrieval."""
```

**CONSTRAINT:**
- Add after the multi-scale indexing config section
- Environment variable: `QUERY_TRANSFORMATION_ENABLED`

**ACCEPTANCE:**
- `settings.query_transformation_enabled` defaults to `False`
- Overridable via environment variable
- All existing config tests pass

---

### Task 4.3 — Integrate QueryTransformer into RAG engine `MEDIUM`

**FILE:** `backend/app/services/rag_engine.py`

**TASK:** When `settings.query_transformation_enabled` is `True`, use `QueryTransformer` to generate multiple queries before the embedding/search step. Embed each query, search with each, and fuse results using RRF before passing to reranking.

**CONSTRAINT:**
- Insert the transformation step in `query()` between the memory intent detection and the `embed_single()` call
- If transformation produces N queries, embed each one via `embed_single()`
- For each query embedding, call `vector_store.search()` with the same parameters
- Fuse the N result lists using RRF (same `k_rrf = 60`) before applying distance threshold filtering
- After RRF fusion, the remaining pipeline (reranking, filtering, LLM call) proceeds unchanged
- When disabled: behavior is identical to current (single query path)
- Log `"Query transformation: original='%s', step_back='%s'"` at DEBUG level
- The `QueryTransformer` should be instantiated using the RAG engine's existing `llm_client`
- Do NOT modify the `__init__` signature of RAGEngine — create QueryTransformer lazily on first use

**ACCEPTANCE:**
- With `QUERY_TRANSFORMATION_ENABLED=false`: behavior unchanged
- With `QUERY_TRANSFORMATION_ENABLED=true`: step-back query is generated and used alongside original
- Results from both queries are fused before reranking
- All existing RAG engine tests pass
- New test: mock QueryTransformer returning 2 queries → verify search is called twice and results are fused

---

## Phase 5: Corrective RAG Self-Evaluation Gate [SMALL]

### Task 5.1 — Create RetrievalEvaluator service `SMALL`

**FILE:** `backend/app/services/retrieval_evaluator.py` (NEW FILE)

**TASK:** Create a `RetrievalEvaluator` class that asks the LLM to assess whether retrieved chunks are actually relevant to the query. This is a lightweight CRAG-style self-evaluation.

The class should:
1. Accept an `LLMClient` instance
2. Implement `async def evaluate(self, query: str, chunks: List[Dict[str, Any]]) -> str` that returns one of: `"CONFIDENT"`, `"AMBIGUOUS"`, `"NO_MATCH"`
3. Send the query and the text of the top 3 chunks to the LLM with an evaluation prompt
4. Parse the LLM response (which should be exactly one of the three labels)
5. If parsing fails or LLM errors, default to `"CONFIDENT"` (fail-open for safety)

**CONSTRAINT:**
- Evaluation prompt template:
  ```
  You are a retrieval quality evaluator. Given a user query and retrieved document chunks, assess whether the chunks contain information relevant to answering the query.
  
  Query: {QUERY}
  
  Retrieved chunks:
  {CHUNK_1_TEXT}
  ---
  {CHUNK_2_TEXT}
  ---
  {CHUNK_3_TEXT}
  
  Respond with exactly one word:
  CONFIDENT - if the chunks clearly contain relevant information to answer the query
  AMBIGUOUS - if the chunks are partially relevant but may not fully answer the query
  NO_MATCH - if the chunks do not contain information relevant to the query
  ```
- Only evaluate the top 3 chunks (by score) to keep LLM calls cheap
- Do NOT implement web search fallback or re-retrieval in this task — only the evaluation
- Maximum LLM response length: 10 tokens
- Truncate each chunk text to 500 characters in the evaluation prompt

**ACCEPTANCE:**
- New file `backend/app/services/retrieval_evaluator.py` exists
- `evaluate()` returns one of the three valid labels
- Invalid LLM response defaults to `"CONFIDENT"`
- LLM failure defaults to `"CONFIDENT"`
- Unit test: mock LLM returning `"NO_MATCH"` → verify return value
- Unit test: mock LLM returning gibberish → verify default to `"CONFIDENT"`

---

### Task 5.2 — Add retrieval evaluation config and integrate into RAG engine `SMALL`

**FILE:** `backend/app/config.py` and `backend/app/services/rag_engine.py`

**TASK:** Add `retrieval_evaluation_enabled: bool = False` config option. In the RAG engine, after reranking but before building the LLM prompt, call `RetrievalEvaluator.evaluate()`. If the result is `"NO_MATCH"`, add a hint to the LLM prompt: `"Note: The retrieved documents may not be directly relevant to this query. Use your best judgment and indicate if you cannot answer from the provided context."`

**CONSTRAINT:**
- Config: add `retrieval_evaluation_enabled: bool = False` with docstring `"""Enable LLM-based retrieval quality evaluation (CRAG-style)."""`
- In RAG engine: insert evaluation after the reranking step and before `_build_messages()`
- When evaluation returns `"NO_MATCH"`, prepend the hint to the context section in `_build_messages()`
- When evaluation returns `"AMBIGUOUS"`, log at WARNING level but do not modify the prompt
- When evaluation returns `"CONFIDENT"`, do nothing
- When disabled: behavior unchanged
- Log the evaluation result at INFO level: `"Retrieval evaluation: %s for query '%s'"`
- The `RetrievalEvaluator` should use the RAG engine's existing `llm_client`

**ACCEPTANCE:**
- With `RETRIEVAL_EVALUATION_ENABLED=false`: behavior unchanged
- With enabled + `"NO_MATCH"`: LLM prompt includes the "may not be directly relevant" hint
- With enabled + `"CONFIDENT"`: LLM prompt is unchanged
- All existing RAG engine tests pass
- New test: verify hint is injected on NO_MATCH, not injected on CONFIDENT

---

## Phase 6: BGE-M3 Tri-Vector Serving (Dense + Sparse + ColBERT) [LARGE]

> **NOTE:** This phase is the most impactful but also the most architecturally significant change. It replaces the current Ollama/OpenAI-compat embedding endpoint with a FlagEmbedding-based server that returns all three BGE-M3 vector types. This should be implemented LAST and tested thoroughly.

### Task 6.1 — Create FlagEmbedding server script `MEDIUM`

**FILE:** `backend/embedding_server/server.py` (NEW FILE in new directory)

**TASK:** Create a standalone FastAPI server that loads BGE-M3 via FlagEmbedding and exposes a `/embed` endpoint returning dense, sparse, and ColBERT vectors in a single response. This server replaces the current embedding endpoint (port 18080).

The server should:
1. Load `BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)` on startup
2. Expose `POST /embed` accepting `{"texts": [str], "return_dense": true, "return_sparse": true, "return_colbert": false}` — ColBERT defaults to false for backward compat
3. Return `{"dense": [[float]], "sparse": [{"token_id": weight}], "colbert": [[[float]]] | null}`
4. Expose `GET /health` returning `{"status": "ok", "model": "BAAI/bge-m3"}`
5. Support batch processing with configurable max batch size (env var `MAX_BATCH_SIZE`, default 64)
6. Use `uvicorn` for serving

**CONSTRAINT:**
- This is a STANDALONE server, not part of the main KnowledgeVault FastAPI app
- It must run in its own Docker container with GPU access
- Do NOT install sentence-transformers in this container — use FlagEmbedding only
- The server must also expose the OpenAI-compatible `/v1/embeddings` endpoint (dense only) for backward compatibility during migration
- Add a `Dockerfile` at `backend/embedding_server/Dockerfile`
- Add a `requirements.txt` at `backend/embedding_server/requirements.txt` with: `fastapi`, `uvicorn`, `FlagEmbedding`, `torch`

**ACCEPTANCE:**
- `POST /embed` with `{"texts": ["hello world"], "return_dense": true, "return_sparse": true}` returns dense (1024-dim) and sparse vectors
- `POST /v1/embeddings` with `{"model": "bge-m3", "input": "hello world"}` returns OpenAI-format dense embeddings (backward compat)
- `GET /health` returns 200 with model info
- Dockerfile builds and runs with GPU access
- Server loads model in under 30 seconds

---

### Task 6.2 — Update EmbeddingService to support tri-vector responses `MEDIUM`

**FILE:** `backend/app/services/embeddings.py`

**TASK:** Add a new method `async def embed_multi(self, texts: List[str], return_sparse: bool = True, return_colbert: bool = False) -> Dict[str, Any]` that calls the new `/embed` endpoint and returns all requested vector types. Keep existing `embed_single()` and `embed_batch()` working via the OpenAI-compat path for backward compatibility.

**CONSTRAINT:**
- `embed_multi()` should call `{base_url}/embed` (auto-detected: if the base URL serves the new FlagEmbedding server, use `/embed`; otherwise fall back to the existing provider mode)
- Return format: `{"dense": [[float]], "sparse": [dict], "colbert": [list] | None}`
- Add a `_detect_flag_embedding_server()` method that does a `GET /health` probe to check if the endpoint is the new FlagEmbedding server
- Cache the detection result as `self._is_flag_embedding_server: bool`
- When the server is NOT a FlagEmbedding server, `embed_multi()` should fall back to dense-only via existing path and return `{"dense": [...], "sparse": None, "colbert": None}`
- Do NOT change `embed_single()` behavior — it always returns a dense vector list
- Use the persistent httpx client from Task 1.1

**ACCEPTANCE:**
- `embed_multi()` returns tri-vector response from FlagEmbedding server
- `embed_multi()` falls back to dense-only from non-FlagEmbedding servers
- `embed_single()` behavior unchanged
- All existing embedding tests pass
- New test: mock `/embed` endpoint → verify tri-vector response parsing
- New test: mock `/health` returning non-FlagEmbedding → verify fallback

---

### Task 6.3 — Store and search with learned sparse vectors `MEDIUM`

**FILE:** `backend/app/services/vector_store.py`

**TASK:** Add support for storing BGE-M3's learned sparse vectors alongside dense vectors, and use them in hybrid search instead of (or in addition to) BM25 FTS.

**CONSTRAINT:**
- Add an optional `sparse_embedding` column to the schema (JSON string of `{token_id: weight}` dict)
- In `add_chunks()`: accept optional `sparse_embedding` field in records
- In `search()`: when sparse vectors are available for the query, compute sparse similarity scores and fuse with dense via RRF (replacing or supplementing BM25 FTS)
- If sparse vectors are not available (old records or non-FlagEmbedding server), fall back to BM25 FTS
- The sparse search should use inverted-index-style scoring: for each document's sparse vector, compute dot product with query sparse vector
- This is computationally expensive at scale — add a config option `sparse_search_max_candidates: int = 1000` to limit the candidate set
- Do NOT remove BM25 FTS support — it should remain as a fallback

**ACCEPTANCE:**
- Records with `sparse_embedding` store and retrieve correctly
- Hybrid search uses learned sparse when available, BM25 FTS as fallback
- All existing vector store tests pass
- New test: insert records with sparse embeddings, search with sparse query, verify results differ from dense-only

---

### Task 6.4 — Integrate tri-vector embedding into document processing pipeline `SMALL`

**FILE:** `backend/app/services/document_processor.py`

**TASK:** When the embedding server supports tri-vector output (detected by `EmbeddingService._is_flag_embedding_server`), use `embed_multi()` instead of `embed_batch()` during document processing, and pass the sparse vectors to `vector_store.add_chunks()`.

**CONSTRAINT:**
- If FlagEmbedding server is detected, call `embed_multi(texts, return_sparse=True, return_colbert=False)` — ColBERT is NOT stored in LanceDB (too expensive for storage; it's only used if a ColBERT reranker is added later)
- Map sparse vectors from the response into each chunk record's `sparse_embedding` field
- If FlagEmbedding server is NOT detected, use existing `embed_batch()` path (no sparse vectors)
- Log `"Using tri-vector embedding (dense + sparse)"` at INFO level when FlagEmbedding is detected
- Do NOT change the reranking step — BGE-Reranker-v2-M3 via TEI is still the reranker

**ACCEPTANCE:**
- With FlagEmbedding server: chunks are stored with both dense and sparse embeddings
- Without FlagEmbedding server: behavior unchanged (dense only)
- All existing document processing tests pass
- New test: mock `embed_multi()` → verify sparse vectors are passed to `add_chunks()`

---

### Task 6.5 — Update docker-compose with FlagEmbedding server service `SMALL`

**FILE:** `docker-compose.yml` (or `docker-compose.override.yml`)

**TASK:** Add a new service `embedding-server` that builds from `backend/embedding_server/Dockerfile` and runs the FlagEmbedding server on port 18080 (replacing the current embedding service).

**CONSTRAINT:**
- Service name: `embedding-server`
- Build context: `backend/embedding_server`
- Expose port 18080
- GPU access via `deploy.resources.reservations.devices` with `capabilities: [gpu]`
- Environment variables: `MAX_BATCH_SIZE=64`
- Health check: `curl -f http://localhost:18080/health`
- The existing backend service's `OLLAMA_EMBEDDING_URL` should point to `http://embedding-server:18080/v1/embeddings` for backward compat
- Add a comment explaining that the new server also supports `/embed` for tri-vector output

**ACCEPTANCE:**
- `docker compose up embedding-server` starts the FlagEmbedding server
- Backend can reach the embedding server at the configured URL
- Health check passes
- Existing embedding flow works via `/v1/embeddings` endpoint

---

## Dependencies

```
Phase 1 → all other phases (connection pooling must be in place first)
1.1 → 1.2 (client must exist before lifecycle wiring)
2.1 → 2.3 (ContextualChunker must exist before integration)
2.2 → 2.3 (config must exist before integration checks it)
3.1 → 3.3 (config must exist before processor uses it)
3.2 → 3.3, 3.4 (schema must include chunk_scale before it's written/queried)
3.3 → 3.4 (chunks must be stored at multiple scales before search aggregates them)
4.1 → 4.3 (QueryTransformer must exist before RAG engine uses it)
4.2 → 4.3 (config must exist before RAG engine checks it)
5.1 → 5.2 (RetrievalEvaluator must exist before integration)
6.1 → 6.2 (server must exist before client supports it)
6.2 → 6.3, 6.4 (multi-vector client must exist before storage/pipeline changes)
6.3 → 6.4 (storage must support sparse before pipeline writes sparse)
6.4 → 6.5 (pipeline integration before Docker orchestration)
```

---

## Final Acceptance Criteria

- [ ] `embed_single()` uses persistent httpx client (no per-call allocation)
- [ ] App shutdown cleanly closes embedding client
- [ ] `CONTEXTUAL_CHUNKING_ENABLED=true` prepends LLM-generated context to chunks before embedding
- [ ] Contextual chunking gracefully handles LLM failures per-chunk
- [ ] `MULTI_SCALE_INDEXING_ENABLED=true` indexes documents at multiple chunk sizes
- [ ] Multi-scale search aggregates results across scales via RRF
- [ ] Multi-scale chunks have unique IDs and `chunk_scale` metadata
- [ ] `QUERY_TRANSFORMATION_ENABLED=true` generates step-back queries and fuses results
- [ ] Query transformation gracefully falls back to single query on LLM failure
- [ ] `RETRIEVAL_EVALUATION_ENABLED=true` evaluates retrieval quality and injects hints on NO_MATCH
- [ ] Retrieval evaluation defaults to CONFIDENT on any error (fail-open)
- [ ] FlagEmbedding server serves dense + sparse + ColBERT vectors via `/embed`
- [ ] FlagEmbedding server maintains backward-compat via `/v1/embeddings`
- [ ] `embed_multi()` auto-detects FlagEmbedding server and returns tri-vector response
- [ ] Document processor stores sparse vectors when FlagEmbedding server is available
- [ ] Hybrid search uses learned sparse vectors (when available) instead of BM25 FTS
- [ ] All new features are gated behind config flags (disabled by default)
- [ ] All existing tests pass with new features disabled
- [ ] Docker compose includes FlagEmbedding server service definition
- [ ] Zero hardcoded model names — all configurable via environment variables
