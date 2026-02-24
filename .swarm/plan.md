<!-- PLAN_HASH: paaagtvikzyc -->
# Refactoring Mission Plan
Swarm: mega
Phase: 6 [IN PROGRESS] | Updated: 2026-02-23

---
## Phase 1: Discovery & Audit [COMPLETE]
- [x] 1.1: Backend Deep Scan (Security, Performance, Architecture) [SMALL]
- [x] 1.2: Frontend Deep Scan (Performance, Re [SMALL]
- [x] 1.3: Generate Refactoring Roadmap [SMALL]

---
## Phase 2: Refactoring - Batch 1: Critical Security (Backend) [COMPLETE]
- [x] 2.1: Fix Path Traversal in File Upload (backend/app/api/routes/documents.py) [SMALL]
- [x] 2.2: Add File Size & Type Validation (backend/app/api/routes/documents.py) [SMALL]
- [x] 2.3: Fix Unsafe Query Parameter in Memory Search (backend/app/services/memory_store.py) [SMALL]
- [x] 2.4: Remove Duplicate Health Check Endpoints (backend/app/main.py) [SMALL]
- [x] 2.5: Fix Global HTTP Client Leak (backend/app/services/llm_client.py) [SMALL]

---
## Phase 3: Refactoring - Batch 2: Core Architecture (Backend) [COMPLETE]
- [x] 3.1: Implement Dependency Injection for Services (Remove Global Singletons) [SMALL]
  - COMMIT: e19c7de
  - FILE: backend/app/services/llm_health.py, backend/app/main.py
  - CHANGE: LLMHealthChecker now accepts embedding_service and llm_client via constructor instead of creating instances
  - NOTE: Full DI infrastructure already in place (deps.py, app.state), removed last singleton violation
  
- [N/A] 3.2: Implement Database Connection Pooling [SKIP]
  - DECISION: Keep SQLite connection-per-request pattern (SME confirmed 2026-02-11)
  - RATIONALE: SQLite connections are cheap; true connection pool adds complexity for embedded DB
  
- [N/A] 3.3: Standardize Async/Sync boundaries [SKIP]
  - STATUS: Already standardized
  - All blocking SQLite operations use asyncio.to_thread
  - Both sync (time.sleep) and async (asyncio.sleep) retry decorators available
  - No blocking I/O found in async paths

---
## Phase 4: Refactoring - Batch 3: Frontend Modernization [PENDING]
- [ ] 4.1: Decompose Monolithic App.tsx into Page Components [SMALL]
- [ ] 4.2: Extract Custom Hooks (useDebounce, useDocuments, etc.) [SMALL]
- [ ] 4.3: optimize Re [SMALL]
- [ ] 4.4: Fix Chat History Loading Loop [SMALL]

---
## Phase 5: Validation & Cleanup [PENDING]
- [ ] 5.1: Add Authentication/Authorization (Basic Auth or Token) [SMALL]
- [ ] 5.2: Comprehensive Regression Testing [SMALL]
- [ ] 5.3: Final Codebase Polish (Docstrings, Type Hints) [SMALL]

---
## Phase 6: RAG Best-Practice Alignment [IN PROGRESS]
Based on comprehensive spec for eliminating UI/backend semantic drift and modernizing RAG pipeline.

### Sprint 1: Settings & Chunking Unit Fix [CRITICAL]
- [x] 6.1.1: Add new character-based settings (chunk_size_chars, chunk_overlap_chars) [SMALL]
  - STATUS: Already implemented in backend/config.py with validators
  
- [x] 6.1.2: Add retrieval_top_k to unify max_context_chunks/vector_top_k [SMALL]
  - STATUS: Already implemented in backend/config.py
  
- [x] 6.1.3: Add settings migration layer (legacy chunk_size → chunk_size_chars*4) [SMALL]
  - STATUS: Field validators in config.py handle migration
  
- [x] 6.1.4: Remove *4 scaling in chunking.py [SMALL]
  - STATUS: chunking.py already uses chunk_size_chars directly
  
- [x] 6.1.5: Update DocumentProcessor to store chunk metadata (file_id, chunk_index, chunk_uid) [SMALL]
  - STATUS: Already storing file_id, chunk_index, chunk_uid in vector store
  
- [ ] 6.1.6: Update frontend to use new field names (chunk_size_chars, chunk_overlap_chars, retrieval_top_k, max_distance_threshold) [MEDIUM]
  - FILE: frontend/src/lib/api.ts, frontend/src/stores/useSettingsStore.ts, frontend/src/pages/SettingsPage.tsx
  - CHANGE: Update SettingsResponse interface to use new field names
  - CHANGE: Update useSettingsStore to use new field names
  - CHANGE: Update SettingsPage form labels and mappings
  - RATIONALE: Frontend currently using legacy names causing semantic drift

### Sprint 2: Retrieval & Threshold Fix [CRITICAL]
- [ ] 6.2.1: Fix vector_store to expose _distance field from LanceDB [SMALL]
- [ ] 6.2.2: Fix RAG engine threshold logic (use _distance not score) [MEDIUM]
- [ ] 6.2.3: Change threshold to max_distance_threshold (lower=better) [SMALL]
- [ ] 6.2.4: Add per-query logging (initial_hits, filtered_count, distance stats) [SMALL]

### Sprint 3: Embeddings & Windowing [HIGH]
- [ ] 6.3.1: Add embedding instruction prefixes (doc_prefix, query_prefix) [SMALL]
- [ ] 6.3.2: Add dimension/schema validation at startup [SMALL]
- [ ] 6.3.3: Implement adjacent-chunk windowing (fetch N±1 chunks) [MEDIUM]
- [ ] 6.3.4: Add deduplication and capping for windowed results [SMALL]
- [x] 6.3.5: Add adaptive embedding batching for low VRAM llama.cpp limits [SMALL]
  - depends: none (hotfix independent of 6.3.1/6.3.2)
  - acceptance: catch OpenAI-mode HTTP 500 messages containing "too large to process" and "current batch size"
  - acceptance: retry by halving request batch length with bounded backoff until success or single-input case
  - acceptance: if a single input still overflows, raise actionable EmbeddingError (reduce chunk_size_chars or increase server batch)
  - acceptance: preserve output ordering and one-embedding-per-input for successful requests
  - acceptance: add unit tests for overflow split behavior and single-input overflow failure path
- [x] 6.3.6: Expose embedding_batch_size in /api/settings for UI tuning [SMALL]
  - acceptance: GET /api/settings returns embedding_batch_size
  - acceptance: POST/PUT /api/settings accepts and persists embedding_batch_size
  - acceptance: backend settings tests cover read and update of embedding_batch_size
- [x] 6.3.7: Fix frontend upload queue stall after first file [SMALL]
  - acceptance: multiple queued uploads process sequentially until no pending files remain
  - acceptance: queue continues after per-file error and does not deadlock in isProcessing=true
- [x] 6.3.8: Reduce pre-embedding document parsing latency [SMALL]
  - acceptance: default parser strategy avoids unconditional hi_res for all docs
  - acceptance: first-GPU-use delay significantly reduced for text PDFs
  - acceptance: retain configurable path for higher-quality parsing when needed
- [x] 6.3.9: Handle single-input embedding overflow via adaptive split+pool fallback [SMALL]
  - acceptance: when len(texts)==1 overflows, split text on boundary-aware midpoint and retry sub-embeddings instead of immediate 500
  - acceptance: preserve one-embedding-per-input contract by mean-pooling split embeddings
  - acceptance: define MIN_SPLIT_CHARS guard; if text cannot be split meaningfully, raise actionable EmbeddingError
  - acceptance: keep existing prefix behavior consistent for split sub-inputs
  - acceptance: update existing single-overflow test + add hard-failure boundary-case test

### Sprint 4: Code Boundary Safety [MEDIUM]
- [ ] 6.4.1: Detect and prevent splitting inside fenced code blocks [MEDIUM]
- [ ] 6.4.2: Detect and prevent splitting inside markdown tables [MEDIUM]

### Sprint 5: Tests & Observability [HIGH]
- [ ] 6.5.1: Add unit tests for settings migration [SMALL]
- [ ] 6.5.2: Add unit tests for threshold filtering [SMALL]
- [ ] 6.5.3: Add unit tests for window expansion [SMALL]
- [ ] 6.5.4: Add integration test for RAG pipeline with tech docs [MEDIUM]

### Sprint 7: UX Polish [LOW]
- [ ] 6.7.1: Resizable Filename column in DocumentsPage table [SMALL]
  - acceptance: drag handle on right edge of Filename `<th>` changes column width
  - acceptance: min width 120px, max 600px, default 250px
  - acceptance: filename `title` tooltip shows full name on hover
  - acceptance: no other columns affected; no new npm packages; build passes

### Sprint 6: Bulk Document Operations [MEDIUM]
- [x] 6.6.1: Add backend endpoints for batch delete and vault delete-all [SMALL]
  - acceptance: DELETE /api/documents/batch accepts array of IDs and deletes all
  - acceptance: DELETE /api/documents/vault/{vault_id}/all deletes all docs in vault
- [x] 6.6.2: Add frontend bulk delete UI in DocumentsPage [MEDIUM]
  - acceptance: checkbox column in document table for multi-select
  - acceptance: "Delete Selected" button appears when items selected
  - acceptance: "Delete All" button to clear entire vault

### Sprint 8: RAG Retrieval Fix [CRITICAL]
- [x] 6.8.1: Fix max_distance_threshold default to enable relevance filtering [SMALL]
  - COMMIT: 924ba70
  - FILE: backend/app/config.py
  - CHANGE: max_distance_threshold: float = 0.5 (was Optional[float] = None)
  
- [x] 6.8.2: Fix fallback bug - remove garbage injection, inform LLM of no context [SMALL]
  - COMMIT: 924ba70
  - FILE: backend/app/services/rag_engine.py
  - CHANGE: Removed fallback injection block (was lines 119-130)
  - CHANGE: _build_messages now includes "No relevant documents found" when context is empty
  
- [x] 6.8.3: Add per-query retrieval logging for debugging [SMALL]
  - COMMIT: 924ba70
  - FILE: backend/app/services/rag_engine.py
  - CHANGE: Added vector search logging after results retrieved
  
- [x] 6.8.4: Fix vault_id defaults - consistent vault_id=1 across all endpoints [SMALL]
  - COMMIT: 924ba70
  - FILE: backend/app/api/routes/chat.py
  - CHANGE: ChatRequest, ChatStreamRequest, stream_chat_response, non_stream_chat_response all default to vault_id=1
  
- [x] 6.8.5: Display sources in chat UI [SMALL] [FRONTEND]
  - STATUS: Already implemented
  - FILE: frontend/src/components/shared/MessageContent.tsx
  - NOTE: SourcesList component was already present; backend sends sources in "done" event, frontend displays via MessageContent
  
- [x] 6.8.6: Update RAG engine tests for new threshold behavior [SMALL]
  - COMMIT: 394a9aa
  - FILE: backend/tests/test_rag_engine.py
  - CHANGE: Updated 4 tests to use _distance instead of score
  - CHANGE: Added test_filter_relevant_filters_by_distance_with_lancedb_results
  - CHANGE: Added test_no_fallback_injection_when_all_chunks_filtered
  - acceptance: All 12 tests pass
