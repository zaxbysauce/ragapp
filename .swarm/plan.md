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
## Phase 3: Refactoring - Batch 2: Core Architecture (Backend) [IN PROGRESS]
- [ ] 3.1: Implement Dependency Injection for Services (Remove Global Singletons) [SMALL]
- [ ] 3.2: Implement Database Connection Pooling (Remove new connection per request) [SMALL]
- [ ] 3.3: Standardize Async/Sync boundaries (Fix blocking I/O) [SMALL]

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
- [ ] 6.1.1: Add new character-based settings (chunk_size_chars, chunk_overlap_chars) [SMALL]
- [ ] 6.1.2: Add retrieval_top_k to unify max_context_chunks/vector_top_k [SMALL]
- [ ] 6.1.3: Add settings migration layer (legacy chunk_size → chunk_size_chars*4) [SMALL]
- [ ] 6.1.4: Remove *4 scaling in chunking.py [SMALL]
- [ ] 6.1.5: Update DocumentProcessor to store chunk metadata (file_id, chunk_index, chunk_uid) [SMALL]
- [ ] 6.1.6: Update frontend SettingsPage labels and new fields [MEDIUM]

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

### Sprint 4: Code Boundary Safety [MEDIUM]
- [ ] 6.4.1: Detect and prevent splitting inside fenced code blocks [MEDIUM]
- [ ] 6.4.2: Detect and prevent splitting inside markdown tables [MEDIUM]

### Sprint 5: Tests & Observability [HIGH]
- [ ] 6.5.1: Add unit tests for settings migration [SMALL]
- [ ] 6.5.2: Add unit tests for threshold filtering [SMALL]
- [ ] 6.5.3: Add unit tests for window expansion [SMALL]
- [ ] 6.5.4: Add integration test for RAG pipeline with tech docs [MEDIUM]
