# KnowledgeVault Development Plan
Swarm: paid
Phase: 6 [IN PROGRESS] | Updated: 2026-02-10

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
- [x] 3.2: Implement Database Connection Pooling (Remove new connection per request) [SMALL]
- [x] 3.3: Standardize Async/Sync boundaries (Fix blocking I/O) [SMALL]

---
## Phase 4: Refactoring - Batch 3: Frontend Modernization [COMPLETE]
- [x] 4.1: Decompose Monolithic App.tsx into Page Components [SMALL]
- [x] 4.2: Extract Custom Hooks (useDebounce, useDocuments, etc.) [SMALL]
- [x] 4.3: Optimize Rendering (React.memo, useMemo) [SMALL]
- [x] 4.4: Fix Memory Search Debounce & Chat History Persistence [SMALL]

---
## Phase 5: Validation & Cleanup [COMPLETE]
- [x] 5.1: Add Authentication/Authorization (Basic Auth or Token) [SMALL]
- [x] 5.2: Comprehensive Regression Testing [SMALL] — 129/129 tests passing
- [x] 5.3: Final Codebase Polish (Docstrings, Type Hints) [SMALL] — Pydantic V2, datetime.UTC, 0 warnings

---
## Phase 6: Multi-Vault Support [COMPLETE]

### Design Decisions
- Vector store: Single LanceDB "chunks" table + vault_id column + filter_expr (already supported)
- Memories: Hybrid — vault_id nullable; NULL = global, set = vault-scoped
- Chat sessions: Vault-scoped via vault_id FK
- Default behavior: Omitted vault_id → default vault (id=1) for backward compat
- Migration: Create "Default" vault (id=1), backfill existing rows, THEN add FK constraints
- Vault names: UNIQUE, flat (no hierarchy)
- Vault deletion: Cascade cleanup — delete files, chunks, sessions; reassign memories to global
- Security: Single-user app, vault isolation is per-vault filtering (not per-user ACL)

### 6.1: Backend — Schema & Migration [COMPLETE]
- [x] 6.1.1: Add `vaults` table + `vault_id` columns to files, memories, chat_sessions [SMALL]
- [x] 6.1.2: Add `vault_id` column to LanceDB chunks schema [SMALL]
- [x] 6.1.3: Migration logic — create Default vault, backfill existing data, re-index chunks [SMALL]

### 6.2: Backend — Vault CRUD API [COMPLETE]
- [x] 6.2.1: Create vault routes (GET/POST/PUT/DELETE /api/vaults) with cascade delete [SMALL]
- [x] 6.2.2: Wire vault routes into main app router [SMALL]

### 6.3: Backend — Vault-Scoped Services [COMPLETE]
- [x] 6.3.1: Update VectorStore — vault_id in add_chunks + filter_expr in search [SMALL] (depends: 6.1.2)
- [x] 6.3.2: Update DocumentProcessor — accept vault_id, pass to file record + chunks [SMALL] (depends: 6.3.1)
- [x] 6.3.3: Update RAGEngine.query() — accept vault_id, pass to vector search + memory search [SMALL] (depends: 6.3.1)
- [x] 6.3.4: Update MemoryStore — add optional vault_id to add/search [SMALL] (depends: 6.1.1)

### 6.4: Backend — Vault-Scoped Routes [COMPLETE]
- [x] 6.4.1: Update chat routes — add vault_id to ChatRequest/ChatStreamRequest [SMALL] (depends: 6.3.3)
- [x] 6.4.2: Update document routes — add vault_id to list/upload/delete [SMALL] (depends: 6.3.2)
- [x] 6.4.3: Update memory routes — add vault_id to create/search/list [SMALL] (depends: 6.3.4)
- [x] 6.4.4: Update search routes — add vault_id to search endpoint [SMALL] (depends: 6.3.3)

### 6.5: Backend — Tests [COMPLETE]
- [x] 6.5.1: Add vault CRUD + isolation tests [SMALL] (depends: 6.2) — 30 tests (CRUD, cascade, edge cases)
- [x] 6.5.2: Add vault-scoped chat/document/search tests [SMALL] (depends: 6.4) — 12 tests (route filtering, passthrough)
- [x] 6.5.3: Verify existing 129 tests still pass (backward compat) [SMALL] (depends: 6.4) — 171/171 passing

### 6.6: Frontend — Vault UI [COMPLETE]
- [x] 6.6.1: Add vault API functions + types to api.ts [SMALL] (depends: 6.2)
- [x] 6.6.2: Create useVaults hook + vault context with localStorage persistence [SMALL]
- [x] 6.6.3: Add VaultSelector component (dropdown on ChatPage + DocumentsPage) [SMALL]
- [x] 6.6.4: Add VaultsPage for vault management (create/edit/delete) [SMALL]
- [x] 6.6.5: Update NavigationRail — add Vaults nav item [SMALL]
- [x] 6.6.6: Update ChatPage — wire vault_id into chat requests [SMALL] (depends: 6.6.2, 6.6.3)
- [x] 6.6.7: Update DocumentsPage — filter by vault, upload to vault [SMALL] (depends: 6.6.2, 6.6.3)
- [x] 6.6.8: Verify frontend build passes [SMALL]
