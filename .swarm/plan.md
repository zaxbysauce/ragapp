# KnowledgeVault Code Review Remediation Plan
Swarm: paid
Phase: 6 [COMPLETE] | Updated: 2026-02-11

## Rollback Strategy
- Git commit after each phase; revert to prior commit if app fails to start
- Phase 2 (highest risk): commit before AND after each singleton removal; test suite gates each step
- Frontend: npm run build gates each task; git revert if build breaks

---

## Phase 1: Bug Fixes (CRITICAL + HIGH) [COMPLETE]

- [x] 1.1: Fix `exempt_when_health_check` no-op decorator (backend/app/limiter.py) [SMALL]
  - Replaced broken decorator with WhitelistLimiter._check_request_limit override + hmac.compare_digest

- [x] 1.2: Fix sync/async mismatch in LLMHealthChecker (backend/app/services/llm_client.py:32) [SMALL]
  - Changed `def start()` to `async def start()`; updated caller in main.py

- [x] 1.3: Fix health endpoint per-request instantiation (backend/app/api/routes/health.py) [SMALL]
  - Added to app.state + deps.py; health.py now uses Depends()

- [CANCELLED] 1.4: CSRF store token validation — Not broken (presence-based scheme, token is in key not value)

- [x] 1.5: Fix deprecated `datetime.utcnow()` + timing attack in admin.py [SMALL]
  - Changed to datetime.now(timezone.utc); added secrets.compare_digest for token comparison

- [x] 1.6: Phase 1 verification — 171/171 tests passing

---

## Phase 2: DI & Connection Consolidation [COMPLETE]
Order: Update deps.py first (2.1), then refactor services in dependency order (leaf services first), verify after each.

- [x] 2.1: Remove module-level singletons + update deps.py [MEDIUM] (depends: 1.2, 1.3)
  - Removed dead singletons from llm_client.py, model_checker.py, llm_health.py
  - background_tasks.py kept (intentional factory pattern)

- [x] 2.2: Refactor toggle_manager to accept DB via DI (backend/app/services/toggle_manager.py) [SMALL]
  - Constructor now accepts SQLiteConnectionPool; uses pool.get_connection()/release_connection()
  - main.py updated to pass app.state.db_pool

- [x] 2.3: Refactor maintenance service to accept DB via DI (backend/app/services/maintenance.py) [SMALL]
  - Constructor accepts SQLiteConnectionPool; updated test_maintenance.py

- [x] 2.4: Refactor memory_store to accept DB via DI (backend/app/services/memory_store.py) [SMALL]
  - Optional pool with fallback for backward compat; updated test_memory_store.py + test_api_routes.py

- [x] 2.5: Refactor document_processor to accept DB via DI (backend/app/services/document_processor.py) [SMALL]
  - Optional pool with fallback; updated background_tasks.py, documents.py, deps.py (new get_db_pool), test files

- [x] 2.6: Refactor file_watcher to accept DB via DI (backend/app/services/file_watcher.py) [SMALL]
  - Optional pool with lazy init; updated main.py, documents.py scan endpoint

- [x] 2.7: Verify all tests pass after DI consolidation — 171/171 passing
  - Zero get_db_connection calls remain in services/

---

## Phase 3: Performance & Query Fixes [COMPLETE]

- [x] 3.1: Fix N+1 query in list_sessions (backend/app/api/routes/chat.py) [SMALL]
  - Replaced N+1 loop with single LEFT JOIN + GROUP BY query
  - Index idx_chat_messages_session_id already existed

- [x] 3.2: Batch embedding API calls (backend/app/services/embeddings.py) [MEDIUM]
  - embed_batch now uses asyncio.gather + Semaphore(10) + sub-batches of 64

- [x] 3.3: Fix vector_store connect/close in delete endpoint (backend/app/api/routes/documents.py) [SMALL]
  - Removed connect()/close() calls that disrupted shared VectorStore state

- [x] 3.4: Verify all tests pass — 171/171 passing

---

## Phase 4: Code Cleanup (Slop + Refactoring) [COMPLETE]

- [x] 4.1: Extract vault query helper (backend/app/api/routes/vaults.py) [SMALL]
  - Extracted _row_to_vault_response, _VAULT_WITH_COUNTS_SQL, _fetch_vault_with_counts, _fetch_all_vaults
  - vaults.py reduced from 431 to 346 lines

- [x] 4.2: Clean up RAG engine cruft (backend/app/services/rag_engine.py) [SMALL]
  - Removed redundant _instance params, test-only getattr check, replaced self.logger with module-level logger
  - Updated 9 test call sites in test_rag_pipeline.py

- [x] 4.3: Extract ChatPage hooks (frontend/src/pages/ChatPage.tsx) [MEDIUM]
  - Created useChatHistory.ts (61 lines) + useSendMessage.ts (178 lines)
  - ChatPage.tsx reduced from 460 to 307 lines (JSX-focused)

- [x] 4.4: Extract MemoryPage hooks (frontend/src/pages/MemoryPage.tsx) [SMALL]
  - Created useMemorySearch.ts (81 lines) + useMemoryCrud.ts (146 lines)
  - MemoryPage.tsx reduced from 339 to 207 lines

- [x] 4.5: Extract shared formatters (frontend/src/pages/DocumentsPage.tsx) [SMALL]
  - Created lib/formatters.ts (formatFileSize, formatDate)
  - Created components/shared/StatusBadge.tsx (status badge component)
  - DocumentsPage.tsx reduced from 427 to ~377 lines

- [CANCELLED] 4.6: Split vault store — 84 lines, already clean; splitting adds unnecessary complexity

- [x] 4.7: Verify frontend build passes — npm run build succeeds (600 KB bundle)

- [x] 4.8: Verify all backend tests pass — 171/171 passing

---

## Phase 5: Frontend Polish [COMPLETE]

- [x] 5.1: Deduplicate health check logic (App.tsx + SettingsPage.tsx) [SMALL]
  - Extracted useHealthCheck hook (50 lines); App.tsx 72→38 lines; SettingsPage simplified

- [CANCELLED] 5.2: Load settings defaults — initializeForm already overwrites defaults from server values on mount

- [x] 5.3: Add API client interceptors (frontend/src/lib/api.ts) [SMALL]
  - Added response interceptor for error normalization; removed redundant try/catch from 3 functions

- [CANCELLED] 5.4: Upload progress — already handled: onProgress(0) triggers "Uploading..." text in UI

- [x] 5.5: Verify frontend build + backend tests — Build succeeds, 171/171 tests passing

---

## Phase 6: Final Validation [COMPLETE]

- [x] 6.1: Full regression test suite [SMALL] (depends: all prior phases)
  - pytest 171/171 passing + npm run build succeeds

- [x] 6.2: Update inline docstrings for changed public APIs [SMALL] (depends: 6.1)
  - Added docstrings to deps.py (get_toggle_manager, get_secret_manager)
  - Added JSDoc to 6 new frontend hooks + 3 helper functions + StatusBadge component
  - 171/171 tests passing, frontend build clean
