# KnowledgeVault Context

**Project:** KnowledgeVault - Self-Hosted RAG Knowledge Base  
**Swarm:** paid  
**Current Phase:** UI/UX Improvements — IN PROGRESS
**Last Updated:** 2026-02-22

---

## Current State

Full 6-category code review COMPLETE. 36 findings across backend and frontend:
- 1 CRITICAL (no-op rate limit bypass decorator)
- 17 HIGH (bugs, tech debt, DI inconsistencies, performance)
- 13 MEDIUM (refactoring, enhancements)
- 5 LOW (polish items)

Remediation plan: Phases 1-6 COMPLETE. Phase 7 in progress (retry exception preservation).

Previous work: Phases 1-6 of original implementation complete (security, DI, async, frontend, validation, multi-vault). 171/171 tests passing.

---

## Decisions

| Date | Decision | Rationale | Status |
|------|----------|-----------|--------|
| 2026-02-11 | Phase order: bugs → DI → perf → cleanup → polish → validation | Fix broken things first, consolidate architecture, then optimize and clean | Pending approval |
| 2026-02-11 | Keep SQLite connection-per-request pattern (not pooling) | SQLite connections are cheap; true connection pool adds complexity for embedded DB | Confirmed by SME |
| 2026-02-11 | Make LLMClient.start() async | It creates httpx.AsyncClient; must be awaitable. All callers already use await. | Confirmed by SME |
| 2026-02-11 | Remove ALL module-level singletons | Contradicts DI pattern in deps.py; causes test isolation issues | Confirmed by SME |

---

## SME Cache

### UI/UX (consulted 2026-02-22)

**Top 3 Priorities:**
1. Mobile navigation (bottom tabs + overflow drawer)
2. Responsive document view (cards on mobile, table on desktop)
3. Chat accessibility (live regions, ARIA labels, keyboard shortcuts)

**Mobile Patterns:**
- Bottom tab bar for primary 3-4 nav items, overflow drawer for rest
- Document table → card grid below sm breakpoint
- Chat sources collapsible accordion on mobile
- All touch targets ≥44×44px (min-w-[44px] min-h-[44px])

**Accessibility Requirements (WCAG 2.1 AA):**
- Chat textarea: `aria-label="Message input"` + `role="textbox"`
- Message list: `role="log"` with `aria-live="polite"`
- Table: `<caption>`, `<th scope="col">`, `<th scope="row">`
- Live regions: Use `react-aria-live` for streaming content
- Keyboard shortcuts: Help dialog triggered by `?` (Shift + /)

**Component Patterns:**
- Chat controls: "More" dropdown (DropdownMenu) with Rename, Delete, Export
- Upload queue: Stacked list with progress bars + cancel buttons
- Empty states: Icon + message + CTA button
- Status indicators: Follow shadcn badge patterns

### Python/FastAPI (consulted 2026-02-11)

1. **No-op decorator fix**: Whitelisted branch should return `await func()` directly; non-whitelisted branch should apply rate limiter via `async with rate_limiter`. Single wrapper, path-based decision.

2. **Sync/async mismatch**: Make `LLMClient.start()` async since it creates `httpx.AsyncClient`. Call once at startup, not per-request. Register close in shutdown hook.

3. **Singletons → DI**: Replace module-level `llm_client = LLMClient()` with `@lru_cache` factory in deps.py. Use `app.dependency_overrides` in tests. Watch for import cycles — keep deps.py imports minimal.

4. **DB connections in services**: Pass connection via constructor, not internal `get_db_connection()`. For background tasks, pass connection explicitly via `background_tasks.add_task(fn, db)`. Use `check_same_thread=False` for async usage.

5. **N+1 query fix**: Use `LEFT JOIN messages ON session_id GROUP BY session_id` with `COUNT(m.id)`. Add index `CREATE INDEX idx_messages_session_id ON messages(session_id)`.

6. **Embedding batching**: Use `asyncio.gather` with `Semaphore(10)` for concurrent HTTP calls. Preserve ordering. Chunk into sub-batches of 64. Handle individual failures with try/except.

---

## Patterns

| Pattern | Description | Used In |
|---------|-------------|---------|
| FastAPI DI | `Depends(get_X)` from deps.py | All route files |
| Test overrides | `app.dependency_overrides[get_X] = lambda: fake` | All test files |
| Connection pool | `SQLiteConnectionPool` in database.py | main.py lifespan |
| Pydantic V2 | `@field_validator`, `ConfigDict` | config.py, models |
| Auth opt-in | `require_auth` with Bearer token, constant-time compare | security.py |
| Custom hooks | Extract logic from pages into hooks/ directory | ChatPage, MemoryPage |
| Shared formatters | Pure functions in lib/formatters.ts | DocumentsPage |
| Shared components | Reusable UI in components/shared/ | StatusBadge, MessageContent |

---

## Key Files

| File | Purpose | Lines |
|------|---------|-------|
| backend/app/main.py | FastAPI app, lifespan, middleware | 197 |
| backend/app/api/deps.py | DI dependency functions | 88 |
| backend/app/config.py | Pydantic Settings | 98 |
| backend/app/security.py | Auth, CSRF, tokens | 213 |
| backend/app/limiter.py | Rate limiting | 87 |
| backend/app/models/database.py | Schema, migrations, pool | 504 |
| backend/app/services/llm_client.py | LLM chat client | 221 |
| backend/app/services/embeddings.py | Embedding service | 216 |
| backend/app/services/rag_engine.py | RAG orchestration | 253 |
| backend/app/services/vector_store.py | LanceDB vector store | 455 |
| backend/app/services/document_processor.py | Doc processing | 436 |
| backend/app/api/routes/chat.py | Chat + sessions | 485 |
| backend/app/api/routes/documents.py | Upload, scan, delete | 617 |
| backend/app/api/routes/vaults.py | Vault CRUD | 431 |

---

## Findings Summary (36 total)

### Category 1: Stubs/Placeholders (1)
- test_integration.py:619 — placeholder auth test

### Category 2: Bugs/Partial Implementations (5)
- limiter.py:42-86 — CRITICAL: no-op health check bypass
- llm_health.py:79 — HIGH: sync/async mismatch
- health.py:31-32 — HIGH: per-request instantiation
- security.py:41 — HIGH: CSRF token validation broken
- admin.py:44 — MEDIUM: deprecated datetime.utcnow()

### Category 3: AI Slop/Redundancy (7)
- rag_engine.py:40-41 — HIGH: redundant constructor params
- rag_engine.py:144 — HIGH: test code in production
- vaults.py (6 places) — HIGH: copy-pasted JOIN query
- ChatPage.tsx:67-73 — MEDIUM: duplicate parseInt logic
- ChatPage.tsx:181-196 — MEDIUM: duplicate load pattern
- MemoryPage.tsx:35-62 — MEDIUM: mixed search/list handler
- DocumentsPage.tsx:155-161 — LOW: inline search filter

### Category 4: Tech Debt (9)
- document_processor.py:363 — HIGH: own DB connection
- maintenance.py — HIGH: own DB connections
- memory_store.py — HIGH: own DB connections
- file_watcher.py:149 — HIGH: own DB connection
- toggle_manager.py:34,48 — HIGH: own DB connections
- Module singletons (4 files) — HIGH: contradicts DI
- useSettingsStore.ts:61-68 — MEDIUM: hardcoded defaults
- App.tsx + SettingsPage.tsx — MEDIUM: duplicate health check
- useVaultStore.ts:6-8 — LOW: unvalidated localStorage

### Category 5: Enhancement Opportunities (7)
- chat.py:246-251 — HIGH: N+1 query
- embeddings.py:207-211 — HIGH: sequential batching
- documents.py:563-576 — HIGH: connect/close disruption
- ChatPage.tsx:43-57 — MEDIUM: excessive reloading
- DocumentsPage.tsx:33-49 — MEDIUM: no cache layer
- api.ts:4-10 — MEDIUM: no interceptors
- DocumentsPage.tsx:67-93 — LOW: per-file progress only

### Category 6: Refactoring Targets (7)
- ChatPage.tsx:67-149 — HIGH: 83-line handleSend
- MemoryPage.tsx:35-73 — HIGH: mixed function
- useVaultStore.ts:25-77 — HIGH: multi-responsibility store
- DocumentsPage.tsx:163-213 — MEDIUM: inline formatters
- ChatPage.tsx:42-57 — MEDIUM: inline history loading
- SettingsPage.tsx:16-33 — MEDIUM: monolithic component
- useSettingsStore.ts:20-59 — LOW: coupled types

---

## Environment

- **Host:** Dual Xeon 5218, 380GB RAM, RTX A1000 8GB
- **OS:** Windows (dev), Linux (Docker host)
- **External Dependencies:** Ollama (user-managed)
- **Test Suite:** 171 backend tests, frontend verified via npm run build

## Agent Activity

| Tool | Calls | Success | Failed | Avg Duration |
|------|-------|---------|--------|--------------|
| read | 220 | 220 | 0 | 6ms |
| bash | 144 | 144 | 0 | 1798ms |
| edit | 90 | 90 | 0 | 195ms |
| glob | 57 | 57 | 0 | 26ms |
| task | 35 | 35 | 0 | 737923ms |
| grep | 20 | 20 | 0 | 133ms |
| invalid | 20 | 20 | 0 | 1ms |
| write | 18 | 18 | 0 | 197ms |
| retrieve_summary | 16 | 16 | 0 | 6ms |
| lint | 11 | 11 | 0 | 2275ms |
| test_runner | 11 | 11 | 0 | 1ms |
| imports | 9 | 9 | 0 | 3ms |
| diff | 5 | 5 | 0 | 15ms |
| secretscan | 5 | 5 | 0 | 33ms |
| todowrite | 2 | 2 | 0 | 3ms |
| apply_patch | 2 | 2 | 0 | 6ms |
| checkpoint | 1 | 1 | 0 | 7ms |
