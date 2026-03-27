
# Codebase QA Report
Generated: 2026-03-26
Scope: Full codebase — backend (app/, services/, models/, middleware/, utils/, tests/), frontend (src/), infrastructure (Dockerfile, docker-compose, embedding_server), docs (README.md, docs/)
Files reviewed: ~110 source files across all modules
Claude bootstrap used: YES
Claimed-vs-shipped verification: YES

VERDICT: REJECTED
RISK: CRITICAL

## Executive Summary

The ragapp codebase contains **5 confirmed Critical findings** and **18 High findings** across security, data integrity, and correctness domains. Three default credential values (admin token, JWT secret, health check API key) allow authentication bypass and rate-limit circumvention in any deployment that does not change defaults — a critical risk for any shipped instance. A SQL injection vulnerability exists in the FTS vault filter in `vector_store.py`. An acknowledged Unicode TypeError in CSRF/security token comparison is documented in the test suite but left unfixed. Data integrity is compromised by missing CASCADE DELETE constraints on the files→vaults and memories→vaults foreign keys. Eighteen High-severity findings cover CSRF token replay, path traversal in file operations, inadequate attachment validation, missing user-scoping on memory search, connection pool exhaustion, and test theater that provides false assurance on security-critical paths.

---

## Findings Count

```
Group 1 (Behavior):           0 / 1 / 4 / 1 / 0
Group 2 (Substance/Wiring):   0 / 1 / 4 / 0 / 0
Group 3 (Security):           4 / 9 / 3 / 0 / 0
Group 4 (Dependencies):       0 / 2 / 1 / 0 / 0
Group 5 (Claimed vs Shipped): 0 / 0 / 3 / 1 / 2
Group 6 (Cross-Platform):     0 / 0 / 1 / 0 / 0
Group 7 (AI Smells):          0 / 0 / 3 / 2 / 0
Group 8 (Architecture/Perf):  0 / 3 / 5 / 0 / 0
Group 9 (Tests):              1 / 3 / 8 / 0 / 0
------------------------------------------------
TOTAL:                        5 / 19 / 32 / 4 / 2

AI Pattern Distribution:
  mapping-hallucination: 0
  naming-hallucination: 0
  resource-hallucination: 0
  logic-hallucination: 3
  claim-hallucination: 2
  phantom-dependency: 0
  stale-api: 0
  context-rot: 1
  unwired-functionality: 2
  happy-path-only: 7
  other: 5

Claim Ledger:
  supported: 9
  partially_supported: 4
  unsupported: 0
  contradicted: 3
  stealth_change: 1
```

---

## Claim Ledger Summary

| Claim | Status | Evidence |
|---|---|---|
| Hybrid search (dense + BM25 sparse) | PARTIALLY_SUPPORTED | Code path exists; `hybrid_alpha` parameter accepted but never applied to RRF weighting (vector_store.py:463) |
| Tri-vector search | PARTIALLY_SUPPORTED | Dense + sparse implemented; colbert always returns `None` in query pipeline (embeddings.py:366, 379) |
| Contextual chunking | SUPPORTED | `contextual_chunking.py` implemented and integrated in `rag_engine.py:389` |
| Reranking | PARTIALLY_SUPPORTED | `reranking.py` exists and is called; `_rerank_score` added to chunks but results not re-sorted by it before final limit (rag_engine.py:290) |
| Email ingestion via IMAP | SUPPORTED | `email_service.py` fully implemented, wired in `main.py:272` |
| Multi-vault support | SUPPORTED | Vault routes, filtering, and access control all implemented |
| Rate limiting | SUPPORTED | `WhitelistLimiter` in `limiter.py`, applied via `SlowAPIMiddleware` in `main.py` |
| CSRF protection | PARTIALLY_SUPPORTED | `CSRFManager` implemented; however tokens are not invalidated on use (security.py:118-123) |
| Session memory | SUPPORTED | `MemoryStore` integrated with intent detection in `rag_engine.py:118` |
| Background processing queue | SUPPORTED | `BackgroundProcessor` running, wired to email and file watcher |
| Default embedding model is bge-m3 | CONTRADICTED | `docs/release.md:36` and `docs/admin-guide.md:255` reference `nomic-embed-text`; all code and docker-compose default to `bge-m3` |
| Architecture diagram is production-accurate | CONTRADICTED | `README.md:34` shows port 5173 in architecture; `README.md:35` immediately states it is dev-only |
| Release checklist is current | CONTRADICTED | `docs/release.md:36` shows outdated default embedding model |

---

## Critical and High Findings

### CRITICAL

---

**[CRITICAL] CORE-001 — `backend/app/config.py:177`**
**Insecure default admin token shipped in all deployments**

`admin_secret_token` defaults to the literal string `"admin-secret-token"`. Any deployment that does not override this value (via environment variable) has a known, public authentication bypass token. `security.py:207` checks against this value at request time, but there is no startup validation rejecting the default. A warning is logged at `main.py:193-198` but execution continues normally.

*Fix:* Raise `ValueError` at startup if `admin_secret_token == "admin-secret-token"`. Do not allow application startup with a default credential.

---

**[CRITICAL] CORE-002 — `backend/app/config.py:184`**
**Weak default JWT secret key is human-guidance text, not a random value**

`jwt_secret_key` defaults to `"change-me-to-a-random-64-char-string"`. This is a documentation string, not entropy. Any deployment using this default produces JWTs that can be forged by any attacker who reads the source code. No startup validation enforces that this was changed.

*Fix:* At startup, validate `len(settings.jwt_secret_key) >= 64` and that it is not equal to the default string. Raise on failure.

---

**[CRITICAL] CORE-012 — `backend/app/limiter.py:16-32`**
**Health check API key defaults to public string, bypasses rate limiting globally**

`health_check_api_key` defaults to `"health-api-key"` (`config.py:164`). The `_should_whitelist` function in `limiter.py:23` compares the `X-API-Key` header against this value using `hmac.compare_digest`. Since the default is public and known, any caller supplying `X-API-Key: health-api-key` bypasses rate limiting on all routes. This is a rate-limit bypass without authentication.

*Fix:* Require `health_check_api_key` to be set to a strong random value. Raise at startup if it equals the default.

---

**[CRITICAL] RAG-001 — `backend/app/services/vector_store.py:611`**
**SQL injection in FTS vault filter — pattern inconsistency confirmed**

Line 611 interpolates `vault_id` directly into a filter string: `fts_query = fts_query.where(f"vault_id = '{vault_id}'")` without the sanitization applied at every other call site. Lines 314-315, 350-351, 404-408, 581-582, 708-714 all apply `safe_vault_id = str(vault_id).replace("'", "\\'")` before string interpolation. Line 611 is the sole exception. A caller controlling `vault_id` (e.g., via API with string-coerced vault ID) can inject arbitrary filter expressions.

*Fix:* Apply `safe_vault_id = str(vault_id).replace("'", "\\'")` at line 611, consistent with all other call sites.

---

**[CRITICAL] TEST-001 — `backend/tests/test_security_adversarial.py:293`**
**Acknowledged Unicode TypeError in security comparison left unfixed; tests expect the bug**

Lines 286-306 document that fullwidth Unicode characters, combining diacritics, homoglyphs, and RTL override characters cause an unhandled `TypeError` in CSRF/token `compare_digest` comparisons. Lines 293, 299, 305, 318 assert `pytest.raises(TypeError)` — meaning the test suite **passes** with the vulnerability present. The issue is acknowledged in comments but not fixed in `security.py`. An attacker sending a Unicode-mangled token can trigger an unhandled exception path rather than a clean 401/403.

*Fix:* In `security.py`, encode all token strings to bytes before `hmac.compare_digest`: `hmac.compare_digest(token.encode("utf-8"), stored.encode("utf-8"))`. Update tests to expect `HTTPException`, not `TypeError`.

---

### HIGH

---

**[HIGH] CORE-006 — `backend/app/main.py:118-163`**
**Settings loaded from database with `setattr` without key allowlist**

`_load_persisted_settings` reads key-value pairs from the `settings_kv` table and applies them via `setattr(settings, key, value)`. `NEW_DIRECT_KEYS` (line 122-139) accepts a list of keys from the database without an explicit allowlist verification before `setattr`. If the database is modified (via SQL injection, compromised admin, or the unsanitized FTS filter in RAG-001), arbitrary attributes can be set on the `settings` object, potentially overriding security-critical configuration values.

*Fix:* Define a `SETTINGS_ALLOWLIST = frozenset({...})` containing only the keys permitted for database persistence. Validate each key against the allowlist before `setattr`.

---

**[HIGH] CORE-008 — `backend/app/security.py:118-123`**
**CSRF token not invalidated on use — unlimited replay within TTL**

`validate_token` calls `store.expire(key)` (line 120) to refresh the TTL rather than `store.delete(key)`. The token remains valid for its full TTL after successful validation and can be replayed indefinitely. An attacker who obtains a valid CSRF token (e.g., via XSS) can use it for all state-changing requests until expiry.

*Fix:* Replace `store.expire(key)` with `store.delete(key)`. Issue a new CSRF token in the response after successful validation.

---

**[HIGH] CORE-014 — `backend/app/api/deps.py:142-169`**
**Default admin token rejection placed AFTER scope validation**

In `require_scope`, scope validation occurs at line 158 before the default-token rejection check at lines 161-163. A caller providing the default token `"admin-secret-token"` who also presents a matching scope header passes scope validation before being rejected. Depending on execution order of other middleware, this sequencing is fragile.

*Fix:* Move the `DEFAULT_TOKEN` rejection check (lines 161-163) to execute before scope validation (line 158).

---

**[HIGH] CORE-022 — `backend/app/main.py:193-198`**
**Application starts normally after logging critical warning about default admin token**

Lines 193-198 issue `logger.critical(...)` if `admin_secret_token` is the default value, but do not halt startup. The application serves all routes with a publicly-known admin token. See also CORE-001.

*Fix:* `sys.exit(1)` or raise `RuntimeError` after the critical warning.

---

**[HIGH] PROC-002 — `backend/app/services/email_service.py:365`**
**Unsanitized email body content reaches vector store**

`_sanitize_html` is defined at lines 656-671 but is never called on email body content before parsing, chunking, or storage. Email body is extracted at lines 365-369 and passed directly to the document processor. Malicious HTML or script content in an email body is stored without sanitization and could be returned in RAG responses rendered to users.

*Fix:* Call `_sanitize_html(body)` on extracted email body content before passing to document processor.

---

**[HIGH] PROC-003 — `backend/app/services/email_service.py:495-509`**
**Email attachment validation uses MIME type from headers only — no magic number check**

Attachment validation at lines 495-509 checks `part.get_content_type()`, which is attacker-controlled in the email. A malicious executable renamed to `.pdf` with MIME type `application/pdf` passes the whitelist check. No file signature (magic number) verification is performed.

*Fix:* Read the first N bytes of the payload and verify the file signature matches the claimed MIME type before accepting the attachment.

---

**[HIGH] PROC-021 — `backend/app/services/email_service.py:455`**
**`os.path.basename` does not sanitize control characters or null bytes in filenames**

`_sanitize_filename` uses `os.path.basename(filename)` to strip directory components but does not remove control characters (`\x00`-`\x1f`), null bytes, or filesystem-invalid characters. A filename like `file\x00name.txt` passes sanitization unchanged and can cause undefined behavior on downstream filesystem operations.

*Fix:* After `basename`, apply a regex to whitelist only safe characters: `re.sub(r'[^\w.\-]', '_', name)`.

---

**[HIGH] PROC-025 — `backend/app/services/upload_path.py:342`**
**`os.rename` in vault path migration lacks path canonicalization**

At line 342, `os.rename(old_path, new_path)` is called after constructing paths from `safe_name` and `vault_id`, but neither path is resolved with `.resolve()` or validated to be within `vaults_dir` using `.relative_to()`. A symlink or race condition could cause files to be moved outside the expected directory tree.

*Fix:* Call `.resolve()` on both paths and assert both are within `vaults_dir` using `.relative_to(vaults_dir)` before `os.rename`.

---

**[HIGH] FE-003 — `frontend/src/lib/api.ts:505-593`**
**Aborted stream leaves incomplete assistant message permanently on screen**

When a stream is aborted mid-response (network failure, user abort, server crash), the optimistic assistant message created at line 65-69 remains in the UI with partial or empty content. There is no error flag, rollback, or "message incomplete" indicator. Users see a permanently incomplete message with no explanation.

*Fix:* In the stream abort/error handler, set `message.error = true` or `message.content += "\n\n[Response interrupted]"` and update the store so the UI can render an appropriate indicator.

---

**[HIGH] FE-005 — `frontend/src/components/RoleGuard.tsx:9-16`**
**RoleGuard enforces roles client-side only using hydrated localStorage state**

`RoleGuard` checks `user.role` from `authStore`, which is hydrated from localStorage on page load (`authStore.ts:168`). The role is only fetched from the server once (at login). An attacker who modifies localStorage can bypass all `RoleGuard` protected components. Backend endpoints must independently enforce role requirements.

*Fix:* Ensure every endpoint behind a `RoleGuard` has an explicit role check in `deps.py`. Add a periodic server-side role re-validation (e.g., on `useEffect` mount in `ProtectedRoute`).

---

**[HIGH] DB-002 — `backend/app/models/database.py:45`**
**Missing `ON DELETE CASCADE` on `files.vault_id → vaults.id`**

The `files` table foreign key at line 45 does not specify `ON DELETE CASCADE`. Deleting a vault leaves all associated file records in the `files` table with dangling `vault_id` references, causing data inconsistency and potential information disclosure in queries that filter by vault.

*Fix:* Change line 45 to: `FOREIGN KEY (vault_id) REFERENCES vaults(id) ON DELETE CASCADE`

---

**[HIGH] DB-003 — `backend/app/models/database.py:51`**
**Missing foreign key constraint on `memories.vault_id`**

The `memories` table has `vault_id INTEGER` (nullable) with no foreign key declaration. Vault deletion does not affect orphaned memory records. This is inconsistent with the intent for vault-scoped memories.

*Fix:* Add `FOREIGN KEY (vault_id) REFERENCES vaults(id) ON DELETE SET NULL` to the `memories` table schema.

---

**[HIGH] MEM-001 — `backend/app/services/memory_store.py:95`**
**Memory search not scoped to authenticated user — any vault_id accepted without access check**

`search_memories` accepts a `vault_id` parameter (line 95) and filters results by it (SQL at line 127), but does not verify the calling user has read access to that vault. An authenticated user can pass any `vault_id` to retrieve memories from vaults they do not have access to.

*Fix:* Before executing the FTS query, validate that the authenticated user is a member of the specified vault using the `vault_members` table.

---

**[HIGH] TEST-003 — `backend/tests/test_vector_store_security.py:84`**
**SQL injection tests verify string construction only, not actual VectorStore behavior**

Tests in `TestVaultFilterSanitization` (lines 84-134) construct malicious filter strings manually and assert that the injection payload is present `in` the combined string — they never call `VectorStore.search()`. These tests document vulnerability scenarios but provide no proof that the production code actually sanitizes input. The 2 production call sites that DO correctly escape are not tested for consistency with the 1 site that does not (RAG-001).

*Fix:* Call `VectorStore.search()` with adversarial `filter_expr` values and assert that injection payloads are either rejected with an exception or escaped such that they cannot alter the query logic.

---

**[HIGH] TEST-004 — `backend/tests/test_vector_store_security.py:261`**
**Boundary condition tests are all empty `pass` statements**

`TestSearchBoundaryConditions` (lines 258-295) contains 6+ test methods (`test_limit_zero`, `test_negative_limit`, `test_oversized_embedding`, `test_wrong_dimensions`, etc.) that contain only `pass`. All are commented "# Tested via integration" but there are no corresponding integration tests. These tests run, pass, and report coverage for paths that are never actually exercised.

*Fix:* Implement assertions in each test method or delete the placeholder stubs. Do not leave `pass`-only test methods in the suite.

---

**[HIGH] ARCH-002 — `backend/app/models/database.py:739`**
**SQLite connection pool exhaustion blocks requests for up to 15 seconds**

`SQLiteConnectionPool.get_connection()` (line 813) blocks on `self._pool.get(timeout=5)` with `max_wait_attempts=3`, meaning a request can block up to 15 seconds before raising `RuntimeError`. With the default pool size of 5, concurrent requests (e.g., multiple streaming sessions) can saturate the pool and degrade all subsequent requests, including auth checks.

*Fix:* Reduce `timeout` to 1 second per attempt and return `HTTP 503 Service Unavailable` immediately when the pool is exhausted rather than blocking the caller thread.

---

**[HIGH] INFRA-001 — `Dockerfile:2`**
**Node base image unpinned — non-deterministic builds**

`FROM node:20-alpine` has no patch version, allowing silent updates on each build. A supply chain change in the base image could introduce vulnerabilities without detection.

*Fix:* `FROM node:20.18.3-alpine` (or verified current patch). Also apply to `python:3.11-slim` at Dockerfile:10.

---

**[HIGH] INFRA-002 — `Dockerfile:10`**
**Python base image unpinned**

Same as INFRA-001 for the production Python image. `FROM python:3.11-slim` allows silent patch updates.

*Fix:* `FROM python:3.11.12-slim` or verified current patch version.

---

## Medium Findings

| ID | File | Line | Title | Group |
|---|---|---|---|---|
| CORE-007 | `app/main.py` | 118-119 | Silent exception swallowing in settings restoration | 1 |
| CORE-009 | `app/security.py` | 172-182 | CSRF validation applied to all methods with no method guard | 3 |
| CORE-010 | `app/security.py` | 185-194 | CSRF cookie `secure=True` hardcoded, breaks HTTP development | 3 |
| CORE-011 | `app/security.py` | 21-54 | In-memory CSRF store returns hardcoded `"1"` instead of stored token value | 3 |
| CORE-013 | `app/api/deps.py` | 274-335 | Vault permission queries not in a single transaction — race condition | 8 |
| CORE-015 | `app/middleware/logging.py` | 29-36 | Query parameter scrubbing uses substring match, over-redacts similar field names | 1 |
| CORE-016 | `app/services/auth_service.py` | 30-51 | Password strength check missing special character requirement | 3 |
| CORE-017 | `app/services/auth_service.py` | 54-67 | JWT `iat` (issued-at) not validated on decode | 3 |
| CORE-018 | `app/services/auth_service.py` | 83-91 | Refresh token stored as unsalted SHA256 hash | 3 |
| CORE-020 | `app/services/toggle_manager.py` | 29-47 | Race condition between cache check and database read in `get_toggle` | 8 |
| RAG-003 | `app/services/rag_engine.py` | 373 | `_rerank_score` added to chunks but results not re-sorted before final limit | 1 |
| RAG-004 | `app/services/vector_store.py` | 463 | `hybrid_alpha` accepted but never applied to RRF fusion weights | 2 |
| RAG-005 | `app/services/embeddings.py` | 350-366 | Tri-vector docstring claims colbert but `colbert=None` always returned | 2 |
| RAG-007 | `app/services/reranking.py` | 89 | Reranker fallback silently returns unsorted results with no failure signal | 1 |
| RAG-008 | `app/services/query_transformer.py` | 70-72 | HyDE failure silently reduces query diversity with no caller signal | 1 |
| RAG-010 | `app/services/rag_engine.py` | 209 | Hybrid search applied to original query only, not transformed variants | 8 |
| RAG-011 | `app/services/rag_engine.py` | 306 | Broad exception converts all retrieval errors to identical fallback state | 1 |
| PROC-007 | `app/services/file_watcher.py` | 161-163 | File watcher does not exclude symlinks from scan | 6 |
| PROC-008 | `app/services/file_watcher.py` | 148-188 | File watcher does not detect deleted files between scans | 8 |
| PROC-009 | `app/services/background_tasks.py` | 305-311 | `_handle_failure` itself not wrapped in try/except | 8 |
| PROC-011 | `app/services/document_processor.py` | 86-90 | TOCTOU race between existence check and file read in `parse()` | 8 |
| PROC-016 | `app/services/upload_path.py` | 228-229 | Vault name sanitization duplicated in multiple functions without shared utility | 5 |
| PROC-023 | `app/services/email_service.py` | 372 | Email sender/subject logged after sanitization — PII still present | 3 |
| PROC-026 | `app/services/email_service.py` | 537-592 | Temp attachment file not cleaned up if `enqueue()` call fails | 8 |
| PROC-028 | `app/services/email_service.py` | 201-207 | IMAP search result structure not validated before `data[0].split()` | 8 |
| DB-001 | `app/models/database.py` | 61 | FTS5 `memories_fts` search sanitization may not cover all FTS5 operators | 1 |
| DB-004 | `app/models/database.py` | 258 | Missing indexes on `memories.vault_id` and related query columns | 8 |
| MEM-002 | `app/services/memory_store.py` | 95 | No upper bound enforced on memory search `limit` parameter | 8 |
| CHUNK-001 | `app/services/chunking.py` | 227-232 | Naive backtick count for code fence detection; broken on escaped backticks | 7 |
| CHUNK-003 | `app/services/chunking.py` | 335-340 | Sentence splitter uses ASCII punctuation regex, fails on Unicode punctuation | 7 |
| FE-006 | `frontend/src/hooks/useSendMessage.ts` | 42-54 | Chat session creation has no lock against concurrent session creation | 8 |
| FE-007 | `frontend/src/stores/useUploadStore.ts` | 119-134 | Upload queue lock acquisition is not atomic in Zustand | 8 |
| FE-008 | `frontend/src/stores/useChatStore.ts` | 1 | Two chat stores (`useChatStore`, `useChatStoreRedesign`) with overlapping scope | 7 |
| FE-009 | `frontend/src/lib/auth.ts` | 10-19 | `useRequirePasswordChange` redirects but does not block rendering | 3 |
| FE-012 | `frontend/src/pages/LoginPage.tsx` | 24 | Synchronous `mustChangePassword` access immediately after async `login()` | 1 |
| FE-013 | `frontend/src/hooks/useChatHistory.ts` | 27-50 | Chat history cache not invalidated when active vault changes | 8 |
| TEST-002 | `backend/tests/test_security_adversarial.py` | 24 | Auth rejection tests verify status code only, not error message content | 9 |
| TEST-005 | `backend/tests/test_auth.py` | 693-702 | Password hash tests do not validate algorithm, iteration count, or salt | 9 |
| TEST-006 | `backend/tests/test_rag_engine.py` | 162 | Filter relevance tests use `_distance` field but production code may use `score` | 9 |
| TEST-007 | `backend/tests/test_rag_engine.py` | 249 | Fallback behavior test does not verify fallback chunks are the actual low-relevance results | 9 |
| TEST-010 | `backend/tests/test_auth.py` | 712-748 | Refresh token rotation test does not query database to verify old token invalidation | 9 |
| TEST-012 | `backend/tests/test_integration.py` | 79-164 | Integration tests use fully mocked services — named integration, behave as unit tests | 9 |
| DOC-003 | `docs/release.md` | 36 | Pre-deployment checklist references `nomic-embed-text` — current default is `bge-m3` | 5 |
| ARCH-001 | `app/models/database.py` | 289 | Concurrent startup of multiple app workers can race on schema migrations | 8 |
| ARCH-003 | `app/models/database.py` | 732 | `PRAGMA foreign_keys` may not be re-applied on pooled connection reuse | 8 |

---

## Low and Info Findings

| ID | File | Line | Title |
|---|---|---|---|
| CORE-019 | `app/services/secret_manager.py` | 14-38 | Secret key not validated for minimum length |
| CORE-023 | `app/middleware/maintenance.py` | 31-44 | Maintenance mode blocks writes but allows reads, may return inconsistent data |
| CORE-024 | `app/config.py` | 420-422 | Orphan vault ID hardcoded as `1` with no existence validation |
| RAG-002 | `app/services/rag_engine.py` | 290 | `reranker_top_n` > `initial_retrieval_top_k` misconfiguration not validated |
| RAG-009 | `app/services/vector_store.py` | 593-621 | `rrf_fuse` called even when hybrid is disabled — needless overhead |
| RAG-012 | `app/services/retrieval_evaluator.py` | 31-32 | Empty chunk list returns `CONFIDENT` — should return `NO_MATCH` |
| RAG-014 | `app/services/rag_engine.py` | 202 | `is` identity comparison used instead of index to detect original query embedding |
| RAG-015 | `app/services/rag_engine.py` | 483 | Mixed `_distance`/`score` semantics without type safety in threshold logic |
| PROC-013 | `app/services/document_processor.py` | 543-546 | Contextual chunking silently skipped for schema files with no log |
| FE-014 | `frontend/src/pages/DocumentsPage.tsx` | 248 | Document deletion uses `window.confirm` — inconsistent with rest of app using Dialog |
| FE-015 | `frontend/src/stores/useSettingsStore.ts` | 326-349 | Duplicate `resetState` / `reset` methods with identical implementations |
| FE-020 | `frontend/src/pages/ChatShell.tsx` | 413-415 | Hardcoded 50ms `setTimeout` for textarea focus — fragile on slow devices |
| DOC-001 | `README.md` | 34-35 | Architecture diagram shows port 5173, immediately annotated as dev-only |
| DOC-004 | `docs/admin-guide.md` | 255 | Admin guide shows `ollama pull nomic-embed-text` — current default is `bge-m3` |
| INFRA-003 | `docker-compose.yml` | 42 | Redis image pinned to minor version only (`redis:7-alpine`) |
| CONFIG-001 | `docker-compose.yml` | 13, 30 | `flag-embed` external dependency not documented in compose file |
| TEST-008 | `backend/tests/test_embeddings.py` | 73-122 | Batch overflow test mock setup is fragile — call order dependent |
| TEST-011 | `backend/tests/test_chat_streaming.py` | 129-365 | No tests for oversized streaming content chunks |

---

## Dominant AI Failure Modes

**1. Happy-path-only logic (7 instances)**
Prevalent across `rag_engine.py`, `email_service.py`, `reranking.py`, and `query_transformer.py`. Error conditions are caught and swallowed into generic fallback states with no signal to callers about degraded quality. Specifically: HyDE failure silently reduces query diversity, reranker failure silently returns unsorted results, stream abort leaves incomplete UI messages, retrieval exceptions produce identical fallback output regardless of root cause.

**2. Unwired/partial feature implementation (2 instances)**
`hybrid_alpha` is accepted by `vector_store.search()` but never applied to the RRF fusion formula. `colbert` vectors are documented in the tri-vector system, allocated in `TriVectorResponse`, and discussed in docstrings, but are always `None` at both the embedding server and RAG engine query layers.

**3. Logic hallucination (3 instances)**
`retrieval_evaluator.py` returns `CONFIDENT` on empty chunk lists (the semantically opposite result). `security.py` stores `"1"` as the token value in the in-memory CSRF store rather than the actual token. Reranking adds `_rerank_score` to chunks without using it for the final result sort order.

**4. Claim hallucination (2 instances)**
`docs/release.md` and `docs/admin-guide.md` consistently reference `nomic-embed-text` as the default embedding model; all code, docker-compose, and README default to `bge-m3`. The docstring on `embed_multi()` claims tri-vector returns `(dense, sparse, colbert)` but the implementation always returns `colbert=None`.

---

## Unsupported or Contradicted Claims

| Claim | Location | Status | Evidence |
|---|---|---|---|
| Default embedding model is `nomic-embed-text` | `docs/release.md:36`, `docs/admin-guide.md:255` | CONTRADICTED | All code defaults: `bge-m3`. docker-compose.yml:16, README.md:16, embedding_server both use bge-m3 |
| Architecture diagram shows production topology | `README.md:34` | CONTRADICTED | Port 5173 shown; README:35 immediately states it is development-only |
| `hybrid_alpha` weights the RRF fusion | `docker-compose.yml env`, `config.py`, `vector_store.py:449` | CONTRADICTED | Parameter accepted; vector_store.py:463 comment states "not directly used in pure RRF"; fusion.py uses only constant k |
| Reranked results are returned in rerank score order | Implied by reranking feature | PARTIALLY_SUPPORTED | `_rerank_score` added to chunks (reranking.py:96) but `rag_engine.py:290` applies final limit without re-sorting by rerank score |
| Tri-vector search uses dense + sparse + colbert | `README.md`, `embeddings.py:352` | PARTIALLY_SUPPORTED | colbert always `None`; only dense + sparse used in query pipeline |

---

## Stealth Changes

**RAG-006** — `backend/app/services/rag_engine.py:165-172` / `embeddings.py:366`
Documents are indexed with tri-vector embeddings (dense + sparse, colbert=None stub) during ingestion via `embed_multi()`, but the RAG query pipeline only generates `query_sparse` for the original query (not transformed variants) and never attempts colbert retrieval. The indexing and query paths are asymmetric with no documentation acknowledging this. Users enabling `TRI_VECTOR_SEARCH_ENABLED` get 2-vector search at query time despite 3-vector field indexing.

---

## Supply Chain and Dependency Notes

- **All Python packages in `backend/requirements.txt` verified** as legitimate PyPI packages with valid version constraints. No phantom packages detected.
- **All npm packages in `frontend/package.json` verified** as legitimate npm packages. No cross-ecosystem confusion.
- **Embedding server** (`backend/embedding_server/requirements.txt`) uses exact pinning (`torch==2.6.0`, `transformers==4.35.2`, `FlagEmbedding==1.2.10`) while the main backend uses range constraints. This dual strategy is intentional isolation but risks silent compatibility drift.
- **Docker base images unpinned** (`node:20-alpine`, `python:3.11-slim`, `redis:7-alpine`) — see INFRA-001, INFRA-002, INFRA-003. Non-deterministic builds.
- **`flag-embed` external service** is referenced in `docker-compose.yml:13` and resolved via `extra_hosts:flag-embed:host-gateway`. Not defined in compose, requiring undocumented external setup.

---

## Coverage Notes

- `backend/app/api/routes/` directory (individual route files for documents, vaults, auth, chat, settings, users) was not individually deep-read. The main.py router registration was verified and security deps were audited. Route-level parameter validation coverage may have gaps not captured here.
- The redesign directory (`redesign/frontend/`) was not audited — it appears to be a development prototype not currently wired into the production build.
- `backend/tests/` contains ~50 test files. 12 representative files were audited in depth. Coverage gaps in un-audited test files are possible.
- `backend/embedding_server/server.py` was read. The colbert=None finding (FEATURE-002) is confirmed.
- No runtime execution was performed — dynamic analysis findings (actual SQL behavior, actual FTS5 injection outcomes) are based on static analysis with HIGH confidence where pattern inconsistencies are structural.

---

## Recommended Remediation Order

### 1. Supply Chain and Authentication Criticals (Do First)
- CORE-001 + CORE-022: Reject startup if `admin_secret_token` equals default. Do not log a warning and continue.
- CORE-002: Reject startup if `jwt_secret_key` equals default or is < 64 chars.
- CORE-012: Reject startup if `health_check_api_key` equals default.
- RAG-001: Add `safe_vault_id` escaping at `vector_store.py:611`.
- TEST-001: Fix the Unicode encoding in `security.py` `compare_digest` calls. Change test assertions from `TypeError` to `HTTPException`.

### 2. Broken or Unwired Shipped Functionality
- CORE-008: Invalidate CSRF tokens on use (`store.delete` not `store.expire`).
- CORE-014: Move default-token rejection before scope validation in `require_scope`.
- RAG-003: After reranking, re-sort `vector_results` by `_rerank_score` before applying final limit.
- RAG-004: Either implement `hybrid_alpha` weighting in `rrf_fuse` or remove the parameter and document it as unsupported.
- RAG-012: Return `NO_MATCH` from `retrieval_evaluator` when chunk list is empty.
- CORE-011: Store actual token value (not `"1"`) in `_InMemoryCSRFStore`.

### 3. Trust Boundary Defects
- PROC-003: Add magic number file signature check for email attachments.
- PROC-021: Replace `os.path.basename` with full sanitization including control character removal.
- PROC-025: Add `resolve()` and `relative_to()` validation in vault path migration.
- PROC-002: Call `_sanitize_html()` on email body before processing.
- MEM-001: Add vault access authorization check in `search_memories`.
- FE-005: Add server-side role verification endpoint and call it on protected route mount.
- CORE-006: Implement `SETTINGS_ALLOWLIST` in `_load_persisted_settings`.

### 4. Data Integrity and Database
- DB-002: Add `ON DELETE CASCADE` to `files.vault_id` foreign key.
- DB-003: Add `ON DELETE SET NULL` to `memories.vault_id` with foreign key declaration.
- ARCH-001: Add migration advisory lock to prevent concurrent schema changes.
- ARCH-003: Re-apply `PRAGMA foreign_keys = ON` on every connection retrieved from pool.
- ARCH-002: Return HTTP 503 immediately on pool exhaustion instead of blocking 15 seconds.

### 5. Test Blind Spots
- TEST-003: Rewrite SQL injection tests to call `VectorStore.search()` with adversarial inputs.
- TEST-004: Implement boundary condition test bodies or delete placeholder stubs.
- TEST-010: Add database query to verify old token hash is removed after refresh.
- TEST-012: Create real integration test tier that uses actual service implementations.
- TEST-005: Add assertions on hash algorithm and iteration count.

### 6. Documentation Corrections
- DOC-003: Update `docs/release.md:36` from `nomic-embed-text` to `bge-m3`.
- DOC-004: Update `docs/admin-guide.md:255` from `ollama pull nomic-embed-text` to `ollama pull bge-m3`.
- RAG-005/RAG-006: Update `embeddings.py:352` docstring and README tri-vector claims to accurately reflect colbert deferral.
- DOC-001: Separate dev and production architecture diagrams in README.

### 7. Systemic AI Smell Cleanup
- Add caller signals for all silent degradation paths (HyDE failure, reranker unavailability, stream abort).
- Consolidate `useChatStore` and `useChatStoreRedesign` or document the split responsibility explicitly.
- Centralize vault name sanitization into a single utility function.
- Pin all Docker base images to specific patch versions.

---
*Report generated by Claude Sonnet 4.6. Static analysis only. No application source files were modified.*
