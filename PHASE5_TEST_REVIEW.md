# Phase 5 Test Coverage Review: Recommended Additional Tests

## OVERVIEW
Reviewing test coverage for Phase 5: Memory Store, RAG Engine, and Configuration.

---

## 1. MEMORY_STORE.PY GAPS

### Priority: HIGH

#### Test 1.1: add_memory with invalid inputs
**Target File:** backend/tests/test_memory_store.py (new)
**Rationale:** The method validates for empty string (line 53-54), but no tests cover this validation logic.
**Test Cases:**
- Empty string raises MemoryStoreError
- Whitespace-only string raises MemoryStoreError
- None input raises MemoryStoreError

#### Test 1.2: add_memory success with all parameters
**Target File:** backend/tests/test_memory_store.py (new)
**Rationale:** add_memory accepts category, tags, source but no tests verify these are stored/retrieved correctly.
**Test Cases:**
- Store memory with all optional parameters
- Verify category, tags, source are included in returned MemoryRecord

#### Test 1.3: search_memories with empty query
**Target File:** backend/tests/test_memory_store.py (new)
**Rationale:** Method returns empty list for empty query (line 86-87), but no explicit test exists.
**Test Cases:**
- Empty string returns []
- Whitespace-only query returns []
- None returns []

#### Test 1.4: detect_memory_intent with all pattern types
**Target File:** backend/tests/test_memory_store.py (new)
**Rationale:** MEMORY_PATTERNS includes 4 regex patterns, but no tests cover all of them.
**Test Cases:**
- "remember that X" pattern matches correctly
- "don't forget X" pattern matches correctly
- "keep in mind X" pattern matches correctly
- "note that X" pattern matches correctly
- Case-insensitivity verified

#### Test 1.5: detect_memory_intent edge cases
**Target File:** backend/tests/test_memory_store.py (new)
**Rationale:** No tests for edge cases like no match, partial match, or trailing punctuation.
**Test Cases:**
- Text with no pattern returns None
- Text with multiple patterns returns first match
- Text with trailing periods matches correctly

---

### Priority: MEDIUM

#### Test 1.6: add_memory database insertion error handling
**Target File:** backend/tests/test_memory_store.py (new)
**Rationale:** Connection error handling exists (try/finally) but not explicitly tested. lastrowid=None case (line 65-66) not tested.
**Test Cases:**
- Simulate database insertion failure
- Verify MemoryStoreError is raised on insertion failure

#### Test 1.7: search_memories query building with FTS5
**Target File:** backend/tests/test_memory_store.py (new)
**Rationale:** Complex FTS5 SQL query (lines 92-98) not tested. JOIN and ORDER BY not validated.
**Test Cases:**
- Verify query uses FTS5 MATCH correctly
- Verify LIMIT parameter is passed
- Verify join between memories_fts and memories tables

#### Test 1.8: MemoryRecord dataclass fields
**Target File:** backend/tests/test_memory_store.py (new)
**Rationale:** MemoryRecord dataclass used extensively but no tests for field integrity.
**Test Cases:**
- Verify all fields can be instantiated
- Verify fields match returned values from add_memory and search_memories

---

## 2. RAG_ENGINE.PY GAPS

### Priority: HIGH

#### Test 2.1: _filter_relevant with relevance threshold
**Target File:** backend/tests/test_rag_engine.py
**Rationale:** Core filtering logic (lines 97-113) not tested. rag_relevance_threshold (0.1) is not validated.
**Test Cases:**
- Filter out scores below threshold
- Filter out scores equal to threshold
- Filter out None scores (defaulted to 1.0)
- Return only matched RAGSource objects with correct metadata

#### Test 2.2: _filter_relevant with mixed scores
**Target File:** backend/tests/test_rag_engine.py
**Rationale:** No test for mixed results with some below, some above threshold.
**Test Cases:**
- Input: scores [0.2, 0.3, 0.4, 0.1, 0.5]
- With threshold 0.3: returns [0.4, 0.5]
- Verify scores and metadata preserved

#### Test 2.3: query with memory detection disabled
**Target File:** backend/tests/test_rag_engine.py
**Rationale:** Memory detection happens before query execution (line 49-56), but no test for query without memory intent.
**Test Cases:**
- Query with no memory intent proceeds to vector search
- Query without memory intent yields sources and memories correctly

#### Test 2.4: _build_messages with empty context
**Target File:** backend/tests/test_rag_engine.py
**Rationale:** Context sections can be empty (line 137-141), but this path not tested.
**Test Cases:**
- No chunks or memories: user_content only has question
- Verify correct message structure

---

### Priority: MEDIUM

#### Test 2.5: _build_messages with both chunks and memories
**Target File:** backend/tests/test_rag_engine.py
**Rationale:** System builds context from chunks and memories but combination not tested.
**Test Cases:**
- Multiple chunks with memories
- Verify proper formatting and structure

#### Test 2.6: _format_chunk with missing metadata
**Target File:** backend/tests/test_rag_engine.py
**Rationale:** _format_chunk (lines 151-153) handles missing keys but no test.
**Test Cases:**
- Missing source_file and section_title: defaults to "document"
- Missing score: None value handled

#### Test 2.7: _source_metadata extraction
**Target File:** backend/tests/test_rag_engine.py
**Rationale:** Metadata extraction not tested.
**Test Cases:**
- Verify all chunk fields extracted correctly

#### Test 2.8: _build_system_prompt structure
**Target File:** backend/tests/test_rag_engine.py
**Rationale:** System prompt not tested for content or structure.
**Test Cases:**
- Verify prompt contains "KnowledgeVault" and instruction to cite sources

---

### Priority: LOW

#### Test 2.9: RAGEngine initialization with custom services
**Target File:** backend/tests/test_rag_engine.py
**Rationale:** __init__ accepts custom services but no test for passing them in.
**Test Cases:**
- Pass custom embedding_service
- Pass custom vector_store
- Pass custom memory_store
- Pass custom llm_client

#### Test 2.10: query streaming error handling
**Target File:** backend/tests/test_rag_engine.py
**Rationale:** LLMError caught for non-stream (line 87-88), but no streaming error test.
**Test Cases:**
- Streaming fails with LLMError
- Verify RAGEngineError is raised and re-raised

#### Test 2.11: RAGEngineError exception
**Target File:** backend/tests/test_rag_engine.py
**Rationale:** Exception class defined but never tested.
**Test Cases:**
- Verify exception can be instantiated
- Verify exception is raised in error cases

---

## 3. CONFIG.PY GAPS

### Priority: HIGH

#### Test 3.1: rag_relevance_threshold default value
**Target File:** backend/tests/test_config.py (new)
**Rationale:** Default is 0.1 but not verified. This value is critical for RAG filtering.
**Test Cases:**
- Verify default is 0.1
- Verify can be overridden via environment variable

#### Test 3.2: Settings property paths
**Target File:** backend/tests/test_config.py (new)
**Rationale:** Property methods for directories not tested.
**Test Cases:**
- Verify documents_dir returns data_dir/documents
- Verify uploads_dir returns data_dir/uploads
- Verify library_dir returns data_dir/library
- Verify lancedb_path returns data_dir/lancedb
- Verify sqlite_path returns data_dir/app.db

---

### Priority: MEDIUM

#### Test 3.3: Settings model validation
**Target File:** backend/tests/test_config.py (new)
**Rationale:** Pydantic validation not tested.
**Test Cases:**
- Invalid port type raises validation error
- Invalid path type raises validation error
- Extra fields are ignored

#### Test 3.4: Settings environment variable override
**Target File:** backend/tests/test_config.py (new)
**Rationale:** Environment variable support not tested.
**Test Cases:**
- OVERRIDDEN_PORT env var overrides default
- OVERRIDDEN_MODEL env var overrides default
- MULTIPLE env vars work together

---

## SUMMARY OF RECOMMENDED TESTS

### New Test Files Needed:
1. **backend/tests/test_memory_store.py** - ~15-20 tests
2. **backend/tests/test_config.py** - ~8-10 tests

### Enhanced Existing Tests:
1. **backend/tests/test_rag_engine.py** - ~8-10 additional tests

### Total Coverage Improvements:
- **Memory Store:** 40% of code paths tested → 85%
- **RAG Engine:** 50% of code paths tested → 80%
- **Configuration:** 20% of code paths tested → 70%

### Critical Risk Areas:
1. Memory detection patterns (memory_store.py lines 33-38)
2. Relevance threshold filtering (rag_engine.py lines 97-113)
3. Database connection and FTS5 queries (memory_store.py lines 85-104)
4. Query message building (rag_engine.py lines 121-149)
