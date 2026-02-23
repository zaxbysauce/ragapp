# Implementation Spec: RAG Best-Practice Alignment (Tech Docs) — zaxbysauce-ragapp

Target: eliminate UI/backend semantic drift, make relevance threshold actually work, add instruction-aware embeddings for Qwen3-Embedding-4B, and implement retrieval behaviors that match modern technical-doc RAG (windowing/merge) without external dependencies.

Repo signals (from gitingest):
- UI labels chunk size/overlap as **characters**.
- Backend chunking treats settings as **token-ish**, then multiplies by 4 for Unstructured max_characters.
- Retrieval filtering uses `record["score"]` but LanceDB returns `_distance`.
- `Max Context Chunks` appears not wired to retrieval `top_k` (backend uses `settings.vector_top_k`).

---

## 0. Guiding Constraints

1) One source of truth for units:
- **ChunkSizeChars** (integer, characters)
- **ChunkOverlapChars** (integer, characters)

2) Retrieval must operate on an explicit score field:
- Use LanceDB `_distance` as returned; convert to similarity only if you can guarantee the metric semantics.

3) Avoid new heavy deps:
- No rerankers requiring additional models.
- Implement “sentence-window” behavior via deterministic adjacent-chunk merge (same doc, contiguous index).

4) Preserve existing settings compatibility:
- Include migration/compat layer for older configs if present.

---

## 1. Fix Chunk Size / Overlap Unit Mismatch (UI vs Backend)

### Problem
UI says “characters”, backend effectively uses “tokens” then `*4` to convert to chars for Unstructured.

### Change
Make the settings explicitly character-based end-to-end.

### Files
- `frontend/.../SettingsPage.*` (where labels are defined)
- `backend/.../DocumentProcessor` (where `max_characters` and `overlap` are passed to Unstructured)

### Implementation
A) Backend: remove implicit `*4` scaling.
- Today: `max_characters=self.chunk_size * 4`, `overlap=self.chunk_overlap * 4`
- New: `max_characters=self.chunk_size_chars`, `overlap=self.chunk_overlap_chars`

B) Settings model:
- Rename/introduce fields:
  - `chunk_size_chars`
  - `chunk_overlap_chars`
- Backward compatibility:
  - If config contains `chunk_size` and no `chunk_size_chars`, treat existing `chunk_size` as **tokens** (legacy) and set:
    - `chunk_size_chars = chunk_size * 4`
  - Same for overlap.

C) Frontend:
- Change labels to match actual units:
  - “Chunk Size (characters)”
  - “Chunk Overlap (characters)”
- If you keep old keys for API payloads, update UI tooltips explicitly.

### Acceptance Criteria
- Setting `Chunk Size = 2048` results in Unstructured being called with `max_characters=2048` (not 8192).
- No new docs get chunked differently due to hidden scaling.
- Existing installs with legacy `chunk_size` keep approximately the same behavior via migration mapping.

---

## 2. Wire “Max Context Chunks” to Actual Retrieval Top-K

### Problem
UI “Max Context Chunks” is not guaranteed to control backend retrieval. Backend uses `settings.vector_top_k`.

### Change
Make a single canonical setting: `retrieval_top_k` (or use existing `vector_top_k` and wire UI to it).

### Implementation
A) Backend settings:
- Add/alias:
  - `retrieval_top_k` (int)
- If `vector_top_k` exists, either:
  - Deprecate it and map `vector_top_k -> retrieval_top_k`, or
  - Keep `vector_top_k` and remove `max_context_chunks` from backend entirely.

B) Frontend:
- “Max Context Chunks” must post to the backend key used by RAGEngine (`retrieval_top_k` / `vector_top_k`).

### Acceptance Criteria
- Changing “Max Context Chunks” changes the number of chunks retrieved in the response context deterministically.

---

## 3. Make Relevance Threshold Actually Work (Distance vs Score)

### Problem
Filtering uses `record.get("score")`, but results contain `_distance`. Defaulting missing score to 1.0 makes threshold a no-op.

### Change
Define a single, explicit field to threshold on:
- `distance = record["_distance"]` from LanceDB

You have two valid options:
- Option A (recommended): threshold on `_distance` directly (lower is better).
- Option B: convert to similarity score and threshold on similarity.

Because LanceDB distance metric can vary (cosine vs L2), Option A is safest unless you explicitly set metric.

### Implementation (Option A: threshold on distance)
A) Rename UI field and semantics:
- `rag_relevance_threshold` becomes `max_distance_threshold`
- Default: `0` disables thresholding (include all), or `null`.

B) Backend filtering:
- If threshold is set and > 0:
  - include only records where `_distance <= threshold`
- Store in logs which records were dropped and why.

C) Expose debug info (optional but recommended):
- In response payload include:
  - `retrieval_debug: { top_k, threshold, returned_count, filtered_count }`

### Suggested Defaults
- Start with no threshold by default until metric confirmed.
- Provide presets:
  - Cosine distance common ranges are stack-dependent; do not hardcode “0.55–0.70” unless you’re using similarity.

### Acceptance Criteria
- With threshold enabled, returned chunks drop deterministically based on `_distance`.
- With threshold disabled, no chunks are filtered due to missing `score`.

---

## 4. Lock Vector Metric + Normalize Threshold Semantics

### Problem
Without specifying metric at table creation/query time, threshold meaning is ambiguous.

### Change
Explicitly define the metric used by LanceDB for vector search and document it in code + UI.

### Implementation
A) When creating LanceDB table or defining vector index:
- Explicitly set metric (prefer cosine for embeddings).
- Ensure query uses same.

B) Store metric in settings:
- `vector_metric: "cosine" | "l2"` (default "cosine")

C) If cosine metric is used:
- Prefer Option B thresholding (similarity) because it’s operator-friendly:
  - `similarity = 1 - cosine_distance`
  - Threshold on `similarity >= min_similarity`

Only do this if you confirm LanceDB returns cosine *distance*.

### Acceptance Criteria
- Metric is deterministic across environments.
- Threshold label matches actual math (“min similarity” or “max distance”).

---

## 5. Add Instruction-Aware Embedding Prefixes (Qwen3-Embedding-4B)

### Problem
EmbeddingService sends raw text; Qwen3 instruction-aware capability unused.

### Change
Add optional `embedding_doc_prefix` and `embedding_query_prefix` settings, applied consistently.

### Implementation
A) Settings:
- `embedding_doc_prefix` (string, default empty)
- `embedding_query_prefix` (string, default empty)

B) Embed paths:
- Document embedding:
  - `text_to_embed = embedding_doc_prefix + original_text`
- Query embedding:
  - `text_to_embed = embedding_query_prefix + query`

C) Provide safe defaults for Qwen3:
- Document prefix:
  - `Instruct: Represent this technical documentation passage for retrieval.\nDocument: `
- Query prefix:
  - `Instruct: Retrieve relevant technical documentation passages.\nQuery: `

D) Important: Keep prefixes identical across a full index lifecycle.
- If prefixes change, embeddings must be regenerated.
- Detect and warn by storing `embedding_prefix_hash` in metadata.

### Acceptance Criteria
- Prefixes are applied to all embeddings (doc + query).
- Index is invalidated or reindexed when prefixes change.

---

## 6. Add Adjacent-Chunk Windowing (Modern Tech-Doc Coherence)

### Problem
Chunking alone cannot guarantee the generator sees enough surrounding context for procedures, code blocks, and multi-step sections.

### Change
Implement deterministic “window expansion” without new models:
- Retrieve top-k chunks.
- For each chunk, also include N adjacent chunks (same document) by chunk index.
- Merge/unique and then cap to `max_context_chunks`.

### Implementation
A) During ingestion, persist these fields per chunk:
- `doc_id` (stable)
- `chunk_index` (0..n-1)
- `chunk_count` (optional)
- `source_path`
- `section_title` (if available from Unstructured metadata)

B) In retrieval pipeline:
- Initial fetch: `top_k = retrieval_top_k`
- Expansion:
  - `window = settings.retrieval_window` (default 1)
  - For each hit, request chunk indices:
    - `[chunk_index - window .. chunk_index + window]`
- Fetch those chunks via secondary query:
  - If LanceDB supports filtering by doc_id and chunk_index, use it.
  - Otherwise, store a `chunk_uid = doc_id:chunk_index` and query by that list.

C) Merge:
- Deduplicate by `chunk_uid`
- Sort by `(doc_id, chunk_index)`
- Cap to `max_context_chunks` (or use retrieval_top_k as the cap)

D) UI settings:
- `retrieval_window` (int, default 1)
- Tooltip: “Includes adjacent chunks around each hit.”

### Acceptance Criteria
- For a hit in the middle of a procedure, context includes surrounding steps.
- Window expansion never crosses document boundaries.
- Deduplication prevents runaway context bloat.

---

## 7. Make Chunk Boundaries More Tech-Doc Safe (No Mid-Code Splits)

### Goal
Reduce splitting inside code blocks/tables.

### Implementation
A) If Unstructured chunker exposes “respect boundaries” flags, enable them.
B) If not available:
- Post-process chunk text:
  - Detect if chunk ends inside a fenced code block (odd count of ```).
  - If so, append next chunk until code block closes, within a max char limit.
- Same for Markdown tables (line starts with `|`).

C) Keep this deterministic and bounded:
- `max_merge_chars` setting (default 8192).

### Acceptance Criteria
- Markdown fenced blocks are not split across two chunks in the final stored chunks except when exceeding `max_merge_chars`.

---

## 8. Harden LanceDB Dimension/Schema Validation

### Problem
Mismatch causes garbage retrieval or insertion errors. You already throw with an instruction to delete DB; improve operator experience.

### Change
Auto-detect embedding dimension and verify schema at startup.

### Implementation
A) On app startup (or first embed call):
- Generate a 1-item embedding for `"dimension_probe"`.
- Set `expected_dim = len(vector)`.
- Verify table schema vector dim matches expected_dim.
- If mismatch:
  - Block ingestion/search with explicit error:
    - “Embedding dimension changed from X to Y; reindex required.”
  - Offer a built-in endpoint/button:
    - “Reset Vector Index” (calls your existing reset path).

B) Store in table metadata:
- `embedding_model_id`
- `embedding_dim`
- `embedding_prefix_hash`

### Acceptance Criteria
- Changing embedding model or dimension produces a clear, immediate failure with remediation.

---

## 9. Observability + Regression Tests (Non-negotiable)

### Add logging counters (per query)
- `top_k`
- `window`
- `threshold`
- counts:
  - `initial_hits`
  - `expanded_hits`
  - `final_context_chunks`
- distance stats:
  - min/max/mean `_distance`

### Add tests
A) Unit tests:
- Settings migration:
  - legacy `chunk_size` => `chunk_size_chars = chunk_size*4`
- Threshold:
  - record with `_distance` filters correctly
- Window expansion:
  - dedup, sorted ordering, boundary clamp

B) Integration test:
- Ingest a synthetic Markdown doc with:
  - headings
  - fenced code block
  - table
  - procedure steps
- Query for a mid-procedure term:
  - confirm returned context contains adjacent steps and intact code block.

### Acceptance Criteria
- CI passes with deterministic outputs.

---

## 10. Minimal Patch Checklist (LLM-Friendly TODOs)

1) Introduce settings:
   - `chunk_size_chars`, `chunk_overlap_chars`
   - `retrieval_top_k`
   - `retrieval_window`
   - `vector_metric`
   - `embedding_doc_prefix`, `embedding_query_prefix`
2) Add migration layer for legacy keys.
3) Remove `*4` scaling in DocumentProcessor.
4) Wire UI controls to real backend keys.
5) Fix thresholding to use `_distance` (or similarity if metric is locked).
6) Add per-chunk metadata fields: `doc_id`, `chunk_index`, `chunk_uid`.
7) Implement window expansion merge + cap logic.
8) Add schema validation (embedding_dim) at startup.
9) Add logs and tests.

---

## Default Values (Safe for Tech Docs)

- `chunk_size_chars`: 2000
- `chunk_overlap_chars`: 200
- `retrieval_top_k`: 12
- `retrieval_window`: 1
- `vector_metric`: "cosine"
- threshold:
  - disabled by default until confirmed semantics
- Qwen3 prefixes:
  - doc/query prefixes enabled only when embedding model contains `"Qwen"` or via explicit toggle

---

## Definition of Done

A) Operator-visible behavior matches UI labels (no hidden scaling).
B) Changing “Max Context Chunks” changes actual retrieved chunk count.
C) Relevance threshold filters based on an explicitly defined distance/similarity field.
D) Windowing improves coherence without materially increasing hallucination rate.
E) Switching embedding models/dimensions fails fast with explicit reindex guidance.
F) All changes covered by unit + integration tests.