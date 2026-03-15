"""Retrieval-augmented generation engine orchestration."""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

from app.config import settings
from app.services.embeddings import EmbeddingService, EmbeddingError
from app.services.llm_client import LLMClient, LLMError
from app.services.memory_store import MemoryRecord, MemoryStore
from app.services.vector_store import VectorStore
from app.services.query_transformer import QueryTransformer
from app.services.retrieval_evaluator import RetrievalEvaluator
from app.utils.fusion import rrf_fuse

# Deferred import to avoid circular dependency
def _get_pool():
    from app.models.database import get_pool
    return get_pool(str(settings.sqlite_path))

if TYPE_CHECKING:
    from app.services.context_distiller import ContextDistiller


logger = logging.getLogger(__name__)


@dataclass
class RAGSource:
    text: str
    file_id: str
    score: float
    metadata: Dict[str, Any]


class RAGEngineError(Exception):
    """Raised when the RAG engine cannot complete a query."""


class RAGEngine:
    """Coordinates embeddings, vector search, memory search, and LLM responses."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        memory_store: Optional[MemoryStore] = None,
        llm_client: Optional[LLMClient] = None,
        reranking_service=None,  # NEW
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        self.memory_store = memory_store or MemoryStore()
        self.llm_client = llm_client or LLMClient()
        self.reranking_service = reranking_service

        # Log warnings for missing dependencies (indicates non-DI usage)
        if embedding_service is None:
            logger.warning("RAGEngine created without injected embedding_service - using default instance")
        if vector_store is None:
            logger.warning("RAGEngine created without injected vector_store - using default instance")
        if memory_store is None:
            logger.warning("RAGEngine created without injected memory_store - using default instance")
        if llm_client is None:
            logger.warning("RAGEngine created without injected llm_client - using default instance")

        # Use new character-based fields, with fallback to legacy fields for backward compatibility
        self.chunk_size_chars = settings.chunk_size_chars
        self.chunk_overlap_chars = settings.chunk_overlap_chars
        self.retrieval_top_k = settings.retrieval_top_k
        self.vector_metric = settings.vector_metric
        self.max_distance_threshold = settings.max_distance_threshold
        self.embedding_doc_prefix = settings.embedding_doc_prefix
        self.embedding_query_prefix = settings.embedding_query_prefix
        self.retrieval_window = settings.retrieval_window

        # Reranking config
        self.reranking_enabled = settings.reranking_enabled
        self.reranker_top_n = settings.reranker_top_n
        self.initial_retrieval_top_k = settings.initial_retrieval_top_k

        # Hybrid search config
        self.hybrid_search_enabled = settings.hybrid_search_enabled
        self.hybrid_alpha = settings.hybrid_alpha

        # Legacy field support (deprecated) - warn if different from canonical fields
        self.relevance_threshold = settings.rag_relevance_threshold
        self.top_k = settings.vector_top_k
        if self.top_k is not None and self.top_k != self.retrieval_top_k:
            logger.warning(
                "vector_top_k (%s) is deprecated and differs from retrieval_top_k (%s). "
                "Using retrieval_top_k. Please update your settings.",
                self.top_k, self.retrieval_top_k
            )
        self.maintenance_mode = settings.maintenance_mode

        # Query transformer instance (lazy-loaded)
        self._query_transformer: Optional[QueryTransformer] = None

        # Retrieval evaluator instance (lazy-loaded)
        self._retrieval_evaluator: Optional[RetrievalEvaluator] = None

        # Context distiller instance (lazy-loaded)
        self._context_distiller: Optional["ContextDistiller"] = None

    async def query(
        self,
        user_input: str,
        chat_history: List[Dict[str, Any]],
        stream: bool = False,
        vault_id: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute a RAG query: embed, search, build prompt, call LLM."""
        try:
            memory_content = self.memory_store.detect_memory_intent(user_input)
            if memory_content:
                memory = self.memory_store.add_memory(memory_content, source="chat", vault_id=vault_id)
                yield {
                    "type": "content",
                    "content": f"Memory stored: {memory.content}",
                }
                return
        except Exception as exc:
            logger.error("Memory intent detection/add failed: %s", exc)

        # Query transformation for broader retrieval (if enabled)
        transformed_queries: List[str] = [user_input]
        if settings.query_transformation_enabled and self.llm_client is not None:
            try:
                if self._query_transformer is None:
                    self._query_transformer = QueryTransformer(self.llm_client)
                transformed_queries = await self._query_transformer.transform(user_input)
                if len(transformed_queries) > 1:
                    logger.info(
                        "Query transformation: original='%s', step_back='%s'",
                        transformed_queries[0],
                        transformed_queries[1]
                    )
            except Exception as e:
                logger.warning("Query transformation failed: %s, using original query only", e)
                transformed_queries = [user_input]

        # Embed all transformed queries
        query_embeddings: List[List[float]] = []
        for query_text in transformed_queries:
            try:
                embedding = await self.embedding_service.embed_single(query_text)
                query_embeddings.append(embedding)
            except EmbeddingError as exc:
                logger.warning("Failed to embed query variant '%s': %s", query_text, exc)

        if not query_embeddings:
            error_msg = "Unable to encode any query variants"
            if stream:
                yield {"type": "error", "message": error_msg, "code": "EMBEDDING_ERROR"}
                return
            raise RAGEngineError(error_msg)

        # Generate sparse query vector for learned sparse retrieval (original query only)
        query_sparse: Optional[dict] = None
        effective_alpha = self.hybrid_alpha
        if settings.tri_vector_search_enabled and getattr(self.embedding_service, "supports_tri_vector", False):
            try:
                query_sparse = await self.embedding_service.embed_query_sparse(user_input)
                logger.debug("Sparse query vector: %d tokens", len(query_sparse))
            except EmbeddingError as e:
                logger.warning("Sparse query vector generation failed, using alpha=1.0 (dense only): %s", e)
                query_sparse = None
                effective_alpha = 1.0  # Use dense-only search on sparse failure

        fallback_reason: Optional[str] = None
        vector_results: List[Dict[str, Any]] = []
        relevance_hint: Optional[str] = None
        reranking_applied = False
        
        # DEBUG: Log pre-vector-search state
        logger.debug(
            "RAG query: retrieval_top_k=%d, vault_id=%s, vector_store_connected=%s",
            self.retrieval_top_k,
            vault_id,
            getattr(self.vector_store, 'is_connected', lambda: 'unknown')()
        )
        
        # Initialize eval_result here so it's always defined regardless of maintenance mode
        eval_result = "CONFIDENT"

        if self.maintenance_mode:
            fallback_reason = "RAG index is under maintenance"
            vector_results = []
        else:
            try:
                # Stage 1: Initial retrieval
                fetch_k = self.initial_retrieval_top_k if self.reranking_enabled else self.retrieval_top_k
                top_k_value = fetch_k if fetch_k is not None else self.retrieval_top_k

                # Search with all query embeddings and fuse results
                all_results: List[List[Dict[str, Any]]] = []
                for embedding in query_embeddings:
                    is_original = embedding is query_embeddings[0]
                    results = await asyncio.to_thread(
                        self.vector_store.search,
                        embedding,
                        int(top_k_value),
                        vault_id=vault_id,
                        query_text=user_input if is_original else "",
                        hybrid=self.hybrid_search_enabled and is_original,
                        hybrid_alpha=effective_alpha,
                        query_sparse=query_sparse if is_original else None,
                    )
                    all_results.append(results)

                # Compute recency scores for tiebreaking in multi-query fusion
                recency_scores: Optional[Dict[str, float]] = None
                if settings.retrieval_recency_weight > 0.0 and len(all_results) > 1:
                    dates: Dict[str, float] = {}
                    for result_list in all_results:
                        for record in result_list:
                            uid = record.get("id", "")
                            if not uid:
                                continue
                            metadata = self._normalize_metadata(record.get("metadata", {}))
                            proc_at = metadata.get("processed_at") or record.get("processed_at")
                            if proc_at:
                                try:
                                    dates[uid] = datetime.fromisoformat(str(proc_at)).timestamp()
                                except (ValueError, TypeError):
                                    pass
                    if len(dates) > 1:
                        min_ts = min(dates.values())
                        max_ts = max(dates.values())
                        span = max_ts - min_ts or 1.0
                        recency_scores = {uid: (ts - min_ts) / span for uid, ts in dates.items()}

                # Fuse results from all query variants using RRF (with optional recency)
                if len(all_results) > 1:
                    vector_results = rrf_fuse(
                        all_results,
                        k=60,
                        limit=fetch_k,
                        recency_scores=recency_scores,
                        recency_weight=settings.retrieval_recency_weight,
                    )
                    logger.info("Fused results from %d query variants: %d results", len(all_results), len(vector_results))
                else:
                    vector_results = all_results[0] if all_results else []
                
                # Log vector search results
                logger.info(
                    "Vector search: vault_id=%s, top_k=%d, results=%d, distances=%s",
                    vault_id,
                    top_k_value,
                    len(vector_results),
                    [r.get("_distance") for r in vector_results[:3]] if vector_results else "N/A"
                )

                # Stage 2: Reranking (if enabled)
                if self.reranking_enabled and self.reranking_service and vector_results:
                    try:
                        vector_results = await self.reranking_service.rerank(
                            query=user_input,
                            chunks=vector_results,
                            top_n=self.reranker_top_n,
                        )
                        reranking_applied = True
                    except Exception as e:
                        logger.warning(f"Reranking failed, using original results: {e}")

                # Pack context by token budget (convert to RAGSource first for packing)
                if vector_results:
                    temp_sources = [
                        RAGSource(
                            text=r.get("text", ""),
                            file_id=str(r.get("file_id", "")),
                            score=r.get("_distance", 0.0),
                            metadata=r.get("metadata", {}),
                        )
                        for r in vector_results
                    ]
                    packed_sources = self._pack_context_by_token_budget(
                        temp_sources, settings.context_max_tokens
                    )
                    # Filter vector_results to match packed sources
                    packed_texts = {s.text for s in packed_sources}
                    vector_results = [r for r in vector_results if r.get("text") in packed_texts]

                # Final limit to retrieval_top_k
                vector_results = vector_results[:self.retrieval_top_k]

                # Retrieval evaluation (CRAG-style self-evaluation)
                relevance_hint = None
                if settings.retrieval_evaluation_enabled and self.llm_client is not None:
                    try:
                        if self._retrieval_evaluator is None:
                            self._retrieval_evaluator = RetrievalEvaluator(self.llm_client)
                        eval_result = await self._retrieval_evaluator.evaluate(user_input, vector_results)
                        if eval_result == "NO_MATCH":
                            logger.info("Retrieval evaluation: NO_MATCH for query '%s'", user_input)
                            relevance_hint = "Note: The retrieved documents may not be directly relevant to your query."
                        elif eval_result == "AMBIGUOUS":
                            logger.warning("Retrieval evaluation: AMBIGUOUS for query '%s'", user_input)
                    except Exception as e:
                        logger.warning("Retrieval evaluation failed: %s", e)
            except Exception as exc:
                fallback_reason = str(exc)
                vector_results = []

        relevant_chunks = self._filter_relevant(vector_results)

        # Context distillation (post-filter, pre-prompt-build)
        if settings.context_distillation_enabled and relevant_chunks:
            pre_count = len(relevant_chunks)
            relevant_chunks = await self._distill_context(relevant_chunks, user_input, eval_result)
            logger.info(
                "Context distillation: %d chunks → %d chunks after dedup",
                pre_count, len(relevant_chunks)
            )

        # Supersession check: warn if retrieved files have newer versions
        if relevant_chunks:
            supersession_warning = await self._check_supersession(relevant_chunks)
            if supersession_warning:
                if relevance_hint:
                    relevance_hint = supersession_warning + "\n" + relevance_hint
                else:
                    relevance_hint = supersession_warning

        if fallback_reason:
            logger.warning("Vector search fallback triggered: %s", fallback_reason)
            yield {
                "type": "fallback",
                "reason": fallback_reason,
                "results": [],
                "total": 0,
                "fallback": True,
            }
        try:
            memories = await asyncio.to_thread(
                self.memory_store.search_memories,
                user_input,
                self.retrieval_top_k,
                vault_id=vault_id,
            )
        except Exception as exc:
            logger.error("Memory search failed: %s", exc)
            memories = []

        messages = self._build_messages(user_input, chat_history, relevant_chunks, memories, relevance_hint)

        if stream:
            try:
                async for chunk in self.llm_client.chat_completion_stream(messages):
                    yield {"type": "content", "content": chunk}
            except LLMError as exc:
                yield {"type": "error", "message": str(exc), "code": "LLM_ERROR"}
                return
        else:
            try:
                content = await self.llm_client.chat_completion(messages)
            except LLMError as exc:
                raise RAGEngineError(f"LLM chat failed: {exc}") from exc
            yield {"type": "content", "content": content}

        # Build retrieval debug info
        retrieval_debug: Dict[str, Any] = {
            "max_distance_threshold": self.max_distance_threshold,
            "vector_metric": self.vector_metric,
            "retrieval_top_k": self.retrieval_top_k,
        }
        
        # Determine score_type based on retrieval method
        if reranking_applied:
            score_type = "rerank_score"
        elif self.hybrid_search_enabled:
            score_type = "hybrid_rrf"
        else:
            score_type = "dense_distance"
        
        yield {
            "type": "done",
            "sources": [self._source_metadata(c) for c in relevant_chunks],
            "memories_used": [mem.content for mem in memories],
            "retrieval_debug": retrieval_debug,
            "score_type": score_type,
        }

    async def _distill_context(
        self,
        sources: List[RAGSource],
        query: str,
        eval_result: str,
    ) -> List[RAGSource]:
        """Run context distillation (sentence dedup + optional LLM synthesis)."""
        from app.services.context_distiller import ContextDistiller
        if self._context_distiller is None:
            self._context_distiller = ContextDistiller(
                self.embedding_service,
                self.llm_client if settings.context_distillation_synthesis_enabled else None,
            )
        return await self._context_distiller.distill(query, sources, eval_result)

    async def _check_supersession(self, sources: List[RAGSource]) -> Optional[str]:
        """Query SQLite to check if any retrieved files have been superseded by newer versions."""
        file_ids = list({src.file_id for src in sources if src.file_id})
        if not file_ids:
            return None
        try:
            placeholders = ",".join("?" * len(file_ids))
            sql = (
                f"SELECT file_name FROM files "
                f"WHERE supersedes_file_id IN ({placeholders}) AND status='indexed'"
            )

            def _query() -> list:
                pool = _get_pool()
                with pool.connection() as conn:
                    rows = conn.execute(sql, file_ids).fetchall()
                    return rows

            rows = await asyncio.to_thread(_query)
            if rows:
                newer_names = [r[0] for r in rows]
                logger.warning(
                    "Supersession warning: retrieved file_ids %s have been superseded by %s",
                    file_ids, newer_names
                )
                return (
                    "\u26a0\ufe0f Note: One or more retrieved documents may have been superseded by a "
                    "newer version in the knowledge base. Verify currency of information where critical."
                )
        except Exception as exc:
            logger.warning("Supersession check failed (suppressed): %s", exc)
        return None

    def _normalize_metadata(self, metadata: Any) -> Dict[str, Any]:
        """Ensure metadata is a dict, parsing JSON string if needed."""
        if isinstance(metadata, dict):
            return metadata
        if isinstance(metadata, str):
            try:
                parsed = json.loads(metadata)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Failed to parse metadata JSON: %s", exc)
                pass
        return {}

    def _filter_relevant(self, results: List[Dict[str, Any]]) -> List[RAGSource]:
        sources: List[RAGSource] = []
        distances: List[float] = []
        
        # DEBUG: Log pre-filtering state
        input_count = len(results)
        logger.debug(
            "Filtering: input_results=%d, max_distance_threshold=%s",
            input_count,
            self.max_distance_threshold
        )
        if results:
            first_distances = [r.get("_distance", r.get("score")) for r in results[:5]]
            logger.debug("First few _distance values: %s", first_distances)
        
        for record in results:
            # Use _distance from LanceDB (lower is better for cosine)
            has_distance = "_distance" in record
            distance = record.get("_distance")
            if distance is None:
                score = record.get("score")
                if score is None:
                    score = 1.0
                distance = score
            
            distances.append(distance)
            
            # Use max_distance_threshold if set (new field)
            # Otherwise use relevance_threshold (legacy field) for backward compatibility
            threshold = self.max_distance_threshold
            if threshold is None:
                threshold = self.relevance_threshold
            
            # Determine if we should skip this record based on threshold
            # For _distance (lower=better): skip if distance > threshold
            # For score (higher=better): skip if score < threshold
            should_skip = False
            if threshold is not None:
                if has_distance:
                    # Using _distance (lower is better)
                    should_skip = distance > threshold
                else:
                    # Using score (higher is better, for backward compatibility)
                    should_skip = distance < threshold
            
            if should_skip:
                continue
                
            sources.append(
                RAGSource(
                    text=record.get("text", ""),
                    file_id=record.get("file_id", ""),
                    score=distance,  # Store distance (lower=better) or score (higher=better)
                    metadata=self._normalize_metadata(record.get("metadata")),
                )
            )
        
        # Log retrieval stats
        if distances:
            initial_count = len(distances)
            filtered_count = len(sources)
            min_dist = min(distances)
            max_dist = max(distances)
            mean_dist = sum(distances) / len(distances)
            
            threshold_str = f"{self.max_distance_threshold:.3f}" if self.max_distance_threshold is not None else "None"
            logger.info(
                "Vector search: initial=%d, filtered=%d, min=%.3f, max=%.3f, mean=%.3f, threshold=%s",
                initial_count, filtered_count, min_dist, max_dist, mean_dist, threshold_str
            )
        
        # Apply window expansion if enabled
        if self.retrieval_window > 0:
            sources = self._expand_window(sources)
        
        # DEBUG: Log filtering results summary
        filtered_count = len(sources)
        if input_count > 0 and filtered_count == 0:
            # All chunks exceeded threshold - return empty list (NO_MATCH)
            logger.warning(
                "All %d chunks exceeded max_distance_threshold - returning NO_MATCH",
                input_count
            )
            return []
        logger.debug("Filtering complete: %d results returned", len(sources))

        return sources
    
    def _normalize_uid_for_dedup(self, uid: str) -> str:
        """
        Normalize a chunk UID for deduplication by stripping scale suffix.
        
        Converts multi-scale UIDs like "file_id_512_0" to "file_id_0" format
        for deduplication purposes. This ensures the same chunk at different
        scales is treated as the same document.
        
        Args:
            uid: Chunk UID in format "file_id_idx" or "file_id_scale_idx"
            
        Returns:
            Normalized UID for deduplication
        """
        if "_" not in uid:
            return uid
        
        parts = uid.rsplit("_", 2)
        if len(parts) == 3:
            # 3-part format: file_id_scale_index -> file_id_index
            file_id, _scale, chunk_index = parts
            return f"{file_id}_{chunk_index}"
        # 2-part format: file_id_index (already normalized)
        return uid

    def _expand_window(self, sources: List[RAGSource]) -> List[RAGSource]:
        """
        Expand search results by fetching adjacent chunks (N±window).
        
        Args:
            sources: Initial list of RAGSource chunks from vector search
            
        Returns:
            Expanded list of RAGSource chunks with adjacent context
        """
        if not sources:
            return sources
        
        window = self.retrieval_window
        
        # Group sources by (file_id, chunk_scale) to avoid cross-scale window mixing
        file_chunks: Dict[str, List[RAGSource]] = {}
        for source in sources:
            # Use (file_id, chunk_scale) as key to separate different scales
            chunk_scale = source.metadata.get("chunk_scale", "default")
            if chunk_scale == "default" or chunk_scale is None:
                group_key = source.file_id
            else:
                group_key = f"{source.file_id}_{chunk_scale}"
            
            if group_key not in file_chunks:
                file_chunks[group_key] = []
            file_chunks[group_key].append(source)
        
        # Collect all chunk_uids to fetch (including adjacent chunks)
        chunk_uids_to_fetch: List[str] = []
        
        # Track chunk_count per file from the initial results
        file_chunk_indices: Dict[str, List[int]] = {}
        
        for group_key, file_sources in file_chunks.items():
            # Extract file_id and chunk_scale from group_key
            # group_key is either "file_id" (default) or "file_id_scale"
            if "_" in group_key:
                parts = group_key.rsplit("_", 1)
                if len(parts) == 2:
                    file_id, chunk_scale = parts
                else:
                    file_id = group_key
                    chunk_scale = "default"
            else:
                file_id = group_key
                chunk_scale = "default"
            
            indices = [int(s.metadata.get("chunk_index", 0)) for s in file_sources]
            
            # Calculate adjacent indices for each chunk
            for chunk_index in indices:
                # Calculate window range: [chunk_index - window, chunk_index + window]
                start_idx = max(0, chunk_index - window)
                end_idx = chunk_index + window
                
                # Generate UIDs for all indices in the window
                # Handle both 2-part (file_id_idx) and 3-part (file_id_scale_idx) formats
                for idx in range(start_idx, end_idx + 1):
                    if chunk_scale != "default":
                        chunk_uid = f"{file_id}_{chunk_scale}_{idx}"
                    else:
                        chunk_uid = f"{file_id}_{idx}"
                    chunk_uids_to_fetch.append(chunk_uid)
        
        # Fetch adjacent chunks from vector store
        if chunk_uids_to_fetch:
            adjacent_chunks = self.vector_store.get_chunks_by_uid(chunk_uids_to_fetch)
            
            # Create a lookup for adjacent chunks by their uid
            adjacent_lookup: Dict[str, Dict[str, Any]] = {}
            for chunk in adjacent_chunks:
                chunk_id = chunk.get("id", "")
                if chunk_id:
                    adjacent_lookup[chunk_id] = chunk
            
            # Build expanded sources list
            expanded_sources: List[RAGSource] = []
            seen_uids: set = set()
            
            # First, add the original sources
            for source in sources:
                chunk_index = source.metadata.get("chunk_index", 0)
                # Handle both 2-part and 3-part UID formats based on chunk_scale
                chunk_scale = source.metadata.get("chunk_scale", "default")
                if chunk_scale and chunk_scale != "default":
                    uid = f"{source.file_id}_{chunk_scale}_{chunk_index}"
                else:
                    uid = f"{source.file_id}_{chunk_index}"
                # Use normalized UID for deduplication (strips scale suffix)
                normalized_uid = self._normalize_uid_for_dedup(uid)
                if normalized_uid not in seen_uids:
                    expanded_sources.append(source)
                    seen_uids.add(normalized_uid)
            
            # Then, add adjacent chunks that aren't already in the results
            for chunk_uid in chunk_uids_to_fetch:
                # Use normalized UID for deduplication (strips scale suffix)
                normalized_uid = self._normalize_uid_for_dedup(chunk_uid)
                if normalized_uid in seen_uids:
                    continue
                
                # Parse uid to get file_id, chunk_scale (optional), and chunk_index
                # Handle both 3-part (file_id_scale_index) and 2-part (file_id_index) formats
                parts = chunk_uid.rsplit("_", 2)
                
                if len(parts) == 3:
                    # 3-part format: file_id_scale_index
                    file_id, chunk_scale, chunk_index_str = parts
                elif len(parts) == 2:
                    # 2-part format: file_id_index
                    file_id, chunk_index_str = parts
                    chunk_scale = None
                else:
                    continue
                
                try:
                    chunk_index = int(chunk_index_str)
                except ValueError:
                    continue
                
                # Check if this chunk exists in adjacent_lookup
                if chunk_uid in adjacent_lookup:
                    chunk = adjacent_lookup[chunk_uid]
                    
                    # Calculate score based on distance (lower is better for cosine)
                    has_distance = "_distance" in chunk
                    distance = chunk.get("_distance")
                    if distance is None:
                        score = chunk.get("score")
                        if score is None:
                            score = 1.0
                        distance = score
                    
                    # Get metadata and add chunk_scale if parsed from 3-part UID
                    metadata = self._normalize_metadata(chunk.get("metadata"))
                    if chunk_scale:
                        metadata["chunk_scale"] = chunk_scale
                    
                    expanded_source = RAGSource(
                        text=chunk.get("text", ""),
                        file_id=file_id,
                        score=distance,
                        metadata=metadata,
                    )
                    expanded_sources.append(expanded_source)
                    seen_uids.add(normalized_uid)
            
            # Sort by (file_id, chunk_index)
            def sort_key(source: RAGSource) -> tuple:
                chunk_index = source.metadata.get("chunk_index", 0)
                try:
                    chunk_index = int(chunk_index)
                except (ValueError, TypeError):
                    chunk_index = 0
                return (source.file_id, chunk_index)
            
            expanded_sources.sort(key=sort_key)
            
            # Cap to retrieval_top_k total
            if len(expanded_sources) > self.retrieval_top_k:
                expanded_sources = expanded_sources[:self.retrieval_top_k]
            
            return expanded_sources
        
        # If no adjacent chunks fetched, return original sources
        return sources

    def _pack_context_by_token_budget(self, chunks: List[RAGSource], max_tokens: int = 6000) -> List[RAGSource]:
        """
        Pack context chunks by token budget, respecting max token limit.

        Args:
            chunks: List of RAGSource chunks to pack
            max_tokens: Maximum tokens allowed (default 6000)

        Returns:
            List of RAGSource chunks that fit within the token budget
        """
        packed, token_count = [], 0
        for chunk in chunks:
            chunk_tokens = len(chunk.text) // 4
            if token_count + chunk_tokens > max_tokens and packed:
                break
            packed.append(chunk)
            token_count += chunk_tokens
        return packed

    def _build_system_prompt(self) -> str:
        CITATION_INSTRUCTION = (
            "You must cite sources inline using [Source: <name>] format. "
            "Every substantive claim must be backed by a citation from the provided context. "
            "If multiple sources support a claim, cite all of them. "
            "If you cannot find a source for a claim, do not make that claim."
        )
        return (
            "You are KnowledgeVault, a highly accurate assistant. "
            "Answer questions using ONLY the context documents provided. "
            "If the context does not contain sufficient information to answer, "
            "say so clearly — do not guess or use outside knowledge. "
            "When you cite sources, reference them by name using [Source: <name>]. "
            "Always respond in clear, well-formatted markdown — use headings, bullet points, "
            "numbered lists, bold, and code blocks where appropriate. "
            "Never output raw JSON, XML, or any structured data format unless the user "
            "explicitly requests it. "
            + CITATION_INSTRUCTION
        )

    def _build_messages(
        self,
        user_input: str,
        chat_history: List[Dict[str, Any]],
        chunks: List[RAGSource],
        memories: List[MemoryRecord],
        relevance_hint: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        context_sections = [self._format_chunk(ch) for ch in chunks]
        memory_context = [mem.content for mem in memories if mem.content]

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
        ]
        for entry in chat_history:
            messages.append({"role": entry["role"], "content": entry["content"]})

        context = "\n\n".join(filter(None, context_sections))
        
        # DEBUG: Log context being sent to LLM
        logger.info(
            "Building messages: chunks=%d, memories=%d, context_length=%d",
            len(chunks),
            len(memory_context),
            len(context)
        )
        if context:
            logger.debug("Context preview (first 500 chars): %s", context[:500])
        
        user_content_parts = []
        if relevance_hint:
            user_content_parts.append(relevance_hint)
        if context:
            user_content_parts.append(f"Context:\n{context}")
        else:
            user_content_parts.append(
                "No relevant context documents were found for this query. "
                "Please let the user know you cannot find relevant information "
                "in the knowledge base and avoid speculating."
            )
        user_content = "\n\n".join(user_content_parts) + "\n\n"

        memory_text = "\n".join(memory_context)
        if memory_text:
            user_content += f"Memories:\n{memory_text}\n\n"

        user_content += f"Question: {user_input}"
        messages.append({"role": "user", "content": user_content})
        return messages

    def _format_chunk(self, chunk: RAGSource) -> str:
        source_title = chunk.metadata.get("source_file") or chunk.metadata.get("section_title") or "document"
        return f"Source {source_title}:\n{chunk.text}"

    def _source_metadata(self, chunk: RAGSource) -> Dict[str, Any]:
        filename = (
            chunk.metadata.get("source_file")
            or chunk.metadata.get("filename")
            or chunk.metadata.get("section_title")
            or "Unknown document"
        )
        return {
            "id": chunk.file_id,
            "file_id": chunk.file_id,
            "filename": filename,
            "snippet": chunk.text[:300] if chunk.text else "",
            "score": chunk.score,
            "metadata": chunk.metadata,
        }
