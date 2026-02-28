"""Retrieval-augmented generation engine orchestration."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from app.config import settings
from app.services.embeddings import EmbeddingService, EmbeddingError
from app.services.llm_client import LLMClient, LLMError
from app.services.memory_store import MemoryRecord, MemoryStore
from app.services.vector_store import VectorStore
from app.services.query_transformer import QueryTransformer
from app.services.retrieval_evaluator import RetrievalEvaluator
from app.utils.fusion import rrf_fuse


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

        fallback_reason: Optional[str] = None
        vector_results: List[Dict[str, Any]]
        relevance_hint: Optional[str] = None
        
        # DEBUG: Log pre-vector-search state
        logger.debug(
            "RAG query: retrieval_top_k=%d, vault_id=%s, vector_store_connected=%s",
            self.retrieval_top_k,
            vault_id,
            getattr(self.vector_store, 'is_connected', lambda: 'unknown')()
        )
        
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
                    results = await asyncio.to_thread(
                        self.vector_store.search,
                        embedding,
                        int(top_k_value),
                        vault_id=str(vault_id) if vault_id is not None else None,
                        query_text=user_input if embedding is query_embeddings[0] else "",
                        hybrid=self.hybrid_search_enabled and embedding is query_embeddings[0],
                        hybrid_alpha=self.hybrid_alpha,
                    )
                    all_results.append(results)

                # Fuse results from all query variants using RRF
                if len(all_results) > 1:
                    vector_results = rrf_fuse(all_results, k=60, limit=fetch_k)
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

                # Filter by distance threshold
                filtered_results = []
                for record in vector_results:
                    distance = record.get("_distance")
                    if distance is not None and self.max_distance_threshold:
                        if distance > self.max_distance_threshold:
                            continue
                    filtered_results.append(record)
                vector_results = filtered_results

                # Stage 2: Reranking (if enabled)
                if self.reranking_enabled and self.reranking_service and vector_results:
                    try:
                        vector_results = await self.reranking_service.rerank(
                            query=user_input,
                            chunks=vector_results,
                            top_n=self.reranker_top_n,
                        )
                    except Exception as e:
                        logger.warning(f"Reranking failed, using original results: {e}")

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
                settings.max_context_chunks,
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
        
        yield {
            "type": "done",
            "sources": [self._source_metadata(c) for c in relevant_chunks],
            "memories_used": [mem.content for mem in memories],
            "retrieval_debug": retrieval_debug,
        }

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
            
            logger.info(
                "Vector search: initial=%d, filtered=%d, min=%.3f, max=%.3f, mean=%.3f, threshold=%.3f",
                initial_count, filtered_count, min_dist, max_dist, mean_dist, self.max_distance_threshold
            )
        
        # Apply window expansion if enabled
        if self.retrieval_window > 0:
            sources = self._expand_window(sources)
        
        # DEBUG: Log filtering results summary
        filtered_count = len(sources)
        if input_count > 0 and filtered_count == 0:
            logger.warning(
                "Filtering issue: %d input results but 0 after filtering. "
                "Check max_distance_threshold (%.3f) - distances may exceed threshold.",
                input_count,
                self.max_distance_threshold
            )
        logger.debug("Filtering complete: %d results returned", filtered_count)
        
        return sources
    
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
                if uid not in seen_uids:
                    expanded_sources.append(source)
                    seen_uids.add(uid)
            
            # Then, add adjacent chunks that aren't already in the results
            for chunk_uid in chunk_uids_to_fetch:
                if chunk_uid in seen_uids:
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
                    seen_uids.add(chunk_uid)
            
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

    def _build_system_prompt(self) -> str:
        return (
            "You are KnowledgeVault, a highly accurate assistant that references sources when "
            "answering questions. Cite the relevant documents or memories by name."
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
            messages.append(entry)

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
            user_content_parts.append("No relevant documents found for this query.")
        user_content = "\n\n".join(user_content_parts) + "\n\n"

        memory_text = "\n".join(memory_context)
        if memory_text:
            user_content += f"Memories:\n{memory_text}\n\n"

        user_content += f"Question: {user_input}"
        messages.append({"role": "user", "content": user_content})
        return messages

    def _format_chunk(self, chunk: RAGSource) -> str:
        source_title = chunk.metadata.get("source_file") or chunk.metadata.get("section_title") or "document"
        return f"Source {source_title} (score: {chunk.score:.2f}):\n{chunk.text}"

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
