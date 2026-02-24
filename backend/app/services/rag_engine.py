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
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        self.memory_store = memory_store or MemoryStore()
        self.llm_client = llm_client or LLMClient()

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

        # Legacy field support (deprecated)
        self.relevance_threshold = settings.rag_relevance_threshold
        self.top_k = settings.vector_top_k
        self.maintenance_mode = settings.maintenance_mode

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

        try:
            query_embedding = await self.embedding_service.embed_single(user_input)
        except EmbeddingError as exc:
            error_msg = f"Unable to encode query: {exc}"
            if stream:
                yield {"type": "error", "message": error_msg, "code": "EMBEDDING_ERROR"}
                return
            raise RAGEngineError(error_msg) from exc

        fallback_reason: Optional[str] = None
        vector_results: List[Dict[str, Any]]
        if self.maintenance_mode:
            fallback_reason = "RAG index is under maintenance"
            vector_results = []
        else:
            try:
                top_k_value = self.top_k if self.top_k is not None else self.retrieval_top_k
                vector_results = await asyncio.to_thread(
                    self.vector_store.search,
                    query_embedding,
                    int(top_k_value),
                    vault_id=str(vault_id) if vault_id is not None else None,
                )
                
                # Log vector search results
                logger.info(
                    "Vector search: vault_id=%s, top_k=%d, results=%d, distances=%s",
                    vault_id,
                    top_k_value,
                    len(vector_results),
                    [r.get("_distance") for r in vector_results[:3]] if vector_results else "N/A"
                )
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

        messages = self._build_messages(user_input, chat_history, relevant_chunks, memories)

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
        
        # Group sources by file_id to track chunk indices per document
        file_chunks: Dict[str, List[RAGSource]] = {}
        for source in sources:
            file_id = source.file_id
            if file_id not in file_chunks:
                file_chunks[file_id] = []
            file_chunks[file_id].append(source)
        
        # Collect all chunk_uids to fetch (including adjacent chunks)
        chunk_uids_to_fetch: List[str] = []
        
        # Track chunk_count per file from the initial results
        file_chunk_indices: Dict[str, List[int]] = {}
        
        for file_id, file_sources in file_chunks.items():
            indices = [int(s.metadata.get("chunk_index", 0)) for s in file_sources]
            file_chunk_indices[file_id] = indices
            
            # Calculate adjacent indices for each chunk
            for chunk_index in indices:
                # Calculate window range: [chunk_index - window, chunk_index + window]
                start_idx = max(0, chunk_index - window)
                end_idx = chunk_index + window
                
                # Generate UIDs for all indices in the window
                for idx in range(start_idx, end_idx + 1):
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
                uid = f"{source.file_id}_{chunk_index}"
                if uid not in seen_uids:
                    expanded_sources.append(source)
                    seen_uids.add(uid)
            
            # Then, add adjacent chunks that aren't already in the results
            for chunk_uid in chunk_uids_to_fetch:
                if chunk_uid in seen_uids:
                    continue
                
                # Parse uid to get file_id and chunk_index
                parts = chunk_uid.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                
                file_id, chunk_index_str = parts
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
                    
                    expanded_source = RAGSource(
                        text=chunk.get("text", ""),
                        file_id=file_id,
                        score=distance,
                        metadata=self._normalize_metadata(chunk.get("metadata")),
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
    ) -> List[Dict[str, str]]:
        context_sections = [self._format_chunk(ch) for ch in chunks]
        memory_context = [mem.content for mem in memories if mem.content]

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
        ]
        for entry in chat_history:
            messages.append(entry)

        context = "\n\n".join(filter(None, context_sections))
        if context:
            user_content = f"Context:\n{context}\n\n"
        else:
            user_content = "No relevant documents found for this query.\n\n"

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
