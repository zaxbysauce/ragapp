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
                vector_results = await asyncio.to_thread(
                    self.vector_store.search,
                    query_embedding,
                    self.top_k,
                    vault_id=str(vault_id) if vault_id is not None else None,
                )
            except Exception as exc:
                fallback_reason = str(exc)
                vector_results = []

        relevant_chunks = self._filter_relevant(vector_results)
        if not relevant_chunks and vector_results:
            fallback_record = vector_results[0]
            fallback_source = RAGSource(
                text=fallback_record.get("text", ""),
                file_id=fallback_record.get("file_id", ""),
                score=fallback_record.get("score") or 1.0,
                metadata=self._normalize_metadata(fallback_record.get("metadata")),
            )
            relevant_chunks = [fallback_source]
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

        yield {
            "type": "done",
            "sources": [self._source_metadata(c) for c in relevant_chunks],
            "memories_used": [mem.content for mem in memories],
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
        for record in results:
            score = record.get("score")
            if score is None:
                score = 1.0
            if score < self.relevance_threshold:
                continue
            sources.append(
                RAGSource(
                    text=record.get("text", ""),
                    file_id=record.get("file_id", ""),
                    score=score,
                    metadata=self._normalize_metadata(record.get("metadata")),
                )
            )
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
            user_content = ""

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
