"""FastAPI dependency functions."""

from fastapi import Request, Depends

from app.config import Settings, settings
from app.services.llm_client import LLMClient
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.memory_store import MemoryStore
from app.services.rag_engine import RAGEngine
from app.services.secret_manager import SecretManager
from app.services.toggle_manager import ToggleManager
from app.security import get_csrf_manager


def get_settings() -> Settings:
    """Return the application settings."""
    return settings


def get_llm_client(request: Request) -> LLMClient:
    """Return the LLM client from app state."""
    return request.app.state.llm_client


def get_embedding_service(request: Request) -> EmbeddingService:
    """Return the embedding service from app state."""
    return request.app.state.embedding_service


def get_vector_store(request: Request) -> VectorStore:
    """Return the vector store from app state."""
    return request.app.state.vector_store


def get_memory_store(request: Request) -> MemoryStore:
    """Return the memory store from app state."""
    return request.app.state.memory_store


def get_rag_engine(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    memory_store: MemoryStore = Depends(get_memory_store),
    llm_client: LLMClient = Depends(get_llm_client),
) -> RAGEngine:
    """Return a new RAGEngine initialized with dependencies."""
    return RAGEngine(
        embedding_service=embedding_service,
        vector_store=vector_store,
        memory_store=memory_store,
        llm_client=llm_client,
    )


def get_toggle_manager(request: Request) -> ToggleManager:
    return request.app.state.toggle_manager


def get_secret_manager(request: Request) -> SecretManager:
    return request.app.state.secret_manager


def get_csrf_manager(request: Request):
    return request.app.state.csrf_manager
