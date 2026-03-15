import httpx
import json
import sqlite3
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional
from app.config import settings
from app.api.deps import get_csrf_manager, get_db, get_current_active_user
from app.security import CSRFManager, issue_csrf_token

router = APIRouter()


class SettingsUpdate(BaseModel):
    # Legacy fields (deprecated, still supported for backward compatibility)
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    max_context_chunks: Optional[int] = None
    rag_relevance_threshold: Optional[float] = None
    vector_top_k: Optional[int] = None

    # New character-based fields
    chunk_size_chars: Optional[int] = None
    chunk_overlap_chars: Optional[int] = None
    retrieval_top_k: Optional[int] = None
    vector_metric: Optional[str] = None
    max_distance_threshold: Optional[float] = None
    embedding_doc_prefix: Optional[str] = None
    embedding_query_prefix: Optional[str] = None
    retrieval_window: Optional[int] = None
    embedding_batch_size: Optional[int] = None

    # Reranker config
    reranker_url: Optional[str] = None
    reranker_model: Optional[str] = None
    reranking_enabled: Optional[bool] = None
    reranker_top_n: Optional[int] = None
    initial_retrieval_top_k: Optional[int] = None

    # Hybrid search config
    hybrid_search_enabled: Optional[bool] = None
    hybrid_alpha: Optional[float] = None

    # Advanced retrieval feature config
    query_transformation_enabled: Optional[bool] = None
    retrieval_evaluation_enabled: Optional[bool] = None
    context_distillation_enabled: Optional[bool] = None
    context_distillation_dedup_threshold: Optional[float] = None
    context_distillation_synthesis_enabled: Optional[bool] = None
    hyde_enabled: Optional[bool] = None
    tri_vector_search_enabled: Optional[bool] = None
    flag_embedding_url: Optional[str] = None
    sparse_search_max_candidates: Optional[int] = None
    retrieval_recency_weight: Optional[float] = None

    # Feature flags (still supported)
    auto_scan_enabled: Optional[bool] = None
    auto_scan_interval_minutes: Optional[int] = None
    maintenance_mode: Optional[bool] = None
    enable_model_validation: Optional[bool] = None

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v):
        if v is not None and v <= 0:
            raise ValueError("chunk_size must be a positive integer")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v):
        if v is not None and v < 0:
            raise ValueError("chunk_overlap must be a non-negative integer")
        return v

    @field_validator("max_context_chunks")
    @classmethod
    def validate_max_context_chunks(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_context_chunks must be a positive integer")
        return v

    @field_validator("rag_relevance_threshold")
    @classmethod
    def validate_rag_relevance_threshold(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("rag_relevance_threshold must be between 0 and 1")
        return v

    @field_validator("vector_top_k")
    @classmethod
    def validate_vector_top_k(cls, v):
        if v is not None and v <= 0:
            raise ValueError("vector_top_k must be a positive integer")
        return v

    @field_validator("chunk_size_chars")
    @classmethod
    def validate_chunk_size_chars(cls, v):
        if v is not None and v <= 0:
            raise ValueError("chunk_size_chars must be a positive integer")
        return v

    @field_validator("chunk_overlap_chars")
    @classmethod
    def validate_chunk_overlap_chars(cls, v):
        if v is not None and v < 0:
            raise ValueError("chunk_overlap_chars must be a non-negative integer")
        return v

    @field_validator("retrieval_top_k")
    @classmethod
    def validate_retrieval_top_k(cls, v):
        if v is not None and v <= 0:
            raise ValueError("retrieval_top_k must be a positive integer")
        return v

    @field_validator("max_distance_threshold")
    @classmethod
    def validate_max_distance_threshold(cls, v):
        if v is not None and v < 0:
            raise ValueError("max_distance_threshold must be a non-negative number")
        return v

    @field_validator("retrieval_window")
    @classmethod
    def validate_retrieval_window(cls, v):
        if v is not None and v <= 0:
            raise ValueError("retrieval_window must be a positive integer")
        return v

    @field_validator("auto_scan_interval_minutes")
    @classmethod
    def validate_auto_scan_interval_minutes(cls, v):
        if v is not None and v <= 0:
            raise ValueError("auto_scan_interval_minutes must be a positive integer")
        return v

    @field_validator("embedding_batch_size")
    @classmethod
    def validate_embedding_batch_size(cls, v):
        if v is not None and (v < 1 or v > 2048):
            raise ValueError("embedding_batch_size must be between 1 and 2048")
        return v

    @field_validator("reranker_top_n")
    @classmethod
    def validate_reranker_top_n(cls, v):
        if v is not None and v <= 0:
            raise ValueError("reranker_top_n must be a positive integer")
        return v

    @field_validator("initial_retrieval_top_k")
    @classmethod
    def validate_initial_retrieval_top_k(cls, v):
        if v is not None and v <= 0:
            raise ValueError("initial_retrieval_top_k must be a positive integer")
        return v

    @field_validator("hybrid_alpha")
    @classmethod
    def validate_hybrid_alpha(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("hybrid_alpha must be between 0 and 1")
        return v

    @field_validator("context_distillation_dedup_threshold")
    @classmethod
    def validate_context_distillation_dedup_threshold(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("context_distillation_dedup_threshold must be between 0 and 1")
        return v

    @field_validator("retrieval_recency_weight")
    @classmethod
    def validate_retrieval_recency_weight(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("retrieval_recency_weight must be between 0 and 1")
        return v

    @field_validator("sparse_search_max_candidates")
    @classmethod
    def validate_sparse_search_max_candidates(cls, v):
        if v is not None and v < 10:
            raise ValueError("sparse_search_max_candidates must be at least 10")
        return v

    @model_validator(mode="after")
    def validate_chunk_overlap_less_than_size(self):
        chunk_overlap = self.chunk_overlap
        chunk_size = self.chunk_size
        if chunk_overlap is not None and chunk_size is not None and chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


ALLOWED_FIELDS = [
    "chunk_size",
    "chunk_overlap",
    "max_context_chunks",
    "auto_scan_enabled",
    "auto_scan_interval_minutes",
    "rag_relevance_threshold",
    "vector_top_k",
    "chunk_size_chars",
    "chunk_overlap_chars",
    "retrieval_top_k",
    "vector_metric",
    "max_distance_threshold",
    "embedding_doc_prefix",
    "embedding_query_prefix",
    "retrieval_window",
    "embedding_batch_size",
    "reranker_url",
    "reranker_model",
    "reranking_enabled",
    "reranker_top_n",
    "initial_retrieval_top_k",
    "hybrid_search_enabled",
    "hybrid_alpha",
    "query_transformation_enabled",
    "retrieval_evaluation_enabled",
    "context_distillation_enabled",
    "context_distillation_dedup_threshold",
    "context_distillation_synthesis_enabled",
    "hyde_enabled",
    "tri_vector_search_enabled",
    "flag_embedding_url",
    "sparse_search_max_candidates",
    "retrieval_recency_weight",
]


def _persist_settings(conn: sqlite3.Connection, update: SettingsUpdate) -> None:
    """Save changed settings to the settings_kv table."""
    for field in ALLOWED_FIELDS:
        value = getattr(update, field)
        if value is not None:
            conn.execute(
                "INSERT OR REPLACE INTO settings_kv (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (field, json.dumps(value)),
            )
    conn.commit()


class SettingsResponse(BaseModel):
    """Public settings response - excludes secrets."""

    # Server config (safe to expose)
    port: int
    data_dir: str

    # Ollama config
    ollama_embedding_url: str
    ollama_chat_url: str

    # Model config
    embedding_model: str
    chat_model: str

    # Document processing (user-configurable)
    chunk_size: Optional[int] = None  # Legacy, deprecated
    chunk_overlap: Optional[int] = None  # Legacy, deprecated
    max_context_chunks: int

    # RAG config (user-configurable)
    rag_relevance_threshold: Optional[float] = None  # Legacy, deprecated
    vector_top_k: Optional[int] = None  # Legacy, deprecated

    # New character-based fields
    chunk_size_chars: int
    chunk_overlap_chars: int
    retrieval_top_k: int
    vector_metric: str
    max_distance_threshold: Optional[float] = None
    embedding_doc_prefix: str
    embedding_query_prefix: str
    retrieval_window: int

    # Feature flags
    maintenance_mode: bool
    auto_scan_enabled: bool
    auto_scan_interval_minutes: int
    enable_model_validation: bool

    # Embedding config
    embedding_batch_size: int

    # Reranker config
    reranker_url: str = ""
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranking_enabled: bool = False
    reranker_top_n: int = 5
    initial_retrieval_top_k: int = 20

    # Hybrid search config
    hybrid_search_enabled: bool = True
    hybrid_alpha: float = 0.5

    # Advanced retrieval feature config
    query_transformation_enabled: bool = False
    retrieval_evaluation_enabled: bool = False
    context_distillation_enabled: bool = False
    context_distillation_dedup_threshold: float = 0.92
    context_distillation_synthesis_enabled: bool = False
    hyde_enabled: bool = False
    tri_vector_search_enabled: bool = False
    flag_embedding_url: str = ""
    sparse_search_max_candidates: int = 1000
    retrieval_recency_weight: float = 0.1

    # Limits (safe to expose)
    max_file_size_mb: int
    allowed_extensions: list[str]

    # CORS (safe to expose)
    backend_cors_origins: list[str]

    @field_validator("data_dir", mode="before")
    @classmethod
    def convert_path_to_str(cls, v):
        return str(v)


def _apply_settings_update(update: SettingsUpdate) -> SettingsResponse:
    """Shared logic to apply settings update and return updated settings."""
    updated = False
    for field in ALLOWED_FIELDS:
        value = getattr(update, field)
        if value is not None:
            setattr(settings, field, value)
            updated = True
    if not updated:
        raise HTTPException(status_code=400, detail="No valid fields provided for update")
    # Convert settings object to dict for validation
    settings_dict = {
        "port": settings.port,
        "data_dir": str(settings.data_dir),
        "ollama_embedding_url": settings.ollama_embedding_url,
        "ollama_chat_url": settings.ollama_chat_url,
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "max_context_chunks": settings.max_context_chunks,
        "rag_relevance_threshold": settings.rag_relevance_threshold,
        "vector_top_k": settings.vector_top_k,
        "chunk_size_chars": settings.chunk_size_chars,
        "chunk_overlap_chars": settings.chunk_overlap_chars,
        "retrieval_top_k": settings.retrieval_top_k,
        "vector_metric": settings.vector_metric,
        "max_distance_threshold": settings.max_distance_threshold,
        "embedding_doc_prefix": settings.embedding_doc_prefix,
        "embedding_query_prefix": settings.embedding_query_prefix,
        "retrieval_window": settings.retrieval_window,
        "embedding_batch_size": settings.embedding_batch_size,
        "maintenance_mode": settings.maintenance_mode,
        "auto_scan_enabled": settings.auto_scan_enabled,
        "auto_scan_interval_minutes": settings.auto_scan_interval_minutes,
        "enable_model_validation": settings.enable_model_validation,
        "max_file_size_mb": settings.max_file_size_mb,
        "allowed_extensions": settings.allowed_extensions,
        "backend_cors_origins": settings.backend_cors_origins,
        "reranker_url": settings.reranker_url,
        "reranker_model": settings.reranker_model,
        "reranking_enabled": settings.reranking_enabled,
        "reranker_top_n": settings.reranker_top_n,
        "initial_retrieval_top_k": settings.initial_retrieval_top_k,
        "hybrid_search_enabled": settings.hybrid_search_enabled,
        "hybrid_alpha": settings.hybrid_alpha,
        "query_transformation_enabled": settings.query_transformation_enabled,
        "retrieval_evaluation_enabled": settings.retrieval_evaluation_enabled,
        "context_distillation_enabled": settings.context_distillation_enabled,
        "context_distillation_dedup_threshold": settings.context_distillation_dedup_threshold,
        "context_distillation_synthesis_enabled": settings.context_distillation_synthesis_enabled,
        "hyde_enabled": settings.hyde_enabled,
        "tri_vector_search_enabled": settings.tri_vector_search_enabled,
        "flag_embedding_url": settings.flag_embedding_url,
        "sparse_search_max_candidates": settings.sparse_search_max_candidates,
        "retrieval_recency_weight": settings.retrieval_recency_weight,
    }
    return SettingsResponse.model_validate(settings_dict)


@router.get("/settings", response_model=SettingsResponse)
def get_settings(
    user: dict = Depends(get_current_active_user),
):
    """Return current public settings dict (including embedding_batch_size)."""
    settings_dict = {
        "port": settings.port,
        "data_dir": str(settings.data_dir),
        "ollama_embedding_url": settings.ollama_embedding_url,
        "ollama_chat_url": settings.ollama_chat_url,
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "max_context_chunks": settings.max_context_chunks,
        "rag_relevance_threshold": settings.rag_relevance_threshold,
        "vector_top_k": settings.vector_top_k,
        "chunk_size_chars": settings.chunk_size_chars,
        "chunk_overlap_chars": settings.chunk_overlap_chars,
        "retrieval_top_k": settings.retrieval_top_k,
        "vector_metric": settings.vector_metric,
        "max_distance_threshold": settings.max_distance_threshold,
        "embedding_doc_prefix": settings.embedding_doc_prefix,
        "embedding_query_prefix": settings.embedding_query_prefix,
        "retrieval_window": settings.retrieval_window,
        "embedding_batch_size": settings.embedding_batch_size,
        "maintenance_mode": settings.maintenance_mode,
        "auto_scan_enabled": settings.auto_scan_enabled,
        "auto_scan_interval_minutes": settings.auto_scan_interval_minutes,
        "enable_model_validation": settings.enable_model_validation,
        "max_file_size_mb": settings.max_file_size_mb,
        "allowed_extensions": settings.allowed_extensions,
        "backend_cors_origins": settings.backend_cors_origins,
        "reranker_url": settings.reranker_url,
        "reranker_model": settings.reranker_model,
        "reranking_enabled": settings.reranking_enabled,
        "reranker_top_n": settings.reranker_top_n,
        "initial_retrieval_top_k": settings.initial_retrieval_top_k,
        "hybrid_search_enabled": settings.hybrid_search_enabled,
        "hybrid_alpha": settings.hybrid_alpha,
        "query_transformation_enabled": settings.query_transformation_enabled,
        "retrieval_evaluation_enabled": settings.retrieval_evaluation_enabled,
        "context_distillation_enabled": settings.context_distillation_enabled,
        "context_distillation_dedup_threshold": settings.context_distillation_dedup_threshold,
        "context_distillation_synthesis_enabled": settings.context_distillation_synthesis_enabled,
        "hyde_enabled": settings.hyde_enabled,
        "tri_vector_search_enabled": settings.tri_vector_search_enabled,
        "flag_embedding_url": settings.flag_embedding_url,
        "sparse_search_max_candidates": settings.sparse_search_max_candidates,
        "retrieval_recency_weight": settings.retrieval_recency_weight,
    }
    return SettingsResponse.model_validate(settings_dict)


@router.post("/settings")
def post_settings(
    update: SettingsUpdate,
    conn: sqlite3.Connection = Depends(get_db),
    user: dict = Depends(get_current_active_user),
):
    """Apply settings update and persist to database."""
    result = _apply_settings_update(update)
    _persist_settings(conn, update)
    return result


@router.put("/settings")
def put_settings(
    update: SettingsUpdate,
    conn: sqlite3.Connection = Depends(get_db),
    user: dict = Depends(get_current_active_user),
):
    """Apply settings update and persist to database."""
    result = _apply_settings_update(update)
    _persist_settings(conn, update)
    return result


@router.get("/csrf-token")
def get_csrf_token(
    response: Response,
    csrf_manager: CSRFManager = Depends(get_csrf_manager),
):
    token = issue_csrf_token(response, csrf_manager)
    return {"csrf_token": token}


@router.get("/settings/connection")
async def test_connection(
    user: dict = Depends(get_current_active_user),
):
    """Test connectivity to Ollama endpoints and reranker."""
    targets = {
        "embeddings": settings.ollama_embedding_url,
        "chat": settings.ollama_chat_url,
    }
    if settings.reranker_url:
        targets["reranker"] = settings.reranker_url
    if settings.tri_vector_search_enabled and settings.flag_embedding_url:
        targets["flag_embedding"] = settings.flag_embedding_url

    async with httpx.AsyncClient(timeout=5.0) as client:
        results = {}
        for name, url in targets.items():
            try:
                response = await client.get(url)
                results[name] = {"url": url, "status": response.status_code, "ok": response.status_code < 300}
            except Exception as exc:
                results[name] = {"url": url, "status": None, "ok": False, "error": str(exc)}
        
        # Only add local mode result if reranker wasn't tested (i.e., reranker_url was not set)
        if "reranker" not in results:
            results["reranker"] = {
                "url": "local (sentence-transformers)",
                "ok": True,
                "status": "local",
                "model": settings.reranker_model,
            }
    return results
