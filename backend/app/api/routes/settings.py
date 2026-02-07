import httpx
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional
from app.config import settings
from app.api.deps import get_csrf_manager
from app.security import CSRFManager, issue_csrf_token

router = APIRouter()


class SettingsUpdate(BaseModel):
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    max_context_chunks: Optional[int] = None
    auto_scan_enabled: Optional[bool] = None
    auto_scan_interval_minutes: Optional[int] = None
    rag_relevance_threshold: Optional[float] = None

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

    @field_validator("auto_scan_interval_minutes")
    @classmethod
    def validate_auto_scan_interval(cls, v):
        if v is not None and v <= 0:
            raise ValueError("auto_scan_interval_minutes must be a positive integer")
        return v

    @field_validator("rag_relevance_threshold")
    @classmethod
    def validate_rag_relevance_threshold(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("rag_relevance_threshold must be between 0 and 1")
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
]


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
    chunk_size: int
    chunk_overlap: int
    max_context_chunks: int

    # RAG config (user-configurable)
    rag_relevance_threshold: float
    vector_top_k: int

    # Feature flags
    maintenance_mode: bool
    auto_scan_enabled: bool
    auto_scan_interval_minutes: int
    enable_model_validation: bool

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
        "maintenance_mode": settings.maintenance_mode,
        "auto_scan_enabled": settings.auto_scan_enabled,
        "auto_scan_interval_minutes": settings.auto_scan_interval_minutes,
        "enable_model_validation": settings.enable_model_validation,
        "max_file_size_mb": settings.max_file_size_mb,
        "allowed_extensions": settings.allowed_extensions,
        "backend_cors_origins": settings.backend_cors_origins,
    }
    return SettingsResponse.model_validate(settings_dict)


@router.get("/settings", response_model=SettingsResponse)
def get_settings():
    """Get public settings - excludes secrets like admin_secret_token."""
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
        "maintenance_mode": settings.maintenance_mode,
        "auto_scan_enabled": settings.auto_scan_enabled,
        "auto_scan_interval_minutes": settings.auto_scan_interval_minutes,
        "enable_model_validation": settings.enable_model_validation,
        "max_file_size_mb": settings.max_file_size_mb,
        "allowed_extensions": settings.allowed_extensions,
        "backend_cors_origins": settings.backend_cors_origins,
    }
    return SettingsResponse.model_validate(settings_dict)


@router.post("/settings")
def update_settings(update: SettingsUpdate):
    """Update runtime settings.

    NOTE: Changes are applied in-memory only and will be lost on server restart.
    DB persistence is not yet implemented.
    """
    return _apply_settings_update(update)


@router.put("/settings")
def update_settings_put(update: SettingsUpdate):
    """Update runtime settings (PUT method).

    NOTE: Changes are applied in-memory only and will be lost on server restart.
    DB persistence is not yet implemented.
    """
    return _apply_settings_update(update)


@router.get("/csrf-token")
def get_csrf_token(
    response: Response,
    csrf_manager: CSRFManager = Depends(get_csrf_manager),
):
    token = issue_csrf_token(response, csrf_manager)
    return {"csrf_token": token}


@router.get("/settings/connection")
async def test_connection():
    targets = {
        "embeddings": settings.ollama_embedding_url,
        "chat": settings.ollama_chat_url,
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        results = {}
        for name, url in targets.items():
            try:
                response = await client.get(url)
                results[name] = {
                    "url": url,
                    "status": response.status_code,
                    "ok": response.status_code < 300,
                }
            except Exception as exc:
                results[name] = {
                    "url": url,
                    "status": None,
                    "ok": False,
                    "error": str(exc),
                }
    return results

