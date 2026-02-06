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


def _apply_settings_update(update: SettingsUpdate) -> dict:
    """Shared logic to apply settings update and return updated settings."""
    updated = False
    for field in ALLOWED_FIELDS:
        value = getattr(update, field)
        if value is not None:
            setattr(settings, field, value)
            updated = True
    if not updated:
        raise HTTPException(status_code=400, detail="No valid fields provided for update")
    return settings.model_dump()


@router.get("/settings")
def get_settings():
    return settings.model_dump()


@router.post("/settings")
def update_settings(update: SettingsUpdate):
    return _apply_settings_update(update)


@router.put("/settings")
def update_settings_put(update: SettingsUpdate):
    return _apply_settings_update(update)


@router.get("/csrf-token")
def get_csrf_token(
    response: Response,
    csrf_manager: CSRFManager = Depends(get_csrf_manager),
):
    token = issue_csrf_token(response, csrf_manager)
    return {"csrf_token": token}


