"""
Memory API routes for CRUD operations on memories.

Provides endpoints for listing, creating, updating, deleting, and searching memories.
"""
import json
import logging
import sqlite3
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, validator

from app.models.database import get_db_connection
from app.services.memory_store import MemoryStore, MemoryRecord, MemoryStoreError
from app.config import settings


logger = logging.getLogger(__name__)


router = APIRouter()


def _normalize_tags(tags: Optional[str]) -> Optional[str]:
    """Normalize tags to a valid JSON string."""
    if tags is None:
        return None
    tags = tags.strip()
    if not tags:
        return None
    # If it looks like a JSON array, validate it
    if tags.startswith('['):
        try:
            parsed = json.loads(tags)
            if not isinstance(parsed, list):
                raise ValueError("Tags must be a JSON array")
            return json.dumps(parsed)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array for tags: {e}")
    # Otherwise, treat as comma-separated and convert to JSON array
    tag_list = [t.strip() for t in tags.split(',') if t.strip()]
    return json.dumps(tag_list) if tag_list else None


class MemoryCreateRequest(BaseModel):
    """Request model for creating a new memory."""
    content: str = Field(..., min_length=1, max_length=10000, description="Memory content")
    category: Optional[str] = Field(None, max_length=255, description="Optional category")
    tags: Optional[str] = Field(None, max_length=1000, description="Optional tags (JSON array or comma-separated)")
    source: Optional[str] = Field(None, max_length=500, description="Optional source reference")

    @validator('tags')
    def validate_tags(cls, v):
        if v is None:
            return v
        return _normalize_tags(v)


class MemoryUpdateRequest(BaseModel):
    """Request model for updating an existing memory."""
    content: Optional[str] = Field(None, min_length=1, max_length=10000, description="Memory content")
    category: Optional[str] = Field(None, max_length=255, description="Optional category")
    tags: Optional[str] = Field(None, max_length=1000, description="Optional tags")
    source: Optional[str] = Field(None, max_length=500, description="Optional source reference")

    @validator('tags')
    def validate_tags(cls, v):
        if v is None:
            return v
        return _normalize_tags(v)


class MemoryResponse(BaseModel):
    """Response model for a memory record."""
    id: int
    content: str
    category: Optional[str]
    tags: Optional[str]
    source: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]

    class Config:
        from_attributes = True


class MemoryListResponse(BaseModel):
    """Response model for listing memories."""
    memories: List[MemoryResponse]


class MemorySearchResponse(BaseModel):
    """Response model for memory search results."""
    results: List[MemoryResponse]


def _memory_record_to_response(record: MemoryRecord) -> MemoryResponse:
    """Convert a MemoryRecord to a MemoryResponse."""
    return MemoryResponse(
        id=record.id,
        content=record.content,
        category=record.category,
        tags=record.tags,
        source=record.source,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get("/memories", response_model=MemoryListResponse)
async def list_memories():
    """
    List all memories.
    
    Returns a list of all memories with their id, content, category, tags, source,
    created_at, and updated_at fields.
    """
    conn = get_db_connection(str(settings.sqlite_path))
    try:
        cursor = conn.execute(
            """
            SELECT id, content, category, tags, source, created_at, updated_at
            FROM memories
            ORDER BY created_at DESC
            """
        )
        rows = cursor.fetchall()
        
        memories = []
        for row in rows:
            memories.append(MemoryResponse(
                id=row[0],
                content=row[1],
                category=row[2],
                tags=row[3],
                source=row[4],
                created_at=row[5],
                updated_at=row[6],
            ))
        
        return MemoryListResponse(memories=memories)
    finally:
        conn.close()


@router.post("/memories", response_model=MemoryResponse)
async def create_memory(request: MemoryCreateRequest):
    """
    Create a new memory.
    
    Uses MemoryStore.add_memory to add a new memory to the database.
    """
    store = MemoryStore()
    try:
        record = store.add_memory(
            content=request.content,
            category=request.category,
            tags=request.tags,
            source=request.source,
        )
    except MemoryStoreError as e:
        logger.exception("MemoryStoreError in create_memory (content length: %d)", len(request.content))
        raise HTTPException(status_code=400, detail=str(e))
    except sqlite3.Error as e:
        logger.exception("Database error in create_memory (content length: %d)", len(request.content))
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        logger.exception("Unexpected error in create_memory (content length: %d)", len(request.content))
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
    
    return _memory_record_to_response(record)


@router.put("/memories/{memory_id}", response_model=MemoryResponse)
async def update_memory(memory_id: int, request: MemoryUpdateRequest):
    """
    Update an existing memory.
    
    Updates content, category, tags, and/or source fields in the database.
    Returns 404 if the memory is not found.
    """
    conn = get_db_connection(str(settings.sqlite_path))
    try:
        try:
            # Check if memory exists
            cursor = conn.execute("SELECT id FROM memories WHERE id = ?", (memory_id,))
            if cursor.fetchone() is None:
                raise HTTPException(status_code=404, detail=f"Memory with id {memory_id} not found")
            
            # Build update query dynamically based on provided fields
            update_fields = []
            params = []
            
            if request.content is not None:
                update_fields.append("content = ?")
                params.append(request.content)
            if request.category is not None:
                update_fields.append("category = ?")
                params.append(request.category)
            if request.tags is not None:
                update_fields.append("tags = ?")
                params.append(request.tags)
            if request.source is not None:
                update_fields.append("source = ?")
                params.append(request.source)
            
            if not update_fields:
                # No fields to update, just fetch and return current record
                cursor = conn.execute(
                    """
                    SELECT id, content, category, tags, source, created_at, updated_at
                    FROM memories WHERE id = ?
                    """,
                    (memory_id,)
                )
                row = cursor.fetchone()
                return MemoryResponse(
                    id=row[0],
                    content=row[1],
                    category=row[2],
                    tags=row[3],
                    source=row[4],
                    created_at=row[5],
                    updated_at=row[6],
                )
            
            # Add memory_id to params
            params.append(memory_id)
            
            # Execute update
            sql = f"""
                UPDATE memories
                SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """
            conn.execute(sql, params)
            conn.commit()
            
            # Fetch updated record
            cursor = conn.execute(
                """
                SELECT id, content, category, tags, source, created_at, updated_at
                FROM memories WHERE id = ?
                """,
                (memory_id,)
            )
            row = cursor.fetchone()
            
            return MemoryResponse(
                id=row[0],
                content=row[1],
                category=row[2],
                tags=row[3],
                source=row[4],
                created_at=row[5],
                updated_at=row[6],
            )
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()


@router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: int):
    """
    Delete a memory.
    
    Deletes the memory with the given id from the database.
    Returns 404 if the memory is not found.
    """
    conn = get_db_connection(str(settings.sqlite_path))
    try:
        # Check if memory exists
        cursor = conn.execute("SELECT id FROM memories WHERE id = ?", (memory_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail=f"Memory with id {memory_id} not found")
        
        # Delete the memory
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        
        return {"message": f"Memory {memory_id} deleted successfully"}
    finally:
        conn.close()


@router.get("/memories/search", response_model=MemorySearchResponse)
async def search_memories(
    query: str = Query(..., min_length=1, description="Search query string"),
    limit: int = Query(5, ge=1, le=100, description="Maximum number of results")
):
    """
    Search memories using full-text search.
    
    Uses MemoryStore.search_memories to search memories via FTS5.
    Returns matching memories ordered by relevance.
    """
    store = MemoryStore()
    try:
        records = store.search_memories(query=query, limit=limit)
    except MemoryStoreError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    results = [_memory_record_to_response(record) for record in records]
    return MemorySearchResponse(results=results)
