"""
Chat API routes for RAG-based conversational interface.

Provides streaming and non-streaming chat endpoints that leverage
the RAG engine for context-aware responses.
"""

import asyncio
import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import (
    get_db,
    get_rag_engine,
    require_vault_permission,
    get_current_active_user,
    get_user_accessible_vault_ids,
)
from app.security import require_auth
from app.services.rag_engine import RAGEngine, RAGEngineError


router = APIRouter()

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str
    history: List[Dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    vault_id: int = 1


class ChatResponse(BaseModel):
    """Response model for non-streaming chat endpoint."""

    content: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    memories_used: List[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatStreamRequest(BaseModel):
    messages: List[ChatMessage]
    vault_id: int = 1


class CreateSessionRequest(BaseModel):
    """Request model for creating a new chat session."""

    title: Optional[str] = None
    vault_id: int = 1


class AddMessageRequest(BaseModel):
    """Request model for adding a message to a chat session."""

    role: str
    content: str
    sources: Optional[List[dict]] = None


class UpdateSessionRequest(BaseModel):
    """Request model for updating a chat session title."""

    title: str


def stream_chat_response(
    message: str,
    history: List[Dict[str, Any]],
    rag_engine: Optional[RAGEngine],
    vault_id: int = 1,
) -> StreamingResponse:
    """
    Generate a streaming chat response using SSE format.

    Yields SSE events with JSON data chunks from the RAG engine.
    Each event is formatted as: data: {json}\n\n
    Ends with a done event containing sources and memories_used.
    """
    if rag_engine is None:

        async def error_generator():
            yield f"data: {json.dumps({'type': 'error', 'message': 'RAG engine not available', 'code': 'SERVICE_UNAVAILABLE'})}\n\n"

        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream",
        )

    async def event_generator():
        collected_content = []
        sources = []
        memories_used = []

        # SSE comment keeps the connection alive during model cold-start
        # (proxies / browsers drop idle connections after ~60 s)
        yield ": ping\n\n"

        try:
            async for chunk in rag_engine.query(
                message, history, stream=True, vault_id=vault_id
            ):
                chunk_type = chunk.get("type")

                if chunk_type == "content":
                    content = chunk.get("content", "")
                    collected_content.append(content)
                    yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                elif chunk_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': chunk.get('message', 'Chat stream failed'), 'code': chunk.get('code', 'UNKNOWN_ERROR')})}\n\n"
                    return
                elif chunk_type == "done":
                    sources = chunk.get("sources", [])
                    memories_used = chunk.get("memories_used", [])
        except Exception as e:
            logger.error(
                "Chat stream failed: message_len=%d, history_len=%d, exception=%s, error=%s",
                len(message),
                len(history),
                type(e).__name__,
                str(e),
                exc_info=False,
            )
            yield f"data: {json.dumps({'type': 'error', 'message': 'An error occurred while processing your request', 'code': 'INTERNAL_ERROR'})}\n\n"
            return

        # Yield final done event with sources and memories
        yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'memories_used': memories_used})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


async def non_stream_chat_response(
    message: str,
    history: List[Dict[str, Any]],
    rag_engine: Optional[RAGEngine],
    vault_id: int = 1,
) -> ChatResponse:
    """
    Generate a non-streaming chat response.

    Collects all chunks from the RAG engine and returns a complete
    response with content, sources, and memories used.
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not available")

    collected_content = []
    sources = []
    memories_used = []

    try:
        async for chunk in rag_engine.query(
            message, history, stream=False, vault_id=vault_id
        ):
            chunk_type = chunk.get("type")

            if chunk_type == "content":
                collected_content.append(chunk.get("content", ""))
            elif chunk_type == "done":
                sources = chunk.get("sources", [])
                memories_used = chunk.get("memories_used", [])
    except RAGEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    full_content = "".join(collected_content)

    return ChatResponse(
        content=full_content,
        sources=sources,
        memories_used=memories_used,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
    user: dict = Depends(require_vault_permission("read")),
):
    """
    Chat endpoint for RAG-based conversational interface.

    Args:
        request: ChatRequest containing message, optional history, and stream flag

    Returns:
        ChatResponse with content, sources, memories_used

    Raises:
        HTTPException: If stream=True is requested (use /chat/stream instead)
    """
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not supported on this endpoint. Use /chat/stream for streaming responses.",
        )
    return await non_stream_chat_response(
        request.message, request.history, rag_engine, vault_id=request.vault_id
    )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatStreamRequest,
    user: dict = Depends(get_current_active_user),
    db=Depends(get_db),
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """Streaming chat endpoint that accepts a sequence of chat messages."""
    # Validate vault_id is in user's accessible list
    user_role = user.get("role", "")

    # superadmin/admin can access any vault
    if user_role not in ("superadmin", "admin"):
        accessible = await get_user_accessible_vault_ids(user, db)
        if request.vault_id not in accessible:
            raise HTTPException(status_code=403, detail="Vault access denied")

    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required")

    last_message = request.messages[-1]
    if last_message.role.lower() != "user":
        raise HTTPException(
            status_code=400, detail="The last message must be from the user"
        )

    history = [msg.model_dump(exclude_none=True) for msg in request.messages[:-1]]
    return stream_chat_response(
        last_message.content, history, rag_engine, vault_id=request.vault_id
    )


# ============================================================================
# Chat Session History Management Endpoints
# ============================================================================


@router.get("/chat/sessions")
async def list_sessions(
    vault_id: Optional[int] = Query(None),
    conn: sqlite3.Connection = Depends(get_db),
    user: dict = Depends(require_vault_permission("read")),
):
    """
    List all chat sessions, optionally filtered by vault_id.

    Returns sessions sorted by updated_at DESC with message count for each session.
    """
    # Build single JOIN query with optional vault_id filter to avoid N+1
    if vault_id is not None:
        query = """
            SELECT s.id, s.vault_id, s.title, s.created_at, s.updated_at, COUNT(m.id) as message_count
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.id
            WHERE s.vault_id = ?
            GROUP BY s.id
            ORDER BY s.updated_at DESC
        """
        params = (vault_id,)
    else:
        query = """
            SELECT s.id, s.vault_id, s.title, s.created_at, s.updated_at, COUNT(m.id) as message_count
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
        """
        params = ()

    result = await asyncio.to_thread(conn.execute, query, params)
    rows = await asyncio.to_thread(result.fetchall)

    # Map rows to dicts (message_count is now the 6th column, index 5)
    sessions_with_count = []
    for row in rows:
        sessions_with_count.append(
            {
                "id": row[0],
                "vault_id": row[1],
                "title": row[2],
                "created_at": row[3],
                "updated_at": row[4],
                "message_count": row[5],
            }
        )

    return {"sessions": sessions_with_count}


@router.get("/chat/sessions/{session_id}")
async def get_session(
    session_id: int,
    conn: sqlite3.Connection = Depends(get_db),
    user: dict = Depends(require_vault_permission("read")),
):
    """
    Get a specific chat session with all its messages.

    Returns session details and messages ordered by created_at ASC.
    Parses the sources field from JSON string to list.
    """
    # Get session
    session_query = "SELECT id, vault_id, title, created_at, updated_at FROM chat_sessions WHERE id = ?"
    session_result = await asyncio.to_thread(conn.execute, session_query, (session_id,))
    session_row = await asyncio.to_thread(session_result.fetchone)

    if session_row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get messages
    messages_query = "SELECT id, role, content, sources, created_at FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC"
    messages_result = await asyncio.to_thread(
        conn.execute, messages_query, (session_id,)
    )
    message_rows = await asyncio.to_thread(messages_result.fetchall)

    # Parse messages with JSON sources
    messages = []
    for msg_row in message_rows:
        sources = None
        if msg_row[3]:
            try:
                sources = json.loads(msg_row[3])
            except json.JSONDecodeError:
                sources = []

        messages.append(
            {
                "id": msg_row[0],
                "role": msg_row[1],
                "content": msg_row[2],
                "sources": sources,
                "created_at": msg_row[4],
            }
        )

    return {
        "id": session_row[0],
        "vault_id": session_row[1],
        "title": session_row[2],
        "created_at": session_row[3],
        "updated_at": session_row[4],
        "messages": messages,
    }


@router.post("/chat/sessions")
async def create_session(
    request: CreateSessionRequest,
    conn: sqlite3.Connection = Depends(get_db),
    user: dict = Depends(require_vault_permission("read")),
):
    """
    Create a new chat session.

    Returns the created session with its ID.
    """
    query = "INSERT INTO chat_sessions (vault_id, title, created_at, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
    cursor = await asyncio.to_thread(
        conn.execute, query, (request.vault_id, request.title)
    )
    await asyncio.to_thread(conn.commit)

    # Get the created session
    session_id = cursor.lastrowid
    select_query = "SELECT id, vault_id, title, created_at, updated_at FROM chat_sessions WHERE id = ?"
    result = await asyncio.to_thread(conn.execute, select_query, (session_id,))
    row = await asyncio.to_thread(result.fetchone)

    return {
        "id": row[0],
        "vault_id": row[1],
        "title": row[2],
        "created_at": row[3],
        "updated_at": row[4],
    }


async def _auto_name_session(first_message: str) -> str:
    """
    Generate a concise, meaningful title for a chat session based on the first message.

    Uses a simple heuristic approach to extract key topics:
    - For questions: Extract the subject being asked about
    - For commands: Use the action verb + object
    - For general text: Extract noun phrases

    Returns a title limited to 50 characters.
    """
    import re

    # Clean and normalize the message
    text = first_message.strip()

    # Remove common prefixes
    prefixes = [
        r"^please\s+",
        r"^can\s+you\s+",
        r"^could\s+you\s+",
        r"^what\s+(is|are)\s+",
        r"^how\s+(do|does|can|to)\s+",
        r"^tell\s+me\s+about\s+",
        r"^explain\s+",
        r"^summarize\s+",
        r"^compare\s+",
        r"^analyze\s+",
    ]

    cleaned = text.lower()
    for prefix in prefixes:
        cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)

    # Extract key phrases
    # Look for quoted text first (often contains the main topic)
    quoted = re.findall(r'["\']([^"\']+)["\']', text)
    if quoted:
        candidate = quoted[0]
    else:
        # Extract the first meaningful phrase (up to first punctuation or 10 words)
        words = cleaned.split()
        if len(words) <= 3:
            candidate = " ".join(words)
        else:
            # Take first 4-6 words that form a coherent phrase
            candidate = " ".join(words[:5])

    # Capitalize first letter
    candidate = candidate.strip().capitalize()

    # Remove trailing punctuation
    candidate = re.sub(r"[.,;:!?]+$", "", candidate)

    # Limit length and add ellipsis if needed
    if len(candidate) > 47:
        candidate = candidate[:47].rsplit(" ", 1)[0] + "..."
    elif len(candidate) < 10 and len(text) > 10:
        # If too short, use more context
        candidate = text[:47].strip()
        if len(candidate) > 47:
            candidate = candidate[:47].rsplit(" ", 1)[0] + "..."

    return candidate if candidate else "New Chat"


@router.post("/chat/sessions/{session_id}/messages")
async def add_message(
    session_id: int,
    request: AddMessageRequest,
    conn: sqlite3.Connection = Depends(get_db),
    user: dict = Depends(require_vault_permission("read")),
):
    """
    Add a message to a chat session.

    If this is the first message and the session has no title,
    auto-titles the session using intelligent topic extraction.
    Updates the session's updated_at timestamp.
    """
    # Verify session exists
    session_query = "SELECT id, title, vault_id FROM chat_sessions WHERE id = ?"
    session_result = await asyncio.to_thread(conn.execute, session_query, (session_id,))
    session_row = await asyncio.to_thread(session_result.fetchone)

    if session_row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if this is the first message
    count_query = "SELECT COUNT(*) FROM chat_messages WHERE session_id = ?"
    count_result = await asyncio.to_thread(conn.execute, count_query, (session_id,))
    message_count_row = await asyncio.to_thread(count_result.fetchone)
    is_first_message = message_count_row[0] == 0

    # Auto-title if first message and session has no title
    if is_first_message and session_row[1] is None:
        auto_title = await _auto_name_session(request.content)
        update_title_query = "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
        await asyncio.to_thread(
            conn.execute, update_title_query, (auto_title, session_id)
        )

    # Serialize sources to JSON
    sources_json = json.dumps(request.sources) if request.sources else None

    # Insert message
    insert_query = """
        INSERT INTO chat_messages (session_id, role, content, sources, created_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    """
    cursor = await asyncio.to_thread(
        conn.execute,
        insert_query,
        (session_id, request.role, request.content, sources_json),
    )

    # Update session's updated_at
    update_query = (
        "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?"
    )
    await asyncio.to_thread(conn.execute, update_query, (session_id,))
    await asyncio.to_thread(conn.commit)

    # Get the created message
    message_id = cursor.lastrowid
    select_query = (
        "SELECT id, role, content, sources, created_at FROM chat_messages WHERE id = ?"
    )
    result = await asyncio.to_thread(conn.execute, select_query, (message_id,))
    row = await asyncio.to_thread(result.fetchone)

    # Parse sources
    sources = None
    if row[3]:
        try:
            sources = json.loads(row[3])
        except json.JSONDecodeError:
            sources = []

    return {
        "id": row[0],
        "role": row[1],
        "content": row[2],
        "sources": sources,
        "created_at": row[4],
    }


@router.put("/chat/sessions/{session_id}")
async def update_session(
    session_id: int,
    request: UpdateSessionRequest,
    conn: sqlite3.Connection = Depends(get_db),
    user: dict = Depends(require_vault_permission("write")),
):
    """
    Update a chat session's title.

    Updates the session's title and updated_at timestamp.
    """
    # Verify session exists
    check_query = "SELECT id FROM chat_sessions WHERE id = ?"
    check_result = await asyncio.to_thread(conn.execute, check_query, (session_id,))
    check_row = await asyncio.to_thread(check_result.fetchone)

    if check_row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Update session
    update_query = "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
    await asyncio.to_thread(conn.execute, update_query, (request.title, session_id))
    await asyncio.to_thread(conn.commit)

    # Get updated session
    select_query = "SELECT id, vault_id, title, created_at, updated_at FROM chat_sessions WHERE id = ?"
    result = await asyncio.to_thread(conn.execute, select_query, (session_id,))
    row = await asyncio.to_thread(result.fetchone)

    return {
        "id": row[0],
        "vault_id": row[1],
        "title": row[2],
        "created_at": row[3],
        "updated_at": row[4],
    }


@router.delete("/chat/sessions/{session_id}")
async def delete_session(
    session_id: int,
    conn: sqlite3.Connection = Depends(get_db),
    user: dict = Depends(require_vault_permission("write")),
):
    """
    Delete a chat session.

    The CASCADE constraint will automatically delete all messages
    associated with the session.
    """
    # Verify session exists
    check_query = "SELECT id FROM chat_sessions WHERE id = ?"
    check_result = await asyncio.to_thread(conn.execute, check_query, (session_id,))
    check_row = await asyncio.to_thread(check_result.fetchone)

    if check_row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Delete session (CASCADE will delete messages)
    delete_query = "DELETE FROM chat_sessions WHERE id = ?"
    await asyncio.to_thread(conn.execute, delete_query, (session_id,))
    await asyncio.to_thread(conn.commit)

    return {"status": "deleted", "session_id": session_id}
