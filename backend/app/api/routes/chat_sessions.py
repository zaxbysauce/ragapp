"""Chat session persistence routes for user-scoped conversations.

This module provides endpoints for managing chat sessions and their messages,
with user-based access control ensuring users can only access their own sessions.
"""

import asyncio
import json
import sqlite3
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.api.deps import get_current_active_user, get_db, get_user_accessible_vault_ids

router = APIRouter(prefix="/chat/sessions", tags=["chat-sessions"])


class MessageCreate(BaseModel):
    """Request model for creating a new message."""

    role: str  # "user" or "assistant"
    content: str
    sources: Optional[List[dict]] = None


class MessageResponse(BaseModel):
    """Response model for a chat message."""

    id: int
    role: str
    content: str
    sources: Optional[List[dict]]
    created_at: str


class SessionCreate(BaseModel):
    """Request model for creating a new chat session."""

    title: str
    vault_id: int


class SessionUpdate(BaseModel):
    """Request model for updating a chat session."""

    title: str


class SessionResponse(BaseModel):
    """Response model for a chat session."""

    id: int
    title: str
    vault_id: int
    user_id: int
    created_at: str
    updated_at: str
    messages: Optional[List[MessageResponse]] = None


async def _get_session_by_id(db: sqlite3.Connection, session_id: int) -> Optional[dict]:
    """Fetch a session by ID, returning None if not found."""
    cursor = db.execute(
        "SELECT id, title, vault_id, user_id, created_at, updated_at "
        "FROM chat_sessions WHERE id = ?",
        (session_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "title": row[1],
        "vault_id": row[2],
        "user_id": row[3],
        "created_at": row[4],
        "updated_at": row[5],
    }


async def _verify_session_ownership(session: dict, user: dict) -> None:
    """Verify that the session belongs to the user, raise 403 if not."""
    if session["user_id"] != user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you do not own this session",
        )


@router.get("/", response_model=List[SessionResponse])
async def list_sessions(
    user: dict = Depends(get_current_active_user),
    db: sqlite3.Connection = Depends(get_db),
) -> List[SessionResponse]:
    """List all chat sessions for the current user.

    Returns a list of sessions without their messages.
    """

    def _fetch_sessions():
        cursor = db.execute(
            "SELECT id, title, vault_id, user_id, created_at, updated_at "
            "FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user["id"],),
        )
        rows = cursor.fetchall()
        return [
            SessionResponse(
                id=row[0],
                title=row[1],
                vault_id=row[2],
                user_id=row[3],
                created_at=row[4],
                updated_at=row[5],
                messages=None,
            )
            for row in rows
        ]

    return await asyncio.to_thread(_fetch_sessions)


@router.post("/", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    session_data: SessionCreate,
    user: dict = Depends(get_current_active_user),
    db: sqlite3.Connection = Depends(get_db),
) -> SessionResponse:
    """Create a new chat session for the current user."""
    # Validate that user has access to the vault
    accessible_vaults = await get_user_accessible_vault_ids(user, db)
    if accessible_vaults and session_data.vault_id not in accessible_vaults:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you do not have permission to access this vault",
        )

    def _create():
        cursor = db.execute(
            "INSERT INTO chat_sessions (title, vault_id, user_id) VALUES (?, ?, ?)",
            (session_data.title, session_data.vault_id, user["id"]),
        )
        db.commit()
        session_id = cursor.lastrowid
        # Fetch the created session
        cursor = db.execute(
            "SELECT id, title, vault_id, user_id, created_at, updated_at "
            "FROM chat_sessions WHERE id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        return SessionResponse(
            id=row[0],
            title=row[1],
            vault_id=row[2],
            user_id=row[3],
            created_at=row[4],
            updated_at=row[5],
            messages=None,
        )

    return await asyncio.to_thread(_create)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: int,
    user: dict = Depends(get_current_active_user),
    db: sqlite3.Connection = Depends(get_db),
) -> SessionResponse:
    """Get a specific chat session with all its messages."""

    def _fetch_session():
        # Fetch session
        cursor = db.execute(
            "SELECT id, title, vault_id, user_id, created_at, updated_at "
            "FROM chat_sessions WHERE id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        session = {
            "id": row[0],
            "title": row[1],
            "vault_id": row[2],
            "user_id": row[3],
            "created_at": row[4],
            "updated_at": row[5],
        }

        # Fetch messages
        cursor = db.execute(
            "SELECT id, role, content, sources, created_at "
            "FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        message_rows = cursor.fetchall()
        messages = []
        for msg_row in message_rows:
            sources = None
            if msg_row[3]:
                try:
                    sources = json.loads(msg_row[3])
                except json.JSONDecodeError:
                    sources = None
            messages.append(
                MessageResponse(
                    id=msg_row[0],
                    role=msg_row[1],
                    content=msg_row[2],
                    sources=sources,
                    created_at=msg_row[4],
                )
            )

        return SessionResponse(
            id=session["id"],
            title=session["title"],
            vault_id=session["vault_id"],
            user_id=session["user_id"],
            created_at=session["created_at"],
            updated_at=session["updated_at"],
            messages=messages,
        )

    result = await asyncio.to_thread(_fetch_session)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Verify ownership
    if result.user_id != user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you do not own this session",
        )

    return result


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: int,
    user: dict = Depends(get_current_active_user),
    db: sqlite3.Connection = Depends(get_db),
) -> None:
    """Delete a chat session and all its messages."""

    def _delete():
        # Check if session exists and get user_id
        cursor = db.execute(
            "SELECT user_id FROM chat_sessions WHERE id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None  # Session not found
        return row[0]

    session_user_id = await asyncio.to_thread(_delete)

    if session_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    if session_user_id != user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you do not own this session",
        )

    def _do_delete():
        # Delete session (messages will be deleted via CASCADE)
        db.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        db.commit()

    await asyncio.to_thread(_do_delete)


@router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: int,
    session_data: SessionUpdate,
    user: dict = Depends(get_current_active_user),
    db: sqlite3.Connection = Depends(get_db),
) -> SessionResponse:
    """Update a chat session's title."""

    def _update():
        # Check if session exists and get user_id
        cursor = db.execute(
            "SELECT user_id FROM chat_sessions WHERE id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None  # Session not found
        return row[0]

    session_user_id = await asyncio.to_thread(_update)

    if session_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    if session_user_id != user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you do not own this session",
        )

    def _do_update():
        # Update session
        db.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE id = ?",
            (session_data.title, session_id),
        )
        db.commit()

        # Fetch updated session
        cursor = db.execute(
            "SELECT id, title, vault_id, user_id, created_at, updated_at "
            "FROM chat_sessions WHERE id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        return SessionResponse(
            id=row[0],
            title=row[1],
            vault_id=row[2],
            user_id=row[3],
            created_at=row[4],
            updated_at=row[5],
            messages=None,
        )

    return await asyncio.to_thread(_do_update)


@router.post(
    "/{session_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_message(
    session_id: int,
    message_data: MessageCreate,
    user: dict = Depends(get_current_active_user),
    db: sqlite3.Connection = Depends(get_db),
) -> MessageResponse:
    """Append a message to a chat session."""

    def _verify_and_create():
        # Check if session exists and get user_id
        cursor = db.execute(
            "SELECT user_id FROM chat_sessions WHERE id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None, None  # Session not found

        session_user_id = row[0]

        # Validate role - only 'user' and 'assistant' allowed per DB constraint
        if message_data.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid role. Must be 'user' or 'assistant'",
            )

        # Insert message
        sources_json = (
            json.dumps(message_data.sources) if message_data.sources else None
        )
        cursor = db.execute(
            "INSERT INTO chat_messages (session_id, role, content, sources) "
            "VALUES (?, ?, ?, ?)",
            (session_id, message_data.role, message_data.content, sources_json),
        )
        message_id = cursor.lastrowid

        # Update session's updated_at timestamp
        db.execute(
            "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,),
        )
        db.commit()

        # Fetch created message
        cursor = db.execute(
            "SELECT id, role, content, sources, created_at "
            "FROM chat_messages WHERE id = ?",
            (message_id,),
        )
        msg_row = cursor.fetchone()

        sources = None
        if msg_row[3]:
            try:
                sources = json.loads(msg_row[3])
            except json.JSONDecodeError:
                sources = None

        message = MessageResponse(
            id=msg_row[0],
            role=msg_row[1],
            content=msg_row[2],
            sources=sources,
            created_at=msg_row[4],
        )

        return session_user_id, message

    result = await asyncio.to_thread(_verify_and_create)

    if result[0] is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    session_user_id, message = result

    if session_user_id != user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you do not own this session",
        )

    return message


@router.get("/{session_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    session_id: int,
    user: dict = Depends(get_current_active_user),
    db: sqlite3.Connection = Depends(get_db),
) -> List[MessageResponse]:
    """Get all messages for a chat session."""

    def _fetch():
        # Check if session exists and get user_id
        cursor = db.execute(
            "SELECT user_id FROM chat_sessions WHERE id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None  # Session not found

        session_user_id = row[0]

        # Fetch messages
        cursor = db.execute(
            "SELECT id, role, content, sources, created_at "
            "FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        message_rows = cursor.fetchall()
        messages = []
        for msg_row in message_rows:
            sources = None
            if msg_row[3]:
                try:
                    sources = json.loads(msg_row[3])
                except json.JSONDecodeError:
                    sources = None
            messages.append(
                MessageResponse(
                    id=msg_row[0],
                    role=msg_row[1],
                    content=msg_row[2],
                    sources=sources,
                    created_at=msg_row[4],
                )
            )

        return session_user_id, messages

    result = await asyncio.to_thread(_fetch)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    session_user_id, messages = result

    if session_user_id != user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you do not own this session",
        )

    return messages
