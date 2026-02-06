"""
Chat API routes for RAG-based conversational interface.

Provides streaming and non-streaming chat endpoints that leverage
the RAG engine for context-aware responses.
"""
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import get_rag_engine
from app.services.rag_engine import RAGEngine


router = APIRouter()


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    history: List[Dict[str, Any]] = Field(default_factory=list)
    stream: bool = False


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


def stream_chat_response(
    message: str,
    history: List[Dict[str, Any]],
    rag_engine: Optional[RAGEngine],
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
        
        try:
            async for chunk in rag_engine.query(message, history, stream=True):
                chunk_type = chunk.get("type")
                
                if chunk_type == "content":
                    content = chunk.get("content", "")
                    collected_content.append(content)
                    # Yield content chunk as SSE event
                    yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                elif chunk_type == "done":
                    sources = chunk.get("sources", [])
                    memories_used = chunk.get("memories_used", [])
        except Exception as e:
            # Emit error event and terminate immediately to avoid additional events
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'code': 'INTERNAL_ERROR'})}\n\n"
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
) -> ChatResponse:
    """
    Generate a non-streaming chat response.

    Collects all chunks from the RAG engine and returns a complete
    response with content, sources, and memories used.
    """
    if rag_engine is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not available"
        )

    collected_content = []
    sources = []
    memories_used = []

    async for chunk in rag_engine.query(message, history, stream=False):
        chunk_type = chunk.get("type")
        
        if chunk_type == "content":
            collected_content.append(chunk.get("content", ""))
        elif chunk_type == "done":
            sources = chunk.get("sources", [])
            memories_used = chunk.get("memories_used", [])
    
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
            detail="Streaming is not supported on this endpoint. Use /chat/stream for streaming responses."
        )
    return await non_stream_chat_response(request.message, request.history, rag_engine)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatStreamRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """Streaming chat endpoint that accepts a sequence of chat messages."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required")

    last_message = request.messages[-1]
    if last_message.role.lower() != "user":
        raise HTTPException(status_code=400, detail="The last message must be from the user")

    history = [msg.model_dump(exclude_none=True) for msg in request.messages[:-1]]
    return stream_chat_response(last_message.content, history, rag_engine)
