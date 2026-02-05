"""
Chat API routes for RAG-based conversational interface.

Provides streaming and non-streaming chat endpoints that leverage
the RAG engine for context-aware responses.
"""
from typing import Any, Dict, List

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.rag_engine import RAGEngine


router = APIRouter()


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    history: List[Dict[str, Any]] = []
    stream: bool = False


class ChatResponse(BaseModel):
    """Response model for non-streaming chat endpoint."""
    content: str
    sources: List[Dict[str, Any]]
    memories_used: List[str]


async def stream_chat_response(
    message: str,
    history: List[Dict[str, Any]],
) -> StreamingResponse:
    """
    Generate a streaming chat response using SSE format.
    
    Yields SSE events with JSON data chunks from the RAG engine.
    Each event is formatted as: data: {json}\n\n
    Ends with a done event containing sources and memories_used.
    """
    async def event_generator():
        rag_engine = RAGEngine()
        collected_content = []
        sources = []
        memories_used = []
        
        async for chunk in rag_engine.query(message, history, stream=True):
            chunk_type = chunk.get("type")
            
            if chunk_type == "content":
                content = chunk.get("content", "")
                collected_content.append(content)
                # Yield content chunk as SSE event
                import json
                yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
            elif chunk_type == "done":
                sources = chunk.get("sources", [])
                memories_used = chunk.get("memories_used", [])
        
        # Yield final done event with sources and memories
        import json
        yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'memories_used': memories_used})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


async def non_stream_chat_response(
    message: str,
    history: List[Dict[str, Any]],
) -> ChatResponse:
    """
    Generate a non-streaming chat response.
    
    Collects all chunks from the RAG engine and returns a complete
    response with content, sources, and memories used.
    """
    rag_engine = RAGEngine()
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
async def chat(request: ChatRequest):
    """
    Chat endpoint for RAG-based conversational interface.
    
    Args:
        request: ChatRequest containing message, optional history, and stream flag
        
    Returns:
        If stream=True: StreamingResponse with SSE events (data: {json})
        If stream=False: ChatResponse with content, sources, memories_used
    """
    if request.stream:
        return await stream_chat_response(request.message, request.history)
    else:
        return await non_stream_chat_response(request.message, request.history)
