# KnowledgeVault: Self-Hosted RAG Knowledge Base
## Implementation Plan for OpenCode Swarm

---

## Executive Summary

**KnowledgeVault** is a self-hosted web service enabling users to ingest thousands of technical documents across diverse formats (docx, xlsx, pptx, pdf, csv, sql, scripts, database schemas) into a local vector database, then interact with this knowledge through natural language chat powered by RAG (Retrieval-Augmented Generation). The system includes persistent memory for user-defined facts and context.

**Key Differentiator:** Structure-aware semantic chunking optimized for technical documentation, based on research showing this approach prevents the "document shredding" problem that plagues naive RAG implementations.

---

## Final Architecture Decisions

| Component | Decision |
|-----------|----------|
| **Architecture** | Self-hosted web service (FastAPI + React) |
| **Deployment** | Docker Compose (app only, Ollama external) |
| **Vector DB** | LanceDB (embedded, bundled) |
| **Memory DB** | SQLite with FTS5 (embedded, bundled) |
| **Document Processing** | Python + Unstructured (bundled in container) |
| **Chunking Strategy** | Semantic/structure-aware, 256-512 tokens |
| **Embedding Model** | nomic-embed-text (user provides via Ollama) |
| **Chat Model** | User's choice via Ollama/vLLM/LM Studio |
| **LLM Connection** | External Ollama (user manages separately) |
| **Document Storage** | Configurable server path, web upload + filesystem scan |
| **Knowledge Base** | Shared across all users |
| **Authentication** | None (user handles network security) |
| **UI Framework** | React + shadcn/ui + Tailwind + Material 3 principles |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Host Server                                     │
│         (Dual Xeon 5218, 380GB RAM, RTX A1000 8GB)                          │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                 KnowledgeVault Container                               │  │
│  │                                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    FastAPI Backend                               │  │  │
│  │  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐       │  │  │
│  │  │  │  Document │ │    RAG    │ │  Memory   │ │  Settings │       │  │  │
│  │  │  │ Processor │ │  Engine   │ │  Service  │ │  Service  │       │  │  │
│  │  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘       │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    React Frontend                                │  │  │
│  │  │         Material 3 + shadcn/ui + Tailwind CSS                   │  │  │
│  │  │    (Chat Panel, File Manager, Memory Panel, Settings)           │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  ┌──────────────────┐  ┌──────────────────┐                          │  │
│  │  │     LanceDB      │  │      SQLite      │                          │  │
│  │  │  (Vector Store)  │  │ (Memory + Meta)  │                          │  │
│  │  └──────────────────┘  └──────────────────┘                          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                               │
│            Volume Mount: /data/knowledgevault                                │
│                              │                                               │
└──────────────────────────────┼───────────────────────────────────────────────┘
                               │
                               ▼ HTTP (configurable endpoint)
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Ollama Service (External)                            │
│                    (User installs and manages separately)                    │
│                                                                              │
│    Embedding Model: nomic-embed-text (GPU - RTX A1000)                      │
│    Chat Model: Qwen 2.5 32B/72B (CPU - 380GB RAM)                           │
│                                                                              │
│    Endpoint: http://localhost:11434                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Backend** | FastAPI (Python 3.11+) | Async, fast, OpenAPI docs, excellent ecosystem |
| **Frontend** | React 18 + TypeScript | Industry standard, excellent tooling |
| **UI Components** | shadcn/ui + Tailwind CSS | Copy-paste ownership, highly customizable |
| **UI Design** | Material 3 Expressive principles | Research-backed UX, 4x faster element discovery |
| **Vector Database** | LanceDB | Embedded, zero-config, handles millions of vectors |
| **Memory/Metadata** | SQLite + FTS5 | Fast, embedded, full-text search built-in |
| **Document Parsing** | Unstructured | Handles 15+ formats, layout-aware parsing |
| **Chunking** | Custom semantic pipeline | Structure-aware, optimized for technical docs |
| **LLM Integration** | OpenAI-compatible API | Works with Ollama, vLLM, LM Studio, etc. |
| **Deployment** | Docker Compose | Single command deployment |

---

## Data Directory Structure

```
/data/knowledgevault/                    # Configurable via settings
├── documents/                           # Raw document storage
│   ├── uploads/                         # User uploads via web UI
│   └── library/                         # Admin can drop files here directly
├── processing/                          # Temp files during ingestion
├── lancedb/                             # Vector embeddings (auto-managed)
│   └── chunks.lance/                    # Chunk vectors and metadata
├── knowledgevault.db                    # SQLite: memories, settings, file index
└── logs/                                # Application logs
    └── knowledgevault.log
```

---

## RAG Implementation Strategy

Based on research from arxiv.org/html/2404.00657v1:

### Chunking Best Practices for Technical Documents

1. **NEVER use fixed character-count chunking** - Research shows this is "disastrous" for technical docs as it splits tables and separates headers from values

2. **Sentence-based embedding + paragraph retrieval** - Embed at sentence granularity but retrieve parent paragraphs for context

3. **Chunk size limits** - Chunks >200 words show spurious similarity scores. Target: 256-512 tokens

4. **Preserve structure** - Tables, code blocks, and lists should never be split

5. **Section awareness** - Track section headers and include them in chunk metadata

6. **Special handling for schemas** - Database DDL files need custom parsing to extract table/column definitions as discrete chunks

### Memory Detection Patterns

The system auto-detects memory storage intent via patterns:
- "remember that..."
- "don't forget..."
- "keep in mind..."
- "note that..."

---

## Project Structure

```
knowledgevault/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application entry
│   │   ├── config.py               # Settings management
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── chat.py         # Chat endpoints
│   │   │   │   ├── documents.py    # Document management
│   │   │   │   ├── memory.py       # Memory CRUD
│   │   │   │   ├── settings.py     # Settings endpoints
│   │   │   │   └── health.py       # Health check
│   │   │   └── deps.py             # Dependency injection
│   │   ├── services/
│   │   │   ├── document_processor.py
│   │   │   ├── chunking.py         # Semantic chunking pipeline
│   │   │   ├── embeddings.py       # Ollama embedding client
│   │   │   ├── vector_store.py     # LanceDB operations
│   │   │   ├── memory_store.py     # SQLite memory operations
│   │   │   ├── rag_engine.py       # RAG orchestration
│   │   │   └── llm_client.py       # Multi-provider LLM client
│   │   ├── models/
│   │   │   ├── schemas.py          # Pydantic models
│   │   │   └── database.py         # SQLite models
│   │   └── utils/
│   │       ├── file_utils.py
│   │       └── text_utils.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── pytest.ini
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/                 # shadcn/ui components
│   │   │   ├── layout/
│   │   │   ├── chat/
│   │   │   ├── documents/
│   │   │   ├── memory/
│   │   │   └── settings/
│   │   ├── hooks/
│   │   ├── lib/
│   │   ├── stores/                 # Zustand state management
│   │   ├── types/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── tailwind.config.js
│   ├── vite.config.ts
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Implementation Phases

### Phase 1: Project Foundation
- Docker Compose configuration
- Multi-stage Dockerfile
- Environment variable management
- Development hot-reload setup

### Phase 2: Backend Core Services
- Configuration management (Pydantic Settings)
- SQLite database setup with FTS5
- LanceDB vector store service

### Phase 3: Document Processing Pipeline
- Semantic chunking service (Unstructured library)
- Schema parser for SQL/DDL files
- Document processor with hash-based deduplication
- Background task processing

### Phase 4: LLM Integration
- Embedding service (Ollama client)
- LLM chat client (OpenAI-compatible)
- Streaming response support
- Health check endpoints

### Phase 5: RAG Engine
- Query embedding generation
- Vector similarity search
- Memory injection into context
- Prompt construction with source attribution
- Streaming response with sources

### Phase 6: API Routes
- FastAPI application with lifespan
- Chat endpoints (streaming + non-streaming)
- Document management endpoints
- Memory CRUD endpoints
- Settings and health endpoints

### Phase 7: Frontend Implementation
- React + TypeScript + Vite setup
- Material 3 design system via Tailwind
- shadcn/ui component integration
- Navigation rail layout
- Chat interface with streaming
- Document manager with drag-drop upload
- Memory panel
- Settings panel with connection status

---

## Deployment Configuration

### docker-compose.yml

```yaml
version: '3.8'

services:
  knowledgevault:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: knowledgevault
    ports:
      - "${PORT:-8080}:8080"
    volumes:
      - ${DATA_DIR:-./data}:/data/knowledgevault
    environment:
      - DATA_DIR=/data/knowledgevault
      - OLLAMA_EMBEDDING_URL=${OLLAMA_EMBEDDING_URL:-http://host.docker.internal:11434}
      - OLLAMA_CHAT_URL=${OLLAMA_CHAT_URL:-http://host.docker.internal:11434}
      - EMBEDDING_MODEL=${EMBEDDING_MODEL:-nomic-embed-text}
      - CHAT_MODEL=${CHAT_MODEL:-qwen2.5:32b}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### .env.example

```bash
# Server Configuration
PORT=8080
DATA_DIR=./data

# Ollama Configuration (external - user manages)
OLLAMA_EMBEDDING_URL=http://localhost:11434
OLLAMA_CHAT_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
CHAT_MODEL=qwen2.5:32b

# Processing Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_CONTEXT_CHUNKS=10

# Logging
LOG_LEVEL=INFO
```

### Dockerfile

```dockerfile
# Stage 1: Build Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Backend with Unstructured dependencies
FROM python:3.11-slim AS backend

# Install system dependencies for Unstructured
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libreoffice \
    pandoc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/app ./app

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist ./static

# Create data directory
RUN mkdir -p /data/knowledgevault

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## Backend Dependencies

### backend/requirements.txt

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6
httpx>=0.26.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
lancedb>=0.4.0
pyarrow>=14.0.0
unstructured[all-docs]>=0.12.0
aiofiles>=23.2.0
python-magic>=0.4.27
```

---

## Frontend Dependencies

### frontend/package.json

```json
{
  "name": "knowledgevault-frontend",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.22.0",
    "zustand": "^4.5.0",
    "@tanstack/react-query": "^5.17.0",
    "axios": "^1.6.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.0",
    "class-variance-authority": "^0.7.0",
    "lucide-react": "^0.312.0",
    "framer-motion": "^11.0.0",
    "react-dropzone": "^14.2.0",
    "react-markdown": "^9.0.0",
    "sonner": "^1.4.0",
    "@radix-ui/react-dialog": "^1.0.5",
    "@radix-ui/react-dropdown-menu": "^2.0.6",
    "@radix-ui/react-scroll-area": "^1.0.5",
    "@radix-ui/react-tabs": "^1.0.4",
    "@radix-ui/react-tooltip": "^1.0.7",
    "@radix-ui/react-progress": "^1.0.3"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "autoprefixer": "^10.4.17",
    "postcss": "^8.4.33",
    "tailwindcss": "^3.4.1",
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "tailwindcss-animate": "^1.0.7"
  }
}
```

---

## Key Backend Service Implementations

### Configuration (backend/app/config.py)

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    data_dir: Path = Field(default=Path("/data/knowledgevault"))
    ollama_embedding_url: str = Field(default="http://localhost:11434")
    ollama_chat_url: str = Field(default="http://localhost:11434")
    embedding_model: str = Field(default="nomic-embed-text")
    chat_model: str = Field(default="qwen2.5:32b")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    max_context_chunks: int = Field(default=10)
    auto_scan_enabled: bool = Field(default=False)
    auto_scan_interval_minutes: int = Field(default=30)
    log_level: str = Field(default="INFO")
    
    class Config:
        env_file = ".env"

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"
    
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "documents" / "uploads"
    
    @property
    def library_dir(self) -> Path:
        return self.data_dir / "documents" / "library"
    
    @property
    def lancedb_path(self) -> str:
        return str(self.data_dir / "lancedb")
    
    @property
    def sqlite_path(self) -> str:
        return str(self.data_dir / "knowledgevault.db")

settings = Settings()
```

### SQLite Schema (backend/app/models/database.py)

```python
SCHEMA = """
-- File tracking
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    file_type TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    modified_at TIMESTAMP
);

-- Memories with FTS5
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    category TEXT,
    tags TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, category, content='memories', content_rowid='id'
);

-- Chat history
CREATE TABLE IF NOT EXISTS chat_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
```

### Semantic Chunking Key Logic

```python
class SemanticChunker:
    """
    Structure-aware semantic chunking for technical documents.
    
    Key principles from research:
    - Chunks >200 words show spurious similarity
    - Tables and code blocks should never be split
    - Section headers provide critical context
    """
    
    ATOMIC_ELEMENTS = {Table, CodeSnippet}  # Never split these
    
    def process_file(self, file_path: str) -> List[ProcessedChunk]:
        # Parse with Unstructured (layout-aware)
        elements = partition(filename=file_path, strategy="hi_res")
        
        # Semantic chunking by title/section
        chunked_elements = chunk_by_title(
            elements,
            max_characters=self.chunk_size * 4,
            overlap=self.chunk_overlap * 4
        )
        
        # Post-process: merge small chunks, split oversized
        return self._balance_chunks(chunks)
```

### RAG Engine Core

```python
class RAGEngine:
    async def query(self, user_input: str, chat_history: List[dict], stream: bool):
        # 1. Check for memory storage intent
        memory_content = self.detect_memory_intent(user_input)
        if memory_content:
            memory_store.add_memory(memory_content, source="chat")
            yield {"type": "content", "content": f"I'll remember that: \"{memory_content}\""}
            return
        
        # 2. Generate query embedding
        query_embedding = await embedding_service.embed_single(user_input)
        
        # 3. Search vector store
        search_results = await vector_store.search(query_embedding, limit=10)
        
        # 4. Get relevant memories
        memories = memory_store.search_memories(user_input, limit=5)
        
        # 5. Build prompt with context
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_prompt(user_input, search_results, memories)}
        ]
        
        # 6. Stream response
        async for chunk in llm_client.chat_completion_stream(messages):
            yield {"type": "content", "content": chunk}
        
        yield {"type": "done", "sources": sources, "memories_used": memories}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Basic health check |
| POST | `/api/chat/query` | RAG query (streaming or non-streaming) |
| GET | `/api/documents/list` | List all indexed files |
| GET | `/api/documents/stats` | Get index statistics |
| POST | `/api/documents/upload` | Upload files for indexing |
| POST | `/api/documents/scan` | Scan directory for new files |
| DELETE | `/api/documents/{id}` | Remove file from index |
| GET | `/api/memory/list` | List all memories |
| POST | `/api/memory/add` | Add a memory |
| PUT | `/api/memory/{id}` | Update a memory |
| DELETE | `/api/memory/{id}` | Delete a memory |
| GET | `/api/memory/search` | Search memories |
| GET | `/api/settings/current` | Get current settings |
| GET | `/api/settings/health` | Check LLM service connectivity |

---

## UI Pages

### Chat Page
- Message input with send button
- Streaming response display with markdown rendering
- Source citations panel (collapsible)
- Memory indicator when memories are used
- Chat history sidebar

### Documents Page
- File list with status indicators (indexed/pending/error)
- Drag-and-drop upload zone
- "Scan Directory" button
- File statistics (total files, chunks, size)
- Delete functionality

### Memory Page
- Memory list with category badges
- Add memory form
- Edit/delete actions
- Search functionality

### Settings Page
- Connection status indicators (green/red)
- Ollama endpoint configuration
- Model selection
- Chunking parameters
- Auto-scan toggle and interval
- Document directory path display

---

## Design Principles (Material 3 Expressive)

1. **Generous touch targets** - 48px minimum for interactive elements
2. **Clear visual hierarchy** - Size, color, and containment
3. **Spring-like animations** - Framer Motion with stiffness: 500, damping: 30
4. **Status visibility** - Always show connection status
5. **Progressive disclosure** - Tooltips for descriptions, not cluttered UI
6. **Smart defaults** - Everything works out of the box
7. **One-click actions** - "Scan for new files" not multi-step wizard

---

## Recommended Models

**Embedding:** nomic-embed-text
- 768 dimensions
- 8192 token context
- ~0.5GB VRAM
- Outperforms OpenAI ada-002 on technical content

**Chat:** Qwen 2.5 32B or 72B (Q4_K_M quantization)
- Run on CPU with 380GB RAM
- 32B: ~22GB RAM, ~15 tok/s
- 72B: ~45GB RAM, ~10 tok/s
- Excellent technical reasoning

---

## Deployment Steps

1. Install Docker and Docker Compose on host
2. Clone repository
3. Copy `.env.example` to `.env` and configure
4. Ensure Ollama is running with required models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen2.5:32b
   ```
5. Start the service:
   ```bash
   docker compose up -d
   ```
6. Access at `http://localhost:8080`
7. Drop documents into `/data/knowledgevault/documents/library/` or upload via UI
8. Click "Scan for new files" or wait for auto-scan

---

## Future Enhancements (Out of Scope for MVP)

- User authentication
- Per-user document collections
- Document versioning
- Conversation branching
- Export/import memories
- Webhook notifications on indexing complete
- GPU passthrough for embeddings in container
- Kubernetes deployment manifests
