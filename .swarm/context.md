# KnowledgeVault Context

**Project:** KnowledgeVault - Self-Hosted RAG Knowledge Base  
**Swarm:** paid  
**Current Phase:** 7 (Complete)
**Last Updated:** 2026-02-04

---

## Current State

Phase 7 Frontend Scaffold is COMPLETE. React project initialized with Vite and TypeScript. shadcn/ui installed and configured with custom theme system following Material 3 principles. Navigation rail component created for primary navigation. API client integration implemented for backend communication. Tailwind CSS and PostCSS configured with custom color tokens. All UI pages implemented: Chat with streaming display, Search with filters, Documents with upload and management, Memory management, and Settings configuration. React Router integrated for navigation. State management with React hooks and context.

Phase 6 API Layer is complete. FastAPI application with lifespan management and dependency wiring is in place. API routes implemented for health checks, chat (streaming and non-streaming), search, documents, memories, and settings. All routes wire into the underlying RAG engine and services. API documentation (OpenAPI) and integration tests are complete.

**Docker Verification COMPLETE:** `docker compose up --build` succeeded. Health check endpoint `/health` returned status "ok". Backend and frontend containers are running successfully.

**Manual API Checks COMPLETE:**
- `/health` - Status: ok
- `/api/memories` - CRUD operations tested successfully (POST, GET, PUT, DELETE)
- `/api/documents/upload` - File upload tested successfully
- `/api/documents/stats` - Document statistics endpoint tested successfully

**Backend Test Suite COMPLETE:** `pytest backend/tests/` ran successfully with all tests passing (deprecation warnings present but non-blocking).

**Documentation COMPLETE:** Comprehensive README.md created with setup instructions, API documentation, and usage guide. Additional guides include deployment and architecture documentation.

**End-to-End RAG Query (blocked):** Non-stream `/api/chat` and `/api/search` calls fail with 500 because the configured Ollama models (gemini-3-flash-preview) are not pulled locally. The pipeline is ready, but run `ollama pull gemini-3-flash-preview` or configure another accessible model before rerunning the full RAG workflow.

**Note:** Ollama is an external dependency (user-managed). The health check or LLM endpoints may show "LLM unavailable" when Ollama is not running - this is expected behavior, not a blocker.

**Files Present:**
- `.swarm/plan.md` - This implementation plan
- `.swarm/context.md` - This context file
- `KnowledgeVault-Implementation-Plan.md` - Original specification document
- `docker-compose.yml` - Main Docker Compose configuration
- `docker-compose.override.yml` - Local development overrides
- `Dockerfile` - Multi-stage backend build
- `frontend/Dockerfile` - Frontend production build
- `backend/requirements.txt` - Python dependencies
- `backend/__init__.py` - Backend package init
- `backend/app/__init__.py` - App package init
- `backend/app/config.py` - Application configuration (Pydantic Settings)
- `backend/app/models/database.py` - SQLite database models and FTS5 setup
- `backend/app/services/vector_store.py` - LanceDB vector store service
- `backend/app/services/chunking.py` - Document chunking service
- `backend/app/services/schema_parser.py` - Schema parsing service
- `backend/app/services/background_tasks.py` - Background task processing
- `backend/app/services/file_watcher.py` - File system watcher
- `backend/app/services/embeddings.py` - Ollama embedding client
- `backend/app/services/llm_client.py` - OpenAI-compatible LLM chat client
- `backend/app/services/llm_health.py` - LLM health monitoring service
- `backend/app/services/model_checker.py` - Ollama model availability checker
- `backend/app/services/memory_store.py` - SQLite memory store with FTS lookup
- `backend/app/services/document_processor.py` - Document processing service
- `backend/app/services/rag_engine.py` - RAG orchestration engine
- `backend/app/main.py` - FastAPI application with lifespan and dependency injection
- `backend/app/api/routes/health.py` - Health check endpoints
- `backend/app/api/routes/chat.py` - Chat endpoints (streaming and non-streaming)
- `backend/app/api/routes/search.py` - Search endpoints
- `backend/app/api/routes/documents.py` - Document CRUD endpoints
- `backend/app/api/routes/memories.py` - Memory management endpoints
- `backend/app/api/routes/settings.py` - Application settings endpoints
- `backend/app/utils/file_utils.py` - File utility functions
- `backend/tests/test_llm_integration.py` - LLM integration tests
- `backend/tests/test_rag_engine.py` - RAG engine tests
- `backend/tests/test_config.py` - Configuration tests
- `backend/tests/test_document_processor.py` - Document processor tests
- `backend/tests/test_database.py` - Database model tests
- `backend/tests/test_memory_store.py` - Memory store tests
- `backend/tests/test_api.py` - API integration tests
- `frontend/package.json` - Node.js dependencies
- `frontend/components.json` - shadcn/ui configuration
- `frontend/tailwind.config.js` - Tailwind CSS configuration with custom theme
- `frontend/postcss.config.js` - PostCSS configuration
- `frontend/vite.config.ts` - Vite build configuration
- `frontend/tsconfig.json` - TypeScript configuration
- `frontend/tsconfig.node.json` - TypeScript configuration for Node
- `frontend/index.html` - HTML entry point
- `frontend/src/App.tsx` - Main React application component
- `frontend/src/main.tsx` - React application entry point
- `frontend/src/vite-env.d.ts` - Vite environment type declarations
- `frontend/src/components/layout/NavigationRail.tsx` - Primary navigation component
- `frontend/src/components/layout/PageShell.tsx` - Page layout wrapper
- `frontend/src/lib/api.ts` - API client for backend communication
- `frontend/src/components/ui/` - shadcn/ui components (button, card, input, dialog, dropdown-menu, progress, tooltip, scroll-area, tabs, badge, textarea, etc.)
- `frontend/src/lib/utils.ts` - Utility functions (cn helper)
- `frontend/src/index.css` - Global styles with CSS variables
- `.env.example` - Environment variable template

**Directory Structure Created:**
```
/
├── .swarm/
├── backend/
│   ├── __init__.py
│   ├── requirements.txt
│   ├── tests/
│   │   ├── test_config.py
│   │   ├── test_database.py
│   │   ├── test_document_processor.py
│   │   ├── test_llm_integration.py
│   │   ├── test_rag_engine.py
│   │   └── test_memory_store.py
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── models/
│       │   └── database.py
│       ├── api/
│       │   └── routes/
│       │       ├── health.py
│       │       ├── chat.py
│       │       ├── search.py
│       │       ├── documents.py
│       │       ├── memories.py
│       │       └── settings.py
│       ├── services/
│       │   ├── vector_store.py
│       │   ├── chunking.py
│       │   ├── schema_parser.py
│       │   ├── background_tasks.py
│       │   ├── file_watcher.py
│       │   ├── embeddings.py
│       │   ├── llm_client.py
│       │   ├── llm_health.py
│       │   ├── model_checker.py
│       │   ├── memory_store.py
│       │   ├── document_processor.py
│       │   └── rag_engine.py
│       └── utils/
│           └── file_utils.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── NavigationRail.tsx
│   │   │   │   └── PageShell.tsx
│   │   │   └── ui/
│   │   │       ├── button.tsx
│   │   │       ├── card.tsx
│   │   │       ├── input.tsx
│   │   │       ├── dialog.tsx
│   │   │       ├── dropdown-menu.tsx
│   │   │       └── ... (shadcn/ui components)
│   │   ├── lib/
│   │   │   ├── utils.ts
│   │   │   └── api.ts
│   │   └── index.css
│   ├── Dockerfile
│   ├── package.json
│   ├── components.json
│   ├── tailwind.config.js
│   └── postcss.config.js
├── docker-compose.yml
├── docker-compose.override.yml
├── Dockerfile
└── .env.example
```

**Files Added (Phase 7 Complete):**
- `frontend/src/pages/ChatPage.tsx` - Chat interface with streaming display
- `frontend/src/pages/SearchPage.tsx` - Search with filters and results
- `frontend/src/pages/DocumentsPage.tsx` - Document upload and management
- `frontend/src/pages/MemoryPage.tsx` - Memory management interface
- `frontend/src/pages/SettingsPage.tsx` - Application settings
- `frontend/src/hooks/useChat.ts` - Chat state management hook
- `frontend/src/hooks/useDocuments.ts` - Document management hook
- `frontend/src/hooks/useSearch.ts` - Search state management hook
- `frontend/src/hooks/useMemory.ts` - Memory management hook
- `frontend/src/hooks/useSettings.ts` - Settings management hook
- `frontend/src/context/AppContext.tsx` - Global application context
- `frontend/src/components/chat/ChatMessage.tsx` - Individual chat message component
- `frontend/src/components/chat/CitationsPanel.tsx` - RAG source references panel
- `frontend/src/components/chat/StreamingText.tsx` - Streaming text display
- `frontend/src/components/documents/UploadDialog.tsx` - Document upload dialog
- `frontend/src/components/search/SearchFilters.tsx` - Search filter controls

---

## Decisions

| Date | Decision | Rationale | Status |
|------|----------|-----------|--------|
| | | | |

---

## SME Cache

| Domain | Expert | Last Consulted | Key Insights |
|--------|--------|----------------|--------------|
| | | | |

---

## Patterns

| Pattern | Description | Used In |
|---------|-------------|---------|
| | | |

---

## Technical Notes

### Architecture Decisions (from spec)

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

### Data Directory Structure

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

### Recommended Models

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

## Blockers

| Issue | Impact | Resolution |
|-------|--------|------------|
| Required Ollama models not pulled | `/api/chat` and `/api/search` fail because the configured embedding/chat models (gemini-3-flash-preview:latest) are missing | Run `ollama pull gemini-3-flash-preview` (and any other models referenced in `.env`) or update `docker-compose.yml` to use locally-available Ollama models before rerunning the RAG workflow |

---

## Next Actions

Phase 7 Tasks COMPLETED:
- 7.6. Streaming chat display with progress indicators
- 7.7. Citations panel for RAG source references
- 7.8. Documents management page with upload and list
- 7.9. Memory management page
- 7.10. Settings/configuration page
- 7.11. Search page with filters
- 7.12. Routing and state management
- 7.13. Final integration and testing

**Next Actions (Optional Follow-ups):**
1. **End-to-End RAG Test:** Run full RAG workflow test with Ollama configured to verify document ingestion, embedding generation, and chat responses with citations
2. **Performance Testing:** Load test with large document collections to validate system performance on the target hardware (380GB RAM, Dual Xeon)

---

## Environment

- **Host:** Dual Xeon 5218, 380GB RAM, RTX A1000 8GB
- **OS:** Linux (Docker host)
- **External Dependencies:** Ollama (user-managed)
- **Target Port:** 8080

## Agent Activity

| Tool | Calls | Success | Failed | Avg Duration |
|------|-------|---------|--------|--------------|
| read | 45 | 45 | 0 | 3ms |
| edit | 20 | 20 | 0 | 239ms |
| bash | 17 | 17 | 0 | 2450ms |
| apply_patch | 10 | 10 | 0 | 13ms |
| task | 10 | 10 | 0 | 68871ms |
| invalid | 6 | 6 | 0 | 1ms |
| todowrite | 5 | 5 | 0 | 2ms |
| grep | 5 | 5 | 0 | 221ms |
| skill | 1 | 1 | 0 | 14ms |
| write | 1 | 1 | 0 | 3ms |
