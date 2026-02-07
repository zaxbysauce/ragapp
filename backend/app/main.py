"""
FastAPI application with lifespan context manager.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import Response

from app.api.routes.admin import router as admin_router
from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.memories import router as memories_router
from app.api.routes.search import router as search_router
from app.api.routes.settings import router as settings_router
from app.config import settings
from app.models.database import init_db
from app.services.llm_client import LLMClient
from app.services.vector_store import VectorStore
from app.services.memory_store import MemoryStore
from app.services.embeddings import EmbeddingService
from app.services.secret_manager import SecretManager
from app.services.toggle_manager import ToggleManager
from app.services.maintenance import MaintenanceService
from app.services.background_tasks import get_background_processor
from app.limiter import limiter
from app.middleware.logging import LoggingMiddleware
from app.middleware.maintenance import MaintenanceMiddleware
from app.security import CSRFManager
from fastapi.exceptions import RequestValidationError
from app.api.routes.documents import validation_exception_handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Initialize database and services
    init_db(str(settings.sqlite_path))
    app.state.llm_client = LLMClient()
    app.state.llm_client.start()
    app.state.embedding_service = EmbeddingService()
    app.state.vector_store = VectorStore()
    app.state.vector_store.connect()
    app.state.memory_store = MemoryStore()
    app.state.secret_manager = SecretManager()
    app.state.toggle_manager = ToggleManager(str(settings.sqlite_path))
    app.state.csrf_manager = CSRFManager(settings.redis_url, settings.csrf_token_ttl)
    app.state.maintenance_service = MaintenanceService(str(settings.sqlite_path))
    app.state.model_validation = (
        settings.enable_model_validation
        or app.state.toggle_manager.get_toggle("model_validation", settings.enable_model_validation)
    )
    # Initialize background processor as singleton (runs continuously)
    app.state.background_processor = get_background_processor(
        max_retries=3,
        retry_delay=1.0,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        vector_store=app.state.vector_store,
        embedding_service=app.state.embedding_service,
        maintenance_service=app.state.maintenance_service,
    )
    await app.state.background_processor.start()
    yield
    # Shutdown: Close services
    await app.state.background_processor.stop()
    await app.state.llm_client.close()
    app.state.vector_store.close()


app = FastAPI(
    title="KnowledgeVault API",
    version="0.1.0",
    description="Self-hosted RAG Knowledge Base API",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.backend_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up rate limiting
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
# Note: MaintenanceMiddleware is initialized with a lazy getter since
# maintenance_service is only available after lifespan startup
app.state._maintenance_service_getter = lambda: getattr(app.state, 'maintenance_service', None)
app.add_middleware(MaintenanceMiddleware, service_getter=app.state._maintenance_service_getter)

app.include_router(health_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(memories_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(settings_router, prefix="/api")
app.include_router(admin_router, prefix="/api")

# Register exception handler for validation errors (empty filename)
app.add_exception_handler(RequestValidationError, validation_exception_handler)


@app.get("/health")
async def health_check():
    """Simple health check endpoint for Docker/tooling."""
    return {"status": "ok"}


# Serve frontend static files
from pathlib import Path
static_dir = Path("/app/static")
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
