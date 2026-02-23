"""
Application configuration using Pydantic Settings.
"""
import logging
from pathlib import Path
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Server configuration
    port: int = 8080
    
    # Base data directory
    data_dir: Path = Path("/data/knowledgevault")
    
    # Ollama configuration
    ollama_embedding_url: str = "http://host.docker.internal:11434"
    ollama_chat_url: str = "http://host.docker.internal:11434"
    
    # Model configuration
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "qwen2.5:32b"
    
    # Embedding dimension (auto-detected from model, but can be overridden)
    embedding_dim: int = 768
    
    # Document processing configuration (character-based - NEW)
    chunk_size_chars: int | None = None
    """Character-based chunk size for document processing. Default ~1500 chars (~375 tokens)."""
    chunk_overlap_chars: int | None = None
    """Character-based overlap between chunks. Default ~150 chars (~37 tokens)."""
    retrieval_top_k: int | None = None
    """Number of top chunks to retrieve (unifies max_context_chunks and vector_top_k)."""
    vector_metric: str = "cosine"
    """Distance metric for vector similarity search."""
    max_distance_threshold: float | None = None
    """Maximum distance threshold for relevance filtering (replaces rag_relevance_threshold)."""
    embedding_doc_prefix: str = ""
    """Prefix to prepend to documents during embedding."""
    embedding_query_prefix: str = ""
    """Prefix to prepend to queries during embedding."""
    retrieval_window: int = 1
    """Window size for retrieval context expansion."""

    # Document processing configuration (legacy - DEPRECATED)
    chunk_size: int | None = None
    """[DEPRECATED] Token-based chunk size. Use chunk_size_chars instead."""
    chunk_overlap: int | None = None
    """[DEPRECATED] Token-based chunk overlap. Use chunk_overlap_chars instead."""
    max_context_chunks: int = 10
    """[DEPRECATED] Number of context chunks. Use retrieval_top_k instead."""

    # RAG configuration (legacy - DEPRECATED)
    rag_relevance_threshold: float | None = None
    """[DEPRECATED] Relevance threshold. Use max_distance_threshold instead."""
    vector_top_k: int | None = None
    """[DEPRECATED] Vector top K. Use retrieval_top_k instead."""
    maintenance_mode: bool = False
    redis_url: str = "redis://localhost:6379/0"
    csrf_token_ttl: int = 900
    admin_rate_limit: str = "10/minute"
    health_check_api_key: str = "health-api-key"
    
    # Auto-scan configuration
    auto_scan_enabled: bool = True
    auto_scan_interval_minutes: int = 60
    
    # Logging configuration
    log_level: str = "INFO"

    # Feature flags
    enable_model_validation: bool = False

    # Admin security
    admin_secret_token: str = "admin-secret-token"
    audit_hmac_key_version: str = "v1"

    # Security settings
    max_file_size_mb: int = 50
    allowed_extensions: set[str] = {
        ".txt", ".md", ".pdf", ".docx", ".csv", ".json",
        ".sql", ".py", ".js", ".ts", ".html", ".css",
        ".xml", ".yaml", ".yml"
    }

    # IMAP Email Ingestion configuration
    imap_enabled: bool = False
    imap_host: str = ""
    imap_port: int = 993
    imap_username: str = ""
    imap_password: SecretStr = SecretStr("")
    imap_use_ssl: bool = True
    imap_mailbox: str = "INBOX"
    imap_poll_interval: int = 60  # seconds
    imap_max_attachment_size: int = 10 * 1024 * 1024  # 10MB
    imap_allowed_mime_types: set[str] = {
        "application/pdf",
        "text/plain",
        "text/markdown",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/csv",
        "application/json",
        "application/sql",
        "text/x-python",
        "application/javascript",
        "text/html",
        "text/css",
        "application/xml",
        "application/x-yaml",
    }

    # CORS settings
    backend_cors_origins: list[str] = ["http://localhost:5173"]

    # Migration validators for backward compatibility
    @field_validator("chunk_size_chars", mode="before")
    @classmethod
    def migrate_chunk_size_chars(cls, v: int | None, values) -> int:
        """Auto-convert from legacy chunk_size if chunk_size_chars not provided."""
        if v is not None:
            return v
        legacy_chunk_size = values.data.get("chunk_size")
        if legacy_chunk_size is not None:
            logger.warning(
                "Deprecated: 'chunk_size' is deprecated. Use 'chunk_size_chars' instead. "
                f"Auto-converting chunk_size={legacy_chunk_size} to chunk_size_chars={legacy_chunk_size * 4}."
            )
            return legacy_chunk_size * 4
        return 1500  # ~375 tokens, leaves room for instruction prefix

    @field_validator("chunk_overlap_chars", mode="before")
    @classmethod
    def migrate_chunk_overlap_chars(cls, v: int | None, values) -> int:
        """Auto-convert from legacy chunk_overlap if chunk_overlap_chars not provided."""
        if v is not None:
            return v
        legacy_chunk_overlap = values.data.get("chunk_overlap")
        if legacy_chunk_overlap is not None:
            logger.warning(
                "Deprecated: 'chunk_overlap' is deprecated. Use 'chunk_overlap_chars' instead. "
                f"Auto-converting chunk_overlap={legacy_chunk_overlap} to chunk_overlap_chars={legacy_chunk_overlap * 4}."
            )
            return legacy_chunk_overlap * 4
        return 200

    @field_validator("retrieval_top_k", mode="before")
    @classmethod
    def migrate_retrieval_top_k(cls, v: int | None, values) -> int:
        """Auto-convert from legacy vector_top_k if retrieval_top_k not provided."""
        if v is not None:
            return v
        legacy_vector_top_k = values.data.get("vector_top_k")
        if legacy_vector_top_k is not None:
            logger.warning(
                "Deprecated: 'vector_top_k' is deprecated. Use 'retrieval_top_k' instead. "
                f"Auto-copying vector_top_k={legacy_vector_top_k} to retrieval_top_k={legacy_vector_top_k}."
            )
            return legacy_vector_top_k
        return 12

    @field_validator("max_distance_threshold", mode="before")
    @classmethod
    def migrate_max_distance_threshold(cls, v: float | None, values) -> float | None:
        """Auto-convert from legacy rag_relevance_threshold if max_distance_threshold not provided."""
        if v is not None:
            return v
        legacy_rag_relevance_threshold = values.data.get("rag_relevance_threshold")
        if legacy_rag_relevance_threshold is not None:
            logger.warning(
                "Deprecated: 'rag_relevance_threshold' is deprecated. Use 'max_distance_threshold' instead. "
                f"Auto-converting rag_relevance_threshold={legacy_rag_relevance_threshold} "
                f"to max_distance_threshold={1.0 - legacy_rag_relevance_threshold}."
            )
            return 1.0 - legacy_rag_relevance_threshold
        return None

    @property
    def documents_dir(self) -> Path:
        """Directory for storing documents."""
        return self.data_dir / "documents"
    
    @property
    def uploads_dir(self) -> Path:
        """Directory for temporary uploads."""
        return self.data_dir / "uploads"
    
    @property
    def library_dir(self) -> Path:
        """Directory for library files."""
        return self.data_dir / "library"
    
    @property
    def lancedb_path(self) -> Path:
        """Path to LanceDB database."""
        return self.data_dir / "lancedb"
    
    @property
    def sqlite_path(self) -> Path:
        """Path to SQLite database."""
        return self.data_dir / "app.db"


# Global settings instance
settings = Settings()
