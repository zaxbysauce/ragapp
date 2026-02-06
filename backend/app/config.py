"""
Application configuration using Pydantic Settings.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    
    # Document processing configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_context_chunks: int = 10
    
    # RAG configuration
    rag_relevance_threshold: float = 0.1
    vector_top_k: int = 10
    maintenance_mode: bool = False
    
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

    # CORS settings
    backend_cors_origins: list[str] = ["http://localhost:5173"]

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
