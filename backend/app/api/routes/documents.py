"""
Documents API routes for file management and processing.

Provides endpoints for listing documents, uploading files, scanning directories,
and managing document processing status.
"""
import asyncio
import hashlib
import hmac
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from app.models.database import get_db_connection
from app.config import settings
from app.services.document_processor import DocumentProcessor, DocumentProcessingError, DuplicateFileError
from app.services.vector_store import VectorStore
from app.services.secret_manager import SecretManager
from app.api.deps import get_secret_manager
from app.security import csrf_protect, require_scope
from app.limiter import limiter
from app.services.background_tasks import BackgroundProcessor


def secure_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent security issues.
    
    - Strips paths using os.path.basename
    - Removes non-ASCII characters
    - Replaces spaces with underscores
    - Allows only alphanumeric, dots, hyphens, and underscores
    """
    # Strip paths
    filename = os.path.basename(filename)
    
    # Replace spaces with underscores
    filename = filename.replace(" ", "_")
    
    # Remove non-ASCII characters
    filename = filename.encode("ascii", "ignore").decode("ascii")
    
    # Allow only alphanumeric, dots, hyphens, and underscores
    filename = re.sub(r"[^a-zA-Z0-9._-]", "", filename)
    
    return filename


logger = logging.getLogger(__name__)


router = APIRouter(prefix="/documents", tags=["documents"])


def _sanitize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            continue
        if key.lower() in {"password", "ssn", "secret", "token"}:
            continue
        if isinstance(value, str) and len(value) > 256:
            sanitized[key] = value[:256]
        else:
            sanitized[key] = value
    return sanitized


def _record_document_action(
    file_id: int,
    action: str,
    status: str,
    user_id: str,
    secret_manager: SecretManager,
) -> None:
    key, key_version = secret_manager.get_hmac_key()
    message = f"{file_id}|{action}|{status}|{user_id}"
    digest = hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest()
    conn = get_db_connection(str(settings.sqlite_path))
    try:
        conn.execute(
            """
            INSERT INTO document_actions(file_id, action, status, user_id, hmac_sha256)
            VALUES (?, ?, ?, ?, ?)
            """,
            (file_id, action, status, user_id, digest),
        )
        conn.commit()
    finally:
        conn.close()


@router.post("/admin/retry/{file_id}")
@limiter.limit(settings.admin_rate_limit)
async def retry_document(
    file_id: int,
    request: Request,
    auth: dict = Depends(require_scope("documents:manage")),
    csrf_token: str = Depends(csrf_protect),
    secret_manager: SecretManager = Depends(get_secret_manager),
) -> dict:
    conn = get_db_connection(str(settings.sqlite_path))
    try:
        cursor = conn.execute("SELECT file_path FROM files WHERE id = ?", (file_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        processor = BackgroundProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            vector_store=getattr(request.app.state, "vector_store", None),
            embedding_service=getattr(request.app.state, "embedding_service", None),
        )
        await processor.start()
        try:
            await processor.enqueue(row["file_path"])
        finally:
            await processor.stop()
        _record_document_action(
            file_id,
            "retry",
            "scheduled",
            auth.get("user_id", "unknown"),
            secret_manager,
        )
        return {"file_id": file_id, "status": "scheduled"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error reprocessing document %d", file_id)
        _record_document_action(
            file_id,
            "retry",
            "error",
            auth.get("user_id", "unknown"),
            secret_manager,
        )
        raise HTTPException(status_code=500, detail=f"Retry failed: {exc}")
    finally:
        conn.close()


class DocumentResponse(BaseModel):
    """Response model for a document record - frontend compatible."""
    id: int
    file_name: str
    filename: str  # Frontend alias
    file_path: str
    status: str
    chunk_count: int
    size: Optional[int] = None  # Frontend expects size
    created_at: Optional[str]
    processed_at: Optional[str]
    metadata: Optional[dict] = None  # Frontend expects metadata

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Response model for listing documents - frontend compatible with total."""
    documents: List[DocumentResponse]
    total: int


class DocumentStatsResponse(BaseModel):
    """Response model for document statistics - frontend compatible."""
    total_documents: int  # Frontend expects this field
    total_chunks: int
    total_size_bytes: int = 0  # Frontend expects this field
    documents_by_status: dict = Field(default_factory=dict)  # Frontend expects this field
    total_files: int = 0  # Backward compatibility alias
    status: str = "success"


class UploadResponse(BaseModel):
    """Response model for file upload - frontend compatible."""
    file_id: int
    file_name: str
    id: int  # Frontend alias for file_id
    filename: str  # Frontend alias for file_name
    status: str
    message: str


class ScanResponse(BaseModel):
    """Response model for directory scan - frontend compatible."""
    files_enqueued: int
    status: str
    message: str
    added: int  # Frontend alias for files_enqueued
    scanned: int  # Frontend expects this field (total files scanned)
    errors: List[str] = Field(default_factory=list)  # Frontend expects this field


class DeleteResponse(BaseModel):
    """Response model for document deletion."""
    file_id: int
    status: str
    message: str


def _row_to_document_response(row: sqlite3.Row) -> DocumentResponse:
    """Convert a database row to a DocumentResponse."""
    file_name = row["file_name"]
    chunk_count = row["chunk_count"] or 0
    status = row["status"]
    return DocumentResponse(
        id=row["id"],
        file_name=file_name,
        filename=file_name,  # Frontend alias
        file_path=row["file_path"],
        status=status,
        chunk_count=chunk_count,
        size=row["file_size"] if "file_size" in row.keys() and row["file_size"] is not None else None,
        created_at=row["created_at"],
        processed_at=row["processed_at"],
        metadata={
            "status": status,
            "chunk_count": chunk_count,
            "chunks": chunk_count,  # Backward compatibility
        },
    )


@router.get("", response_model=DocumentListResponse)
@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """
    List all documents from the files table.
    
    Returns a list of all files with their id, file_name, file_path, status,
    chunk_count, created_at, and processed_at fields.
    """
    conn = get_db_connection(str(settings.sqlite_path))
    try:
        cursor = conn.execute(
            """
            SELECT id, file_name, file_path, status, chunk_count, created_at, processed_at
            FROM files
            ORDER BY created_at DESC
            """
        )
        rows = cursor.fetchall()
        
        documents = [_row_to_document_response(row) for row in rows]
        
        return DocumentListResponse(documents=documents, total=len(documents))
    finally:
        conn.close()


@router.get("/stats", response_model=DocumentStatsResponse)
async def get_document_stats():
    """
    Get counts of files and chunks.
    
    Returns total number of files in the database, total chunks,
    total size in bytes, and documents grouped by status.
    """
    conn = get_db_connection(str(settings.sqlite_path))
    try:
        # Get total files count
        cursor = conn.execute("SELECT COUNT(*) as total_files FROM files")
        total_files = cursor.fetchone()["total_files"]
        
        # Get total chunks count
        cursor = conn.execute("SELECT COALESCE(SUM(chunk_count), 0) as total_chunks FROM files")
        total_chunks = cursor.fetchone()["total_chunks"]
        
        # Get total size (sum of file_size if column exists, otherwise 0)
        try:
            cursor = conn.execute("SELECT COALESCE(SUM(file_size), 0) as total_size FROM files")
            total_size_bytes = cursor.fetchone()["total_size"] or 0
        except sqlite3.OperationalError:
            total_size_bytes = 0
        
        # Get documents grouped by status
        cursor = conn.execute("SELECT status, COUNT(*) as count FROM files GROUP BY status")
        documents_by_status = {row["status"]: row["count"] for row in cursor.fetchall()}
        
        return DocumentStatsResponse(
            total_documents=total_files,  # Frontend field
            total_chunks=total_chunks,
            total_size_bytes=total_size_bytes,
            documents_by_status=documents_by_status,
            total_files=total_files,  # Backward compatibility
        )
    finally:
        conn.close()


@router.post("", response_model=UploadResponse)
@router.post("/", response_model=UploadResponse)
async def upload_document_root(request: Request, file: Optional[UploadFile] = None):
    """
    Upload endpoint at root /documents for frontend compatibility.
    Delegates to the main upload handler.
    """
    return await upload_document(request, file)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: Optional[UploadFile] = None):
    """
    Upload a file and process it with strict security controls.
    
    Validates filename, extension, and file size before saving.
    Saves the uploaded file to settings.uploads_dir using aiofiles,
    then processes it via DocumentProcessor.process_file in asyncio.to_thread.
    """
    # Validate file is provided
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate filename is not empty
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")
    
    # Ensure uploads directory exists
    uploads_dir = settings.uploads_dir
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize filename
    file_name = secure_filename(file.filename or "unnamed_file")
    if not file_name:
        file_name = "unnamed_file.txt"

    # Ensure file has an extension for validation
    if not Path(file_name).suffix:
        file_name = f"{file_name}.txt"
    
    # Validate file extension
    file_suffix = Path(file_name).suffix.lower()
    if file_suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File extension '{file_suffix}' not allowed. Allowed: {settings.allowed_extensions}"
        )
    
    # Validate file size from content-length header
    max_size_bytes = settings.max_file_size_mb * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_size_bytes:
                raise HTTPException(status_code=413, detail=f"File too large. Max size: {settings.max_file_size_mb}MB")
        except ValueError:
            pass  # Invalid content-length header, will check during streaming
    
    # Generate safe file path
    file_path = uploads_dir / file_name
    
    # Handle duplicate file names
    counter = 1
    original_path = file_path
    while file_path.exists():
        stem = original_path.stem
        suffix = original_path.suffix
        file_path = uploads_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    
    # Path safety: ensure file_path is within uploads_dir
    try:
        resolved_path = file_path.resolve()
        resolved_uploads_dir = uploads_dir.resolve()
        if not str(resolved_path).startswith(str(resolved_uploads_dir)):
            raise HTTPException(status_code=400, detail="Invalid file path")
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    temp_file_path = None
    try:
        # Save file using aiofiles with chunked reading and size validation
        total_bytes = 0
        temp_file_path = file_path
        async with aiofiles.open(temp_file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # Read 1MB chunks
                total_bytes += len(chunk)
                if total_bytes > max_size_bytes:
                    # Close and delete partial file
                    await f.close()
                    if temp_file_path.exists():
                        temp_file_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail=f"File too large. Max size: {settings.max_file_size_mb}MB")
                await f.write(chunk)
        
        # Process file with app-state dependencies when available
        processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            vector_store=getattr(request.app.state, "vector_store", None),
            embedding_service=getattr(request.app.state, "embedding_service", None),
        )

        try:
            result = await processor.process_file(str(file_path))
            
            return UploadResponse(
                file_id=result.file_id,
                file_name=file_name,
                id=result.file_id,  # Frontend alias
                filename=file_name,  # Frontend alias
                status="indexed",
                message=f"File '{file_name}' uploaded and processed successfully with {len(result.chunks)} chunks",
            )
        except DuplicateFileError as e:
            # File is a duplicate, remove the uploaded file
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=409, detail=f"{e} (uploaded file was cleaned up)")
        except DocumentProcessingError as e:
            logger.exception("Document processing error for file: %s", file_name)
            raise HTTPException(status_code=500, detail=f"Processing error: {e}")
        except Exception as e:
            logger.exception("Unexpected error processing file: %s", file_name)
            raise HTTPException(status_code=500, detail=f"Server error: {e}")
            
        except HTTPException:
            # Clean up partial file if it exists
            if temp_file_path and temp_file_path.exists():
                temp_file_path.unlink(missing_ok=True)
            raise
    except Exception as e:
        logger.exception("Error uploading file: %s", file_name)
        # Clean up file if it was created
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@router.post("/scan", response_model=ScanResponse)
async def scan_directories(request: Request):
    """
    Trigger a scan of configured directories for new files.
    
    Calls FileWatcher.scan_once() to find and enqueue new files
    from uploads_dir and library_dir that are not in the database.
    """
    from app.services.file_watcher import FileWatcher
    from app.services.background_tasks import BackgroundProcessor
    
    processor = BackgroundProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        vector_store=getattr(request.app.state, "vector_store", None),
        embedding_service=getattr(request.app.state, "embedding_service", None),
        maintenance_service=getattr(request.app.state, "maintenance_service", None),
    )
    try:
        await processor.start()
        watcher = FileWatcher(processor)
        
        # Perform scan
        files_enqueued = await watcher.scan_once()
        
        if files_enqueued > 0:
            message = f"Scan complete: {files_enqueued} new files enqueued for processing"
        else:
            message = "Scan complete: no new files found"
        
        return ScanResponse(
            files_enqueued=files_enqueued,
            status="success",
            message=message,
            added=files_enqueued,  # Frontend alias
            scanned=files_enqueued,  # Frontend expects this (at least files_enqueued)
            errors=[],  # Frontend expects this field
        )
    except Exception as e:
        logger.exception("Error during directory scan")
        raise HTTPException(status_code=500, detail=f"Scan failed: {e}")
    finally:
        await processor.stop()


@router.delete("/{file_id}", response_model=DeleteResponse)
@limiter.limit(settings.admin_rate_limit)
async def delete_document(
    file_id: int,
    request: Request,
    auth: dict = Depends(require_scope("documents:manage")),
    csrf_token: str = Depends(csrf_protect),
    secret_manager: SecretManager = Depends(get_secret_manager),
):
    """
    Delete a document by ID.
    
    Deletes the file record from the database and removes all associated
    chunks from the vector store. Returns 404 if the file is not found.
    """
    conn = get_db_connection(str(settings.sqlite_path))
    try:
        # Check if file exists
        cursor = conn.execute("SELECT id, file_name FROM files WHERE id = ?", (file_id,))
        row = cursor.fetchone()
        
        if row is None:
            raise HTTPException(status_code=404, detail=f"Document with id {file_id} not found")
        
        file_name = row["file_name"]
        
        try:
            # Delete from vector store first
            vector_store = VectorStore()
            try:
                vector_store.connect()
                # Check if chunks table exists before attempting deletion
                db = vector_store.db
                if db is not None and "chunks" in db.table_names():
                    vector_store.table = db.open_table("chunks")
                    deleted_chunks = vector_store.delete_by_file(str(file_id))
                    logger.info("Deleted %d chunks from vector store for file_id %s", deleted_chunks, file_id)
                else:
                    logger.debug("Chunks table not found, skipping vector store deletion for file_id %s", file_id)
            except Exception as e:
                logger.warning("Error deleting chunks from vector store: %s", e)
                # Continue with database deletion even if vector store fails
            finally:
                vector_store.close()
            
            # Delete from database
            conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
            _record_document_action(
                file_id,
                "delete",
                "success",
                auth.get("user_id", "unknown"),
                secret_manager,
            )
            conn.commit()
            
            return DeleteResponse(
                file_id=file_id,
                status="success",
                message=f"Document '{file_name}' (id: {file_id}) deleted successfully",
            )
        except Exception:
            conn.rollback()
            raise
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deleting document %d", file_id)
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
    finally:
        conn.close()


# Exception handler for validation errors (e.g., empty filename)
# This is registered at the app level in main.py
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Convert validation errors to 400 for empty filename cases."""
    errors = exc.errors()
    for error in errors:
        if error.get("loc") == ("body", "file") and "filename" in str(error.get("input", "")).lower():
            raise HTTPException(status_code=400, detail="Filename cannot be empty")
    # Re-raise as 400 for any validation error
    raise HTTPException(status_code=400, detail="Invalid request")

