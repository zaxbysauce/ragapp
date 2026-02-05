"""
Documents API routes for file management and processing.

Provides endpoints for listing documents, uploading files, scanning directories,
and managing document processing status.
"""
import asyncio
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import List, Optional

import aiofiles
from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from pydantic import BaseModel, Field

from app.models.database import get_db_connection
from app.config import settings
from app.services.document_processor import DocumentProcessor, DocumentProcessingError, DuplicateFileError
from app.services.vector_store import VectorStore


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


class DocumentResponse(BaseModel):
    """Response model for a document record."""
    id: int
    file_name: str
    file_path: str
    status: str
    chunk_count: int
    created_at: Optional[str]
    processed_at: Optional[str]

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: List[DocumentResponse]


class DocumentStatsResponse(BaseModel):
    """Response model for document statistics."""
    total_files: int
    total_chunks: int
    status: str = "success"


class UploadResponse(BaseModel):
    """Response model for file upload."""
    file_id: int
    file_name: str
    status: str
    message: str


class ScanResponse(BaseModel):
    """Response model for directory scan."""
    files_enqueued: int
    status: str
    message: str


class DeleteResponse(BaseModel):
    """Response model for document deletion."""
    file_id: int
    status: str
    message: str


def _row_to_document_response(row: sqlite3.Row) -> DocumentResponse:
    """Convert a database row to a DocumentResponse."""
    return DocumentResponse(
        id=row["id"],
        file_name=row["file_name"],
        file_path=row["file_path"],
        status=row["status"],
        chunk_count=row["chunk_count"] or 0,
        created_at=row["created_at"],
        processed_at=row["processed_at"],
    )


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
        
        return DocumentListResponse(documents=documents)
    finally:
        conn.close()


@router.get("/stats", response_model=DocumentStatsResponse)
async def get_document_stats():
    """
    Get counts of files and chunks.
    
    Returns total number of files in the database and total chunks
    across all indexed files.
    """
    conn = get_db_connection(str(settings.sqlite_path))
    try:
        # Get total files count
        cursor = conn.execute("SELECT COUNT(*) as total_files FROM files")
        total_files = cursor.fetchone()["total_files"]
        
        # Get total chunks count
        cursor = conn.execute("SELECT COALESCE(SUM(chunk_count), 0) as total_chunks FROM files")
        total_chunks = cursor.fetchone()["total_chunks"]
        
        return DocumentStatsResponse(
            total_files=total_files,
            total_chunks=total_chunks,
        )
    finally:
        conn.close()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a file and process it with strict security controls.
    
    Validates filename, extension, and file size before saving.
    Saves the uploaded file to settings.uploads_dir using aiofiles,
    then processes it via DocumentProcessor.process_file in asyncio.to_thread.
    """
    # Ensure uploads directory exists
    uploads_dir = settings.uploads_dir
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize filename
    file_name = secure_filename(file.filename or "unnamed_file")
    if not file_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
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
        
        # Process file in thread pool to avoid blocking
        processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        try:
            result = await asyncio.to_thread(processor.process_file, str(file_path))
            
            return UploadResponse(
                file_id=result.file_id,
                file_name=file_name,
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
async def scan_directories():
    """
    Trigger a scan of configured directories for new files.
    
    Calls FileWatcher.scan_once() to find and enqueue new files
    from uploads_dir and library_dir that are not in the database.
    """
    from app.services.file_watcher import FileWatcher
    from app.services.background_tasks import BackgroundProcessor
    
    processor = BackgroundProcessor()
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
        )
    except Exception as e:
        logger.exception("Error during directory scan")
        raise HTTPException(status_code=500, detail=f"Scan failed: {e}")
    finally:
        await processor.stop()


@router.delete("/{file_id}", response_model=DeleteResponse)
async def delete_document(file_id: int):
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
