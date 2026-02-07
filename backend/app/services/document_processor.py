"""
Document processing service with orchestration, status tracking, and deduplication.

Provides DocumentProcessor class that coordinates parsing, chunking, and schema extraction
while tracking processing status in SQLite and handling file deduplication.
"""

import asyncio
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Any, Optional, Union

from unstructured.partition.auto import partition

from ..config import settings
from ..models.database import get_db_connection
from ..utils.file_utils import compute_file_hash
from .chunking import SemanticChunker, ProcessedChunk
from .embeddings import EmbeddingService
from .schema_parser import SchemaParser
from .vector_store import VectorStore


@dataclass
class ProcessedDocument:
    """
    Result of processing a document file.

    Attributes:
        file_id: The database ID of the processed file
        chunks: List of processed chunks from the document
    """
    file_id: int
    chunks: List[ProcessedChunk]


class DuplicateFileError(Exception):
    """Exception raised when a file with the same hash already exists and is indexed."""
    pass


class DocumentProcessingError(Exception):
    """Exception raised when document processing fails due to database errors."""
    pass


class DocumentProcessor:
    """
    Orchestrates document processing with status tracking and deduplication.

    Coordinates DocumentParser, SemanticChunker, and SchemaParser to process
    files while maintaining processing status in SQLite and handling duplicates.
    """

    # File extensions that should use SchemaParser instead of DocumentParser
    SCHEMA_EXTENSIONS = {'.sql', '.ddl'}

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Target chunk size in tokens for semantic chunking
            chunk_overlap: Overlap between chunks in tokens
            vector_store: VectorStore instance for storing chunk embeddings
            embedding_service: EmbeddingService instance for generating embeddings
        """
        self.parser = DocumentParser()
        self.chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.schema_parser = SchemaParser()
        self.sqlite_path = str(settings.sqlite_path)
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def _check_duplicate(self, file_hash: str, conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
        """
        Check if a file with the given hash already exists and is indexed.

        Args:
            file_hash: The hash of the file to check
            conn: Database connection

        Returns:
            The existing file row if found and indexed, None otherwise
        """
        cursor = conn.execute(
            "SELECT * FROM files WHERE file_hash = ? AND status = 'indexed'",
            (file_hash,)
        )
        return cursor.fetchone()

    def _insert_or_get_file_record(
        self,
        file_path: str,
        file_hash: str,
        conn: sqlite3.Connection
    ) -> int:
        """
        Insert a new file record or update existing one, returning the file ID.

        Args:
            file_path: Path to the file
            file_hash: Computed hash of the file
            conn: Database connection

        Returns:
            The file ID (database row ID)

        Raises:
            DocumentProcessingError: If database operations fail
        """
        path = Path(file_path)
        file_name = path.name
        file_size = path.stat().st_size
        file_type = path.suffix.lower() if path.suffix else None
        now = datetime.utcnow().isoformat()
        path_str = str(file_path)

        try:
            # Check if file record already exists by path
            cursor = conn.execute(
                "SELECT id FROM files WHERE file_path = ?",
                (path_str,)
            )
            existing = cursor.fetchone()

            if existing:
                # Validate existing row id
                existing_id = existing['id']
                if existing_id is None:
                    raise DocumentProcessingError(
                        f"Existing file record for '{path_str}' has invalid NULL id"
                    )
                file_id = int(existing_id)

                # Update existing record
                conn.execute(
                    """UPDATE files
                       SET file_hash = ?, file_size = ?, file_type = ?,
                           status = 'pending', error_message = NULL,
                           modified_at = ?, processed_at = NULL
                       WHERE id = ?""",
                    (file_hash, file_size, file_type, now, file_id)
                )
            else:
                # Insert new record
                cursor = conn.execute(
                    """INSERT INTO files
                       (file_path, file_name, file_hash, file_size, file_type,
                        status, created_at, modified_at)
                       VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
                    (path_str, file_name, file_hash, file_size, file_type, now, now)
                )
                lastrowid = cursor.lastrowid
                if lastrowid is None:
                    raise DocumentProcessingError(
                        f"Failed to insert file record for '{path_str}': lastrowid is None"
                    )
                file_id = int(lastrowid)

            # Commit within the context of this method
            conn.commit()
            return file_id

        except sqlite3.Error as e:
            # Rollback on error and wrap in DocumentProcessingError
            conn.rollback()
            raise DocumentProcessingError(
                f"Database error while inserting/updating file record for '{path_str}': {str(e)}"
            ) from e

    def _update_status(
        self,
        file_id: int,
        status: str,
        conn: sqlite3.Connection,
        chunk_count: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update the processing status of a file.

        Args:
            file_id: The database ID of the file
            status: New status ('pending', 'processing', 'indexed', 'error')
            conn: Database connection
            chunk_count: Number of chunks produced (optional)
            error_message: Error message if status is 'error' (optional)

        Note:
            This method does not commit - caller is responsible for transaction management.
        """
        now = datetime.utcnow().isoformat()

        if status == 'indexed':
            conn.execute(
                """UPDATE files
                   SET status = ?, chunk_count = ?, processed_at = ?, modified_at = ?
                   WHERE id = ?""",
                (status, chunk_count, now, now, file_id)
            )
        elif status == 'error':
            conn.execute(
                """UPDATE files
                   SET status = ?, error_message = ?, modified_at = ?
                   WHERE id = ?""",
                (status, error_message, now, file_id)
            )
        else:
            conn.execute(
                """UPDATE files
                   SET status = ?, modified_at = ?
                   WHERE id = ?""",
                (status, now, file_id)
            )
        # Note: No commit here - caller manages transactions

    def _is_schema_file(self, file_path: str) -> bool:
        """
        Check if a file should be processed as a schema file.

        Args:
            file_path: Path to the file

        Returns:
            True if the file has a schema extension (.sql, .ddl)
        """
        return Path(file_path).suffix.lower() in self.SCHEMA_EXTENSIONS

    async def _process_schema_file(self, file_path: str) -> List[ProcessedChunk]:
        """
        Process a schema file using SchemaParser.

        Args:
            file_path: Path to the schema file

        Returns:
            List of ProcessedChunk objects
        """
        schema_chunks = await asyncio.to_thread(self.schema_parser.parse, file_path)

        processed_chunks = []
        for idx, chunk_data in enumerate(schema_chunks):
            chunk = ProcessedChunk(
                text=chunk_data['text'],
                metadata={
                    **chunk_data['metadata'],
                    'chunk_index': idx,
                    'total_chunks': len(schema_chunks)
                },
                chunk_index=idx
            )
            processed_chunks.append(chunk)

        return processed_chunks

    async def _process_document_file(self, file_path: str) -> List[ProcessedChunk]:
        """
        Process a document file using DocumentParser and SemanticChunker.

        Args:
            file_path: Path to the document file

        Returns:
            List of ProcessedChunk objects
        """
        elements = await asyncio.to_thread(self.parser.parse, file_path)
        return await asyncio.to_thread(self.chunker.chunk_elements, elements)

    async def _process_file_async(self, file_path: str) -> ProcessedDocument:
        """
        Internal async implementation of file processing.

        Args:
            file_path: Path to the file to process

        Returns:
            ProcessedDocument containing file_id and chunks
        """
        # Validate file exists
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise FileNotFoundError(f"Path is not a file: {file_path}")

        # Compute file hash
        file_hash = compute_file_hash(file_path)

        # Get database connection
        conn = get_db_connection(self.sqlite_path)

        try:
            # Check for duplicates
            duplicate = self._check_duplicate(file_hash, conn)
            if duplicate:
                raise DuplicateFileError(
                    f"File with hash {file_hash} already indexed as '{duplicate['file_path']}'"
                )

            # Insert or get file record (handles its own commit)
            file_id = self._insert_or_get_file_record(file_path, file_hash, conn)

            try:
                # Update status to processing
                self._update_status(file_id, 'processing', conn)

                # Process the file based on type
                if self._is_schema_file(file_path):
                    chunks = await self._process_schema_file(file_path)
                else:
                    chunks = await self._process_document_file(file_path)

                # Generate embeddings and store in vector store
                if self.embedding_service is not None and self.vector_store is not None:
                    # Skip embedding/indexing if no chunks (status indexed with 0 chunks is acceptable)
                    if chunks:
                        # Extract texts from chunks
                        texts = [c.text for c in chunks]
                        # Generate embeddings
                        embeddings = await self.embedding_service.embed_batch(texts)
                        # Validate embeddings count matches chunks count
                        if len(embeddings) != len(chunks):
                            raise DocumentProcessingError(
                                f"Embedding count mismatch: expected {len(chunks)}, got {len(embeddings)}"
                            )
                        # Validate first embedding is a non-empty list
                        if not embeddings[0] or not isinstance(embeddings[0], list):
                            raise DocumentProcessingError(
                                "First embedding is empty or not a list"
                            )
                        # Map chunks to records for vector store
                        records = []
                        for chunk, embedding in zip(chunks, embeddings):
                            records.append({
                                "id": f"{file_id}_{chunk.chunk_index}",
                                "text": chunk.text,
                                "file_id": str(file_id),
                                "chunk_index": chunk.chunk_index,
                                "metadata": json.dumps(chunk.metadata),
                                "embedding": embedding
                            })
                        # Initialize vector table with embedding dimension and add chunks
                        embedding_dim = len(embeddings[0])
                        await asyncio.to_thread(self.vector_store.init_table, embedding_dim)
                        await asyncio.to_thread(self.vector_store.add_chunks, records)

                # Update status to indexed only after successful vector operations
                self._update_status(file_id, 'indexed', conn, chunk_count=len(chunks))
                conn.commit()

                return ProcessedDocument(file_id=file_id, chunks=chunks)

            except Exception as e:
                # Update status to error and commit
                error_msg = str(e)
                self._update_status(file_id, 'error', conn, error_message=error_msg)
                conn.commit()
                raise

        finally:
            conn.close()

    def process_file(self, file_path: str):
        """
        Process a file with status tracking and deduplication.

        This method can be called from both sync and async contexts.
        When called from a sync context (like unit tests), it will run
        the async operations using asyncio.run().
        When called from an async context, it returns a coroutine that
        should be awaited.

        Args:
            file_path: Path to the file to process

        Returns:
            ProcessedDocument when called from sync context, or a coroutine
            that resolves to ProcessedDocument when called from async context.

        Raises:
            FileNotFoundError: If the file does not exist
            DuplicateFileError: If a file with the same hash is already indexed
            DocumentParseError: If parsing fails
        """
        try:
            # Check if we're in an async context (event loop is running)
            loop = asyncio.get_running_loop()
            # We're in an async context - return the coroutine
            # The caller must await this
            return self._process_file_async(file_path)
        except RuntimeError:
            # No event loop running - we're in a sync context
            # Use asyncio.run() to execute the async code synchronously
            return asyncio.run(self._process_file_async(file_path))


class DocumentParser:
    """
    Parser for extracting text elements from documents using unstructured.io.
    
    Supports various formats: PDF, DOCX, TXT, HTML, and more.
    Uses hi_res strategy for optimal extraction quality.
    """
    
    def parse(self, file_path: str) -> List[Any]:
        """
        Parse a document and extract text elements.
        
        Args:
            file_path: Path to the document file to parse.
            
        Returns:
            List of extracted text elements from the document.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            DocumentParseError: If parsing fails for any reason.
        """
        # Validate file exists
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")
        
        if not path.is_file():
            raise FileNotFoundError(f"Path is not a file: {file_path}")
        
        try:
            # Use unstructured with hi_res strategy for best extraction quality
            elements = partition(
                filename=str(path),
                strategy="hi_res"
            )
            return elements
        except Exception as e:
            # Wrap exceptions with clear, actionable message
            raise DocumentParseError(
                f"Failed to parse document '{file_path}': {str(e)}"
            ) from e


class DocumentParseError(Exception):
    """Exception raised when document parsing fails."""
    pass
