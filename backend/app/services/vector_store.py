"""
LanceDB vector store service for semantic search.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import lancedb
import pyarrow as pa
import numpy as np

from app.config import settings


class VectorStoreError(Exception):
    """Custom exception for vector store errors."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """Exception raised when connection to LanceDB fails."""
    pass


class VectorStoreValidationError(VectorStoreError):
    """Exception raised when record validation fails."""
    pass


class VectorStore:
    """LanceDB-based vector store for document chunk embeddings."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the vector store.
        
        Args:
            db_path: Path to LanceDB database. Defaults to settings.lancedb_path.
        """
        self.db_path = db_path or settings.lancedb_path
        self.db: Optional[lancedb.DBConnection] = None
        self.table: Optional[lancedb.table.Table] = None
        self._embedding_dim: Optional[int] = None
    
    def connect(self) -> "VectorStore":
        """Connect to LanceDB.
        
        Raises:
            VectorStoreConnectionError: If connection to LanceDB fails.
        """
        try:
            self.db = lancedb.connect(str(self.db_path))
        except Exception as e:
            raise VectorStoreConnectionError(f"Failed to connect to LanceDB at {self.db_path}: {e}") from e
        return self
    
    def init_table(self, embedding_dim: int) -> "VectorStore":
        """
        Initialize or open the 'chunks' table.
        
        Args:
            embedding_dim: Dimension of embedding vectors.
            
        Returns:
            Self for method chaining.
            
        Raises:
            VectorStoreConnectionError: If connection or table operations fail.
        """
        if self.db is None:
            self.connect()
        
        if self.db is None:
            raise VectorStoreConnectionError("Database connection is not available.")
        
        self._embedding_dim = embedding_dim
        
        # Define schema for chunks table
        schema = pa.schema([
            ("id", pa.string()),
            ("text", pa.string()),
            ("file_id", pa.string()),
            ("chunk_index", pa.int32()),
            ("metadata", pa.string()),  # JSON string for flexibility
            ("embedding", pa.list_(pa.float32(), embedding_dim)),
        ])
        
        # Create or open table with error handling
        try:
            if "chunks" in self.db.table_names():
                self.table = self.db.open_table("chunks")
            else:
                self.table = self.db.create_table("chunks", schema=schema)
        except Exception as e:
            raise VectorStoreConnectionError(f"Failed to initialize 'chunks' table: {e}") from e
        
        return self
    
    def add_chunks(self, records: List[Dict[str, Any]]) -> None:
        """
        Add chunk records to the vector store.
        
        Args:
            records: List of records with keys: id, text, file_id, chunk_index, 
                     metadata, embedding.
                     
        Raises:
            RuntimeError: If table is not initialized.
            VectorStoreValidationError: If records validation fails.
        """
        if self.table is None:
            raise RuntimeError("Table not initialized. Call init_table() first.")
        
        # Handle empty records
        if not records:
            return
        
        # Required fields for validation
        required_fields = ["id", "text", "file_id", "chunk_index", "embedding"]
        
        # Convert records to arrow-compatible format
        processed_records = []
        for record in records:
            # Validate required fields
            missing_fields = [field for field in required_fields if field not in record]
            if missing_fields:
                raise VectorStoreValidationError(
                    f"Record missing required fields: {', '.join(missing_fields)}"
                )
            
            # Ensure embedding is a list (convert from numpy if needed)
            embedding = record["embedding"]
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            elif not isinstance(embedding, list):
                raise VectorStoreValidationError(
                    f"Embedding must be a list or numpy array, got {type(embedding).__name__}"
                )
            
            processed_record = {
                "id": record["id"],
                "text": record["text"],
                "file_id": record["file_id"],
                "chunk_index": record["chunk_index"],
                "metadata": record.get("metadata", "{}"),
                "embedding": embedding,
            }
            processed_records.append(processed_record)
        
        self.table.add(processed_records)
    
    def search(
        self, 
        embedding: List[float], 
        limit: int = 10,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks by embedding.
        
        Args:
            embedding: Query embedding vector.
            limit: Maximum number of results.
            filter_expr: Optional filter expression (LanceDB syntax).
            
        Returns:
            List of matching records with similarity scores. Empty list if no table exists.
        """
        # Ensure DB connection exists
        if self.db is None:
            self.connect()
        
        # Try to open existing table if not already loaded
        if self.table is None:
            try:
                table_names = self.db.table_names()
            except Exception as e:
                raise VectorStoreConnectionError(f"Failed to list table names: {e}") from e
            
            if "chunks" not in table_names:
                # No table exists yet - graceful no-docs behavior
                return []
            
            # Table exists, try to open it
            try:
                self.table = self.db.open_table("chunks")
            except Exception as e:
                raise VectorStoreConnectionError(f"Failed to open 'chunks' table: {e}") from e
            
            # Set embedding_dim from table schema if available
            if self._embedding_dim is None:
                try:
                    schema = self.table.schema
                    embedding_field = schema.field("embedding")
                    # Extract dimension from fixed size list type
                    if hasattr(embedding_field.type, 'list_size'):
                        self._embedding_dim = embedding_field.type.list_size
                except Exception:
                    # If we can't determine embedding_dim, leave it as None
                    pass
        
        query = self.table.search(embedding)
        
        if filter_expr:
            query = query.where(filter_expr)
        
        results = query.limit(limit).to_list()
        return results
    
    def delete_by_file(self, file_id: str) -> int:
        """
        Delete all chunks for a given file_id.
        
        Args:
            file_id: The file ID to delete chunks for.
            
        Returns:
            Number of records deleted.
        """
        # Ensure DB connection exists
        if self.db is None:
            self.connect()
        
        # Try to open existing table if not already loaded
        if self.table is None:
            try:
                table_names = self.db.table_names()
            except Exception as e:
                raise VectorStoreConnectionError(f"Failed to list table names: {e}") from e
            
            if "chunks" not in table_names:
                # No table exists yet - nothing to delete
                return 0
            
            # Table exists, try to open it
            try:
                self.table = self.db.open_table("chunks")
            except Exception as e:
                raise VectorStoreConnectionError(f"Failed to open 'chunks' table: {e}") from e
            
            # Set embedding_dim from table schema if available
            if self._embedding_dim is None:
                try:
                    schema = self.table.schema
                    embedding_field = schema.field("embedding")
                    # Extract dimension from fixed size list type
                    if hasattr(embedding_field.type, 'list_size'):
                        self._embedding_dim = embedding_field.type.list_size
                except Exception:
                    # If we can't determine embedding_dim, leave it as None
                    pass
        
        # Query count before delete to return accurate deletion count
        try:
            count_before = self.table.count_rows(f'file_id = "{file_id}"')
        except Exception:
            # If count_rows fails, safely default to 0
            count_before = 0
        
        # LanceDB delete using filter expression
        self.table.delete(f'file_id = "{file_id}"')
        
        return count_before
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with stats like total chunks, embedding dimension.
        """
        if self.table is None:
            return {"total_chunks": 0, "embedding_dim": self._embedding_dim}
        
        return {
            "total_chunks": self.table.count_rows(),
            "embedding_dim": self._embedding_dim,
        }
    
    def close(self) -> None:
        """Close the database connection."""
        # LanceDB connections are typically stateless
        self.db = None
        self.table = None



