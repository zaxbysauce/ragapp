"""
LanceDB vector store service for semantic search.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import lancedb
import pyarrow as pa
import numpy as np
import logging

from app.config import settings

logger = logging.getLogger(__name__)


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
            ("vault_id", pa.string()),  # Vault isolation
            ("chunk_index", pa.int32()),
            ("metadata", pa.string()),  # JSON string for flexibility
            ("embedding", pa.list_(pa.float32(), embedding_dim)),
        ])
        
        # Create or open table with error handling
        try:
            if "chunks" in self.db.table_names():
                try:
                    self.table = self.db.open_table("chunks")
                except Exception:
                    # Stale table reference — drop and recreate
                    try:
                        self.db.drop_table("chunks")
                    except Exception:
                        pass
                    self.table = self.db.create_table("chunks", schema=schema, mode="overwrite")
            else:
                self.table = self.db.create_table("chunks", schema=schema)
        except Exception as e:
            raise VectorStoreConnectionError(f"Failed to initialize 'chunks' table: {e}") from e
        
        return self
    
    def _get_expected_embedding_dim(self) -> Optional[int]:
        """Get the expected embedding dimension from the table schema."""
        if self.table is None:
            return self._embedding_dim
        
        try:
            schema = self.table.schema
            embedding_field = schema.field("embedding")
            if hasattr(embedding_field.type, 'list_size'):
                return embedding_field.type.list_size
        except Exception:
            pass
        return self._embedding_dim
    
    def add_chunks(self, records: List[Dict[str, Any]]) -> None:
        """
        Add chunk records to the vector store.
        
        Args:
            records: List of records with keys: id, text, file_id, chunk_index, 
                     metadata, embedding, vault_id (optional, defaults to "1").
                     
        Raises:
            RuntimeError: If table is not initialized.
            VectorStoreValidationError: If records validation fails.
        """
        if self.table is None:
            raise RuntimeError("Table not initialized. Call init_table() first.")
        
        # Handle empty records
        if not records:
            return
        
        # Get expected embedding dimension from table schema
        expected_dim = self._get_expected_embedding_dim()
        
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
            
            # Validate embedding dimension matches table schema
            actual_dim = len(embedding)
            if expected_dim is not None and actual_dim != expected_dim:
                raise VectorStoreValidationError(
                    f"Embedding dimension mismatch: expected {expected_dim} dimensions, "
                    f"got {actual_dim}. The table was created with a different embedding model. "
                    f"Delete the lancedb directory at {self.db_path} and restart to use the new model."
                )
            
            processed_record = {
                "id": record["id"],
                "text": record["text"],
                "file_id": record["file_id"],
                "vault_id": record.get("vault_id", "1"),  # Default to vault "1"
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
        filter_expr: Optional[str] = None,
        vault_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks by embedding.

        Args:
            embedding: Query embedding vector.
            limit: Maximum number of results.
            filter_expr: Optional filter expression (LanceDB syntax).
            vault_id: Optional vault ID to filter results. If provided, only returns
                     chunks from the specified vault.

        Returns:
            List of matching records with similarity scores. Each record includes:
            - All original fields (id, text, file_id, chunk_index, metadata, etc.)
            - _distance: Cosine distance from query embedding (lower = more similar)
            Empty list if no table exists.
            
        Note:
            For cosine distance metric:
            - Distance of 0 = identical vectors (perfect match)
            - Distance of 1 = orthogonal vectors
            - Distance of 2 = opposite vectors (perfect mismatch)
            The _distance field is provided by LanceDB's vector search.
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
        
        query = self.table.search(embedding, metric="cosine")

        # Apply vault filter if specified
        if vault_id is not None:
            safe_vault_id = str(vault_id).replace("'", "\\'")
            vault_filter = f"vault_id = '{safe_vault_id}'"
            if filter_expr:
                filter_expr = f"({filter_expr}) AND ({vault_filter})"
            else:
                filter_expr = vault_filter

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
        safe_file_id = str(file_id).replace('"', '\\"')
        try:
            count_before = self.table.count_rows(f'file_id = "{safe_file_id}"')
        except Exception:
            # If count_rows fails, safely default to 0
            count_before = 0

        # LanceDB delete using filter expression
        self.table.delete(f'file_id = "{safe_file_id}"')

        return count_before

    def delete_by_vault(self, vault_id: str) -> int:
        """
        Delete all chunks for a given vault_id.

        Args:
            vault_id: The vault ID to delete all chunks for.

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
                return 0

            try:
                self.table = self.db.open_table("chunks")
            except Exception as e:
                raise VectorStoreConnectionError(f"Failed to open 'chunks' table: {e}") from e

        safe_vault_id = str(vault_id).replace("'", "\\'")
        try:
            count_before = self.table.count_rows(f"vault_id = '{safe_vault_id}'")
        except Exception:
            count_before = 0

        self.table.delete(f"vault_id = '{safe_vault_id}'")
        return count_before

    def migrate_add_vault_id(self) -> int:
        """
        Migration: Backfill vault_id='1' on existing chunks that lack it.

        LanceDB doesn't support ALTER TABLE or UPDATE, so this reads all data,
        adds the vault_id field, and rewrites the table. This is idempotent —
        safe to call multiple times (no-op if all records already have vault_id).

        Returns:
            Number of records migrated. 0 if no migration was needed.
        """
        if self.db is None:
            self.connect()

        if self.db is None:
            logger.info("LanceDB vault_id migration: no connection available")
            return 0

        try:
            table_names = self.db.table_names()
        except Exception as e:
            logger.warning(f"LanceDB vault_id migration failed: {e}")
            return 0

        if "chunks" not in table_names:
            logger.info("LanceDB vault_id migration: no table exists")
            return 0

        try:
            table = self.db.open_table("chunks")
        except Exception as e:
            logger.warning(f"LanceDB vault_id migration failed: {e}")
            return 0

        # Check if vault_id column exists in schema
        schema = table.schema
        field_names = [schema.field(i).name for i in range(len(schema))]

        if "vault_id" in field_names:
            # Column exists — check if any rows have null vault_id
            try:
                df = table.to_pandas()
                null_count = df["vault_id"].isna().sum()
                if null_count == 0:
                    logger.info("LanceDB vault_id migration: no migration needed")
                    return 0  # All records already have vault_id

                # Backfill null vault_ids with "1"
                df["vault_id"] = df["vault_id"].fillna("1")
                count = int(null_count)

                # Drop and recreate table with updated data
                self.db.drop_table("chunks")
                try:
                    self.table = self.db.create_table("chunks", data=df)
                except Exception as create_err:
                    logger.critical(f"LanceDB vault_id migration: table dropped but recreate failed: {create_err}. Data may need manual recovery from backup.")
                    raise
                logger.info(f"LanceDB vault_id migration: backfilled {count} records")
                return count
            except Exception as e:
                logger.warning(f"LanceDB vault_id migration failed: {e}")
                return 0
        else:
            # Column doesn't exist — add it to all records
            try:
                df = table.to_pandas()
                if len(df) == 0:
                    # Empty table — just drop and recreate with new schema
                    # Try to get embedding_dim from existing schema before dropping
                    if self._embedding_dim is None:
                        try:
                            embedding_field = table.schema.field("embedding")
                            if hasattr(embedding_field.type, 'list_size'):
                                self._embedding_dim = embedding_field.type.list_size
                        except Exception:
                            pass

                    self.db.drop_table("chunks")
                    try:
                        if self._embedding_dim:
                            self.init_table(self._embedding_dim)
                    except Exception as create_err:
                        logger.critical(f"LanceDB vault_id migration: empty table dropped but recreate failed: {create_err}")
                        raise
                    logger.info("LanceDB vault_id migration: empty table, recreated with new schema")
                    return 0

                # Add vault_id column with default "1"
                df["vault_id"] = "1"
                migrated_count = len(df)

                # Drop and recreate table with updated data
                self.db.drop_table("chunks")
                try:
                    self.table = self.db.create_table("chunks", data=df)
                except Exception as create_err:
                    logger.critical(f"LanceDB vault_id migration: table dropped but recreate failed: {create_err}. Data may need manual recovery from backup.")
                    raise
                logger.info(f"LanceDB vault_id migration: backfilled {migrated_count} records")
                return migrated_count
            except Exception as e:
                logger.warning(f"LanceDB vault_id migration failed: {e}")
                return 0

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



