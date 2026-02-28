"""
LanceDB vector store service for semantic search.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import lancedb
import pyarrow as pa
import numpy as np
import logging

from app.config import settings
from app.utils.fusion import rrf_fuse

logger = logging.getLogger(__name__)

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
            ("chunk_scale", pa.string()),  # Scale label like "512", "1024", "default"
            ("sparse_embedding", pa.string()),  # JSON string for sparse vectors (BGE-M3)
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
        
        # Create vector index with configured metric (only for new tables)
        if "chunks" not in self.db.table_names():
            try:
                self.table.create_index(
                    metric=settings.vector_metric,  # "cosine" or "l2"
                    num_partitions=256,
                    num_sub_vectors=96,
                    replace=True,
                )
                logger.info(f"Vector index created with metric={settings.vector_metric}")
            except Exception as e:
                logger.warning(f"Vector index creation failed: {e}")
        
        # Create full-text search index on 'text' column for hybrid search
        try:
            self.table.create_fts_index("text", replace=True)
            logger.info("Full-text search index created on 'text' column")
        except Exception as e:
            logger.warning(f"FTS index creation failed (hybrid search will be unavailable): {e}")
        
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
                "chunk_scale": record.get("chunk_scale", "default"),  # Scale label
                "sparse_embedding": record.get("sparse_embedding"),  # JSON string for sparse vectors
                "metadata": record.get("metadata", "{}"),
                "embedding": embedding,
            }
            
            # Validate sparse_embedding JSON format if provided
            sparse_emb = processed_record.get("sparse_embedding")
            if sparse_emb is not None:
                try:
                    json.loads(sparse_emb)  # Validate it's valid JSON
                except json.JSONDecodeError:
                    raise VectorStoreValidationError("sparse_embedding must be valid JSON")
            processed_records.append(processed_record)
        
        self.table.add(processed_records)
    
    def _search_single_scale(
        self,
        embedding: List[float],
        scale: str,
        fetch_k: int,
        filter_expr: Optional[str] = None,
        vault_id: Optional[str] = None,
        query_text: str = "",
        hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search within a single chunk scale.

        Args:
            embedding: Query embedding vector.
            scale: The chunk_scale value to filter by (e.g., "512", "1024", "default").
            fetch_k: Number of results to fetch per search type.
            filter_expr: Optional additional filter expression.
            vault_id: Optional vault ID to filter results.
            query_text: Raw query text for BM25 FTS search.
            hybrid: If True, combine dense vector search with BM25 FTS using RRF.

        Returns:
            List of matching records for this scale with RRF scores.
        """
        # Build scale filter
        safe_scale = str(scale).replace("'", "\\'")
        scale_filter = f"chunk_scale = '{safe_scale}'"
        
        # Combine with vault filter if present
        if vault_id is not None:
            safe_vault_id = str(vault_id).replace("'", "\\'")
            vault_filter = f"vault_id = '{safe_vault_id}'"
            if filter_expr:
                combined_filter = f"({filter_expr}) AND ({scale_filter}) AND ({vault_filter})"
            else:
                combined_filter = f"{scale_filter} AND ({vault_filter})"
        elif filter_expr:
            combined_filter = f"({filter_expr}) AND ({scale_filter})"
        else:
            combined_filter = scale_filter

        # Dense vector search with scale filter
        query = self.table.search(embedding)
        if combined_filter:
            query = query.where(combined_filter)
        dense_results = query.limit(fetch_k).to_list()

        # If hybrid disabled, return dense results only
        if not hybrid or not query_text:
            return dense_results

        # Run FTS search with scale filter
        try:
            fts_query = self.table.search(query_text)
            fts_filter_parts = [scale_filter]
            if vault_id:
                safe_vault_id = str(vault_id).replace("'", "\\'")
                fts_filter_parts.append(f"vault_id = '{safe_vault_id}'")
            if filter_expr:
                fts_filter_parts.append(f"({filter_expr})")
            fts_combined_filter = " AND ".join(f"({f})" for f in fts_filter_parts)
            fts_query = fts_query.where(fts_combined_filter)
            fts_results = fts_query.limit(fetch_k).to_list()
        except Exception as e:
            logger.warning(f"FTS search failed for scale {scale} (falling back to dense-only): {e}")
            fts_results = []

        # RRF Fusion for this scale
        k_rrf = 60
        rrf_scores: dict = {}
        id_to_record: dict = {}

        for rank, record in enumerate(dense_results):
            uid = record.get("id", f"dense_{rank}")
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + 1.0 / (k_rrf + rank + 1)
            id_to_record[uid] = record

        for rank, record in enumerate(fts_results):
            uid = record.get("id", f"fts_{rank}")
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + 1.0 / (k_rrf + rank + 1)
            if uid not in id_to_record:
                id_to_record[uid] = record

        # Return results with RRF scores (unsorted, to be sorted in cross-scale RRF)
        result_list = []
        for uid in rrf_scores:
            record = dict(id_to_record[uid])
            record["_rrf_score"] = rrf_scores[uid]
            result_list.append(record)
        return result_list

    def search(
        self,
        embedding: List[float],
        limit: int = 10,
        filter_expr: Optional[str] = None,
        vault_id: Optional[str] = None,
        query_text: str = "",
        hybrid: bool = True,
        hybrid_alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks by embedding.

        Args:
            embedding: Query embedding vector.
            limit: Maximum number of results.
            filter_expr: Optional filter expression (LanceDB syntax).
            vault_id: Optional vault ID to filter results. If provided, only returns
                      chunks from the specified vault.
            query_text: Raw query text for BM25 FTS search (used in hybrid search).
            hybrid: If True, combine dense vector search with BM25 FTS using RRF.
            hybrid_alpha: Weight for RRF fusion (not directly used in pure RRF).

        Returns:
            List of matching records with similarity scores. Each record includes:
            - All original fields (id, text, file_id, chunk_index, metadata, etc.)
            - _distance: Cosine distance from query embedding (lower = more similar)
            - _rrf_score: Reciprocal Rank Fusion score (when hybrid=True)
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
        
        # Check for multi-scale search
        fetch_k = limit * 2
        
        if (
            settings.multi_scale_indexing_enabled
            and settings.multi_scale_chunk_sizes
        ):
            # Parse scale sizes
            scale_strs = [
                s.strip() for s in settings.multi_scale_chunk_sizes.split(",")
                if s.strip()
            ]
            
            if len(scale_strs) > 1:
                # Multi-scale search: query each scale and perform cross-scale RRF
                all_scale_results: List[Dict[str, Any]] = []
                
                for scale in scale_strs:
                    scale_results = self._search_single_scale(
                        embedding=embedding,
                        scale=scale,
                        fetch_k=fetch_k,
                        filter_expr=filter_expr,
                        vault_id=vault_id,
                        query_text=query_text,
                        hybrid=hybrid,
                    )
                    all_scale_results.extend(scale_results)
                
                # Cross-scale RRF fusion
                k_rrf = 60
                cross_scale_scores: dict = {}
                id_to_record: dict = {}
                
                # Add results from each scale with their RRF contributions
                for rank, record in enumerate(all_scale_results):
                    uid = record.get("id", f"scale_{rank}")
                    # Use the per-scale RRF score as contribution
                    scale_rrf = record.get("_rrf_score", 0.0)
                    cross_scale_scores[uid] = cross_scale_scores.get(uid, 0.0) + scale_rrf
                    if uid not in id_to_record:
                        id_to_record[uid] = record
                
                # Sort by cross-scale RRF score and return top limit
                sorted_uids = sorted(
                    cross_scale_scores, key=lambda u: cross_scale_scores[u], reverse=True
                )
                fused = []
                for uid in sorted_uids[:limit]:
                    record = dict(id_to_record[uid])
                    record["_rrf_score"] = cross_scale_scores[uid]
                    fused.append(record)
                
                logger.info(
                    f"Multi-scale search: queried {len(scale_strs)} scales, "
                    f"returning {len(fused)} results"
                )
                return fused
        
        # Single-scale or multi-scale disabled: use existing behavior
        
        # Dense vector search
        query = self.table.search(embedding)

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
        
        dense_results = query.limit(fetch_k).to_list()
        
        # If hybrid disabled, return dense results only
        if not hybrid or not query_text:
            return dense_results
        
        # If hybrid enabled, run FTS search
        try:
            fts_query = self.table.search(query_text)  # LanceDB FTS
            if vault_id:
                fts_query = fts_query.where(f"vault_id = '{vault_id}'")
            if filter_expr:
                # FTS doesn't support complex filter_expr, apply basic vault filter only
                fts_query = fts_query.where(filter_expr)
            fts_results = fts_query.limit(fetch_k).to_list()
        except Exception as e:
            logger.warning(f"FTS search failed (falling back to dense-only): {e}")
            fts_results = []

        # RRF Fusion using shared utility
        return rrf_fuse([dense_results, fts_results], k=60, limit=limit)
    
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

    def migrate_add_chunk_scale(self) -> int:
        """
        Migration: Backfill chunk_scale='default' on existing chunks that lack it.

        LanceDB doesn't support ALTER TABLE or UPDATE, so this reads all data,
        adds the chunk_scale field, and rewrites the table. This is idempotent —
        safe to call multiple times (no-op if all records already have chunk_scale).

        Returns:
            Number of records migrated. 0 if no migration was needed.
        """
        if self.db is None:
            self.connect()

        if self.db is None:
            logger.info("LanceDB chunk_scale migration: no connection available")
            return 0

        try:
            table_names = self.db.table_names()
        except Exception as e:
            logger.warning(f"LanceDB chunk_scale migration failed: {e}")
            return 0

        if "chunks" not in table_names:
            logger.info("LanceDB chunk_scale migration: no table exists")
            return 0

        try:
            table = self.db.open_table("chunks")
        except Exception as e:
            logger.warning(f"LanceDB chunk_scale migration failed: {e}")
            return 0

        # Check if chunk_scale column exists in schema
        schema = table.schema
        field_names = [schema.field(i).name for i in range(len(schema))]

        if "chunk_scale" in field_names:
            # Column exists — check if any rows have null chunk_scale
            try:
                df = table.to_pandas()
                null_count = df["chunk_scale"].isna().sum()
                if null_count == 0:
                    logger.info("LanceDB chunk_scale migration: no migration needed")
                    return 0  # All records already have chunk_scale

                # Backfill null chunk_scales with "default"
                df["chunk_scale"] = df["chunk_scale"].fillna("default")
                count = int(null_count)

                # Drop and recreate table with updated data
                self.db.drop_table("chunks")
                try:
                    self.table = self.db.create_table("chunks", data=df)
                except Exception as create_err:
                    logger.critical(f"LanceDB chunk_scale migration: table dropped but recreate failed: {create_err}. Data may need manual recovery from backup.")
                    raise
                logger.info(f"LanceDB chunk_scale migration: backfilled {count} records")
                return count
            except Exception as e:
                logger.warning(f"LanceDB chunk_scale migration failed: {e}")
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
                        logger.critical(f"LanceDB chunk_scale migration: empty table dropped but recreate failed: {create_err}")
                        raise
                    logger.info("LanceDB chunk_scale migration: empty table, recreated with new schema")
                    return 0

                # Add chunk_scale column with default "default"
                df["chunk_scale"] = "default"
                migrated_count = len(df)

                # Drop and recreate table with updated data
                self.db.drop_table("chunks")
                try:
                    self.table = self.db.create_table("chunks", data=df)
                except Exception as create_err:
                    logger.critical(f"LanceDB chunk_scale migration: table dropped but recreate failed: {create_err}. Data may need manual recovery from backup.")
                    raise
                logger.info(f"LanceDB chunk_scale migration: backfilled {migrated_count} records")
                return migrated_count
            except Exception as e:
                logger.warning(f"LanceDB chunk_scale migration failed: {e}")
                return 0

    def migrate_add_sparse_embedding(self) -> int:
        """
        Migration: Add sparse_embedding column to existing chunks table.

        LanceDB doesn't support ALTER TABLE, so this reads all data,
        adds the sparse_embedding field (default null), and rewrites the table.
        This is idempotent — safe to call multiple times.

        Returns:
            Number of records migrated. 0 if no migration was needed.
        """
        if self.db is None:
            self.connect()

        if self.db is None:
            logger.info("LanceDB sparse_embedding migration: no connection available")
            return 0

        try:
            table_names = self.db.table_names()
        except Exception as e:
            logger.warning(f"LanceDB sparse_embedding migration failed: {e}")
            return 0

        if "chunks" not in table_names:
            logger.info("LanceDB sparse_embedding migration: no table exists")
            return 0

        try:
            table = self.db.open_table("chunks")
        except Exception as e:
            logger.warning(f"LanceDB sparse_embedding migration failed: {e}")
            return 0

        # Check if sparse_embedding column exists in schema
        schema = table.schema
        field_names = [schema.field(i).name for i in range(len(schema))]

        if "sparse_embedding" in field_names:
            logger.info("LanceDB sparse_embedding migration: column already exists")
            return 0

        # Column doesn't exist — add it to all records (default None/null)
        try:
            df = table.to_pandas()
            if len(df) == 0:
                # Empty table — just drop and recreate with new schema
                if self._embedding_dim is None:
                    try:
                        embedding_field = table.schema.field("embedding")
                        if hasattr(embedding_field.type, 'list_size'):
                            self._embedding_dim = embedding_field.type.list_size
                    except Exception:
                        pass
                
                # Explicit error if embedding_dim cannot be determined
                if self._embedding_dim is None:
                    raise VectorStoreError("Cannot determine embedding dimension for migration")
                
                self.db.drop_table("chunks")
                try:
                    if self._embedding_dim:
                        self.init_table(self._embedding_dim)
                except Exception as create_err:
                    logger.critical(f"LanceDB sparse_embedding migration: empty table dropped but recreate failed: {create_err}")
                    raise
                logger.info("LanceDB sparse_embedding migration: empty table, recreated with new schema")
                return 0

            # Add sparse_embedding column with default None (null)
            df["sparse_embedding"] = None
            migrated_count = len(df)

            # Drop and recreate table with updated data
            self.db.drop_table("chunks")
            try:
                self.table = self.db.create_table("chunks", data=df)
                # Recreate vector index
                self.table.create_index(
                    metric=settings.vector_metric,
                    num_partitions=256,
                    num_sub_vectors=96,
                    replace=True,
                )
                logger.info(f"Vector index recreated with metric={settings.vector_metric}")
                # Recreate FTS index
                self.table.create_fts_index("text", replace=True)
                logger.info("Full-text search index recreated on 'text' column")
            except Exception as create_err:
                logger.critical(f"LanceDB sparse_embedding migration: table dropped but recreate failed: {create_err}")
                raise
            logger.info(f"LanceDB sparse_embedding migration: added column to {migrated_count} records")
            return migrated_count
        except Exception as e:
            logger.warning(f"LanceDB sparse_embedding migration failed: {e}")
            return 0

    def get_chunks_by_uid(self, chunk_uids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch chunks by their unique IDs.
        
        Args:
            chunk_uids: List of chunk UIDs in format "{file_id}_{chunk_index}"
            
        Returns:
            List of matching chunk records from LanceDB.
        """
        if self.table is None:
            return []
        
        if not chunk_uids:
            return []
        
        try:
            # Build IN clause for chunk_uids
            # Each uid is in format "{file_id}_{chunk_index}"
            # Escape single quotes in uids for SQL-like syntax
            escaped_uids = [uid.replace("'", "''") for uid in chunk_uids]
            quoted_uids = [f"'{uid}'" for uid in escaped_uids]
            uid_list = ", ".join(quoted_uids)
            
            # Query chunks where id is in the list of chunk_uids
            query = f"id IN ({uid_list})"
            results = self.table.search() \
                .where(query) \
                .to_list()
            
            return results
        except Exception as e:
            logger.warning(f"Failed to fetch chunks by UID: {e}")
            return []
    
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
    
    def get_stored_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get stored metadata from the table's metadata.
        
        Returns:
            Dictionary with stored metadata (embedding_model_id, embedding_dim, embedding_prefix_hash)
            or None if table doesn't exist or no metadata is stored.
        """
        if self.table is None:
            return None
        
        try:
            # Try to get table metadata
            table_metadata = self.table.schema.metadata
            if table_metadata:
                # Convert bytes keys/values to strings if needed
                metadata = {}
                for key, value in table_metadata.items():
                    if isinstance(key, bytes):
                        key = key.decode('utf-8')
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    metadata[key] = value
                
                # Extract our stored fields
                result = {}
                if b'embedding_model_id' in table_metadata or 'embedding_model_id' in metadata:
                    result['embedding_model_id'] = metadata.get('embedding_model_id')
                if b'embedding_dim' in table_metadata or 'embedding_dim' in metadata:
                    result['embedding_dim'] = int(metadata.get('embedding_dim', 0))
                if b'embedding_prefix_hash' in table_metadata or 'embedding_prefix_hash' in metadata:
                    result['embedding_prefix_hash'] = metadata.get('embedding_prefix_hash')
                
                if result:
                    return result
        except Exception as e:
            logger.debug(f"Failed to read table metadata: {e}")
        
        return None
    
    def validate_schema(self, embedding_model_id: str, embedding_dim: int) -> Dict[str, Any]:
        """
        Validate that the table schema matches the current embedding configuration.
        
        Args:
            embedding_model_id: The embedding model identifier
            embedding_dim: The expected embedding dimension
            
        Returns:
            Dictionary with validation results
            
        Raises:
            VectorStoreValidationError: If embedding dimension mismatch is detected
        """
        # Generate a probe embedding for "dimension_probe" text
        probe_text = "dimension_probe"
        try:
            probe_embedding = self._generate_probe_embedding(probe_text, embedding_dim)
        except Exception as e:
            logger.warning(f"Failed to generate probe embedding: {e}")
            probe_embedding = None
        
        # Get expected dimension from the provided parameter
        expected_dim = embedding_dim
        
        # Check if table exists
        table_exists = False
        if self.db is not None:
            try:
                table_names = self.db.table_names()
                table_exists = "chunks" in table_names
            except Exception as e:
                logger.warning(f"Failed to check table existence: {e}")
        
        stored_metadata = None
        if table_exists:
            try:
                if self.table is None:
                    self.table = self.db.open_table("chunks")
                
                # Get schema and compare vector dimension
                schema = self.table.schema
                embedding_field = schema.field("embedding")
                actual_dim = None
                if hasattr(embedding_field.type, 'list_size'):
                    actual_dim = embedding_field.type.list_size
                
                if actual_dim is not None and actual_dim != expected_dim:
                    error_msg = f"Embedding dimension changed from {actual_dim} to {expected_dim}; reindex required."
                    logger.error(error_msg)
                    raise VectorStoreValidationError(error_msg)
                
                # Get stored metadata
                stored_metadata = self.get_stored_metadata()
                
            except VectorStoreValidationError:
                raise
            except Exception as e:
                logger.warning(f"Failed to validate schema: {e}")
        
        # Prepare metadata to store
        import hashlib
        prefix_hash = hashlib.sha256(embedding_model_id.encode('utf-8')).hexdigest()[:16]
        
        metadata_to_store = {
            'embedding_model_id': embedding_model_id,
            'embedding_dim': str(expected_dim),
            'embedding_prefix_hash': prefix_hash
        }
        
        # Update table metadata if table exists
        if table_exists and self.table is not None:
            try:
                # Get existing metadata
                current_metadata = dict(self.table.schema.metadata) if self.table.schema.metadata else {}
                
                # Update with our metadata
                for key, value in metadata_to_store.items():
                    if isinstance(value, str):
                        current_metadata[key.encode('utf-8')] = value.encode('utf-8')
                    else:
                        current_metadata[key.encode('utf-8')] = str(value).encode('utf-8')
                
                # Note: LanceDB doesn't support direct metadata update on existing table
                # We'll log the metadata that should be stored for future reference
                logger.info(f"Table metadata to store/update: {metadata_to_store}")
                
            except Exception as e:
                logger.warning(f"Failed to update table metadata: {e}")
        
        return {
            'table_exists': table_exists,
            'expected_dim': expected_dim,
            'actual_dim': expected_dim if table_exists else None,
            'stored_metadata': stored_metadata,
            'probe_embedding_generated': probe_embedding is not None,
            'metadata_to_store': metadata_to_store
        }
    
    def _generate_probe_embedding(self, text: str, dim: int) -> List[float]:
        """
        Generate a probe embedding for dimension validation.
        
        Args:
            text: The text to generate embedding for
            dim: Expected dimension
            
        Returns:
            Generated embedding vector
        """
        # Use a deterministic hash-based approach for probe embedding
        import hashlib
        import random
        
        # Create a deterministic seed from the text
        seed_value = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (2**32)
        random.seed(seed_value)
        
        # Generate a random vector of expected dimension
        # This simulates what a real embedding would look like
        probe = [random.gauss(0, 1) for _ in range(dim)]
        
        # Normalize the vector (typical for embeddings)
        magnitude = sum(x*x for x in probe) ** 0.5
        if magnitude > 0:
            probe = [x / magnitude for x in probe]
        
        return probe



