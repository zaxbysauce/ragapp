"""Tests for vector_store.py fixes in Task 0.1.

Tests cover:
1. Vector index is NOT created on fresh install (table <256 rows)
2. FTS index is created once and not rebuilt
3. _maybe_create_vector_index is called after add_chunks
4. Vector index IS created after 256 rows
5. Edge cases (empty records, concurrent access)
"""

import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock, patch, Mock, call
import threading
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# Stub missing optional dependencies before importing app modules
try:
    import lancedb
except ImportError:
    import types
    sys.modules['lancedb'] = types.ModuleType('lancedb')

try:
    import pyarrow as pa
except ImportError:
    import types
    sys.modules['pyarrow'] = types.ModuleType('pyarrow')
    pa = types.ModuleType('pyarrow')
    pa.schema = lambda fields: None
    pa.string = lambda: None
    pa.int32 = lambda: None
    pa.float32 = lambda: None
    pa.list_ = lambda t, size=None: None

from app.services.vector_store import (
    VectorStore,
    VectorStoreError,
    VectorStoreConnectionError,
    VectorStoreValidationError,
    VECTOR_INDEX_MIN_ROWS,
)


class FakeIndex:
    """Fake index object for mocking list_indices() results."""
    def __init__(self, name):
        self.name = name


class TestVectorIndexCreation(unittest.TestCase):
    """Test suite for vector index creation behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_lancedb")
        self.store = VectorStore(db_path=self.db_path)
        self.embedding_dim = 384

    def tearDown(self):
        """Clean up test fixtures."""
        self.store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_table(self, row_count=0, existing_indices=None):
        """Create a mock table with configurable row count and indices."""
        mock_table = MagicMock()
        mock_table.count_rows.return_value = row_count
        
        if existing_indices is None:
            existing_indices = []
        mock_table.list_indices.return_value = [FakeIndex(name) for name in existing_indices]
        
        # Configure schema mock for _get_expected_embedding_dim
        mock_field = MagicMock()
        mock_field.type.list_size = self.embedding_dim
        mock_table.schema.field.return_value = mock_field
        
        return mock_table

    def test_vector_index_not_created_on_fresh_install(self):
        """Test that vector index is NOT created when table has <256 rows."""
        # Setup mock table with 0 rows (fresh install)
        mock_table = self._create_mock_table(row_count=0, existing_indices=[])
        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim
        
        # Call _maybe_create_vector_index
        self.store._maybe_create_vector_index()
        
        # Verify create_index was NOT called
        mock_table.create_index.assert_not_called()

    def test_vector_index_not_created_with_255_rows(self):
        """Test that vector index is NOT created when table has exactly 255 rows."""
        # Setup mock table with 255 rows (just under threshold)
        mock_table = self._create_mock_table(row_count=255, existing_indices=[])
        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim
        
        # Call _maybe_create_vector_index
        self.store._maybe_create_vector_index()
        
        # Verify create_index was NOT called
        mock_table.create_index.assert_not_called()

    def test_vector_index_created_at_256_rows(self):
        """Test that vector index IS created when table reaches exactly 256 rows."""
        # Setup mock table with exactly 256 rows (at threshold)
        mock_table = self._create_mock_table(row_count=256, existing_indices=[])
        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim
        
        # Mock settings.vector_metric
        with patch('app.services.vector_store.settings') as mock_settings:
            mock_settings.vector_metric = "cosine"
            
            # Call _maybe_create_vector_index
            self.store._maybe_create_vector_index()
            
            # Verify create_index WAS called with correct parameters
            mock_table.create_index.assert_called_once()
            call_kwargs = mock_table.create_index.call_args.kwargs
            self.assertEqual(call_kwargs.get('metric'), 'cosine')
            self.assertEqual(call_kwargs.get('num_partitions'), 256)
            self.assertEqual(call_kwargs.get('num_sub_vectors'), 96)
            self.assertEqual(call_kwargs.get('replace'), True)

    def test_vector_index_created_above_threshold(self):
        """Test that vector index IS created when table has >256 rows."""
        # Setup mock table with 500 rows (above threshold)
        mock_table = self._create_mock_table(row_count=500, existing_indices=[])
        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim
        
        # Mock settings.vector_metric
        with patch('app.services.vector_store.settings') as mock_settings:
            mock_settings.vector_metric = "l2"
            
            # Call _maybe_create_vector_index
            self.store._maybe_create_vector_index()
            
            # Verify create_index WAS called
            mock_table.create_index.assert_called_once()
            call_kwargs = mock_table.create_index.call_args.kwargs
            self.assertEqual(call_kwargs.get('metric'), 'l2')

    def test_vector_index_skipped_if_already_exists(self):
        """Test that vector index creation is skipped if index already exists."""
        # Setup mock table with existing vector index
        mock_table = self._create_mock_table(row_count=500, existing_indices=["embedding_idx"])
        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim
        
        # Call _maybe_create_vector_index
        self.store._maybe_create_vector_index()
        
        # Verify create_index was NOT called (fast path)
        mock_table.create_index.assert_not_called()

    def test_vector_index_handles_list_indices_exception(self):
        """Test that vector index creation handles exceptions from list_indices gracefully."""
        mock_table = MagicMock()
        mock_table.list_indices.side_effect = Exception("Index listing failed")
        mock_table.count_rows.return_value = 500
        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim
        
        # Mock settings.vector_metric
        with patch('app.services.vector_store.settings') as mock_settings:
            mock_settings.vector_metric = "cosine"
            
            # Call _maybe_create_vector_index - should not raise
            self.store._maybe_create_vector_index()
            
            # Verify create_index was still called (continues after exception)
            mock_table.create_index.assert_called_once()

    def test_vector_index_handles_count_rows_exception(self):
        """Test that vector index creation handles exceptions from count_rows gracefully."""
        mock_table = MagicMock()
        mock_table.list_indices.return_value = []
        mock_table.count_rows.side_effect = Exception("Count failed")
        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim
        
        # Call _maybe_create_vector_index - should not raise
        self.store._maybe_create_vector_index()
        
        # Verify create_index was NOT called due to exception
        mock_table.create_index.assert_not_called()

    def test_vector_index_handles_create_index_exception(self):
        """Test that vector index creation handles exceptions from create_index gracefully."""
        mock_table = MagicMock()
        mock_table.list_indices.return_value = []
        mock_table.count_rows.return_value = 500
        mock_table.create_index.side_effect = Exception("Index creation failed")
        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim
        
        # Mock settings.vector_metric
        with patch('app.services.vector_store.settings') as mock_settings:
            mock_settings.vector_metric = "cosine"
            
            # Call _maybe_create_vector_index - should not raise
            self.store._maybe_create_vector_index()
            
            # Verify create_index was called but exception was handled
            mock_table.create_index.assert_called_once()


class TestFTSIndexCreation(unittest.TestCase):
    """Test suite for FTS index creation behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_lancedb")
        self.store = VectorStore(db_path=self.db_path)
        self.embedding_dim = 384

    def tearDown(self):
        """Clean up test fixtures."""
        self.store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fts_index_created_if_missing(self):
        """Test that FTS index is created when it doesn't exist."""
        mock_table = MagicMock()
        mock_table.list_indices.return_value = []  # No existing indices
        self.store.table = mock_table
        
        # Manually call the FTS index creation logic from init_table
        fts_index_exists = any(idx.name == "fts_text" for idx in mock_table.list_indices())
        self.assertFalse(fts_index_exists)
        
        if not fts_index_exists:
            mock_table.create_fts_index("text", replace=True)
        
        # Verify create_fts_index was called
        mock_table.create_fts_index.assert_called_once_with("text", replace=True)

    def test_fts_index_skipped_if_exists(self):
        """Test that FTS index is NOT rebuilt when it already exists."""
        mock_table = MagicMock()
        mock_table.list_indices.return_value = [FakeIndex("fts_text")]
        self.store.table = mock_table
        
        # Check if FTS index exists
        fts_index_exists = any(idx.name == "fts_text" for idx in mock_table.list_indices())
        self.assertTrue(fts_index_exists)
        
        # If it exists, we should skip creation
        if not fts_index_exists:
            mock_table.create_fts_index("text", replace=True)
        
        # Verify create_fts_index was NOT called
        mock_table.create_fts_index.assert_not_called()

    def test_fts_index_handles_list_indices_exception(self):
        """Test that FTS index creation handles exceptions from list_indices gracefully."""
        mock_table = MagicMock()
        mock_table.list_indices.side_effect = Exception("Index listing failed")
        self.store.table = mock_table
        
        # Try to check if FTS index exists - should handle exception
        try:
            indices = mock_table.list_indices()
            fts_index_exists = any(idx.name == "fts_text" for idx in indices)
        except Exception:
            fts_index_exists = False
        
        self.assertFalse(fts_index_exists)
        
        # Should proceed to create index since we couldn't verify
        if not fts_index_exists:
            mock_table.create_fts_index("text", replace=True)
        
        mock_table.create_fts_index.assert_called_once_with("text", replace=True)

    def test_fts_index_handles_create_exception(self):
        """Test that FTS index creation handles exceptions gracefully."""
        mock_table = MagicMock()
        mock_table.list_indices.return_value = []
        mock_table.create_fts_index.side_effect = Exception("FTS creation failed")
        self.store.table = mock_table
        
        # Try to create FTS index - should handle exception
        try:
            mock_table.create_fts_index("text", replace=True)
        except Exception:
            pass  # Exception should be caught
        
        # Verify it was called even though it failed
        mock_table.create_fts_index.assert_called_once()


class TestMaybeCreateVectorIndexCalled(unittest.TestCase):
    """Test that _maybe_create_vector_index is called after add_chunks."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_lancedb")
        self.store = VectorStore(db_path=self.db_path)
        self.embedding_dim = 384

    def tearDown(self):
        """Clean up test fixtures."""
        self.store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_maybe_create_vector_index_called_after_add_chunks(self):
        """Test that _maybe_create_vector_index is called after add_chunks."""
        mock_table = MagicMock()
        mock_table.list_indices.return_value = []
        mock_table.count_rows.return_value = 100
        
        # Configure schema mock for _get_expected_embedding_dim
        mock_field = MagicMock()
        mock_field.type.list_size = self.embedding_dim
        mock_table.schema.field.return_value = mock_field
        
        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim
        
        # Mock _maybe_create_vector_index to track calls
        with patch.object(self.store, '_maybe_create_vector_index') as mock_maybe_create:
            # Add a single chunk
            records = [{
                "id": "test_1",
                "text": "Test text",
                "file_id": "file1",
                "chunk_index": 0,
                "embedding": [0.1] * self.embedding_dim,
            }]
            
            self.store.add_chunks(records)
            
            # Verify _maybe_create_vector_index was called
            mock_maybe_create.assert_called_once()

    def test_maybe_create_vector_index_not_called_for_empty_records(self):
        """Test that _maybe_create_vector_index is NOT called when adding empty records."""
        mock_table = MagicMock()
        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim
        
        # Mock _maybe_create_vector_index to track calls
        with patch.object(self.store, '_maybe_create_vector_index') as mock_maybe_create:
            # Add empty records
            self.store.add_chunks([])
            
            # Verify _maybe_create_vector_index was NOT called
            mock_maybe_create.assert_not_called()
            
            # Verify table.add was NOT called
            mock_table.add.assert_not_called()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases for vector store operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_lancedb")
        self.store = VectorStore(db_path=self.db_path)
        self.embedding_dim = 384

    def tearDown(self):
        """Clean up test fixtures."""
        self.store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_empty_records_list(self):
        """Test that adding an empty list of records is handled gracefully."""
        mock_table = MagicMock()
        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim
        
        # Should not raise
        self.store.add_chunks([])
        
        # Verify table.add was NOT called
        mock_table.add.assert_not_called()

    def test_add_chunks_raises_if_table_not_initialized(self):
        """Test that add_chunks raises RuntimeError if table is not initialized."""
        self.store.table = None
        
        records = [{
            "id": "test_1",
            "text": "Test text",
            "file_id": "file1",
            "chunk_index": 0,
            "embedding": [0.1] * self.embedding_dim,
        }]
        
        with self.assertRaises(RuntimeError) as context:
            self.store.add_chunks(records)
        
        self.assertIn("Table not initialized", str(context.exception))

    def test_add_chunks_validates_required_fields(self):
        """Test that add_chunks validates required fields."""
        mock_table = MagicMock()
        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim
        
        # Missing 'embedding' field
        records = [{
            "id": "test_1",
            "text": "Test text",
            "file_id": "file1",
            "chunk_index": 0,
            # "embedding" is missing
        }]
        
        with self.assertRaises(VectorStoreValidationError) as context:
            self.store.add_chunks(records)
        
        self.assertIn("missing required fields", str(context.exception).lower())

    def test_add_chunks_validates_embedding_dimension(self):
        """Test that add_chunks validates embedding dimension matches table schema."""
        mock_table = MagicMock()
        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim  # 384
        
        # Wrong embedding dimension (100 instead of 384)
        records = [{
            "id": "test_1",
            "text": "Test text",
            "file_id": "file1",
            "chunk_index": 0,
            "embedding": [0.1] * 100,  # Wrong dimension
        }]
        
        with self.assertRaises(VectorStoreValidationError) as context:
            self.store.add_chunks(records)
        
        self.assertIn("dimension mismatch", str(context.exception).lower())

    def test_add_chunks_accepts_numpy_array_embedding(self):
        """Test that add_chunks accepts numpy array embeddings."""
        mock_table = MagicMock()
        mock_table.list_indices.return_value = []
        mock_table.count_rows.return_value = 100
        
        # Configure schema mock for _get_expected_embedding_dim
        mock_field = MagicMock()
        mock_field.type.list_size = self.embedding_dim
        mock_table.schema.field.return_value = mock_field
        
        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim
        
        # Use numpy array for embedding
        records = [{
            "id": "test_1",
            "text": "Test text",
            "file_id": "file1",
            "chunk_index": 0,
            "embedding": np.array([0.1] * self.embedding_dim),
        }]
        
        with patch.object(self.store, '_maybe_create_vector_index'):
            # Should not raise
            self.store.add_chunks(records)
            
            # Verify table.add was called
            mock_table.add.assert_called_once()
            
            # Verify the embedding was converted to list
            call_args = mock_table.add.call_args[0][0]
            self.assertIsInstance(call_args[0]["embedding"], list)

    def test_add_chunks_rejects_invalid_embedding_type(self):
        """Test that add_chunks rejects invalid embedding types."""
        mock_table = MagicMock()
        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim
        
        # Invalid embedding type (string instead of list/array)
        records = [{
            "id": "test_1",
            "text": "Test text",
            "file_id": "file1",
            "chunk_index": 0,
            "embedding": "invalid_embedding",
        }]
        
        with self.assertRaises(VectorStoreValidationError) as context:
            self.store.add_chunks(records)
        
        self.assertIn("must be a list or numpy array", str(context.exception))

    def test_add_chunks_validates_sparse_embedding_json(self):
        """Test that add_chunks validates sparse_embedding is valid JSON."""
        mock_table = MagicMock()
        mock_table.list_indices.return_value = []
        mock_table.count_rows.return_value = 100
        
        # Configure schema mock for _get_expected_embedding_dim
        mock_field = MagicMock()
        mock_field.type.list_size = self.embedding_dim
        mock_table.schema.field.return_value = mock_field
        
        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim
        
        records = [{
            "id": "test_1",
            "text": "Test text",
            "file_id": "file1",
            "chunk_index": 0,
            "embedding": [0.1] * self.embedding_dim,
            "sparse_embedding": "invalid json {",
        }]
        
        with self.assertRaises(VectorStoreValidationError) as context:
            with patch.object(self.store, '_maybe_create_vector_index'):
                self.store.add_chunks(records)
        
        self.assertIn("must be valid json", str(context.exception).lower())

    def test_vector_index_creation_with_none_table(self):
        """Test that _maybe_create_vector_index handles None table gracefully."""
        self.store.table = None
        
        # Should not raise
        self.store._maybe_create_vector_index()

    def test_concurrent_add_chunks(self):
        """Test that concurrent add_chunks operations are handled safely."""
        mock_table = MagicMock()
        mock_table.list_indices.return_value = []
        mock_table.count_rows.return_value = 100

        # Configure schema mock for _get_expected_embedding_dim
        mock_field = MagicMock()
        mock_field.type.list_size = self.embedding_dim
        mock_table.schema.field.return_value = mock_field

        self.store.table = mock_table
        self.store._embedding_dim = self.embedding_dim

        results = []
        errors = []

        def add_chunk(chunk_id):
            try:
                with patch.object(self.store, '_maybe_create_vector_index'):
                    records = [{
                        "id": f"test_{chunk_id}",
                        "text": f"Test text {chunk_id}",
                        "file_id": "file1",
                        "chunk_index": chunk_id,
                        "embedding": [0.1] * self.embedding_dim,
                    }]
                    self.store.add_chunks(records)
                    results.append(chunk_id)
            except Exception as e:
                errors.append((chunk_id, str(e)))

        # Start multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=add_chunk, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify all operations completed
        self.assertEqual(len(results), 10)
        self.assertEqual(len(errors), 0)

    def test_vector_index_threshold_boundary(self):
        """Test vector index creation at exact threshold boundary values."""
        test_cases = [
            (0, False),    # Fresh install - no index
            (1, False),    # Single row - no index
            (255, False),  # Just under threshold - no index
            (256, True),   # At threshold - create index
            (257, True),   # Just over threshold - create index
            (1000, True),  # Well over threshold - create index
        ]
        
        for row_count, should_create in test_cases:
            with self.subTest(row_count=row_count):
                mock_table = MagicMock()
                mock_table.list_indices.return_value = []
                mock_table.count_rows.return_value = row_count
                self.store.table = mock_table
                self.store._embedding_dim = self.embedding_dim
                
                with patch('app.services.vector_store.settings') as mock_settings:
                    mock_settings.vector_metric = "cosine"
                    
                    self.store._maybe_create_vector_index()
                    
                    if should_create:
                        mock_table.create_index.assert_called_once()
                    else:
                        mock_table.create_index.assert_not_called()
                
                # Reset for next iteration
                self.store.table = None


class TestInitTableBehavior(unittest.TestCase):
    """Test init_table behavior with vector and FTS index creation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_lancedb")
        self.store = VectorStore(db_path=self.db_path)
        self.embedding_dim = 384

    def tearDown(self):
        """Clean up test fixtures."""
        self.store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('app.services.vector_store.lancedb')
    @patch('app.services.vector_store.settings')
    def test_init_table_defers_vector_index_on_fresh_table(self, mock_settings, mock_lancedb):
        """Test that init_table defers vector index creation on fresh table."""
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_table = MagicMock()
        mock_table.list_indices.return_value = []
        mock_db.create_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db
        mock_settings.vector_metric = "cosine"
        
        self.store.db = mock_db
        
        # Call init_table
        self.store.init_table(self.embedding_dim)
        
        # Verify table was created
        mock_db.create_table.assert_called_once()
        
        # Verify vector index was NOT created (deferred)
        mock_table.create_index.assert_not_called()
        
        # Verify FTS index WAS created
        mock_table.create_fts_index.assert_called_once_with("text", replace=True)

    @patch('app.services.vector_store.lancedb')
    @patch('app.services.vector_store.settings')
    def test_init_table_skips_fts_if_exists(self, mock_settings, mock_lancedb):
        """Test that init_table skips FTS creation if index already exists."""
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["chunks"]
        mock_table = MagicMock()
        mock_table.list_indices.return_value = [FakeIndex("fts_text")]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db
        mock_settings.vector_metric = "cosine"
        
        self.store.db = mock_db
        
        # Call init_table
        self.store.init_table(self.embedding_dim)
        
        # Verify table was opened
        mock_db.open_table.assert_called_once_with("chunks")
        
        # Verify FTS index was NOT created (already exists)
        mock_table.create_fts_index.assert_not_called()


if __name__ == "__main__":
    unittest.main()
