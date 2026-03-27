"""
Unit tests for vector store index creation fixes.

Tests Bug B fix: Vector index deferred until >=256 rows
Tests Bug C fix: FTS index created without blocking startup
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa

# Use try/except for optional lancedb dependency
try:
    import lancedb

    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

# Conditional imports
if LANCEDB_AVAILABLE:
    from app.services.vector_store import VectorStore, VECTOR_INDEX_MIN_ROWS
else:
    VECTOR_INDEX_MIN_ROWS = 256  # Fallback for tests without lancedb


@unittest.skipUnless(LANCEDB_AVAILABLE, "LanceDB not available")
class TestVectorStoreIndexCreation(unittest.TestCase):
    """Test vector store index creation behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.lancedb"
        self.store = VectorStore(db_path=self.db_path)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_vector_index_deferred_on_table_creation(self):
        """Test that vector index is deferred when table is first created."""
        # Initialize table with small dimension for testing
        self.store.init_table(embedding_dim=384)

        # Check that table exists but vector index does not (too few rows)
        self.assertIsNotNone(self.store.table)

        # List indices - should be empty or only have FTS
        indices = self.store.table.list_indices()
        index_names = [idx.name for idx in indices]

        # Vector index should not exist yet (no rows added)
        self.assertNotIn("embedding_idx", index_names)

    def test_vector_index_created_after_threshold(self):
        """Test that vector index is created after reaching row threshold."""
        # Initialize table
        self.store.init_table(embedding_dim=384)

        # Add enough records to trigger index creation
        records = []
        for i in range(VECTOR_INDEX_MIN_ROWS + 10):
            records.append({
                "id": f"test_{i}",
                "text": f"Test text {i}",
                "file_id": "file1",
                "vault_id": "1",
                "chunk_index": i,
                "chunk_scale": "default",
                "sparse_embedding": None,
                "metadata": "{}",
                "embedding": [0.1] * 384,
            })

        # Add chunks - this should trigger index creation
        self.store.add_chunks(records)

        # Check that vector index now exists
        indices = self.store.table.list_indices()
        index_names = [idx.name for idx in indices]

        self.assertIn("embedding_idx", index_names)

    def test_vector_index_not_created_below_threshold(self):
        """Test that vector index is NOT created below row threshold."""
        # Initialize table
        self.store.init_table(embedding_dim=384)

        # Add fewer records than threshold
        records = []
        for i in range(10):  # Well below 256 threshold
            records.append({
                "id": f"test_{i}",
                "text": f"Test text {i}",
                "file_id": "file1",
                "vault_id": "1",
                "chunk_index": i,
                "chunk_scale": "default",
                "sparse_embedding": None,
                "metadata": "{}",
                "embedding": [0.1] * 384,
            })

        self.store.add_chunks(records)

        # Check that vector index still does not exist
        indices = self.store.table.list_indices()
        index_names = [idx.name for idx in indices]

        self.assertNotIn("embedding_idx", index_names)

    def test_fts_index_created_on_init(self):
        """Test that FTS index is created during table initialization.

        LanceDB names the index "text_idx" when created via create_fts_index("text").
        We accept either "text_idx" or the legacy "fts_text" name.
        """
        self.store.init_table(embedding_dim=384)

        # FTS index should exist
        indices = self.store.table.list_indices()
        index_names = [idx.name for idx in indices]

        fts_exists = any(name in ("text_idx", "fts_text") for name in index_names)
        self.assertTrue(fts_exists, f"No FTS index found; got: {index_names}")

    def test_fts_index_not_recreated_if_exists(self):
        """Test that FTS index is not recreated if it already exists."""
        # First initialization creates FTS index
        self.store.init_table(embedding_dim=384)

        # Close and reopen store (simulates restart)
        store2 = VectorStore(db_path=self.db_path)
        store2.init_table(embedding_dim=384)

        # Should still have exactly one FTS index (either name)
        indices = store2.table.list_indices()
        fts_indices = [idx for idx in indices if idx.name in ("text_idx", "fts_text")]

        self.assertEqual(len(fts_indices), 1)

    def test_maybe_create_vector_index_idempotent(self):
        """Test that _maybe_create_vector_index is idempotent."""
        self.store.init_table(embedding_dim=384)

        # Add threshold records
        records = []
        for i in range(VECTOR_INDEX_MIN_ROWS + 10):
            records.append({
                "id": f"test_{i}",
                "text": f"Test text {i}",
                "file_id": "file1",
                "vault_id": "1",
                "chunk_index": i,
                "chunk_scale": "default",
                "sparse_embedding": None,
                "metadata": "{}",
                "embedding": [0.1] * 384,
            })

        self.store.add_chunks(records)

        # Call _maybe_create_vector_index again - should not fail
        self.store._maybe_create_vector_index()

        # Should still have exactly one vector index
        indices = self.store.table.list_indices()
        vector_indices = [idx for idx in indices if idx.name == "embedding_idx"]

        self.assertEqual(len(vector_indices), 1)


@unittest.skipUnless(LANCEDB_AVAILABLE, "LanceDB not available")
class TestVectorStoreDeferredIndexLogging(unittest.TestCase):
    """Test logging behavior for deferred index creation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.lancedb"
        self.store = VectorStore(db_path=self.db_path)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    @patch("app.services.vector_store.logger")
    def test_defer_message_logged_on_table_creation(self, mock_logger):
        """Test that defer message is logged when table is created."""
        self.store.init_table(embedding_dim=384)

        # Check that defer message was logged
        info_calls = [call for call in mock_logger.info.call_args_list
                      if "deferred" in str(call).lower()]
        self.assertTrue(len(info_calls) > 0)

    @patch("app.services.vector_store.logger")
    def test_index_creation_logged_at_threshold(self, mock_logger):
        """Test that index creation is logged when threshold is reached."""
        self.store.init_table(embedding_dim=384)

        # Add threshold records
        records = []
        for i in range(VECTOR_INDEX_MIN_ROWS + 10):
            records.append({
                "id": f"test_{i}",
                "text": f"Test text {i}",
                "file_id": "file1",
                "vault_id": "1",
                "chunk_index": i,
                "chunk_scale": "default",
                "sparse_embedding": None,
                "metadata": "{}",
                "embedding": [0.1] * 384,
            })

        self.store.add_chunks(records)

        # Check that creation message was logged
        info_calls = [call for call in mock_logger.info.call_args_list
                      if "created" in str(call).lower() and "index" in str(call).lower()]
        self.assertTrue(len(info_calls) > 0)


class TestVectorStoreIndexConstants(unittest.TestCase):
    """Test constants related to vector store indexing."""

    def test_vector_index_min_rows_value(self):
        """Test that VECTOR_INDEX_MIN_ROWS has expected value."""
        if LANCEDB_AVAILABLE:
            from app.services.vector_store import VECTOR_INDEX_MIN_ROWS
            self.assertEqual(VECTOR_INDEX_MIN_ROWS, 256)
        else:
            self.assertEqual(VECTOR_INDEX_MIN_ROWS, 256)


if __name__ == "__main__":
    unittest.main()
