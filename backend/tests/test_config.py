"""Unit tests for config Settings defaults."""

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.config import Settings


class TestSettingsDefaults(unittest.TestCase):
    """Test that Settings loads correct default values."""

    def test_settings_defaults(self):
        """Test Settings() loads all defaults with no env overrides."""
        settings = Settings()

        self.assertEqual(settings.data_dir, Path("/data/knowledgevault"))
        self.assertEqual(settings.ollama_embedding_url, "http://host.docker.internal:11434")
        self.assertEqual(settings.ollama_chat_url, "http://host.docker.internal:11434")
        self.assertEqual(settings.embedding_model, "nomic-embed-text")
        self.assertEqual(settings.chat_model, "qwen2.5:32b")
        self.assertEqual(settings.chunk_size, 512)
        self.assertEqual(settings.chunk_overlap, 50)
        self.assertEqual(settings.max_context_chunks, 10)
        self.assertTrue(settings.auto_scan_enabled)
        self.assertEqual(settings.auto_scan_interval_minutes, 60)
        self.assertEqual(settings.log_level, "INFO")
        self.assertEqual(settings.port, 8080)


class TestRagRelevanceThreshold(unittest.TestCase):
    """Test rag_relevance_threshold default and env override."""

    def test_rag_relevance_threshold_default(self):
        """Test that default rag_relevance_threshold is 0.1."""
        settings = Settings()
        self.assertEqual(settings.rag_relevance_threshold, 0.1)

    def test_rag_relevance_threshold_env_override(self):
        """Test that rag_relevance_threshold can be overridden via environment variable."""
        original_value = os.environ.get("RAG_RELEVANCE_THRESHOLD")
        try:
            os.environ["RAG_RELEVANCE_THRESHOLD"] = "0.5"
            settings = Settings()
            self.assertEqual(settings.rag_relevance_threshold, 0.5)
        finally:
            if original_value is not None:
                os.environ["RAG_RELEVANCE_THRESHOLD"] = original_value
            else:
                del os.environ["RAG_RELEVANCE_THRESHOLD"]


class TestSettingsPropertyPaths(unittest.TestCase):
    """Test Settings property paths based on data_dir."""

    def test_documents_dir_property(self):
        """Test documents_dir returns data_dir/documents."""
        settings = Settings()
        self.assertEqual(settings.documents_dir, Path("/data/knowledgevault/documents"))

    def test_uploads_dir_property(self):
        """Test uploads_dir returns data_dir/uploads."""
        settings = Settings()
        self.assertEqual(settings.uploads_dir, Path("/data/knowledgevault/uploads"))

    def test_library_dir_property(self):
        """Test library_dir returns data_dir/library."""
        settings = Settings()
        self.assertEqual(settings.library_dir, Path("/data/knowledgevault/library"))

    def test_lancedb_path_property(self):
        """Test lancedb_path returns data_dir/lancedb."""
        settings = Settings()
        self.assertEqual(settings.lancedb_path, Path("/data/knowledgevault/lancedb"))

    def test_sqlite_path_property(self):
        """Test sqlite_path returns data_dir/app.db."""
        settings = Settings()
        self.assertEqual(settings.sqlite_path, Path("/data/knowledgevault/app.db"))

    def test_property_paths_with_custom_data_dir(self):
        """Test property paths update correctly when data_dir is overridden."""
        original_value = os.environ.get("DATA_DIR")
        try:
            os.environ["DATA_DIR"] = "/custom/path"
            settings = Settings()
            self.assertEqual(settings.data_dir, Path("/custom/path"))
            self.assertEqual(settings.documents_dir, Path("/custom/path/documents"))
            self.assertEqual(settings.uploads_dir, Path("/custom/path/uploads"))
            self.assertEqual(settings.library_dir, Path("/custom/path/library"))
            self.assertEqual(settings.lancedb_path, Path("/custom/path/lancedb"))
            self.assertEqual(settings.sqlite_path, Path("/custom/path/app.db"))
        finally:
            if original_value is not None:
                os.environ["DATA_DIR"] = original_value
            else:
                del os.environ["DATA_DIR"]


if __name__ == "__main__":
    unittest.main()
