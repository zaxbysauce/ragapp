"""Unit tests for DocumentProcessor using SQL file path."""

import asyncio
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Stub missing optional dependencies
try:
    import lancedb
except ImportError:
    import types
    sys.modules['lancedb'] = types.ModuleType('lancedb')

try:
    import pyarrow
except ImportError:
    import types
    sys.modules['pyarrow'] = types.ModuleType('pyarrow')

try:
    from unstructured.partition.auto import partition
except ImportError:
    import types
    _unstructured = types.ModuleType('unstructured')
    _unstructured.__path__ = []
    _unstructured.partition = types.ModuleType('unstructured.partition')
    _unstructured.partition.__path__ = []
    _unstructured.partition.auto = types.ModuleType('unstructured.partition.auto')
    _unstructured.partition.auto.partition = lambda *args, **kwargs: []
    _unstructured.chunking = types.ModuleType('unstructured.chunking')
    _unstructured.chunking.__path__ = []
    _unstructured.chunking.title = types.ModuleType('unstructured.chunking.title')
    _unstructured.chunking.title.chunk_by_title = lambda *args, **kwargs: []
    _unstructured.documents = types.ModuleType('unstructured.documents')
    _unstructured.documents.__path__ = []
    _unstructured.documents.elements = types.ModuleType('unstructured.documents.elements')
    _unstructured.documents.elements.Element = type('Element', (), {})
    sys.modules['unstructured'] = _unstructured
    sys.modules['unstructured.partition'] = _unstructured.partition
    sys.modules['unstructured.partition.auto'] = _unstructured.partition.auto
    sys.modules['unstructured.chunking'] = _unstructured.chunking
    sys.modules['unstructured.chunking.title'] = _unstructured.chunking.title
    sys.modules['unstructured.documents'] = _unstructured.documents
    sys.modules['unstructured.documents.elements'] = _unstructured.documents.elements

from app.models.database import init_db
from app.services.document_processor import (
    DocumentProcessor,
    DuplicateFileError,
    ProcessedDocument
)
from app.config import settings


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for DocumentProcessor with SQL file processing."""

    def setUp(self):
        """Create temporary database and SQL file for each test."""
        # Create temp directory for all test files
        self.temp_dir = tempfile.mkdtemp()

        # Create temp sqlite file
        self.temp_db_path = os.path.join(self.temp_dir, 'test.db')

        # Initialize the database
        init_db(self.temp_db_path)

        # Create temp .sql file with CREATE TABLE statement
        self.sql_file_path = os.path.join(self.temp_dir, 'test_schema.sql')
        sql_content = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""
        with open(self.sql_file_path, 'w', encoding='utf-8') as f:
            f.write(sql_content)

        # Monkeypatch settings.data_dir so sqlite_path resolves to temp path
        self._original_data_dir = settings.data_dir
        settings.data_dir = Path(self.temp_dir)

        # Create DocumentProcessor instance
        self.processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
        # Override sqlite_path directly on instance
        self.processor.sqlite_path = self.temp_db_path

    def tearDown(self):
        """Clean up temporary files."""
        settings.data_dir = self._original_data_dir

        # Remove temp files
        if os.path.exists(self.sql_file_path):
            os.remove(self.sql_file_path)
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_process_file_returns_valid_result(self):
        """Test that process_file returns valid ProcessedDocument with SQL file."""
        # Process the SQL file
        result = asyncio.run(self.processor.process_file(self.sql_file_path))

        # Assert result is ProcessedDocument
        self.assertIsInstance(result, ProcessedDocument)

        # Assert file_id is int
        self.assertIsInstance(result.file_id, int)
        self.assertGreater(result.file_id, 0)

        # Assert chunks list is not empty
        self.assertIsInstance(result.chunks, list)
        self.assertGreater(len(result.chunks), 0)

    def test_process_file_updates_db_status(self):
        """Test that process_file updates DB status to indexed with chunk_count."""
        # Process the SQL file
        result = asyncio.run(self.processor.process_file(self.sql_file_path))

        # Query database to verify status
        conn = sqlite3.connect(self.temp_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT status, chunk_count FROM files WHERE id = ?",
            (result.file_id,)
        )
        row = cursor.fetchone()
        conn.close()

        # Assert status is 'indexed'
        self.assertIsNotNone(row)
        self.assertEqual(row['status'], 'indexed')

        # Assert chunk_count matches number of chunks
        self.assertEqual(row['chunk_count'], len(result.chunks))

    def test_process_file_raises_duplicate_error_on_second_call(self):
        """Test that second call with same file raises DuplicateFileError."""
        # Process the SQL file first time
        asyncio.run(self.processor.process_file(self.sql_file_path))

        # Second call should raise DuplicateFileError
        with self.assertRaises(DuplicateFileError):
            asyncio.run(self.processor.process_file(self.sql_file_path))

    def test_process_file_extracts_correct_chunks(self):
        """Test that SQL file is correctly parsed into chunks."""
        # Process the SQL file
        result = asyncio.run(self.processor.process_file(self.sql_file_path))

        # Should have 2 chunks (users table and posts table)
        self.assertEqual(len(result.chunks), 2)

        # Verify chunk content contains expected table names
        chunk_texts = [chunk.text for chunk in result.chunks]
        self.assertTrue(
            any('users' in text for text in chunk_texts),
            "Expected one chunk to contain 'users' table"
        )
        self.assertTrue(
            any('posts' in text for text in chunk_texts),
            "Expected one chunk to contain 'posts' table"
        )


if __name__ == '__main__':
    unittest.main()
