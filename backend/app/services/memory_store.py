"""Memory storage service backed by SQLite + FTS5."""

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings
from app.models.database import get_db_connection


class MemoryStoreError(Exception):
    """General memory store error."""


class MemoryDetectionError(MemoryStoreError):
    """Raised when a memory pattern cannot be parsed."""


@dataclass
class MemoryRecord:
    id: int
    content: str
    category: Optional[str]
    tags: Optional[str]
    source: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


class MemoryStore:
    """Provides memory storage and retrieval backed by SQLite + FTS5."""

    MEMORY_PATTERNS = [
        re.compile(r"remember that\s+(?P<memory>.+?)(?:\.|$)", re.IGNORECASE),
        re.compile(r"don't forget\s+(?P<memory>.+?)(?:\.|$)", re.IGNORECASE),
        re.compile(r"keep in mind\s+(?P<memory>.+?)(?:\.|$)", re.IGNORECASE),
        re.compile(r"note that\s+(?P<memory>.+?)(?:\.|$)", re.IGNORECASE),
    ]

    def __init__(self, sqlite_path: Optional[Path] = None) -> None:
        self.sqlite_path = str(sqlite_path or settings.sqlite_path)

    def _connect(self):
        return get_db_connection(self.sqlite_path)

    def add_memory(
        self,
        content: str,
        category: Optional[str] = None,
        tags: Optional[str] = None,
        source: Optional[str] = None,
    ) -> MemoryRecord:
        if not content or not content.strip():
            raise MemoryStoreError("Memory content cannot be empty")

        sql = """
        INSERT INTO memories (content, category, tags, source)
        VALUES (?, ?, ?, ?)
        """
        conn = self._connect()
        try:
            cursor = conn.execute(sql, (content, category, tags, source))
            conn.commit()
            memory_id = cursor.lastrowid
            if memory_id is None:
                raise MemoryStoreError("Failed to insert memory")
            # Fetch created_at and updated_at for the inserted row
            cursor = conn.execute(
                "SELECT created_at, updated_at FROM memories WHERE id = ?", (memory_id,)
            )
            row = cursor.fetchone()
            created_at = row[0] if row else None
            updated_at = row[1] if row else None
        finally:
            conn.close()

        return MemoryRecord(
            id=memory_id,
            content=content,
            category=category,
            tags=tags,
            source=source,
            created_at=created_at,
            updated_at=updated_at,
        )

    def search_memories(self, query: str, limit: int = 5) -> List[MemoryRecord]:
        if not query or not query.strip():
            return []

        # Sanitize query to prevent FTS5 syntax injection
        # Keep alphanumeric, spaces, and basic punctuation; strip FTS5 special chars
        sanitized_query = re.sub(r'[*:"^()]+', '', query)
        sanitized_query = re.sub(r'[^\w\s.,?]+', '', sanitized_query)

        if not sanitized_query.strip():
            return []

        conn = self._connect()
        try:
            try:
                cursor = conn.execute(
                    """
                    SELECT m.id, m.content, m.category, m.tags, m.source, m.created_at, m.updated_at
                    FROM memories_fts f
                    JOIN memories m ON f.rowid = m.id
                    WHERE memories_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (sanitized_query, limit),
                )
                rows = cursor.fetchall()
            except sqlite3.Error as e:
                raise MemoryStoreError(f"FTS query failed: {e}")
        finally:
            conn.close()

        records: List[MemoryRecord] = []
        for row in rows:
            records.append(
                MemoryRecord(
                    id=row[0],
                    content=row[1],
                    category=row[2],
                    tags=row[3],
                    source=row[4],
                    created_at=row[5],
                    updated_at=row[6],
                )
            )
        return records

    def detect_memory_intent(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None

        for pattern in self.MEMORY_PATTERNS:
            match = pattern.search(text)
            if match and match.groupdict().get("memory"):
                memory_content = match.group("memory").strip()
                if memory_content:
                    return memory_content
        return None



