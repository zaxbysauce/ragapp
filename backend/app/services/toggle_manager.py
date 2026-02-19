"""Toggle manager with caching and persistence."""

import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional

from app.models.database import SQLiteConnectionPool
from app.utils.retry import with_retry


@dataclass
class ToggleCacheEntry:
    timestamp: float
    enabled: bool


class ToggleManager:
    """Reads and writes feature toggles backed by SQLite."""

    CACHE_TTL = 30.0  # seconds

    def __init__(self, pool: SQLiteConnectionPool) -> None:
        self.pool = pool
        self._cache: dict[str, ToggleCacheEntry] = {}
        self._lock = threading.Lock()

    @with_retry(max_attempts=3, retry_exceptions=(sqlite3.Error,), raise_last_exception=True)
    def get_toggle(self, feature: str, default: bool = False) -> bool:
        now = time.time()
        with self._lock:
            entry = self._cache.get(feature)
            if entry and now - entry.timestamp < self.CACHE_TTL:
                return entry.enabled
        conn = self.pool.get_connection()
        try:
            row = conn.execute(
                "SELECT enabled FROM admin_toggles WHERE feature = ?",
                (feature,)
            ).fetchone()
            value = default if row is None else bool(row["enabled"])
        finally:
            self.pool.release_connection(conn)
        with self._lock:
            self._cache[feature] = ToggleCacheEntry(timestamp=now, enabled=value)
        return value

    @with_retry(max_attempts=3, retry_exceptions=(sqlite3.Error,), raise_last_exception=True)
    def set_toggle(self, feature: str, enabled: bool) -> None:
        conn = self.pool.get_connection()
        try:
            conn.execute(
                "INSERT INTO admin_toggles(feature, enabled) VALUES(?, ?) ON CONFLICT(feature) DO UPDATE SET enabled=excluded.enabled, updated_at=CURRENT_TIMESTAMP",
                (feature, int(enabled))
            )
            conn.commit()
        finally:
            self.pool.release_connection(conn)
        with self._lock:
            self._cache[feature] = ToggleCacheEntry(timestamp=time.time(), enabled=enabled)

    def clear_cache(self, feature: Optional[str] = None) -> None:
        with self._lock:
            if feature:
                self._cache.pop(feature, None)
            else:
                self._cache.clear()
