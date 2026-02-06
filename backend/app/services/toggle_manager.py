"""Toggle manager with caching and persistence."""

import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional

from app.models.database import get_db_connection


@dataclass
class ToggleCacheEntry:
    timestamp: float
    enabled: bool


class ToggleManager:
    """Reads and writes feature toggles backed by SQLite."""

    CACHE_TTL = 30.0  # seconds

    def __init__(self, sqlite_path: str) -> None:
        self.sqlite_path = sqlite_path
        self._cache: dict[str, ToggleCacheEntry] = {}
        self._lock = threading.Lock()

    def get_toggle(self, feature: str, default: bool = False) -> bool:
        now = time.time()
        with self._lock:
            entry = self._cache.get(feature)
            if entry and now - entry.timestamp < self.CACHE_TTL:
                return entry.enabled
        conn = get_db_connection(self.sqlite_path)
        try:
            row = conn.execute(
                "SELECT enabled FROM admin_toggles WHERE feature = ?",
                (feature,)
            ).fetchone()
            value = default if row is None else bool(row["enabled"])
        finally:
            conn.close()
        with self._lock:
            self._cache[feature] = ToggleCacheEntry(timestamp=now, enabled=value)
        return value

    def set_toggle(self, feature: str, enabled: bool) -> None:
        conn = get_db_connection(self.sqlite_path)
        try:
            conn.execute(
                "INSERT INTO admin_toggles(feature, enabled) VALUES(?, ?) ON CONFLICT(feature) DO UPDATE SET enabled=excluded.enabled, updated_at=CURRENT_TIMESTAMP",
                (feature, int(enabled))
            )
            conn.commit()
        finally:
            conn.close()
        with self._lock:
            self._cache[feature] = ToggleCacheEntry(timestamp=time.time(), enabled=enabled)

    def clear_cache(self, feature: Optional[str] = None) -> None:
        with self._lock:
            if feature:
                self._cache.pop(feature, None)
            else:
                self._cache.clear()
