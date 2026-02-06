"""Maintenance mode flag service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.config import settings
from app.models.database import get_db_connection


class MaintenanceError(Exception):
    pass


@dataclass
class MaintenanceFlag:
    enabled: bool
    reason: str
    version: int
    updated_at: Optional[str]


class MaintenanceService:
    FLAG_NAME = "maintenance"

    def __init__(self, sqlite_path: str) -> None:
        self.sqlite_path = sqlite_path
        self._ensure_flag_row()

    def _ensure_flag_row(self) -> None:
        conn = get_db_connection(self.sqlite_path)
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO system_flags(name, value, version, reason)
                VALUES (?, 0, 0, '')
                """,
                (self.FLAG_NAME,)
            )
            conn.commit()
        finally:
            conn.close()

    def get_flag(self) -> MaintenanceFlag:
        conn = get_db_connection(self.sqlite_path)
        try:
            row = conn.execute(
                "SELECT value, reason, version, updated_at FROM system_flags WHERE name = ?",
                (self.FLAG_NAME,),
            ).fetchone()
            if row is None:
                raise MaintenanceError("Maintenance flag missing")
            return MaintenanceFlag(
                enabled=bool(row[0]),
                reason=row[1] or "",
                version=row[2] or 0,
                updated_at=row[3],
            )
        finally:
            conn.close()

    def set_flag(self, enabled: bool, reason: str = "") -> None:
        attempts = 0
        while True:
            flag = self.get_flag()
            conn = get_db_connection(self.sqlite_path)
            try:
                cursor = conn.execute(
                    """
                    UPDATE system_flags
                    SET value = ?, reason = ?, version = version + 1, updated_at = CURRENT_TIMESTAMP
                    WHERE name = ? AND version = ?
                    """,
                    (int(enabled), reason, self.FLAG_NAME, flag.version),
                )
                if cursor.rowcount:
                    conn.commit()
                    return
            finally:
                conn.close()
            attempts += 1
            if attempts > 3:
                raise MaintenanceError("Failed to update maintenance flag")
