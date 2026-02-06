"""Migration helper that toggles maintenance, backups, and schema updates."""

import argparse
import shutil
import sys
import time
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from backend.app.config import settings
from backend.app.models.database import run_migrations
from backend.app.services.maintenance import MaintenanceService
from backend.app.services.secret_manager import SecretManager
from scripts.backup_sqlite import backup_sqlite


def decrypt_backup(path: Path, key: bytes) -> Path:
    data = path.read_bytes()
    nonce = data[:12]
    ciphertext = data[12:]
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    target = Path(settings.sqlite_path)
    target.write_bytes(plaintext)
    return target


def migrate(rollback: bool, backup: Path | None, retention: int) -> None:
    maintenance = MaintenanceService(str(settings.sqlite_path))
    secret_manager = SecretManager()
    key, key_version = secret_manager.get_aes_key()
    if rollback:
        if not backup:
            raise SystemExit("Rollback requires --backup")
        decrypt_backup(backup, key)
        print(f"Restored {backup} using key {key_version}")
        sys.exit(0)

    maintenance.set_flag(True, "migration in progress")
    try:
        backup_path = backup_sqlite(Path("backups"))
        print(f"Backup created: {backup_path}")
        run_migrations(str(settings.sqlite_path))
    finally:
        maintenance.set_flag(False, "migration complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run migrations with maintenance mode + backups")
    parser.add_argument("--rollback", action="store_true", help="Restore from backup")
    parser.add_argument("--backup", type=Path, help="Encrypted backup to restore when rolling back")
    parser.add_argument("--retention", type=int, default=30, help="Retention days for backups")
    args = parser.parse_args()
    migrate(args.rollback, args.backup, args.retention)


if __name__ == "__main__":
    main()
