"""Create AES-GCM encrypted backups of the SQLite database."""

import argparse
import secrets
import sys
from datetime import datetime
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from backend.app.config import settings
from backend.app.services.secret_manager import SecretManager


def backup_sqlite(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    key, version = SecretManager().get_aes_key()
    sqlite_path = Path(settings.sqlite_path)
    nonce = secrets.token_bytes(12)
    aesgcm = AESGCM(key)
    data = sqlite_path.read_bytes()
    ciphertext = aesgcm.encrypt(nonce, data, None)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    backup_path = output_dir / f"db_{timestamp}_k{version}.enc"
    with open(backup_path, "wb") as fout:
        fout.write(nonce)
        fout.write(ciphertext)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Encrypt SQLite backup using AES-GCM")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backups"),
        help="Directory where encrypted backups are stored",
    )
    args = parser.parse_args()

    backup_path = backup_sqlite(args.output)
    print(f"Backup written to {backup_path}")


if __name__ == "__main__":
    main()
