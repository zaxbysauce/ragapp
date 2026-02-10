"""Clear embeddings and reset file statuses for model migration."""

import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from backend.app.config import settings
from backend.app.models.database import get_db_connection


def reset_all_embeddings():
    """Delete all embeddings and reset file statuses."""
    
    print("=== Migration: Clearing Embeddings ===")
    print(f"SQLite DB: {settings.sqlite_path}")
    print(f"LanceDB path: {settings.lancedb_path}")
    print()
    
    # 1. Clear LanceDB by deleting entire database directory
    print("[1/3] Clearing LanceDB database...")
    lancedb_path = settings.lancedb_path
    
    if lancedb_path.exists():
        # Option A: Delete entire lancedb directory and recreate
        shutil.rmtree(lancedb_path)
        lancedb_path.mkdir(parents=True, exist_ok=True)
        print(f"      Deleted LanceDB directory and recreated")
    else:
        print(f"      LanceDB directory does not exist - nothing to clear")
    
    # 2. Reset file statuses in SQLite
    print("[2/3] Resetting file statuses...")
    conn = get_db_connection(str(settings.sqlite_path))
    try:
        # Update all indexed files to pending
        cursor = conn.execute("""
            UPDATE files 
            SET status = 'pending', 
                chunk_count = 0, 
                processed_at = NULL, 
                modified_at = CURRENT_TIMESTAMP
            WHERE status = 'indexed'
        """)
        updated_count = cursor.rowcount
        conn.commit()
        print(f"      Reset {updated_count} files from 'indexed' to 'pending'")
    finally:
        conn.close()
    
    print("[3/3] Migration complete")
    print()
    print("Next steps:")
    print("1. Update EMBEDDING_MODEL in .env or docker-compose.yml")
    print("   Recommended: qwen3-embed:4b (text embedding)")
    print("   Note: qwen3-vl-embedding-2b is vision-language, not ideal for text RAG")
    print("2. Restart backend service:")
    print("   docker compose restart backend")
    print("3. Background processor will auto-reprocess all pending files")
    print()


if __name__ == "__main__":
    reset_all_embeddings()
