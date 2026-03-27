"""
Vault-specific upload path provider.

Provides per-vault upload directories and handles migration from legacy flat structure.
"""
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class UploadPathProvider:
    """Provides vault-specific upload paths."""
    
    def get_upload_dir(self, vault_id: int) -> Path:
        """
        Returns the upload directory for a specific vault.

        Creates the directory if it doesn't exist.
        """
        vault_dir = settings.vault_uploads_dir(vault_id)
        vault_dir.mkdir(parents=True, exist_ok=True)
        return vault_dir
    
    def resolve(self, filename: str, vault_id: int) -> Path:
        """Returns full path for an existing upload file."""
        return self.get_upload_dir(vault_id) / filename
    
    def get_legacy(self, filename: str) -> Path:
        """Fallback: returns path in the old flat uploads directory."""
        return settings.uploads_dir / filename
    
    def is_migrated(self, filename: str, vault_id: int) -> bool:
        """Check if a file has been migrated to the new vault location."""
        return self.resolve(filename, vault_id).exists()
    
    def file_exists(self, filename: str, vault_id: int) -> bool:
        """Check if file exists in either new or legacy location."""
        if self.is_migrated(filename, vault_id):
            return True
        # Check legacy location
        return self.get_legacy(filename).exists()
    
    def resolve_any(self, filename: str, vault_id: int) -> Optional[Path]:
        """Find file in new location, then legacy, return None if not found."""
        new_path = self.resolve(filename, vault_id)
        if new_path.exists():
            return new_path
        legacy_path = self.get_legacy(filename)
        if legacy_path.exists():
            return legacy_path
        return None


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    total: int
    migrated: int
    failed: list[str]
    can_rollback: bool


def migrate_uploads(dry_run: bool = False) -> MigrationResult:
    """
    Migrate files from flat uploads directory to per-vault directories.
    
    Args:
        dry_run: If True, only count files without moving them
        
    Returns:
        MigrationResult with counts and any failures
    """
    provider = UploadPathProvider()
    
    # Get all files in old location
    old_dir = settings.uploads_dir
    if not old_dir.exists():
        logger.info("No legacy uploads directory found, skipping migration")
        return MigrationResult(total=0, migrated=0, failed=[], can_rollback=False)
    
    old_files = list(old_dir.glob("*"))
    if not old_files:
        logger.info("No files in legacy uploads directory, skipping migration")
        return MigrationResult(total=0, migrated=0, failed=[], can_rollback=False)
    
    logger.info(f"Starting migration of {len(old_files)} files")
    
    migrated = 0
    failed = []
    
    for f in old_files:
        if not f.is_file():
            continue
            
        # Lookup vault_id from database
        vault_id = _lookup_vault_id(f.name)
        
        # Determine destination
        dest_dir = provider.get_upload_dir(vault_id)
        dest_path = dest_dir / f.name
        # Validate no traversal
        try:
            dest_path.resolve().relative_to(dest_dir.resolve())
        except ValueError:
            logger.warning("Skipping file with suspicious name: %s", f.name)
            continue

        if dry_run:
            logger.info(f"[DRY RUN] Would move {f.name} to vault {vault_id}")
            migrated += 1
            continue
        
        try:
            # Copy to new location
            shutil.copy2(f, dest_path)
            
            # Verify size match
            if f.stat().st_size == dest_path.stat().st_size:
                # Rename source instead of deleting - safe backup
                backup_path = f.with_suffix(f.suffix + ".migrated")
                if backup_path.exists():
                    stem = f.stem
                    suffix = f.suffix
                    counter = 1
                    while backup_path.exists():
                        backup_path = f.parent / f"{stem}.migrated.{counter}{suffix}"
                        counter += 1
                f.rename(backup_path)
                migrated += 1
                logger.debug(f"Migrated {f.name} to vault {vault_id}")
            else:
                dest_path.unlink()  # Rollback copy
                failed.append(f.name)
                logger.warning(f"Size mismatch for {f.name}, skipping")
        except Exception as e:
            failed.append(f.name)
            logger.error(f"Failed to migrate {f.name}: {e}")
    
    logger.info(f"Migration complete: {migrated}/{len(old_files)} files migrated, {len(failed)} failed")
    
    return MigrationResult(
        total=len(old_files),
        migrated=migrated,
        failed=failed,
        can_rollback=migrated > 0
    )


def _lookup_vault_id(filename: str) -> int:
    """
    Lookup vault_id for a file from the database.
    
    Returns vault_id if found, falls back to orphan vault (default vault).
    """
    try:
        from app.models.database import get_pool
        pool = get_pool(str(settings.sqlite_path))
        conn = pool.get_connection()
        try:
            # Try to find by file_name
            row = conn.execute(
                "SELECT vault_id FROM files WHERE file_name = ?",
                (filename,)
            ).fetchone()
            if row:
                return row[0]
        finally:
            pool.release_connection(conn)
    except Exception as e:
        logger.warning(f"Failed to lookup vault_id for {filename}: {e}")
    
    # Fallback to default vault
    return settings.orphan_vault_id


def rollback_migration() -> dict:
    """
    Rollback migration: move files from vault directories back to flat uploads.
    
    Returns dict with 'moved', 'skipped', 'failed' lists.
    """
    provider = UploadPathProvider()
    results = {"moved": [], "skipped": [], "failed": []}
    
    old_dir = settings.uploads_dir
    old_dir.mkdir(parents=True, exist_ok=True)
    
    # Iterate through all vault directories
    vaults_dir = settings.vaults_dir
    if not vaults_dir.exists():
        return results
    
    for vault_dir in vaults_dir.iterdir():
        if not vault_dir.is_dir():
            continue
        
        uploads_dir = vault_dir / "uploads"
        if not uploads_dir.exists():
            continue
        
        for f in uploads_dir.glob("*"):
            if not f.is_file():
                continue
            
            dest = old_dir / f.name
            
            # Handle name collision
            if dest.exists():
                # Rename with suffix
                stem = f.stem
                suffix = 1
                while dest.exists():
                    dest = old_dir / f"{stem}_{suffix}{f.suffix}"
                    suffix += 1
            
            try:
                try:
                    f.resolve().relative_to(old_dir.resolve())
                    dest.resolve().relative_to(old_dir.resolve())
                except ValueError:
                    logger.warning("Skipping rollback: path escapes old_dir — %s", f)
                    continue
                shutil.move(str(f), str(dest))
                results["moved"].append(str(f))
            except Exception as e:
                results["failed"].append(f"{f}: {e}")
    
    logger.info(f"Rollback complete: {len(results['moved'])} moved, {len(results['skipped'])} skipped, {len(results['failed'])} failed")
    return results


def _rename_vault_folder(old_name: str, new_name: str) -> bool:
    """
    Rename vault upload folder when vault is renamed.
    
    Returns True if renamed, False if old folder doesn't exist.
    """
    try:
        # Sanitize names the same way as get_upload_dir
        safe_old = "".join(c if c.isalnum() or c in "-_" else "_" for c in old_name)
        safe_new = "".join(c if c.isalnum() or c in "-_" else "_" for c in new_name)
        
        old_path = settings.vaults_dir / safe_old
        new_path = settings.vaults_dir / safe_new
        
        if not old_path.exists():
            return False
        
        # Handle case where new folder already exists
        if new_path.exists():
            # Move contents from old to new, rename conflicts with suffix
            for item in old_path.glob("*"):
                dest = new_path / item.name
                if dest.exists():
                    # Rename with suffix to avoid conflict
                    stem = item.stem
                    suffix = item.suffix
                    counter = 1
                    while dest.exists():
                        dest = new_path / f"{stem}_{counter}{suffix}"
                        counter += 1
                shutil.move(str(item), str(dest))
            old_path.rmdir()
        else:
            old_path.rename(new_path)
        
        return True
    except (OSError, PermissionError, shutil.Error) as e:
        logger.error(f"Failed to rename vault folder: {e}")
        return False


def migrate_vault_paths(sqlite_path: str, data_dir: Path) -> None:
    """
    One-time migration: rename vaults/{sanitized_name}/ → vaults/{id}/ directories.
    Idempotent — safe to run on every startup.
    Creates automatic backup before migration.

    Algorithm:
    1. Query vaults table for all (id, name) pairs
    2. old_path = data_dir/vaults/{sanitize(name)}/
    3. new_path = data_dir/vaults/{id}/
    4. If old_path exists and new_path does not: os.rename(old_path, new_path)
    5. If both exist: merge contents, log warning
    6. If only new_path exists: already migrated, skip
    
    Backup:
    - Creates backup directory at data_dir/vaults/.migration_backup/
    - Copies all vault directories before migration
    """
    import os

    from app.models.database import get_pool

    vaults_dir = data_dir / "vaults"
    backup_dir = data_dir / "vaults" / ".migration_backup"

    # Create backup before migration
    if vaults_dir.exists() and not backup_dir.exists():
        logger.info("Creating vault path migration backup...")
        backup_dir.mkdir(parents=True, exist_ok=True)
        for item in vaults_dir.iterdir():
            if item.is_dir() and item.name != ".migration_backup":
                backup_path = backup_dir / item.name
                shutil.copytree(item, backup_path, dirs_exist_ok=True)
        logger.info(f"Backup created at {backup_dir}")

    # Get vault mappings from database
    conn = None
    try:
        pool = get_pool(sqlite_path)
        conn = pool.get_connection()
        try:
            rows = conn.execute("SELECT id, name FROM vaults").fetchall()
        finally:
            pool.release_connection(conn)
            conn = None
    except Exception as e:
        logger.warning(f"Could not query vaults for migration: {e}")
        if conn is not None:
            try:
                pool.release_connection(conn)
            except Exception:
                pass
        return

    migrated_count = 0
    for vault_id, vault_name in rows:
        # Sanitize name the same way as old code
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in vault_name)
        old_path = vaults_dir / safe_name
        new_path = vaults_dir / str(vault_id)

        if not old_path.exists():
            # Already migrated or never existed
            continue

        if new_path.exists():
            # Both exist - merge contents
            logger.warning(f"Both paths exist for vault {vault_id} ({vault_name}). Merging contents.")
            for item in old_path.iterdir():
                dest = new_path / item.name
                if dest.exists():
                    logger.warning(f"Skipping {item} - already exists at destination")
                else:
                    shutil.move(str(item), str(dest))
            # Remove empty old directory
            try:
                old_path.rmdir()
            except OSError:
                pass
        else:
            # Rename old to new
            # Validate paths are within vaults_dir before renaming
            try:
                old_resolved = old_path.resolve()
                new_resolved = new_path.resolve()
                vaults_resolved = vaults_dir.resolve()
                old_resolved.relative_to(vaults_resolved)  # raises ValueError if outside
                new_resolved.relative_to(vaults_resolved)  # raises ValueError if outside
            except ValueError:
                logger.warning("Skipping rename: path escapes vaults_dir — old=%s new=%s", old_path, new_path)
                continue
            os.rename(old_path, new_path)
            logger.info(f"Migrated vault {vault_id}: {safe_name} → {vault_id}")
            migrated_count += 1

    if migrated_count > 0:
        logger.info(f"Vault path migration complete. Migrated {migrated_count} vaults.")

    # Verify data integrity after migration
    if migrated_count > 0:
        logger.info(f"Verifying migration integrity...")
        for vault_id, _ in rows:
            new_path = vaults_dir / str(vault_id)
            if new_path.exists():
                file_count = len(list(new_path.rglob('*')))
                logger.debug(f"Vault {vault_id}: {file_count} files")
