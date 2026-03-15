"""
Rollback script for vault path migration.

Restores vault directories from the migration backup.
Usage: python -m backend.scripts.rollback_vault_paths [--data-dir PATH] [--dry-run]
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import settings

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for the rollback script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def rollback_vault_paths(data_dir: Path, dry_run: bool = False) -> dict:
    """
    Rollback vault path migration by restoring from backup.

    Args:
        data_dir: Root data directory containing vaults/
        dry_run: If True, only log what would be done without making changes

    Returns:
        Dict with rollback statistics:
        - restored: list of vault IDs that were restored
        - skipped: list of vault IDs that were skipped (backup not found)
        - failed: list of vault IDs that failed to restore
    """
    vaults_dir = data_dir / "vaults"
    backup_dir = vaults_dir / ".migration_backup"

    result = {
        "restored": [],
        "skipped": [],
        "failed": [],
    }

    if not backup_dir.exists():
        logger.error(f"Backup directory not found: {backup_dir}")
        logger.error("Cannot rollback - no backup exists.")
        return result

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Starting rollback from {backup_dir}")

    # Iterate through backup directories
    for backup_vault_dir in backup_dir.iterdir():
        if not backup_vault_dir.is_dir():
            continue

        vault_name = backup_vault_dir.name
        current_vault_dir = vaults_dir / vault_name

        # Check if current vault directory exists (post-migration)
        if current_vault_dir.exists():
            if dry_run:
                logger.info(f"[DRY RUN] Would remove current: {current_vault_dir}")
            else:
                try:
                    shutil.rmtree(current_vault_dir)
                    logger.info(f"Removed current directory: {current_vault_dir}")
                except Exception as e:
                    logger.error(f"Failed to remove {current_vault_dir}: {e}")
                    result["failed"].append(vault_name)
                    continue

        # Restore from backup
        if dry_run:
            logger.info(f"[DRY RUN] Would restore: {backup_vault_dir} -> {current_vault_dir}")
            result["restored"].append(vault_name)
        else:
            try:
                shutil.copytree(backup_vault_dir, current_vault_dir)
                logger.info(f"Restored: {vault_name}")
                result["restored"].append(vault_name)
            except Exception as e:
                logger.error(f"Failed to restore {vault_name}: {e}")
                result["failed"].append(vault_name)

    # Summary
    logger.info("=" * 50)
    logger.info("Rollback Summary:")
    logger.info(f"  Restored: {len(result['restored'])}")
    logger.info(f"  Skipped: {len(result['skipped'])}")
    logger.info(f"  Failed: {len(result['failed'])}")

    if result["failed"]:
        logger.error(f"Failed vaults: {result['failed']}")

    return result


def cleanup_backup(data_dir: Path, dry_run: bool = False) -> bool:
    """
    Remove the migration backup directory after successful rollback.

    Args:
        data_dir: Root data directory containing vaults/
        dry_run: If True, only log what would be done

    Returns:
        True if backup was removed (or would be removed in dry_run), False otherwise
    """
    backup_dir = data_dir / "vaults" / ".migration_backup"

    if not backup_dir.exists():
        logger.info("No backup directory to clean up")
        return True

    if dry_run:
        logger.info(f"[DRY RUN] Would remove backup directory: {backup_dir}")
        return True

    try:
        shutil.rmtree(backup_dir)
        logger.info(f"Removed backup directory: {backup_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to remove backup directory: {e}")
        return False


def main() -> int:
    """Main entry point for the rollback script."""
    parser = argparse.ArgumentParser(
        description="Rollback vault path migration from backup"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory path (defaults to settings.data_dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove backup directory after successful rollback",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only remove backup directory, skip restore",
    )

    args = parser.parse_args()
    setup_logging()

    data_dir = args.data_dir or settings.data_dir

    if args.cleanup_only:
        success = cleanup_backup(data_dir, args.dry_run)
        return 0 if success else 1

    # Perform rollback
    result = rollback_vault_paths(data_dir, args.dry_run)

    # Cleanup if requested and rollback was successful
    if args.cleanup and not result["failed"]:
        cleanup_backup(data_dir, args.dry_run)

    # Return exit code based on success
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
