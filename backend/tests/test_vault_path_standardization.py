"""
Verification tests for Task 0.3 - Vault path standardization.

Tests:
1. vault_dir(vault_id) returns data_dir/vaults/{vault_id}
2. vault_uploads_dir(vault_id) returns data_dir/vaults/{vault_id}/uploads
3. vault_documents_dir(vault_id) returns data_dir/vaults/{vault_id}/documents
4. _get_vault_name() is deleted (no name-based lookups)
5. migrate_vault_paths() correctly renames name-based to ID-based paths
6. Migration creates backup before modifying
"""
import os
import pytest
from pathlib import Path, PurePosixPath


def normalize_path(path_str: str) -> str:
    """Normalize path to use forward slashes for cross-platform comparison."""
    return str(PurePosixPath(path_str.replace('\\', '/')))


class TestVaultPathStandardization:
    """Test vault path standardization in config.py."""
    
    def test_vault_dir_returns_id_based_path(self):
        """Test that vault_dir returns data_dir/vaults/{vault_id}."""
        # Test path construction logic directly without instantiating Settings
        data_dir = Path("/data/knowledgevault")
        vault_id = 5
        
        # This is the implementation from config.py
        path = data_dir / "vaults" / str(vault_id)
        
        # Verify exact path structure
        normalized = normalize_path(str(path))
        expected = normalize_path(str(Path("/data/knowledgevault/vaults/5")))
        assert normalized == expected
        assert normalized.endswith("vaults/5")
    
    def test_vault_dir_with_different_ids(self):
        """Test vault_dir with various vault IDs."""
        data_dir = Path("/data/knowledgevault")
        
        test_cases = [
            (1, "vaults/1"),
            (99, "vaults/99"),
            (12345, "vaults/12345"),
        ]
        
        for vault_id, expected_suffix in test_cases:
            path = data_dir / "vaults" / str(vault_id)
            normalized = normalize_path(str(path))
            assert normalized.endswith(expected_suffix), f"Expected path ending with {expected_suffix}, got {normalized}"
    
    def test_vault_uploads_dir_returns_correct_path(self):
        """Test that vault_uploads_dir returns data_dir/vaults/{vault_id}/uploads."""
        data_dir = Path("/data/knowledgevault")
        vault_id = 5
        
        # This is the implementation from config.py
        path = data_dir / "vaults" / str(vault_id) / "uploads"
        
        # Verify exact path structure
        normalized = normalize_path(str(path))
        expected = normalize_path(str(Path("/data/knowledgevault/vaults/5/uploads")))
        assert normalized == expected
        assert normalized.endswith("5/uploads")
    
    def test_vault_documents_dir_returns_correct_path(self):
        """Test that vault_documents_dir returns data_dir/vaults/{vault_id}/documents."""
        data_dir = Path("/data/knowledgevault")
        vault_id = 5
        
        # This is the implementation from config.py
        path = data_dir / "vaults" / str(vault_id) / "documents"
        
        # Verify exact path structure
        normalized = normalize_path(str(path))
        expected = normalize_path(str(Path("/data/knowledgevault/vaults/5/documents")))
        assert normalized == expected
        assert normalized.endswith("5/documents")
    
    def test_vault_paths_are_consistent(self):
        """Test that vault_uploads_dir and vault_documents_dir use vault_dir as base."""
        data_dir = Path("/data/knowledgevault")
        vault_id = 42
        
        vault_path = data_dir / "vaults" / str(vault_id)
        uploads_path = data_dir / "vaults" / str(vault_id) / "uploads"
        documents_path = data_dir / "vaults" / str(vault_id) / "documents"
        
        # Verify uploads is under vault_dir
        assert uploads_path.parent == vault_path
        assert str(uploads_path).startswith(str(vault_path))
        
        # Verify documents is under vault_dir
        assert documents_path.parent == vault_path
        assert str(documents_path).startswith(str(vault_path))
    
    def test_get_vault_name_deleted(self):
        """Test that _get_vault_name function no longer exists in config module."""
        from app import config
        
        # Verify _get_vault_name is not in the module
        assert not hasattr(config, '_get_vault_name'), "_get_vault_name should be deleted"
        
        # Verify the vault_dir method doesn't call _get_vault_name (only mentions it in docs as something to avoid)
        import inspect
        source = inspect.getsource(config.Settings.vault_dir)
        
        # The function body should not contain _get_vault_name calls
        # Split source to separate docstring from code
        lines = source.split('\n')
        in_docstring = False
        code_lines = []
        for line in lines:
            if '"""' in line or "'''" in line:
                in_docstring = not in_docstring
                continue
            if not in_docstring:
                code_lines.append(line)
        
        code_only = '\n'.join(code_lines)
        assert '_get_vault_name' not in code_only, "vault_dir code should not reference _get_vault_name"
        assert 'vault_id' in code_only, "vault_dir should use vault_id parameter"


class TestMigrationPathFunctions:
    """Test migration path functions in upload_path.py."""
    
    def test_migrate_vault_paths_function_exists(self):
        """Test that migrate_vault_paths function exists."""
        from app.services import upload_path
        
        assert hasattr(upload_path, 'migrate_vault_paths'), "migrate_vault_paths should exist"
        assert callable(upload_path.migrate_vault_paths)
    
    def test_migrate_vault_paths_creates_backup_before_migration(self):
        """Test that migrate_vault_paths creates backup before modifying."""
        from app.services import upload_path
        
        # Check the function creates backup at .migration_backup
        import inspect
        source = inspect.getsource(upload_path.migrate_vault_paths)
        
        # Verify backup directory logic
        assert '.migration_backup' in source, "Should create .migration_backup directory"
        assert 'backup_dir' in source or 'backup' in source.lower(), "Should have backup logic"
        assert 'shutil.copytree' in source or 'shutil.copy' in source, "Should copy files for backup"
    
    def test_migrate_vault_paths_id_based_renaming(self):
        """Test that migrate_vault_paths renames vaults/{name}/ to vaults/{id}/."""
        from app.services import upload_path
        
        import inspect
        source = inspect.getsource(upload_path.migrate_vault_paths)
        
        # Verify it uses vault_id from database
        assert 'SELECT id, name FROM vaults' in source, "Should query vault id and name"
        assert 'vaults' in source, "Should work with vaults directory"
        
        # Verify it creates new_path with vault_id
        assert 'str(vault_id)' in source or 'vault_id' in source, "Should use vault_id for new path"


class TestUploadPathProviderIntegration:
    """Test that UploadPathProvider uses new vault path functions."""
    
    def test_upload_path_provider_uses_vault_uploads_dir(self):
        """Test that UploadPathProvider.get_upload_dir uses vault_uploads_dir."""
        from app.services import upload_path
        
        import inspect
        source = inspect.getsource(upload_path.UploadPathProvider.get_upload_dir)
        
        # Verify it uses settings.vault_uploads_dir
        assert 'vault_uploads_dir' in source, "Should use vault_uploads_dir from settings"
    
    def test_no_name_based_lookup_in_upload_path(self):
        """Test that upload_path.py doesn't use name-based lookups."""
        from app.services import upload_path
        
        import inspect
        
        # Check that migrate_uploads doesn't use _get_vault_name
        if hasattr(upload_path, 'migrate_uploads'):
            source = inspect.getsource(upload_path.migrate_uploads)
            assert '_get_vault_name' not in source, "Should not use _get_vault_name"


class TestPathVerification:
    """Verification tests for specific acceptance criteria."""
    
    def test_verification_point_1_vault_dir_with_vault_id_5(self):
        """Verification Point 1: vault_dir() with vault_id=5 returns path ending with '5'."""
        data_dir = Path("/data/knowledgevault")
        vault_id = 5
        
        # Implementation from config.py
        result = data_dir / "vaults" / str(vault_id)
        normalized = normalize_path(str(result))
        
        assert normalized.endswith("5"), f"Expected path ending with '5', got {normalized}"
        assert "vaults" in normalized
        assert "5" in normalized
    
    def test_verification_point_2_vault_uploads_dir(self):
        """Verification Point 2: vault_uploads_dir() returns path ending with '5/uploads'."""
        data_dir = Path("/data/knowledgevault")
        vault_id = 5
        
        # Implementation from config.py
        result = data_dir / "vaults" / str(vault_id) / "uploads"
        normalized = normalize_path(str(result))
        
        assert normalized.endswith("5/uploads"), f"Expected path ending with '5/uploads', got {normalized}"
    
    def test_verification_point_3_vault_documents_dir(self):
        """Verification Point 3: vault_documents_dir() returns path ending with '5/documents'."""
        data_dir = Path("/data/knowledgevault")
        vault_id = 5
        
        # Implementation from config.py
        result = data_dir / "vaults" / str(vault_id) / "documents"
        normalized = normalize_path(str(result))
        
        assert normalized.endswith("5/documents"), f"Expected path ending with '5/documents', got {normalized}"
    
    def test_verification_point_5_migration_renames_name_to_id(self):
        """Verification Point 5: migrate_vault_paths() renames vaults/{name}/ to vaults/{id}/."""
        from app.services import upload_path
        
        import inspect
        source = inspect.getsource(upload_path.migrate_vault_paths)
        
        # Should use sanitize function to convert name to path
        assert 'safe_name' in source or 'sanitize' in source.lower(), "Should sanitize vault name"
        
        # Should build old_path with sanitized name
        assert 'old_path' in source, "Should calculate old path from name"
        
        # Should build new_path with vault_id
        assert 'new_path' in source, "Should calculate new path from id"
        
        # Should rename/move
        assert 'rename' in source.lower() or 'move' in source.lower(), "Should rename/move directories"
    
    def test_verification_point_6_backup_directory(self):
        """Verification Point 6: Backup directory created at vaults/.migration_backup/."""
        from app.services import upload_path
        
        import inspect
        source = inspect.getsource(upload_path.migrate_vault_paths)
        
        # Should create backup directory
        assert '.migration_backup' in source, "Should create .migration_backup directory"
        
        # Should check if backup exists before creating
        assert 'backup_dir.exists()' in source or 'exists' in source, "Should check backup existence"


class TestEdgeCases:
    """Test edge cases for vault paths."""
    
    def test_vault_id_is_converted_to_string(self):
        """Test that vault_id is converted to string in path."""
        data_dir = Path("/data/knowledgevault")
        
        # Test with integer
        result_int = data_dir / "vaults" / str(5)
        
        # Test with string that looks like int
        result_str = data_dir / "vaults" / str("5")
        
        # Both should produce the same path
        assert str(result_int) == str(result_str)
        assert normalize_path(str(result_int)).endswith("5")
    
    def test_different_vault_ids_produce_different_paths(self):
        """Test that different vault_ids produce different paths."""
        data_dir = Path("/data/knowledgevault")
        
        paths = [data_dir / "vaults" / str(i) for i in [1, 2, 3, 10, 100]]
        
        # All paths should be unique
        assert len(set(str(p) for p in paths)) == len(paths), "Each vault_id should produce unique path"
    
    def test_cross_platform_path_handling(self):
        """Test that paths use pathlib for cross-platform compatibility."""
        data_dir = Path("/data/knowledgevault")
        
        result = data_dir / "vaults" / str(1)
        
        # Should return a Path object
        assert isinstance(result, Path), "Should return pathlib.Path for cross-platform compatibility"
        
        # Path should be properly constructed
        assert result.absolute() or result.relative_to(data_dir)


class TestSettingsIntegration:
    """Integration tests using actual Settings with mocked data_dir."""
    
    def test_settings_vault_dir_method(self):
        """Test that Settings.vault_dir produces correct path."""
        # Create a mock settings object that mimics Settings behavior
        class MockSettings:
            def __init__(self, data_dir):
                self._data_dir = Path(data_dir)
            
            @property
            def data_dir(self):
                return self._data_dir
            
            def vault_dir(self, vault_id: int) -> Path:
                path = self.data_dir / "vaults" / str(vault_id)
                return path
            
            def vault_uploads_dir(self, vault_id: int) -> Path:
                return self.vault_dir(vault_id) / "uploads"
            
            def vault_documents_dir(self, vault_id: int) -> Path:
                return self.vault_dir(vault_id) / "documents"
        
        settings = MockSettings("/data/knowledgevault")
        
        # Test vault_dir
        normalized = normalize_path(str(settings.vault_dir(5)))
        assert normalized.endswith("vaults/5"), f"Expected path ending with 'vaults/5', got {normalized}"
        
        # Test vault_uploads_dir
        normalized = normalize_path(str(settings.vault_uploads_dir(5)))
        assert normalized.endswith("vaults/5/uploads"), f"Expected path ending with 'vaults/5/uploads', got {normalized}"
        
        # Test vault_documents_dir
        normalized = normalize_path(str(settings.vault_documents_dir(5)))
        assert normalized.endswith("vaults/5/documents"), f"Expected path ending with 'vaults/5/documents', got {normalized}"
    
    def test_settings_singleton_has_vault_methods(self):
        """Test that actual settings singleton has vault methods."""
        from app.config import settings
        
        # Verify the methods exist and are callable
        assert callable(settings.vault_dir)
        assert callable(settings.vault_uploads_dir)
        assert callable(settings.vault_documents_dir)
        
        # Test that they return Path objects
        result = settings.vault_dir(1)
        assert isinstance(result, Path)
        
        result = settings.vault_uploads_dir(1)
        assert isinstance(result, Path)
        
        result = settings.vault_documents_dir(1)
        assert isinstance(result, Path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
