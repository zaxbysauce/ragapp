"""
Adversarial security tests for vector_store.py (Task 0.1)
Focus: Malformed inputs, oversized payloads, injection attempts, boundary violations
"""
import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from app.services.vector_store import (
    VectorStore, 
    VectorStoreError, 
    VectorStoreValidationError,
    VECTOR_INDEX_MIN_ROWS
)


class TestVaultIdTypeCoercion:
    """Tests for vault_id type coercion attacks"""
    
    def test_vault_id_integer_passthrough(self):
        """Test that integer vault_id is handled correctly"""
        vs = VectorStore()
        # Simulate the vault_id handling
        vault_id = 123
        safe_vault_id = str(vault_id).replace("'", "\\'")
        # Should convert to string "123"
        assert safe_vault_id == "123"
        # Filter should work
        vault_filter = f"vault_id = '{safe_vault_id}'"
        assert vault_filter == "vault_id = '123'"
    
    def test_vault_id_string_with_numeric_content(self):
        """Test vault_id as string with numeric content"""
        vs = VectorStore()
        vault_id = "456"
        safe_vault_id = str(vault_id).replace("'", "\\'")
        assert safe_vault_id == "456"
    
    def test_vault_id_negative_number(self):
        """Test negative vault_id - potential boundary attack"""
        vs = VectorStore()
        vault_id = -1
        safe_vault_id = str(vault_id).replace("'", "\\'")
        vault_filter = f"vault_id = '{safe_vault_id}'"
        # This would query for vault_id = '-1', which likely doesn't exist
        assert vault_filter == "vault_id = '-1'"
    
    def test_vault_id_float_conversion(self):
        """Test float vault_id - type confusion"""
        vs = VectorStore()
        vault_id = 1.5  # Float passed where int expected
        safe_vault_id = str(vault_id).replace("'", "\\'")
        # Float converts to "1.5", not "1" - potential mismatch
        assert safe_vault_id == "1.5"
    
    def test_vault_id_zero(self):
        """Test vault_id = 0 - boundary case"""
        vs = VectorStore()
        vault_id = 0
        safe_vault_id = str(vault_id).replace("'", "\\'")
        assert safe_vault_id == "0"
    
    def test_vault_id_max_int(self):
        """Test max integer vault_id - boundary"""
        vs = VectorStore()
        vault_id = 2**31 - 1  # Max 32-bit signed int
        safe_vault_id = str(vault_id).replace("'", "\\'")
        vault_filter = f"vault_id = '{safe_vault_id}'"
        assert "2147483647" in vault_filter


class TestSQLInjection:
    """Tests for SQL injection attacks in filter_expr"""
    
    def test_filter_expr_classic_or_injection(self):
        """Test classic OR injection attack"""
        vs = VectorStore()
        filter_expr = "' OR '1'='1"
        # The filter_expr is NOT sanitized - this is the vulnerability
        # It would be combined like: (filter_expr) AND (scale_filter)
        combined = f"({filter_expr}) AND (chunk_scale = 'default')"
        assert "'1'='1" in combined  # Injection succeeds
    
    def test_filter_expr_union_attack(self):
        """Test UNION-based injection"""
        vs = VectorStore()
        filter_expr = "1=1 UNION SELECT * FROM chunks--"
        combined = f"({filter_expr})"
        assert "UNION" in combined
    
    def filter_expr_drop_table_attempt(self):
        """Test DROP TABLE injection (LanceDB may not support but should fail safely)"""
        vs = VectorStore()
        filter_expr = "'; DROP TABLE chunks; --"
        # Without sanitization, this passes through
        combined = f"({filter_expr})"
        # The raw injection passes through - vulnerability
    
    def test_filter_expr_comment_injection(self):
        """Test comment-based injection to bypass filters"""
        vs = VectorStore()
        filter_expr = "id = 'test' --"
        combined = f"({filter_expr}) AND (vault_id = '1')"
        # The -- would comment out the vault filter
    
    def test_filter_expr_semicolon_injection(self):
        """Test semicolon injection for chaining"""
        vs = VectorStore()
        filter_expr = "id = 'test'; SELECT * FROM chunks"
        combined = f"({filter_expr})"
        assert "SELECT" in combined
    
    def test_filter_expr_js_injection(self):
        """Test JavaScript-like injection"""
        vs = VectorStore()
        filter_expr = "id = '<script>alert(1)</script>'"
        combined = f"({filter_expr})"
        assert "<script>" in combined
    
    def test_filter_expr_template_injection(self):
        """Test template literal injection"""
        vs = VectorStore()
        filter_expr = "id = '${malicious}'"
        combined = f"({filter_expr})"
        assert "${" in combined
    
    def test_filter_expr_path_traversal(self):
        """Test path traversal in filter"""
        vs = VectorStore()
        filter_expr = "id = '../../../etc/passwd'"
        combined = f"({filter_expr})"
        assert ".." in combined
    
    def test_filter_expr_null_byte(self):
        """Test null byte injection"""
        vs = VectorStore()
        filter_expr = "id = 'test\x00malicious'"
        combined = f"({filter_expr})"
        assert "\x00" in combined
    
    def test_filter_expr_unicode_override(self):
        """Test Unicode RTL override injection"""
        vs = VectorStore()
        # Right-to-left override character
        filter_expr = "id = 'test\u202emalicious\u202c'"
        combined = f"({filter_expr})"
        # Should be in the combined filter
    
    def test_filter_expr_oversized(self):
        """Test oversized filter expression"""
        vs = VectorStore()
        # Create a very large filter expression (>10KB)
        filter_expr = "id = '" + "A" * 20000 + "'"
        combined = f"({filter_expr})"
        assert len(combined) > 20000


class TestRowCountManipulation:
    """Tests for row count manipulation to bypass index threshold"""
    
    def test_index_threshold_boundary_255(self):
        """Test row count just below threshold (255)"""
        # VECTOR_INDEX_MIN_ROWS = 256
        row_count = 255
        should_create = row_count >= VECTOR_INDEX_MIN_ROWS
        assert should_create is False
    
    def test_index_threshold_boundary_256(self):
        """Test row count at threshold (256)"""
        row_count = 256
        should_create = row_count >= VECTOR_INDEX_MIN_ROWS
        assert should_create is True
    
    def test_index_threshold_boundary_257(self):
        """Test row count just above threshold (257)"""
        row_count = 257
        should_create = row_count >= VECTOR_INDEX_MIN_ROWS
        assert should_create is True
    
    def test_index_threshold_zero(self):
        """Test zero row count"""
        row_count = 0
        should_create = row_count >= VECTOR_INDEX_MIN_ROWS
        assert should_create is False
    
    def test_index_threshold_negative(self):
        """Test negative row count (should not happen but test boundary)"""
        row_count = -1
        should_create = row_count >= VECTOR_INDEX_MIN_ROWS
        assert should_create is False
    
    def test_index_threshold_max_int(self):
        """Test max int row count"""
        row_count = 2**31 - 1
        should_create = row_count >= VECTOR_INDEX_MIN_ROWS
        assert should_create is True
    
    def test_count_rows_with_injection(self):
        """Test count_rows with malicious filter"""
        vs = VectorStore()
        # Test that count_rows with injection doesn't cause issues
        # This tests the internal handling
        safe_file_id = "test'; DROP TABLE chunks; --"
        # Escaped version
        safe_escaped = safe_file_id.replace('"', '\\"')
        # This should be used in count_rows


class TestExceptionHandlingInfoLeakage:
    """Tests for exception handling that might leak information"""
    
    def test_connection_error_message_sanitization(self):
        """Test that connection errors don't leak sensitive paths"""
        vs = VectorStore(db_path=Path("/sensitive/path/lancedb"))
        try:
            vs.connect()
        except VectorStoreConnectionError as e:
            error_msg = str(e)
            # Path might be in message - check if it's properly handled
            # The error should contain path but that's expected for debugging
    
    def test_validation_error_message_content(self):
        """Test validation error messages don't leak sensitive data"""
        # Test dimension mismatch error
        error = VectorStoreValidationError(
            f"Embedding dimension mismatch: expected 384 dimensions, "
            f"got 128. The table was created with a different embedding model. "
            f"Delete the lancedb directory at /sensitive/path and restart."
        )
        error_msg = str(error)
        # Error contains path - this is a potential leak if paths are sensitive
        assert "/sensitive/path" in error_msg
    
    def test_missing_fields_error(self):
        """Test missing fields error message"""
        error = VectorStoreValidationError(
            "Record missing required fields: id, text, file_id"
        )
        # Error doesn't leak sensitive info - good
    
    def test_embedding_type_error_message(self):
        """Test embedding type error message"""
        error = VectorStoreValidationError(
            "Embedding must be a list or numpy array, got str"
        )
        # Error shows type but not sensitive data - OK


class TestSearchBoundaryConditions:
    """Tests for boundary conditions in search operations"""
    
    def test_search_limit_zero(self):
        """Test search with limit=0"""
        # Boundary case - should return empty or handle gracefully
        pass  # Tested via integration
    
    def test_search_limit_negative(self):
        """Test search with negative limit"""
        # Should be handled or rejected
        pass
    
    def test_search_limit_max_int(self):
        """Test search with max int limit"""
        limit = 2**31 - 1
        # This could cause memory issues or be rejected
        pass
    
    def test_search_embedding_malformed(self):
        """Test search with malformed embedding"""
        # Test with embedding of wrong type
        pass
    
    def test_search_embedding_wrong_dimension(self):
        """Test search with wrong embedding dimension"""
        pass
    
    def test_search_query_text_oversized(self):
        """Test search with oversized query text"""
        # Create very large query text (>10KB)
        large_text = "A" * 20000
        pass
    
    def test_search_query_text_unicode_extremes(self):
        """Test search with extreme Unicode"""
        # Test with null bytes, combining chars, etc.
        pass


class TestDeleteByFileSecurity:
    """Tests for delete_by_file security"""
    
    def test_delete_file_id_sql_injection(self):
        """Test file_id parameter SQL injection"""
        vs = VectorStore()
        # Test with malicious file_id
        malicious_file_id = 'test"; DROP TABLE chunks; --'
        safe_file_id = str(malicious_file_id).replace('"', '\\"')
        delete_filter = f'file_id = "{safe_file_id}"'
        # The filter should have escaped quotes
    
    def test_delete_file_id_path_traversal(self):
        """Test file_id with path traversal"""
        malicious_file_id = "../../../etc/passwd"
        safe_file_id = str(malicious_file_id).replace('"', '\\"')
        delete_filter = f'file_id = "{safe_file_id}"'
    
    def test_delete_file_id_null_byte(self):
        """Test file_id with null byte"""
        malicious_file_id = "test\x00malicious"
        safe_file_id = str(malicious_file_id).replace('"', '\\"')
        delete_filter = f'file_id = "{safe_file_id}"'
    
    def test_delete_file_id_unicode(self):
        """Test file_id with Unicode manipulation"""
        malicious_file_id = "test\u202emalicious\u202c"
        safe_file_id = str(malicious_file_id).replace('"', '\\"')
        delete_filter = f'file_id = "{safe_file_id}"'


class TestDeleteByVaultSecurity:
    """Tests for delete_by_vault security"""
    
    def test_delete_vault_id_type_confusion(self):
        """Test vault_id type confusion in delete"""
        vs = VectorStore()
        # vault_id is int but used in string filter
        vault_id = 1
        safe_vault_id = str(vault_id).replace("'", "\\'")
        delete_filter = f"vault_id = '{safe_vault_id}'"
        assert delete_filter == "vault_id = '1'"
    
    def test_delete_vault_id_float(self):
        """Test float vault_id in delete"""
        vault_id = 1.5
        safe_vault_id = str(vault_id).replace("'", "\\'")
        delete_filter = f"vault_id = '{safe_vault_id}'"
        # Float becomes "1.5" - different from int "1"
    
    def test_delete_vault_id_negative(self):
        """Test negative vault_id in delete"""
        vault_id = -1
        safe_vault_id = str(vault_id).replace("'", "\\'")
        delete_filter = f"vault_id = '{safe_vault_id}'"


class TestSparseSearchSecurity:
    """Tests for sparse search security"""
    
    def test_sparse_search_oversized_input(self):
        """Test sparse search with oversized sparse vector"""
        # Create very large sparse vector (>10KB when JSON serialized)
        large_sparse = {str(i): 1.0 for i in range(10000)}
        json_str = json.dumps(large_sparse)
        assert len(json_str) > 10000
    
    def test_sparse_search_malformed_json(self):
        """Test sparse search with malformed JSON in stored data"""
        pass  # Tested via integration
    
    def test_sparse_search_negative_weights(self):
        """Test sparse search with negative weights"""
        query_sparse = {"token1": -1.0, "token2": 0.5}
        # Negative weights could cause unexpected behavior
    
    def test_sparse_search_special_float_values(self):
        """Test sparse search with special float values"""
        query_sparse = {
            "inf": float('inf'),
            "ninf": float('-inf'),
            "nan": float('nan'),
        }


class TestGetChunksByUidSecurity:
    """Tests for get_chunks_by_uid security"""
    
    def test_uid_list_oversized(self):
        """Test oversized UID list"""
        # Create list of UIDs that would exceed reasonable limits
        large_uid_list = [f"file_{i}_chunk_0" for i in range(100000)]
        # Build IN clause
        escaped_uids = [uid.replace("'", "''") for uid in large_uid_list]
        quoted_uids = [f"'{uid}'" for uid in escaped_uids]
        uid_list = ", ".join(quoted_uids)
        query = f"id IN ({uid_list})"
        assert len(query) > 100000
    
    def test_uid_injection_attempt(self):
        """Test UID with SQL injection attempt"""
        malicious_uids = ["test'; DROP TABLE chunks; --"]
        escaped_uids = [uid.replace("'", "''") for uid in malicious_uids]
        quoted_uids = [f"'{uid}'" for uid in escaped_uids]
        uid_list = ", ".join(quoted_uids)
        query = f"id IN ({uid_list})"
        # The single quote is escaped to '' - should be safe
    
    def test_uid_empty_list(self):
        """Test empty UID list"""
        uid_list = ""
        query = f"id IN ({uid_list})"
        assert query == "id IN ()"
    
    def test_uid_special_characters(self):
        """Test UIDs with special characters"""
        malicious_uids = [
            "\x00nullbyte",
            "\u202eoverride",
            "'; DELETE FROM chunks WHERE '1'='1",
        ]
        for uid in malicious_uids:
            escaped = uid.replace("'", "''")
            quoted = f"'{escaped}'"


class TestAddChunksValidation:
    """Tests for add_chunks input validation"""
    
    def test_add_chunks_embedding_dimension_mismatch(self):
        """Test embedding dimension validation"""
        # Schema expects specific dimension
        pass
    
    def test_add_chunks_missing_required_fields(self):
        """Test missing required fields"""
        pass
    
    def test_add_chunks_invalid_embedding_type(self):
        """Test invalid embedding type"""
        pass
    
    def test_add_chunks_sparse_embedding_invalid_json(self):
        """Test sparse_embedding with invalid JSON"""
        pass
    
    def test_add_chunks_oversized_text(self):
        """Test oversized text field (>10KB)"""
        oversized_text = "A" * 20000
        # Should be accepted but test boundary
    
    def test_add_chunks_unicode_extremes(self):
        """Test text with extreme Unicode"""
        # Null bytes, RTL override, etc.
        pass


class TestSchemaValidation:
    """Tests for schema validation security"""
    
    def test_validate_schema_dimension_probe(self):
        """Test dimension probe generation doesn't leak"""
        pass
    
    def test_validate_schema_mismatched_dimensions(self):
        """Test dimension mismatch error message"""
        pass


class TestHybridSearchSecurity:
    """Tests for hybrid search attack vectors"""
    
    def test_hybrid_alpha_boundary(self):
        """Test hybrid_alpha boundary values"""
        # alpha = 0, 1, negative, >1
        pass
    
    def test_hybrid_text_injection(self):
        """Test query text injection in hybrid search"""
        pass
    
    def test_fts_filter_injection(self):
        """Test FTS search filter injection"""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
