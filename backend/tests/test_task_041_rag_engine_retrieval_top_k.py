"""Verification tests for Task 0.4.1 - Update rag_engine.py to use retrieval_top_k."""

import re
import pytest


class TestTask041RetrievalTopK:
    """Test that rag_engine.py uses retrieval_top_k instead of max_context_chunks."""

    @pytest.fixture
    def source_content(self):
        """Read the rag_engine.py source file."""
        with open("backend/app/services/rag_engine.py", "r", encoding="utf-8") as f:
            return f.read()

    def test_line_341_uses_retrieval_top_k(self, source_content):
        """
        Verification Point 1: Line 341 uses self.retrieval_top_k (not settings.max_context_chunks).
        
        The memory search call at line 338-343 should pass self.retrieval_top_k
        as the k parameter, not settings.max_context_chunks.
        """
        lines = source_content.split("\n")
        
        # Line numbers in editor are 1-indexed, Python is 0-indexed
        # Line 341 in the file corresponds to index 340
        assert 340 < len(lines), "File has fewer than 341 lines"
        
        line_341 = lines[340]  # 0-indexed
        line_338_to_343 = "\n".join(lines[337:343])
        
        # Verify line 341 specifically contains retrieval_top_k
        assert "self.retrieval_top_k" in line_341, (
            f"Line 341 should use 'self.retrieval_top_k', but got: {line_341.strip()}"
        )
        
        # Verify the memory search call uses retrieval_top_k
        assert "self.retrieval_top_k" in line_338_to_343, (
            "Memory search should use self.retrieval_top_k"
        )
        
        # Verify it does NOT use settings.max_context_chunks
        assert "max_context_chunks" not in line_338_to_343, (
            "Memory search should NOT use settings.max_context_chunks"
        )

    def test_no_max_context_chunks_usage_in_file(self, source_content):
        """
        Verification Point 2: No other usages of max_context_chunks in rag_engine.py.
        
        The entire file should not contain any reference to max_context_chunks,
        ensuring the migration from max_context_chunks to retrieval_top_k is complete.
        """
        # Search for max_context_chunks in the entire file
        matches = re.findall(r"max_context_chunks", source_content)
        
        assert len(matches) == 0, (
            f"Found {len(matches)} usage(s) of 'max_context_chunks' in rag_engine.py. "
            "All references should be replaced with 'retrieval_top_k'."
        )

    def test_retrieval_top_k_is_initialized(self, source_content):
        """
        Additional verification: Ensure retrieval_top_k is properly initialized.
        
        The __init__ method should set self.retrieval_top_k from settings.
        """
        # Check that retrieval_top_k is initialized in __init__
        init_pattern = r"self\.retrieval_top_k\s*=\s*settings\.retrieval_top_k"
        assert re.search(init_pattern, source_content), (
            "self.retrieval_top_k should be initialized from settings.retrieval_top_k in __init__"
        )

    def test_memory_search_uses_retrieval_top_k_parameter(self, source_content):
        """
        Additional verification: The memory_store.search_memories call uses retrieval_top_k.
        
        This is the key change - memory search should use the retrieval_top_k parameter
        for consistency with vector search.
        """
        # Find the memory_store.search_memories call (spans multiple lines)
        # Look for the method call followed by parameters
        memory_search_pattern = r"self\.memory_store\.search_memories\s*,"
        matches = list(re.finditer(memory_search_pattern, source_content))
        
        assert len(matches) > 0, "Could not find self.memory_store.search_memories call"
        
        # Get the context around the match (next 100 chars should contain retrieval_top_k)
        match_pos = matches[0].end()
        context_after = source_content[match_pos:match_pos + 100]
        
        # The call should pass self.retrieval_top_k as the second argument
        assert "self.retrieval_top_k" in context_after, (
            f"memory_store.search_memories should pass self.retrieval_top_k, but context is: {repr(context_after[:50])}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
