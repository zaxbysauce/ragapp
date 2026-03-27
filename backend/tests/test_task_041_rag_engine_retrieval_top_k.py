"""Verification tests for Task 0.4.1 - Update rag_engine.py to use retrieval_top_k."""

import re
import pytest


class TestTask041RetrievalTopK:
    """Test that rag_engine.py uses retrieval_top_k instead of max_context_chunks."""

    @pytest.fixture
    def source_content(self):
        """Read the rag_engine.py source file."""
        with open("app/services/rag_engine.py", "r", encoding="utf-8") as f:
            return f.read()

    def test_line_341_uses_retrieval_top_k(self, source_content):
        """
        Verification Point 1: The memory search call uses self.retrieval_top_k
        (not settings.max_context_chunks).

        Originally checked line 341, but line numbers shift as the file evolves.
        This test now searches structurally for self.retrieval_top_k near the
        search_memories / memory_store.search invocation.
        """
        # Find any line that calls memory search AND uses retrieval_top_k nearby
        idx = source_content.find("self.retrieval_top_k")
        assert idx != -1, "self.retrieval_top_k not found anywhere in rag_engine.py"

        # The context around the main usage block must NOT reference max_context_chunks
        context = source_content[max(0, idx - 200): idx + 200]
        assert "max_context_chunks" not in context, (
            "Memory search context should NOT use settings.max_context_chunks near "
            "self.retrieval_top_k"
        )

        # Count the total usages of retrieval_top_k — there should be at least 2
        # (initialization and at least one usage site)
        count = source_content.count("self.retrieval_top_k")
        assert count >= 2, (
            f"Expected at least 2 uses of self.retrieval_top_k, found {count}"
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
