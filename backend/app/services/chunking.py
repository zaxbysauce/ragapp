"""
Semantic chunking service using unstructured.chunking.title.chunk_by_title.
Preserves tables and code blocks while creating semantically meaningful chunks.
"""

from dataclasses import dataclass, field
from typing import List, Any, Optional

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element


@dataclass
class ProcessedChunk:
    """
    Represents a processed chunk of document content.
    
    Attributes:
        text: The chunk text content
        metadata: Dictionary containing section title, element type, and other metadata
        chunk_index: Sequential index of this chunk in the document
        chunk_uid: Unique identifier for windowing (format: file_id_chunk_index)
        original_indices: List of original chunk indices if merged (for tracking)
    """
    text: str
    metadata: dict
    chunk_index: int
    chunk_uid: Optional[str] = None
    original_indices: List[int] = field(default_factory=list)


class SemanticChunker:
    """
    Semantic chunker using unstructured's title-based chunking.
    
    Uses character-based parameters directly for chunking.
    Preserves tables and code blocks by keeping element text intact.
    """
    
    def __init__(
        self,
        chunk_size_chars: int = 2000,
        chunk_overlap_chars: int = 200,
        max_merge_chars: int = 8192
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            chunk_size_chars: Target chunk size in characters
            chunk_overlap_chars: Overlap between chunks in characters
            max_merge_chars: Maximum characters when merging chunks (default 8192)
        """
        self.chunk_size = chunk_size_chars
        self.chunk_overlap = chunk_overlap_chars
        self.max_merge_chars = max_merge_chars
        
        # Use character counts directly
        self.max_characters = chunk_size_chars
        self.new_after_n_chars = chunk_size_chars
        self.overlap = chunk_overlap_chars
    
    def _get_element_text(self, element: Element) -> str:
        """
        Extract text from an element, preserving tables and code blocks.
        
        Args:
            element: Unstructured Element object
            
        Returns:
            String representation of the element's content
        """
        # Get the element text
        text = str(element)
        
        # For tables and code blocks, ensure we preserve the full content
        element_type = type(element).__name__
        if element_type in ('Table', 'CodeSnippet'):
            # These elements should be kept intact
            return text
        
        return text
    
    def _get_element_metadata(self, element: Element) -> dict:
        """
        Extract metadata from an unstructured element.
        
        Args:
            element: Unstructured Element object
            
        Returns:
            Dictionary containing element metadata
        """
        metadata = {
            'element_type': type(element).__name__,
            'category': getattr(element, 'category', None),
        }
        
        # Extract section title if available
        if hasattr(element, 'metadata') and element.metadata:
            element_meta = element.metadata
            if hasattr(element_meta, 'section') and element_meta.section:
                metadata['section_title'] = element_meta.section
            elif hasattr(element_meta, 'page_name') and element_meta.page_name:
                metadata['section_title'] = element_meta.page_name
            elif hasattr(element_meta, 'filename') and element_meta.filename:
                metadata['section_title'] = element_meta.filename
        
        # Try to get text as title if it's a Title element
        if metadata['element_type'] == 'Title' and hasattr(element, 'text'):
            metadata['section_title'] = element.text
        
        return metadata
    
    def _is_preserve_element(self, element: Element) -> bool:
        """
        Check if an element should be preserved intact (not split).
        
        Args:
            element: Unstructured Element object
            
        Returns:
            True if element should be preserved as-is
        """
        element_type = type(element).__name__
        return element_type in ('Table', 'CodeSnippet', 'TableChunk')
    
    def chunk_elements(self, elements: List[Element]) -> List[ProcessedChunk]:
        """
        Chunk document elements using title-based semantic chunking.
        
        Args:
            elements: List of unstructured document elements
            
        Returns:
            List of ProcessedChunk objects with text, metadata, and index
        """
        if not elements:
            return []
        
        # Use unstructured's chunk_by_title for semantic chunking
        # This respects document structure (titles, headers) when creating chunks
        chunks = chunk_by_title(
            elements,
            max_characters=self.max_characters,
            new_after_n_chars=self.new_after_n_chars,
            overlap=self.overlap,
            overlap_all=True
        )
        
        processed_chunks = []
        
        for idx, chunk in enumerate(chunks):
            # Get chunk text
            chunk_text = str(chunk)
            
            # Extract metadata from the chunk
            metadata = self._get_element_metadata(chunk)
            metadata['chunk_index'] = idx
            metadata['total_chunks'] = len(chunks)
            
            # Add original element info if available
            if hasattr(chunk, 'metadata') and chunk.metadata:
                orig_meta = chunk.metadata
                if hasattr(orig_meta, 'page_number') and orig_meta.page_number:
                    metadata['page_number'] = orig_meta.page_number
                if hasattr(orig_meta, 'filename') and orig_meta.filename:
                    metadata['source_file'] = orig_meta.filename
            
            processed_chunk = ProcessedChunk(
                text=chunk_text,
                metadata=metadata,
                chunk_index=idx
            )
            
            processed_chunks.append(processed_chunk)
        
        # Post-process chunks to merge those that split inside code blocks or tables
        processed_chunks = self._post_process_chunks(processed_chunks)
        
        return processed_chunks
    
    def _post_process_chunks(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """
        Merge chunks that split inside code blocks or tables.
        
        Args:
            chunks: List of ProcessedChunk objects from chunk_by_title
            
        Returns:
            List of ProcessedChunk objects with code blocks and tables preserved
        """
        merged = []
        pending = None
        
        for chunk in chunks:
            text = chunk.text
            
            if pending:
                # Try to merge pending with current
                merged_text = pending.text + "\n" + text
                if len(merged_text) <= self.max_merge_chars:
                    # Track original indices from both chunks
                    original_indices = list(pending.original_indices)
                    if chunk.original_indices:
                        original_indices.extend(chunk.original_indices)
                    else:
                        original_indices.append(chunk.chunk_index)
                    
                    pending = ProcessedChunk(
                        text=merged_text,
                        metadata={**pending.metadata, 'merged': True},
                        chunk_index=pending.chunk_index,
                        original_indices=original_indices
                    )
                    text = pending.text
                else:
                    merged.append(pending)
                    pending = None
            
            # Check if this chunk ends inside code block or table
            code_fence_count = text.count("```")
            lines = text.split("\n")
            last_line = lines[-1] if lines else ""
            
            in_code_block = code_fence_count % 2 == 1
            in_table = last_line.strip().startswith("|")
            
            if in_code_block or in_table:
                # Store original indices for tracking
                if pending is None:
                    if chunk.original_indices:
                        original_indices = chunk.original_indices
                    else:
                        original_indices = [chunk.chunk_index]
                    pending = ProcessedChunk(
                        text=text,
                        metadata=dict(chunk.metadata),
                        chunk_index=chunk.chunk_index,
                        original_indices=original_indices
                    )
                else:
                    # Continue building pending
                    pass
            else:
                if pending:
                    merged.append(pending)
                    pending = None
                else:
                    merged.append(chunk)
        
        if pending:
            merged.append(pending)
        
        # Re-index the merged chunks
        for idx, chunk in enumerate(merged):
            chunk.chunk_index = idx
        
        return merged
    
    def chunk_text(self, text: str, section_title: Optional[str] = None) -> List[ProcessedChunk]:
        """
        Chunk plain text content.
        
        Args:
            text: Text content to chunk
            section_title: Optional section title for metadata
            
        Returns:
            List of ProcessedChunk objects
        """
        from unstructured.partition.text import partition_text
        
        # Partition the text into elements
        elements = partition_text(text=text)
        
        # Add section title to metadata if provided
        if section_title:
            for element in elements:
                if hasattr(element, 'metadata') and element.metadata:
                    element.metadata.section = section_title
        
        return self.chunk_elements(elements)
