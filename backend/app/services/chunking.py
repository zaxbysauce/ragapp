"""
Semantic chunking service using unstructured.chunking.title.chunk_by_title.
Preserves tables and code blocks while creating semantically meaningful chunks.
"""

from dataclasses import dataclass
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
    """
    text: str
    metadata: dict
    chunk_index: int


class SemanticChunker:
    """
    Semantic chunker using unstructured's title-based chunking.
    
    Converts token-based parameters to character-based limits (approx 4x ratio).
    Preserves tables and code blocks by keeping element text intact.
    """
    
    # Approximate ratio: 1 token ≈ 4 characters
    TOKEN_TO_CHAR_RATIO = 4
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            chunk_size: Target chunk size in tokens (converted to characters)
            chunk_overlap: Overlap between chunks in tokens (converted to characters)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Convert token counts to character counts
        self.max_characters = chunk_size * self.TOKEN_TO_CHAR_RATIO
        self.new_after_n_chars = self.max_characters
        self.overlap = chunk_overlap * self.TOKEN_TO_CHAR_RATIO
    
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
        
        return processed_chunks
    
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
