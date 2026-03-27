"""
Semantic chunking service using unstructured.chunking.title.chunk_by_title.
Preserves tables and code blocks while creating semantically meaningful chunks.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Any, Optional, Callable

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element

logger = logging.getLogger(__name__)


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


class ThresholdType(Enum):
    """Types of threshold strategies for embedding semantic chunking."""
    PERCENTILE = "percentile"
    STDDEV = "stddev"
    GRADIENT = "gradient"


class EmbeddingSemanticChunker:
    """
    Semantic chunker using embedding-based cosine similarity breakpoints.
    
    Divides text into chunks at semantic boundaries determined by embedding
    similarity drops. Falls back to title-based chunking on embedding errors.
    
    Uses configurable threshold types: percentile, stddev, or gradient-based.
    """
    
    def __init__(
        self,
        embedding_service: Any,
        threshold_type: ThresholdType = ThresholdType.PERCENTILE,
        threshold_value: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        window_size: int = 1,
    ):
        """
        Initialize the embedding semantic chunker.
        
        Args:
            embedding_service: Service with embed_single(text) -> List[float] method
            threshold_type: Strategy for determining breakpoints (percentile/stddev/gradient)
            threshold_value: Threshold value (percentile 0-100, stddev multiplier, or gradient threshold)
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
            window_size: Number of sentences to combine for embedding context
        """
        self.embedding_service = embedding_service
        self.threshold_type = threshold_type
        self.threshold_value = threshold_value
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.window_size = window_size
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristic."""
        import re
        # Simple sentence splitting on period, exclamation, question mark
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def _calculate_breakpoints(self, similarities: List[float]) -> List[int]:
        """
        Calculate chunk breakpoints based on similarity drops.
        
        Args:
            similarities: List of cosine similarities between adjacent sentence windows
            
        Returns:
            List of indices where chunks should break
        """
        if not similarities:
            return []
        
        breakpoints = []
        
        if self.threshold_type == ThresholdType.PERCENTILE:
            # Break where similarity drops below the Nth percentile
            sorted_sims = sorted(similarities)
            percentile_idx = int(len(sorted_sims) * self.threshold_value / 100)
            threshold = sorted_sims[max(0, min(percentile_idx, len(sorted_sims) - 1))]
            
            for i, sim in enumerate(similarities):
                if sim < threshold:
                    breakpoints.append(i + 1)  # Break after sentence i
                    
        elif self.threshold_type == ThresholdType.STDDEV:
            # Break where similarity is more than N stddevs below mean
            mean = sum(similarities) / len(similarities)
            variance = sum((s - mean) ** 2 for s in similarities) / len(similarities)
            stddev = math.sqrt(variance)
            threshold = mean - (self.threshold_value * stddev)
            
            for i, sim in enumerate(similarities):
                if sim < threshold:
                    breakpoints.append(i + 1)
                    
        elif self.threshold_type == ThresholdType.GRADIENT:
            # Break where similarity gradient (drop) exceeds threshold
            for i in range(1, len(similarities)):
                gradient = similarities[i - 1] - similarities[i]
                if gradient > self.threshold_value:
                    breakpoints.append(i + 1)
        
        return breakpoints
    
    async def chunk_text(self, text: str, section_title: Optional[str] = None) -> List[ProcessedChunk]:
        """
        Chunk text using embedding-based semantic breakpoints.
        
        Args:
            text: Text content to chunk
            section_title: Optional section title for metadata
            
        Returns:
            List of ProcessedChunk objects
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        # Handle single sentence
        if len(sentences) == 1:
            metadata = {"section_title": section_title, "element_type": "Text"}
            return [ProcessedChunk(text=text, metadata=metadata, chunk_index=0)]
        
        try:
            # Create windows for embedding (sliding window of sentences)
            windows = []
            for i in range(len(sentences)):
                end_idx = min(i + self.window_size, len(sentences))
                window_text = " ".join(sentences[i:end_idx])
                windows.append(window_text)
            
            # Get embeddings for all windows
            embeddings = []
            for window in windows:
                try:
                    embedding = await self.embedding_service.embed_single(window)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning("Failed to embed window, using zero vector: %s", e)
                    # Use zero vector as placeholder - will create low similarity
                    if embeddings:
                        embeddings.append([0.0] * len(embeddings[0]))
                    else:
                        # Cannot determine embedding dimension, fall back
                        raise
            
            # Calculate similarities between adjacent windows
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                similarities.append(sim)
            
            # Determine breakpoints
            breakpoints = self._calculate_breakpoints(similarities)
            
            # Build chunks from breakpoints
            chunks = []
            start_idx = 0
            
            # Always add final boundary
            all_breakpoints = sorted(set(breakpoints + [len(sentences)]))
            
            for bp in all_breakpoints:
                chunk_sentences = sentences[start_idx:bp]
                chunk_text = " ".join(chunk_sentences)
                
                # Merge small chunks with previous if possible
                if chunks and len(chunk_text) < self.min_chunk_size:
                    # Extend previous chunk
                    prev_chunk = chunks[-1]
                    merged_text = prev_chunk.text + " " + chunk_text
                    if len(merged_text) <= self.max_chunk_size:
                        prev_chunk.text = merged_text
                        start_idx = bp
                        continue
                
                # Split large chunks
                while len(chunk_text) > self.max_chunk_size:
                    # Find a good split point (prefer sentence boundary)
                    split_at = self.max_chunk_size
                    for i in range(self.max_chunk_size - 1, self.min_chunk_size, -1):
                        if chunk_text[i] in '.!?' and chunk_text[i + 1] == ' ':
                            split_at = i + 1
                            break
                    
                    first_part = chunk_text[:split_at].strip()
                    metadata = {
                        "section_title": section_title,
                        "element_type": "Text",
                    }
                    chunks.append(ProcessedChunk(
                        text=first_part,
                        metadata=metadata,
                        chunk_index=len(chunks)
                    ))
                    chunk_text = chunk_text[split_at:].strip()
                
                if chunk_text:
                    metadata = {
                        "section_title": section_title,
                        "element_type": "Text",
                    }
                    chunks.append(ProcessedChunk(
                        text=chunk_text,
                        metadata=metadata,
                        chunk_index=len(chunks)
                    ))
                
                start_idx = bp
            
            # Update total_chunks in metadata
            for chunk in chunks:
                chunk.metadata["total_chunks"] = len(chunks)
            
            return chunks
            
        except Exception as e:
            logger.warning("Embedding-based chunking failed: %s. Falling back to title-based chunking.", e)
            return await self._fallback_chunk_text(text, section_title)
    
    async def _fallback_chunk_text(self, text: str, section_title: Optional[str] = None) -> List[ProcessedChunk]:
        """
        Fallback to title-based chunking when embedding chunking fails.

        Tries SemanticChunker first; if it returns nothing (e.g. unstructured not
        available in test environments), falls back to a simple paragraph/size-based
        splitter so that the caller always receives at least one chunk.

        Args:
            text: Text content to chunk
            section_title: Optional section title for metadata

        Returns:
            List of ProcessedChunk objects
        """
        # Try the unstructured-based chunker first
        fallback_chunker = SemanticChunker(
            chunk_size_chars=self.max_chunk_size,
            chunk_overlap_chars=max(100, self.max_chunk_size // 10),
        )
        chunks = fallback_chunker.chunk_text(text, section_title)

        if chunks:
            return chunks

        # Final fallback: split by max_chunk_size characters so the caller always
        # receives at least one chunk even when unstructured is unavailable.
        if not text or not text.strip():
            return []

        result: List[ProcessedChunk] = []
        step = max(self.max_chunk_size, 100)
        for start in range(0, len(text), step):
            chunk_text = text[start: start + step].strip()
            if not chunk_text:
                continue
            result.append(ProcessedChunk(
                text=chunk_text,
                metadata={
                    "section_title": section_title or "",
                    "element_type": "Text",
                    "total_chunks": 0,  # Updated below
                },
                chunk_index=len(result),
            ))

        for chunk in result:
            chunk.metadata["total_chunks"] = len(result)

        return result
    
    async def chunk_elements(self, elements: List[Element]) -> List[ProcessedChunk]:
        """
        Chunk document elements using embedding-based semantic chunking.
        
        Falls back to title-based chunking if embedding service is unavailable.
        
        Args:
            elements: List of unstructured document elements
            
        Returns:
            List of ProcessedChunk objects
        """
        if not elements:
            return []
        
        try:
            # Try to use embedding-based chunking on combined text
            combined_text = "\n\n".join(str(el) for el in elements)
            return await self.chunk_text(combined_text)
        except Exception as e:
            logger.warning("Embedding chunking failed for elements: %s. Using title-based fallback.", e)
            fallback_chunker = SemanticChunker(
                chunk_size_chars=self.max_chunk_size,
                chunk_overlap_chars=max(100, self.max_chunk_size // 10),
            )
            return fallback_chunker.chunk_elements(elements)
