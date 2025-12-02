#!/usr/bin/env python3
"""
chunking.py: Advanced text chunking for context storage

This module implements improved chunking strategies for better retrieval:
- Target chunk size: 600-800 tokens with ~80 token overlap
- Metadata injection (doc title, section headers)
- Boilerplate removal (TOC, navigation, etc.)
- Semantic boundary detection
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available - using basic word-based tokenization")


class AdvancedChunker:
    """Advanced text chunker with configurable parameters."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize chunker with configuration.
        
        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        
        # Chunking parameters
        self.target_tokens = self.config.get("target_tokens", 700)  # Middle of 600-800 range
        self.overlap_tokens = self.config.get("overlap_tokens", 80)
        self.min_chunk_tokens = self.config.get("min_chunk_tokens", 100)
        self.max_chunk_tokens = self.config.get("max_chunk_tokens", 1000)
        
        # Content processing options
        self.inject_metadata = self.config.get("inject_metadata", True)
        self.strip_boilerplate = self.config.get("strip_boilerplate", True)
        self.preserve_structure = self.config.get("preserve_structure", True)
        
        # Tokenizer setup
        self.encoding_name = self.config.get("encoding", "cl100k_base")  # GPT-4 encoding
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding(self.encoding_name)
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding {self.encoding_name}: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
        
        # Boilerplate patterns to remove
        self.boilerplate_patterns = [
            r'^(Table of Contents|Contents)$',
            r'^(Navigation|Nav)$', 
            r'^(Menu|Site Menu)$',
            r'^(Header|Footer)$',
            r'^(Sidebar|Side Navigation)$',
            r'^(\d+\.\s*)+$',  # Numbered list items with no content
            r'^\s*(Home|Back|Next|Previous|Skip to|Jump to)\s*$',
            r'^\s*©.*$',  # Copyright notices
            r'^\s*(Privacy Policy|Terms of Service|Legal)\s*$'
        ]
        
        # Section header patterns
        self.header_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headers
            r'^([A-Z][^.!?]*):?\s*$',  # Title case lines
            r'^\d+(\.\d+)*\.?\s+(.+)$',  # Numbered sections
        ]
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: approximate token count using word count * 1.3
            words = len(text.split())
            return int(words * 1.3)
    
    def clean_boilerplate(self, text: str) -> str:
        """Remove boilerplate content from text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not self.strip_boilerplate:
            return text
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check against boilerplate patterns
            is_boilerplate = False
            for pattern in self.boilerplate_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_boilerplate = True
                    break
            
            if not is_boilerplate:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_headers(self, text: str) -> List[Tuple[str, int]]:
        """Extract section headers from text.
        
        Args:
            text: Input text
            
        Returns:
            List of (header_text, line_number) tuples
        """
        headers = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            for pattern in self.header_patterns:
                match = re.match(pattern, line)
                if match:
                    # Extract the header text (first capturing group)
                    if match.groups():
                        header_text = match.group(1).strip()
                    else:
                        header_text = line.strip()
                    headers.append((header_text, i))
                    break
        
        return headers
    
    def create_chunk_metadata(
        self, 
        chunk_text: str,
        chunk_index: int, 
        total_chunks: int,
        source_metadata: Optional[Dict[str, Any]] = None,
        current_headers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create metadata for a chunk.
        
        Args:
            chunk_text: The chunk text
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks
            source_metadata: Original document metadata
            current_headers: Active section headers for this chunk
            
        Returns:
            Chunk metadata dictionary
        """
        metadata = {
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "token_count": self.count_tokens(chunk_text),
            "char_count": len(chunk_text)
        }
        
        # Include source metadata
        if source_metadata:
            metadata.update({
                "source_title": source_metadata.get("title", ""),
                "source_type": source_metadata.get("type", "document"),
                "source_id": source_metadata.get("id", "")
            })
        
        # Include section context
        if current_headers:
            metadata["section_headers"] = current_headers
            metadata["section_context"] = " > ".join(current_headers)
        
        return metadata
    
    def inject_chunk_metadata(
        self, 
        chunk_text: str, 
        metadata: Dict[str, Any]
    ) -> str:
        """Inject metadata into chunk text for better embedding.
        
        Args:
            chunk_text: Original chunk text
            metadata: Chunk metadata
            
        Returns:
            Enhanced chunk text with metadata
        """
        if not self.inject_metadata:
            return chunk_text
        
        prefixes = []
        
        # Add document title
        if metadata.get("source_title"):
            prefixes.append(f"Document: {metadata['source_title']}")
        
        # Add section context
        if metadata.get("section_context"):
            prefixes.append(f"Section: {metadata['section_context']}")
        
        # Add chunk position context
        if metadata.get("chunk_index") is not None and metadata.get("total_chunks"):
            prefixes.append(f"Part {metadata['chunk_index'] + 1} of {metadata['total_chunks']}")
        
        if prefixes:
            prefix_text = " | ".join(prefixes)
            return f"[{prefix_text}]\n\n{chunk_text}"
        
        return chunk_text
    
    def chunk_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Chunk text into optimally sized pieces.
        
        Args:
            text: Input text to chunk
            metadata: Optional source document metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        # Clean boilerplate content
        cleaned_text = self.clean_boilerplate(text)
        if not cleaned_text.strip():
            return []
        
        # Extract headers for context
        headers = self.extract_headers(cleaned_text)
        
        # Split into sentences/paragraphs for semantic boundaries
        paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in cleaned_text.split('\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        current_headers = []
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            # Update current headers based on this paragraph
            for header_text, _ in headers:
                if header_text in para:
                    # This paragraph contains a header - update context
                    if header_text not in current_headers:
                        current_headers.append(header_text)
                    # Keep only the last few headers to avoid bloat
                    current_headers = current_headers[-3:]
            
            # Check if adding this paragraph would exceed target
            if current_tokens + para_tokens > self.target_tokens and current_chunk:
                # Finalize current chunk
                chunk_metadata = self.create_chunk_metadata(
                    current_chunk,
                    len(chunks),
                    0,  # Will update total_chunks later
                    metadata,
                    current_headers.copy()
                )
                
                enhanced_text = self.inject_chunk_metadata(current_chunk, chunk_metadata)
                
                chunks.append({
                    "text": enhanced_text,
                    "original_text": current_chunk,
                    "metadata": chunk_metadata
                })
                
                # Start new chunk with overlap
                if self.overlap_tokens > 0:
                    # Try to keep some content for overlap
                    overlap_text = self._get_overlap_text(current_chunk, self.overlap_tokens)
                    current_chunk = overlap_text + "\n\n" + para if overlap_text else para
                    current_tokens = self.count_tokens(current_chunk)
                else:
                    current_chunk = para
                    current_tokens = para_tokens
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                    current_tokens += para_tokens + 2  # Account for newlines
                else:
                    current_chunk = para
                    current_tokens = para_tokens
        
        # Handle final chunk
        if current_chunk and current_tokens >= self.min_chunk_tokens:
            chunk_metadata = self.create_chunk_metadata(
                current_chunk,
                len(chunks), 
                0,  # Will update below
                metadata,
                current_headers
            )
            
            enhanced_text = self.inject_chunk_metadata(current_chunk, chunk_metadata)
            
            chunks.append({
                "text": enhanced_text,
                "original_text": current_chunk,
                "metadata": chunk_metadata
            })
        
        # Update total_chunks in all chunk metadata
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)
        
        logger.info(
            f"Chunked text into {len(chunks)} chunks. "
            f"Avg tokens: {sum(self.count_tokens(c['original_text']) for c in chunks) / len(chunks):.0f}"
        )
        
        return chunks
    
    def _get_overlap_text(self, text: str, max_tokens: int) -> str:
        """Get trailing text for chunk overlap.
        
        Args:
            text: Source text
            max_tokens: Maximum tokens for overlap
            
        Returns:
            Overlap text
        """
        if not text:
            return ""
        
        # Split into sentences and take from the end
        sentences = re.split(r'[.!?]+\s+', text)
        overlap_text = ""
        tokens = 0
        
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_tokens = self.count_tokens(sentence)
            if tokens + sentence_tokens <= max_tokens:
                overlap_text = sentence + ". " + overlap_text
                tokens += sentence_tokens
            else:
                break
        
        return overlap_text.strip()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chunker configuration and statistics."""
        return {
            "target_tokens": self.target_tokens,
            "overlap_tokens": self.overlap_tokens,
            "min_chunk_tokens": self.min_chunk_tokens,
            "max_chunk_tokens": self.max_chunk_tokens,
            "inject_metadata": self.inject_metadata,
            "strip_boilerplate": self.strip_boilerplate,
            "tiktoken_available": TIKTOKEN_AVAILABLE,
            "encoding_name": self.encoding_name
        }


# Global chunker instance
_global_chunker: Optional[AdvancedChunker] = None


def get_chunker(config: Optional[Dict[str, Any]] = None) -> AdvancedChunker:
    """Get or create global chunker instance.
    
    Args:
        config: Optional configuration dict
        
    Returns:
        AdvancedChunker instance
    """
    global _global_chunker
    
    if _global_chunker is None:
        _global_chunker = AdvancedChunker(config)
    
    return _global_chunker


def chunk_text(
    text: str, 
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Convenience function to chunk text.
    
    Args:
        text: Text to chunk
        metadata: Optional source metadata
        config: Optional chunker configuration
        
    Returns:
        List of chunk dictionaries
    """
    chunker = get_chunker(config)
    return chunker.chunk_text(text, metadata)