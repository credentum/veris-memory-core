#!/usr/bin/env python3
"""
BM25-based text search backend for full-text search capabilities.

This backend provides BM25 (Best Matching 25) text search functionality
to complement vector and graph backends with traditional information retrieval.
"""

import asyncio
import logging
import math
import time
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from ..interfaces.backend_interface import (
    BackendSearchInterface, 
    SearchOptions, 
    BackendHealthStatus, 
    BackendSearchError
)
from ..interfaces.memory_result import MemoryResult
from ..utils.logging_middleware import search_logger


@dataclass
class DocumentIndex:
    """Represents an indexed document for BM25 search."""
    doc_id: str
    text: str
    tokens: List[str]
    token_frequencies: Dict[str, int]
    doc_length: int
    metadata: Dict[str, Any]


class BM25Scorer:
    """
    BM25 scoring implementation for ranking documents by relevance.
    
    BM25 is a probabilistic ranking function used by search engines to estimate
    the relevance of documents to a given search query. It's based on the
    bag-of-words retrieval function used by search engines to rank matching
    documents according to their relevance to a query.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 scorer with tuning parameters.
        
        Args:
            k1: Controls term frequency scaling (typically 1.2-2.0)
            b: Controls field length normalization (0-1, 0.75 is common)
        """
        self.k1 = k1
        self.b = b
        self.idf_cache: Dict[str, float] = {}
    
    def calculate_idf(self, term: str, total_docs: int, docs_with_term: int) -> float:
        """
        Calculate Inverse Document Frequency for a term.
        
        IDF measures how informative a term is - rare terms get higher scores.
        """
        if term in self.idf_cache:
            return self.idf_cache[term]
        
        # BM25 IDF formula: log((N - df + 0.5) / (df + 0.5))
        # where N = total documents, df = documents containing term
        if docs_with_term == 0:
            idf = 0.0
        else:
            idf = math.log((total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5))
        
        self.idf_cache[term] = idf
        return idf
    
    def score_document(
        self, 
        query_terms: List[str],
        document: DocumentIndex,
        avg_doc_length: float,
        total_docs: int,
        term_doc_frequencies: Dict[str, int]
    ) -> float:
        """
        Calculate BM25 score for a document given a query.
        
        Args:
            query_terms: List of query terms
            document: The document to score
            avg_doc_length: Average document length in corpus
            total_docs: Total number of documents
            term_doc_frequencies: Number of documents containing each term
            
        Returns:
            BM25 score for the document
        """
        score = 0.0
        
        for term in query_terms:
            if term not in document.token_frequencies:
                continue
                
            # Term frequency in this document
            tf = document.token_frequencies[term]
            
            # IDF for this term
            docs_with_term = term_doc_frequencies.get(term, 0)
            idf = self.calculate_idf(term, total_docs, docs_with_term)
            
            # BM25 formula component for this term
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (document.doc_length / avg_doc_length))
            
            term_score = idf * (numerator / denominator)
            score += term_score
        
        return max(0.0, score)  # Ensure non-negative scores


class TextSearchBackend(BackendSearchInterface):
    """
    BM25-based text search backend.
    
    Provides full-text search capabilities using the BM25 ranking algorithm,
    complementing vector-based semantic search with traditional keyword matching.
    """
    
    backend_name = "text"
    
    def __init__(self):
        """Initialize the text search backend."""
        self.documents: Dict[str, DocumentIndex] = {}
        self.term_doc_frequencies: Dict[str, int] = defaultdict(int)
        self.total_doc_length = 0
        self.scorer = BM25Scorer()
        self.last_indexed_count = 0
        self.index_dirty = False
        
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25 indexing.
        
        In production, this could be enhanced with:
        - Stemming/lemmatization
        - Stop word removal
        - N-gram extraction
        - Language-specific processing
        """
        # Convert to lowercase and split on whitespace/punctuation
        import re
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        return tokens
    
    async def index_document(
        self, 
        doc_id: str, 
        text: str, 
        content_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Index a document for BM25 search.
        
        Args:
            doc_id: Unique document identifier
            text: Document text content
            content_type: Type of content being indexed
            metadata: Additional document metadata
        """
        try:
            # Tokenize the document
            tokens = self.tokenize(text)
            token_frequencies = Counter(tokens)
            
            # Create document index
            doc_index = DocumentIndex(
                doc_id=doc_id,
                text=text,
                tokens=tokens,
                token_frequencies=token_frequencies,
                doc_length=len(tokens),
                metadata=metadata or {}
            )
            
            # Update global statistics if this is a new document
            if doc_id not in self.documents:
                self.total_doc_length += len(tokens)
                
                # Update term document frequencies
                for term in set(tokens):  # Use set to count each term only once per document
                    self.term_doc_frequencies[term] += 1
            else:
                # Remove old document statistics
                old_doc = self.documents[doc_id]
                self.total_doc_length -= old_doc.doc_length
                self.total_doc_length += len(tokens)
                
                # Update term frequencies (remove old, add new)
                old_terms = set(old_doc.tokens)
                new_terms = set(tokens)
                
                for term in old_terms - new_terms:
                    self.term_doc_frequencies[term] -= 1
                    if self.term_doc_frequencies[term] <= 0:
                        del self.term_doc_frequencies[term]
                
                for term in new_terms - old_terms:
                    self.term_doc_frequencies[term] += 1
            
            # Store the document
            self.documents[doc_id] = doc_index
            self.index_dirty = True
            
            search_logger.debug(
                f"Indexed document for text search",
                doc_id=doc_id,
                token_count=len(tokens),
                unique_terms=len(token_frequencies),
                total_documents=len(self.documents)
            )
            
        except Exception as e:
            search_logger.error(f"Failed to index document {doc_id}: {e}")
            raise BackendSearchError(self.backend_name, f"Indexing failed: {e}")
    
    async def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the index.
        
        Args:
            doc_id: Document identifier to remove
            
        Returns:
            True if document was removed, False if not found
        """
        if doc_id not in self.documents:
            return False
        
        try:
            doc = self.documents[doc_id]
            
            # Update global statistics
            self.total_doc_length -= doc.doc_length
            
            # Update term document frequencies
            for term in set(doc.tokens):
                self.term_doc_frequencies[term] -= 1
                if self.term_doc_frequencies[term] <= 0:
                    del self.term_doc_frequencies[term]
            
            # Remove document
            del self.documents[doc_id]
            self.index_dirty = True
            
            search_logger.debug(f"Removed document from text index", doc_id=doc_id)
            return True
            
        except Exception as e:
            search_logger.error(f"Failed to remove document {doc_id}: {e}")
            return False
    
    async def search(self, query: str, options: SearchOptions) -> List[MemoryResult]:
        """
        Execute BM25 text search.
        
        Args:
            query: Search query string
            options: Search configuration options
            
        Returns:
            List of MemoryResult objects ranked by BM25 score
            
        Raises:
            BackendSearchError: If search execution fails
        """
        start_time = time.time()
        
        try:
            if not self.documents:
                return []
            
            # Tokenize query
            query_terms = self.tokenize(query)
            if not query_terms:
                return []
            
            search_logger.debug(
                f"Executing BM25 text search",
                query_terms=query_terms,
                index_size=len(self.documents),
                term_vocabulary_size=len(self.term_doc_frequencies)
            )
            
            # Calculate average document length
            avg_doc_length = self.total_doc_length / len(self.documents) if self.documents else 1.0
            
            # Score all documents
            scored_docs = []
            total_docs = len(self.documents)
            
            for doc_id, document in self.documents.items():
                score = self.scorer.score_document(
                    query_terms=query_terms,
                    document=document,
                    avg_doc_length=avg_doc_length,
                    total_docs=total_docs,
                    term_doc_frequencies=self.term_doc_frequencies
                )
                
                # Apply score threshold
                if score >= options.score_threshold:
                    scored_docs.append((doc_id, document, score))
            
            # Sort by score (highest first)
            scored_docs.sort(key=lambda x: x[2], reverse=True)
            
            # Convert to MemoryResult objects
            results = []
            for doc_id, document, score in scored_docs[:options.limit]:
                # Normalize BM25 score to [0, 1] range (approximate)
                # BM25 scores can vary widely, so we use a sigmoid-like normalization
                normalized_score = min(1.0, score / (score + 1.0))
                
                result = MemoryResult(
                    id=doc_id,
                    text=document.text,
                    content_type=document.metadata.get("content_type", "text"),
                    source=self.backend_name,
                    score=normalized_score,
                    tags=document.metadata.get("tags", []),
                    metadata={
                        **document.metadata,
                        "bm25_raw_score": score,
                        "matched_terms": [term for term in query_terms if term in document.token_frequencies],
                        "doc_length": document.doc_length
                    }
                )
                results.append(result)
            
            search_time = (time.time() - start_time) * 1000
            
            search_logger.info(
                f"BM25 text search completed",
                query_terms=query_terms,
                total_candidates=len(self.documents),
                results_above_threshold=len(scored_docs),
                final_results=len(results),
                search_time_ms=search_time
            )
            
            return results
            
        except Exception as e:
            search_time = (time.time() - start_time) * 1000
            error_msg = f"BM25 text search failed: {e}"
            
            search_logger.error(
                error_msg,
                query=query,
                search_time_ms=search_time,
                error=str(e)
            )
            
            raise BackendSearchError(self.backend_name, error_msg, e)
    
    async def health_check(self) -> BackendHealthStatus:
        """
        Perform health check on the text search backend.
        
        Returns:
            BackendHealthStatus object with current backend status
        """
        start_time = time.time()
        
        try:
            # Basic health metrics
            document_count = len(self.documents)
            vocabulary_size = len(self.term_doc_frequencies)
            avg_doc_length = self.total_doc_length / document_count if document_count > 0 else 0
            
            # Test basic search functionality with simple query
            if document_count > 0:
                test_results = await self.search("test", SearchOptions(limit=1))
                search_functional = True
            else:
                test_results = []
                search_functional = True  # Empty index is still functional
            
            response_time = (time.time() - start_time) * 1000
            
            return BackendHealthStatus(
                status="healthy",
                response_time_ms=response_time,
                error_message=None,
                metadata={
                    "document_count": document_count,
                    "vocabulary_size": vocabulary_size,
                    "average_document_length": avg_doc_length,
                    "index_dirty": self.index_dirty,
                    "search_functional": search_functional,
                    "test_query_results": len(test_results),
                    "backend_type": "bm25_text_search"
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return BackendHealthStatus(
                status="unhealthy",
                response_time_ms=response_time,
                error_message=f"Health check failed: {e}",
                metadata={
                    "error_type": type(e).__name__,
                    "backend_type": "bm25_text_search"
                }
            )
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the text search index.
        
        Returns:
            Dictionary with comprehensive index statistics
        """
        if not self.documents:
            return {
                "document_count": 0,
                "vocabulary_size": 0,
                "total_tokens": 0,
                "average_document_length": 0,
                "top_terms": []
            }
        
        # Calculate statistics
        document_count = len(self.documents)
        vocabulary_size = len(self.term_doc_frequencies)
        average_doc_length = self.total_doc_length / document_count
        
        # Find most common terms
        top_terms = sorted(
            self.term_doc_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]  # Top 20 terms
        
        return {
            "document_count": document_count,
            "vocabulary_size": vocabulary_size,
            "total_tokens": self.total_doc_length,
            "average_document_length": average_doc_length,
            "top_terms": [{"term": term, "document_frequency": freq} for term, freq in top_terms],
            "index_dirty": self.index_dirty,
            "scorer_parameters": {
                "k1": self.scorer.k1,
                "b": self.scorer.b
            }
        }
    
    async def rebuild_index(self) -> Dict[str, Any]:
        """
        Rebuild the entire text search index.
        
        This is useful for maintenance, optimization, or recovering from corruption.
        
        Returns:
            Dictionary with rebuild statistics
        """
        start_time = time.time()
        
        try:
            search_logger.info("Starting text search index rebuild")
            
            # Store current documents
            old_documents = dict(self.documents)
            
            # Reset index
            self.documents.clear()
            self.term_doc_frequencies.clear()
            self.total_doc_length = 0
            self.scorer.idf_cache.clear()
            
            # Re-index all documents
            rebuild_count = 0
            for doc_id, old_doc in old_documents.items():
                await self.index_document(
                    doc_id=doc_id,
                    text=old_doc.text,
                    metadata=old_doc.metadata
                )
                rebuild_count += 1
            
            rebuild_time = (time.time() - start_time) * 1000
            self.index_dirty = False
            
            search_logger.info(
                f"Text search index rebuild completed",
                documents_rebuilt=rebuild_count,
                rebuild_time_ms=rebuild_time
            )
            
            return {
                "success": True,
                "documents_rebuilt": rebuild_count,
                "rebuild_time_ms": rebuild_time,
                "new_vocabulary_size": len(self.term_doc_frequencies),
                "new_total_tokens": self.total_doc_length
            }
            
        except Exception as e:
            rebuild_time = (time.time() - start_time) * 1000
            error_msg = f"Index rebuild failed: {e}"
            
            search_logger.error(error_msg, rebuild_time_ms=rebuild_time)
            
            return {
                "success": False,
                "error": error_msg,
                "rebuild_time_ms": rebuild_time
            }


# Global instance for easy access
_text_backend_instance: Optional[TextSearchBackend] = None


def get_text_backend() -> Optional[TextSearchBackend]:
    """Get the global text backend instance."""
    return _text_backend_instance


def initialize_text_backend() -> TextSearchBackend:
    """
    Initialize and return the global text search backend.
    
    Returns:
        TextSearchBackend instance
    """
    global _text_backend_instance
    _text_backend_instance = TextSearchBackend()
    search_logger.info("Text search backend initialized")
    return _text_backend_instance


def set_text_backend(backend: TextSearchBackend) -> None:
    """Set the global text backend instance."""
    global _text_backend_instance
    _text_backend_instance = backend
    search_logger.info("Text search backend instance set globally")