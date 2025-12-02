#!/usr/bin/env python3
"""
Comprehensive tests for the BM25 text search backend.

Tests cover BM25 scoring accuracy, indexing operations, search functionality,
and integration with the broader search architecture.
"""

import pytest
import asyncio
from datetime import datetime
from typing import List, Dict, Any

from src.backends.text_backend import (
    TextSearchBackend, BM25Scorer, DocumentIndex,
    initialize_text_backend, get_text_backend, set_text_backend
)
from src.interfaces.backend_interface import SearchOptions
from src.interfaces.memory_result import MemoryResult


class TestBM25Scorer:
    """Test BM25 scoring algorithm implementation."""
    
    def test_bm25_scorer_initialization(self):
        """Test BM25 scorer initialization with default parameters."""
        scorer = BM25Scorer()
        assert scorer.k1 == 1.5
        assert scorer.b == 0.75
        assert scorer.idf_cache == {}
    
    def test_bm25_scorer_custom_parameters(self):
        """Test BM25 scorer with custom parameters."""
        scorer = BM25Scorer(k1=2.0, b=0.5)
        assert scorer.k1 == 2.0
        assert scorer.b == 0.5
    
    def test_calculate_idf(self):
        """Test IDF calculation."""
        scorer = BM25Scorer()
        
        # Test IDF for term appearing in half the documents
        idf = scorer.calculate_idf("test", total_docs=100, docs_with_term=50)
        assert idf == pytest.approx(0.0, abs=0.1)  # log((100-50+0.5)/(50+0.5)) ≈ 0
        
        # Test IDF for rare term
        idf_rare = scorer.calculate_idf("rare", total_docs=100, docs_with_term=1)
        assert idf_rare > 0  # Should be positive for rare terms
        
        # Test IDF for common term
        idf_common = scorer.calculate_idf("common", total_docs=100, docs_with_term=90)
        assert idf_common < 0  # Should be negative for very common terms
    
    def test_idf_caching(self):
        """Test that IDF values are properly cached."""
        scorer = BM25Scorer()
        
        # First calculation
        idf1 = scorer.calculate_idf("cached", total_docs=100, docs_with_term=10)
        assert "cached" in scorer.idf_cache
        assert scorer.idf_cache["cached"] == idf1
        
        # Second calculation should use cache
        idf2 = scorer.calculate_idf("cached", total_docs=200, docs_with_term=20)  # Different values
        assert idf2 == idf1  # Should return cached value, not recalculate
    
    def test_score_document(self):
        """Test document scoring with BM25."""
        scorer = BM25Scorer()
        
        # Create test document
        doc = DocumentIndex(
            doc_id="test1",
            text="The quick brown fox jumps over the lazy dog",
            tokens=["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
            token_frequencies={"the": 2, "quick": 1, "brown": 1, "fox": 1, "jumps": 1, "over": 1, "lazy": 1, "dog": 1},
            doc_length=9,
            metadata={}
        )
        
        # Simple term frequencies for test
        term_doc_frequencies = {"the": 10, "quick": 5, "brown": 3, "missing": 0}
        
        # Test scoring with query terms
        score = scorer.score_document(
            query_terms=["quick", "brown"],
            document=doc,
            avg_doc_length=10.0,
            total_docs=100,
            term_doc_frequencies=term_doc_frequencies
        )
        
        assert score > 0  # Should get positive score
        
        # Test with query terms not in document
        score_missing = scorer.score_document(
            query_terms=["missing", "words"],
            document=doc,
            avg_doc_length=10.0,
            total_docs=100,
            term_doc_frequencies=term_doc_frequencies
        )
        
        assert score_missing == 0  # Should get zero score for missing terms


class TestFuzzyMatcher:
    """Test fuzzy string matching utilities."""
    
    def test_levenshtein_distance(self):
        """Test Levenshtein distance calculation."""
        from src.backends.text_backend import FuzzyMatcher
        
        # Identical strings
        assert FuzzyMatcher.levenshtein_distance("test", "test") == 0
        
        # Single character difference
        assert FuzzyMatcher.levenshtein_distance("test", "best") == 1
        
        # Insertion
        assert FuzzyMatcher.levenshtein_distance("test", "tests") == 1
        
        # Deletion
        assert FuzzyMatcher.levenshtein_distance("tests", "test") == 1
        
        # Empty strings
        assert FuzzyMatcher.levenshtein_distance("", "") == 0
        assert FuzzyMatcher.levenshtein_distance("test", "") == 4
    
    def test_similarity_ratio(self):
        """Test similarity ratio calculation."""
        from src.backends.text_backend import FuzzyMatcher
        
        # Identical strings
        assert FuzzyMatcher.similarity_ratio("test", "test") == 1.0
        
        # Completely different
        ratio = FuzzyMatcher.similarity_ratio("abc", "xyz")
        assert 0 <= ratio <= 1
        
        # Similar strings
        ratio = FuzzyMatcher.similarity_ratio("testing", "tester")
        assert ratio > 0.5
    
    def test_fuzzy_match(self):
        """Test fuzzy matching with threshold."""
        from src.backends.text_backend import FuzzyMatcher
        
        # Exact match
        assert FuzzyMatcher.fuzzy_match("test", "test", threshold=1.0)
        
        # Similar match
        assert FuzzyMatcher.fuzzy_match("testing", "tester", threshold=0.5)
        
        # Different match
        assert not FuzzyMatcher.fuzzy_match("completely", "different", threshold=0.8)


class TestTextSearchBackend:
    """Test the main text search backend functionality."""
    
    @pytest.fixture
    def backend(self):
        """Create a fresh text search backend for each test."""
        return TextSearchBackend()
    
    def test_backend_initialization(self, backend):
        """Test backend initialization."""
        assert backend.backend_name == "text"
        assert len(backend.documents) == 0
        assert len(backend.term_doc_frequencies) == 0
        assert backend.total_doc_length == 0
        assert backend.last_indexed_count == 0
        assert not backend.index_dirty
    
    def test_tokenization(self, backend):
        """Test text tokenization."""
        # Simple text
        tokens = backend.tokenize("Hello world")
        assert tokens == ["hello", "world"]
        
        # Text with punctuation
        tokens = backend.tokenize("Hello, world! How are you?")
        assert "hello" in tokens
        assert "world" in tokens
        assert "how" in tokens
        assert "are" in tokens
        assert "you" in tokens
        
        # Empty text
        tokens = backend.tokenize("")
        assert tokens == []
        
        # Numbers and mixed content
        tokens = backend.tokenize("Test 123 with numbers and symbols!")
        assert "test" in tokens
        assert "123" in tokens
        assert "with" in tokens
    
    @pytest.mark.asyncio
    async def test_document_indexing(self, backend):
        """Test document indexing functionality."""
        # Index a simple document
        await backend.index_document(
            doc_id="doc1",
            text="The quick brown fox jumps over the lazy dog",
            content_type="text",
            metadata={"author": "test"}
        )
        
        assert len(backend.documents) == 1
        assert "doc1" in backend.documents
        
        doc = backend.documents["doc1"]
        assert doc.doc_id == "doc1"
        assert "quick" in doc.token_frequencies
        assert "the" in doc.token_frequencies
        assert doc.token_frequencies["the"] == 2  # "the" appears twice
        assert doc.metadata["author"] == "test"
    
    @pytest.mark.asyncio
    async def test_document_reindexing(self, backend):
        """Test updating an existing document."""
        # Index initial document
        await backend.index_document("doc1", "Original text", metadata={"version": 1})
        original_length = backend.total_doc_length
        
        # Re-index with new content
        await backend.index_document("doc1", "Updated text content", metadata={"version": 2})
        
        assert len(backend.documents) == 1  # Still only one document
        doc = backend.documents["doc1"]
        assert "updated" in doc.token_frequencies
        assert doc.metadata["version"] == 2
        assert backend.total_doc_length != original_length
    
    @pytest.mark.asyncio
    async def test_document_removal(self, backend):
        """Test document removal from index."""
        # Index documents
        await backend.index_document("doc1", "First document")
        await backend.index_document("doc2", "Second document") 
        
        assert len(backend.documents) == 2
        
        # Remove one document
        removed = await backend.remove_document("doc1")
        assert removed is True
        assert len(backend.documents) == 1
        assert "doc1" not in backend.documents
        assert "doc2" in backend.documents
        
        # Try to remove non-existent document
        removed = await backend.remove_document("nonexistent")
        assert removed is False
    
    @pytest.mark.asyncio
    async def test_basic_search(self, backend):
        """Test basic text search functionality."""
        # Index test documents
        await backend.index_document("doc1", "Python programming language", metadata={"topic": "programming"})
        await backend.index_document("doc2", "Java programming tutorial", metadata={"topic": "programming"})
        await backend.index_document("doc3", "Machine learning with Python", metadata={"topic": "ml"})
        
        # Search for programming
        options = SearchOptions(limit=10)
        results = await backend.search("programming", options)
        
        assert len(results) >= 2
        # Results should be ranked by relevance
        assert all(isinstance(r, MemoryResult) for r in results)
        assert all(r.source == "text" for r in results)
        
        # Verify BM25 scoring
        assert all(0 <= r.score <= 1 for r in results)
    
    @pytest.mark.asyncio
    async def test_search_relevance_ranking(self, backend):
        """Test that search results are properly ranked by relevance."""
        # Index documents with different relevance levels
        await backend.index_document("doc1", "machine learning artificial intelligence")
        await backend.index_document("doc2", "machine learning machine learning machine learning")  # More repetition
        await backend.index_document("doc3", "artificial intelligence AI")
        
        options = SearchOptions(limit=10)
        results = await backend.search("machine learning", options)
        
        # Results should be in descending order of score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        
        # Document with more term repetition should score higher
        doc2_result = next((r for r in results if r.id == "doc2"), None)
        doc1_result = next((r for r in results if r.id == "doc1"), None)
        
        if doc2_result and doc1_result:
            assert doc2_result.score >= doc1_result.score
    
    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, backend):
        """Test search with score thresholding."""
        # Index documents
        await backend.index_document("doc1", "highly relevant matching content")
        await backend.index_document("doc2", "somewhat related material")
        await backend.index_document("doc3", "completely unrelated information")
        
        # Search with high score threshold
        options = SearchOptions(limit=10, score_threshold=0.3)
        results = await backend.search("highly relevant", options)
        
        # Should only return results above threshold
        assert all(r.score >= 0.3 for r in results)
        
        # Should have fewer results than without threshold
        options_no_threshold = SearchOptions(limit=10, score_threshold=0.0)
        all_results = await backend.search("highly relevant", options_no_threshold)
        assert len(results) <= len(all_results)
    
    @pytest.mark.asyncio
    async def test_empty_query_search(self, backend):
        """Test search behavior with empty query."""
        await backend.index_document("doc1", "Some content")
        
        options = SearchOptions(limit=10)
        results = await backend.search("", options)
        
        # Should return empty results for empty query
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_search_nonexistent_terms(self, backend):
        """Test search for terms not in index."""
        await backend.index_document("doc1", "existing content here")
        
        options = SearchOptions(limit=10)
        results = await backend.search("nonexistent terms", options)
        
        # Should return empty results
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, backend):
        """Test backend health check functionality."""
        # Health check on empty index
        health = await backend.health_check()
        assert health.status == "healthy"
        assert health.response_time_ms > 0
        assert health.metadata["document_count"] == 0
        
        # Add documents and check again
        await backend.index_document("doc1", "Test content")
        health = await backend.health_check()
        assert health.status == "healthy"
        assert health.metadata["document_count"] == 1
        assert health.metadata["search_functional"] is True
    
    def test_index_statistics(self, backend):
        """Test index statistics functionality."""
        # Empty index statistics
        stats = backend.get_index_statistics()
        assert stats["document_count"] == 0
        assert stats["vocabulary_size"] == 0
        assert stats["total_tokens"] == 0
        
    @pytest.mark.asyncio
    async def test_index_statistics_with_content(self, backend):
        """Test index statistics with content."""
        # Add some documents
        await backend.index_document("doc1", "python programming language tutorial", metadata={"type": "tutorial"})
        await backend.index_document("doc2", "java programming language guide", metadata={"type": "guide"})
        
        stats = backend.get_index_statistics()
        assert stats["document_count"] == 2
        assert stats["vocabulary_size"] > 0
        assert stats["total_tokens"] > 0
        assert stats["average_document_length"] > 0
        assert len(stats["top_terms"]) > 0
        
        # Verify top terms structure
        top_term = stats["top_terms"][0]
        assert "term" in top_term
        assert "document_frequency" in top_term
    
    @pytest.mark.asyncio
    async def test_rebuild_index(self, backend):
        """Test index rebuilding functionality."""
        # Add documents
        await backend.index_document("doc1", "Content for rebuilding")
        await backend.index_document("doc2", "More content here")
        
        original_stats = backend.get_index_statistics()
        
        # Rebuild index
        rebuild_result = await backend.rebuild_index()
        
        assert rebuild_result["success"] is True
        assert rebuild_result["documents_rebuilt"] == 2
        assert rebuild_result["rebuild_time_ms"] > 0
        
        # Verify index is rebuilt correctly
        new_stats = backend.get_index_statistics()
        assert new_stats["document_count"] == original_stats["document_count"]
        assert not backend.index_dirty


class TestTextBackendIntegration:
    """Test integration scenarios with the text search backend."""
    
    @pytest.mark.asyncio
    async def test_large_document_indexing(self):
        """Test indexing and searching large documents."""
        backend = TextSearchBackend()
        
        # Create a large document
        large_text = " ".join(["word" + str(i) for i in range(1000)])  # 1000 unique words
        
        await backend.index_document("large_doc", large_text)
        
        stats = backend.get_index_statistics()
        assert stats["document_count"] == 1
        assert stats["vocabulary_size"] == 1000
        
        # Search should still work
        options = SearchOptions(limit=5)
        results = await backend.search("word500", options)
        assert len(results) == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test thread safety with concurrent operations."""
        backend = TextSearchBackend()
        
        # Index multiple documents concurrently
        tasks = []
        for i in range(10):
            task = backend.index_document(f"doc{i}", f"Document {i} content here")
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        assert len(backend.documents) == 10
        
        # Concurrent searches
        search_tasks = []
        for i in range(5):
            task = backend.search("content", SearchOptions(limit=10))
            search_tasks.append(task)
        
        results_list = await asyncio.gather(*search_tasks)
        
        # All searches should return same results
        first_results = results_list[0]
        for results in results_list[1:]:
            assert len(results) == len(first_results)
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory usage with many documents."""
        backend = TextSearchBackend()
        
        # Index many small documents
        for i in range(100):
            await backend.index_document(f"doc{i}", f"Short document {i}")
        
        stats = backend.get_index_statistics()
        assert stats["document_count"] == 100
        
        # Verify search performance is still reasonable
        import time
        start_time = time.time()
        
        options = SearchOptions(limit=20)
        results = await backend.search("document", options)
        
        search_time = time.time() - start_time
        assert search_time < 1.0  # Should complete within 1 second
        assert len(results) == 20  # Should return requested number of results


class TestGlobalBackendManagement:
    """Test global backend instance management."""
    
    def test_initialize_text_backend(self):
        """Test global backend initialization."""
        backend = initialize_text_backend()
        assert isinstance(backend, TextSearchBackend)
        assert get_text_backend() is backend
    
    def test_set_text_backend(self):
        """Test setting custom backend instance."""
        custom_backend = TextSearchBackend()
        set_text_backend(custom_backend)
        assert get_text_backend() is custom_backend
    
    def test_get_text_backend_none(self):
        """Test getting backend when none is set."""
        # Reset global instance
        import src.backends.text_backend
        src.backends.text_backend._text_backend_instance = None
        
        assert get_text_backend() is None


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_search_with_special_characters(self):
        """Test search with special characters and Unicode."""
        backend = TextSearchBackend()
        
        # Index documents with special characters
        await backend.index_document("doc1", "Special chars: !@#$%^&*()")
        await backend.index_document("doc2", "Unicode: café résumé naïve")
        await backend.index_document("doc3", "Mixed: test123 hello@world.com")
        
        # Search should handle special characters gracefully
        options = SearchOptions(limit=10)
        results = await backend.search("café", options)
        # May or may not find results depending on tokenization, but should not crash
        
        results = await backend.search("hello", options)
        # Should find the document with email
    
    @pytest.mark.asyncio
    async def test_extremely_long_queries(self):
        """Test behavior with very long search queries."""
        backend = TextSearchBackend()
        
        await backend.index_document("doc1", "Normal content here")
        
        # Very long query
        long_query = " ".join(["word"] * 1000)
        options = SearchOptions(limit=10)
        
        # Should not crash, though may be slow
        results = await backend.search(long_query, options)
        # Results depend on content, but operation should complete
    
    @pytest.mark.asyncio
    async def test_duplicate_document_ids(self):
        """Test handling of duplicate document IDs."""
        backend = TextSearchBackend()
        
        # Index document
        await backend.index_document("doc1", "Original content")
        original_count = len(backend.documents)
        
        # Index with same ID (should update, not duplicate)
        await backend.index_document("doc1", "Updated content")
        
        assert len(backend.documents) == original_count
        assert "updated" in backend.documents["doc1"].text
    
    @pytest.mark.asyncio
    async def test_search_empty_index(self):
        """Test searching an empty index."""
        backend = TextSearchBackend()
        
        options = SearchOptions(limit=10)
        results = await backend.search("anything", options)
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_malformed_metadata(self):
        """Test handling of malformed metadata."""
        backend = TextSearchBackend()
        
        # Should handle None metadata gracefully
        await backend.index_document("doc1", "Content", metadata=None)
        
        # Should handle empty metadata
        await backend.index_document("doc2", "Content", metadata={})
        
        # Should handle complex nested metadata
        complex_meta = {
            "nested": {"deep": {"value": 123}},
            "list": [1, 2, 3],
            "none_value": None
        }
        await backend.index_document("doc3", "Content", metadata=complex_meta)
        
        assert len(backend.documents) == 3