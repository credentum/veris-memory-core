#!/usr/bin/env python3
"""
Integration tests for text search backend with the broader search architecture.

Tests the integration of BM25 text search with QueryDispatcher, RetrievalCore,
and the overall search pipeline including ranking and filtering.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List

from src.backends.text_backend import TextSearchBackend, initialize_text_backend
from src.core.query_dispatcher import QueryDispatcher, SearchMode
from src.core.retrieval_core import RetrievalCore, initialize_retrieval_core
from src.interfaces.backend_interface import SearchOptions
from src.interfaces.memory_result import MemoryResult


class MockVectorBackend:
    """Mock vector backend for integration testing."""
    
    backend_name = "vector"
    
    def __init__(self):
        self.documents = {}
    
    async def index_document(self, doc_id: str, text: str, **kwargs):
        """Mock document indexing."""
        self.documents[doc_id] = {
            "text": text,
            "metadata": kwargs.get("metadata", {})
        }
    
    async def search(self, query: str, options: SearchOptions) -> List[MemoryResult]:
        """Mock vector search - returns documents with vector scores."""
        results = []
        
        # Simple keyword matching for mock
        for doc_id, doc in self.documents.items():
            if any(term.lower() in doc["text"].lower() for term in query.split()):
                result = MemoryResult(
                    id=doc_id,
                    text=doc["text"],
                    content_type=doc["metadata"].get("content_type", "text"),
                    source=self.backend_name,
                    score=0.75,  # Mock vector score
                    tags=doc["metadata"].get("tags", []),
                    metadata=doc["metadata"]
                )
                results.append(result)
        
        return results[:options.limit]
    
    async def health_check(self):
        """Mock health check."""
        from src.interfaces.backend_interface import BackendHealthStatus
        return BackendHealthStatus(
            status="healthy",
            response_time_ms=10.0,
            error_message=None,
            metadata={"mock": True, "documents": len(self.documents)}
        )


class TestTextSearchIntegration:
    """Test text search backend integration with the search architecture."""
    
    @pytest.fixture
    async def integrated_dispatcher(self):
        """Create a query dispatcher with both text and mock vector backends."""
        # Initialize text backend
        text_backend = TextSearchBackend()
        
        # Create mock vector backend
        vector_backend = MockVectorBackend()
        
        # Create dispatcher and register backends
        dispatcher = QueryDispatcher()
        dispatcher.register_backend("text", text_backend)
        dispatcher.register_backend("vector", vector_backend)
        
        # Index test documents in both backends
        test_docs = [
            {
                "id": "doc1",
                "text": "Python programming tutorial for beginners",
                "metadata": {"content_type": "tutorial", "tags": ["python", "programming"], "difficulty": "easy"}
            },
            {
                "id": "doc2", 
                "text": "Advanced machine learning algorithms and techniques",
                "metadata": {"content_type": "article", "tags": ["ml", "algorithms"], "difficulty": "hard"}
            },
            {
                "id": "doc3",
                "text": "JavaScript fundamentals and web development",
                "metadata": {"content_type": "guide", "tags": ["javascript", "web"], "difficulty": "medium"}
            },
            {
                "id": "doc4",
                "text": "Database design patterns and best practices",
                "metadata": {"content_type": "reference", "tags": ["database", "patterns"], "difficulty": "medium"}
            }
        ]
        
        # Index in text backend
        for doc in test_docs:
            await text_backend.index_document(
                doc_id=doc["id"],
                text=doc["text"],
                metadata=doc["metadata"]
            )
        
        # Index in vector backend
        for doc in test_docs:
            await vector_backend.index_document(
                doc_id=doc["id"],
                text=doc["text"],
                metadata=doc["metadata"]
            )
        
        return dispatcher
    
    @pytest.mark.asyncio
    async def test_text_only_search(self, integrated_dispatcher):
        """Test text-only search mode."""
        options = SearchOptions(limit=10)
        
        response = await integrated_dispatcher.dispatch_query(
            query="python programming",
            search_mode=SearchMode.TEXT,
            options=options
        )
        
        assert response.success is True
        assert len(response.results) >= 1
        assert response.search_mode_used == "text"
        assert "text" in response.backends_used
        assert "vector" not in response.backends_used
        
        # Verify results are from text backend
        for result in response.results:
            assert result.source == "text"
            assert "python" in result.text.lower() or "programming" in result.text.lower()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_with_text(self, integrated_dispatcher):
        """Test hybrid search including text backend."""
        options = SearchOptions(limit=10)
        
        response = await integrated_dispatcher.dispatch_query(
            query="machine learning",
            search_mode=SearchMode.HYBRID,
            options=options
        )
        
        assert response.success is True
        assert response.search_mode_used == "hybrid"
        
        # Should use both backends
        assert "text" in response.backends_used
        assert "vector" in response.backends_used
        
        # Should have results from both backends
        text_results = [r for r in response.results if r.source == "text"]
        vector_results = [r for r in response.results if r.source == "vector"]
        
        assert len(text_results) >= 1
        assert len(vector_results) >= 1
    
    @pytest.mark.asyncio
    async def test_auto_search_mode_text_selection(self, integrated_dispatcher):
        """Test auto mode selecting text search for keyword queries."""
        options = SearchOptions(limit=10)
        
        # Query with keyword indicators should favor text search
        response = await integrated_dispatcher.dispatch_query(
            query='find "exact phrase" in content',
            search_mode=SearchMode.AUTO,
            options=options
        )
        
        assert response.success is True
        assert "text" in response.backends_used
    
    @pytest.mark.asyncio
    async def test_text_search_ranking_integration(self, integrated_dispatcher):
        """Test that text search results are properly ranked with other backends."""
        options = SearchOptions(limit=10)
        
        response = await integrated_dispatcher.dispatch_query(
            query="programming tutorial",
            search_mode=SearchMode.HYBRID,
            options=options
        )
        
        assert response.success is True
        assert len(response.results) > 0
        
        # Results should be sorted by score (descending)
        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)
        
        # Should have mixed sources if both backends have relevant results
        sources = set(r.source for r in response.results)
        assert len(sources) >= 1
    
    @pytest.mark.asyncio
    async def test_text_search_performance_metrics(self, integrated_dispatcher):
        """Test that text search backend reports proper timing metrics."""
        options = SearchOptions(limit=5)
        
        response = await integrated_dispatcher.dispatch_query(
            query="database patterns",
            search_mode=SearchMode.TEXT,
            options=options
        )
        
        assert response.success is True
        assert response.response_time_ms > 0
        assert "text" in response.backend_timings
        assert response.backend_timings["text"] > 0
    
    @pytest.mark.asyncio
    async def test_text_search_with_score_threshold(self, integrated_dispatcher):
        """Test text search with score thresholding."""
        options = SearchOptions(limit=10, score_threshold=0.5)
        
        response = await integrated_dispatcher.dispatch_query(
            query="advanced algorithms",
            search_mode=SearchMode.TEXT,
            options=options
        )
        
        assert response.success is True
        
        # All results should meet score threshold
        for result in response.results:
            assert result.score >= 0.5
    
    @pytest.mark.asyncio
    async def test_text_backend_health_check_integration(self, integrated_dispatcher):
        """Test text backend health check through dispatcher."""
        health_results = await integrated_dispatcher.health_check_all_backends()
        
        assert "text" in health_results
        text_health = health_results["text"]
        
        assert text_health["status"] == "healthy"
        assert text_health["response_time_ms"] > 0
        assert text_health["metadata"]["backend_type"] == "bm25_text_search"
    
    @pytest.mark.asyncio
    async def test_empty_text_search_results(self, integrated_dispatcher):
        """Test text search with no matching results."""
        options = SearchOptions(limit=10)
        
        response = await integrated_dispatcher.dispatch_query(
            query="nonexistent terms that match nothing",
            search_mode=SearchMode.TEXT,
            options=options
        )
        
        assert response.success is True
        assert len(response.results) == 0
        assert "text" in response.backends_used
    
    @pytest.mark.asyncio
    async def test_text_search_error_handling(self, integrated_dispatcher):
        """Test error handling in text search integration."""
        # Corrupt the text backend to simulate error
        text_backend = integrated_dispatcher.get_backend("text")
        original_search = text_backend.search
        
        async def failing_search(*args, **kwargs):
            raise Exception("Simulated text search failure")
        
        text_backend.search = failing_search
        
        try:
            options = SearchOptions(limit=10)
            
            response = await integrated_dispatcher.dispatch_query(
                query="test query",
                search_mode=SearchMode.HYBRID,  # Should fall back to vector
                options=options
            )
            
            # Hybrid search should still succeed with vector backend
            assert response.success is True
            # Should not have text results due to failure
            assert "text" not in response.backends_used or response.backend_timings.get("text", 1) == 0
            
        finally:
            # Restore original search method
            text_backend.search = original_search


class TestRetrievalCoreTextIntegration:
    """Test RetrievalCore integration with text search."""
    
    @pytest.fixture
    async def retrieval_core_with_text(self):
        """Create RetrievalCore with text backend."""
        # Initialize text backend
        text_backend = TextSearchBackend()
        
        # Index test documents
        test_docs = [
            ("doc1", "Python web development with Django framework"),
            ("doc2", "Machine learning model training and evaluation"),
            ("doc3", "JavaScript frontend development best practices")
        ]
        
        for doc_id, text in test_docs:
            await text_backend.index_document(doc_id, text)
        
        # Create dispatcher and register text backend
        dispatcher = QueryDispatcher()
        dispatcher.register_backend("text", text_backend)
        
        # Create and return RetrievalCore
        return RetrievalCore(dispatcher)
    
    @pytest.mark.asyncio
    async def test_retrieval_core_text_search(self, retrieval_core_with_text):
        """Test RetrievalCore text search functionality."""
        response = await retrieval_core_with_text.search(
            query="python development",
            limit=5,
            search_mode="text"
        )
        
        assert len(response.results) >= 1
        assert response.backends_used == ["text"]
        assert "text" in response.backend_timings
        
        # Should find Python document
        python_results = [r for r in response.results if "python" in r.text.lower()]
        assert len(python_results) >= 1
    
    @pytest.mark.asyncio
    async def test_retrieval_core_health_check_with_text(self, retrieval_core_with_text):
        """Test RetrievalCore health check including text backend."""
        health = await retrieval_core_with_text.health_check()
        
        assert health["overall_status"] in ["healthy", "degraded"]
        assert "text" in health["backends"]
        assert health["backends"]["text"]["status"] == "healthy"


class TestTextSearchEdgeCases:
    """Test edge cases in text search integration."""
    
    @pytest.mark.asyncio
    async def test_large_result_set_text_search(self):
        """Test text search with large number of documents."""
        text_backend = TextSearchBackend()
        dispatcher = QueryDispatcher()
        dispatcher.register_backend("text", text_backend)
        
        # Index many documents
        for i in range(100):
            await text_backend.index_document(
                f"doc{i}",
                f"Document {i} about programming and software development topic {i % 10}"
            )
        
        options = SearchOptions(limit=50)
        response = await dispatcher.dispatch_query(
            query="programming software",
            search_mode=SearchMode.TEXT,
            options=options
        )
        
        assert response.success is True
        assert len(response.results) <= 50  # Should respect limit
        assert response.total_count >= 50  # Should have many matches
        
        # Results should still be ranked
        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_concurrent_text_searches(self):
        """Test concurrent text search operations."""
        text_backend = TextSearchBackend()
        dispatcher = QueryDispatcher()
        dispatcher.register_backend("text", text_backend)
        
        # Index test documents
        for i in range(10):
            await text_backend.index_document(f"doc{i}", f"Content {i} with various topics")
        
        # Run multiple searches concurrently
        search_tasks = []
        queries = ["content", "topics", "various", "doc"]
        
        for query in queries:
            task = dispatcher.dispatch_query(
                query=query,
                search_mode=SearchMode.TEXT,
                options=SearchOptions(limit=5)
            )
            search_tasks.append(task)
        
        results = await asyncio.gather(*search_tasks)
        
        # All searches should succeed
        for response in results:
            assert response.success is True
            assert "text" in response.backends_used
    
    @pytest.mark.asyncio
    async def test_text_search_unicode_handling(self):
        """Test text search with Unicode content."""
        text_backend = TextSearchBackend()
        dispatcher = QueryDispatcher()
        dispatcher.register_backend("text", text_backend)
        
        # Index documents with Unicode content
        unicode_docs = [
            ("doc1", "Café programming with résumé parsing"),
            ("doc2", "Naïve Bayes algorithm for text classification"),
            ("doc3", "机器学习 and artificial intelligence")
        ]
        
        for doc_id, text in unicode_docs:
            await text_backend.index_document(doc_id, text)
        
        # Search with Unicode query
        options = SearchOptions(limit=10)
        response = await dispatcher.dispatch_query(
            query="café programming",
            search_mode=SearchMode.TEXT,
            options=options
        )
        
        # Should handle Unicode gracefully (may or may not match depending on tokenization)
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_text_search_memory_usage(self):
        """Test memory efficiency of text search backend."""
        text_backend = TextSearchBackend()
        
        # Index moderate number of documents
        for i in range(200):
            await text_backend.index_document(
                f"doc{i}",
                f"Short document {i} with content about topic {i % 20}"
            )
        
        # Get index statistics
        stats = text_backend.get_index_statistics()
        
        assert stats["document_count"] == 200
        assert stats["vocabulary_size"] > 0
        assert stats["average_document_length"] > 0
        
        # Search should still be reasonably fast
        import time
        start_time = time.time()
        
        options = SearchOptions(limit=20)
        await text_backend.search("document topic", options)
        
        search_time = time.time() - start_time
        assert search_time < 2.0  # Should complete within 2 seconds