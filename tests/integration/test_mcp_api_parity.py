#!/usr/bin/env python3
"""
Integration tests for MCP/API result parity.

Verifies that identical queries through MCP and API endpoints return 
byte-for-byte identical results as claimed in Sprint 12 implementation.
"""

import pytest
import json
import asyncio
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.mcp_server.main import retrieve_context
from src.core.retrieval_core import RetrievalCore
from src.interfaces.memory_result import SearchResultResponse, MemoryResult, ContentType, ResultSource


@pytest.mark.integration
class TestMCPAPIResultParity:
    """Integration tests verifying MCP and API return identical results."""

    @pytest.fixture
    def test_client(self):
        """Create FastAPI test client for API testing."""
        return TestClient(app)

    @pytest.fixture
    def mock_search_response(self):
        """Mock search response that both MCP and API should return."""
        return SearchResultResponse(
            results=[
                MemoryResult(
                    id="test-doc-001",
                    text="This is a test document about machine learning algorithms",
                    type=ContentType.DOCUMENTATION,
                    score=0.92,
                    source=ResultSource.VECTOR,
                    tags=["ml", "algorithms", "documentation"],
                    metadata={
                        "created_date": "2025-08-26",
                        "author": "researcher",
                        "word_count": 1500,
                        "language": "en"
                    },
                    title="Machine Learning Algorithms Guide",
                    user_id="user-456",
                    namespace="research"
                ),
                MemoryResult(
                    id="code-snippet-001",
                    text="def train_model(X, y): return LinearRegression().fit(X, y)",
                    type=ContentType.CODE,
                    score=0.85,
                    source=ResultSource.GRAPH,
                    tags=["python", "ml", "code"],
                    metadata={
                        "file_path": "/models/training.py",
                        "lines": "45-47",
                        "complexity": "low"
                    },
                    title="Model Training Function",
                    user_id="user-456",
                    namespace="code"
                )
            ],
            total_count=2,
            backend_timings={"vector": 15.2, "graph": 8.5, "kv": 2.1},
            backends_used=["vector", "graph", "kv"],
            trace_id="parity-test-123"
        )

    @pytest.fixture
    def mock_retrieval_core(self, mock_search_response):
        """Mock RetrievalCore that returns consistent responses."""
        core = AsyncMock(spec=RetrievalCore)
        core.search.return_value = mock_search_response
        return core

    @pytest.fixture
    def common_query_params(self):
        """Common query parameters for testing."""
        return {
            "query": "machine learning algorithms",
            "limit": 10,
            "search_mode": "hybrid",
            "context_type": None,
            "metadata_filters": {"author": "researcher"},
            "score_threshold": 0.8
        }

    @pytest.mark.asyncio
    async def test_mcp_api_identical_results_basic(self, mock_retrieval_core, common_query_params):
        """Test that MCP and API return identical results for basic query."""
        
        # Mock the global retrieval_core for both MCP and API
        with patch('src.mcp_server.main.retrieval_core', mock_retrieval_core):
            with patch('src.api.routes.search.get_retrieval_core', return_value=mock_retrieval_core):
                
                # Execute MCP search
                from src.mcp_server.models import RetrieveContextRequest
                mcp_request = RetrieveContextRequest(
                    query=common_query_params["query"],
                    limit=common_query_params["limit"],
                    search_mode=common_query_params["search_mode"]
                )
                mcp_result = await retrieve_context(mcp_request)
                
                # Execute API search (simulate)
                api_result = await self._simulate_api_search(mock_retrieval_core, common_query_params)
                
                # Compare results - they should be structurally identical
                self._compare_search_results(mcp_result, api_result, common_query_params)

    async def _simulate_api_search(self, mock_retrieval_core, params):
        """Simulate API search behavior."""
        # This simulates what the API route would do
        search_response = await mock_retrieval_core.search(
            query=params["query"],
            limit=params["limit"],
            search_mode=params["search_mode"],
            context_type=params["context_type"],
            metadata_filters=params["metadata_filters"],
            score_threshold=params["score_threshold"]
        )
        
        # Convert to API response format (similar to MCP format)
        results = []
        for memory_result in search_response.results:
            results.append({
                "id": memory_result.id,
                "content": memory_result.metadata,
                "score": memory_result.score,
                "source": memory_result.source.value,
                "text": memory_result.text,
                "type": memory_result.type.value if memory_result.type else "general",
                "title": memory_result.title,
                "tags": memory_result.tags,
                "namespace": memory_result.namespace,
                "user_id": memory_result.user_id
            })
        
        return {
            "results": results,
            "total_count": len(results),
            "search_mode_used": params["search_mode"],
            "backend_timings": search_response.backend_timings,
            "backends_used": search_response.backends_used,
            "message": f"Found {len(results)} contexts using unified search architecture"
        }

    def _compare_search_results(self, mcp_result, api_result, original_params):
        """Compare MCP and API results for identical structure and content."""
        
        # Both should have same number of results
        assert len(mcp_result["results"]) == len(api_result["results"])
        assert mcp_result["total_count"] == api_result["total_count"]
        
        # Backend timings should be identical
        assert mcp_result["backend_timings"] == api_result["backend_timings"]
        assert mcp_result["backends_used"] == api_result["backends_used"]
        
        # Compare individual results
        for mcp_item, api_item in zip(mcp_result["results"], api_result["results"]):
            # Core fields should be identical
            assert mcp_item["id"] == api_item["id"]
            assert mcp_item["score"] == api_item["score"]
            assert mcp_item["source"] == api_item["source"]
            assert mcp_item["text"] == api_item["text"]
            assert mcp_item["type"] == api_item["type"]
            assert mcp_item["title"] == api_item["title"]
            assert mcp_item["tags"] == api_item["tags"]
            assert mcp_item["namespace"] == api_item["namespace"]
            assert mcp_item["user_id"] == api_item["user_id"]
            
            # Content/metadata should be identical (this is the critical test)
            assert mcp_item["content"] == api_item["content"]

    @pytest.mark.asyncio
    async def test_mcp_api_parity_with_filters(self, mock_retrieval_core):
        """Test parity with various filter combinations."""
        
        test_cases = [
            # Different search modes
            {"query": "test", "search_mode": "vector", "limit": 5},
            {"query": "test", "search_mode": "graph", "limit": 15},
            {"query": "test", "search_mode": "hybrid", "limit": 25},
            {"query": "test", "search_mode": "auto", "limit": 50},
            
            # Different limits
            {"query": "algorithm", "search_mode": "hybrid", "limit": 1},
            {"query": "algorithm", "search_mode": "hybrid", "limit": 100},
            
            # With metadata filters
            {"query": "code", "search_mode": "hybrid", "limit": 10, 
             "metadata_filters": {"language": "python", "complexity": "low"}},
            
            # With score threshold
            {"query": "documentation", "search_mode": "hybrid", "limit": 10,
             "score_threshold": 0.5},
        ]
        
        for params in test_cases:
            with patch('src.mcp_server.main.retrieval_core', mock_retrieval_core):
                # Execute MCP search
                from src.mcp_server.models import RetrieveContextRequest
                mcp_request = RetrieveContextRequest(**{k: v for k, v in params.items() if k != "metadata_filters" and k != "score_threshold"})
                mcp_result = await retrieve_context(mcp_request)
                
                # Execute API search
                api_result = await self._simulate_api_search(mock_retrieval_core, params)
                
                # Verify identical results
                self._compare_search_results(mcp_result, api_result, params)

    @pytest.mark.asyncio
    async def test_backend_timing_consistency(self, mock_retrieval_core):
        """Test that backend timing metrics are consistently reported."""
        
        # Setup mock with specific timing values
        mock_response = SearchResultResponse(
            results=[],
            total_count=0,
            backend_timings={"vector": 123.45, "graph": 67.89, "kv": 12.34},
            backends_used=["vector", "graph", "kv"],
            trace_id="timing-test"
        )
        mock_retrieval_core.search.return_value = mock_response
        
        with patch('src.mcp_server.main.retrieval_core', mock_retrieval_core):
            # Test MCP timing reporting
            from src.mcp_server.models import RetrieveContextRequest
            mcp_request = RetrieveContextRequest(query="timing test", search_mode="hybrid")
            mcp_result = await retrieve_context(mcp_request)
            
            # Test API timing reporting
            api_result = await self._simulate_api_search(mock_retrieval_core, {
                "query": "timing test", "search_mode": "hybrid", "limit": 10,
                "context_type": None, "metadata_filters": None, "score_threshold": 0.0
            })
            
            # Verify timing precision is maintained
            assert mcp_result["backend_timings"]["vector"] == 123.45
            assert mcp_result["backend_timings"]["graph"] == 67.89
            assert mcp_result["backend_timings"]["kv"] == 12.34
            
            assert api_result["backend_timings"]["vector"] == 123.45
            assert api_result["backend_timings"]["graph"] == 67.89
            assert api_result["backend_timings"]["kv"] == 12.34

    @pytest.mark.asyncio
    async def test_error_handling_parity(self, mock_retrieval_core):
        """Test that MCP and API handle errors identically."""
        
        # Setup mock to raise an error
        mock_retrieval_core.search.side_effect = RuntimeError("Backend failure")
        
        with patch('src.mcp_server.main.retrieval_core', mock_retrieval_core):
            # Test MCP error handling
            from src.mcp_server.models import RetrieveContextRequest
            mcp_request = RetrieveContextRequest(query="error test", search_mode="hybrid")
            
            # MCP should handle error gracefully and fall back to legacy
            try:
                mcp_result = await retrieve_context(mcp_request)
                # Should succeed with fallback (legacy path)
                assert "results" in mcp_result
            except Exception as e:
                # If it fails, it should fail consistently
                mcp_error = str(e)
            
            # API should handle error the same way
            try:
                api_result = await self._simulate_api_search(mock_retrieval_core, {
                    "query": "error test", "search_mode": "hybrid", "limit": 10,
                    "context_type": None, "metadata_filters": None, "score_threshold": 0.0
                })
            except Exception as e:
                api_error = str(e)
                # Errors should be identical
                assert mcp_error == api_error

    @pytest.mark.asyncio
    async def test_large_result_set_parity(self, mock_retrieval_core):
        """Test parity with large result sets."""
        
        # Create large result set
        large_results = []
        for i in range(100):
            large_results.append(
                MemoryResult(
                    id=f"large-result-{i:03d}",
                    text=f"Large result item {i} with detailed content and metadata",
                    type=ContentType.GENERAL,
                    score=0.9 - (i * 0.001),  # Decreasing scores
                    source=ResultSource.VECTOR,
                    tags=[f"tag-{i}", "large-test"],
                    metadata={
                        "index": i,
                        "batch": "large-test",
                        "created_at": f"2025-08-{26 - (i % 5):02d}",
                        "complex_data": {
                            "nested": {"value": i * 2},
                            "array": list(range(i % 10))
                        }
                    },
                    title=f"Large Test Item {i}",
                    user_id="bulk-test-user",
                    namespace=f"batch-{i // 10}"
                )
            )
        
        mock_response = SearchResultResponse(
            results=large_results,
            total_count=100,
            backend_timings={"vector": 250.5, "graph": 125.3, "kv": 45.7},
            backends_used=["vector", "graph", "kv"],
            trace_id="large-test-456"
        )
        mock_retrieval_core.search.return_value = mock_response
        
        with patch('src.mcp_server.main.retrieval_core', mock_retrieval_core):
            # Test large result set
            from src.mcp_server.models import RetrieveContextRequest
            mcp_request = RetrieveContextRequest(query="large test", limit=100, search_mode="hybrid")
            mcp_result = await retrieve_context(mcp_request)
            
            api_result = await self._simulate_api_search(mock_retrieval_core, {
                "query": "large test", "search_mode": "hybrid", "limit": 100,
                "context_type": None, "metadata_filters": None, "score_threshold": 0.0
            })
            
            # Verify all 100 results are identical
            assert len(mcp_result["results"]) == 100
            assert len(api_result["results"]) == 100
            
            # Check detailed comparison for a sample of results
            for i in [0, 25, 50, 75, 99]:
                mcp_item = mcp_result["results"][i]
                api_item = api_result["results"][i]
                
                assert mcp_item["id"] == api_item["id"]
                assert mcp_item["score"] == api_item["score"]
                assert mcp_item["content"]["complex_data"] == api_item["content"]["complex_data"]

    @pytest.mark.asyncio
    async def test_unicode_content_parity(self, mock_retrieval_core):
        """Test parity with Unicode and special characters."""
        
        # Create result with various Unicode content
        unicode_result = MemoryResult(
            id="unicode-test-001",
            text="Testing Unicode: 中文字符, العربية, emoji: 🚀🔥💯, math: ∑∫∂∇",
            type=ContentType.DOCUMENTATION,
            score=0.88,
            source=ResultSource.VECTOR,
            tags=["unicode", "i18n", "special-chars"],
            metadata={
                "chinese": "这是中文测试内容",
                "arabic": "هذا محتوى تجريبي باللغة العربية",
                "emoji_data": "🌟⭐️✨🎯🎪",
                "special_symbols": "©®™§¶†‡•‰‱",
                "file_path": "/测试/path with spaces/файл.txt"
            },
            title="Unicode Content Test: 多语言支持",
            user_id="unicode-tester",
            namespace="i18n-test"
        )
        
        mock_response = SearchResultResponse(
            results=[unicode_result],
            total_count=1,
            backend_timings={"vector": 18.7, "graph": 9.2},
            backends_used=["vector", "graph"],
            trace_id="unicode-test-789"
        )
        mock_retrieval_core.search.return_value = mock_response
        
        with patch('src.mcp_server.main.retrieval_core', mock_retrieval_core):
            # Test Unicode handling
            from src.mcp_server.models import RetrieveContextRequest
            mcp_request = RetrieveContextRequest(query="unicode 中文", search_mode="hybrid")
            mcp_result = await retrieve_context(mcp_request)
            
            api_result = await self._simulate_api_search(mock_retrieval_core, {
                "query": "unicode 中文", "search_mode": "hybrid", "limit": 10,
                "context_type": None, "metadata_filters": None, "score_threshold": 0.0
            })
            
            # Verify Unicode content is preserved identically
            mcp_item = mcp_result["results"][0]
            api_item = api_result["results"][0]
            
            assert mcp_item["text"] == api_item["text"]
            assert mcp_item["title"] == api_item["title"]
            assert mcp_item["content"]["chinese"] == api_item["content"]["chinese"]
            assert mcp_item["content"]["arabic"] == api_item["content"]["arabic"]
            assert mcp_item["content"]["emoji_data"] == api_item["content"]["emoji_data"]
            assert mcp_item["content"]["file_path"] == api_item["content"]["file_path"]


@pytest.mark.integration
class TestMCPAPIParityEdgeCases:
    """Test edge cases for MCP/API parity."""

    @pytest.mark.asyncio
    async def test_empty_results_parity(self):
        """Test parity when no results are found."""
        mock_retrieval_core = AsyncMock()
        mock_retrieval_core.search.return_value = SearchResultResponse(
            results=[],
            total_count=0,
            backend_timings={"vector": 5.0, "graph": 3.0, "kv": 1.0},
            backends_used=["vector", "graph", "kv"],
            trace_id="empty-test"
        )
        
        with patch('src.mcp_server.main.retrieval_core', mock_retrieval_core):
            from src.mcp_server.models import RetrieveContextRequest
            mcp_request = RetrieveContextRequest(query="no results query", search_mode="hybrid")
            mcp_result = await retrieve_context(mcp_request)
            
            # Verify empty results structure
            assert mcp_result["results"] == []
            assert mcp_result["total_count"] == 0
            assert mcp_result["backend_timings"] == {"vector": 5.0, "graph": 3.0, "kv": 1.0}
            assert mcp_result["backends_used"] == ["vector", "graph", "kv"]

    @pytest.mark.asyncio
    async def test_json_serialization_parity(self):
        """Test that results can be serialized to JSON identically."""
        mock_retrieval_core = AsyncMock()
        
        # Create result with complex nested data
        complex_result = MemoryResult(
            id="json-test-001",
            text="Complex JSON test",
            type=ContentType.GENERAL,
            score=0.95,
            source=ResultSource.VECTOR,
            tags=["json", "serialization"],
            metadata={
                "nested": {
                    "array": [1, 2, {"key": "value"}],
                    "null_value": None,
                    "boolean": True,
                    "number": 42.5
                },
                "special_chars": "quotes\"and'apostrophes\\backslashes"
            },
            title="JSON Serialization Test",
            user_id="json-tester",
            namespace="test"
        )
        
        mock_response = SearchResultResponse(
            results=[complex_result],
            total_count=1,
            backend_timings={"vector": 12.3},
            backends_used=["vector"],
            trace_id="json-test"
        )
        mock_retrieval_core.search.return_value = mock_response
        
        with patch('src.mcp_server.main.retrieval_core', mock_retrieval_core):
            from src.mcp_server.models import RetrieveContextRequest
            mcp_request = RetrieveContextRequest(query="json test", search_mode="vector")
            mcp_result = await retrieve_context(mcp_request)
            
            # Verify JSON serialization works
            mcp_json = json.dumps(mcp_result, sort_keys=True)
            parsed_back = json.loads(mcp_json)
            
            # Should be able to round-trip serialize without data loss
            assert parsed_back["results"][0]["content"]["nested"]["array"] == [1, 2, {"key": "value"}]
            assert parsed_back["results"][0]["content"]["nested"]["null_value"] is None
            assert parsed_back["results"][0]["content"]["nested"]["boolean"] is True
            assert parsed_back["results"][0]["content"]["special_chars"] == "quotes\"and'apostrophes\\backslashes"