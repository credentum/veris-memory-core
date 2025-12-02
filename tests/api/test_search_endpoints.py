#!/usr/bin/env python3
"""
Tests for Search API Endpoints.

Tests the search functionality, parameter validation, error handling,
and response formatting for all search-related endpoints.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import create_app, get_query_dispatcher
from src.api.models import SearchRequest, SearchResponse
from src.core.query_dispatcher import SearchMode, DispatchPolicy
from src.interfaces.memory_result import MemoryResult, ContentType, ResultSource


@pytest.fixture
def mock_query_dispatcher():
    """Create a mock query dispatcher."""
    dispatcher = AsyncMock()
    
    # Mock basic methods
    dispatcher.list_backends.return_value = ["vector", "graph", "kv"]
    dispatcher.get_available_ranking_policies.return_value = ["default", "code_boost", "recency"]
    dispatcher.get_ranking_policy_info.return_value = {
        "description": "Test policy",
        "configuration": {"boost_factor": 2.0}
    }
    dispatcher.get_filter_capabilities.return_value = {
        "time_window_filtering": True,
        "tag_filtering": True,
        "content_type_filtering": True
    }
    dispatcher.get_performance_stats.return_value = {
        "timing_summary": {},
        "registered_backends": ["vector", "graph", "kv"]
    }
    dispatcher.health_check_all_backends.return_value = {
        "vector": {"status": "healthy", "response_time_ms": 25.0},
        "graph": {"status": "healthy", "response_time_ms": 30.0},
        "kv": {"status": "healthy", "response_time_ms": 15.0}
    }
    
    return dispatcher


@pytest.fixture
def sample_search_results():
    """Create sample search results."""
    return [
        MemoryResult(
            id="result_1",
            text="Python function for data processing",
            type=ContentType.CODE,
            score=0.92,
            source=ResultSource.VECTOR,
            timestamp=datetime.now(timezone.utc),
            tags=["python", "function", "data"]
        ),
        MemoryResult(
            id="result_2", 
            text="API documentation for authentication",
            type=ContentType.DOCUMENTATION,
            score=0.85,
            source=ResultSource.GRAPH,
            timestamp=datetime.now(timezone.utc),
            tags=["api", "auth", "docs"]
        )
    ]


@pytest.fixture
def api_client(mock_query_dispatcher):
    """Create test client with mocked dependencies."""
    app = create_app()
    
    # Override dependency
    app.dependency_overrides[get_query_dispatcher] = lambda: mock_query_dispatcher
    
    return TestClient(app)


class TestSearchEndpoint:
    """Test the main search endpoint."""
    
    def test_basic_search_request(self, api_client, mock_query_dispatcher, sample_search_results):
        """Test basic search request and response."""
        # Mock successful search response
        mock_search_result = MagicMock()
        mock_search_result.success = True
        mock_search_result.results = sample_search_results
        mock_search_result.total_count = len(sample_search_results)
        mock_search_result.search_mode_used = "hybrid"
        mock_search_result.response_time_ms = 45.2
        mock_search_result.backend_timings = {"vector": 25.1, "graph": 20.1}
        mock_search_result.backends_used = ["vector", "graph"]
        mock_search_result.trace_id = "test_trace_123"
        
        mock_query_dispatcher.dispatch_query.return_value = mock_search_result
        
        # Make request
        request_data = {
            "query": "python function examples",
            "limit": 10
        }
        
        response = api_client.post("/api/v1/search", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert data["success"] is True
        assert len(data["results"]) == 2
        assert data["total_count"] == 2
        assert data["search_mode_used"] == "hybrid"
        assert data["query"] == "python function examples"
        assert data["response_time_ms"] == 45.2
        
        # Check results structure
        result = data["results"][0]
        assert "id" in result
        assert "text" in result
        assert "type" in result
        assert "score" in result
        assert "source" in result
    
    def test_search_with_advanced_parameters(self, api_client, mock_query_dispatcher, sample_search_results):
        """Test search with advanced filtering and ranking parameters."""
        mock_search_result = MagicMock()
        mock_search_result.success = True
        mock_search_result.results = sample_search_results
        mock_search_result.total_count = len(sample_search_results)
        mock_search_result.search_mode_used = "vector"
        mock_search_result.response_time_ms = 30.5
        mock_search_result.backend_timings = {"vector": 30.5}
        mock_search_result.backends_used = ["vector"]
        mock_search_result.trace_id = "test_trace_456"
        
        mock_query_dispatcher.dispatch_query.return_value = mock_search_result
        
        # Make request with advanced parameters
        request_data = {
            "query": "authentication middleware",
            "search_mode": "vector",
            "dispatch_policy": "parallel",
            "ranking_policy": "code_boost",
            "limit": 20,
            "content_types": ["code", "documentation"],
            "tags": ["python", "security"],
            "min_score": 0.7,
            "time_window": {"hours_ago": 24}
        }
        
        response = api_client.post("/api/v1/search", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["ranking_policy_used"] == "code_boost"
        
        # Verify dispatcher was called with correct parameters
        mock_query_dispatcher.dispatch_query.assert_called_once()
        call_args = mock_query_dispatcher.dispatch_query.call_args
        assert call_args.kwargs["query"] == "authentication middleware"
        assert call_args.kwargs["search_mode"] == SearchMode.VECTOR
        assert call_args.kwargs["ranking_policy"] == "code_boost"
    
    def test_search_validation_errors(self, api_client):
        """Test search request validation errors."""
        # Test empty query
        response = api_client.post("/api/v1/search", json={"query": ""})
        assert response.status_code == 422
        
        # Test invalid limit
        response = api_client.post("/api/v1/search", json={
            "query": "test",
            "limit": 1000  # Exceeds maximum
        })
        assert response.status_code == 422
        
        # Test invalid score range
        response = api_client.post("/api/v1/search", json={
            "query": "test",
            "min_score": 0.8,
            "max_score": 0.6  # max < min
        })
        assert response.status_code == 422
    
    def test_search_backend_error(self, api_client, mock_query_dispatcher):
        """Test search handling of backend errors."""
        # Mock backend error
        mock_query_dispatcher.dispatch_query.side_effect = Exception("Backend connection failed")
        
        request_data = {"query": "test query"}
        response = api_client.post("/api/v1/search", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "Backend connection failed" in data["error"]["message"]


class TestSearchMetadataEndpoints:
    """Test search metadata and configuration endpoints."""
    
    def test_get_search_modes(self, api_client):
        """Test getting available search modes."""
        response = api_client.get("/api/v1/search/modes")
        
        assert response.status_code == 200
        data = response.json()
        
        expected_modes = ["vector", "graph", "kv", "hybrid", "auto"]
        assert all(mode in data for mode in expected_modes)
    
    def test_get_dispatch_policies(self, api_client):
        """Test getting available dispatch policies."""
        response = api_client.get("/api/v1/search/policies")
        
        assert response.status_code == 200
        data = response.json()
        
        expected_policies = ["parallel", "sequential", "fallback", "smart"]
        assert all(policy in data for policy in expected_policies)
    
    def test_get_ranking_policies(self, api_client, mock_query_dispatcher):
        """Test getting available ranking policies."""
        response = api_client.get("/api/v1/search/ranking")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check structure of policy info
        policy = data[0]
        assert "name" in policy
        assert "description" in policy
        assert "configuration" in policy
    
    def test_get_available_backends(self, api_client, mock_query_dispatcher):
        """Test getting available backends."""
        response = api_client.get("/api/v1/search/backends")
        
        assert response.status_code == 200
        data = response.json()
        
        expected_backends = ["vector", "graph", "kv"]
        assert data == expected_backends
    
    def test_get_system_info(self, api_client, mock_query_dispatcher):
        """Test getting comprehensive system information."""
        response = api_client.get("/api/v1/search/system-info")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "version" in data
        assert "backends" in data
        assert "ranking_policies" in data
        assert "filter_capabilities" in data
        assert "rate_limits" in data
        assert "features" in data
        
        # Check structure
        assert isinstance(data["backends"], list)
        assert isinstance(data["ranking_policies"], list)
        assert isinstance(data["filter_capabilities"], dict)
        assert isinstance(data["features"], list)


class TestSearchParameterValidation:
    """Test comprehensive parameter validation."""
    
    def test_time_window_validation(self, api_client):
        """Test time window parameter validation."""
        # Valid time window
        request_data = {
            "query": "test",
            "time_window": {"hours_ago": 24}
        }
        response = api_client.post("/api/v1/search", json=request_data)
        # Should not fail on validation (might fail on execution due to mock)
        
        # Invalid time window - end before start
        request_data = {
            "query": "test", 
            "time_window": {
                "start_time": "2024-01-02T00:00:00Z",
                "end_time": "2024-01-01T00:00:00Z"
            }
        }
        response = api_client.post("/api/v1/search", json=request_data)
        assert response.status_code == 422
    
    def test_filter_criteria_validation(self, api_client):
        """Test filter criteria validation."""
        # Valid filter criteria
        request_data = {
            "query": "test",
            "pre_filters": [
                {
                    "field": "type",
                    "operator": "EQUALS",
                    "value": "code"
                }
            ]
        }
        response = api_client.post("/api/v1/search", json=request_data)
        # Should not fail on validation
        
        # Invalid operator
        request_data = {
            "query": "test",
            "pre_filters": [
                {
                    "field": "type",
                    "operator": "INVALID_OPERATOR", 
                    "value": "code"
                }
            ]
        }
        response = api_client.post("/api/v1/search", json=request_data)
        assert response.status_code == 422
    
    def test_enum_validation(self, api_client):
        """Test enum parameter validation."""
        # Invalid search mode
        response = api_client.post("/api/v1/search", json={
            "query": "test",
            "search_mode": "invalid_mode"
        })
        assert response.status_code == 422
        
        # Invalid dispatch policy
        response = api_client.post("/api/v1/search", json={
            "query": "test", 
            "dispatch_policy": "invalid_policy"
        })
        assert response.status_code == 422
        
        # Invalid content type
        response = api_client.post("/api/v1/search", json={
            "query": "test",
            "content_types": ["invalid_type"]
        })
        assert response.status_code == 422


class TestSearchResponseFormat:
    """Test search response format and structure."""
    
    def test_successful_response_structure(self, api_client, mock_query_dispatcher, sample_search_results):
        """Test structure of successful search response."""
        mock_search_result = MagicMock()
        mock_search_result.success = True
        mock_search_result.results = sample_search_results
        mock_search_result.total_count = len(sample_search_results)
        mock_search_result.search_mode_used = "hybrid"
        mock_search_result.response_time_ms = 45.2
        mock_search_result.backend_timings = {"vector": 25.1}
        mock_search_result.backends_used = ["vector"]
        mock_search_result.trace_id = "test_trace"
        
        mock_query_dispatcher.dispatch_query.return_value = mock_search_result
        
        response = api_client.post("/api/v1/search", json={"query": "test"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields are present
        required_fields = [
            "success", "results", "total_count", "search_mode_used", 
            "query", "response_time_ms", "backend_timings", "backends_used",
            "ranking_policy_used", "filters_applied", "trace_id", "timestamp"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check data types
        assert isinstance(data["success"], bool)
        assert isinstance(data["results"], list)
        assert isinstance(data["total_count"], int)
        assert isinstance(data["response_time_ms"], (int, float))
        assert isinstance(data["backend_timings"], dict)
        assert isinstance(data["backends_used"], list)
    
    def test_result_item_structure(self, api_client, mock_query_dispatcher, sample_search_results):
        """Test structure of individual result items."""
        mock_search_result = MagicMock()
        mock_search_result.success = True
        mock_search_result.results = sample_search_results
        mock_search_result.total_count = len(sample_search_results)
        mock_search_result.search_mode_used = "hybrid"
        mock_search_result.response_time_ms = 45.2
        mock_search_result.backend_timings = {}
        mock_search_result.backends_used = []
        mock_search_result.trace_id = "test_trace"
        
        mock_query_dispatcher.dispatch_query.return_value = mock_search_result
        
        response = api_client.post("/api/v1/search", json={"query": "test"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Check result item structure
        result_item = data["results"][0]
        
        required_result_fields = [
            "id", "text", "type", "score", "source", 
            "timestamp", "tags", "metadata"
        ]
        
        for field in required_result_fields:
            assert field in result_item, f"Missing result field: {field}"
        
        # Check result field types
        assert isinstance(result_item["id"], str)
        assert isinstance(result_item["text"], str)
        assert isinstance(result_item["score"], (int, float))
        assert isinstance(result_item["tags"], list)
        assert isinstance(result_item["metadata"], dict)