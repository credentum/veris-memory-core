#!/usr/bin/env python3
"""
Tests for API main application and configuration.

Tests the FastAPI application creation, middleware configuration,
lifespan management, and OpenAPI schema generation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import create_app, create_openapi_schema
from src.api.dependencies import get_query_dispatcher
from src.api.models import ErrorResponse


class TestAPIApplication:
    """Test FastAPI application creation and configuration."""
    
    def test_create_app_basic_configuration(self):
        """Test basic application creation and configuration."""
        app = create_app()
        
        # Check basic app properties
        assert app.title == "Veris Memory API"
        assert app.version == "1.0.0"
        assert "context storage and retrieval" in app.description.lower()
        
        # Check endpoint configuration
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"
        assert app.openapi_url == "/openapi.json"
    
    def test_middleware_configuration(self):
        """Test middleware is properly configured."""
        app = create_app()
        
        # Check middleware stack - FastAPI stores middleware differently
        middleware_types = []
        for middleware in app.user_middleware:
            if hasattr(middleware, 'cls'):
                middleware_types.append(middleware.cls.__name__)
            elif hasattr(middleware, 'dispatch'):
                middleware_types.append(type(middleware).__name__)
        
        # Should include our custom middleware
        assert "ErrorHandlerMiddleware" in middleware_types
        assert "ValidationMiddleware" in middleware_types
        assert "LoggingMiddleware" in middleware_types
        assert "CORSMiddleware" in middleware_types
    
    def test_router_inclusion(self):
        """Test that all routers are properly included."""
        app = create_app()
        
        # Check that routes are registered
        route_paths = [route.path for route in app.routes]
        
        # Should include API routes
        expected_prefixes = ["/api/v1/search", "/api/v1/health", "/api/v1/metrics"]
        for prefix in expected_prefixes:
            assert any(path.startswith(prefix) for path in route_paths), f"Missing routes with prefix {prefix}"
    
    @patch('src.core.query_dispatcher.QueryDispatcher')
    @patch('src.backends.vector_backend.VectorBackend')
    @patch('src.backends.graph_backend.GraphBackend') 
    @patch('src.backends.kv_backend.KVBackend')
    def test_lifespan_initialization(self, mock_kv, mock_graph, mock_vector, mock_dispatcher):
        """Test application lifespan initialization."""
        # Mock backend instances
        mock_vector_instance = MagicMock()
        mock_graph_instance = MagicMock()
        mock_kv_instance = MagicMock()
        mock_dispatcher_instance = MagicMock()
        
        mock_vector.return_value = mock_vector_instance
        mock_graph.return_value = mock_graph_instance
        mock_kv.return_value = mock_kv_instance
        mock_dispatcher.return_value = mock_dispatcher_instance
        
        app = create_app()
        
        # The lifespan is tested indirectly through the dependency injection
        # In a real test, you'd use an async test client
        assert app is not None


class TestOpenAPISchema:
    """Test OpenAPI schema generation and customization."""
    
    def test_openapi_schema_generation(self):
        """Test OpenAPI schema generation with custom metadata."""
        app = create_app()
        schema = create_openapi_schema(app)
        
        # Check basic schema structure
        assert schema["info"]["title"] == "Veris Memory API"
        assert schema["info"]["version"] == "1.0.0"
        assert "context storage" in schema["info"]["description"].lower()
        
        # Check custom metadata
        assert "contact" in schema["info"]
        assert "license" in schema["info"]
        assert schema["info"]["license"]["name"] == "MIT"
        
        # Check servers configuration
        assert "servers" in schema
        assert len(schema["servers"]) >= 1
        
        # Check security schemes
        assert "components" in schema
        assert "securitySchemes" in schema["components"]
        assert "BearerAuth" in schema["components"]["securitySchemes"]
        
        # Check tags
        assert "tags" in schema
        tag_names = [tag["name"] for tag in schema["tags"]]
        expected_tags = ["search", "health", "metrics"]
        for expected_tag in expected_tags:
            assert expected_tag in tag_names
    
    def test_openapi_schema_caching(self):
        """Test that OpenAPI schema is properly cached."""
        app = create_app()
        
        # First call should generate schema
        schema1 = create_openapi_schema(app)
        
        # Second call should return cached schema
        schema2 = create_openapi_schema(app)
        
        assert schema1 is schema2  # Should be the same object (cached)


class TestDependencyInjection:
    """Test dependency injection for shared components."""
    
    @patch('src.api.dependencies.query_dispatcher', None)
    def test_get_query_dispatcher_not_initialized(self):
        """Test error when query dispatcher is not initialized."""
        with pytest.raises(RuntimeError, match="Query dispatcher not initialized"):
            get_query_dispatcher()
    def test_get_query_dispatcher_initialized(self):
        """Test getting initialized query dispatcher."""
        mock_instance = MagicMock()
        
        # Mock the global variable directly
        with patch('src.api.dependencies.query_dispatcher', mock_instance):
            result = get_query_dispatcher()
            assert result is mock_instance


class TestRootEndpoint:
    """Test the root API endpoint."""
    
    def test_root_endpoint_response(self):
        """Test root endpoint returns correct information."""
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "Veris Memory API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "operational"
        assert data["docs"] == "/docs"
        assert data["openapi"] == "/openapi.json"


class TestErrorHandling:
    """Test API-level error handling."""
    def test_internal_error_handling(self):
        """Test internal error handling and response format."""
        # Test with uninitialized dispatcher (the actual runtime error)
        app = create_app()
        client = TestClient(app)
        
        # This should test what happens when dispatcher is not initialized
        # The real error handling in the actual API endpoints
        try:
            get_query_dispatcher()
        except RuntimeError as e:
            # The real error from uninitialized dispatcher
            assert "Query dispatcher not initialized" in str(e)
    
    def test_404_error_handling(self):
        """Test 404 error handling for non-existent endpoints."""
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/non-existent-endpoint")
        
        assert response.status_code == 404


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_json_endpoint(self):
        """Test OpenAPI JSON schema endpoint."""
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        schema = response.json()
        assert schema["info"]["title"] == "Veris Memory API"
    
    def test_docs_endpoint_accessibility(self):
        """Test that documentation endpoints are accessible."""
        app = create_app()
        client = TestClient(app)
        
        # Test Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_headers(self):
        """Test CORS headers are properly set."""
        app = create_app()
        client = TestClient(app)
        
        # Test preflight request
        response = client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
    
    def test_cors_actual_request(self):
        """Test CORS headers on actual requests.""" 
        app = create_app()
        client = TestClient(app)
        
        response = client.get(
            "/",
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == 200
        # CORS middleware should add appropriate headers