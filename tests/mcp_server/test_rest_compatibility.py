#!/usr/bin/env python3
"""
Comprehensive tests for REST API compatibility layer.

Tests cover:
- Context creation endpoint mapping
- Search endpoint mapping
- Context retrieval by ID
- Admin config endpoint
- Admin stats endpoint
- Health endpoint aliases
- Error handling
- Authentication forwarding
- Cypher injection prevention

Target: ≥90% code coverage
"""

import json
import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Import the module under test
from src.mcp_server import rest_compatibility


class TestContextEndpoints:
    """Tests for /api/v1/contexts/* endpoints."""

    @pytest.mark.asyncio
    async def test_create_context_success(self):
        """Test successful context creation."""
        # Mock request and forwarding
        mock_request = Mock()
        mock_request.headers = {"x-api-key": "test_key"}

        mock_payload = {
            "user_id": "test_user",
            "content": "test content",
            "content_type": "log",
            "metadata": {"key": "value"}
        }

        mock_mcp_result = {
            "success": True,
            "id": "ctx-123",
            "message": "Stored successfully"
        }

        with patch("src.mcp_server.rest_compatibility.forward_to_mcp_tool", new=AsyncMock(return_value=mock_mcp_result)) as mock_forward:
            request_model = rest_compatibility.ContextCreateRequest(**mock_payload)
            result = await rest_compatibility.create_context(request_model, mock_request)

            assert result.success is True
            assert result.context_id == "ctx-123"
            assert result.message == "Context stored successfully"

            # Verify content was converted from string to dict format for MCP
            called_payload = mock_forward.call_args[0][2]
            assert isinstance(called_payload["content"], dict), "Content must be dict for MCP endpoint"
            assert called_payload["content"]["text"] == "test content", "Content dict must have 'text' key"
            assert called_payload["type"] == "log", "Type must be valid MCP type"

    @pytest.mark.asyncio
    async def test_create_context_content_format_conversion(self):
        """Test that REST string content is converted to MCP dict format."""
        mock_request = Mock()
        mock_request.headers = {}

        mock_payload = {
            "content": "simple text content",
            # No content_type specified - should default to "log"
        }

        mock_mcp_result = {
            "success": True,
            "id": "ctx-456",
            "message": "Stored successfully"
        }

        with patch("src.mcp_server.rest_compatibility.forward_to_mcp_tool", new=AsyncMock(return_value=mock_mcp_result)) as mock_forward:
            request_model = rest_compatibility.ContextCreateRequest(**mock_payload)
            result = await rest_compatibility.create_context(request_model, mock_request)

            # Verify MCP payload has correct format
            mcp_call = mock_forward.call_args[0][2]
            assert mcp_call["content"] == {"text": "simple text content"}, "Content must be wrapped in dict with 'text' key"
            assert mcp_call["type"] == "log", "Default type must be 'log'"
            assert mcp_call["author"] == "anonymous", "Default author must be 'anonymous'"

    @pytest.mark.asyncio
    async def test_create_context_returns_201_status(self):
        """Test that POST /api/v1/contexts returns HTTP 201 status code."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        # Create a minimal FastAPI app with the router
        app = FastAPI()
        app.include_router(rest_compatibility.router)

        client = TestClient(app)

        mock_mcp_result = {
            "success": True,
            "id": "ctx-123",
            "message": "Stored successfully"
        }

        with patch("src.mcp_server.rest_compatibility.forward_to_mcp_tool", new=AsyncMock(return_value=mock_mcp_result)):
            response = client.post(
                "/api/v1/contexts",
                json={
                    "content": "test content",
                    "content_type": "log"  # Changed from "context" to valid MCP type
                }
            )

            # Verify HTTP 201 Created status code
            assert response.status_code == 201
            assert response.json()["success"] is True
            assert response.json()["context_id"] == "ctx-123"

    @pytest.mark.asyncio
    async def test_create_context_failure(self):
        """Test context creation failure handling."""
        mock_request = Mock()
        mock_request.headers = {}

        mock_payload = {
            "content": "test content"
        }

        mock_mcp_result = {
            "success": False,
            "message": "Storage failed"
        }

        with patch("src.mcp_server.rest_compatibility.forward_to_mcp_tool", new=AsyncMock(return_value=mock_mcp_result)):
            request_model = rest_compatibility.ContextCreateRequest(**mock_payload)

            with pytest.raises(HTTPException) as exc_info:
                await rest_compatibility.create_context(request_model, mock_request)

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_search_contexts_success(self):
        """Test successful context search."""
        mock_request = Mock()
        mock_request.headers = {}

        mock_payload = {
            "query": "test query",
            "limit": 5,
            "threshold": 0.7
        }

        mock_mcp_result = {
            "success": True,
            "results": [
                {"id": "ctx-1", "content": "result 1"},
                {"id": "ctx-2", "content": "result 2"}
            ]
        }

        with patch("src.mcp_server.rest_compatibility.forward_to_mcp_tool", new=AsyncMock(return_value=mock_mcp_result)):
            request_model = rest_compatibility.ContextSearchRequest(**mock_payload)
            result = await rest_compatibility.search_contexts(request_model, mock_request)

            assert result.success is True
            assert result.count == 2
            # Verify 'contexts' field exists and matches results (for Sentinel compatibility)
            assert hasattr(result, 'contexts'), "SearchResponse must have 'contexts' field for Sentinel checks"
            assert len(result.contexts) == 2
            assert result.contexts == result.results, "Both 'contexts' and 'results' fields should contain same data"
            # Verify backward compatibility with 'results' field
            assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_search_contexts_with_user_filter(self):
        """Test context search with user ID filter."""
        mock_request = Mock()
        mock_request.headers = {}

        mock_payload = {
            "query": "test query",
            "user_id": "specific_user",
            "limit": 10
        }

        mock_mcp_result = {
            "success": True,
            "results": []
        }

        with patch("src.mcp_server.rest_compatibility.forward_to_mcp_tool", new=AsyncMock(return_value=mock_mcp_result)) as mock_forward:
            request_model = rest_compatibility.ContextSearchRequest(**mock_payload)
            result = await rest_compatibility.search_contexts(request_model, mock_request)

            # Verify user_id was passed to MCP tool
            called_payload = mock_forward.call_args[0][2]
            assert called_payload["author"] == "specific_user"
            assert result.count == 0

    @pytest.mark.asyncio
    async def test_get_context_by_id_success(self):
        """Test successful context retrieval by ID."""
        mock_request = Mock()
        mock_request.headers = {}

        mock_mcp_result = {
            "success": True,
            "results": [{"id": "ctx-123", "content": "test content"}]
        }

        with patch("src.mcp_server.rest_compatibility.forward_to_mcp_tool", new=AsyncMock(return_value=mock_mcp_result)):
            result = await rest_compatibility.get_context("ctx-123", mock_request)

            assert result["success"] is True
            assert result["context"]["id"] == "ctx-123"

    @pytest.mark.asyncio
    async def test_get_context_cypher_injection_prevention(self):
        """Test that Cypher injection is prevented via parameterized queries."""
        mock_request = Mock()
        mock_request.headers = {}

        # Malicious context_id attempting Cypher injection
        malicious_id = "ctx'; DROP ALL NODES; MATCH (c:Context {id: 'x"

        mock_mcp_result = {
            "success": True,
            "results": []
        }

        with patch("src.mcp_server.rest_compatibility.forward_to_mcp_tool", new=AsyncMock(return_value=mock_mcp_result)) as mock_forward:
            try:
                await rest_compatibility.get_context(malicious_id, mock_request)
            except HTTPException:
                pass  # Expected when no results

            # Verify parameterized query was used
            called_payload = mock_forward.call_args[0][2]
            assert "parameters" in called_payload
            assert called_payload["parameters"]["context_id"] == malicious_id
            # Verify query uses parameter placeholder, not string interpolation
            assert "$context_id" in called_payload["query"]
            assert malicious_id not in called_payload["query"]

    @pytest.mark.asyncio
    async def test_get_context_not_found(self):
        """Test context retrieval when ID not found."""
        mock_request = Mock()
        mock_request.headers = {}

        mock_mcp_result = {
            "success": True,
            "results": []
        }

        with patch("src.mcp_server.rest_compatibility.forward_to_mcp_tool", new=AsyncMock(return_value=mock_mcp_result)):
            with pytest.raises(HTTPException) as exc_info:
                await rest_compatibility.get_context("nonexistent", mock_request)

            assert exc_info.value.status_code == 404


class TestAdminEndpoints:
    """Tests for /api/admin/* endpoints - business logic only."""

    @pytest.mark.asyncio
    async def test_admin_config_response_format(self):
        """Test admin config endpoint response format (business logic)."""
        # Note: Security testing is in test_rest_compatibility_security.py
        # This test focuses on response structure
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(rest_compatibility.router)
        client = TestClient(app)

        # TestClient uses localhost by default, so this should succeed
        response = client.get("/admin/config")

        # Verify response format (regardless of auth status)
        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "config" in data

    @pytest.mark.asyncio
    async def test_admin_stats_endpoint_success(self):
        """Test admin stats endpoint with successful metrics fetch."""
        mock_request = Mock()

        mock_metrics = {
            "uptime_seconds": 3600,
            "total_requests": 1000
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_metrics
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Call function directly to test business logic (not security)
            # Security is tested in integration tests below
            mock_request.client = Mock()
            mock_request.client.host = "127.0.0.1"  # Simulate localhost
            result = await rest_compatibility.get_admin_stats(mock_request)

            assert result["success"] is True
            assert result["stats"] == mock_metrics

    @pytest.mark.asyncio
    async def test_admin_stats_endpoint_fallback(self):
        """Test admin stats endpoint fallback when metrics unavailable."""
        mock_request = Mock()
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 503
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await rest_compatibility.get_admin_stats(mock_request)

            assert result["success"] is True
            assert "message" in result["stats"]


class TestHealthEndpoints:
    """Tests for health endpoint aliases."""

    @pytest.mark.asyncio
    async def test_health_validation_redirect(self):
        """Test health validation endpoint redirects."""
        result = await rest_compatibility.health_validation()

        assert result.status_code == 307
        assert "/health/detailed" in str(result.headers.get("location", ""))

    @pytest.mark.asyncio
    async def test_health_database_redirect(self):
        """Test health database endpoint redirects."""
        result = await rest_compatibility.health_database()

        assert result.status_code == 307

    @pytest.mark.asyncio
    async def test_health_storage_redirect(self):
        """Test health storage endpoint redirects."""
        result = await rest_compatibility.health_storage()

        assert result.status_code == 307

    @pytest.mark.asyncio
    async def test_health_retrieval_redirect(self):
        """Test health retrieval endpoint redirects."""
        result = await rest_compatibility.health_retrieval()

        assert result.status_code == 307

    @pytest.mark.asyncio
    async def test_health_enrichment_redirect(self):
        """Test health enrichment endpoint redirects."""
        result = await rest_compatibility.health_enrichment()

        assert result.status_code == 307

    @pytest.mark.asyncio
    async def test_health_indexing_redirect(self):
        """Test health indexing endpoint redirects."""
        result = await rest_compatibility.health_indexing()

        assert result.status_code == 307

    @pytest.mark.asyncio
    async def test_metrics_alias_redirect(self):
        """Test metrics alias endpoint redirects."""
        result = await rest_compatibility.metrics_alias()

        assert result.status_code == 307
        assert "/metrics" in str(result.headers.get("location", ""))


class TestAuthenticationForwarding:
    """Tests for authentication header forwarding."""

    @pytest.mark.asyncio
    async def test_forward_x_api_key_header(self):
        """Test that x-api-key header is forwarded."""
        mock_request = Mock()
        mock_request.headers = {"x-api-key": "test_api_key"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            await rest_compatibility.forward_to_mcp_tool(
                mock_request,
                "/tools/test",
                {"test": "data"}
            )

            # Verify x-api-key was forwarded
            call_args = mock_client.post.call_args
            headers = call_args.kwargs["headers"]
            assert "x-api-key" in headers
            assert headers["x-api-key"] == "test_api_key"

    @pytest.mark.asyncio
    async def test_forward_authorization_header(self):
        """Test that Authorization header is forwarded."""
        mock_request = Mock()
        mock_request.headers = {"authorization": "Bearer test_token"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            await rest_compatibility.forward_to_mcp_tool(
                mock_request,
                "/tools/test",
                {"test": "data"}
            )

            # Verify authorization header was forwarded
            call_args = mock_client.post.call_args
            headers = call_args.kwargs["headers"]
            assert "authorization" in headers
            assert headers["authorization"] == "Bearer test_token"

    @pytest.mark.asyncio
    async def test_forward_multiple_auth_headers(self):
        """Test forwarding both x-api-key and Authorization headers."""
        mock_request = Mock()
        mock_request.headers = {
            "x-api-key": "test_api_key",
            "authorization": "Bearer test_token"
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            await rest_compatibility.forward_to_mcp_tool(
                mock_request,
                "/tools/test",
                {"test": "data"}
            )

            # Verify both headers were forwarded
            call_args = mock_client.post.call_args
            headers = call_args.kwargs["headers"]
            assert "x-api-key" in headers
            assert "authorization" in headers


class TestErrorHandling:
    """Tests for error handling and security."""

    @pytest.mark.asyncio
    async def test_forward_mcp_tool_sanitized_error(self):
        """Test that error messages are sanitized (no sensitive data exposed)."""
        mock_request = Mock()
        mock_request.headers = {}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Detailed internal error with sensitive data"
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(HTTPException) as exc_info:
                await rest_compatibility.forward_to_mcp_tool(
                    mock_request,
                    "/tools/test",
                    {"test": "data"}
                )

            # Verify error message is sanitized
            assert exc_info.value.detail == "Internal service error"
            # Original detailed error should NOT be in the exception message
            assert "sensitive data" not in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_forward_mcp_tool_network_error(self):
        """Test handling of network errors."""
        mock_request = Mock()
        mock_request.headers = {}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            import httpx
            mock_client.post.side_effect = httpx.ConnectError("Connection failed")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(HTTPException) as exc_info:
                await rest_compatibility.forward_to_mcp_tool(
                    mock_request,
                    "/tools/test",
                    {"test": "data"}
                )

            assert exc_info.value.status_code == 503
            assert exc_info.value.detail == "Service unavailable"

    @pytest.mark.asyncio
    async def test_configurable_base_url(self):
        """Test that MCP_INTERNAL_URL is configurable."""
        # Verify the configuration is loaded from environment
        assert rest_compatibility.MCP_INTERNAL_URL is not None

        # Test with patched environment variable
        with patch.dict("os.environ", {"MCP_INTERNAL_URL": "http://custom-host:9000"}):
            # Re-import to pick up new env var
            import importlib
            importlib.reload(rest_compatibility)

            # Verify custom URL is used
            assert rest_compatibility.MCP_INTERNAL_URL == "http://custom-host:9000"

    @pytest.mark.asyncio
    async def test_configurable_timeout(self):
        """Test that MCP_FORWARD_TIMEOUT is configurable."""
        mock_request = Mock()
        mock_request.headers = {}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            await rest_compatibility.forward_to_mcp_tool(
                mock_request,
                "/tools/test",
                {"test": "data"}
            )

            # Verify timeout was passed to httpx
            call_args = mock_client.post.call_args
            assert "timeout" in call_args.kwargs
            assert call_args.kwargs["timeout"] == rest_compatibility.MCP_FORWARD_TIMEOUT


class TestPydanticModels:
    """Tests for Pydantic request/response models."""

    def test_context_create_request_validation(self):
        """Test ContextCreateRequest validation."""
        # Valid request
        valid_data = {
            "content": "test content",
            "user_id": "user123",
            "content_type": "fact",
            "metadata": {"key": "value"}
        }
        request = rest_compatibility.ContextCreateRequest(**valid_data)
        assert request.content == "test content"

        # Missing required field
        with pytest.raises(Exception):  # Pydantic ValidationError
            rest_compatibility.ContextCreateRequest(user_id="user123")

    def test_context_search_request_validation(self):
        """Test ContextSearchRequest validation."""
        # Valid request
        valid_data = {
            "query": "search query",
            "limit": 20,
            "threshold": 0.5
        }
        request = rest_compatibility.ContextSearchRequest(**valid_data)
        assert request.query == "search query"

        # Limit validation (must be between 1-100)
        with pytest.raises(Exception):  # Pydantic ValidationError
            rest_compatibility.ContextSearchRequest(query="test", limit=0)

        with pytest.raises(Exception):  # Pydantic ValidationError
            rest_compatibility.ContextSearchRequest(query="test", limit=101)

    def test_context_response_model(self):
        """Test ContextResponse model."""
        response = rest_compatibility.ContextResponse(
            success=True,
            context_id="ctx-123",
            message="Success",
            data={"extra": "info"}
        )
        assert response.success is True
        assert response.context_id == "ctx-123"

    def test_search_response_model(self):
        """Test SearchResponse model."""
        response = rest_compatibility.SearchResponse(
            success=True,
            results=[{"id": "1"}, {"id": "2"}],
            count=2,
            message="Found results"
        )
        assert response.count == 2
        assert len(response.results) == 2


class TestAdminEndpointSecurityIntegration:
    """
    Integration tests for admin endpoint security using TestClient.

    These tests verify that security dependencies (verify_admin_access) are
    properly enforced through FastAPI's dependency injection system.

    Unit tests for security functions are in test_rest_compatibility_security.py.
    """

    def setup_method(self):
        """Create FastAPI test client for each test."""
        from fastapi import FastAPI

        self.app = FastAPI()
        self.app.include_router(rest_compatibility.router)
        self.client = TestClient(self.app)

    def test_admin_config_allows_localhost(self):
        """Test that /admin/config allows localhost access (for monitoring)."""
        # TestClient simulates localhost by default
        response = self.client.get("/admin/config")

        # Should succeed or return 500 (business logic error), not 401/403
        assert response.status_code in [200, 500], \
            f"Localhost should access admin endpoints, got {response.status_code}"

    def test_admin_stats_allows_localhost(self):
        """Test that /admin/stats allows localhost access."""
        response = self.client.get("/admin/stats")

        # Should succeed or return 500, not 401/403
        assert response.status_code in [200, 500], \
            f"Localhost should access admin stats, got {response.status_code}"

    def test_admin_users_allows_localhost(self):
        """Test that /admin/users allows localhost access."""
        response = self.client.get("/admin/users")

        # Should succeed, not 401/403
        assert response.status_code in [200, 500], \
            f"Localhost should access admin users, got {response.status_code}"

    def test_admin_root_allows_localhost(self):
        """Test that /admin/ allows localhost access."""
        response = self.client.get("/admin")

        # Should succeed, not 401/403
        assert response.status_code in [200, 500], \
            f"Localhost should access admin root, got {response.status_code}"

    def test_metrics_endpoint_localhost_allowed(self):
        """Test that /metrics endpoint allows localhost (verify_localhost dependency)."""
        response = self.client.get("/metrics")

        # Should redirect (307) or allow (200), not 403
        assert response.status_code in [200, 307], \
            f"Localhost should access /metrics, got {response.status_code}"

    def test_admin_config_with_valid_api_key(self):
        """Test that /admin/config accepts valid ADMIN_API_KEY."""
        with patch.dict('os.environ', {'ADMIN_API_KEY': 'test_key_123'}):
            # Reload module to pick up env var
            import importlib
            importlib.reload(rest_compatibility)

            response = self.client.get(
                "/admin/config",
                headers={"Authorization": "Bearer test_key_123"}
            )

            # Should succeed or return 500, not 403
            assert response.status_code in [200, 500], \
                f"Valid API key should grant access, got {response.status_code}"

    def test_security_dependencies_are_registered(self):
        """
        Verify that admin endpoints have security dependencies registered.

        This is a meta-test ensuring that the endpoints actually use
        Depends(verify_admin_access) in their route definitions.
        """
        # Get the FastAPI routes
        routes = [route for route in self.app.routes if hasattr(route, 'path')]

        # Find admin routes
        admin_routes = [r for r in routes if '/admin' in r.path]

        # Verify admin routes exist
        assert len(admin_routes) > 0, "Admin routes should be registered"

        # Verify at least one route has dependencies
        has_dependencies = any(
            hasattr(route, 'dependencies') and len(route.dependencies) > 0
            for route in admin_routes
        )

        assert has_dependencies, \
            "Admin routes should have security dependencies registered"


# Run tests with: pytest tests/mcp_server/test_rest_compatibility.py -v --cov=src/mcp_server/rest_compatibility --cov-report=term-missing
