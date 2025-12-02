#!/usr/bin/env python3
"""
Comprehensive tests for src/mcp_server/main.py

Tests cover:
- Global embedding model management and fallback behavior
- FastAPI application configuration and middleware
- Health check endpoints with service validation
- MCP tool endpoints (store_context, retrieve_context, query_graph, etc.)
- Dashboard endpoints and WebSocket streaming
- Error handling and exception management
- Service startup and shutdown lifecycle
- Request/response validation
- Cross-cutting concerns like logging and monitoring
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, mock_open, call
import pytest
from fastapi.testclient import TestClient
from fastapi import WebSocket, WebSocketDisconnect
import websockets

# Import the module under test
from src.mcp_server.main import (
    _get_embedding_model,
    _generate_embedding,
    global_exception_handler,
    StoreContextRequest,
    RetrieveContextRequest,
    QueryGraphRequest,
    UpdateScratchpadRequest,
    GetAgentStateRequest,
    startup_event,
    shutdown_event,
    _check_service_with_retries,
    _is_in_startup_grace_period,
    app
)


class TestEmbeddingModel:
    """Test embedding model management and generation."""

    def test_get_embedding_model_unavailable(self):
        """Test _get_embedding_model when sentence-transformers is unavailable."""
        with patch('src.mcp_server.main.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            result = _get_embedding_model()
            assert result is None

    def test_get_embedding_model_available_success(self):
        """Test successful embedding model loading."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        with patch('src.mcp_server.main.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('src.mcp_server.main.SentenceTransformer', return_value=mock_model) as mock_st:
                with patch('src.mcp_server.main._embedding_model', None):
                    result = _get_embedding_model()
                    
                    assert result == mock_model
                    mock_st.assert_called_once_with("all-MiniLM-L6-v2")
                    mock_model.get_sentence_embedding_dimension.assert_called_once()

    def test_get_embedding_model_custom_model_name(self):
        """Test embedding model loading with custom model name."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        
        with patch('src.mcp_server.main.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('src.mcp_server.main.SentenceTransformer', return_value=mock_model) as mock_st:
                with patch('src.mcp_server.main._embedding_model', None):
                    with patch.dict(os.environ, {'EMBEDDING_MODEL': 'custom-model'}):
                        result = _get_embedding_model()
                        
                        assert result == mock_model
                        mock_st.assert_called_once_with("custom-model")

    def test_get_embedding_model_loading_failure(self):
        """Test embedding model loading failure."""
        with patch('src.mcp_server.main.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('src.mcp_server.main.SentenceTransformer', side_effect=Exception("Loading failed")):
                with patch('src.mcp_server.main._embedding_model', None):
                    with patch('src.mcp_server.main.logger') as mock_logger:
                        result = _get_embedding_model()
                        
                        assert result is None
                        mock_logger.error.assert_called_once()

    def test_get_embedding_model_singleton_behavior(self):
        """Test that embedding model is singleton."""
        mock_model = MagicMock()
        
        with patch('src.mcp_server.main.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('src.mcp_server.main.SentenceTransformer', return_value=mock_model) as mock_st:
                with patch('src.mcp_server.main._embedding_model', mock_model):
                    result1 = _get_embedding_model()
                    result2 = _get_embedding_model()
                    
                    assert result1 is result2
                    assert result1 == mock_model
                    # SentenceTransformer should not be called again
                    mock_st.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_embedding_dict_content_with_title(self):
        """Test embedding generation from dict content with title."""
        content = {
            "title": "Test Title",
            "description": "Test Description",
            "body": "Test Body"
        }
        
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        
        with patch('src.mcp_server.main._get_embedding_model', return_value=mock_model):
            result = await _generate_embedding(content)
            
            assert result == [0.1, 0.2, 0.3]
            mock_model.encode.assert_called_once_with("Test Title Test Description Test Body")

    @pytest.mark.asyncio
    async def test_generate_embedding_dict_content_with_content_field(self):
        """Test embedding generation from dict with content field."""
        content = {"content": {"nested": "data"}}
        
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.4, 0.5, 0.6]
        
        with patch('src.mcp_server.main._get_embedding_model', return_value=mock_model):
            result = await _generate_embedding(content)
            
            assert result == [0.4, 0.5, 0.6]
            expected_text = '{"nested": "data"}'
            mock_model.encode.assert_called_once_with(expected_text)

    @pytest.mark.asyncio
    async def test_generate_embedding_dict_content_fallback_json(self):
        """Test embedding generation with fallback to JSON string."""
        content = {"random": "data", "number": 42}
        
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.7, 0.8, 0.9]
        
        with patch('src.mcp_server.main._get_embedding_model', return_value=mock_model):
            result = await _generate_embedding(content)
            
            assert result == [0.7, 0.8, 0.9]
            expected_text = '{"number": 42, "random": "data"}'
            mock_model.encode.assert_called_once_with(expected_text)

    @pytest.mark.asyncio
    async def test_generate_embedding_string_content(self):
        """Test embedding generation from string content."""
        content = "Simple string content"
        
        mock_model = MagicMock()
        mock_model.encode.return_value = [1.0, 1.1, 1.2]
        
        with patch('src.mcp_server.main._get_embedding_model', return_value=mock_model):
            result = await _generate_embedding(content)
            
            assert result == [1.0, 1.1, 1.2]
            mock_model.encode.assert_called_once_with("Simple string content")

    @pytest.mark.asyncio
    async def test_generate_embedding_no_model_available(self):
        """Test embedding generation when no model is available."""
        content = {"test": "data"}
        
        with patch('src.mcp_server.main._get_embedding_model', return_value=None):
            result = await _generate_embedding(content)
            
            assert isinstance(result, list)
            assert len(result) == 384  # Default dimension for hash-based fallback
            assert all(isinstance(x, float) for x in result)


class TestRequestModels:
    """Test Pydantic request models."""

    def test_store_context_request_valid(self):
        """Test valid StoreContextRequest."""
        data = {
            "content": {"title": "Test", "body": "Content"},
            "agent_id": "agent_123",
            "metadata": {"type": "document"}
        }
        
        request = StoreContextRequest(**data)
        
        assert request.content == data["content"]
        assert request.agent_id == "agent_123"
        assert request.metadata == {"type": "document"}

    def test_store_context_request_minimal(self):
        """Test minimal StoreContextRequest."""
        data = {
            "content": {"title": "Test"},
            "agent_id": "agent_123"
        }
        
        request = StoreContextRequest(**data)
        
        assert request.content == data["content"]
        assert request.agent_id == "agent_123"
        assert request.metadata is None

    def test_retrieve_context_request_valid(self):
        """Test valid RetrieveContextRequest."""
        data = {
            "query": "test query",
            "agent_id": "agent_123",
            "limit": 10,
            "similarity_threshold": 0.7
        }
        
        request = RetrieveContextRequest(**data)
        
        assert request.query == "test query"
        assert request.agent_id == "agent_123"
        assert request.limit == 10
        assert request.similarity_threshold == 0.7

    def test_retrieve_context_request_defaults(self):
        """Test RetrieveContextRequest with defaults."""
        data = {
            "query": "test query",
            "agent_id": "agent_123"
        }
        
        request = RetrieveContextRequest(**data)
        
        assert request.query == "test query"
        assert request.agent_id == "agent_123"
        assert request.limit == 10  # Default
        assert request.similarity_threshold == 0.5  # Default

    def test_query_graph_request_valid(self):
        """Test valid QueryGraphRequest."""
        data = {
            "cypher_query": "MATCH (n) RETURN n LIMIT 5",
            "agent_id": "agent_123",
            "parameters": {"param1": "value1"}
        }
        
        request = QueryGraphRequest(**data)
        
        assert request.cypher_query == "MATCH (n) RETURN n LIMIT 5"
        assert request.agent_id == "agent_123"
        assert request.parameters == {"param1": "value1"}

    def test_update_scratchpad_request_valid(self):
        """Test valid UpdateScratchpadRequest."""
        data = {
            "content": "Updated scratchpad content",
            "agent_id": "agent_123",
            "operation": "append"
        }
        
        request = UpdateScratchpadRequest(**data)
        
        assert request.content == "Updated scratchpad content"
        assert request.agent_id == "agent_123"
        assert request.operation == "append"

    def test_get_agent_state_request_valid(self):
        """Test valid GetAgentStateRequest."""
        data = {
            "agent_id": "agent_123",
            "include_scratchpad": True,
            "include_metrics": False
        }
        
        request = GetAgentStateRequest(**data)
        
        assert request.agent_id == "agent_123"
        assert request.include_scratchpad is True
        assert request.include_metrics is False

    def test_get_agent_state_request_defaults(self):
        """Test GetAgentStateRequest with defaults."""
        data = {"agent_id": "agent_123"}
        
        request = GetAgentStateRequest(**data)
        
        assert request.agent_id == "agent_123"
        assert request.include_scratchpad is True  # Default
        assert request.include_metrics is True  # Default


class TestExceptionHandler:
    """Test global exception handler."""

    @pytest.mark.asyncio
    async def test_global_exception_handler_generic_error(self):
        """Test global exception handler with generic error."""
        mock_request = MagicMock()
        mock_request.url.path = "/test/path"
        mock_request.method = "POST"
        
        exception = Exception("Test error")
        
        with patch('src.mcp_server.main.handle_generic_error') as mock_handler:
            mock_handler.return_value = {"error": "handled", "status": "error"}
            
            response = await global_exception_handler(mock_request, exception)
            
            assert response.status_code == 500
            mock_handler.assert_called_once_with(exception, "POST /test/path")

    @pytest.mark.asyncio
    async def test_global_exception_handler_http_exception(self):
        """Test global exception handler with HTTPException."""
        from fastapi import HTTPException
        
        mock_request = MagicMock()
        mock_request.url.path = "/api/endpoint"
        mock_request.method = "GET"
        
        exception = HTTPException(status_code=404, detail="Not found")
        
        response = await global_exception_handler(mock_request, exception)
        
        assert response.status_code == 404
        response_data = json.loads(response.body)
        assert "error" in response_data
        assert "detail" in response_data


class TestStartupShutdown:
    """Test application startup and shutdown lifecycle."""

    @pytest.mark.asyncio
    async def test_startup_event_success(self):
        """Test successful startup event."""
        with patch('src.mcp_server.main.logger') as mock_logger:
            with patch('src.mcp_server.main.validate_all_configs') as mock_validate:
                mock_validate.return_value = True
                
                await startup_event()
                
                mock_logger.info.assert_called()
                mock_validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_event_config_validation_failure(self):
        """Test startup event with config validation failure."""
        with patch('src.mcp_server.main.logger') as mock_logger:
            with patch('src.mcp_server.main.validate_all_configs') as mock_validate:
                mock_validate.side_effect = Exception("Config validation failed")
                
                with pytest.raises(Exception, match="Config validation failed"):
                    await startup_event()
                
                mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_shutdown_event_success(self):
        """Test successful shutdown event."""
        with patch('src.mcp_server.main.logger') as mock_logger:
            await shutdown_event()
            
            mock_logger.info.assert_called_with("MCP server shutdown complete")

    @pytest.mark.asyncio
    async def test_shutdown_event_with_error(self):
        """Test shutdown event with cleanup error."""
        with patch('src.mcp_server.main.logger') as mock_logger:
            # Simulate error during shutdown
            with patch('src.mcp_server.main.asyncio.sleep', side_effect=Exception("Shutdown error")):
                await shutdown_event()
                
                # Should still log completion despite error
                mock_logger.info.assert_called_with("MCP server shutdown complete")


class TestServiceHealthChecks:
    """Test service health check functions."""

    @pytest.mark.asyncio
    async def test_check_service_with_retries_success_first_try(self):
        """Test service check success on first try."""
        mock_service = AsyncMock()
        mock_service.ping.return_value = True
        
        result = await _check_service_with_retries(
            mock_service, "test_service", check_method="ping"
        )
        
        assert result is True
        mock_service.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_service_with_retries_success_after_retry(self):
        """Test service check success after retries."""
        mock_service = AsyncMock()
        mock_service.health_check.side_effect = [Exception("Failed"), True]
        
        with patch('src.mcp_server.main.asyncio.sleep'):
            result = await _check_service_with_retries(
                mock_service, "test_service", check_method="health_check", max_retries=2
            )
        
        assert result is True
        assert mock_service.health_check.call_count == 2

    @pytest.mark.asyncio
    async def test_check_service_with_retries_all_failures(self):
        """Test service check fails after all retries."""
        mock_service = AsyncMock()
        mock_service.ping.side_effect = Exception("Service unavailable")
        
        with patch('src.mcp_server.main.asyncio.sleep'):
            with patch('src.mcp_server.main.logger') as mock_logger:
                result = await _check_service_with_retries(
                    mock_service, "test_service", check_method="ping", max_retries=2
                )
        
        assert result is False
        assert mock_service.ping.call_count == 2
        mock_logger.warning.assert_called()

    def test_is_in_startup_grace_period_within_period(self):
        """Test startup grace period check within period."""
        # Mock startup time to be recent
        import src.mcp_server.main
        with patch.object(src.mcp_server.main, '_startup_time', time.time() - 30):
            result = _is_in_startup_grace_period(grace_period_seconds=60)
            
            assert result is True

    def test_is_in_startup_grace_period_outside_period(self):
        """Test startup grace period check outside period."""
        # Mock startup time to be old
        import src.mcp_server.main
        with patch.object(src.mcp_server.main, '_startup_time', time.time() - 120):
            result = _is_in_startup_grace_period(grace_period_seconds=60)
            
            assert result is False

    def test_is_in_startup_grace_period_no_startup_time(self):
        """Test startup grace period when no startup time set."""
        import src.mcp_server.main
        with patch.object(src.mcp_server.main, '_startup_time', None):
            result = _is_in_startup_grace_period()
            
            assert result is False


class TestHealthEndpoints:
    """Test health and status endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    @pytest.mark.asyncio 
    async def test_root_endpoint(self):
        """Test root endpoint."""
        from src.mcp_server.main import root
        
        result = await root()
        
        assert "service" in result
        assert "version" in result
        assert "status" in result
        assert result["service"] == "veris-memory-mcp-server"

    def test_health_endpoint_basic(self):
        """Test basic health endpoint."""
        client = TestClient(app)
        
        with patch('src.mcp_server.main._check_service_with_retries', return_value=asyncio.create_task(asyncio.coroutine(lambda: True)())):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "timestamp" in data

    def test_status_endpoint_all_services_healthy(self):
        """Test status endpoint with all services healthy."""
        client = TestClient(app)
        
        async def mock_check_healthy(*args, **kwargs):
            return True
        
        with patch('src.mcp_server.main._check_service_with_retries', side_effect=mock_check_healthy):
            response = client.get("/status")
            
            assert response.status_code == 200
            data = response.json()
            assert "agent_ready" in data
            assert "dependencies" in data
            assert "deps" in data
            assert "tools" in data

    def test_verify_readiness_endpoint(self):
        """Test readiness verification endpoint."""
        client = TestClient(app)
        
        async def mock_check_ready(*args, **kwargs):
            return "healthy", ""
        
        with patch('src.mcp_server.main._check_service_with_retries', side_effect=mock_check_ready):
            response = client.post("/tools/verify_readiness")
            
            assert response.status_code == 200
            data = response.json()
            assert "ready" in data
            assert "readiness_level" in data
            assert "service_status" in data
            assert "recommended_actions" in data


class TestMCPToolEndpoints:
    """Test MCP tool endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_store_context_endpoint_success(self):
        """Test store_context endpoint success."""
        client = TestClient(app)
        
        request_data = {
            "content": {"title": "Test", "body": "Content"},
            "agent_id": "agent_123",
            "metadata": {"type": "document"}
        }
        
        with patch('src.mcp_server.main.validate_mcp_request', return_value=[]):
            with patch('src.mcp_server.main.KVStore') as mock_kv:
                mock_kv_instance = AsyncMock()
                mock_kv.return_value = mock_kv_instance
                mock_kv_instance.store_context.return_value = {"status": "success", "id": "ctx_123"}
                
                response = client.post("/tools/store_context", json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"

    def test_store_context_endpoint_validation_error(self):
        """Test store_context endpoint with validation error."""
        client = TestClient(app)
        
        request_data = {
            "content": {"title": "Test"},
            "agent_id": "agent_123"
        }
        
        with patch('src.mcp_server.main.validate_mcp_request', return_value=["Validation error"]):
            with patch('src.mcp_server.main.handle_validation_error') as mock_handler:
                mock_handler.return_value = {"error": "validation_failed"}
                
                response = client.post("/tools/store_context", json=request_data)
                
                assert response.status_code == 200  # Error handled gracefully
                mock_handler.assert_called_once()

    def test_retrieve_context_endpoint_success(self):
        """Test retrieve_context endpoint success."""
        client = TestClient(app)
        
        request_data = {
            "query": "test query",
            "agent_id": "agent_123",
            "limit": 5
        }
        
        with patch('src.mcp_server.main.validate_mcp_request', return_value=[]):
            with patch('src.mcp_server.main.KVStore') as mock_kv:
                mock_kv_instance = AsyncMock()
                mock_kv.return_value = mock_kv_instance
                mock_kv_instance.retrieve_context.return_value = {
                    "results": [{"id": "ctx_1", "content": {"title": "Result 1"}}],
                    "total": 1
                }
                
                response = client.post("/tools/retrieve_context", json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                assert "results" in data
                assert len(data["results"]) == 1

    def test_query_graph_endpoint_success(self):
        """Test query_graph endpoint success."""
        client = TestClient(app)
        
        request_data = {
            "cypher_query": "MATCH (n) RETURN n LIMIT 5",
            "agent_id": "agent_123"
        }
        
        with patch('src.mcp_server.main.validate_mcp_request', return_value=[]):
            with patch('src.mcp_server.main.validate_cypher_query', return_value=True):
                with patch('src.mcp_server.main.Neo4jClient') as mock_neo4j:
                    mock_neo4j_instance = AsyncMock()
                    mock_neo4j.return_value = mock_neo4j_instance
                    mock_neo4j_instance.execute_read_query.return_value = [{"n": {"id": 1}}]
                    
                    response = client.post("/tools/query_graph", json=request_data)
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "results" in data

    def test_query_graph_endpoint_invalid_cypher(self):
        """Test query_graph endpoint with invalid Cypher."""
        client = TestClient(app)
        
        request_data = {
            "cypher_query": "INVALID CYPHER QUERY",
            "agent_id": "agent_123"
        }
        
        with patch('src.mcp_server.main.validate_mcp_request', return_value=[]):
            with patch('src.mcp_server.main.validate_cypher_query', return_value=False):
                response = client.post("/tools/query_graph", json=request_data)
                
                # Should return error for invalid Cypher
                assert response.status_code == 200  # Error handled gracefully
                data = response.json()
                assert "error" in data

    def test_update_scratchpad_endpoint_success(self):
        """Test update_scratchpad endpoint success."""
        client = TestClient(app)
        
        request_data = {
            "content": "Updated content",
            "agent_id": "agent_123",
            "operation": "append"
        }
        
        with patch('src.mcp_server.main.validate_mcp_request', return_value=[]):
            with patch('src.mcp_server.main.KVStore') as mock_kv:
                mock_kv_instance = AsyncMock()
                mock_kv.return_value = mock_kv_instance
                mock_kv_instance.update_scratchpad.return_value = {"status": "success"}
                
                response = client.post("/tools/update_scratchpad", json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"

    def test_get_agent_state_endpoint_success(self):
        """Test get_agent_state endpoint success."""
        client = TestClient(app)
        
        request_data = {
            "agent_id": "agent_123",
            "include_scratchpad": True,
            "include_metrics": True
        }
        
        with patch('src.mcp_server.main.validate_mcp_request', return_value=[]):
            with patch('src.mcp_server.main.KVStore') as mock_kv:
                mock_kv_instance = AsyncMock()
                mock_kv.return_value = mock_kv_instance
                mock_kv_instance.get_agent_state.return_value = {
                    "agent_id": "agent_123",
                    "scratchpad": "Current scratchpad",
                    "metrics": {"requests": 10}
                }
                
                response = client.post("/tools/get_agent_state", json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                assert data["agent_id"] == "agent_123"

    def test_list_tools_endpoint(self):
        """Test list_tools endpoint."""
        client = TestClient(app)
        
        response = client.get("/tools")
        
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)
        assert len(data["tools"]) > 0


class TestDashboardEndpoints:
    """Test dashboard-related endpoints."""

    def test_dashboard_endpoints_when_available(self):
        """Test dashboard endpoints when dashboard is available."""
        client = TestClient(app)
        
        with patch('src.mcp_server.main.DASHBOARD_AVAILABLE', True):
            with patch('src.mcp_server.main.UnifiedDashboard') as mock_dashboard:
                mock_dashboard_instance = AsyncMock()
                mock_dashboard.return_value = mock_dashboard_instance
                mock_dashboard_instance.get_dashboard_json.return_value = {"status": "ok"}
                
                response = client.get("/dashboard")
                
                assert response.status_code == 200

    def test_dashboard_endpoints_when_unavailable(self):
        """Test dashboard endpoints when dashboard is unavailable."""
        client = TestClient(app)
        
        with patch('src.mcp_server.main.DASHBOARD_AVAILABLE', False):
            response = client.get("/dashboard")
            
            assert response.status_code == 503  # Service Unavailable
            data = response.json()
            assert "error" in data

    def test_dashboard_ascii_endpoint(self):
        """Test ASCII dashboard endpoint."""
        client = TestClient(app)
        
        with patch('src.mcp_server.main.DASHBOARD_AVAILABLE', True):
            with patch('src.mcp_server.main.UnifiedDashboard') as mock_dashboard:
                mock_dashboard_instance = AsyncMock()
                mock_dashboard.return_value = mock_dashboard_instance
                mock_dashboard_instance.get_dashboard_ascii.return_value = "ASCII Dashboard"
                
                response = client.get("/dashboard/ascii")
                
                assert response.status_code == 200
                assert "ASCII Dashboard" in response.text

    def test_metrics_endpoints(self):
        """Test various metrics endpoints."""
        client = TestClient(app)
        endpoints = ["/metrics/system", "/metrics/service", "/metrics/security"]
        
        for endpoint in endpoints:
            with patch('src.mcp_server.main.DASHBOARD_AVAILABLE', True):
                with patch('src.mcp_server.main.UnifiedDashboard') as mock_dashboard:
                    mock_dashboard_instance = AsyncMock()
                    mock_dashboard.return_value = mock_dashboard_instance
                    mock_dashboard_instance.get_system_metrics.return_value = {"cpu": 50}
                    mock_dashboard_instance.get_service_metrics.return_value = {"requests": 100}
                    mock_dashboard_instance.get_security_metrics.return_value = {"threats": 0}
                    
                    response = client.get(endpoint)
                    
                    assert response.status_code == 200

    def test_dashboard_refresh_endpoint(self):
        """Test dashboard refresh endpoint."""
        client = TestClient(app)
        
        with patch('src.mcp_server.main.DASHBOARD_AVAILABLE', True):
            with patch('src.mcp_server.main.UnifiedDashboard') as mock_dashboard:
                mock_dashboard_instance = AsyncMock()
                mock_dashboard.return_value = mock_dashboard_instance
                mock_dashboard_instance.refresh.return_value = None
                
                response = client.post("/dashboard/refresh")
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "refreshed"


class TestWebSocketDashboard:
    """Test WebSocket dashboard functionality."""

    @pytest.mark.asyncio
    async def test_websocket_dashboard_connection(self):
        """Test WebSocket dashboard connection."""
        with patch('src.mcp_server.main.DASHBOARD_AVAILABLE', True):
            with patch('src.mcp_server.main._stream_dashboard_updates') as mock_stream:
                mock_stream.return_value = None
                
                # Simulate WebSocket connection
                mock_websocket = AsyncMock()
                mock_websocket.accept = AsyncMock()
                
                from src.mcp_server.main import websocket_dashboard
                
                await websocket_dashboard(mock_websocket)
                
                mock_websocket.accept.assert_called_once()
                mock_stream.assert_called_once_with(mock_websocket)

    @pytest.mark.asyncio
    async def test_websocket_dashboard_unavailable(self):
        """Test WebSocket dashboard when unavailable."""
        with patch('src.mcp_server.main.DASHBOARD_AVAILABLE', False):
            mock_websocket = AsyncMock()
            mock_websocket.close = AsyncMock()
            
            from src.mcp_server.main import websocket_dashboard
            
            await websocket_dashboard(mock_websocket)
            
            mock_websocket.close.assert_called_once_with(code=1011, reason="Dashboard not available")

    @pytest.mark.asyncio
    async def test_stream_dashboard_updates(self):
        """Test dashboard update streaming."""
        mock_websocket = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        with patch('src.mcp_server.main.UnifiedDashboard') as mock_dashboard:
            mock_dashboard_instance = AsyncMock()
            mock_dashboard.return_value = mock_dashboard_instance
            mock_dashboard_instance.get_dashboard_json.return_value = {"timestamp": "2023-01-01"}
            
            # Mock asyncio.sleep to prevent infinite loop
            with patch('src.mcp_server.main.asyncio.sleep', side_effect=[None, WebSocketDisconnect()]):
                from src.mcp_server.main import _stream_dashboard_updates
                
                try:
                    await _stream_dashboard_updates(mock_websocket)
                except WebSocketDisconnect:
                    pass
                
                mock_websocket.send_text.assert_called()

    @pytest.mark.asyncio 
    async def test_broadcast_to_websockets(self):
        """Test broadcasting to WebSocket connections."""
        from src.mcp_server.main import _broadcast_to_websockets, websocket_connections
        
        # Mock WebSocket connections
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws_closed = AsyncMock()
        mock_ws_closed.send_text.side_effect = Exception("Connection closed")
        
        websocket_connections.clear()
        websocket_connections.extend([mock_ws1, mock_ws2, mock_ws_closed])
        
        message = {"type": "update", "data": "test"}
        
        await _broadcast_to_websockets(message)
        
        mock_ws1.send_text.assert_called_once_with(json.dumps(message))
        mock_ws2.send_text.assert_called_once_with(json.dumps(message))
        # Closed connection should be removed from list
        assert mock_ws_closed not in websocket_connections


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_mcp_tool_storage_error(self):
        """Test MCP tool handling storage errors."""
        client = TestClient(app)
        
        request_data = {
            "content": {"title": "Test"},
            "agent_id": "agent_123"
        }
        
        with patch('src.mcp_server.main.validate_mcp_request', return_value=[]):
            with patch('src.mcp_server.main.KVStore') as mock_kv:
                mock_kv_instance = AsyncMock()
                mock_kv.return_value = mock_kv_instance
                mock_kv_instance.store_context.side_effect = Exception("Storage error")
                
                with patch('src.mcp_server.main.handle_storage_error') as mock_handler:
                    mock_handler.return_value = {"error": "storage_failed"}
                    
                    response = client.post("/tools/store_context", json=request_data)
                    
                    assert response.status_code == 200
                    mock_handler.assert_called_once()

    def test_embedding_generation_error(self):
        """Test embedding generation error handling."""
        content = {"test": "data"}
        
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        
        with patch('src.mcp_server.main._get_embedding_model', return_value=mock_model):
            # Should fall back to hash-based embedding
            import asyncio
            result = asyncio.run(_generate_embedding(content))
            
            assert isinstance(result, list)
            assert len(result) > 0

    def test_service_initialization_error(self):
        """Test service initialization error handling."""
        client = TestClient(app)
        
        with patch('src.mcp_server.main.KVStore', side_effect=Exception("Init failed")):
            request_data = {
                "content": {"title": "Test"},
                "agent_id": "agent_123"
            }
            
            response = client.post("/tools/store_context", json=request_data)
            
            # Should handle initialization error gracefully
            assert response.status_code == 200
            data = response.json()
            assert "error" in data


class TestEdgeCasesAndIntegration:
    """Test edge cases and integration scenarios."""

    def test_app_cors_configuration(self):
        """Test CORS middleware configuration."""
        # Check that CORS middleware is properly configured
        assert any(
            middleware.cls.__name__ == "CORSMiddleware" 
            for middleware in app.user_middleware
        )

    def test_app_exception_handler_registration(self):
        """Test global exception handler registration."""
        # Verify exception handler is registered
        assert Exception in app.exception_handlers

    def test_startup_shutdown_event_registration(self):
        """Test startup and shutdown event registration."""
        # Check that startup and shutdown events are registered
        startup_handlers = [handler for handler in app.router.on_startup]
        shutdown_handlers = [handler for handler in app.router.on_shutdown]
        
        assert len(startup_handlers) > 0
        assert len(shutdown_handlers) > 0

    def test_large_content_handling(self):
        """Test handling of large content in store_context."""
        client = TestClient(app)
        
        # Create large content
        large_content = {
            "title": "Large Document",
            "body": "x" * 10000  # 10KB content
        }
        
        request_data = {
            "content": large_content,
            "agent_id": "agent_123"
        }
        
        with patch('src.mcp_server.main.validate_mcp_request', return_value=[]):
            with patch('src.mcp_server.main.KVStore') as mock_kv:
                mock_kv_instance = AsyncMock()
                mock_kv.return_value = mock_kv_instance
                mock_kv_instance.store_context.return_value = {"status": "success", "id": "large_ctx"}
                
                response = client.post("/tools/store_context", json=request_data)
                
                assert response.status_code == 200

    def test_concurrent_requests_handling(self):
        """Test handling of concurrent requests."""
        client = TestClient(app)
        
        request_data = {
            "content": {"title": "Concurrent Test"},
            "agent_id": "agent_123"
        }
        
        with patch('src.mcp_server.main.validate_mcp_request', return_value=[]):
            with patch('src.mcp_server.main.KVStore') as mock_kv:
                mock_kv_instance = AsyncMock()
                mock_kv.return_value = mock_kv_instance
                mock_kv_instance.store_context.return_value = {"status": "success"}
                
                # Send multiple concurrent requests
                import concurrent.futures
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [
                        executor.submit(client.post, "/tools/store_context", json=request_data)
                        for _ in range(5)
                    ]
                    
                    responses = [future.result() for future in futures]
                    
                    # All requests should succeed
                    assert all(response.status_code == 200 for response in responses)

    def test_malformed_request_handling(self):
        """Test handling of malformed requests."""
        client = TestClient(app)
        
        # Send malformed JSON
        response = client.post(
            "/tools/store_context",
            data="invalid json content",
            headers={"Content-Type": "application/json"}
        )
        
        # Should return 422 for malformed JSON
        assert response.status_code == 422

    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        client = TestClient(app)
        
        # Missing required fields
        request_data = {"content": {"title": "Test"}}  # Missing agent_id
        
        response = client.post("/tools/store_context", json=request_data)
        
        # Should return 422 for missing required fields
        assert response.status_code == 422

    def test_environment_variable_configuration(self):
        """Test environment variable configuration."""
        with patch.dict(os.environ, {
            'EMBEDDING_MODEL': 'custom-model',
            'STRICT_EMBEDDINGS': 'true',
            'LOG_LEVEL': 'DEBUG'
        }):
            # Test environment variables are properly used
            with patch('src.mcp_server.main.SENTENCE_TRANSFORMERS_AVAILABLE', True):
                with patch('src.mcp_server.main.SentenceTransformer') as mock_st:
                    mock_model = MagicMock()
                    mock_model.get_sentence_embedding_dimension.return_value = 768
                    mock_st.return_value = mock_model
                    
                    with patch('src.mcp_server.main._embedding_model', None):
                        model = _get_embedding_model()
                        
                        mock_st.assert_called_with('custom-model')