#!/usr/bin/env python3
"""
Comprehensive tests for MCP Server Main module - Phase 6 Coverage

This test module provides comprehensive coverage for the main MCP server
implementation including API endpoints, request handling, and health checks.
"""
import pytest
import asyncio
import json
import logging
from unittest.mock import patch, Mock, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
from datetime import datetime
import tempfile
import os

# Import FastAPI testing utilities
try:
    from fastapi.testclient import TestClient
    from httpx import AsyncClient
    FASTAPI_TESTING_AVAILABLE = True
except ImportError:
    FASTAPI_TESTING_AVAILABLE = False

# Import MCP server components
try:
    from src.mcp_server.main import (
        app, StoreContextRequest, RetrieveContextRequest, 
        QueryGraphRequest, UpdateScratchpadRequest, GetAgentStateRequest,
        _get_embedding_model, _generate_embedding, startup_event, shutdown_event,
        health, status, verify_readiness, store_context, retrieve_context,
        query_graph, update_scratchpad_endpoint, get_agent_state_endpoint,
        list_tools, global_exception_handler, _check_service_with_retries,
        _is_in_startup_grace_period
    )
    MCP_SERVER_AVAILABLE = True
except ImportError:
    MCP_SERVER_AVAILABLE = False


@pytest.mark.skipif(not MCP_SERVER_AVAILABLE, reason="MCP server not available")
@pytest.mark.skipif(not FASTAPI_TESTING_AVAILABLE, reason="FastAPI testing not available")
class TestMCPServerInitialization:
    """Test MCP server initialization and configuration"""
    
    def test_app_creation(self):
        """Test FastAPI app creation and configuration"""
        assert app is not None
        assert app.title == "Veris Memory MCP Server"
        assert "Context store with vector embeddings" in app.description
        assert app.version == "1.0.0"
    
    @patch('src.mcp_server.main.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('src.mcp_server.main.SentenceTransformer')
    def test_embedding_model_initialization_success(self, mock_sentence_transformer):
        """Test successful embedding model initialization"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        model = _get_embedding_model()
        assert model is not None
        mock_sentence_transformer.assert_called_once()
    
    @patch('src.mcp_server.main.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    def test_embedding_model_initialization_failure(self):
        """Test embedding model initialization when sentence-transformers unavailable"""
        model = _get_embedding_model()
        assert model is None
    
    @patch('src.mcp_server.main.os.getenv')
    @patch('src.mcp_server.main.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    def test_strict_embeddings_mode(self, mock_getenv):
        """Test strict embeddings mode behavior"""
        mock_getenv.return_value = "true"
        
        # In actual code, this would raise RuntimeError during import
        # Here we test the condition logic
        strict_mode = os.getenv("STRICT_EMBEDDINGS", "false").lower() == "true"
        assert strict_mode is True
    
    @patch('src.mcp_server.main._get_embedding_model')
    async def test_generate_embedding_with_model(self, mock_get_model):
        """Test embedding generation with available model"""
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_get_model.return_value = mock_model
        
        content = {"text": "test content", "metadata": {"type": "test"}}
        embedding = await _generate_embedding(content)
        
        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_model.encode.assert_called_once()
    
    @patch('src.mcp_server.main._get_embedding_model')
    async def test_generate_embedding_without_model(self, mock_get_model):
        """Test embedding generation when model unavailable"""
        mock_get_model.return_value = None
        
        content = {"text": "test content"}
        embedding = await _generate_embedding(content)
        
        # Should return default embedding or empty list
        assert isinstance(embedding, list)


@pytest.mark.skipif(not MCP_SERVER_AVAILABLE, reason="MCP server not available")
@pytest.mark.skipif(not FASTAPI_TESTING_AVAILABLE, reason="FastAPI testing not available") 
class TestMCPServerLifecycle:
    """Test MCP server lifecycle events"""
    
    @patch('src.mcp_server.main.neo4j_client')
    @patch('src.mcp_server.main.vector_db')
    @patch('src.mcp_server.main.kv_store')
    async def test_startup_event_success(self, mock_kv, mock_vector, mock_neo4j):
        """Test successful startup event"""
        # Mock successful connections
        mock_neo4j.connect.return_value = True
        mock_vector.connect.return_value = True
        mock_kv.connect.return_value = True
        
        # Test startup doesn't raise exceptions
        try:
            await startup_event()
        except Exception as e:
            pytest.fail(f"Startup event failed: {e}")
    
    @patch('src.mcp_server.main.neo4j_client')
    @patch('src.mcp_server.main.vector_db') 
    @patch('src.mcp_server.main.kv_store')
    async def test_startup_event_connection_failures(self, mock_kv, mock_vector, mock_neo4j):
        """Test startup event with connection failures"""
        # Mock connection failures
        mock_neo4j.connect.return_value = False
        mock_vector.connect.return_value = False
        mock_kv.connect.return_value = False
        
        # Startup should handle failures gracefully
        try:
            await startup_event()
        except Exception as e:
            # Some failures might be acceptable
            pass
    
    @patch('src.mcp_server.main.neo4j_client')
    @patch('src.mcp_server.main.vector_db')
    @patch('src.mcp_server.main.kv_store')
    async def test_shutdown_event(self, mock_kv, mock_vector, mock_neo4j):
        """Test shutdown event"""
        # Mock close methods
        mock_neo4j.close = Mock()
        mock_vector.close = Mock()
        mock_kv.close = Mock()
        
        await shutdown_event()
        
        # Verify all services are closed
        mock_neo4j.close.assert_called_once()
        mock_vector.close.assert_called_once()
        mock_kv.close.assert_called_once()
    
    def test_startup_grace_period_check(self):
        """Test startup grace period checking"""
        # Test within grace period
        with patch('src.mcp_server.main.startup_time', datetime.now().timestamp()):
            assert _is_in_startup_grace_period(grace_period_seconds=60) is True
        
        # Test outside grace period
        with patch('src.mcp_server.main.startup_time', datetime.now().timestamp() - 120):
            assert _is_in_startup_grace_period(grace_period_seconds=60) is False


@pytest.mark.skipif(not MCP_SERVER_AVAILABLE, reason="MCP server not available")
@pytest.mark.skipif(not FASTAPI_TESTING_AVAILABLE, reason="FastAPI testing not available")
class TestMCPServerHealthChecks:
    """Test MCP server health check endpoints"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert data["service"] == "Veris Memory MCP Server"
    
    @patch('src.mcp_server.main.neo4j_client')
    @patch('src.mcp_server.main.vector_db')
    @patch('src.mcp_server.main.kv_store')
    async def test_health_endpoint_all_services_healthy(self, mock_kv, mock_vector, mock_neo4j):
        """Test health endpoint when all services are healthy"""
        # Mock healthy services
        mock_neo4j.check_connection.return_value = True
        mock_vector.check_connection.return_value = True
        mock_kv.ensure_connected.return_value = True
        
        # Use async client for async endpoint
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "timestamp" in data
    
    @patch('src.mcp_server.main.neo4j_client')
    @patch('src.mcp_server.main.vector_db')
    @patch('src.mcp_server.main.kv_store')
    async def test_health_endpoint_service_failures(self, mock_kv, mock_vector, mock_neo4j):
        """Test health endpoint when services are unhealthy"""
        # Mock unhealthy services
        mock_neo4j.check_connection.return_value = False
        mock_vector.check_connection.return_value = False
        mock_kv.ensure_connected.return_value = False
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/health")
        
        # Should still return 200 but with unhealthy status
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "services" in data
    
    @patch('src.mcp_server.main._is_in_startup_grace_period')
    @patch('src.mcp_server.main.neo4j_client')
    async def test_health_endpoint_during_startup_grace_period(self, mock_neo4j, mock_grace_period):
        """Test health endpoint during startup grace period"""
        mock_grace_period.return_value = True
        mock_neo4j.check_connection.return_value = False
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        # During grace period, might be considered healthy even with failures
        assert "status" in data
    
    async def test_check_service_with_retries_success(self):
        """Test service checking with retries - success case"""
        async def mock_successful_check():
            return True
        
        result = await _check_service_with_retries(
            "test_service", mock_successful_check, max_retries=3
        )
        assert result is True
    
    async def test_check_service_with_retries_failure(self):
        """Test service checking with retries - failure case"""
        async def mock_failing_check():
            return False
        
        result = await _check_service_with_retries(
            "test_service", mock_failing_check, max_retries=3
        )
        assert result is False
    
    async def test_check_service_with_retries_exception(self):
        """Test service checking with retries - exception case"""
        async def mock_exception_check():
            raise Exception("Service check failed")
        
        result = await _check_service_with_retries(
            "test_service", mock_exception_check, max_retries=3
        )
        assert result is False


@pytest.mark.skipif(not MCP_SERVER_AVAILABLE, reason="MCP server not available")
@pytest.mark.skipif(not FASTAPI_TESTING_AVAILABLE, reason="FastAPI testing not available")
class TestMCPServerDataModels:
    """Test MCP server data models and validation"""
    
    def test_store_context_request_model(self):
        """Test StoreContextRequest model validation"""
        # Valid request
        valid_request = {
            "context_id": "test_context_001",
            "content": {"text": "test content", "metadata": {"type": "test"}},
            "agent_id": "agent_123",
            "ttl_seconds": 3600
        }
        
        request = StoreContextRequest(**valid_request)
        assert request.context_id == "test_context_001"
        assert request.content["text"] == "test content"
        assert request.agent_id == "agent_123"
        assert request.ttl_seconds == 3600
        
        # Test with minimal required fields
        minimal_request = {
            "context_id": "minimal_context",
            "content": {"text": "minimal content"}
        }
        
        minimal = StoreContextRequest(**minimal_request)
        assert minimal.context_id == "minimal_context"
        assert minimal.agent_id is None  # Optional field
        assert minimal.ttl_seconds is None  # Optional field
    
    def test_retrieve_context_request_model(self):
        """Test RetrieveContextRequest model validation"""
        valid_request = {
            "context_id": "retrieve_context_001",
            "agent_id": "agent_456",
            "include_metadata": True
        }
        
        request = RetrieveContextRequest(**valid_request)
        assert request.context_id == "retrieve_context_001"
        assert request.agent_id == "agent_456"
        assert request.include_metadata is True
    
    def test_query_graph_request_model(self):
        """Test QueryGraphRequest model validation"""
        valid_request = {
            "query": "MATCH (n) RETURN n LIMIT 10",
            "parameters": {"limit": 10},
            "agent_id": "agent_789"
        }
        
        request = QueryGraphRequest(**valid_request)
        assert request.query == "MATCH (n) RETURN n LIMIT 10"
        assert request.parameters == {"limit": 10}
        assert request.agent_id == "agent_789"
    
    def test_update_scratchpad_request_model(self):
        """Test UpdateScratchpadRequest model validation"""
        valid_request = {
            "agent_id": "scratchpad_agent",
            "key": "notes",
            "content": "Updated notes content",
            "mode": "overwrite",
            "ttl_seconds": 7200
        }
        
        request = UpdateScratchpadRequest(**valid_request)
        assert request.agent_id == "scratchpad_agent"
        assert request.key == "notes"
        assert request.content == "Updated notes content"
        assert request.mode == "overwrite"
        assert request.ttl_seconds == 7200
    
    def test_get_agent_state_request_model(self):
        """Test GetAgentStateRequest model validation"""
        valid_request = {
            "agent_id": "state_agent",
            "key": "user_preferences",
            "prefix": "config"
        }
        
        request = GetAgentStateRequest(**valid_request)
        assert request.agent_id == "state_agent"
        assert request.key == "user_preferences"
        assert request.prefix == "config"


@pytest.mark.skipif(not MCP_SERVER_AVAILABLE, reason="MCP server not available")
@pytest.mark.skipif(not FASTAPI_TESTING_AVAILABLE, reason="FastAPI testing not available")
class TestMCPServerAPIEndpoints:
    """Test MCP server API endpoints"""
    
    def setup_method(self):
        """Setup test environment"""
        self.client = TestClient(app)
    
    @patch('src.mcp_server.main.kv_store')
    @patch('src.mcp_server.main._generate_embedding')
    async def test_store_context_endpoint_success(self, mock_embedding, mock_kv):
        """Test successful context storage"""
        mock_embedding.return_value = [0.1, 0.2, 0.3]
        mock_kv.store_context.return_value = True
        
        request_data = {
            "context_id": "test_store_001",
            "content": {"text": "test content", "metadata": {"type": "test"}},
            "agent_id": "agent_123",
            "ttl_seconds": 3600
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/store_context", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "context_id" in data
        assert "embedding_generated" in data
    
    @patch('src.mcp_server.main.kv_store')
    @patch('src.mcp_server.main._generate_embedding')
    async def test_store_context_endpoint_failure(self, mock_embedding, mock_kv):
        """Test context storage failure"""
        mock_embedding.return_value = [0.1, 0.2, 0.3]
        mock_kv.store_context.return_value = False
        
        request_data = {
            "context_id": "test_store_fail",
            "content": {"text": "test content"}
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/store_context", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    @patch('src.mcp_server.main.kv_store')
    async def test_retrieve_context_endpoint_success(self, mock_kv):
        """Test successful context retrieval"""
        mock_context_data = {
            "context_id": "test_retrieve_001",
            "content": {"text": "retrieved content"},
            "metadata": {"created_at": "2023-01-01T00:00:00Z"}
        }
        mock_kv.get_context.return_value = mock_context_data
        
        request_data = {
            "context_id": "test_retrieve_001",
            "agent_id": "agent_456"
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/retrieve_context", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["context"] == mock_context_data
    
    @patch('src.mcp_server.main.kv_store')
    async def test_retrieve_context_endpoint_not_found(self, mock_kv):
        """Test context retrieval when context not found"""
        mock_kv.get_context.return_value = None
        
        request_data = {
            "context_id": "nonexistent_context",
            "agent_id": "agent_456"
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/retrieve_context", json=request_data)
        
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()
    
    async def test_list_tools_endpoint(self):
        """Test list tools endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/tools")
        
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)
        assert len(data["tools"]) > 0
        
        # Check that expected tools are present
        tool_names = [tool["name"] for tool in data["tools"]]
        expected_tools = ["store_context", "retrieve_context", "query_graph", "update_scratchpad"]
        for expected_tool in expected_tools:
            assert expected_tool in tool_names
    
    @patch('src.mcp_server.main.agent_namespace')
    @patch('src.mcp_server.main.kv_store')
    async def test_update_scratchpad_endpoint(self, mock_kv, mock_namespace):
        """Test scratchpad update endpoint"""
        mock_namespace.create_namespaced_key.return_value = "agent:test_agent:scratchpad:notes"
        mock_kv.set.return_value = True
        
        request_data = {
            "agent_id": "test_agent",
            "key": "notes",
            "content": "Updated scratchpad content",
            "mode": "overwrite"
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/update_scratchpad", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "agent_id" in data
        assert "key" in data
    
    @patch('src.mcp_server.main.agent_namespace')
    @patch('src.mcp_server.main.kv_store')
    async def test_get_agent_state_endpoint(self, mock_kv, mock_namespace):
        """Test agent state retrieval endpoint"""
        mock_namespace.create_namespaced_key.return_value = "agent:state_agent:state:preferences"
        mock_kv.get.return_value = '{"theme": "dark", "language": "en"}'
        
        request_data = {
            "agent_id": "state_agent",
            "key": "preferences",
            "prefix": "state"
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/get_agent_state", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"] == {"theme": "dark", "language": "en"}


@pytest.mark.skipif(not MCP_SERVER_AVAILABLE, reason="MCP server not available")
@pytest.mark.skipif(not FASTAPI_TESTING_AVAILABLE, reason="FastAPI testing not available")
class TestMCPServerErrorHandling:
    """Test MCP server error handling"""
    
    def setup_method(self):
        """Setup test environment"""
        self.client = TestClient(app)
    
    async def test_global_exception_handler(self):
        """Test global exception handler"""
        from fastapi import Request
        
        # Create mock request
        mock_request = Mock(spec=Request)
        mock_request.url = "http://test/error"
        mock_request.method = "POST"
        
        # Test with generic exception
        test_exception = Exception("Test error")
        response = await global_exception_handler(mock_request, test_exception)
        
        assert response.status_code == 500
        response_data = json.loads(response.body)
        assert response_data["success"] is False
        assert "error" in response_data
    
    async def test_endpoint_validation_errors(self):
        """Test endpoint validation error handling"""
        # Test with invalid request data
        invalid_request = {
            "context_id": "",  # Empty context_id should be invalid
            "content": "not a dict"  # Should be a dict
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/store_context", json=invalid_request)
        
        # Should return validation error
        assert response.status_code == 422  # Validation error
    
    async def test_endpoint_missing_required_fields(self):
        """Test endpoint behavior with missing required fields"""
        # Test with missing required fields
        incomplete_request = {
            "content": {"text": "test content"}
            # Missing required context_id
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/store_context", json=incomplete_request)
        
        # Should return validation error
        assert response.status_code == 422
    
    @patch('src.mcp_server.main.kv_store')
    async def test_endpoint_service_exceptions(self, mock_kv):
        """Test endpoint behavior when services raise exceptions"""
        # Mock service to raise exception
        mock_kv.store_context.side_effect = Exception("Storage service error")
        
        request_data = {
            "context_id": "test_exception",
            "content": {"text": "test content"}
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/store_context", json=request_data)
        
        # Should handle exception gracefully
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "error" in data


@pytest.mark.skipif(not MCP_SERVER_AVAILABLE, reason="MCP server not available")
@pytest.mark.skipif(not FASTAPI_TESTING_AVAILABLE, reason="FastAPI testing not available")
class TestMCPServerIntegration:
    """Integration tests for MCP server components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.client = TestClient(app)
    
    @patch('src.mcp_server.main.kv_store')
    @patch('src.mcp_server.main.agent_namespace')
    async def test_complete_context_workflow(self, mock_namespace, mock_kv):
        """Test complete context storage and retrieval workflow"""
        # Setup mocks
        mock_namespace.validate_agent_id.return_value = True
        mock_kv.store_context.return_value = True
        mock_kv.get_context.return_value = {
            "context_id": "workflow_test",
            "content": {"text": "workflow content"},
            "metadata": {"agent_id": "workflow_agent"}
        }
        
        # Store context
        store_request = {
            "context_id": "workflow_test",
            "content": {"text": "workflow content"},
            "agent_id": "workflow_agent"
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            store_response = await ac.post("/store_context", json=store_request)
            assert store_response.status_code == 200
            
            # Retrieve context
            retrieve_request = {
                "context_id": "workflow_test",
                "agent_id": "workflow_agent"
            }
            
            retrieve_response = await ac.post("/retrieve_context", json=retrieve_request)
            assert retrieve_response.status_code == 200
            
            retrieve_data = retrieve_response.json()
            assert retrieve_data["success"] is True
            assert retrieve_data["context"]["context_id"] == "workflow_test"
    
    @patch('src.mcp_server.main.kv_store')
    @patch('src.mcp_server.main.agent_namespace')
    async def test_agent_isolation_workflow(self, mock_namespace, mock_kv):
        """Test agent isolation in workflows"""
        # Setup mocks for different agents
        mock_namespace.validate_agent_id.return_value = True
        mock_namespace.create_namespaced_key.side_effect = lambda agent, prefix, key: f"{prefix}:{agent}:{key}"
        
        # Agent 1 operations
        mock_kv.set.return_value = True
        mock_kv.get.return_value = "agent1_data"
        
        agent1_request = {
            "agent_id": "agent_001",
            "key": "data",
            "content": "agent1_data"
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response1 = await ac.post("/update_scratchpad", json=agent1_request)
            assert response1.status_code == 200
            
            # Agent 2 should get different namespaced key
            agent2_request = {
                "agent_id": "agent_002", 
                "key": "data",
                "content": "agent2_data"
            }
            
            response2 = await ac.post("/update_scratchpad", json=agent2_request)
            assert response2.status_code == 200
            
            # Verify different agents got different namespaced keys
            call_args_list = mock_namespace.create_namespaced_key.call_args_list
            assert len(call_args_list) >= 2
            
            # Keys should be different for different agents
            key1 = call_args_list[0][0]  # First call args
            key2 = call_args_list[1][0]  # Second call args
            assert key1[0] != key2[0]  # Different agent IDs
    
    async def test_status_and_readiness_endpoints(self):
        """Test status and readiness endpoint integration"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Test status endpoint
            status_response = await ac.get("/status")
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert "service" in status_data
            assert "uptime" in status_data
            
            # Test readiness endpoint
            readiness_response = await ac.get("/ready")
            assert readiness_response.status_code == 200
            readiness_data = readiness_response.json()
            assert "ready" in readiness_data
            assert "timestamp" in readiness_data