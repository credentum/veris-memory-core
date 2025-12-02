#!/usr/bin/env python3
"""
Comprehensive tests for MCP Server Implementation - Phase 6 Coverage

This test module provides comprehensive coverage for the official MCP SDK server
implementation including storage initialization, protocol handlers, and tool execution.
"""
import pytest
import asyncio
import json
import logging
import os
import tempfile
from unittest.mock import patch, Mock, MagicMock, AsyncMock, mock_open
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import MCP SDK components for testing
try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.types import EmbeddedResource, ImageContent, Resource, TextContent, Tool
    MCP_SDK_AVAILABLE = True
except ImportError:
    MCP_SDK_AVAILABLE = False

# Import server components
try:
    from src.mcp_server.server import (
        server, initialize_storage_clients, cleanup_storage_clients,
        list_resources, read_resource, list_tools, call_tool,
        store_context_tool, retrieve_context_tool, query_graph_tool,
        update_scratchpad_tool, get_agent_state_tool, detect_communities_tool,
        get_health_status, get_tools_info, select_tools_tool,
        list_available_tools_tool, get_tool_info_tool, main
    )
    SERVER_AVAILABLE = True
except ImportError:
    SERVER_AVAILABLE = False


@pytest.mark.skipif(not SERVER_AVAILABLE, reason="MCP server not available")
@pytest.mark.skipif(not MCP_SDK_AVAILABLE, reason="MCP SDK not available")
@pytest.mark.asyncio
class TestMCPServerInitialization:
    """Test MCP server initialization and storage client setup"""
    
    def test_server_creation(self):
        """Test MCP server instance creation"""
        assert server is not None
        assert server.name == "context-store"
        
    def test_global_variable_initialization(self):
        """Test global variable initialization"""
        # Import globals from server module
        from src.mcp_server.server import neo4j_client, qdrant_client, kv_store, embedding_generator
        
        # Initially None before initialization
        assert neo4j_client is None
        assert qdrant_client is None
        assert kv_store is None
        assert embedding_generator is None
    
    @patch('src.mcp_server.server.validate_all_configs')
    @patch('src.mcp_server.server.SSLConfigManager')
    @patch('src.mcp_server.server.Neo4jInitializer')
    @patch('src.mcp_server.server.VectorDBInitializer')
    @patch('src.mcp_server.server.ContextKV')
    @patch('src.mcp_server.server.create_embedding_generator')
    async def test_initialize_storage_clients_success(self, mock_embedding, mock_kv, 
                                                     mock_vector, mock_neo4j, mock_ssl, mock_config):
        """Test successful storage client initialization"""
        # Mock configuration validation
        mock_config.return_value = {"valid": True, "config": {"base": "config"}}
        
        # Mock SSL manager
        mock_ssl_instance = Mock()
        mock_ssl_instance.validate_ssl_certificates.return_value = {
            "neo4j": True, "qdrant": True, "redis": True
        }
        mock_ssl_instance.get_neo4j_ssl_config.return_value = {"encrypted": True}
        mock_ssl_instance.get_qdrant_ssl_config.return_value = {"https": True}
        mock_ssl_instance.get_redis_ssl_config.return_value = {"ssl": True}
        mock_ssl.return_value = mock_ssl_instance
        
        # Mock successful client initialization
        mock_neo4j_instance = Mock()
        mock_neo4j_instance.connect.return_value = True
        mock_neo4j.return_value = mock_neo4j_instance
        
        mock_vector_instance = Mock()
        mock_vector_instance.connect.return_value = True
        mock_vector.return_value = mock_vector_instance
        
        mock_kv_instance = Mock()
        mock_kv_instance.connect.return_value = True
        mock_kv.return_value = mock_kv_instance
        
        mock_embedding.return_value = Mock()
        
        # Set required environment variables
        with patch.dict(os.environ, {
            'NEO4J_PASSWORD': 'test_password',
            'NEO4J_URI': 'bolt://localhost:7687',
            'QDRANT_URL': 'http://localhost:6333',
            'REDIS_URL': 'redis://localhost:6379'
        }):
            result = await initialize_storage_clients()
        
        assert result["success"] is True
        assert result["neo4j_initialized"] is True
        assert result["qdrant_initialized"] is True
        assert result["kv_store_initialized"] is True
        assert result["embedding_initialized"] is True
    
    @patch('src.mcp_server.server.validate_all_configs')
    async def test_initialize_storage_clients_no_env_vars(self, mock_config):
        """Test storage initialization with missing environment variables"""
        mock_config.return_value = {"valid": True, "config": {"base": "config"}}
        
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            result = await initialize_storage_clients()
        
        assert result["success"] is True  # Still successful but with warnings
        assert result["neo4j_initialized"] is False
        assert result["qdrant_initialized"] is False
        assert result["kv_store_initialized"] is False
    
    @patch('src.mcp_server.server.validate_all_configs')
    @patch('src.mcp_server.server.SSLConfigManager')
    async def test_initialize_storage_clients_connection_failures(self, mock_ssl, mock_config):
        """Test storage initialization with connection failures"""
        mock_config.return_value = {"valid": True, "config": {"base": "config"}}
        
        # Mock SSL manager
        mock_ssl_instance = Mock()
        mock_ssl_instance.validate_ssl_certificates.return_value = {
            "neo4j": False, "qdrant": False, "redis": False
        }
        mock_ssl.return_value = mock_ssl_instance
        
        with patch.dict(os.environ, {
            'NEO4J_PASSWORD': 'test_password',
            'QDRANT_URL': 'http://localhost:6333',
            'REDIS_URL': 'redis://localhost:6379'
        }):
            with patch('src.mcp_server.server.Neo4jInitializer') as mock_neo4j:
                mock_neo4j_instance = Mock()
                mock_neo4j_instance.connect.return_value = False
                mock_neo4j.return_value = mock_neo4j_instance
                
                result = await initialize_storage_clients()
        
        assert result["success"] is True  # Graceful degradation
        assert result["neo4j_initialized"] is False
    
    @patch('src.mcp_server.server.validate_all_configs')
    async def test_initialize_storage_clients_exception_handling(self, mock_config):
        """Test storage initialization with exceptions"""
        mock_config.side_effect = Exception("Configuration error")
        
        result = await initialize_storage_clients()
        
        assert result["success"] is False
        assert "Failed to initialize storage clients" in result["message"]
    
    @patch('src.mcp_server.server.neo4j_client')
    @patch('src.mcp_server.server.qdrant_client')
    @patch('src.mcp_server.server.kv_store')
    async def test_cleanup_storage_clients(self, mock_kv, mock_qdrant, mock_neo4j):
        """Test storage client cleanup"""
        # Mock client instances
        mock_neo4j.close = Mock()
        mock_kv.close = Mock()
        
        await cleanup_storage_clients()
        
        mock_neo4j.close.assert_called_once()
        mock_kv.close.assert_called_once()


@pytest.mark.skipif(not SERVER_AVAILABLE, reason="MCP server not available")
@pytest.mark.skipif(not MCP_SDK_AVAILABLE, reason="MCP SDK not available")
@pytest.mark.asyncio
class TestMCPServerResources:
    """Test MCP server resource handling"""
    
    async def test_list_resources(self):
        """Test resource listing"""
        resources = await list_resources()
        
        assert isinstance(resources, list)
        assert len(resources) == 2
        
        # Check health resource
        health_resource = next((r for r in resources if r.uri == "context://health"), None)
        assert health_resource is not None
        assert health_resource.name == "Health Status"
        assert health_resource.mimeType == "application/json"
        
        # Check tools resource
        tools_resource = next((r for r in resources if r.uri == "context://tools"), None)
        assert tools_resource is not None
        assert tools_resource.name == "Available Tools"
        assert tools_resource.mimeType == "application/json"
    
    @patch('src.mcp_server.server.get_health_status')
    async def test_read_health_resource(self, mock_health):
        """Test reading health resource"""
        mock_health.return_value = {"status": "healthy", "services": {}}
        
        content = await read_resource("context://health")
        
        assert isinstance(content, str)
        health_data = json.loads(content)
        assert health_data["status"] == "healthy"
        mock_health.assert_called_once()
    
    @patch('src.mcp_server.server.get_tools_info')
    async def test_read_tools_resource(self, mock_tools):
        """Test reading tools resource"""
        mock_tools.return_value = {"tools": [], "server_version": "1.0.0"}
        
        content = await read_resource("context://tools")
        
        assert isinstance(content, str)
        tools_data = json.loads(content)
        assert "tools" in tools_data
        assert "server_version" in tools_data
        mock_tools.assert_called_once()
    
    async def test_read_unknown_resource(self):
        """Test reading unknown resource"""
        with pytest.raises(ValueError, match="Unknown resource"):
            await read_resource("context://unknown")


@pytest.mark.skipif(not SERVER_AVAILABLE, reason="MCP server not available")
@pytest.mark.skipif(not MCP_SDK_AVAILABLE, reason="MCP SDK not available")
@pytest.mark.asyncio
class TestMCPServerTools:
    """Test MCP server tool listing and execution"""
    
    async def test_list_tools(self):
        """Test tool listing"""
        tools = await list_tools()
        
        assert isinstance(tools, list)
        assert len(tools) >= 8  # At least 8 tools defined
        
        # Check required tools exist
        tool_names = [tool.name for tool in tools]
        expected_tools = [
            "store_context", "retrieve_context", "query_graph",
            "update_scratchpad", "get_agent_state", "detect_communities",
            "select_tools", "list_available_tools", "get_tool_info"
        ]
        
        for expected in expected_tools:
            assert expected in tool_names
    
    async def test_tool_schemas_validation(self):
        """Test tool input schema validation"""
        tools = await list_tools()
        
        for tool in tools:
            assert hasattr(tool, 'inputSchema')
            assert isinstance(tool.inputSchema, dict)
            assert "type" in tool.inputSchema
            assert tool.inputSchema["type"] == "object"
            assert "properties" in tool.inputSchema
    
    @patch('src.mcp_server.server.store_context_tool')
    async def test_call_store_context_tool(self, mock_store):
        """Test calling store_context tool"""
        mock_store.return_value = {"success": True, "id": "test_id"}
        
        arguments = {
            "content": {"title": "Test"},
            "type": "design",
            "metadata": {}
        }
        
        result = await call_tool("store_context", arguments)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        
        response_data = json.loads(result[0].text)
        assert response_data["success"] is True
        mock_store.assert_called_once_with(arguments)
    
    @patch('src.mcp_server.server.retrieve_context_tool')
    async def test_call_retrieve_context_tool(self, mock_retrieve):
        """Test calling retrieve_context tool"""
        mock_retrieve.return_value = {"success": True, "results": []}
        
        arguments = {"query": "test query", "limit": 5}
        
        result = await call_tool("retrieve_context", arguments)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        
        response_data = json.loads(result[0].text)
        assert response_data["success"] is True
        mock_retrieve.assert_called_once_with(arguments)
    
    async def test_call_unknown_tool(self):
        """Test calling unknown tool"""
        with pytest.raises(ValueError, match="Unknown tool"):
            await call_tool("unknown_tool", {})


@pytest.mark.skipif(not SERVER_AVAILABLE, reason="MCP server not available")
@pytest.mark.asyncio
class TestStoreContextTool:
    """Test store_context tool implementation"""
    
    @patch('src.mcp_server.server.rate_limit_check')
    @patch('src.mcp_server.server.input_validator')
    @patch('src.mcp_server.server.qdrant_client')
    @patch('src.mcp_server.server.neo4j_client')
    @patch('src.mcp_server.server.embedding_generator')
    async def test_store_context_success(self, mock_embedding, mock_neo4j, 
                                       mock_qdrant, mock_validator, mock_rate_limit):
        """Test successful context storage"""
        # Mock rate limiting
        mock_rate_limit.return_value = (True, None)
        
        # Mock input validation
        mock_validator.validate_input.return_value = Mock(valid=True)
        mock_validator.validate_json_input.return_value = Mock(valid=True)
        
        # Mock embedding generation
        mock_embedding.generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock Qdrant client
        mock_qdrant.client = Mock()
        mock_qdrant.config = {"qdrant": {"collection_name": "test_collection"}}
        mock_qdrant.store_vector = Mock()
        
        # Mock Neo4j client
        mock_neo4j.driver = Mock()
        mock_session = Mock()
        mock_result = Mock()
        mock_record = Mock()
        mock_record.__getitem__ = Mock(return_value="test_node_id")
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j.driver.session.return_value = mock_session
        mock_neo4j.database = "neo4j"
        
        arguments = {
            "content": {"title": "Test Context", "description": "Test description"},
            "type": "design",
            "metadata": {"author": "test"},
            "relationships": [{"type": "implements", "target": "req-001"}]
        }
        
        result = await store_context_tool(arguments)
        
        assert result["success"] is True
        assert "id" in result
        assert result["id"].startswith("ctx_")
        assert "vector_id" in result
        assert "graph_id" in result
        assert "backend_status" in result
    
    @patch('src.mcp_server.server.rate_limit_check')
    async def test_store_context_rate_limited(self, mock_rate_limit):
        """Test context storage with rate limiting"""
        mock_rate_limit.return_value = (False, "Rate limit exceeded")
        
        arguments = {"content": {"title": "Test"}, "type": "design"}
        result = await store_context_tool(arguments)
        
        assert result["success"] is False
        assert "Rate limit exceeded" in result["message"]
        assert result["error_type"] == "rate_limit"
    
    @patch('src.mcp_server.server.rate_limit_check')
    @patch('src.mcp_server.server.input_validator')
    async def test_store_context_validation_failure(self, mock_validator, mock_rate_limit):
        """Test context storage with validation failure"""
        mock_rate_limit.return_value = (True, None)
        
        # Mock validation failure
        mock_validator.validate_input.return_value = Mock(
            valid=False, error="Content too large"
        )
        
        arguments = {"content": {"title": "Test"}, "type": "design"}
        result = await store_context_tool(arguments)
        
        assert result["success"] is False
        assert "Content validation failed" in result["message"]
    
    @patch('src.mcp_server.server.rate_limit_check')
    @patch('src.mcp_server.server.input_validator')
    @patch('src.mcp_server.server.qdrant_client')
    @patch('src.mcp_server.server.embedding_generator')
    async def test_store_context_fallback_embedding(self, mock_embedding, mock_qdrant,
                                                  mock_validator, mock_rate_limit):
        """Test context storage with fallback embedding generation"""
        mock_rate_limit.return_value = (True, None)
        mock_validator.validate_input.return_value = Mock(valid=True)
        mock_validator.validate_json_input.return_value = Mock(valid=True)
        
        # No embedding generator available
        mock_embedding = None
        
        # Mock Qdrant client
        mock_qdrant.client = Mock()
        mock_qdrant.config = {"qdrant": {"collection_name": "test", "dimensions": 384}}
        mock_qdrant.store_vector = Mock()
        
        with patch('src.mcp_server.server.Config') as mock_config:
            mock_config.EMBEDDING_DIMENSIONS = 384
            
            arguments = {"content": {"title": "Test"}, "type": "design"}
            result = await store_context_tool(arguments)
        
        assert result["success"] is True
        # Should use fallback hash-based embedding
        mock_qdrant.store_vector.assert_called_once()


@pytest.mark.skipif(not SERVER_AVAILABLE, reason="MCP server not available")
@pytest.mark.asyncio
class TestRetrieveContextTool:
    """Test retrieve_context tool implementation"""
    
    @patch('src.mcp_server.server.rate_limit_check')
    @patch('src.mcp_server.server.input_validator')
    @patch('src.mcp_server.server.qdrant_client')
    @patch('src.mcp_server.server.neo4j_client')
    @patch('src.mcp_server.server.embedding_generator')
    @patch('src.mcp_server.server.get_reranker')
    async def test_retrieve_context_hybrid_search(self, mock_reranker, mock_embedding, 
                                                 mock_neo4j, mock_qdrant, 
                                                 mock_validator, mock_rate_limit):
        """Test hybrid context retrieval"""
        mock_rate_limit.return_value = (True, None)
        mock_validator.validate_input.return_value = Mock(valid=True)
        
        # Mock embedding generation
        mock_embedding.generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock Qdrant client
        mock_qdrant.client = Mock()
        mock_qdrant.config = {"qdrant": {"collection_name": "test"}}
        mock_qdrant.search.return_value = [
            {"id": "ctx_1", "score": 0.9, "payload": {"content": {"title": "Result 1"}}}
        ]
        
        # Mock Neo4j client
        mock_neo4j.driver = Mock()
        mock_session = Mock()
        mock_neo4j.query.return_value = [
            {"id": "ctx_2", "type": "design", "content": '{"title": "Result 2"}',
             "metadata": '{}', "created_at": "2023-01-01T00:00:00Z"}
        ]
        mock_neo4j.driver.session.return_value = mock_session
        mock_neo4j.database = "neo4j"
        
        # Mock reranker
        mock_reranker_instance = Mock()
        mock_reranker_instance.enabled = False
        mock_reranker.return_value = mock_reranker_instance
        
        arguments = {
            "query": "test query",
            "search_mode": "hybrid",
            "retrieval_mode": "hybrid",
            "limit": 10
        }
        
        result = await retrieve_context_tool(arguments)
        
        assert result["success"] is True
        assert "results" in result
        assert result["search_mode_used"] == "hybrid"
        assert result["retrieval_mode_used"] == "hybrid"
        assert "graphrag_metadata" in result
    
    @patch('src.mcp_server.server.rate_limit_check')
    @patch('src.mcp_server.server.input_validator')
    async def test_retrieve_context_validation_failure(self, mock_validator, mock_rate_limit):
        """Test context retrieval with validation failure"""
        mock_rate_limit.return_value = (True, None)
        mock_validator.validate_input.return_value = Mock(
            valid=False, error="Query too long"
        )
        
        arguments = {"query": "test query"}
        result = await retrieve_context_tool(arguments)
        
        assert result["success"] is False
        assert "Query validation failed" in result["message"]
    
    @patch('src.mcp_server.server.rate_limit_check')
    @patch('src.mcp_server.server.input_validator')
    async def test_retrieve_context_invalid_limit(self, mock_validator, mock_rate_limit):
        """Test context retrieval with invalid limit"""
        mock_rate_limit.return_value = (True, None)
        mock_validator.validate_input.return_value = Mock(valid=True)
        
        arguments = {"query": "test query", "limit": 101}  # Too high
        result = await retrieve_context_tool(arguments)
        
        assert result["success"] is False
        assert "Limit must be an integer between 1 and 100" in result["message"]


@pytest.mark.skipif(not SERVER_AVAILABLE, reason="MCP server not available")
@pytest.mark.asyncio
class TestQueryGraphTool:
    """Test query_graph tool implementation"""
    
    @patch('src.mcp_server.server.rate_limit_check')
    @patch('src.mcp_server.server.cypher_validator')
    @patch('src.mcp_server.server.neo4j_client')
    async def test_query_graph_success(self, mock_neo4j, mock_validator, mock_rate_limit):
        """Test successful graph query"""
        mock_rate_limit.return_value = (True, None)
        
        # Mock query validation
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.warnings = []
        mock_validation_result.complexity_score = 5
        mock_validator.validate_query.return_value = mock_validation_result
        
        # Mock Neo4j client
        mock_neo4j.driver = Mock()
        mock_neo4j.query.return_value = [
            {"n.title": "Test Node", "n.id": "test_id"}
        ]
        
        arguments = {
            "query": "MATCH (n:Context) RETURN n.title, n.id LIMIT 10",
            "parameters": {},
            "limit": 10
        }
        
        result = await query_graph_tool(arguments)
        
        assert result["success"] is True
        assert "results" in result
        assert "row_count" in result
        mock_neo4j.query.assert_called_once()
    
    @patch('src.mcp_server.server.rate_limit_check')
    @patch('src.mcp_server.server.cypher_validator')
    async def test_query_graph_validation_failure(self, mock_validator, mock_rate_limit):
        """Test graph query with validation failure"""
        mock_rate_limit.return_value = (True, None)
        
        # Mock validation failure
        mock_validation_result = Mock()
        mock_validation_result.is_valid = False
        mock_validation_result.error_message = "Write operation not allowed"
        mock_validation_result.error_type = "write_operation"
        mock_validator.validate_query.return_value = mock_validation_result
        
        arguments = {"query": "CREATE (n:Node) RETURN n"}
        result = await query_graph_tool(arguments)
        
        assert result["success"] is False
        assert "Query validation failed" in result["error"]
        assert result["error_type"] == "write_operation"
    
    @patch('src.mcp_server.server.rate_limit_check')
    @patch('src.mcp_server.server.cypher_validator')
    @patch('src.mcp_server.server.neo4j_client')
    async def test_query_graph_readonly_client_fallback(self, mock_neo4j, 
                                                       mock_validator, mock_rate_limit):
        """Test graph query with readonly client fallback"""
        mock_rate_limit.return_value = (True, None)
        
        # Mock query validation
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.warnings = []
        mock_validation_result.complexity_score = 3
        mock_validator.validate_query.return_value = mock_validation_result
        
        # Mock readonly client import failure (fallback to main client)
        with patch('src.mcp_server.server.get_readonly_client', side_effect=ImportError):
            mock_neo4j.driver = Mock()
            mock_neo4j.query.return_value = []
            
            arguments = {"query": "MATCH (n) RETURN n LIMIT 5"}
            result = await query_graph_tool(arguments)
            
            assert result["success"] is True
            mock_neo4j.query.assert_called_once()


@pytest.mark.skipif(not SERVER_AVAILABLE, reason="MCP server not available")
@pytest.mark.asyncio
class TestAgentTools:
    """Test agent-specific tools (scratchpad and state)"""
    
    @patch('src.mcp_server.server.rate_limit_check')
    @patch('src.mcp_server.server.agent_namespace')
    @patch('src.mcp_server.server.input_validator')
    @patch('src.mcp_server.server.kv_store')
    async def test_update_scratchpad_success(self, mock_kv, mock_validator, 
                                           mock_namespace, mock_rate_limit):
        """Test successful scratchpad update"""
        mock_rate_limit.return_value = (True, None)
        mock_namespace.validate_agent_id.return_value = True
        mock_namespace.validate_key.return_value = True
        mock_namespace.create_namespaced_key.return_value = "agent:test_agent:scratchpad:notes"
        mock_validator.validate_input.return_value = Mock(valid=True)
        
        # Mock Redis client
        mock_redis = Mock()
        mock_redis.setex.return_value = True
        mock_kv.redis = mock_redis
        
        arguments = {
            "agent_id": "test_agent",
            "key": "notes",
            "content": "Updated notes",
            "mode": "overwrite",
            "ttl": 3600
        }
        
        result = await update_scratchpad_tool(arguments)
        
        assert result["success"] is True
        assert "key" in result
        assert result["ttl"] == 3600
        mock_redis.setex.assert_called_once()
    
    @patch('src.mcp_server.server.rate_limit_check')
    @patch('src.mcp_server.server.agent_namespace')
    @patch('src.mcp_server.server.input_validator')
    @patch('src.mcp_server.server.kv_store')
    async def test_update_scratchpad_append_mode(self, mock_kv, mock_validator,
                                               mock_namespace, mock_rate_limit):
        """Test scratchpad update in append mode"""
        mock_rate_limit.return_value = (True, None)
        mock_namespace.validate_agent_id.return_value = True
        mock_namespace.validate_key.return_value = True
        mock_namespace.create_namespaced_key.return_value = "agent:test_agent:scratchpad:notes"
        mock_validator.validate_input.return_value = Mock(valid=True)
        
        # Mock Redis client with existing content
        mock_redis = Mock()
        mock_redis.get.return_value = "Existing content"
        mock_redis.setex.return_value = True
        mock_kv.redis = mock_redis
        
        arguments = {
            "agent_id": "test_agent",
            "key": "notes",
            "content": "New content",
            "mode": "append"
        }
        
        result = await update_scratchpad_tool(arguments)
        
        assert result["success"] is True
        # Should append to existing content
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]
        assert "Existing content\nNew content" in call_args[2]
    
    @patch('src.mcp_server.server.rate_limit_check')
    @patch('src.mcp_server.server.agent_namespace')
    @patch('src.mcp_server.server.kv_store')
    async def test_get_agent_state_specific_key(self, mock_kv, mock_namespace, mock_rate_limit):
        """Test retrieving specific agent state key"""
        mock_rate_limit.return_value = (True, None)
        mock_namespace.validate_agent_id.return_value = True
        mock_namespace.validate_prefix.return_value = True
        mock_namespace.validate_key.return_value = True
        mock_namespace.create_namespaced_key.return_value = "agent:test_agent:state:config"
        mock_namespace.verify_agent_access.return_value = True
        
        # Mock Redis client
        mock_redis = Mock()
        mock_redis.get.return_value = "state content"
        mock_kv.redis = mock_redis
        
        arguments = {
            "agent_id": "test_agent",
            "key": "config",
            "prefix": "state"
        }
        
        result = await get_agent_state_tool(arguments)
        
        assert result["success"] is True
        assert result["data"]["key"] == "config"
        assert result["data"]["content"] == "state content"
    
    @patch('src.mcp_server.server.rate_limit_check')
    @patch('src.mcp_server.server.agent_namespace')
    @patch('src.mcp_server.server.kv_store')
    async def test_get_agent_state_all_keys(self, mock_kv, mock_namespace, mock_rate_limit):
        """Test retrieving all agent state keys"""
        mock_rate_limit.return_value = (True, None)
        mock_namespace.validate_agent_id.return_value = True
        mock_namespace.validate_prefix.return_value = True
        
        # Mock Redis client
        mock_redis = Mock()
        mock_redis.keys.return_value = [
            "agent:test_agent:state:config1",
            "agent:test_agent:state:config2"
        ]
        mock_redis.get.side_effect = ["content1", "content2"]
        mock_kv.redis = mock_redis
        
        arguments = {
            "agent_id": "test_agent",
            "prefix": "state"
        }
        
        result = await get_agent_state_tool(arguments)
        
        assert result["success"] is True
        assert "data" in result
        assert "keys" in result
        assert len(result["keys"]) == 2


@pytest.mark.skipif(not SERVER_AVAILABLE, reason="MCP server not available")
@pytest.mark.asyncio
class TestHealthAndUtilityTools:
    """Test health checking and utility tools"""
    
    @patch('src.mcp_server.server.neo4j_client')
    @patch('src.mcp_server.server.qdrant_client')
    @patch('src.mcp_server.server.kv_store')
    async def test_get_health_status_all_healthy(self, mock_kv, mock_qdrant, mock_neo4j):
        """Test health status with all services healthy"""
        # Mock healthy Neo4j
        mock_neo4j.driver = Mock()
        mock_neo4j.query.return_value = [{"result": 1}]
        
        # Mock healthy Qdrant
        mock_qdrant.client = Mock()
        mock_qdrant.get_collections.return_value = ["collection1"]
        
        # Mock healthy Redis
        mock_redis = Mock()
        mock_redis.redis_client.ping.return_value = True
        mock_kv.redis = mock_redis
        
        health = await get_health_status()
        
        assert health["status"] == "healthy"
        assert health["services"]["neo4j"] == "healthy"
        assert health["services"]["qdrant"] == "healthy"
        assert health["services"]["redis"] == "healthy"
    
    @patch('src.mcp_server.server.neo4j_client')
    @patch('src.mcp_server.server.qdrant_client')
    @patch('src.mcp_server.server.kv_store')
    async def test_get_health_status_degraded(self, mock_kv, mock_qdrant, mock_neo4j):
        """Test health status with some services unhealthy"""
        # Mock unhealthy Neo4j
        mock_neo4j.driver = Mock()
        mock_neo4j.query.side_effect = Exception("Connection failed")
        
        # Mock healthy Qdrant
        mock_qdrant.client = Mock()
        mock_qdrant.get_collections.return_value = ["collection1"]
        
        # Mock Redis not available
        mock_kv = None
        
        health = await get_health_status()
        
        assert health["status"] == "degraded"
        assert health["services"]["neo4j"] == "unhealthy"
        assert health["services"]["qdrant"] == "healthy"
    
    @patch('src.mcp_server.server.Path')
    async def test_get_tools_info(self, mock_path):
        """Test tools info retrieval"""
        # Mock contract files
        mock_contracts_dir = Mock()
        mock_contracts_dir.exists.return_value = True
        mock_contracts_dir.glob.return_value = []
        
        mock_path.return_value.parent.parent.parent = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__ = Mock(return_value=mock_contracts_dir)
        
        tools_info = await get_tools_info()
        
        assert "tools" in tools_info
        assert "server_version" in tools_info
        assert "mcp_version" in tools_info
    
    @patch('src.mcp_server.server.tool_selector_bridge')
    @patch('src.mcp_server.server.rate_limit_check')
    async def test_select_tools_tool_success(self, mock_rate_limit, mock_bridge):
        """Test tool selection success"""
        mock_rate_limit.return_value = (True, None)
        mock_bridge.select_tools.return_value = {
            "success": True,
            "tools": [{"name": "tool1", "score": 0.9}],
            "message": "Tools selected"
        }
        
        arguments = {"query": "test query", "max_tools": 5}
        result = await select_tools_tool(arguments)
        
        assert result["success"] is True
        assert "tools" in result
        mock_bridge.select_tools.assert_called_once_with(arguments)
    
    @patch('src.mcp_server.server.tool_selector_bridge')
    async def test_select_tools_tool_bridge_unavailable(self, mock_bridge):
        """Test tool selection with bridge unavailable"""
        mock_bridge = None
        
        # Temporarily set bridge to None
        with patch('src.mcp_server.server.tool_selector_bridge', None):
            arguments = {"query": "test query"}
            result = await select_tools_tool(arguments)
            
            assert result["success"] is False
            assert result["error_type"] == "bridge_unavailable"
    
    @patch('src.mcp_server.server.rate_limit_check')
    async def test_detect_communities_tool_import_error(self, mock_rate_limit):
        """Test community detection with import error"""
        mock_rate_limit.return_value = (True, None)
        
        # Mock import error for GraphRAG bridge
        with patch('src.mcp_server.server.get_graphrag_bridge', side_effect=ImportError):
            arguments = {"algorithm": "louvain"}
            result = await detect_communities_tool(arguments)
            
            assert result["success"] is False
            assert result["error_type"] == "import_error"


@pytest.mark.skipif(not SERVER_AVAILABLE, reason="MCP server not available")
@pytest.mark.asyncio
class TestServerMainAndIntegration:
    """Test server main function and integration scenarios"""
    
    @patch('src.mcp_server.server.initialize_storage_clients')
    @patch('src.mcp_server.server.cleanup_storage_clients')
    @patch('src.mcp_server.server.stdio_server')
    @patch('src.mcp_server.server.server')
    async def test_main_function_success(self, mock_server, mock_stdio, 
                                       mock_cleanup, mock_initialize):
        """Test main function successful execution"""
        # Mock storage initialization
        mock_initialize.return_value = {"success": True}
        
        # Mock stdio server context manager
        mock_read_stream = Mock()
        mock_write_stream = Mock()
        mock_stdio_context = Mock()
        mock_stdio_context.__aenter__ = AsyncMock(return_value=(mock_read_stream, mock_write_stream))
        mock_stdio_context.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value = mock_stdio_context
        
        # Mock server run
        mock_server.run = AsyncMock()
        
        await main()
        
        mock_initialize.assert_called_once()
        mock_server.run.assert_called_once()
        mock_cleanup.assert_called_once()
    
    @patch('src.mcp_server.server.initialize_storage_clients')
    @patch('src.mcp_server.server.cleanup_storage_clients')
    @patch('src.mcp_server.server.stdio_server')
    @patch('src.mcp_server.server.server')
    async def test_main_function_with_exception(self, mock_server, mock_stdio,
                                              mock_cleanup, mock_initialize):
        """Test main function with exception during execution"""
        mock_initialize.return_value = {"success": True}
        
        # Mock stdio server to raise exception
        mock_stdio.side_effect = Exception("Server error")
        
        # Should still call cleanup even with exception
        try:
            await main()
        except Exception:
            pass
        
        mock_cleanup.assert_called_once()
    
    @patch('src.mcp_server.server.rate_limit_check')
    @patch('src.mcp_server.server.input_validator')
    async def test_tool_rate_limiting_integration(self, mock_validator, mock_rate_limit):
        """Test rate limiting integration across tools"""
        mock_rate_limit.return_value = (False, "Too many requests")
        mock_validator.validate_input.return_value = Mock(valid=True)
        
        # Test multiple tools respect rate limiting
        tools_to_test = [
            ("store_context", {"content": {"title": "Test"}, "type": "design"}),
            ("retrieve_context", {"query": "test"}),
            ("update_scratchpad", {"agent_id": "test", "key": "test", "content": "test"}),
            ("get_agent_state", {"agent_id": "test"})
        ]
        
        for tool_name, arguments in tools_to_test:
            if tool_name == "store_context":
                result = await store_context_tool(arguments)
            elif tool_name == "retrieve_context":
                result = await retrieve_context_tool(arguments)
            elif tool_name == "update_scratchpad":
                result = await update_scratchpad_tool(arguments)
            elif tool_name == "get_agent_state":
                result = await get_agent_state_tool(arguments)
            
            assert result["success"] is False
            assert result["error_type"] == "rate_limit"
    
    async def test_error_handling_consistency(self):
        """Test consistent error handling across tools"""
        # Test that all tools handle missing arguments gracefully
        test_cases = [
            (store_context_tool, {}),  # Missing required fields
            (retrieve_context_tool, {}),  # Missing query
            (update_scratchpad_tool, {}),  # Missing required fields
            (get_agent_state_tool, {}),  # Missing agent_id
        ]
        
        for tool_func, empty_args in test_cases:
            try:
                result = await tool_func(empty_args)
                # Should return error result, not raise exception
                assert isinstance(result, dict)
                assert "success" in result
                assert result["success"] is False
            except Exception as e:
                # Some tools might raise KeyError for missing required fields
                assert isinstance(e, KeyError)
    
    async def test_response_format_consistency(self):
        """Test consistent response format across tools"""
        # Mock successful responses
        with patch('src.mcp_server.server.rate_limit_check', return_value=(True, None)):
            with patch('src.mcp_server.server.input_validator') as mock_validator:
                mock_validator.validate_input.return_value = Mock(valid=True)
                mock_validator.validate_json_input.return_value = Mock(valid=True)
                
                # All tools should return dict with 'success' field
                test_tools = [
                    (store_context_tool, {"content": {"title": "Test"}, "type": "design"}),
                    (retrieve_context_tool, {"query": "test"}),
                ]
                
                for tool_func, args in test_tools:
                    with patch.multiple(
                        'src.mcp_server.server',
                        qdrant_client=Mock(client=None),
                        neo4j_client=Mock(driver=None),
                        embedding_generator=None
                    ):
                        result = await tool_func(args)
                        assert isinstance(result, dict)
                        assert "success" in result
                        assert "message" in result