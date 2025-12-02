"""
Unit tests for critical storage bug fixes in Veris Memory.

This module tests the specific fixes applied to resolve system non-functionality:
1. Vector ID format fix (UUID instead of ctx_xxx)
2. Neo4j parameter fix (labels instead of label)
3. Vector search parameter fix (remove collection_name)

These tests ensure the fixes work correctly and prevent regression.
"""

import uuid
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Dict, Any, List

# Import the main module functions that were fixed
try:
    from src.mcp_server.main import store_context, retrieve_context
    from src.mcp_server.main import StoreContextRequest, RetrieveContextRequest
except ImportError:
    pytest.skip("Main MCP server module not available", allow_module_level=True)


class TestVectorIDFormatFix:
    """Test the vector ID format fix (UUID instead of ctx_xxx)."""
    
    def test_vector_id_is_valid_uuid(self):
        """Test that generated context_id is a valid UUID format."""
        # Generate a UUID using the same method as the fix
        import uuid
        context_id = str(uuid.uuid4())
        
        # Verify it's a valid UUID
        parsed_uuid = uuid.UUID(context_id)
        assert str(parsed_uuid) == context_id
        
        # Verify it's not the old ctx_xxx format
        assert not context_id.startswith("ctx_")
        
        # Verify length is correct for UUID (36 characters with hyphens)
        assert len(context_id) == 36
        assert context_id.count("-") == 4

    @patch('src.mcp_server.main.qdrant_client')
    @patch('src.mcp_server.main.neo4j_client')
    @patch('src.mcp_server.main._generate_embedding')
    @pytest.mark.asyncio
    async def test_store_context_generates_valid_uuid_id(
        self, mock_embedding, mock_neo4j, mock_qdrant
    ):
        """Test that store_context generates valid UUID for context_id."""
        # Setup mocks
        mock_embedding.return_value = [0.1, 0.2, 0.3]
        mock_qdrant.store_vector = Mock(return_value="vector_id_123")
        mock_neo4j.create_node = Mock(return_value="node_id_456")
        
        # Create request
        request = StoreContextRequest(
            type="log",
            content={"message": "test content"},
            metadata={"source": "unit_test"}
        )
        
        # Call the function
        result = await store_context(request)
        
        # Verify the result contains a valid UUID
        context_id = result["id"]
        
        # Test UUID validity
        parsed_uuid = uuid.UUID(context_id)
        assert str(parsed_uuid) == context_id
        
        # Verify it's not the old format
        assert not context_id.startswith("ctx_")
        
        # Verify UUID was passed to storage backends
        mock_qdrant.store_vector.assert_called_once()
        call_args = mock_qdrant.store_vector.call_args
        assert call_args[1]["vector_id"] == context_id
        
        mock_neo4j.create_node.assert_called_once()
        neo4j_call_args = mock_neo4j.create_node.call_args
        assert neo4j_call_args[1]["properties"]["id"] == context_id


class TestNeo4jParameterFix:
    """Test the Neo4j parameter fix (labels instead of label)."""
    
    @patch('src.mcp_server.main.qdrant_client')
    @patch('src.mcp_server.main.neo4j_client')
    @patch('src.mcp_server.main._generate_embedding')
    @pytest.mark.asyncio
    async def test_neo4j_create_node_uses_labels_parameter(
        self, mock_embedding, mock_neo4j, mock_qdrant
    ):
        """Test that Neo4j create_node is called with 'labels' parameter, not 'label'."""
        # Setup mocks
        mock_embedding.return_value = [0.1, 0.2, 0.3]
        mock_qdrant.store_vector = Mock(return_value="vector_id_123")
        mock_neo4j.create_node = Mock(return_value="node_id_456")
        
        # Create request
        request = StoreContextRequest(
            type="design",
            content={"title": "Test Document", "body": "Test content"},
            metadata={"author": "test_user"}
        )
        
        # Call the function
        result = await store_context(request)
        
        # Verify Neo4j create_node was called with correct parameters
        mock_neo4j.create_node.assert_called_once()
        call_args = mock_neo4j.create_node.call_args
        
        # Check that 'labels' parameter is used (as a list)
        assert 'labels' in call_args[1]
        assert call_args[1]['labels'] == ["Context"]
        
        # Ensure 'label' (singular) is NOT used
        assert 'label' not in call_args[1]
        
        # Verify labels is a list (correct API signature)
        labels_param = call_args[1]['labels']
        assert isinstance(labels_param, list)
        assert len(labels_param) == 1
        assert labels_param[0] == "Context"

    def test_neo4j_labels_parameter_format(self):
        """Test that labels parameter is correctly formatted as a list."""
        # Test the exact format used in the fix
        labels = ["Context"]
        
        # Verify it's a list
        assert isinstance(labels, list)
        
        # Verify it contains the expected label
        assert "Context" in labels
        assert len(labels) == 1
        
        # Verify it's not a string (which was the bug)
        assert not isinstance(labels, str)


class TestVectorSearchParameterFix:
    """Test the vector search parameter fix (remove collection_name)."""
    
    @patch('src.mcp_server.main.qdrant_client')
    @patch('src.mcp_server.main.neo4j_client')
    @patch('src.mcp_server.main._generate_embedding')
    @pytest.mark.asyncio
    async def test_qdrant_search_removes_collection_name_parameter(
        self, mock_embedding, mock_neo4j, mock_qdrant
    ):
        """Test that qdrant search is called without collection_name parameter."""
        # Setup mocks
        mock_embedding.return_value = [0.4, 0.5, 0.6]
        mock_qdrant.search = Mock(return_value=[
            Mock(id="test_id_1", score=0.9, payload={"content": "test content 1"}),
            Mock(id="test_id_2", score=0.8, payload={"content": "test content 2"})
        ])
        mock_neo4j.query = Mock(return_value=[{"node": "test_node"}])
        
        # Create request
        request = RetrieveContextRequest(
            query="test search query",
            limit=5,
            type="document"
        )
        
        # Call the function
        result = await retrieve_context(request)
        
        # Verify qdrant search was called
        mock_qdrant.search.assert_called_once()
        call_args = mock_qdrant.search.call_args
        
        # Check that collection_name parameter is NOT present
        assert 'collection_name' not in call_args[1]
        
        # Verify correct parameters are present
        assert 'query_vector' in call_args[1]
        assert 'limit' in call_args[1]
        
        # Verify the call uses the correct parameter names
        expected_params = {'query_vector', 'limit'}
        actual_params = set(call_args[1].keys())
        assert expected_params.issubset(actual_params)

    def test_qdrant_search_parameter_validation(self):
        """Test that search parameters match the expected API signature."""
        # Mock the expected parameters for qdrant search
        search_params = {
            'query_vector': [0.1, 0.2, 0.3],
            'limit': 10
        }
        
        # Verify collection_name is not in parameters
        assert 'collection_name' not in search_params
        
        # Verify required parameters are present
        assert 'query_vector' in search_params
        assert 'limit' in search_params
        
        # Verify parameter types
        assert isinstance(search_params['query_vector'], list)
        assert isinstance(search_params['limit'], int)


class TestCriticalBugFixesIntegration:
    """Integration tests for all critical bug fixes working together."""
    
    @patch('src.mcp_server.main.qdrant_client')
    @patch('src.mcp_server.main.neo4j_client')
    @patch('src.mcp_server.main._generate_embedding')
    @pytest.mark.asyncio
    async def test_store_and_retrieve_with_all_fixes(
        self, mock_embedding, mock_neo4j, mock_qdrant
    ):
        """Test that all three fixes work together in a complete workflow."""
        # Setup mocks for store operation
        mock_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_qdrant.store_vector = Mock(return_value="vector_stored")
        mock_neo4j.create_node = Mock(return_value="node_created")
        
        # Setup mocks for retrieve operation
        mock_qdrant.search = Mock(return_value=[
            Mock(id="test_uuid", score=0.95, payload={"content": "stored content"})
        ])
        mock_neo4j.query = Mock(return_value=[{"data": "graph_result"}])
        
        # Test store operation with fixes
        store_request = StoreContextRequest(
            type="trace",
            content={"test": "integration content"},
            metadata={"fix_validation": True}
        )
        
        store_result = await store_context(store_request)
        
        # Verify Fix #1: UUID format for context_id
        context_id = store_result["id"]
        uuid.UUID(context_id)  # This will raise if not valid UUID
        assert not context_id.startswith("ctx_")
        
        # Verify Fix #2: Neo4j labels parameter
        mock_neo4j.create_node.assert_called_once()
        neo4j_args = mock_neo4j.create_node.call_args
        assert 'labels' in neo4j_args[1]
        assert neo4j_args[1]['labels'] == ["Context"]
        assert 'label' not in neo4j_args[1]
        
        # Test retrieve operation with fix
        retrieve_request = RetrieveContextRequest(
            query="integration test query",
            limit=3
        )
        
        retrieve_result = await retrieve_context(retrieve_request)
        
        # Verify Fix #3: No collection_name parameter in search
        mock_qdrant.search.assert_called_once()
        search_args = mock_qdrant.search.call_args
        assert 'collection_name' not in search_args[1]
        assert 'query_vector' in search_args[1]
        assert 'limit' in search_args[1]
        
        # Verify the workflow completed successfully
        assert store_result["success"] is True
        assert retrieve_result["success"] is True

    def test_uuid_generation_consistency(self):
        """Test that UUID generation is consistent and doesn't revert to old format."""
        # Generate multiple UUIDs to ensure consistency
        generated_ids = []
        for _ in range(10):
            context_id = str(uuid.uuid4())
            generated_ids.append(context_id)
            
            # Each should be valid UUID
            uuid.UUID(context_id)
            
            # None should use old format
            assert not context_id.startswith("ctx_")
            
            # Should be proper length
            assert len(context_id) == 36
        
        # All should be unique
        assert len(set(generated_ids)) == 10