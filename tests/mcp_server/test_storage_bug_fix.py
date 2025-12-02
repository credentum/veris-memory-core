#!/usr/bin/env python3
"""
Tests for storage bug fix - ensuring data persistence and retrieval.

Tests for issue #61: Storage/Retrieval Mechanism Broken
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# Import the functions we're testing
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.mcp_server.server import store_context_tool, retrieve_context_tool


class TestStorageBugFix:
    """Test storage and retrieval functionality after bug fix."""
    
    @pytest.fixture
    def sample_context(self):
        """Sample context data for testing."""
        return {
            "content": {
                "title": "Test Storage Bug Fix",
                "description": "Testing that storage actually persists data",
                "unique_marker": f"TEST_MARKER_{uuid.uuid4().hex[:8]}"
            },
            "type": "trace",
            "metadata": {
                "test_case": "storage_bug_fix",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    @pytest.mark.asyncio
    async def test_storage_returns_context_id_not_id(self, sample_context):
        """Test that storage returns context_id field, not id field."""
        with patch('src.mcp_server.server.qdrant_client') as mock_qdrant, \
             patch('src.mcp_server.server.neo4j_client') as mock_neo4j, \
             patch('src.mcp_server.server.rate_limit_check', return_value=(True, None)):
            
            # Mock successful storage
            mock_qdrant.client = Mock()
            mock_qdrant.client.retrieve.return_value = [Mock()]  # Verify returns data
            mock_qdrant.store_vector = Mock()
            mock_qdrant.collection_name = "test_collection"
            
            mock_neo4j.driver = Mock()
            mock_session = Mock()
            mock_neo4j.driver.session.return_value = mock_session
            mock_session.__enter__ = Mock(return_value=mock_session)
            mock_session.__exit__ = Mock(return_value=None)
            mock_result = Mock()
            mock_result.single.return_value = {"node_id": "test_graph_id"}
            mock_session.run.return_value = mock_result
            
            # Store context
            result = await store_context_tool(sample_context)
            
            # Verify response structure
            assert result["success"] is True
            assert "context_id" in result  # Should be context_id, not id
            assert "id" not in result  # Should NOT have id field
            assert result["context_id"] is not None
            assert result["context_id"].startswith("ctx_")
    
    @pytest.mark.asyncio
    async def test_storage_fails_when_no_backends_succeed(self, sample_context):
        """Test that storage reports failure when no backends succeed."""
        with patch('src.mcp_server.server.qdrant_client', None), \
             patch('src.mcp_server.server.neo4j_client', None), \
             patch('src.mcp_server.server.rate_limit_check', return_value=(True, None)):
            
            # Store context with no backends
            result = await store_context_tool(sample_context)
            
            # Verify failure response
            assert result["success"] is False
            assert result["context_id"] is None
            assert "Storage failed" in result["message"]
            assert result["error_type"] == "storage_failure"
    
    @pytest.mark.asyncio 
    async def test_storage_succeeds_with_one_backend(self, sample_context):
        """Test that storage succeeds when at least one backend works."""
        with patch('src.mcp_server.server.qdrant_client') as mock_qdrant, \
             patch('src.mcp_server.server.neo4j_client', None), \
             patch('src.mcp_server.server.rate_limit_check', return_value=(True, None)):
            
            # Mock only Qdrant working
            mock_qdrant.client = Mock()
            mock_qdrant.client.retrieve.return_value = [Mock()]  # Verify returns data
            mock_qdrant.store_vector = Mock()
            
            # Store context
            result = await store_context_tool(sample_context)
            
            # Verify success with warning
            assert result["success"] is True
            assert result["context_id"] is not None
            assert "warning" in result["message"]
            assert result["backend_status"]["vector"] == "success"
            assert result["backend_status"]["graph"] == "failed"
    
    @pytest.mark.asyncio
    async def test_vector_storage_verification(self, sample_context):
        """Test that vector storage is verified after write."""
        with patch('src.mcp_server.server.qdrant_client') as mock_qdrant, \
             patch('src.mcp_server.server.neo4j_client', None), \
             patch('src.mcp_server.server.rate_limit_check', return_value=(True, None)), \
             patch('time.sleep'):  # Mock sleep to speed up test
            
            # Mock Qdrant client
            mock_qdrant.client = Mock()
            mock_qdrant.store_vector = Mock()
            mock_qdrant.collection_name = "test_collection"
            
            # Test: Storage verification succeeds
            mock_qdrant.client.retrieve.return_value = [Mock()]  # Data found
            
            result = await store_context_tool(sample_context)
            
            # Verify storage was checked
            mock_qdrant.client.retrieve.assert_called_once()
            assert result["success"] is True
            assert result["vector_id"] is not None
    
    @pytest.mark.asyncio
    async def test_vector_storage_verification_fails(self, sample_context):
        """Test that storage fails when verification shows data wasn't written."""
        with patch('src.mcp_server.server.qdrant_client') as mock_qdrant, \
             patch('src.mcp_server.server.neo4j_client', None), \
             patch('src.mcp_server.server.rate_limit_check', return_value=(True, None)), \
             patch('time.sleep'):  # Mock sleep to speed up test
            
            # Mock Qdrant client
            mock_qdrant.client = Mock()
            mock_qdrant.store_vector = Mock()
            mock_qdrant.collection_name = "test_collection"
            
            # Test: Storage verification fails - no data found
            mock_qdrant.client.retrieve.return_value = []  # No data found
            
            result = await store_context_tool(sample_context)
            
            # Verify storage was checked and failed
            mock_qdrant.client.retrieve.assert_called_once()
            assert result["success"] is False  # Should fail when no backends succeed
            assert result["context_id"] is None
    
    @pytest.mark.asyncio
    async def test_exception_handling_returns_context_id(self, sample_context):
        """Test that exceptions return context_id field, not id field."""
        with patch('src.mcp_server.server.rate_limit_check', return_value=(True, None)), \
             patch('uuid.uuid4') as mock_uuid:
            
            # Force an exception during processing
            mock_uuid.side_effect = Exception("Test exception")
            
            result = await store_context_tool(sample_context)
            
            # Verify error response structure
            assert result["success"] is False
            assert "context_id" in result  # Should be context_id, not id
            assert "id" not in result  # Should NOT have id field
            assert result["context_id"] is None
            assert result["error_type"] == "exception"


class TestStorageRetrieval:
    """Test that stored data can be immediately retrieved."""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_cycle(self):
        """Test complete store/retrieve cycle works."""
        unique_marker = f"CYCLE_TEST_{uuid.uuid4().hex[:8]}"
        
        # Test data with unique identifier
        store_data = {
            "content": {
                "title": "Store and Retrieve Test",
                "unique_marker": unique_marker,
                "test_type": "full_cycle"
            },
            "type": "trace",
            "metadata": {"test": "store_retrieve_cycle"}
        }
        
        with patch('src.mcp_server.server.qdrant_client') as mock_qdrant, \
             patch('src.mcp_server.server.neo4j_client') as mock_neo4j, \
             patch('src.mcp_server.server.rate_limit_check', return_value=(True, None)), \
             patch('time.sleep'):
            
            # Mock successful storage and verification
            mock_qdrant.client = Mock()
            mock_qdrant.client.retrieve.return_value = [Mock()]
            mock_qdrant.store_vector = Mock()
            mock_qdrant.collection_name = "test_collection"
            
            mock_neo4j.driver = Mock()
            mock_session = Mock()
            mock_neo4j.driver.session.return_value = mock_session
            mock_session.__enter__ = Mock(return_value=mock_session)
            mock_session.__exit__ = Mock(return_value=None)
            mock_result = Mock()
            mock_result.single.return_value = {"node_id": "test_id"}
            mock_session.run.return_value = mock_result
            
            # Store the data
            store_result = await store_context_tool(store_data)
            assert store_result["success"] is True
            context_id = store_result["context_id"]
            
            # Mock retrieval to return the stored data
            with patch('src.mcp_server.server.qdrant_client') as mock_qdrant_search:
                mock_qdrant_search.client = Mock()
                mock_qdrant_search.client.search.return_value = [
                    Mock(id=context_id, payload={
                        "content": store_data["content"],
                        "type": store_data["type"], 
                        "metadata": store_data["metadata"]
                    }, score=0.95)
                ]
                
                # Retrieve the data
                retrieve_result = await retrieve_context_tool({
                    "query": unique_marker,
                    "limit": 10
                })
                
                # Verify retrieval works
                assert retrieve_result["success"] is True
                assert len(retrieve_result["results"]) > 0
                
                # Find our stored context in results
                found = False
                for result in retrieve_result["results"]:
                    if unique_marker in str(result.get("content", {})):
                        found = True
                        break
                
                assert found, f"Stored data with marker {unique_marker} not found in retrieval results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])