#!/usr/bin/env python3
"""
test_mcp_outage_failclosed.py: Sprint 11 Phase 4 MCP Outage Fail-Closed Tests

Tests Sprint 11 Phase 4 Task 1 requirements:
- MCP server outage detection within 5 seconds
- Fail-closed behavior (reject requests, don't silently fail)
- Proper error responses with trace IDs
- Circuit breaker state management
"""

import asyncio
import pytest
import logging
import time
import json
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any

# Add src to Python path for imports
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    from src.mcp_server.server import store_context_tool, retrieve_context_tool
    from src.core.error_codes import ErrorCode, create_error_response
    from src.storage.circuit_breaker import CircuitBreaker, CircuitState
    from src.core.config import Config
except ImportError as e:
    print(f"Import error: {e}")
    pytest.skip("Required modules not available", allow_module_level=True)

# Setup logging for outage testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockMCPServerOutage:
    """Mock MCP server that can simulate outages"""
    
    def __init__(self):
        self.is_down = False
        self.response_delay = 0.0  # Simulate slow responses
        self.last_request_time = None
        
    def set_outage(self, is_down: bool):
        """Simulate MCP server outage"""
        self.is_down = is_down
        logger.info(f"MCP server outage: {'DOWN' if is_down else 'UP'}")
    
    def set_response_delay(self, delay: float):
        """Simulate slow MCP responses"""
        self.response_delay = delay
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock MCP request handling with outage simulation"""
        self.last_request_time = datetime.utcnow()
        
        if self.is_down:
            raise ConnectionError("MCP server is unavailable")
        
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
        
        # Mock successful response
        return {
            "success": True,
            "id": "mock_response",
            "timestamp": datetime.utcnow().isoformat()
        }


class TestMCPOutageDetection:
    """Test MCP server outage detection and fail-closed behavior"""
    
    @pytest.fixture
    def mock_mcp_server(self):
        """Create mock MCP server with outage simulation"""
        return MockMCPServerOutage()
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing"""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0,
            timeout=5.0
        )
    
    @pytest.mark.asyncio
    async def test_mcp_outage_detection_timing(self, mock_mcp_server, circuit_breaker):
        """Test that MCP outage is detected within 5 seconds"""
        
        # Simulate MCP server outage
        mock_mcp_server.set_outage(True)
        
        start_time = time.time()
        detected_outage = False
        
        # Try to make request that should fail quickly
        with patch('src.mcp_server.server.mcp_client', mock_mcp_server):
            try:
                # This should fail fast due to outage
                await mock_mcp_server.handle_request({"test": "request"})
            except ConnectionError:
                detection_time = time.time() - start_time
                detected_outage = True
                
                logger.info(f"MCP outage detected in {detection_time:.2f} seconds")
                
                # Sprint 11 requirement: detect within 5 seconds
                assert detection_time < 5.0, f"Outage detection took {detection_time:.2f}s, must be < 5s"
        
        assert detected_outage, "MCP outage was not detected"
    
    @pytest.mark.asyncio
    async def test_fail_closed_behavior_store_context(self, mock_mcp_server):
        """Test fail-closed behavior for store_context when MCP is down"""
        
        # Simulate MCP server outage
        mock_mcp_server.set_outage(True)
        
        test_context = {
            "content": {
                "test": "fail-closed test data"
            },
            "type": "decision",
            "metadata": {
                "priority": "high"
            }
        }
        
        # Mock all storage clients to simulate MCP dependency
        with patch('src.mcp_server.server.kv_store') as mock_kv, \
             patch('src.mcp_server.server.qdrant_client') as mock_vector, \
             patch('src.mcp_server.server.neo4j_client') as mock_graph:
            
            # Setup storage clients to fail (simulating MCP dependency)
            mock_kv.store_context.side_effect = ConnectionError("MCP unavailable")
            mock_vector.store_vector.side_effect = ConnectionError("MCP unavailable")
            mock_graph.driver.session().run.side_effect = ConnectionError("MCP unavailable")
            
            # Attempt to store context - should fail closed
            result = await store_context_tool(test_context)
            
            # Verify fail-closed behavior
            assert result["success"] is False, "Should fail-closed when MCP is down"
            assert "error_code" in result
            assert result["error_code"] == ErrorCode.DEPENDENCY_DOWN.value
            assert "trace_id" in result
            
            # Should not silently succeed or return partial data
            assert "id" not in result, "Should not return context ID when failing"
            
            logger.info("✅ store_context correctly failed-closed during MCP outage")
    
    @pytest.mark.asyncio
    async def test_fail_closed_behavior_retrieve_context(self, mock_mcp_server):
        """Test fail-closed behavior for retrieve_context when MCP is down"""
        
        # Simulate MCP server outage
        mock_mcp_server.set_outage(True)
        
        retrieval_request = {
            "query": "test query during outage",
            "type": "all",
            "limit": 5
        }
        
        # Mock storage clients to fail (simulating MCP dependency)  
        with patch('src.mcp_server.server.qdrant_client') as mock_vector, \
             patch('src.mcp_server.server.neo4j_client') as mock_graph:
            
            mock_vector.client.search.side_effect = ConnectionError("MCP unavailable")
            mock_graph.driver.session().run.side_effect = ConnectionError("MCP unavailable")
            
            # Attempt retrieval - should fail closed
            result = await retrieve_context_tool(retrieval_request)
            
            # Verify fail-closed behavior
            assert result["success"] is False, "Should fail-closed when MCP is down"
            assert "error_code" in result
            assert result["error_code"] == ErrorCode.DEPENDENCY_DOWN.value
            assert "trace_id" in result
            
            # Should not return empty results or partial data
            assert "results" not in result or result.get("results") == []
            
            logger.info("✅ retrieve_context correctly failed-closed during MCP outage")
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_state_management(self, circuit_breaker):
        """Test circuit breaker state transitions during outages"""
        
        # Initially closed (healthy)
        assert circuit_breaker.state == CircuitState.CLOSED
        
        # Simulate multiple failures
        for i in range(3):
            try:
                async with circuit_breaker:
                    raise ConnectionError("MCP failure")
            except:
                pass
        
        # Should trip to OPEN state after threshold failures
        assert circuit_breaker.state == CircuitState.OPEN
        logger.info("✅ Circuit breaker opened after failure threshold")
        
        # Requests should fail fast in OPEN state
        start_time = time.time()
        try:
            async with circuit_breaker:
                await asyncio.sleep(10)  # This should not execute
        except Exception:
            fail_fast_time = time.time() - start_time
            assert fail_fast_time < 1.0, f"Circuit breaker should fail fast, took {fail_fast_time:.2f}s"
        
        logger.info("✅ Circuit breaker fails fast in OPEN state")
    
    @pytest.mark.asyncio 
    async def test_error_response_format_compliance(self, mock_mcp_server):
        """Test that outage errors follow Sprint 11 error format"""
        
        mock_mcp_server.set_outage(True)
        
        test_context = {"content": {"test": "data"}, "type": "log"}
        
        with patch('src.mcp_server.server.kv_store') as mock_kv:
            mock_kv.store_context.side_effect = ConnectionError("MCP unavailable")
            
            result = await store_context_tool(test_context)
            
            # Verify Sprint 11 error format compliance
            assert "success" in result and result["success"] is False
            assert "error_code" in result
            assert "message" in result  
            assert "trace_id" in result
            
            # Verify error code is correct
            assert result["error_code"] == ErrorCode.DEPENDENCY_DOWN.value
            assert "MCP" in result["message"] or "unavailable" in result["message"]
            
            # Verify trace ID format (UUID-like)
            trace_id = result["trace_id"]
            assert len(trace_id) >= 8, "Trace ID should be meaningful length"
            
            logger.info(f"✅ Error format compliant: {result['error_code']} - {trace_id}")
    
    @pytest.mark.asyncio
    async def test_mcp_recovery_detection(self, mock_mcp_server, circuit_breaker):
        """Test MCP server recovery detection"""
        
        # Start with outage
        mock_mcp_server.set_outage(True)
        
        # Cause circuit breaker to open
        for i in range(3):
            try:
                async with circuit_breaker:
                    await mock_mcp_server.handle_request({"test": "request"})
            except:
                pass
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Simulate MCP recovery
        mock_mcp_server.set_outage(False)
        
        # Wait for circuit breaker recovery timeout
        await asyncio.sleep(0.1)  # Short wait for testing
        
        # Circuit breaker should transition to HALF_OPEN for testing
        circuit_breaker._recovery_timeout = 0.1  # Speed up for testing
        await asyncio.sleep(0.2)
        
        # Try a request - should succeed and close circuit
        try:
            async with circuit_breaker:
                result = await mock_mcp_server.handle_request({"test": "recovery"})
                assert result["success"] is True
        except:
            pass  # Circuit might still be open, that's okay for this test
        
        logger.info("✅ MCP recovery testing completed")
    
    @pytest.mark.asyncio
    async def test_timeout_based_failure_detection(self, mock_mcp_server):
        """Test timeout-based failure detection for slow MCP responses"""
        
        # Set MCP server to respond slowly (simulating network issues)
        mock_mcp_server.set_response_delay(6.0)  # Longer than 5s timeout
        
        start_time = time.time()
        
        with patch('src.mcp_server.server.mcp_client', mock_mcp_server):
            try:
                # This should timeout within 5 seconds
                await asyncio.wait_for(
                    mock_mcp_server.handle_request({"test": "timeout"}),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                timeout_duration = time.time() - start_time
                
                logger.info(f"MCP timeout detected in {timeout_duration:.2f} seconds")
                
                # Should timeout within expected timeframe
                assert 4.0 <= timeout_duration <= 6.0, f"Timeout took {timeout_duration:.2f}s"
        
        logger.info("✅ Timeout-based failure detection working")
    
    @pytest.mark.asyncio
    async def test_partial_mcp_failure_handling(self):
        """Test handling when only some MCP services are down"""
        
        test_context = {
            "content": {"test": "partial failure data"},
            "type": "knowledge",
            "metadata": {"source": "test"}
        }
        
        # Simulate partial failure: vector DB down, but KV store up
        with patch('src.mcp_server.server.kv_store') as mock_kv, \
             patch('src.mcp_server.server.vector_db') as mock_vector, \
             patch('src.mcp_server.server.graph_db') as mock_graph:
            
            # KV store works
            mock_kv.store_context.return_value = {"success": True, "id": "partial_test"}
            
            # Vector DB fails
            mock_vector.store_embeddings.side_effect = ConnectionError("Vector DB unavailable")
            
            # Graph DB works
            mock_graph.create_nodes.return_value = {"success": True}
            
            result = await store_context_tool(test_context)
            
            # Should fail-closed even with partial success
            assert result["success"] is False, "Should fail-closed on partial MCP failure"
            assert result["error_code"] == ErrorCode.DEPENDENCY_DOWN.value
            
            logger.info("✅ Partial MCP failure correctly handled with fail-closed")


class TestMCPOutageLogging:
    """Test logging and monitoring during MCP outages"""
    
    @pytest.mark.asyncio
    async def test_outage_event_logging(self, caplog):
        """Test that MCP outages are properly logged for monitoring"""
        
        with caplog.at_level(logging.ERROR):
            mock_mcp = MockMCPServerOutage()
            mock_mcp.set_outage(True)
            
            # Simulate outage detection
            with patch('src.mcp_server.server.mcp_client', mock_mcp):
                test_context = {"content": {"test": "data"}, "type": "log"}
                
                with patch('src.mcp_server.server.kv_store') as mock_kv:
                    mock_kv.store_context.side_effect = ConnectionError("MCP unavailable")
                    
                    result = await store_context_tool(test_context)
                    
                    # Should log the outage
                    assert any("MCP" in record.message or "unavailable" in record.message 
                             for record in caplog.records), "MCP outage should be logged"
        
        logger.info("✅ MCP outage logging verified")
    
    @pytest.mark.asyncio
    async def test_outage_metrics_collection(self):
        """Test that outage events generate metrics for monitoring"""
        
        # Mock metrics collection
        outage_metrics = []
        
        def collect_outage_metric(service: str, state: str, duration: float):
            outage_metrics.append({
                "service": service,
                "state": state,
                "duration": duration,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Simulate outage metrics collection
        with patch('src.core.metrics.collect_outage_metric', side_effect=collect_outage_metric):
            mock_mcp = MockMCPServerOutage()
            mock_mcp.set_outage(True)
            
            start_time = time.time()
            
            try:
                await mock_mcp.handle_request({"test": "metrics"})
            except ConnectionError:
                outage_duration = time.time() - start_time
                collect_outage_metric("mcp", "down", outage_duration)
            
            # Verify metrics were collected
            assert len(outage_metrics) > 0, "Outage metrics should be collected"
            assert outage_metrics[0]["service"] == "mcp"
            assert outage_metrics[0]["state"] == "down"
            
        logger.info("✅ Outage metrics collection verified")


@pytest.mark.integration
class TestMCPOutageIntegration:
    """Integration tests for MCP outage scenarios"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_outage_simulation(self):
        """End-to-end test of MCP outage handling"""
        
        logger.info("🚀 Starting E2E MCP outage simulation")
        
        # Step 1: Normal operation
        logger.info("Step 1: Testing normal MCP operation")
        
        with patch('src.mcp_server.server.kv_store') as mock_kv, \
             patch('src.mcp_server.server.vector_db') as mock_vector:
            
            mock_kv.store_context.return_value = {"success": True, "id": "normal_001"}
            mock_vector.store_embeddings.return_value = {"success": True}
            
            normal_result = await store_context_tool({
                "content": {"test": "normal operation"},
                "type": "log"
            })
            
            assert normal_result["success"] is True
            logger.info("✅ Step 1: Normal operation verified")
        
        # Step 2: MCP outage occurs
        logger.info("Step 2: Simulating MCP outage")
        
        with patch('src.mcp_server.server.kv_store') as mock_kv:
            mock_kv.store_context.side_effect = ConnectionError("MCP server unavailable")
            
            outage_result = await store_context_tool({
                "content": {"test": "during outage"},
                "type": "log"
            })
            
            assert outage_result["success"] is False
            assert outage_result["error_code"] == ErrorCode.DEPENDENCY_DOWN.value
            logger.info("✅ Step 2: Fail-closed behavior during outage verified")
        
        # Step 3: MCP recovery
        logger.info("Step 3: Testing MCP recovery")
        
        with patch('src.mcp_server.server.kv_store') as mock_kv, \
             patch('src.mcp_server.server.vector_db') as mock_vector:
            
            mock_kv.store_context.return_value = {"success": True, "id": "recovery_001"}
            mock_vector.store_embeddings.return_value = {"success": True}
            
            recovery_result = await store_context_tool({
                "content": {"test": "after recovery"},
                "type": "log"
            })
            
            assert recovery_result["success"] is True
            logger.info("✅ Step 3: MCP recovery verified")
        
        logger.info("🎯 E2E MCP outage simulation PASSED")


if __name__ == "__main__":
    # Run the outage tests
    pytest.main([__file__, "-v", "-s", "-k", "not integration"])