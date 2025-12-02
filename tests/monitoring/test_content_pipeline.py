#!/usr/bin/env python3
"""
Unit tests for S10 Content Pipeline Monitoring Check.

Tests the ContentPipelineMonitoring check with mocked HTTP calls and pipeline validation.
"""

import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import aiohttp

from src.monitoring.sentinel.checks.s10_content_pipeline import ContentPipelineMonitoring
from src.monitoring.sentinel.models import SentinelConfig


class TestContentPipelineMonitoring:
    """Test suite for ContentPipelineMonitoring check."""
    
    @pytest.fixture
    def config(self) -> SentinelConfig:
        """Create a test configuration."""
        return SentinelConfig({
            "veris_memory_url": "http://test.example.com:8000",
            "s10_pipeline_timeout_sec": 30,
            "s10_pipeline_stages": ["ingestion", "validation", "storage", "retrieval"],
            "s10_performance_thresholds": {
                "ingestion_latency_ms": 3000,
                "retrieval_latency_ms": 1000,
                "pipeline_throughput_per_min": 5,
                "storage_consistency_ratio": 0.9
            },
            "s10_test_content_samples": [
                {
                    "type": "test_doc",
                    "content": {
                        "text": "Test document content",
                        "title": "Test Document"
                    },
                    "expected_features": ["test", "document"]
                }
            ]
        })
    
    @pytest.fixture
    def check(self, config: SentinelConfig) -> ContentPipelineMonitoring:
        """Create a ContentPipelineMonitoring check instance."""
        return ContentPipelineMonitoring(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, config: SentinelConfig) -> None:
        """Test check initialization."""
        check = ContentPipelineMonitoring(config)
        
        assert check.check_id == "S10-content-pipeline"
        assert check.description == "Content pipeline monitoring"
        assert check.veris_memory_url == "http://test.example.com:8000"
        assert check.timeout_seconds == 30
        assert len(check.pipeline_stages) == 4
        assert len(check.test_content_samples) == 1
        assert check.performance_thresholds["ingestion_latency_ms"] == 3000
    
    @pytest.mark.asyncio
    async def test_run_check_all_pass(self, check: ContentPipelineMonitoring) -> None:
        """Test run_check when all pipeline tests pass."""
        mock_results = [
            {"passed": True, "message": "Content ingestion successful"},
            {"passed": True, "message": "Pipeline stages operational"},
            {"passed": True, "message": "Content retrieval working"},
            {"passed": True, "message": "Storage consistency validated"},
            {"passed": True, "message": "Pipeline performance acceptable"},
            {"passed": True, "message": "Error handling working"},
            {"passed": True, "message": "Content lifecycle complete"}
        ]
        
        with patch.object(check, '_test_content_ingestion', return_value=mock_results[0]):
            with patch.object(check, '_validate_pipeline_stages', return_value=mock_results[1]):
                with patch.object(check, '_test_content_retrieval', return_value=mock_results[2]):
                    with patch.object(check, '_validate_storage_consistency', return_value=mock_results[3]):
                        with patch.object(check, '_test_pipeline_performance', return_value=mock_results[4]):
                            with patch.object(check, '_validate_error_handling', return_value=mock_results[5]):
                                with patch.object(check, '_test_content_lifecycle', return_value=mock_results[6]):
                                    
                                    result = await check.run_check()
        
        assert result.check_id == "S10-content-pipeline"
        assert result.status == "pass"
        assert "All content pipeline checks passed: 7 tests successful" in result.message
        assert result.details["total_tests"] == 7
        assert result.details["passed_tests"] == 7
        assert result.details["failed_tests"] == 0
    
    @pytest.mark.asyncio
    async def test_run_check_with_failures(self, check: ContentPipelineMonitoring) -> None:
        """Test run_check when some pipeline tests fail."""
        mock_results = [
            {"passed": False, "message": "Content ingestion failed"},
            {"passed": False, "message": "Pipeline stages unhealthy"},
            {"passed": True, "message": "Content retrieval working"},
            {"passed": True, "message": "Storage consistency validated"},
            {"passed": True, "message": "Pipeline performance acceptable"},
            {"passed": True, "message": "Error handling working"},
            {"passed": True, "message": "Content lifecycle complete"}
        ]
        
        with patch.object(check, '_test_content_ingestion', return_value=mock_results[0]):
            with patch.object(check, '_validate_pipeline_stages', return_value=mock_results[1]):
                with patch.object(check, '_test_content_retrieval', return_value=mock_results[2]):
                    with patch.object(check, '_validate_storage_consistency', return_value=mock_results[3]):
                        with patch.object(check, '_test_pipeline_performance', return_value=mock_results[4]):
                            with patch.object(check, '_validate_error_handling', return_value=mock_results[5]):
                                with patch.object(check, '_test_content_lifecycle', return_value=mock_results[6]):
                                    
                                    result = await check.run_check()
        
        assert result.status == "fail"
        assert "Content pipeline issues detected: 2 problems found" in result.message
        assert result.details["passed_tests"] == 5
        assert result.details["failed_tests"] == 2
    
    @pytest.mark.asyncio
    async def test_content_ingestion_success(self, check: ContentPipelineMonitoring) -> None:
        """Test successful content ingestion."""
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value={"context_id": "test_ctx_123"})
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._test_content_ingestion()
        
        assert result["passed"] is True
        assert "Content ingestion" in result["message"]
        assert result["success_rate"] >= 0.0
        assert result["successful_ingestions"] >= 0
        assert len(result["ingestion_tests"]) > 0
    
    @pytest.mark.asyncio
    async def test_content_ingestion_failure(self, check: ContentPipelineMonitoring) -> None:
        """Test content ingestion failure handling."""
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._test_content_ingestion()
        
        assert result["passed"] is False
        assert result["success_rate"] < 0.8  # Below threshold
        assert result["successful_ingestions"] == 0
        
        # Check that error details are captured
        for test in result["ingestion_tests"]:
            assert test["ingestion_successful"] is False
            assert test["status_code"] == 400
    
    @pytest.mark.asyncio
    async def test_pipeline_stages_validation_success(self, check: ContentPipelineMonitoring) -> None:
        """Test successful pipeline stages validation."""
        mock_health_response = AsyncMock()
        mock_health_response.status = 200
        mock_health_response.json = AsyncMock(return_value={"status": "healthy"})
        
        mock_general_response = AsyncMock()
        mock_general_response.status = 200
        
        mock_session = AsyncMock()
        
        def mock_get(url, **kwargs):
            ctx = AsyncMock()
            if "/health/ready" in url:
                ctx.__aenter__ = AsyncMock(return_value=mock_general_response)
            else:
                ctx.__aenter__ = AsyncMock(return_value=mock_health_response)
            return ctx
        
        mock_session.get.side_effect = mock_get
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._validate_pipeline_stages()
        
        assert result["passed"] is True
        assert "Pipeline stages" in result["message"]
        assert result["stage_health_ratio"] >= 0.0
        assert len(result["stage_validations"]) == len(check.pipeline_stages)
    
    @pytest.mark.asyncio
    async def test_content_retrieval_success(self, check: ContentPipelineMonitoring) -> None:
        """Test successful content retrieval."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "contexts": [
                {"context_id": "ctx1", "content": {"text": "Test content"}, "score": 0.9}
            ]
        })
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._test_content_retrieval()
        
        assert result["passed"] is True
        assert "Content retrieval" in result["message"]
        assert result["success_rate"] >= 0.0
        assert len(result["retrieval_tests"]) > 0
    
    @pytest.mark.asyncio
    async def test_storage_consistency_success(self, check: ContentPipelineMonitoring) -> None:
        """Test successful storage consistency validation."""
        # Mock successful storage
        mock_store_response = AsyncMock()
        mock_store_response.status = 201
        mock_store_response.json = AsyncMock(return_value={"context_id": "test_ctx_123"})
        
        # Mock successful retrieval
        mock_get_response = AsyncMock()
        mock_get_response.status = 200
        mock_get_response.json = AsyncMock(return_value={
            "context_id": "test_ctx_123",
            "content": {"text": "Test content"}
        })
        
        mock_session = AsyncMock()
        
        def mock_request(method_url, **kwargs):
            ctx = AsyncMock()
            if isinstance(method_url, str) and "/contexts/" in method_url:
                ctx.__aenter__ = AsyncMock(return_value=mock_get_response)
            else:
                ctx.__aenter__ = AsyncMock(return_value=mock_store_response)
            return ctx
        
        mock_session.post.side_effect = mock_request
        mock_session.get.side_effect = mock_request
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with patch('asyncio.sleep', return_value=None):  # Speed up test
                result = await check._validate_storage_consistency()
        
        assert result["passed"] is True
        assert "Storage consistency" in result["message"]
        assert result["consistency_ratio"] >= 0.0
        assert len(result["consistency_tests"]) > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_performance_success(self, check: ContentPipelineMonitoring) -> None:
        """Test successful pipeline performance validation."""
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value={"context_id": "test_ctx_123"})
        
        mock_search_response = AsyncMock()
        mock_search_response.status = 200
        mock_search_response.json = AsyncMock(return_value={
            "contexts": [{"context_id": "test_ctx_123"}]
        })
        
        mock_session = AsyncMock()
        
        def mock_request(*args, **kwargs):
            ctx = AsyncMock()
            if "search" in str(args):
                ctx.__aenter__ = AsyncMock(return_value=mock_search_response)
            else:
                ctx.__aenter__ = AsyncMock(return_value=mock_response)
            return ctx
        
        mock_session.post.side_effect = mock_request
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._test_pipeline_performance()
        
        assert result["passed"] is True
        assert "Pipeline performance" in result["message"]
        assert "performance_metrics" in result
        assert result["all_thresholds_met"] is True
    
    @pytest.mark.asyncio
    async def test_error_handling_validation_success(self, check: ContentPipelineMonitoring) -> None:
        """Test successful error handling validation."""
        mock_responses = [
            AsyncMock(status=400, text=AsyncMock(return_value="Bad Request")),  # Invalid content
            AsyncMock(status=413, text=AsyncMock(return_value="Payload Too Large")),  # Oversized
            AsyncMock(status=422, text=AsyncMock(return_value="Unprocessable Entity")),  # Invalid search
            AsyncMock(status=404, text=AsyncMock(return_value="Not Found"))  # Non-existent context
        ]
        
        response_iter = iter(mock_responses)
        
        def mock_request(*args, **kwargs):
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=next(response_iter))
            return ctx
        
        mock_session = AsyncMock()
        mock_session.post.side_effect = mock_request
        mock_session.get.side_effect = mock_request
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._validate_error_handling()
        
        assert result["passed"] is True
        assert "Error handling" in result["message"]
        assert result["error_handling_ratio"] >= 0.75
        assert len(result["error_handling_tests"]) == 4
        
        # Verify that errors were properly handled
        for test in result["error_handling_tests"]:
            assert test["properly_handled"] is True
    
    @pytest.mark.asyncio
    async def test_content_lifecycle_success(self, check: ContentPipelineMonitoring) -> None:
        """Test successful content lifecycle validation."""
        # Mock creation
        mock_create_response = AsyncMock()
        mock_create_response.status = 201
        mock_create_response.json = AsyncMock(return_value={"context_id": "lifecycle_test_123"})
        
        # Mock retrieval by ID
        mock_get_response = AsyncMock()
        mock_get_response.status = 200
        mock_get_response.json = AsyncMock(return_value={
            "context_id": "lifecycle_test_123",
            "content": {"text": "Content lifecycle test"}
        })
        
        # Mock search discovery
        mock_search_response = AsyncMock()
        mock_search_response.status = 200
        mock_search_response.json = AsyncMock(return_value={
            "contexts": [
                {"context_id": "lifecycle_test_123", "content": {"text": "Content lifecycle test"}}
            ]
        })
        
        mock_session = AsyncMock()
        
        def mock_request(method, url, **kwargs):
            ctx = AsyncMock()
            if method == "post" and "search" in url:
                ctx.__aenter__ = AsyncMock(return_value=mock_search_response)
            elif method == "get":
                ctx.__aenter__ = AsyncMock(return_value=mock_get_response)
            else:  # POST to create
                ctx.__aenter__ = AsyncMock(return_value=mock_create_response)
            return ctx
        
        mock_session.post.side_effect = lambda url, **kwargs: mock_request("post", url, **kwargs)
        mock_session.get.side_effect = lambda url, **kwargs: mock_request("get", url, **kwargs)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with patch('asyncio.sleep', return_value=None):  # Speed up test
                result = await check._test_content_lifecycle()
        
        assert result["passed"] is True
        assert "Content lifecycle" in result["message"]
        assert result["lifecycle_success_ratio"] >= 0.75
        assert len(result["lifecycle_stages"]) >= 3
        
        # Verify all stages completed successfully
        successful_stages = [stage for stage in result["lifecycle_stages"] if stage.get("successful")]
        assert len(successful_stages) >= 3
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, check: ContentPipelineMonitoring) -> None:
        """Test handling of API errors."""
        mock_session = AsyncMock()
        mock_session.post.side_effect = aiohttp.ClientError("Connection failed")
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._test_content_ingestion()
        
        assert result["passed"] is False
        assert "error" in result
        assert "Connection failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_performance_thresholds(self, check: ContentPipelineMonitoring) -> None:
        """Test performance threshold validation."""
        # Test with slow responses
        slow_mock_response = AsyncMock()
        slow_mock_response.status = 201
        slow_mock_response.json = AsyncMock(return_value={"context_id": "test_ctx"})
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = slow_mock_response
        
        # Mock time.time to simulate slow response
        with patch('time.time', side_effect=[0, 5]):  # 5 second delay
            with patch('aiohttp.ClientSession', return_value=mock_session):
                result = await check._test_content_ingestion()
        
        # Should pass even with slow response if success rate is good
        assert result["passed"] is True
        assert result["avg_ingestion_latency_ms"] > 1000  # Should be > 1 second
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, check: ContentPipelineMonitoring) -> None:
        """Test concurrent operation handling in performance test."""
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value={"context_id": "test_ctx"})
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._test_pipeline_performance()
        
        assert result["passed"] is True
        concurrent_metrics = result["performance_metrics"]["concurrent_load_test"]
        assert concurrent_metrics["concurrent_operations"] == 5
        assert concurrent_metrics["success_rate"] >= 0.8
    
    @pytest.mark.asyncio
    async def test_pipeline_stages_fallback(self, check: ContentPipelineMonitoring) -> None:
        """Test pipeline stages validation with fallback to general health."""
        # Mock stage-specific endpoints failing, but general health succeeding
        mock_general_response = AsyncMock()
        mock_general_response.status = 200
        
        mock_session = AsyncMock()
        
        def mock_get(url, **kwargs):
            ctx = AsyncMock()
            if "/health/ready" in url:
                ctx.__aenter__ = AsyncMock(return_value=mock_general_response)
            else:
                ctx.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("Not found"))
            return ctx
        
        mock_session.get.side_effect = mock_get
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._validate_pipeline_stages()
        
        assert result["passed"] is True
        # All stages should be marked as "assumed_healthy"
        for validation in result["stage_validations"]:
            assert validation["stage_operational"] is True
            assert validation["status"] == "assumed_healthy"
    
    @pytest.mark.asyncio
    async def test_run_check_with_exception(self, check: ContentPipelineMonitoring) -> None:
        """Test run_check when an exception occurs."""
        with patch.object(check, '_test_content_ingestion', side_effect=Exception("Pipeline error")):
            result = await check.run_check()
        
        assert result.status == "fail"
        assert "Content pipeline monitoring failed with error: Pipeline error" in result.message
        assert result.details["error"] == "Pipeline error"
    
    @pytest.mark.asyncio
    async def test_default_test_samples(self, check: ContentPipelineMonitoring) -> None:
        """Test default test content samples."""
        # Test with default configuration
        default_config = SentinelConfig({})
        default_check = ContentPipelineMonitoring(default_config)
        
        assert len(default_check.test_content_samples) == 5
        
        # Verify structure of default samples
        for sample in default_check.test_content_samples:
            assert "type" in sample
            assert "content" in sample
            assert "expected_features" in sample
            assert "text" in sample["content"]
            assert "title" in sample["content"]
    
    @pytest.mark.asyncio
    async def test_storage_consistency_failures(self, check: ContentPipelineMonitoring) -> None:
        """Test storage consistency with retrieval failures."""
        # Mock successful storage but failed retrieval
        mock_store_response = AsyncMock()
        mock_store_response.status = 201
        mock_store_response.json = AsyncMock(return_value={"context_id": "test_ctx_123"})
        
        mock_get_response = AsyncMock()
        mock_get_response.status = 404  # Not found
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_store_response
        mock_session.get.return_value.__aenter__.return_value = mock_get_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with patch('asyncio.sleep', return_value=None):
                result = await check._validate_storage_consistency()
        
        assert result["passed"] is False
        assert result["consistency_ratio"] < 0.95  # Below threshold
        assert result["successful_retrievals"] == 0
    
    @pytest.mark.asyncio
    async def test_lifecycle_creation_failure(self, check: ContentPipelineMonitoring) -> None:
        """Test content lifecycle with creation failure."""
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")

        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._test_content_lifecycle()

        assert result["passed"] is False
        assert "Content lifecycle test failed at creation stage" in result["message"]
        assert len(result["lifecycle_stages"]) == 1
        assert result["lifecycle_stages"][0]["successful"] is False

    @pytest.mark.asyncio
    async def test_lifecycle_content_format_dict(self, check: ContentPipelineMonitoring) -> None:
        """Test content lifecycle with nested dict content format (legacy format)."""
        # Mock creation
        mock_create_response = AsyncMock()
        mock_create_response.status = 201
        mock_create_response.json = AsyncMock(return_value={"context_id": "lifecycle_test_dict"})

        # Mock retrieval with NESTED dict content format
        mock_get_response = AsyncMock()
        mock_get_response.status = 200
        mock_get_response.json = AsyncMock(return_value={
            "context": {
                "context_id": "lifecycle_test_dict",
                "content": {
                    "text": "Content lifecycle test - 2025-01-01T00:00:00.000000"  # Nested dict format
                }
            }
        })

        # Mock search discovery
        mock_search_response = AsyncMock()
        mock_search_response.status = 200
        mock_search_response.json = AsyncMock(return_value={
            "contexts": [
                {"context_id": "lifecycle_test_dict", "content": {"text": "Content lifecycle test"}}
            ]
        })

        mock_session = AsyncMock()

        def mock_request(method, url, **kwargs):
            ctx = AsyncMock()
            if method == "post" and "search" in url:
                ctx.__aenter__ = AsyncMock(return_value=mock_search_response)
            elif method == "get":
                ctx.__aenter__ = AsyncMock(return_value=mock_get_response)
            else:  # POST to create
                ctx.__aenter__ = AsyncMock(return_value=mock_create_response)
            return ctx

        mock_session.post.side_effect = lambda url, **kwargs: mock_request("post", url, **kwargs)
        mock_session.get.side_effect = lambda url, **kwargs: mock_request("get", url, **kwargs)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            with patch('asyncio.sleep', return_value=None):
                result = await check._test_content_lifecycle()

        assert result["passed"] is True
        assert "Content lifecycle" in result["message"]

        # Verify retrieval stage handled nested dict format correctly
        retrieval_stage = next((s for s in result["lifecycle_stages"] if s.get("stage") == "retrieval_by_id"), None)
        assert retrieval_stage is not None
        assert retrieval_stage["successful"] is True
        # data_integrity check should work with nested dict format
        assert "data_integrity" in retrieval_stage

    @pytest.mark.asyncio
    async def test_lifecycle_content_format_string(self, check: ContentPipelineMonitoring) -> None:
        """Test content lifecycle with flat string content format (current format)."""
        test_content_string = f"Content lifecycle test - {datetime.utcnow().isoformat()}"

        # Mock creation
        mock_create_response = AsyncMock()
        mock_create_response.status = 201
        mock_create_response.json = AsyncMock(return_value={"context_id": "lifecycle_test_string"})

        # Mock retrieval with FLAT string content format
        mock_get_response = AsyncMock()
        mock_get_response.status = 200
        mock_get_response.json = AsyncMock(return_value={
            "context": {
                "context_id": "lifecycle_test_string",
                "content": test_content_string  # Flat string format
            }
        })

        # Mock search discovery
        mock_search_response = AsyncMock()
        mock_search_response.status = 200
        mock_search_response.json = AsyncMock(return_value={
            "contexts": [
                {"context_id": "lifecycle_test_string", "content": test_content_string}
            ]
        })

        mock_session = AsyncMock()

        def mock_request(method, url, **kwargs):
            ctx = AsyncMock()
            if method == "post" and "search" in url:
                ctx.__aenter__ = AsyncMock(return_value=mock_search_response)
            elif method == "get":
                ctx.__aenter__ = AsyncMock(return_value=mock_get_response)
            else:  # POST to create
                ctx.__aenter__ = AsyncMock(return_value=mock_create_response)
            return ctx

        mock_session.post.side_effect = lambda url, **kwargs: mock_request("post", url, **kwargs)
        mock_session.get.side_effect = lambda url, **kwargs: mock_request("get", url, **kwargs)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            with patch('asyncio.sleep', return_value=None):
                result = await check._test_content_lifecycle()

        assert result["passed"] is True
        assert "Content lifecycle" in result["message"]

        # Verify retrieval stage handled flat string format correctly
        retrieval_stage = next((s for s in result["lifecycle_stages"] if s.get("stage") == "retrieval_by_id"), None)
        assert retrieval_stage is not None
        assert retrieval_stage["successful"] is True
        # data_integrity check should work with flat string format
        assert "data_integrity" in retrieval_stage


class TestContentPipelineTargetBaseURL:
    """Test TARGET_BASE_URL environment variable usage in ContentPipelineMonitoring."""

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'TARGET_BASE_URL': 'http://context-store:8000'}, clear=False)
    async def test_uses_target_base_url_when_set(self) -> None:
        """Test check uses TARGET_BASE_URL environment variable when set."""
        config = SentinelConfig(enabled_checks=["S10-content-pipeline"])
        check = ContentPipelineMonitoring(config)

        # Verify TARGET_BASE_URL is used
        assert check.veris_memory_url == "http://context-store:8000"

    @pytest.mark.asyncio
    @patch.dict('os.environ', {}, clear=True)
    async def test_falls_back_to_localhost_when_unset(self) -> None:
        """Test check falls back to localhost:8000 when TARGET_BASE_URL is not set."""
        config = SentinelConfig(enabled_checks=["S10-content-pipeline"])
        check = ContentPipelineMonitoring(config)

        # Verify fallback to localhost
        assert check.veris_memory_url == "http://localhost:8000"

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'TARGET_BASE_URL': 'http://custom-host:9999'}, clear=False)
    async def test_config_overrides_env_var(self) -> None:
        """Test config veris_memory_url overrides TARGET_BASE_URL if explicitly set."""
        config = SentinelConfig(
            enabled_checks=["S10-content-pipeline"],
            veris_memory_url="http://config-override:7777"
        )
        check = ContentPipelineMonitoring(config)

        # Config should override env var
        assert check.veris_memory_url == "http://config-override:7777"

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'TARGET_BASE_URL': 'http://context-store:8000'}, clear=False)
    async def test_can_connect_using_target_base_url(self) -> None:
        """Test check can successfully connect to context-store using TARGET_BASE_URL."""
        config = SentinelConfig(enabled_checks=["S10-content-pipeline"])
        check = ContentPipelineMonitoring(config)

        # Mock successful HTTP connection to context-store
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value={"id": "test123", "success": True})

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_response)
        ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = ctx

        # Also mock GET for retrieval
        mock_get_response = AsyncMock()
        mock_get_response.status = 200
        mock_get_response.json = AsyncMock(return_value={"contexts": [{"id": "test123"}]})

        get_ctx = AsyncMock()
        get_ctx.__aenter__ = AsyncMock(return_value=mock_get_response)
        get_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = get_ctx

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check.run_check()

        # Verify connection was attempted to context-store URL
        assert mock_session.post.called
        call_url = str(mock_session.post.call_args[0][0])
        assert call_url.startswith("http://context-store:8000")


class TestS10_404FallbackHandling:
    """Test suite for S10 graceful 404 handling when stage endpoints are missing."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig(
            enabled_checks=["S10-content-pipeline"],
            veris_memory_url="http://localhost:8000"
        )

    @pytest.mark.asyncio
    async def test_validate_pipeline_stages_404_fallback(self, config):
        """Test that 404 from stage endpoint falls back to /health/ready."""
        check = ContentPipelineMonitoring(config)

        # Create mock session
        mock_session = AsyncMock()

        # Mock responses for stage endpoints (404) and general health (200)
        async def mock_get(url, *args, **kwargs):
            mock_response = AsyncMock()

            # Stage endpoints return 404
            if any(stage in url for stage in ["ingestion", "validation", "enrichment", "storage", "indexing", "retrieval"]):
                mock_response.status = 404
                mock_response.json = AsyncMock(return_value={"detail": "Not found"})

            # General health endpoint returns 200
            elif "/health/ready" in url:
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    "status": "ready",
                    "components": [
                        {"name": "qdrant", "status": "ok"},
                        {"name": "neo4j", "status": "ok"}
                    ]
                })

            else:
                mock_response.status = 404
                mock_response.json = AsyncMock(return_value={"detail": "Not found"})

            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=mock_response)
            ctx.__aexit__ = AsyncMock(return_value=None)
            return ctx

        mock_session.get.side_effect = mock_get

        # Call the validation method
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._validate_pipeline_stages(mock_session)

        # All stages should be marked as operational despite 404s
        assert result["passed"] is True
        assert len(result["stage_validations"]) == 6

        # Check that stages were marked as not_implemented_but_healthy
        for validation in result["stage_validations"]:
            assert validation["stage_operational"] is True
            assert validation["status"] == "not_implemented_but_healthy"
            assert validation["status_code"] == 404
            assert "not implemented" in validation["note"].lower()

    @pytest.mark.asyncio
    async def test_404_fallback_when_general_health_fails(self, config):
        """Test 404 handling when general health check also fails."""
        check = ContentPipelineMonitoring(config)

        mock_session = AsyncMock()

        # Mock all endpoints returning errors
        async def mock_get(url, *args, **kwargs):
            mock_response = AsyncMock()

            if any(stage in url for stage in ["ingestion", "validation", "enrichment", "storage", "indexing", "retrieval"]):
                # Stage endpoints return 404
                mock_response.status = 404
                mock_response.json = AsyncMock(return_value={"detail": "Not found"})
            elif "/health/ready" in url:
                # General health also fails
                mock_response.status = 503
                mock_response.json = AsyncMock(return_value={"status": "unhealthy"})
            else:
                mock_response.status = 500
                mock_response.json = AsyncMock(return_value={"detail": "Server error"})

            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=mock_response)
            ctx.__aexit__ = AsyncMock(return_value=None)
            return ctx

        mock_session.get.side_effect = mock_get

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._validate_pipeline_stages(mock_session)

        # Should fail when general health is not good
        assert result["passed"] is False

        # Stages should be marked as not operational
        for validation in result["stage_validations"]:
            if validation["status_code"] == 404:
                assert validation["stage_operational"] is False

    @pytest.mark.asyncio
    async def test_mixed_404_and_200_responses(self, config):
        """Test handling of mixed responses (some 404, some 200)."""
        check = ContentPipelineMonitoring(config)

        mock_session = AsyncMock()

        # Mock mixed responses
        async def mock_get(url, *args, **kwargs):
            mock_response = AsyncMock()

            # ingestion and validation return 200
            if "ingestion" in url or "validation" in url:
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={"status": "operational"})

            # Other stages return 404
            elif any(stage in url for stage in ["enrichment", "storage", "indexing", "retrieval"]):
                mock_response.status = 404
                mock_response.json = AsyncMock(return_value={"detail": "Not found"})

            # General health returns 200
            elif "/health/ready" in url:
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    "status": "ready",
                    "components": [{"name": "qdrant", "status": "ok"}]
                })

            else:
                mock_response.status = 404
                mock_response.json = AsyncMock(return_value={"detail": "Not found"})

            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=mock_response)
            ctx.__aexit__ = AsyncMock(return_value=None)
            return ctx

        mock_session.get.side_effect = mock_get

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._validate_pipeline_stages(mock_session)

        # All stages should be operational
        assert result["passed"] is True

        # Check individual stage statuses
        implemented_count = sum(1 for v in result["stage_validations"] if v["status_code"] == 200)
        not_implemented_count = sum(1 for v in result["stage_validations"] if v["status"] == "not_implemented_but_healthy")

        assert implemented_count == 2  # ingestion and validation
        assert not_implemented_count == 4  # enrichment, storage, indexing, retrieval


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
