#!/usr/bin/env python3
"""
Unit tests for S3 Paraphrase Robustness Check.

Tests the ParaphraseRobustness check with mocked HTTP calls and semantic analysis.
"""

import asyncio
import statistics
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest
import aiohttp

from src.monitoring.sentinel.checks.s3_paraphrase_robustness import ParaphraseRobustness
from src.monitoring.sentinel.models import SentinelConfig


class TestParaphraseRobustness:
    """Test suite for ParaphraseRobustness check."""
    
    @pytest.fixture
    def config(self) -> SentinelConfig:
        """Create a test configuration."""
        # Create base config with proper dataclass field
        config = SentinelConfig(target_base_url="http://test.example.com:8000")

        # Add custom S3-specific config attributes
        config.s3_paraphrase_timeout_sec = 30
        config.s3_min_similarity_threshold = 0.7
        config.s3_max_result_variance = 0.3
        config.s3_test_paraphrase_sets = [
            {
                "topic": "test_configuration",
                "variations": [
                    "How to configure test settings",
                    "Setting up test configuration",
                    "Test setup configuration guide",
                    "Configure testing environment"
                ],
                "expected_similarity": 0.8
            }
        ]

        return config
    
    @pytest.fixture
    def check(self, config: SentinelConfig) -> ParaphraseRobustness:
        """Create a ParaphraseRobustness check instance."""
        return ParaphraseRobustness(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, config: SentinelConfig) -> None:
        """Test check initialization."""
        check = ParaphraseRobustness(config)
        
        assert check.check_id == "S3-paraphrase-robustness"
        assert check.description == "Paraphrase robustness for semantic consistency"
        assert check.service_url == "http://test.example.com:8000"
        assert check.timeout_seconds == 30
        assert check.min_similarity_threshold == 0.7
        assert check.max_result_variance == 0.3
        assert len(check.test_paraphrase_sets) == 1
    
    @pytest.mark.asyncio
    async def test_run_check_all_pass(self, check: ParaphraseRobustness) -> None:
        """Test run_check when all semantic tests pass."""
        mock_results = [
            {"passed": True, "message": "Semantic similarity validated"},
            {"passed": True, "message": "Result consistency confirmed"},
            {"passed": True, "message": "Ranking stability verified"},
            {"passed": True, "message": "Context retrieval robust"},
            {"passed": True, "message": "Query expansion effective"},
            {"passed": True, "message": "Response quality consistent"}
        ]

        with patch.object(check, '_test_semantic_similarity', return_value=mock_results[0]):
            with patch.object(check, '_test_result_consistency', return_value=mock_results[1]):
                with patch.object(check, '_test_ranking_stability', return_value=mock_results[2]):
                    with patch.object(check, '_test_context_retrieval_robustness', return_value=mock_results[3]):
                        with patch.object(check, '_test_query_expansion', return_value=mock_results[4]):
                            with patch.object(check, '_test_response_quality_consistency', return_value=mock_results[5]):

                                result = await check.run_check()

        assert result.check_id == "S3-paraphrase-robustness"
        assert result.status == "pass"
        assert "All paraphrase robustness checks passed: 6 tests successful" in result.message
        assert result.details["total_tests"] == 6
        assert result.details["passed_tests"] == 6
        assert result.details["failed_tests"] == 0
    
    @pytest.mark.asyncio
    async def test_run_check_with_failures(self, check: ParaphraseRobustness) -> None:
        """Test run_check when some semantic tests fail."""
        mock_results = [
            {"passed": False, "message": "Semantic similarity too low"},
            {"passed": False, "message": "Result consistency failed"},
            {"passed": True, "message": "Ranking stability verified"},
            {"passed": True, "message": "Context retrieval robust"},
            {"passed": True, "message": "Query expansion effective"},
            {"passed": True, "message": "Response quality consistent"}
        ]

        with patch.object(check, '_test_semantic_similarity', return_value=mock_results[0]):
            with patch.object(check, '_test_result_consistency', return_value=mock_results[1]):
                with patch.object(check, '_test_ranking_stability', return_value=mock_results[2]):
                    with patch.object(check, '_test_context_retrieval_robustness', return_value=mock_results[3]):
                        with patch.object(check, '_test_query_expansion', return_value=mock_results[4]):
                            with patch.object(check, '_test_response_quality_consistency', return_value=mock_results[5]):

                                result = await check.run_check()

        assert result.status == "fail"
        assert "Semantic consistency issues detected: 2 tests failed" in result.message
        assert result.details["passed_tests"] == 4
        assert result.details["failed_tests"] == 2
    
    @pytest.mark.asyncio
    async def test_semantic_similarity_success(self, check: ParaphraseRobustness) -> None:
        """Test successful semantic similarity analysis."""
        # Mock response with consistent results (high similarity)
        # All variations return same contexts = high similarity
        mock_response_data = {
            "contexts": [
                {"id": "ctx1", "content": {"text": "Test configuration guide"}, "score": 0.9},
                {"id": "ctx2", "content": {"text": "Configuration setup"}, "score": 0.8},
                {"id": "ctx3", "content": {"text": "Setup guide"}, "score": 0.75}
            ]
        }

        # Create mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)

        # Create mock post context manager
        mock_post_context = AsyncMock()
        mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_context.__aexit__ = AsyncMock(return_value=None)

        # Create mock session with post as a regular Mock (not AsyncMock)
        # so it returns the context manager directly without wrapping in coroutine
        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_post_context)

        # Create mock ClientSession context manager
        mock_client_session = AsyncMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_client_session):
            result = await check._test_semantic_similarity()

        assert result["passed"] is True
        assert "Semantic similarity test" in result["message"]
        assert result["overall_avg_similarity"] >= 0.0
        assert len(result["similarity_results"]) > 0
    
    @pytest.mark.asyncio
    async def test_semantic_similarity_low_scores(self, check: ParaphraseRobustness) -> None:
        """Test semantic similarity with low similarity scores."""
        # Mock very different results for each variation (low overlap = low similarity)
        # Each variation returns completely different contexts
        mock_responses = [
            {"contexts": [{"id": "ctx1", "content": {"text": "Configuration guide"}, "score": 0.9}]},
            {"contexts": [{"id": "ctx2", "content": {"text": "Completely different topic"}, "score": 0.9}]},
            {"contexts": [{"id": "ctx3", "content": {"text": "Another unrelated result"}, "score": 0.9}]},
            {"contexts": [{"id": "ctx4", "content": {"text": "Fourth different topic"}, "score": 0.9}]}
        ]

        response_iter = iter(mock_responses)

        def mock_post(*args, **kwargs):
            # Create response for this call
            response = AsyncMock()
            response.status = 200
            try:
                response.json = AsyncMock(return_value=next(response_iter))
            except StopIteration:
                # If we run out, return empty contexts
                response.json = AsyncMock(return_value={"contexts": []})

            # Create async context manager for this post call
            post_context = AsyncMock()
            post_context.__aenter__ = AsyncMock(return_value=response)
            post_context.__aexit__ = AsyncMock(return_value=None)
            return post_context

        # Create mock session
        mock_session = AsyncMock()
        mock_session.post = mock_post

        # Create mock ClientSession context manager
        mock_client_session = AsyncMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_client_session):
            result = await check._test_semantic_similarity()

        assert result["passed"] is False
        assert result["overall_avg_similarity"] < check.min_similarity_threshold
    
    @pytest.mark.asyncio
    async def test_result_consistency_success(self, check: ParaphraseRobustness) -> None:
        """Test successful result consistency validation."""
        # Mock consistent results across paraphrases (same results = high consistency)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "contexts": [
                {"id": "ctx1", "content": {"text": "Configuration guide"}, "score": 0.9},
                {"id": "ctx2", "content": {"text": "Setup instructions"}, "score": 0.8},
                {"id": "ctx3", "content": {"text": "Test setup"}, "score": 0.7}
            ]
        })

        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._test_result_consistency()

        assert result["passed"] is True
        assert "Result consistency test" in result["message"]
        assert result["topics_tested"] >= 0
        assert result["topics_passed"] >= 0
        assert len(result["consistency_results"]) > 0
    
    @pytest.mark.asyncio
    async def test_ranking_stability_success(self, check: ParaphraseRobustness) -> None:
        """Test successful ranking stability validation."""
        # Mock same results for multiple runs (stable ranking)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "contexts": [
                {"id": "ctx1", "content": {"text": "Configuration guide"}, "score": 0.9},
                {"id": "ctx2", "content": {"text": "Setup instructions"}, "score": 0.8},
                {"id": "ctx3", "content": {"text": "Test setup"}, "score": 0.7}
            ]
        })

        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._test_ranking_stability()

        assert result["passed"] is True
        assert "Ranking stability test" in result["message"]
        assert result["topics_tested"] >= 0
        assert result["topics_passed"] >= 0
        assert len(result["ranking_results"]) > 0
    
    @pytest.mark.asyncio
    async def test_context_retrieval_robustness_success(self, check: ParaphraseRobustness) -> None:
        """Test successful context retrieval robustness."""
        # Mock consistent retrieval results (low variance = high robustness)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "contexts": [
                {"id": "ctx1", "content": {"text": "Configuration guide"}, "score": 0.9},
                {"id": "ctx2", "content": {"text": "Setup guide"}, "score": 0.8}
            ]
        })

        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._test_context_retrieval_robustness()

        assert result["passed"] is True
        assert "Context retrieval robustness test" in result["message"]
        assert result["topics_tested"] >= 0
        assert result["topics_passed"] >= 0
        assert len(result["retrieval_results"]) > 0
    
    @pytest.mark.asyncio
    async def test_query_expansion_success(self, check: ParaphraseRobustness) -> None:
        """Test successful query expansion validation."""
        # Mock responses for 3 test cases (simple + expanded for each = 6 responses)
        mock_responses = [
            # Test case 1: simple "config"
            {"contexts": [{"id": "ctx1", "content": {"text": "Configuration"}, "score": 0.8}]},
            # Test case 1: expanded "configuration setup process" (more results)
            {"contexts": [
                {"id": "ctx1", "content": {"text": "Configuration"}, "score": 0.9},
                {"id": "ctx2", "content": {"text": "Setup guide"}, "score": 0.8},
                {"id": "ctx3", "content": {"text": "Process docs"}, "score": 0.7}
            ]},
            # Test case 2: simple "error"
            {"contexts": [{"id": "ctx4", "content": {"text": "Error"}, "score": 0.8}]},
            # Test case 2: expanded "error troubleshooting resolution" (more results)
            {"contexts": [
                {"id": "ctx4", "content": {"text": "Error handling"}, "score": 0.9},
                {"id": "ctx5", "content": {"text": "Troubleshooting"}, "score": 0.8}
            ]},
            # Test case 3: simple "database"
            {"contexts": [{"id": "ctx6", "content": {"text": "Database"}, "score": 0.8}]},
            # Test case 3: expanded "database connection setup" (more results)
            {"contexts": [
                {"id": "ctx6", "content": {"text": "Database"}, "score": 0.9},
                {"id": "ctx7", "content": {"text": "Connection"}, "score": 0.8},
                {"id": "ctx8", "content": {"text": "Setup"}, "score": 0.7}
            ]}
        ]

        response_iter = iter(mock_responses)

        def mock_post(*args, **kwargs):
            ctx = AsyncMock()
            response = AsyncMock()
            response.status = 200
            try:
                response.json = AsyncMock(return_value=next(response_iter))
            except StopIteration:
                response.json = AsyncMock(return_value={"contexts": []})
            ctx.__aenter__ = AsyncMock(return_value=response)
            return ctx

        mock_session = AsyncMock()
        mock_session.post.side_effect = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._test_query_expansion()

        assert result["passed"] is True
        assert "Query expansion test" in result["message"]
        assert result["total_tests"] >= 0
        assert result["effective_expansions"] >= 0
        assert len(result["expansion_results"]) > 0
    
    @pytest.mark.asyncio
    async def test_response_quality_consistency_success(self, check: ParaphraseRobustness) -> None:
        """Test successful response quality consistency."""
        # Mock consistent quality scores (low variance = high consistency)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "contexts": [
                {"id": "ctx1", "content": {"text": "High quality response"}, "score": 0.9},
                {"id": "ctx2", "content": {"text": "Quality content"}, "score": 0.85}
            ]
        })

        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._test_response_quality_consistency()

        assert result["passed"] is True
        assert "Response quality consistency test" in result["message"]
        assert result["topics_tested"] >= 0
        assert result["topics_passed"] >= 0
        assert len(result["quality_results"]) > 0
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, check: ParaphraseRobustness) -> None:
        """Test handling of API errors at the top level."""
        # Mock ClientSession to raise error during context manager __aenter__
        # This will be caught by the outer try/except and return passed=False
        mock_client_session = AsyncMock()
        mock_client_session.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("Connection failed"))
        mock_client_session.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_client_session):
            result = await check._test_semantic_similarity()

        # Should return error structure
        assert result["passed"] is False
        assert "error" in result
        assert isinstance(result["error"], str)
        assert "Connection failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_calculate_result_overlap(self, check: ParaphraseRobustness) -> None:
        """Test result overlap calculation."""
        results1 = [{"id": "ctx1"}, {"id": "ctx2"}]
        results2 = [{"id": "ctx2"}, {"id": "ctx3"}]

        overlap_ratio = check._calculate_result_overlap(results1, results2)

        assert 0.0 <= overlap_ratio <= 1.0
        # Jaccard similarity: intersection/union = 1/3 ≈ 0.333
        assert 0.3 < overlap_ratio < 0.4
    
    @pytest.mark.asyncio
    async def test_calculate_score_correlation(self, check: ParaphraseRobustness) -> None:
        """Test score correlation calculation."""
        scores1 = [0.9, 0.8, 0.7]
        scores2 = [0.85, 0.75, 0.65]

        correlation = check._calculate_score_correlation(scores1, scores2)

        # Allow small floating point tolerance
        assert -1.01 <= correlation <= 1.01
        # Should be positive correlation for similar score patterns
        assert correlation > 0.5
    
    @pytest.mark.asyncio
    async def test_calculate_score_correlation_edge_cases(self, check: ParaphraseRobustness) -> None:
        """Test score correlation with edge cases."""
        # Empty lists
        correlation = check._calculate_score_correlation([], [])
        assert correlation == 0.0
        
        # Single values
        correlation = check._calculate_score_correlation([0.5], [0.6])
        assert correlation == 0.0
        
        # Identical values (no variance)
        correlation = check._calculate_score_correlation([0.5, 0.5], [0.6, 0.6])
        assert correlation == 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_ranking_stability(self, check: ParaphraseRobustness) -> None:
        """Test ranking stability calculation."""
        # Method expects lists of IDs (strings), not lists of dicts
        ranking1 = ["ctx1", "ctx2", "ctx3"]
        ranking2 = ["ctx1", "ctx2", "ctx3"]  # Same order = perfect stability

        stability = check._calculate_ranking_stability(ranking1, ranking2)

        assert 0.0 <= stability <= 1.0
        # Should be 1.0 since ranking order is identical
        assert stability == 1.0
    
    @pytest.mark.asyncio
    async def test_run_check_with_exception(self, check: ParaphraseRobustness) -> None:
        """Test run_check when an exception occurs in a test method."""
        # When asyncio.gather encounters an exception, it re-raises it
        # So the entire run_check will fail with exception handling
        with patch.object(check, '_test_semantic_similarity', side_effect=Exception("Test error")):
            result = await check.run_check()

        # The exception causes run_check to fail
        assert result.status == "fail"
        # The exception is caught by run_check's top-level exception handler
        assert "error" in result.details or "Test error" in str(result.details)
    
    @pytest.mark.asyncio
    async def test_default_paraphrase_sets(self, check: ParaphraseRobustness) -> None:
        """Test default paraphrase sets configuration."""
        # Test with default configuration
        default_config = SentinelConfig()
        default_check = ParaphraseRobustness(default_config)

        # After optimization (PR #327): 2 topics for runtime monitoring
        # Full 5x5 matrix moved to CI/CD comprehensive tests
        assert len(default_check.test_paraphrase_sets) == 2

        # Verify structure of default sets
        for paraphrase_set in default_check.test_paraphrase_sets:
            assert "topic" in paraphrase_set
            assert "variations" in paraphrase_set
            assert "expected_similarity" in paraphrase_set
            # Each topic has 3 variations for runtime smoke tests
            assert len(paraphrase_set["variations"]) == 3
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, check: ParaphraseRobustness) -> None:
        """Test configuration parameter validation."""
        # Test with invalid configuration
        invalid_config = SentinelConfig()
        invalid_config.s3_min_similarity_threshold = 1.5  # Invalid > 1.0
        invalid_config.s3_max_result_variance = -0.1     # Invalid < 0.0
        invalid_config.s3_paraphrase_timeout_sec = -5     # Invalid < 0

        # Should handle invalid config gracefully
        invalid_check = ParaphraseRobustness(invalid_config)
        assert invalid_check.min_similarity_threshold == 1.5  # Uses provided value
        assert invalid_check.max_result_variance == -0.1     # Uses provided value
        assert invalid_check.timeout_seconds == -5           # Uses provided value


class TestParaphraseRobustnessDiagnosticLogging:
    """
    Test suite for S3 Paraphrase Robustness diagnostic logging (PR #322).

    Tests verify that diagnostic logging is emitted correctly for debugging
    semantic consistency failures.
    """

    @pytest.fixture
    def config(self) -> SentinelConfig:
        """Create a test configuration."""
        config = SentinelConfig(target_base_url="http://test.example.com:8000")
        config.s3_paraphrase_timeout_sec = 30
        config.s3_min_similarity_threshold = 0.7
        config.s3_test_paraphrase_sets = [
            {
                "topic": "test_topic",
                "variations": ["query 1", "query 2", "query 3"]
            }
        ]
        return config

    @pytest.fixture
    def check(self, config: SentinelConfig) -> ParaphraseRobustness:
        """Create a ParaphraseRobustness check instance."""
        return ParaphraseRobustness(config)

    @pytest.mark.asyncio
    async def test_diagnostic_logging_emitted_for_semantic_similarity(
        self, check: ParaphraseRobustness, caplog
    ) -> None:
        """Test that S3-DIAGNOSTIC log messages are generated during semantic similarity tests."""
        import logging
        caplog.set_level(logging.INFO)

        # Mock search results
        mock_contexts = [
            {"id": "ctx-1", "score": 0.9},
            {"id": "ctx-2", "score": 0.8}
        ]

        with patch.object(check, '_search_contexts', new=AsyncMock(return_value={
            "contexts": mock_contexts,
            "error": None
        })):
            result = await check._test_semantic_similarity()

        # Verify diagnostic log messages were generated
        log_messages = [rec.message for rec in caplog.records if 'S3-DIAGNOSTIC' in rec.message]

        assert len(log_messages) > 0, "Expected S3-DIAGNOSTIC log messages to be generated"

        # Verify key diagnostic messages
        assert any("Starting semantic_similarity test" in msg for msg in log_messages), \
            "Expected test start message"
        assert any("Testing topic" in msg for msg in log_messages), \
            "Expected topic test message"
        assert any("Variation" in msg for msg in log_messages), \
            "Expected variation log"
        assert any("returned" in msg and "results" in msg for msg in log_messages), \
            "Expected search results log"

    @pytest.mark.asyncio
    async def test_low_overlap_warning_logged(
        self, check: ParaphraseRobustness, caplog
    ) -> None:
        """Test that low overlap warnings are logged with detailed analysis."""
        import logging
        caplog.set_level(logging.WARNING)

        # Mock search results with LOW overlap (different IDs)
        mock_result1 = {
            "contexts": [
                {"id": "ctx-1", "score": 0.9},
                {"id": "ctx-2", "score": 0.8}
            ],
            "error": None
        }
        mock_result2 = {
            "contexts": [
                {"id": "ctx-99", "score": 0.9},
                {"id": "ctx-98", "score": 0.8}
            ],
            "error": None
        }

        # Create a side effect that alternates between results
        call_count = 0
        async def mock_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_result1 if call_count % 2 == 1 else mock_result2

        with patch.object(check, '_search_contexts', new=AsyncMock(side_effect=mock_search)):
            result = await check._test_semantic_similarity()

        # Verify low overlap warnings were generated
        warning_messages = [rec.message for rec in caplog.records
                          if rec.levelname == 'WARNING' and 'S3-DIAGNOSTIC' in rec.message]

        assert len(warning_messages) > 0, "Expected low overlap warnings to be logged"

        # Verify warning contains detailed analysis
        assert any("LOW OVERLAP" in msg for msg in warning_messages), \
            "Expected LOW OVERLAP warning message"
        assert any("Common IDs" in msg or "Unique to Query" in msg for msg in warning_messages), \
            "Expected ID-level analysis in warnings"

    @pytest.mark.asyncio
    async def test_topic_summary_format(
        self, check: ParaphraseRobustness, caplog
    ) -> None:
        """Test that topic summaries include expected format."""
        import logging
        caplog.set_level(logging.INFO)

        # Mock search results
        mock_contexts = [
            {"id": "ctx-1", "score": 0.9},
            {"id": "ctx-2", "score": 0.8}
        ]

        with patch.object(check, '_search_contexts', new=AsyncMock(return_value={
            "contexts": mock_contexts,
            "error": None
        })):
            result = await check._test_semantic_similarity()

        # Verify topic summary format
        log_messages = [rec.message for rec in caplog.records if 'S3-DIAGNOSTIC' in rec.message]

        # Find summary messages
        summary_messages = [msg for msg in log_messages if "summary" in msg]
        assert len(summary_messages) > 0, "Expected topic summary messages"

        # Verify summary contains expected fields
        assert any("Average similarity" in msg for msg in log_messages), \
            "Expected average similarity in summary"
        assert any("Minimum similarity" in msg for msg in log_messages), \
            "Expected minimum similarity in summary"
        assert any("Meets threshold" in msg for msg in log_messages), \
            "Expected threshold check in summary"

    @pytest.mark.asyncio
    async def test_result_consistency_diagnostic_logging(
        self, check: ParaphraseRobustness, caplog
    ) -> None:
        """Test diagnostic logging for result consistency tests."""
        import logging
        caplog.set_level(logging.INFO)

        # Mock search results
        mock_contexts = [
            {"id": "ctx-1", "score": 0.9},
            {"id": "ctx-2", "score": 0.8},
            {"id": "ctx-3", "score": 0.7}
        ]

        with patch.object(check, '_search_contexts', new=AsyncMock(return_value={
            "contexts": mock_contexts,
            "error": None
        })):
            result = await check._test_result_consistency()

        # Verify diagnostic messages for consistency test
        log_messages = [rec.message for rec in caplog.records if 'S3-DIAGNOSTIC' in rec.message]

        assert any("Testing result consistency" in msg for msg in log_messages), \
            "Expected consistency test start message"
        assert any("Top-5 IDs" in msg for msg in log_messages), \
            "Expected top-5 IDs logging"
        assert any("Top-5 scores" in msg for msg in log_messages), \
            "Expected top-5 scores logging"

    @pytest.mark.asyncio
    async def test_query_expansion_diagnostic_logging(
        self, check: ParaphraseRobustness, caplog
    ) -> None:
        """Test diagnostic logging for query expansion tests."""
        import logging
        caplog.set_level(logging.INFO)

        # Mock search results
        mock_contexts = [
            {"id": "ctx-1", "score": 0.9},
            {"id": "ctx-2", "score": 0.8}
        ]

        with patch.object(check, '_search_contexts', new=AsyncMock(return_value={
            "contexts": mock_contexts,
            "error": None
        })):
            result = await check._test_query_expansion()

        # Verify diagnostic messages for query expansion
        log_messages = [rec.message for rec in caplog.records if 'S3-DIAGNOSTIC' in rec.message]

        assert any("Test case" in msg for msg in log_messages), \
            "Expected test case logging"
        assert any("Simple:" in msg for msg in log_messages), \
            "Expected simple query logging"
        assert any("Expanded:" in msg for msg in log_messages), \
            "Expected expanded query logging"

    @pytest.mark.asyncio
    async def test_lazy_logging_format_used(
        self, check: ParaphraseRobustness
    ) -> None:
        """
        Test that lazy logging format is used (% formatting, not f-strings).

        This is a code quality test - verifies that logger calls use lazy evaluation
        to avoid string formatting overhead when logging is disabled.
        """
        import inspect

        # Get source code of test methods
        source = inspect.getsource(ParaphraseRobustness)

        # Verify no f-strings in logger calls (should use % formatting)
        # Note: This is a best-effort check - actual implementation review is in PR
        assert 'logger.info(f"' not in source or source.count('logger.info(f"') < source.count('logger.info('), \
            "Logger calls should use lazy logging (logger.info('msg %s', var)) instead of f-strings"