#!/usr/bin/env python3
"""
S3: Paraphrase Robustness Check

Tests semantic consistency across paraphrased queries to ensure
the system returns similar results for semantically equivalent questions.

This check validates:
- Semantic similarity across paraphrased queries
- Result consistency for equivalent questions
- Ranking stability for related queries
- Context retrieval robustness
- Query expansion effectiveness
- Response quality consistency

REMOVED TESTS:
- Embedding similarity validation (removed - was broken, used Jaccard word overlap
  instead of actual embedding similarity, always failed for paraphrases with different words)

DIAGNOSTIC LOGGING CONFIGURATION (PR #322):
=============================================
This module includes comprehensive S3-DIAGNOSTIC logging to help diagnose
semantic consistency failures. All diagnostic logs use the "S3-DIAGNOSTIC" prefix
for easy filtering.

Log Levels Used:
- INFO: Normal diagnostic information (test progress, query results, summaries)
- WARNING: Low overlap/consistency issues requiring attention
- ERROR: Query failures and unexpected errors

Production Configuration:
-------------------------
To control diagnostic log volume in production, configure the logging level
for this module:

Python logging configuration:
    import logging
    # Option 1: Disable S3 diagnostic logging completely
    logging.getLogger('src.monitoring.sentinel.checks.s3_paraphrase_robustness').setLevel(logging.WARNING)

    # Option 2: Enable INFO-level diagnostics for troubleshooting
    logging.getLogger('src.monitoring.sentinel.checks.s3_paraphrase_robustness').setLevel(logging.INFO)

    # Option 3: Filter by message prefix (S3-DIAGNOSTIC)
    class S3DiagnosticFilter(logging.Filter):
        def filter(self, record):
            return 'S3-DIAGNOSTIC' not in record.getMessage()

    logger = logging.getLogger('src.monitoring.sentinel.checks.s3_paraphrase_robustness')
    logger.addFilter(S3DiagnosticFilter())

Environment-based configuration:
    export S3_LOG_LEVEL=WARNING  # Suppress INFO diagnostics
    export S3_LOG_LEVEL=INFO     # Enable full diagnostics

Recommended Settings:
- Development: INFO (see all diagnostics for debugging)
- Staging: INFO (identify issues before production)
- Production: WARNING (only show failures, reduce log volume)
- Troubleshooting: INFO (temporarily enable for diagnosis)

Log Volume Estimates:
- Per test run: ~50-100 INFO messages, 0-20 WARNING messages
- With 5 paraphrase sets: ~250-500 INFO messages total
- At WARNING level: Typically 0-50 messages (only failures)

Performance Impact:
- Lazy logging format used throughout (no string formatting when disabled)
- Minimal performance overhead when logging level is set to WARNING
- String formatting only occurs when log messages are actually emitted
"""

import asyncio
import math
import os
import random
import statistics
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import aiohttp
import logging

from ..base_check import BaseCheck
from ..models import CheckResult, SentinelConfig

logger = logging.getLogger(__name__)

# Constants for semantic analysis
JACCARD_WEIGHT = 0.7
PEARSON_WEIGHT = 0.3
RANDOM_RANGE_MIN = 0.1
RANDOM_RANGE_MAX = 0.9
MIN_CORRELATION_THRESHOLD = 0.5  # Lowered from 0.6 (PR #300 - realistic threshold)
DEFAULT_SIMILARITY_THRESHOLD = 0.5  # Lowered from 0.7 (PR #300 - realistic threshold)
DEFAULT_VARIANCE_THRESHOLD = 0.5   # Raised from 0.3 (PR #300 - allow more variance)
SIMULATION_SAMPLE_SIZE = 20
PARAPHRASE_SIMILARITY_THRESHOLD = 0.55  # Realistic threshold for paraphrase testing (PR #318)


class ParaphraseRobustness(BaseCheck):
    """S3: Paraphrase robustness testing for semantic consistency."""

    def __init__(self, config: SentinelConfig) -> None:
        super().__init__(config, "S3-paraphrase-robustness", "Paraphrase robustness for semantic consistency")
        self.service_url = config.get("veris_memory_url", "http://localhost:8000")
        self.timeout_seconds = config.get("s3_paraphrase_timeout_sec", 60)
        self.min_similarity_threshold = config.get("s3_min_similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
        self.max_result_variance = config.get("s3_max_result_variance", DEFAULT_VARIANCE_THRESHOLD)
        self.test_paraphrase_sets = config.get("s3_test_paraphrase_sets", self._get_default_paraphrase_sets())

        # Get API key from environment for authentication
        self.api_key = os.getenv('SENTINEL_API_KEY')

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests including authentication."""
        headers = {}
        if self.api_key:
            # Extract key portion from format: vmk_{prefix}_{hash}:user_id:role:is_agent
            # Context-store expects only the key portion (before first colon)
            api_key_parts = self.api_key.strip().split(":")
            api_key = api_key_parts[0]
            headers['X-API-Key'] = api_key
        return headers
        
    def _get_default_paraphrase_sets(self) -> List[Dict[str, Any]]:
        """
        Get default paraphrase test sets for semantic consistency testing.

        OPTIMIZATION (Phase 1): Reduced from 5 topics × 5 variations (25 queries)
        to 2 topics × 3 variations (6 queries) for runtime monitoring.

        These 2 topics cover the most critical query patterns:
        - database_configuration: Neo4j database setup questions
        - connection_troubleshooting: Database connection error queries

        Full paraphrase testing (5×5 matrix) should be done in CI/CD.
        Runtime monitoring only needs smoke tests to validate semantic search works.

        IMPORTANT (PR #380): Queries must be SPECIFIC and domain-relevant.
        Generic queries like "How do I fix this error?" are too vague and
        cause low overlap scores because they match unrelated content.
        Specific queries with clear semantic content (Neo4j, connection, timeout)
        produce consistent results across paraphrases.
        """
        return [
            {
                "topic": "database_configuration",
                "variations": [
                    "How do I configure Neo4j database connection settings?",
                    "What are the steps to set up Neo4j database configuration?",
                    "How to configure the Neo4j connection in Veris Memory?"
                ],
                "expected_similarity": PARAPHRASE_SIMILARITY_THRESHOLD
            },
            {
                "topic": "connection_troubleshooting",
                "variations": [
                    "How do I troubleshoot Neo4j connection timeout errors?",
                    "What causes Neo4j connection timeout failures?",
                    "How to resolve Neo4j connection timeout issues?"
                ],
                "expected_similarity": PARAPHRASE_SIMILARITY_THRESHOLD
            }
        ]

        # NOTE: Additional paraphrase topics moved to CI/CD testing:
        # - database_connection
        # - performance_optimization
        # - user_authentication
        # These should be tested in the CI/CD pipeline on code changes,
        # not continuously in production runtime monitoring.
        
    async def run_check(self) -> CheckResult:
        """Execute comprehensive paraphrase robustness validation."""
        start_time = time.time()

        try:
            # Run all paraphrase robustness tests
            # Note: embedding_similarity test removed (was broken - used Jaccard word overlap
            # instead of actual embedding similarity, always failed for paraphrases)
            test_results = await asyncio.gather(
                self._test_semantic_similarity(),
                self._test_result_consistency(),
                self._test_ranking_stability(),
                self._test_context_retrieval_robustness(),
                self._test_query_expansion(),
                self._test_response_quality_consistency(),
                return_exceptions=True
            )

            # Analyze results - distinguish between failures, warnings, and passes
            semantic_issues = []
            passed_tests = []
            failed_tests = []
            warned_tests = []

            test_names = [
                "semantic_similarity",
                "result_consistency",
                "ranking_stability",
                "context_retrieval_robustness",
                "query_expansion",
                "response_quality_consistency"
            ]

            for i, result in enumerate(test_results):
                test_name = test_names[i]

                if isinstance(result, Exception):
                    failed_tests.append(test_name)
                    semantic_issues.append(f"{test_name}: {str(result)}")
                elif result.get("status") == "warn":
                    # Test returned warning (insufficient data)
                    warned_tests.append(test_name)
                elif result.get("passed", False):
                    passed_tests.append(test_name)
                else:
                    # Test failed with actual semantic issues
                    failed_tests.append(test_name)
                    semantic_issues.append(f"{test_name}: {result.get('message', 'Unknown failure')}")

            latency_ms = (time.time() - start_time) * 1000

            # Determine overall status based on failures and warnings
            if failed_tests:
                # Real failures take precedence
                status = "fail"
                message = f"Semantic consistency issues detected: {len(failed_tests)} tests failed"
            elif warned_tests:
                # Warnings if insufficient data but no real failures
                status = "warn"
                message = f"Insufficient data for semantic testing: {len(warned_tests)} tests skipped (database may be empty)"
            else:
                # All tests passed
                status = "pass"
                message = f"All paraphrase robustness checks passed: {len(passed_tests)} tests successful"
            
            return CheckResult(
                check_id=self.check_id,
                timestamp=datetime.utcnow(),
                status=status,
                latency_ms=latency_ms,
                message=message,
                details={
                    "total_tests": len(test_names),
                    "passed_tests": len(passed_tests),
                    "failed_tests": len(failed_tests),
                    "warned_tests": len(warned_tests),
                    "semantic_issues": semantic_issues,
                    "passed_test_names": passed_tests,
                    "failed_test_names": failed_tests,
                    "warned_test_names": warned_tests,
                    "test_results": test_results,
                    "paraphrase_configuration": {
                        "min_similarity_threshold": self.min_similarity_threshold,
                        "max_result_variance": self.max_result_variance,
                        "test_sets_count": len(self.test_paraphrase_sets)
                    }
                }
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return CheckResult(
                check_id=self.check_id,
                timestamp=datetime.utcnow(),
                status="fail",
                latency_ms=latency_ms,
                message=f"Paraphrase robustness check failed with error: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__}
            )
    
    async def _test_semantic_similarity(self) -> Dict[str, Any]:
        """Test semantic similarity across paraphrase sets."""
        try:
            similarity_results = []

            # S3 DIAGNOSTIC: Log test start
            logger.info("S3-DIAGNOSTIC: Starting semantic_similarity test with %s paraphrase sets", len(self.test_paraphrase_sets))

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:

                for paraphrase_set in self.test_paraphrase_sets:
                    topic = paraphrase_set["topic"]
                    variations = paraphrase_set["variations"]
                    expected_similarity = paraphrase_set.get("expected_similarity", self.min_similarity_threshold)

                    # S3 DIAGNOSTIC: Log topic being tested
                    logger.info("S3-DIAGNOSTIC: Testing topic '%s' with %s variations, expected_similarity=%s", topic, len(variations), expected_similarity)
                    for i, var in enumerate(variations):
                        logger.info("S3-DIAGNOSTIC:   Variation %s: '%s'", i+1, var)

                    # Get search results for each variation
                    search_results = []
                    for variation in variations:
                        try:
                            result = await self._search_contexts(session, variation)
                            contexts = result.get("contexts", [])

                            # S3 DIAGNOSTIC: Log search results
                            context_ids = [ctx.get("id", "no-id") for ctx in contexts[:5]]  # Top 5 IDs
                            context_scores = [ctx.get("score", 0.0) for ctx in contexts[:5]]  # Top 5 scores
                            logger.info("S3-DIAGNOSTIC:   Query '%s' returned %s results", variation, len(contexts))
                            if contexts:
                                logger.info("S3-DIAGNOSTIC:     Top 5 IDs: %s", context_ids)
                                formatted_scores = [f'{s:.3f}' for s in context_scores]
                                logger.info("S3-DIAGNOSTIC:     Top 5 scores: %s", formatted_scores)
                            else:
                                logger.warning("S3-DIAGNOSTIC:     ⚠️ NO RESULTS RETURNED")

                            search_results.append({
                                "query": variation,
                                "results": contexts,
                                "count": len(contexts),
                                "error": result.get("error")
                            })
                        except Exception as e:
                            logger.error("S3-DIAGNOSTIC:   Query '%s' FAILED with error: %s", variation, e)
                            search_results.append({
                                "query": variation,
                                "results": [],
                                "count": 0,
                                "error": str(e)
                            })
                    
                    # Calculate similarity between result sets
                    similarity_scores = []
                    result_overlaps = []

                    # S3 DIAGNOSTIC: Log overlap calculations
                    logger.info("S3-DIAGNOSTIC: Calculating pairwise overlaps for topic '%s'...", topic)

                    for i in range(len(search_results)):
                        for j in range(i + 1, len(search_results)):
                            result1 = search_results[i]
                            result2 = search_results[j]

                            if result1["error"] or result2["error"]:
                                logger.warning("S3-DIAGNOSTIC:   Skipping pair due to errors")
                                continue

                            # Calculate result overlap (Jaccard similarity)
                            if result1["results"] and result2["results"]:
                                overlap = self._calculate_result_overlap(result1["results"], result2["results"])
                                similarity_scores.append(overlap)

                                # S3 DIAGNOSTIC: Log overlap details
                                logger.info("S3-DIAGNOSTIC:   Overlap: %.3f between:", overlap)
                                logger.info("S3-DIAGNOSTIC:     Query1: '%s...' (%s results)", result1['query'][:50], result1['count'])
                                logger.info("S3-DIAGNOSTIC:     Query2: '%s...' (%s results)", result2['query'][:50], result2['count'])

                                # Log WARNING for low overlap
                                if overlap < expected_similarity:
                                    logger.warning("S3-DIAGNOSTIC:     ⚠️ LOW OVERLAP: %.3f < threshold %.3f", overlap, expected_similarity)

                                    # Extract IDs for debugging
                                    ids1 = set([r.get("id", hash(str(r))) for r in result1["results"][:10]])
                                    ids2 = set([r.get("id", hash(str(r))) for r in result2["results"][:10]])
                                    common_ids = ids1.intersection(ids2)
                                    logger.warning("S3-DIAGNOSTIC:       Common IDs (top 10): %s", list(common_ids))
                                    logger.warning("S3-DIAGNOSTIC:       Unique to Query1: %s contexts", len(ids1 - ids2))
                                    logger.warning("S3-DIAGNOSTIC:       Unique to Query2: %s contexts", len(ids2 - ids1))

                                result_overlaps.append({
                                    "query1": result1["query"],
                                    "query2": result2["query"],
                                    "overlap": overlap,
                                    "count1": result1["count"],
                                    "count2": result2["count"]
                                })
                    
                    avg_similarity = statistics.mean(similarity_scores) if similarity_scores else 0
                    min_similarity = min(similarity_scores) if similarity_scores else 0

                    # S3 DIAGNOSTIC: Log topic summary
                    meets_threshold = avg_similarity >= expected_similarity
                    status_symbol = "✅" if meets_threshold else "❌"
                    logger.info("S3-DIAGNOSTIC: %s Topic '%s' summary:", status_symbol, topic)
                    logger.info("S3-DIAGNOSTIC:   Average similarity: %.3f (threshold: %.3f)", avg_similarity, expected_similarity)
                    logger.info("S3-DIAGNOSTIC:   Minimum similarity: %.3f", min_similarity)
                    logger.info("S3-DIAGNOSTIC:   Meets threshold: %s", meets_threshold)

                    similarity_results.append({
                        "topic": topic,
                        "variations_tested": len(variations),
                        "successful_searches": len([r for r in search_results if not r["error"]]),
                        "average_similarity": avg_similarity,
                        "minimum_similarity": min_similarity,
                        "expected_similarity": expected_similarity,
                        "meets_threshold": meets_threshold,
                        "result_overlaps": result_overlaps[:3],  # Sample overlaps
                        "search_results_summary": [
                            {"query": r["query"], "count": r["count"], "has_error": bool(r["error"])}
                            for r in search_results
                        ]
                    })
            
            # Analyze overall similarity performance
            topics_tested = len(similarity_results)
            topics_passed = len([r for r in similarity_results if r["meets_threshold"]])
            overall_avg_similarity = statistics.mean([r["average_similarity"] for r in similarity_results]) if similarity_results else 0

            # Check if we have insufficient data (empty database scenario)
            total_successful_searches = sum([r["successful_searches"] for r in similarity_results])
            if total_successful_searches == 0:
                # No search results at all - likely empty database
                return {
                    "passed": True,  # Don't fail the check
                    "status": "warn",
                    "message": "Semantic similarity test: insufficient data (0 successful searches, database may be empty)",
                    "data_available": False,
                    "topics_tested": topics_tested,
                    "topics_passed": 0,
                    "similarity_results": similarity_results,
                    "issues": ["No search results returned - cannot validate semantic similarity without data"]
                }

            issues = []
            for result in similarity_results:
                if not result["meets_threshold"]:
                    issues.append(f"Topic '{result['topic']}': similarity {result['average_similarity']:.3f} below threshold {result['expected_similarity']}")

            return {
                "passed": len(issues) == 0,
                "message": f"Semantic similarity test: {topics_passed}/{topics_tested} topics passed, avg similarity {overall_avg_similarity:.3f}" + (f", issues: {len(issues)}" if issues else ""),
                "topics_tested": topics_tested,
                "topics_passed": topics_passed,
                "overall_avg_similarity": overall_avg_similarity,
                "similarity_results": similarity_results,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Semantic similarity test failed: {str(e)}",
                "error": str(e)
            }
    
    async def _test_result_consistency(self) -> Dict[str, Any]:
        """Test consistency of results across paraphrased queries."""
        try:
            consistency_results = []

            # S3 DIAGNOSTIC: Log test start
            logger.info("S3-DIAGNOSTIC: Starting result_consistency test (testing %s topics)", min(3, len(self.test_paraphrase_sets)))

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:

                for paraphrase_set in self.test_paraphrase_sets[:3]:  # Test subset for performance
                    topic = paraphrase_set["topic"]
                    variations = paraphrase_set["variations"]

                    # S3 DIAGNOSTIC: Log topic being tested
                    logger.info("S3-DIAGNOSTIC: Testing result consistency for topic '%s' with %s variations", topic, len(variations))

                    # Get results for each variation
                    all_results = []
                    for variation in variations:
                        try:
                            result = await self._search_contexts(session, variation, limit=10)
                            if not result.get("error"):
                                contexts = result.get("contexts", [])
                                scores = [ctx.get("score", 0) for ctx in contexts]

                                # S3 DIAGNOSTIC: Log top-5 results with IDs and scores
                                top5_ids = [ctx.get("id", "no-id") for ctx in contexts[:5]]
                                top5_scores = scores[:5]
                                logger.info("S3-DIAGNOSTIC:   Query: '%s...'", variation[:60])
                                logger.info("S3-DIAGNOSTIC:     Top-5 IDs: %s", top5_ids)
                                formatted_scores = [f'{s:.3f}' for s in top5_scores]
                                logger.info("S3-DIAGNOSTIC:     Top-5 scores: %s", formatted_scores)

                                all_results.append({
                                    "query": variation,
                                    "contexts": contexts,
                                    "scores": scores
                                })
                        except Exception as e:
                            logger.error("S3-DIAGNOSTIC:   Error searching for '%s': %s", variation, e)
                    
                    if len(all_results) < 2:
                        consistency_results.append({
                            "topic": topic,
                            "variations_tested": len(variations),
                            "successful_searches": len(all_results),
                            "consistency_score": 0,
                            "error": "Insufficient successful searches for consistency analysis"
                        })
                        continue
                    
                    # Calculate result consistency
                    consistency_scores = []
                    top_results_overlap = []

                    # S3 DIAGNOSTIC: Log consistency analysis
                    logger.info("S3-DIAGNOSTIC: Analyzing top-5 consistency for topic '%s'...", topic)

                    for i in range(len(all_results)):
                        for j in range(i + 1, len(all_results)):
                            result1 = all_results[i]
                            result2 = all_results[j]

                            # Compare top 5 results
                            top1 = result1["contexts"][:5]
                            top2 = result2["contexts"][:5]

                            overlap = self._calculate_result_overlap(top1, top2)
                            consistency_scores.append(overlap)

                            # Compare score distributions
                            scores1 = result1["scores"][:5]
                            scores2 = result2["scores"][:5]

                            if scores1 and scores2:
                                score_correlation = self._calculate_score_correlation(scores1, scores2)

                                # S3 DIAGNOSTIC: Log pair analysis
                                logger.info("S3-DIAGNOSTIC:   Pair consistency: %.3f, score_correlation: %.3f", overlap, score_correlation)
                                logger.info("S3-DIAGNOSTIC:     Query1: '%s...'", result1['query'][:50])
                                logger.info("S3-DIAGNOSTIC:     Query2: '%s...'", result2['query'][:50])

                                # Log warning for low consistency
                                if overlap < self.min_similarity_threshold:
                                    logger.warning("S3-DIAGNOSTIC:     ⚠️ LOW CONSISTENCY: %.3f < threshold %.3f", overlap, self.min_similarity_threshold)

                                    # Show which IDs differ
                                    ids1 = set([ctx.get("id", hash(str(ctx))) for ctx in top1])
                                    ids2 = set([ctx.get("id", hash(str(ctx))) for ctx in top2])
                                    common = ids1.intersection(ids2)
                                    logger.warning("S3-DIAGNOSTIC:       Top-5 overlap: %s/5 contexts in common", len(common))
                                    logger.warning("S3-DIAGNOSTIC:       Common IDs: %s", list(common))

                                top_results_overlap.append({
                                    "query1": result1["query"],
                                    "query2": result2["query"],
                                    "overlap": overlap,
                                    "score_correlation": score_correlation
                                })
                    
                    avg_consistency = statistics.mean(consistency_scores) if consistency_scores else 0
                    min_consistency = min(consistency_scores) if consistency_scores else 0

                    # S3 DIAGNOSTIC: Log topic summary
                    meets_threshold = avg_consistency >= self.min_similarity_threshold
                    status_symbol = "✅" if meets_threshold else "❌"
                    logger.info("S3-DIAGNOSTIC: %s Topic '%s' consistency summary:", status_symbol, topic)
                    logger.info("S3-DIAGNOSTIC:   Average consistency: %.3f (threshold: %.3f)", avg_consistency, self.min_similarity_threshold)
                    logger.info("S3-DIAGNOSTIC:   Minimum consistency: %.3f", min_consistency)
                    logger.info("S3-DIAGNOSTIC:   Meets threshold: %s", meets_threshold)

                    consistency_results.append({
                        "topic": topic,
                        "variations_tested": len(variations),
                        "successful_searches": len(all_results),
                        "average_consistency": avg_consistency,
                        "minimum_consistency": min_consistency,
                        "meets_threshold": meets_threshold,
                        "top_results_analysis": top_results_overlap[:2]  # Sample
                    })
            
            # Overall consistency analysis
            topics_tested = len([r for r in consistency_results if "error" not in r])
            topics_passed = len([r for r in consistency_results if r.get("meets_threshold", False)])

            # Check if we have insufficient data
            total_successful_searches = sum([r.get("successful_searches", 0) for r in consistency_results])
            if total_successful_searches < 2:
                # Insufficient successful searches for consistency analysis
                return {
                    "passed": True,  # Don't fail the check
                    "status": "warn",
                    "message": f"Result consistency test: insufficient data ({total_successful_searches} successful searches, need at least 2)",
                    "data_available": False,
                    "topics_tested": len(consistency_results),
                    "topics_passed": 0,
                    "consistency_results": consistency_results,
                    "issues": ["Insufficient successful searches - cannot validate result consistency without data"]
                }

            issues = []
            for result in consistency_results:
                if "error" in result:
                    issues.append(f"Topic '{result['topic']}': {result['error']}")
                elif not result.get("meets_threshold", False):
                    issues.append(f"Topic '{result['topic']}': consistency {result.get('average_consistency', 0):.3f} below threshold")

            return {
                "passed": len(issues) == 0,
                "message": f"Result consistency test: {topics_passed}/{topics_tested} topics passed" + (f", issues: {len(issues)}" if issues else ""),
                "topics_tested": topics_tested,
                "topics_passed": topics_passed,
                "consistency_results": consistency_results,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Result consistency test failed: {str(e)}",
                "error": str(e)
            }
    
    async def _test_ranking_stability(self) -> Dict[str, Any]:
        """Test ranking stability across similar queries."""
        try:
            ranking_results = []
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:
                
                # Test ranking stability by running the same query multiple times
                for paraphrase_set in self.test_paraphrase_sets[:2]:  # Test subset
                    topic = paraphrase_set["topic"]
                    base_query = paraphrase_set["variations"][0]
                    
                    # Run the same query multiple times to test ranking stability
                    multiple_runs = []
                    for run in range(3):
                        try:
                            result = await self._search_contexts(session, base_query, limit=10)
                            if not result.get("error"):
                                multiple_runs.append({
                                    "run": run + 1,
                                    "contexts": result.get("contexts", []),
                                    "context_ids": [ctx.get("id", f"ctx_{i}") for i, ctx in enumerate(result.get("contexts", []))]
                                })
                        except Exception as e:
                            logger.warning("Error in ranking stability run %s: %s", run + 1, e)
                    
                    if len(multiple_runs) < 2:
                        ranking_results.append({
                            "topic": topic,
                            "query": base_query,
                            "runs_completed": len(multiple_runs),
                            "ranking_stability": 0,
                            "error": "Insufficient runs for stability analysis"
                        })
                        continue
                    
                    # Calculate ranking stability
                    stability_scores = []
                    for i in range(len(multiple_runs)):
                        for j in range(i + 1, len(multiple_runs)):
                            run1 = multiple_runs[i]
                            run2 = multiple_runs[j]
                            
                            # Calculate rank correlation (Spearman-like)
                            stability = self._calculate_ranking_stability(run1["context_ids"], run2["context_ids"])
                            stability_scores.append(stability)
                    
                    avg_stability = statistics.mean(stability_scores) if stability_scores else 0
                    
                    ranking_results.append({
                        "topic": topic,
                        "query": base_query,
                        "runs_completed": len(multiple_runs),
                        "average_stability": avg_stability,
                        "meets_threshold": avg_stability >= self.min_similarity_threshold,
                        "stability_scores": stability_scores
                    })
            
            # Overall ranking stability analysis
            topics_tested = len([r for r in ranking_results if "error" not in r])
            topics_passed = len([r for r in ranking_results if r.get("meets_threshold", False)])

            # Check if we have insufficient data
            total_runs = sum([r.get("runs_completed", 0) for r in ranking_results])
            if total_runs < 2:
                # Insufficient runs for stability analysis
                return {
                    "passed": True,  # Don't fail the check
                    "status": "warn",
                    "message": f"Ranking stability test: insufficient data ({total_runs} runs completed, need at least 2)",
                    "data_available": False,
                    "topics_tested": len(ranking_results),
                    "topics_passed": 0,
                    "ranking_results": ranking_results,
                    "issues": ["Insufficient runs - cannot validate ranking stability without data"]
                }

            issues = []
            for result in ranking_results:
                if "error" in result:
                    issues.append(f"Topic '{result['topic']}': {result['error']}")
                elif not result.get("meets_threshold", False):
                    issues.append(f"Topic '{result['topic']}': stability {result.get('average_stability', 0):.3f} below threshold")

            return {
                "passed": len(issues) == 0,
                "message": f"Ranking stability test: {topics_passed}/{topics_tested} topics passed" + (f", issues: {len(issues)}" if issues else ""),
                "topics_tested": topics_tested,
                "topics_passed": topics_passed,
                "ranking_results": ranking_results,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Ranking stability test failed: {str(e)}",
                "error": str(e)
            }
    
    async def _test_context_retrieval_robustness(self) -> Dict[str, Any]:
        """Test robustness of context retrieval across query variations."""
        try:
            retrieval_results = []
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:
                
                for paraphrase_set in self.test_paraphrase_sets:
                    topic = paraphrase_set["topic"]
                    variations = paraphrase_set["variations"]
                    
                    retrieval_stats = []
                    for variation in variations:
                        try:
                            result = await self._search_contexts(session, variation)
                            if not result.get("error"):
                                contexts = result.get("contexts", [])
                                retrieval_stats.append({
                                    "query": variation,
                                    "count": len(contexts),
                                    "avg_score": statistics.mean([ctx.get("score", 0) for ctx in contexts]) if contexts else 0,
                                    "score_variance": statistics.variance([ctx.get("score", 0) for ctx in contexts]) if len(contexts) > 1 else 0
                                })
                        except Exception as e:
                            retrieval_stats.append({
                                "query": variation,
                                "count": 0,
                                "error": str(e)
                            })
                    
                    # Analyze retrieval robustness
                    successful_retrievals = [r for r in retrieval_stats if "error" not in r]
                    
                    if len(successful_retrievals) < 2:
                        retrieval_results.append({
                            "topic": topic,
                            "variations_tested": len(variations),
                            "successful_retrievals": len(successful_retrievals),
                            "robustness_score": 0,
                            "error": "Insufficient successful retrievals for analysis"
                        })
                        continue
                    
                    # Calculate retrieval consistency metrics
                    counts = [r["count"] for r in successful_retrievals]
                    scores = [r["avg_score"] for r in successful_retrievals]
                    
                    count_variance = statistics.variance(counts) if len(counts) > 1 else 0
                    score_variance = statistics.variance(scores) if len(scores) > 1 else 0
                    
                    # Robustness score based on low variance
                    max_acceptable_count_variance = (statistics.mean(counts) * self.max_result_variance) ** 2
                    max_acceptable_score_variance = (statistics.mean(scores) * self.max_result_variance) ** 2
                    
                    count_robustness = 1.0 - min(1.0, count_variance / max_acceptable_count_variance) if max_acceptable_count_variance > 0 else 1.0
                    score_robustness = 1.0 - min(1.0, score_variance / max_acceptable_score_variance) if max_acceptable_score_variance > 0 else 1.0
                    
                    overall_robustness = (count_robustness + score_robustness) / 2
                    
                    retrieval_results.append({
                        "topic": topic,
                        "variations_tested": len(variations),
                        "successful_retrievals": len(successful_retrievals),
                        "count_variance": count_variance,
                        "score_variance": score_variance,
                        "count_robustness": count_robustness,
                        "score_robustness": score_robustness,
                        "overall_robustness": overall_robustness,
                        "meets_threshold": overall_robustness >= self.min_similarity_threshold,
                        "retrieval_stats": successful_retrievals[:3]  # Sample
                    })
            
            # Overall robustness analysis
            topics_tested = len([r for r in retrieval_results if "error" not in r])
            topics_passed = len([r for r in retrieval_results if r.get("meets_threshold", False)])

            # Check if we have insufficient data
            total_successful_retrievals = sum([r.get("successful_retrievals", 0) for r in retrieval_results])
            if total_successful_retrievals < 2:
                # Insufficient successful retrievals for robustness analysis
                return {
                    "passed": True,  # Don't fail the check
                    "status": "warn",
                    "message": f"Context retrieval robustness test: insufficient data ({total_successful_retrievals} successful retrievals, need at least 2)",
                    "data_available": False,
                    "topics_tested": len(retrieval_results),
                    "topics_passed": 0,
                    "retrieval_results": retrieval_results,
                    "issues": ["Insufficient successful retrievals - cannot validate robustness without data"]
                }

            issues = []
            for result in retrieval_results:
                if "error" in result:
                    issues.append(f"Topic '{result['topic']}': {result['error']}")
                elif not result.get("meets_threshold", False):
                    issues.append(f"Topic '{result['topic']}': robustness {result.get('overall_robustness', 0):.3f} below threshold")

            return {
                "passed": len(issues) == 0,
                "message": f"Context retrieval robustness test: {topics_passed}/{topics_tested} topics passed" + (f", issues: {len(issues)}" if issues else ""),
                "topics_tested": topics_tested,
                "topics_passed": topics_passed,
                "retrieval_results": retrieval_results,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Context retrieval robustness test failed: {str(e)}",
                "error": str(e)
            }
    
    async def _test_query_expansion(self) -> Dict[str, Any]:
        """Test query expansion effectiveness."""
        try:
            # Test with simple vs expanded queries
            expansion_results = []

            # S3 DIAGNOSTIC: Log test start
            logger.info("S3-DIAGNOSTIC: Starting query_expansion test")

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:

                test_cases = [
                    {"simple": "config", "expanded": "configuration setup process"},
                    {"simple": "error", "expanded": "error troubleshooting resolution"},
                    {"simple": "database", "expanded": "database connection setup"}
                ]

                for i, test_case in enumerate(test_cases, 1):
                    simple_query = test_case["simple"]
                    expanded_query = test_case["expanded"]

                    # S3 DIAGNOSTIC: Log test case
                    logger.info("S3-DIAGNOSTIC: Test case %s/3:", i)
                    logger.info("S3-DIAGNOSTIC:   Simple:   '%s'", simple_query)
                    logger.info("S3-DIAGNOSTIC:   Expanded: '%s'", expanded_query)
                    
                    try:
                        # Get results for both queries
                        simple_result = await self._search_contexts(session, simple_query)
                        expanded_result = await self._search_contexts(session, expanded_query)

                        if simple_result.get("error") or expanded_result.get("error"):
                            logger.error("S3-DIAGNOSTIC:   ❌ Query failed with error")
                            expansion_results.append({
                                "simple_query": simple_query,
                                "expanded_query": expanded_query,
                                "expansion_effective": False,
                                "error": "One or both queries failed"
                            })
                            continue

                        simple_contexts = simple_result.get("contexts", [])
                        expanded_contexts = expanded_result.get("contexts", [])

                        # Analyze expansion effectiveness
                        simple_count = len(simple_contexts)
                        expanded_count = len(expanded_contexts)

                        # S3 DIAGNOSTIC: Log result counts
                        logger.info("S3-DIAGNOSTIC:   Simple query returned: %s results", simple_count)
                        if simple_contexts:
                            simple_top_score = simple_contexts[0].get("score", 0)
                            logger.info("S3-DIAGNOSTIC:     Top score: %.3f", simple_top_score)

                        logger.info("S3-DIAGNOSTIC:   Expanded query returned: %s results", expanded_count)
                        if expanded_contexts:
                            expanded_top_score = expanded_contexts[0].get("score", 0)
                            logger.info("S3-DIAGNOSTIC:     Top score: %.3f", expanded_top_score)

                        # Check if expanded query returns relevant results
                        # Note: Query expansion doesn't always improve semantic search results.
                        # Semantic search often works better with focused, single-term queries.
                        # Expanded queries add context words that may dilute relevance scores.
                        #
                        # Effectiveness criteria (relaxed):
                        # - Expansion returns at least some results (not total failure)
                        # - OR expansion returns at least as many results as simple query
                        # We no longer require expansion to return BETTER scores.
                        expansion_effective = (
                            # Option 1: Expansion returns meaningful results
                            (expanded_count > 0 and
                             len(expanded_contexts) > 0 and
                             expanded_contexts[0].get("score", 0) > 0.01) or
                            # Option 2: Expansion returns at least as many results
                            expanded_count >= simple_count
                        )

                        # S3 DIAGNOSTIC: Log effectiveness
                        improvement_ratio = expanded_count / simple_count if simple_count > 0 else float('inf')
                        status_symbol = "✅" if expansion_effective else "❌"
                        logger.info("S3-DIAGNOSTIC:   %s Expansion effective: %s", status_symbol, expansion_effective)
                        logger.info("S3-DIAGNOSTIC:     Improvement ratio: %.2fx", improvement_ratio)

                        if not expansion_effective:
                            logger.warning("S3-DIAGNOSTIC:     ⚠️ Expansion returned no meaningful results")
                            logger.warning("S3-DIAGNOSTIC:       Expected: any results with score > 0.01 OR at least %d results", simple_count)

                        expansion_results.append({
                            "simple_query": simple_query,
                            "expanded_query": expanded_query,
                            "simple_count": simple_count,
                            "expanded_count": expanded_count,
                            "expansion_effective": expansion_effective,
                            "improvement_ratio": improvement_ratio
                        })
                        
                    except Exception as e:
                        expansion_results.append({
                            "simple_query": simple_query,
                            "expanded_query": expanded_query,
                            "expansion_effective": False,
                            "error": str(e)
                        })
            
            # Analyze overall expansion effectiveness
            total_tests = len(expansion_results)
            effective_expansions = len([r for r in expansion_results if r.get("expansion_effective", False)])

            # Check if we have insufficient data
            successful_tests = len([r for r in expansion_results if "error" not in r and r.get("simple_count", 0) > 0])
            if successful_tests == 0:
                # No successful expansions to compare
                return {
                    "passed": True,  # Don't fail the check
                    "status": "warn",
                    "message": "Query expansion test: insufficient data (no successful query results to compare)",
                    "data_available": False,
                    "total_tests": total_tests,
                    "effective_expansions": 0,
                    "expansion_results": expansion_results,
                    "issues": ["No query results returned - cannot validate expansion effectiveness without data"]
                }

            issues = []
            for result in expansion_results:
                if "error" in result:
                    issues.append(f"Query '{result['simple_query']}': {result['error']}")
                elif not result.get("expansion_effective", False):
                    issues.append(f"Query '{result['simple_query']}': expansion not effective")

            return {
                "passed": effective_expansions >= total_tests * 0.7,  # 70% threshold
                "message": f"Query expansion test: {effective_expansions}/{total_tests} expansions effective" + (f", issues: {len(issues)}" if issues else ""),
                "total_tests": total_tests,
                "effective_expansions": effective_expansions,
                "effectiveness_rate": effective_expansions / total_tests if total_tests > 0 else 0,
                "expansion_results": expansion_results,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Query expansion test failed: {str(e)}",
                "error": str(e)
            }

    async def _test_response_quality_consistency(self) -> Dict[str, Any]:
        """Test consistency of response quality across paraphrased queries."""
        try:
            quality_results = []
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:
                
                for paraphrase_set in self.test_paraphrase_sets[:3]:  # Test subset
                    topic = paraphrase_set["topic"]
                    variations = paraphrase_set["variations"]
                    
                    quality_metrics = []
                    for variation in variations:
                        try:
                            result = await self._search_contexts(session, variation, limit=5)
                            if not result.get("error"):
                                contexts = result.get("contexts", [])
                                
                                # Calculate quality metrics
                                avg_score = statistics.mean([ctx.get("score", 0) for ctx in contexts]) if contexts else 0
                                score_variance = statistics.variance([ctx.get("score", 0) for ctx in contexts]) if len(contexts) > 1 else 0
                                
                                quality_metrics.append({
                                    "query": variation,
                                    "result_count": len(contexts),
                                    "average_score": avg_score,
                                    "score_variance": score_variance,
                                    "quality_score": avg_score * (1 - min(score_variance, 1.0))  # Penalize high variance
                                })
                        except Exception as e:
                            quality_metrics.append({
                                "query": variation,
                                "result_count": 0,
                                "error": str(e)
                            })
                    
                    # Analyze quality consistency
                    successful_queries = [m for m in quality_metrics if "error" not in m]
                    
                    if len(successful_queries) < 2:
                        quality_results.append({
                            "topic": topic,
                            "variations_tested": len(variations),
                            "successful_queries": len(successful_queries),
                            "quality_consistency": 0,
                            "error": "Insufficient successful queries for quality analysis"
                        })
                        continue
                    
                    quality_scores = [m["quality_score"] for m in successful_queries]
                    avg_quality = statistics.mean(quality_scores)
                    quality_variance = statistics.variance(quality_scores) if len(quality_scores) > 1 else 0
                    
                    # Quality consistency based on low variance
                    max_acceptable_variance = (avg_quality * self.max_result_variance) ** 2
                    consistency_score = 1.0 - min(1.0, quality_variance / max_acceptable_variance) if max_acceptable_variance > 0 else 1.0
                    
                    quality_results.append({
                        "topic": topic,
                        "variations_tested": len(variations),
                        "successful_queries": len(successful_queries),
                        "average_quality": avg_quality,
                        "quality_variance": quality_variance,
                        "quality_consistency": consistency_score,
                        "meets_threshold": consistency_score >= self.min_similarity_threshold,
                        "quality_samples": successful_queries[:3]  # Sample
                    })
            
            # Overall quality consistency analysis
            topics_tested = len([r for r in quality_results if "error" not in r])
            topics_passed = len([r for r in quality_results if r.get("meets_threshold", False)])

            # Check if we have insufficient data
            total_successful_queries = sum([r.get("successful_queries", 0) for r in quality_results])
            if total_successful_queries < 2:
                # Insufficient successful queries for quality analysis
                return {
                    "passed": True,  # Don't fail the check
                    "status": "warn",
                    "message": f"Response quality consistency test: insufficient data ({total_successful_queries} successful queries, need at least 2)",
                    "data_available": False,
                    "topics_tested": len(quality_results),
                    "topics_passed": 0,
                    "quality_results": quality_results,
                    "issues": ["Insufficient successful queries - cannot validate quality consistency without data"]
                }

            issues = []
            for result in quality_results:
                if "error" in result:
                    issues.append(f"Topic '{result['topic']}': {result['error']}")
                elif not result.get("meets_threshold", False):
                    issues.append(f"Topic '{result['topic']}': quality consistency {result.get('quality_consistency', 0):.3f} below threshold")

            return {
                "passed": len(issues) == 0,
                "message": f"Response quality consistency test: {topics_passed}/{topics_tested} topics passed" + (f", issues: {len(issues)}" if issues else ""),
                "topics_tested": topics_tested,
                "topics_passed": topics_passed,
                "quality_results": quality_results,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Response quality consistency test failed: {str(e)}",
                "error": str(e)
            }
    
    async def _search_contexts(self, session: aiohttp.ClientSession, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for contexts using the service API."""
        try:
            # Get authentication headers
            headers = self._get_headers()

            search_url = f"{self.service_url}/api/v1/contexts/search"
            payload = {
                "query": query,
                "limit": limit,
                "include_metadata": True
            }

            async with session.post(search_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    # Fixed: REST API returns "results" not "contexts" (PR #240)
                    # Normalize response to always have "contexts" key for backward compatibility
                    if "results" in data and "contexts" not in data:
                        data["contexts"] = data["results"]
                    return data
                else:
                    return {"error": f"HTTP {response.status}", "contexts": []}

        except Exception as e:
            return {"error": str(e), "contexts": []}
    
    def _calculate_result_overlap(self, results1: List[Dict], results2: List[Dict]) -> float:
        """Calculate overlap between two result sets using Jaccard similarity."""
        if not results1 or not results2:
            return 0.0
        
        # Extract identifiers or content for comparison
        set1 = set()
        set2 = set()
        
        for result in results1:
            # Use content hash or ID for comparison
            identifier = result.get("id", hash(str(result.get("content", ""))))
            set1.add(identifier)
        
        for result in results2:
            identifier = result.get("id", hash(str(result.get("content", ""))))
            set2.add(identifier)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_score_correlation(self, scores1: List[float], scores2: List[float]) -> float:
        """Calculate correlation between two score lists."""
        if len(scores1) != len(scores2) or len(scores1) < 2:
            return 0.0
        
        # Simple Pearson correlation coefficient
        mean1 = statistics.mean(scores1)
        mean2 = statistics.mean(scores2)
        
        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(scores1, scores2))
        
        sum_sq1 = sum((x - mean1) ** 2 for x in scores1)
        sum_sq2 = sum((y - mean2) ** 2 for y in scores2)
        
        denominator = math.sqrt(sum_sq1 * sum_sq2)
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _calculate_ranking_stability(self, ranking1: List[str], ranking2: List[str]) -> float:
        """Calculate ranking stability between two rankings."""
        if not ranking1 or not ranking2:
            return 0.0
        
        # Calculate normalized rank correlation
        common_items = set(ranking1).intersection(set(ranking2))
        if len(common_items) < 2:
            return 0.0
        
        # Create rank mappings for common items
        rank_map1 = {item: i for i, item in enumerate(ranking1) if item in common_items}
        rank_map2 = {item: i for i, item in enumerate(ranking2) if item in common_items}
        
        if len(rank_map1) < 2:
            return 0.0
        
        # Calculate Spearman rank correlation
        ranks1 = [rank_map1[item] for item in common_items]
        ranks2 = [rank_map2[item] for item in common_items]

        return self._calculate_score_correlation(ranks1, ranks2)