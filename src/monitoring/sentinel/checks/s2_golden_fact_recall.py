#!/usr/bin/env python3
"""
S2: Golden Fact Recall Check

Tests the ability to store and recall specific facts through natural language queries.
This validates that the core store/retrieve functionality works correctly.
"""

import json
import re
import time
import uuid
import aiohttp
from datetime import datetime
from typing import Dict, Any

from ..base_check import BaseCheck, APITestMixin
from ..models import CheckResult, SentinelConfig


class GoldenFactRecall(BaseCheck, APITestMixin):
    """
    S2: Comprehensive Semantic Search Validation (Enhanced)

    OPTIMIZATION (Phase 2): Consolidated S2/S9/S10 into unified semantic search test.

    This check now validates:
    - Core semantic search (original S2 golden facts)
    - Graph relationship traversal (from S9)
    - Content pipeline (from S10 - implicit in store/retrieve cycle)

    Previous structure:
    - S2: 6 queries (3 facts × 2 questions)
    - S9: 8 queries (graph intent validation)
    - S10: 5 queries (pipeline monitoring)
    - Total: 19 queries

    Enhanced structure (CURRENT):
    - Golden facts (semantic): 6 queries (3 facts × 2 questions)
    - Graph relationships: 6 queries (3 relationships × 2 questions)
    - Total: 12 queries (6 test cases × 2 questions each)

    Net savings: 7 queries per cycle (37% reduction from original S2/S9/S10 total of 19)

    IMPORTANT: Initial implementation used 4 test cases (8 queries) but was
    enhanced to 6 test cases (12 queries) after code review to provide better
    graph relationship coverage (3 graph test cases instead of 1).

    The 12-query implementation is CORRECT. This enhancement increased the total
    Sentinel query count from 18 to 22 queries/cycle. Earlier commit messages
    referencing "8 queries" or "18 total" reflect the initial design before
    code review enhancements.

    Comprehensive testing of graph intent and pipeline stages should be
    done in CI/CD, not runtime monitoring.
    """

    def __init__(self, config: SentinelConfig) -> None:
        super().__init__(config, "S2-golden-fact-recall", "Comprehensive semantic search validation")
        self.test_dataset = [
            # Original golden facts (semantic search validation)
            {
                "kv": {"name": "Matt"},
                "questions": ["What's my name?", "Who am I?"],
                "expect_contains": "Matt",
                "test_type": "semantic_search"
            },
            {
                "kv": {"food": "spicy"},
                "questions": ["What kind of food do I like?", "What food preference do I have?"],
                "expect_contains": "spicy",
                "test_type": "semantic_search"
            },
            {
                "kv": {"location": "San Francisco"},
                "questions": ["Where do I live?", "What's my location?"],
                "expect_contains": "San Francisco",
                "test_type": "semantic_search"
            },
            # Graph relationship validation (from S9) - Enhanced with 3 test cases
            {
                "kv": {"project": "Veris Memory", "tech_stack": "Python FastAPI Neo4j"},
                "questions": [
                    "What technology stack is used in Veris Memory?",
                    "How is Veris Memory implemented?"
                ],
                "expect_contains": "Python",
                "test_type": "graph_relationship"
            },
            {
                "kv": {"feature": "semantic search", "dependencies": "Qdrant vector database embeddings"},
                "questions": [
                    "What does semantic search depend on?",
                    "What components are needed for semantic search?"
                ],
                "expect_contains": "Qdrant",
                "test_type": "graph_relationship"
            },
            {
                "kv": {"component": "monitoring", "related_systems": "Sentinel checks metrics Neo4j"},
                "questions": [
                    "What systems are related to monitoring?",
                    "How does monitoring connect to other components?"
                ],
                "expect_contains": "Sentinel",
                "test_type": "graph_relationship"
            }
        ]
        
    async def run_check(self) -> CheckResult:
        """Execute golden fact recall check."""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                test_results = []
                total_tests = 0
                passed_tests = 0
                
                for i, test_case in enumerate(self.test_dataset):
                    test_result = await self._run_single_test(session, test_case, i)
                    test_results.append(test_result)
                    
                    total_tests += test_result["total_questions"]
                    passed_tests += test_result["passed_questions"]
                
                # Determine overall status
                success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
                
                if success_rate >= 0.8:
                    status = "pass"
                    message = f"Golden fact recall successful: {passed_tests}/{total_tests} tests passed"
                elif success_rate >= 0.6:
                    status = "warn"
                    message = f"Golden fact recall degraded: {passed_tests}/{total_tests} tests passed"
                else:
                    status = "fail"
                    message = f"Golden fact recall failed: {passed_tests}/{total_tests} tests passed"
                
                latency_ms = (time.time() - start_time) * 1000
                return CheckResult(
                    check_id=self.check_id,
                    timestamp=datetime.utcnow(),
                    status=status,
                    latency_ms=latency_ms,
                    message=message,
                    details={
                        "total_tests": total_tests,
                        "passed_tests": passed_tests,
                        "success_rate": success_rate,
                        "test_results": test_results,
                        "latency_ms": latency_ms
                    }
                )
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return CheckResult(
                check_id=self.check_id,
                timestamp=datetime.utcnow(),
                status="fail",
                latency_ms=latency_ms,
                message=f"Golden fact recall exception: {str(e)}",
                details={"error": str(e)}
            )
    
    async def _run_single_test(self, session: aiohttp.ClientSession, test_case: Dict[str, Any], test_index: int) -> Dict[str, Any]:
        """Run a single golden fact test case.

        PR #399: Now includes cleanup after testing to prevent database bloat.
        Test facts are deleted after recall verification.
        """
        user_id = f"sentinel_test_{uuid.uuid4().hex[:8]}"
        context_id = None

        # Store the fact
        store_result = await self._store_fact(session, test_case["kv"], user_id)
        if not store_result["success"]:
            return {
                "test_index": test_index,
                "test_case": test_case,
                "store_success": False,
                "store_error": store_result["message"],
                "total_questions": len(test_case["questions"]),
                "passed_questions": 0,
                "question_results": [],
                "cleanup_result": None
            }

        # Extract context_id for cleanup
        response_data = store_result.get("response", {})
        context_id = response_data.get("id") or response_data.get("context_id")

        # Test each question
        question_results = []
        passed_questions = 0

        for question in test_case["questions"]:
            question_result = await self._test_recall(session, question, test_case["expect_contains"], user_id)
            question_results.append(question_result)

            if question_result["success"]:
                passed_questions += 1

        # PR #399: Clean up test fact after recall test to prevent database bloat
        cleanup_result = None
        if context_id:
            cleanup_result = await self._cleanup_fact(session, context_id)

        return {
            "test_index": test_index,
            "test_case": test_case,
            "store_success": True,
            "store_result": store_result,
            "total_questions": len(test_case["questions"]),
            "passed_questions": passed_questions,
            "question_results": question_results,
            "cleanup_result": cleanup_result
        }
    
    async def _store_fact(self, session: aiohttp.ClientSession, fact_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Store a fact in the system."""
        # Fixed: Use correct MCP API format (PR #240 + #241)
        # - content: Dict, not JSON string
        # - type: must be one of: design, decision, trace, sprint, log (PR #241 fix)
        # - author: must be in metadata, not top-level (per contracts/store_context.json)
        # Note: Using "log" type since "fact" is not in allowed values
        store_payload = {
            "content": fact_data,
            "type": "log",
            "metadata": {
                "author": user_id,  # Author goes in metadata per API contract
                "test_type": "golden_recall",
                "sentinel": True,
                "content_type": "fact"
            }
        }
        
        success, message, latency, response_data = await self.test_api_call(
            session,
            "POST",
            f"{self.config.target_base_url}/tools/store_context",
            data=store_payload,
            expected_status=200,
            timeout=10.0
        )
        
        return {
            "success": success,
            "message": message,
            "latency_ms": latency,
            "response": response_data,
            "payload": store_payload
        }
    
    async def _test_recall(self, session: aiohttp.ClientSession, question: str, expected_content: str, user_id: str) -> Dict[str, Any]:
        """Test recalling a fact through a natural language question."""
        # Fixed: Use correct MCP API format (PR #240)
        # - metadata_filters: {"author": user_id} for filtering by metadata fields (per contracts/retrieve_context.json)
        # - "filters" parameter only supports date_from, date_to, status, tags
        query_payload = {
            "query": question,
            "limit": 5,
            "metadata_filters": {"author": user_id}  # Use metadata_filters for custom metadata fields
        }
        
        success, message, latency, response_data = await self.test_api_call(
            session,
            "POST",
            f"{self.config.target_base_url}/tools/retrieve_context",
            data=query_payload,
            expected_status=200,
            timeout=15.0
        )
        
        if not success:
            return {
                "question": question,
                "expected_content": expected_content,
                "success": False,
                "message": f"API call failed: {message}",
                "latency_ms": latency
            }
        
        # Check if the expected content is in the response
        # Fixed: API returns "results" not "contexts" (PR #240)
        if not response_data or "results" not in response_data:
            return {
                "question": question,
                "expected_content": expected_content,
                "success": False,
                "message": "No results returned",
                "latency_ms": latency,
                "response": response_data
            }

        contexts = response_data["results"]
        found_expected = False

        for context in contexts:
            # Fixed: content is a dict, need to convert to string for searching (PR #240)
            content = context.get("content", {})
            # Convert content dict to string for searching
            content_str = json.dumps(content) if isinstance(content, dict) else str(content)
            if expected_content.lower() in content_str.lower():
                found_expected = True
                break
        
        return {
            "question": question,
            "expected_content": expected_content,
            "success": found_expected,
            "message": "Expected content found" if found_expected else f"Expected '{expected_content}' not found in response",
            "latency_ms": latency,
            "response": response_data,
            "contexts_count": len(contexts)
        }

    # Pattern for valid context IDs (UUIDs, hex strings, or alphanumeric with hyphens/underscores)
    # This prevents injection by rejecting any ID with special characters
    VALID_CONTEXT_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')

    async def _cleanup_fact(self, session: aiohttp.ClientSession, context_id: str) -> Dict[str, Any]:
        """Clean up a test fact after recall test to prevent database bloat.

        PR #399: Sentinel test data was filling Veris Memory with permanent entries.
        PR #403: Now deletes from ALL backends (Neo4j + Qdrant) via internal cleanup endpoint.

        Uses /internal/sentinel/cleanup endpoint which:
        - Deletes from Neo4j (graph database)
        - Deletes from Qdrant (vector database)
        - Requires SENTINEL_API_KEY for authentication
        - Bypasses human-only restrictions for test data cleanup

        Security: context_id is validated against a safe pattern before sending.
        """
        import os

        if not context_id:
            return {"success": False, "message": "No context_id provided"}

        # Security: Validate context_id format
        if not self.VALID_CONTEXT_ID_PATTERN.match(context_id):
            return {"success": False, "message": f"Invalid context_id format: {context_id[:50]}"}

        # Get sentinel API key from environment
        sentinel_key = os.getenv("SENTINEL_API_KEY", "")

        # Use internal sentinel cleanup endpoint for multi-backend deletion
        cleanup_payload = {
            "context_id": context_id,
            "sentinel_key": sentinel_key
        }

        try:
            success, message, latency, response_data = await self.test_api_call(
                session,
                "POST",
                f"{self.config.target_base_url}/internal/sentinel/cleanup",
                data=cleanup_payload,
                expected_status=200,
                timeout=5.0
            )

            if success and response_data:
                return {
                    "success": response_data.get("success", False),
                    "context_id": context_id,
                    "deleted_from": response_data.get("deleted_from", []),
                    "errors": response_data.get("errors"),
                    "latency_ms": latency,
                    "message": response_data.get("message", "")
                }
            else:
                # Fallback to Neo4j-only cleanup if endpoint not available
                return await self._cleanup_fact_neo4j_only(session, context_id)
        except Exception as e:
            # Fallback to Neo4j-only cleanup on error
            return await self._cleanup_fact_neo4j_only(session, context_id)

    async def _cleanup_fact_neo4j_only(self, session: aiohttp.ClientSession, context_id: str) -> Dict[str, Any]:
        """Fallback cleanup using Neo4j only (for backward compatibility).

        Used when the new /internal/sentinel/cleanup endpoint is not available.
        """
        cleanup_payload = {
            "query": f"MATCH (n:Context {{id: '{context_id}'}}) DETACH DELETE n RETURN count(n) as deleted"
        }

        try:
            success, message, latency, response_data = await self.test_api_call(
                session,
                "POST",
                f"{self.config.target_base_url}/tools/query_graph",
                data=cleanup_payload,
                expected_status=200,
                timeout=5.0
            )

            if success and response_data:
                deleted_count = response_data.get("results", [{}])[0].get("deleted", 0)
                return {
                    "success": True,
                    "context_id": context_id,
                    "deleted_from": ["neo4j"] if deleted_count > 0 else [],
                    "deleted_count": deleted_count,
                    "latency_ms": latency,
                    "message": "Fallback to Neo4j-only cleanup (Qdrant vectors may remain)"
                }
            else:
                return {
                    "success": False,
                    "context_id": context_id,
                    "message": message,
                    "latency_ms": latency
                }
        except Exception as e:
            return {
                "success": False,
                "context_id": context_id,
                "message": f"Cleanup failed: {str(e)}"
            }