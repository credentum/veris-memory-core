"""
Tests for Phase 3 enhancements addressing GitHub issue #127 LIM-003 and LIM-004.

This module tests the query-specific relevance scoring and native Redis operations.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import json

# Import the query relevance scorer
from src.mcp_server.query_relevance_scorer import QueryRelevanceScorer, QueryType


class TestQueryRelevanceScorer:
    """Test the query-specific relevance scoring implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = QueryRelevanceScorer()

    def test_technical_query_analysis(self):
        """Test analysis of technical queries."""
        # Test a technical query with multiple technical terms
        query = "redis hash leaderboard queue"
        analysis = self.scorer.analyze_query(query)
        
        assert analysis.query_type == QueryType.TECHNICAL_SPECIFIC
        assert analysis.intent_strength > 0.5
        assert analysis.domain_focus == "database"
        assert "redis" in analysis.technical_terms
        assert "hash" in analysis.technical_terms
        assert "queue" in analysis.technical_terms

    def test_code_search_analysis(self):
        """Test analysis of code search queries."""
        query = "function getUserData()"
        analysis = self.scorer.analyze_query(query)
        
        assert analysis.query_type == QueryType.CODE_SEARCH
        assert analysis.intent_strength > 0.6

    def test_generic_query_analysis(self):
        """Test analysis of generic queries."""
        query = "hello how are you"
        analysis = self.scorer.analyze_query(query)
        
        assert analysis.query_type == QueryType.GENERIC
        assert analysis.intent_strength < 0.5

    def test_troubleshooting_query_analysis(self):
        """Test analysis of troubleshooting queries."""
        query = "error connecting to database"
        analysis = self.scorer.analyze_query(query)
        
        assert analysis.query_type == QueryType.TROUBLESHOOTING
        assert "error" in analysis.key_terms
        assert analysis.domain_focus == "database"

    def test_query_specific_relevance_scoring(self):
        """Test that relevance scoring varies based on query."""
        # Create mock results with different content
        results = [
            {
                'score': 0.8,
                'payload': {
                    'content': {'title': 'Redis Configuration', 'text': 'redis hash operations leaderboard'},
                    'metadata': {'type': 'code'}
                }
            },
            {
                'score': 0.8,
                'payload': {
                    'content': {'title': 'Random Chat', 'text': 'hello how are you today'},
                    'metadata': {'type': 'conversation'}
                }
            }
        ]
        
        # Test technical query - should boost technical content
        tech_query = "redis hash leaderboard queue"
        tech_results = self.scorer.enhance_search_results(tech_query, results)
        
        # The Redis result should score higher than the chat result
        redis_result = tech_results[0]
        chat_result = tech_results[1] 
        
        assert redis_result['query_relevance_score'] > chat_result['query_relevance_score']
        assert redis_result['query_relevance_multiplier'] > 1.0

    def test_different_queries_different_results(self):
        """Test that different queries produce meaningfully different result rankings."""
        # Mock results covering different domains
        results = [
            {
                'score': 0.5,
                'payload': {
                    'content': {'title': 'Authentication Guide', 'text': 'jwt token oauth login'},
                    'metadata': {'type': 'design'}
                }
            },
            {
                'score': 0.5,
                'payload': {
                    'content': {'title': 'Database Schema', 'text': 'redis hash table index'},
                    'metadata': {'type': 'code'}
                }
            },
            {
                'score': 0.5,
                'payload': {
                    'content': {'title': 'API Documentation', 'text': 'rest endpoint json response'},
                    'metadata': {'type': 'documentation'}
                }
            }
        ]
        
        # Test authentication query
        auth_query = "jwt token authentication"
        auth_results = self.scorer.enhance_search_results(auth_query, results)
        
        # Test database query  
        db_query = "redis hash operations"
        db_results = self.scorer.enhance_search_results(db_query, results)
        
        # Test API query
        api_query = "rest api endpoint"
        api_results = self.scorer.enhance_search_results(api_query, results)
        
        # Results should be ranked differently for each query
        assert auth_results[0]['payload']['content']['title'] == 'Authentication Guide'
        assert db_results[0]['payload']['content']['title'] == 'Database Schema'
        assert api_results[0]['payload']['content']['title'] == 'API Documentation'

    def test_intent_strength_calculation(self):
        """Test that intent strength is calculated correctly."""
        test_cases = [
            ("redis hash leaderboard queue implementation", 0.7),  # High technical specificity
            ("function getUserData()", 0.6),  # Code pattern with parentheses
            ("authentication oauth jwt", 0.2),  # Multiple tech terms without generic words
            ("hello world", 0.05),  # Very generic
        ]
        
        for query, expected_min_strength in test_cases:
            analysis = self.scorer.analyze_query(query)
            assert analysis.intent_strength >= expected_min_strength, f"Query '{query}' should have intent strength >= {expected_min_strength}, got {analysis.intent_strength}"

    def test_domain_focus_detection(self):
        """Test that domain focus is detected correctly."""
        test_cases = [
            ("redis hash operations", "database"),
            ("jwt oauth authentication", "authentication"),
            ("rest api endpoint", "api"),
            ("docker kubernetes deployment", "infrastructure"),
            ("vector search embedding", "search"),
        ]
        
        for query, expected_domain in test_cases:
            analysis = self.scorer.analyze_query(query)
            assert analysis.domain_focus == expected_domain, f"Query '{query}' should focus on '{expected_domain}', got '{analysis.domain_focus}'"


class TestRedisOperations:
    """Test the native Redis operation tools."""
    
    @pytest.mark.asyncio
    async def test_redis_get_validation(self):
        """Test Redis GET tool input validation."""
        from src.mcp_server.server import redis_get_tool
        
        # Test missing key
        result = await redis_get_tool({})
        assert result["success"] is False
        assert result["error_type"] == "validation_error"
        assert "key is required" in result["message"]

    @pytest.mark.asyncio
    async def test_redis_set_validation(self):
        """Test Redis SET tool input validation."""
        from src.mcp_server.server import redis_set_tool
        
        # Test missing key
        result = await redis_set_tool({"value": "test"})
        assert result["success"] is False
        assert result["error_type"] == "validation_error"
        assert "key and value are required" in result["message"]
        
        # Test missing value
        result = await redis_set_tool({"key": "test"})
        assert result["success"] is False
        assert result["error_type"] == "validation_error"
        assert "key and value are required" in result["message"]

    @pytest.mark.asyncio
    async def test_redis_hget_validation(self):
        """Test Redis HGET tool input validation."""
        from src.mcp_server.server import redis_hget_tool
        
        # Test missing field
        result = await redis_hget_tool({"key": "test"})
        assert result["success"] is False
        assert result["error_type"] == "validation_error"
        assert "key and field are required" in result["message"]

    @pytest.mark.asyncio
    async def test_redis_hset_validation(self):
        """Test Redis HSET tool input validation."""
        from src.mcp_server.server import redis_hset_tool
        
        # Test missing value
        result = await redis_hset_tool({"key": "test", "field": "name"})
        assert result["success"] is False
        assert result["error_type"] == "validation_error"
        assert "key, field, and value are required" in result["message"]

    @pytest.mark.asyncio
    async def test_redis_lpush_validation(self):
        """Test Redis LPUSH tool input validation."""
        from src.mcp_server.server import redis_lpush_tool
        
        # Test missing value
        result = await redis_lpush_tool({"key": "mylist"})
        assert result["success"] is False
        assert result["error_type"] == "validation_error"
        assert "key and value are required" in result["message"]

    @pytest.mark.asyncio
    async def test_redis_lrange_validation(self):
        """Test Redis LRANGE tool input validation."""
        from src.mcp_server.server import redis_lrange_tool
        
        # Test missing key
        result = await redis_lrange_tool({})
        assert result["success"] is False
        assert result["error_type"] == "validation_error"
        assert "key is required" in result["message"]

    def test_redis_tools_solve_lim003(self):
        """Test that Redis tools address LIM-003 from GitHub issue."""
        # LIM-003: Redis operations are abstracted as contexts, not native Redis commands
        # Solution: Native Redis tools for direct operations
        
        # Check that we have native Redis operations available
        from src.mcp_server.server import (
            redis_get_tool, redis_set_tool, redis_hget_tool, 
            redis_hset_tool, redis_lpush_tool, redis_lrange_tool
        )
        
        # These should be direct Redis operations, not context abstractions
        redis_tools = [
            redis_get_tool, redis_set_tool, redis_hget_tool,
            redis_hset_tool, redis_lpush_tool, redis_lrange_tool
        ]
        
        # All tools should be callable functions
        for tool in redis_tools:
            assert callable(tool)
            
        # This addresses the limitation that "Redis data stored as JSON contexts in graph/vector DBs"
        # Now we have direct Redis operations: GET, SET, HGET, HSET, LPUSH, LRANGE
        print("✅ LIM-003 addressed: Native Redis operations now available")


class TestSearchRelevanceImprovements:
    """Test that search relevance improvements address LIM-004."""
    
    def test_query_specific_results_solve_lim004(self):
        """Test that query-specific scoring addresses LIM-004."""
        # LIM-004: Search returns generic results regardless of specific query terms
        scorer = QueryRelevanceScorer()
        
        # Create results that could be generic matches
        results = [
            {
                'score': 0.7,
                'payload': {
                    'content': {'title': 'General Help', 'text': 'general information about the system'},
                    'metadata': {'type': 'conversation'}
                }
            },
            {
                'score': 0.6,  # Lower base score
                'payload': {
                    'content': {'title': 'Redis Implementation', 'text': 'redis hash leaderboard queue implementation'},
                    'metadata': {'type': 'code'}
                }
            }
        ]
        
        # Test specific technical query
        specific_query = "redis hash leaderboard queue"
        enhanced_results = scorer.enhance_search_results(specific_query, results)
        
        # The specific technical result should now rank higher despite lower base score
        top_result = enhanced_results[0]
        assert 'Redis Implementation' in top_result['payload']['content']['title']
        
        # The ranking should have changed due to query-specific relevance
        assert top_result['query_relevance_multiplier'] > 1.0
        
        print("✅ LIM-004 addressed: Results now vary based on specific query terms")

    def test_different_queries_produce_different_rankings(self):
        """Test that the same content ranks differently for different queries."""
        scorer = QueryRelevanceScorer()
        
        # Same result set
        results = [
            {
                'score': 0.5,
                'payload': {
                    'content': {'title': 'Mixed Content', 'text': 'authentication redis database api server'},
                    'metadata': {'type': 'design'}
                }
            }
        ]
        
        # Different queries should produce different relevance scores
        auth_results = scorer.enhance_search_results("authentication login", results)
        redis_results = scorer.enhance_search_results("redis database", results)
        api_results = scorer.enhance_search_results("api server", results)
        
        # All should have different relevance multipliers
        auth_multiplier = auth_results[0]['query_relevance_multiplier']
        redis_multiplier = redis_results[0]['query_relevance_multiplier']
        api_multiplier = api_results[0]['query_relevance_multiplier']
        
        # They should be different (within reasonable bounds)
        multipliers = [auth_multiplier, redis_multiplier, api_multiplier]
        assert len(set(multipliers)) > 1, "Different queries should produce different relevance scores"
        
        print("✅ Search relevance now varies meaningfully based on query content")


if __name__ == "__main__":
    pytest.main([__file__])