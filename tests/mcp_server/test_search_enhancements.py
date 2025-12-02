#!/usr/bin/env python3
"""
Test suite for search enhancement improvements.

These tests validate the fixes for issues identified by workspace 001:
- Exact filename searches should return the file as top result
- Technical searches should prioritize code/documentation over conversations
- Recent content shouldn't completely overshadow older relevant content
- All technical term recognition should work correctly

Author: Workspace 002
Date: 2025-08-19
"""

import pytest
from datetime import datetime, timedelta
from src.mcp_server.search_enhancements import (
    calculate_exact_match_boost,
    apply_context_type_weight,
    calculate_recency_decay,
    calculate_technical_boost,
    apply_search_enhancements,
    is_technical_query,
    CONTEXT_TYPE_WEIGHTS,
    TECHNICAL_TERMS
)


class TestExactMatchBoosting:
    """Test exact match boosting functionality."""
    
    def test_exact_filename_match(self) -> None:
        """Test that exact filename matches get highest boost."""
        query = "server.py"
        content = "This is the main server implementation"
        metadata = {"file_path": "/src/mcp_server/server.py"}
        
        boost = calculate_exact_match_boost(query, content, metadata)
        assert boost == 5.0, "Exact filename match should get 5x boost"
    
    def test_partial_filename_match(self) -> None:
        """Test that partial filename matches get moderate boost."""
        query = "server"
        content = "Server implementation code"
        metadata = {"file_path": "/src/mcp_server/server.py"}
        
        boost = calculate_exact_match_boost(query, content, metadata)
        # 3.0 (partial filename) but phrase "server" is also in content so doesn't get additional boost
        assert boost >= 3.0, "Partial filename match should get at least 3x boost"
    
    def test_title_match(self) -> None:
        """Test that title matches get boosted."""
        query = "search enhancement"
        content = "Implementation details"
        metadata = {"title": "Search Enhancement Analysis"}
        
        boost = calculate_exact_match_boost(query, content, metadata)
        assert boost == 2.0, "Title match should get 2x boost"
    
    def test_exact_phrase_in_content(self) -> None:
        """Test that exact phrase matches in content get boosted."""
        query = "MCP Python SDK"
        content = "This module implements the Model Context Protocol server using MCP Python SDK"
        
        boost = calculate_exact_match_boost(query, content, {})
        # With new logic, should only get phrase match OR keyword match, not both
        assert boost == 1.5, "Exact phrase match should get 1.5x boost"
    
    def test_all_keywords_present(self) -> None:
        """Test that having all keywords present gets boosted."""
        query = "vector embedding search"
        content = "The vector database stores embeddings for semantic search capabilities"
        
        boost = calculate_exact_match_boost(query, content, {})
        assert boost >= 1.3, "All keywords present should get at least 1.3x boost"
    
    def test_combined_boosts(self) -> None:
        """Test that multiple boost factors multiply correctly."""
        query = "server.py"
        content = "server.py contains the main implementation"
        metadata = {"file_path": "/src/server.py", "title": "server.py documentation"}
        
        boost = calculate_exact_match_boost(query, content, metadata)
        # 5.0 (exact filename) * 2.0 (title match) = 10.0
        # Phrase boost doesn't apply when filename match exists (by design)
        assert boost == 10.0, "Combined boosts should multiply (filename + title)"


class TestContextTypeWeighting:
    """Test context type weighting functionality."""
    
    def test_code_type_highest_weight(self) -> None:
        """Test that code contexts get highest weight."""
        result = {"payload": {"category": "python_code"}}
        weight = apply_context_type_weight(result)
        assert weight == CONTEXT_TYPE_WEIGHTS["code"], "Code should get highest weight"
    
    def test_documentation_weight(self) -> None:
        """Test that documentation gets moderate weight."""
        result = {"payload": {"category": "documentation"}}
        weight = apply_context_type_weight(result)
        assert weight == CONTEXT_TYPE_WEIGHTS["documentation"], "Documentation should get moderate weight"
    
    def test_conversation_lowest_weight(self) -> None:
        """Test that conversations get lowest weight."""
        result = {"payload": {"type": "conversation"}}
        weight = apply_context_type_weight(result)
        assert weight == CONTEXT_TYPE_WEIGHTS["conversation"], "Conversations should get lowest weight"
    
    def test_unknown_type_default_weight(self) -> None:
        """Test that unknown types get default weight."""
        result = {"payload": {"type": "something_else"}}
        weight = apply_context_type_weight(result)
        assert weight == 1.0, "Unknown types should get default weight of 1.0"
    
    def test_category_overrides_type(self) -> None:
        """Test that category field takes priority over type field."""
        result = {"payload": {"category": "code", "type": "conversation"}}
        weight = apply_context_type_weight(result)
        assert weight == CONTEXT_TYPE_WEIGHTS["code"], "Category should override type"


class TestRecencyDecay:
    """Test recency decay functionality."""
    
    def test_recent_content_no_decay(self) -> None:
        """Test that very recent content has minimal decay."""
        now = datetime.now()
        score = calculate_recency_decay(now, 1.0)
        assert score >= 0.99, "Content from today should have minimal decay"
    
    def test_one_week_old_half_decay(self) -> None:
        """Test that week-old content has ~50% decay."""
        week_ago = datetime.now() - timedelta(days=7)
        score = calculate_recency_decay(week_ago, 1.0, decay_rate=7.0)
        # exp(-7/7) = exp(-1) ≈ 0.368
        assert 0.35 <= score <= 0.40, "Week-old content should have ~37% weight"
    
    def test_two_weeks_old_quarter_decay(self) -> None:
        """Test that two-week-old content has ~25% decay."""
        two_weeks_ago = datetime.now() - timedelta(days=14)
        score = calculate_recency_decay(two_weeks_ago, 1.0, decay_rate=7.0)
        # exp(-14/7) = exp(-2) ≈ 0.135
        assert 0.13 <= score <= 0.15, "Two-week-old content should have ~14% weight"
    
    def test_minimum_decay_threshold(self) -> None:
        """Test that decay never goes below 10%."""
        very_old = datetime.now() - timedelta(days=365)
        score = calculate_recency_decay(very_old, 1.0)
        assert score >= 0.1, "Decay should never go below 10%"
    
    def test_invalid_timestamp_no_decay(self) -> None:
        """Test that invalid timestamps don't cause errors."""
        score = calculate_recency_decay(None, 1.0)
        assert score == 1.0, "Invalid timestamp should return original score"
        
        score = calculate_recency_decay("invalid", 1.0)
        assert score == 1.0, "Invalid timestamp string should return original score"


class TestTechnicalBoost:
    """Test technical term boosting functionality."""
    
    def test_technical_query_with_technical_content(self) -> None:
        """Test that technical queries boost technical content."""
        query = "python function implementation"
        content = "This function implements the algorithm using python classes and methods"
        
        boost = calculate_technical_boost(query, content)
        assert boost > 1.0, "Technical content should be boosted for technical queries"
    
    def test_file_extension_query(self) -> None:
        """Test that queries with file extensions are recognized as technical."""
        query = "server.py"
        content = "Server implementation in Python"
        
        boost = calculate_technical_boost(query, content)
        assert boost >= 1.0, "File extension queries should trigger technical boost"
    
    def test_non_technical_query_no_boost(self) -> None:
        """Test that non-technical queries don't boost technical content."""
        query = "hello world"
        content = "This function implements python code"
        
        boost = calculate_technical_boost(query, content)
        assert boost == 1.0, "Non-technical queries shouldn't boost technical content"
    
    def test_logarithmic_boost_scaling(self) -> None:
        """Test that boost scales logarithmically with technical term count."""
        query = "python api implementation"
        
        # Content with few technical terms
        content1 = "python function"
        boost1 = calculate_technical_boost(query, content1)
        
        # Content with many technical terms
        content2 = " ".join(list(TECHNICAL_TERMS)[:20])
        boost2 = calculate_technical_boost(query, content2)
        
        # More terms should give higher boost, but not linearly
        assert boost2 > boost1, "More technical terms should give higher boost"
        assert boost2 < boost1 * 10, "Boost should scale logarithmically, not linearly"


class TestIsTechnicalQuery:
    """Test technical query detection."""
    
    def test_detects_technical_terms(self) -> None:
        """Test that queries with technical terms are detected."""
        assert is_technical_query("python function implementation")
        assert is_technical_query("docker deployment kubernetes")
        assert is_technical_query("api endpoint json")
    
    def test_detects_file_extensions(self) -> None:
        """Test that queries with file extensions are detected."""
        assert is_technical_query("server.py")
        assert is_technical_query("config.json")
        assert is_technical_query("README.md")
    
    def test_detects_code_patterns(self) -> None:
        """Test that queries with code patterns are detected."""
        assert is_technical_query("function()")
        assert is_technical_query("array[]")
        assert is_technical_query("object{}")
        assert is_technical_query("arrow ->")
    
    def test_non_technical_queries(self) -> None:
        """Test that non-technical queries are not detected."""
        assert not is_technical_query("hello world")
        assert not is_technical_query("what is the weather")
        assert not is_technical_query("congratulations on your work")


class TestSearchEnhancementsIntegration:
    """Test the full search enhancement pipeline."""
    
    def test_workspace_001_case_server_py(self) -> None:
        """Test workspace 001's case: searching for 'server.py' should return server.py file first."""
        results = [
            {
                "id": "1",
                "score": 0.8,
                "payload": {
                    "content": "Congratulations on backup system",
                    "type": "conversation",
                    "created_at": datetime.now().isoformat()
                }
            },
            {
                "id": "2", 
                "score": 0.7,
                "payload": {
                    "content": "MCP server implementation code",
                    "file_path": "/src/mcp_server/server.py",
                    "category": "python_code",
                    "created_at": (datetime.now() - timedelta(days=5)).isoformat()
                }
            }
        ]
        
        enhanced = apply_search_enhancements(results, "server.py")
        
        # server.py file should be ranked first despite lower original score
        assert enhanced[0]["id"] == "2", "server.py file should rank first"
        assert enhanced[0]["enhanced_score"] > enhanced[1]["enhanced_score"]
    
    def test_workspace_001_case_technical_search(self) -> None:
        """Test workspace 001's case: technical searches should prioritize code over conversations."""
        results = [
            {
                "id": "conv",
                "score": 0.9,
                "payload": {
                    "content": "Great work on the vector implementation",
                    "type": "conversation",
                    "created_at": datetime.now().isoformat()
                }
            },
            {
                "id": "code",
                "score": 0.6,
                "payload": {
                    "content": "def generate_embedding(self, text): return vector",
                    "category": "code",
                    "created_at": (datetime.now() - timedelta(days=3)).isoformat()
                }
            }
        ]
        
        enhanced = apply_search_enhancements(results, "vector embedding implementation")
        
        # Code should rank higher than conversation for technical query
        assert enhanced[0]["id"] == "code", "Code should rank first for technical queries"
    
    def test_workspace_001_case_recency_bias(self) -> None:
        """Test workspace 001's case: recent content shouldn't completely overshadow older relevant content."""
        results = [
            {
                "id": "recent_chat",
                "score": 0.3,  # Lower initial score for conversation
                "payload": {
                    "content": "Nice work today",
                    "type": "conversation",
                    "created_at": datetime.now().isoformat()
                }
            },
            {
                "id": "old_docs",
                "score": 0.65,
                "payload": {
                    "content": "Hybrid memory combines vector embeddings with graph relationships",
                    "category": "documentation",
                    "file_path": "README.md",
                    "created_at": (datetime.now() - timedelta(days=30)).isoformat()
                }
            }
        ]
        
        enhanced = apply_search_enhancements(results, "hybrid memory vector embeddings")
        
        # Documentation should rank higher due to content match and type weight
        # even though it's older
        assert enhanced[0]["id"] == "old_docs", "Relevant documentation should rank first despite age"
    
    def test_all_enhancements_combined(self) -> None:
        """Test that all enhancements work together correctly."""
        now = datetime.now()
        results = [
            {
                "id": "A",
                "score": 0.5,
                "payload": {
                    "content": "Random conversation",
                    "type": "conversation",
                    "created_at": now.isoformat()
                }
            },
            {
                "id": "B",
                "score": 0.4,
                "payload": {
                    "content": "search_enhancements.py implementation",
                    "file_path": "search_enhancements.py",
                    "category": "code",
                    "created_at": (now - timedelta(days=2)).isoformat()
                }
            }
        ]
        
        enhanced = apply_search_enhancements(results, "search_enhancements.py")
        
        # File B should win due to:
        # - Exact filename match (5x or more with phrase boost)
        # - Code type weight (2x)
        # - Technical boost
        # Despite slightly older and lower original score
        assert enhanced[0]["id"] == "B"
        assert enhanced[0]["score_boosts"]["exact_match"] >= 5.0  # At least 5x for exact filename
        assert enhanced[0]["score_boosts"]["type_weight"] == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])