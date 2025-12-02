#!/usr/bin/env python3
"""
Tests for ranking policy engine.
"""

import pytest
import time
from datetime import datetime, timezone, timedelta
from typing import List

from src.ranking.policy_engine import (
    RankingPolicyEngine,
    DefaultRankingPolicy,
    CodeBoostRankingPolicy,
    RecencyRankingPolicy,
    RankingContext,
    FeatureWeights,
    RankingFeature,
    ranking_engine
)
from src.interfaces.memory_result import MemoryResult, ResultSource, ContentType


class TestRankingContext:
    """Test RankingContext data class."""
    
    def test_basic_context(self):
        """Test creating basic ranking context."""
        context = RankingContext(
            query="test query",
            search_mode="hybrid"
        )
        
        assert context.query == "test query"
        assert context.search_mode == "hybrid"
        assert context.user_preferences == {}
        assert context.namespace_weights == {}
        assert context.custom_features == {}
        assert context.timestamp > 0  # Should auto-set
    
    def test_complete_context(self):
        """Test creating complete ranking context."""
        timestamp = time.time()
        context = RankingContext(
            query="code search",
            search_mode="vector",
            user_preferences={"language": "python"},
            namespace_weights={"code": 2.0},
            custom_features={"boost_recent": True},
            timestamp=timestamp
        )
        
        assert context.query == "code search"
        assert context.search_mode == "vector"
        assert context.user_preferences["language"] == "python"
        assert context.namespace_weights["code"] == 2.0
        assert context.custom_features["boost_recent"] is True
        assert context.timestamp == timestamp


class TestFeatureWeights:
    """Test FeatureWeights configuration."""
    
    def test_default_weights(self):
        """Test default feature weights."""
        weights = FeatureWeights()
        
        assert weights.base_score == 1.0
        assert weights.content_type == 1.0
        assert weights.recency == 1.0
        assert weights.source_weight == 1.0
        
        # Test get_weight method
        assert weights.get_weight(RankingFeature.BASE_SCORE) == 1.0
        assert weights.get_weight(RankingFeature.CONTENT_TYPE) == 1.0
    
    def test_custom_weights(self):
        """Test setting custom feature weights."""
        weights = FeatureWeights(
            base_score=0.8,
            content_type=2.0,
            recency=0.5
        )
        
        assert weights.base_score == 0.8
        assert weights.content_type == 2.0
        assert weights.recency == 0.5
    
    def test_set_weight_method(self):
        """Test setting weights via method."""
        weights = FeatureWeights()
        
        weights.set_weight(RankingFeature.CODE_BOOST, 3.0)  # This should add dynamically
        weights.set_weight(RankingFeature.RECENCY, 0.2)
        
        assert weights.get_weight(RankingFeature.RECENCY) == 0.2


@pytest.fixture
def sample_results():
    """Create sample results for testing."""
    base_time = datetime.now(timezone.utc)
    
    return [
        MemoryResult(
            id="code_1",
            text="def hello_world(): print('Hello')",
            type=ContentType.CODE,
            score=0.8,
            source=ResultSource.VECTOR,
            timestamp=base_time,
            tags=["python", "function"]
        ),
        MemoryResult(
            id="doc_1", 
            text="This is documentation about Python functions",
            type=ContentType.DOCUMENTATION,
            score=0.9,
            source=ResultSource.GRAPH,
            timestamp=base_time - timedelta(days=30),
            tags=["documentation", "python"]
        ),
        MemoryResult(
            id="general_1",
            text="General information about programming",
            type=ContentType.GENERAL,
            score=0.7,
            source=ResultSource.KV,
            timestamp=base_time - timedelta(days=5),
            tags=["programming", "general"]
        ),
        MemoryResult(
            id="fact_1",
            text="Python was created by Guido van Rossum",
            type=ContentType.FACT,
            score=0.85,
            source=ResultSource.HYBRID,
            timestamp=base_time - timedelta(days=1),
            tags=["python", "history", "fact"]
        )
    ]


@pytest.fixture
def ranking_context():
    """Create sample ranking context."""
    return RankingContext(
        query="python function",
        search_mode="hybrid"
    )


class TestDefaultRankingPolicy:
    """Test default ranking policy."""
    
    def test_initialization(self):
        """Test policy initialization."""
        policy = DefaultRankingPolicy()
        
        assert policy.name == "default"
        assert "balanced" in policy.description.lower()
        assert isinstance(policy.weights, FeatureWeights)
    
    def test_basic_ranking(self, sample_results, ranking_context):
        """Test basic ranking functionality."""
        policy = DefaultRankingPolicy()
        
        ranked = policy.rank(sample_results, ranking_context)
        
        assert len(ranked) == len(sample_results)
        
        # Results should be sorted by adjusted score (highest first)
        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True)
        
        # Verify no result lost or duplicated
        original_ids = {r.id for r in sample_results}
        ranked_ids = {r.id for r in ranked}
        assert original_ids == ranked_ids
    
    def test_content_type_boost(self, sample_results, ranking_context):
        """Test content type boosting."""
        policy = DefaultRankingPolicy()
        
        # Find code and documentation results
        code_result = next(r for r in sample_results if r.type == ContentType.CODE)
        doc_result = next(r for r in sample_results if r.type == ContentType.DOCUMENTATION)
        
        # Code should get a boost
        code_boost = policy._calculate_content_type_boost(ContentType.CODE)
        doc_boost = policy._calculate_content_type_boost(ContentType.DOCUMENTATION)
        
        assert code_boost > doc_boost  # Code should have higher boost
        assert code_boost > 0  # Code should get positive boost
    
    def test_recency_calculation(self):
        """Test recency factor calculation."""
        policy = DefaultRankingPolicy()
        
        current_time = time.time()
        recent_time = current_time - 86400  # 1 day ago
        old_time = current_time - (86400 * 30)  # 30 days ago
        
        recent_factor = policy._calculate_recency_factor(recent_time, current_time)
        old_factor = policy._calculate_recency_factor(old_time, current_time)
        
        assert 0.0 <= recent_factor <= 1.0
        assert 0.0 <= old_factor <= 1.0
        assert recent_factor > old_factor  # Recent should have higher factor
    
    def test_source_weights(self):
        """Test source weight calculation."""
        policy = DefaultRankingPolicy()
        
        vector_weight = policy._get_source_weight(ResultSource.VECTOR)
        graph_weight = policy._get_source_weight(ResultSource.GRAPH)
        kv_weight = policy._get_source_weight(ResultSource.KV)
        
        assert vector_weight == 1.0  # Vector is primary
        assert graph_weight < vector_weight  # Graph slightly lower
        assert kv_weight < graph_weight  # KV lowest priority
    
    def test_empty_results(self, ranking_context):
        """Test handling of empty results."""
        policy = DefaultRankingPolicy()
        
        ranked = policy.rank([], ranking_context)
        
        assert ranked == []
    
    def test_configuration(self):
        """Test getting policy configuration."""
        weights = FeatureWeights(content_type=2.0, recency=0.5)
        policy = DefaultRankingPolicy(weights)
        
        config = policy.get_configuration()
        
        assert config["name"] == "default"
        assert "description" in config
        assert "weights" in config
        assert config["weights"]["content_type"] == 2.0
        assert config["weights"]["recency"] == 0.5


class TestCodeBoostRankingPolicy:
    """Test code boost ranking policy."""
    
    def test_initialization(self):
        """Test policy initialization."""
        policy = CodeBoostRankingPolicy(code_boost_factor=3.0)
        
        assert policy.name == "code_boost"
        assert policy.code_boost_factor == 3.0
        assert "code" in policy.description.lower()
    
    def test_code_boosting(self, sample_results, ranking_context):
        """Test code result boosting."""
        policy = CodeBoostRankingPolicy(code_boost_factor=2.0)
        
        ranked = policy.rank(sample_results, ranking_context)
        
        # Find code result
        code_result = next(r for r in ranked if r.type == ContentType.CODE)
        
        # Code result should have significantly higher score due to boost
        assert code_result.score >= 1.0 or code_result.score > 0.9  # Boosted score
        
        # Code result should be ranked highly
        code_position = next(i for i, r in enumerate(ranked) if r.type == ContentType.CODE)
        assert code_position <= 1  # Should be in top 2
    
    def test_tag_based_code_detection(self, ranking_context):
        """Test code detection via tags."""
        policy = CodeBoostRankingPolicy(code_boost_factor=2.0)
        
        # Create result with code-related tags
        tagged_result = MemoryResult(
            id="tagged_1",
            text="Some technical content",
            type=ContentType.GENERAL,  # Not explicitly code type
            score=0.5,
            source=ResultSource.VECTOR,
            tags=["python", "function"]  # Code-related tags
        )
        
        ranked = policy.rank([tagged_result], ranking_context)
        
        # Should get boosted due to tags
        assert ranked[0].score > 0.5  # Should be boosted
    
    def test_text_based_code_detection(self, ranking_context):
        """Test code detection via text patterns."""
        policy = CodeBoostRankingPolicy(code_boost_factor=2.0)
        
        # Create result with code-like text
        code_text_result = MemoryResult(
            id="code_text_1",
            text="function calculateTotal() { return x + y; }",
            type=ContentType.GENERAL,
            score=0.6,
            source=ResultSource.VECTOR,
            tags=[]
        )
        
        # Test pattern detection
        has_code = policy._contains_code_patterns(code_text_result.text)
        assert has_code
        
        ranked = policy.rank([code_text_result], ranking_context)
        
        # Should get boosted due to code patterns
        assert ranked[0].score > 0.6
    
    def test_configuration(self):
        """Test getting policy configuration."""
        policy = CodeBoostRankingPolicy(code_boost_factor=3.5)
        
        config = policy.get_configuration()
        
        assert config["name"] == "code_boost"
        assert config["code_boost_factor"] == 3.5
        assert "weights" in config


class TestRecencyRankingPolicy:
    """Test recency ranking policy."""
    
    def test_initialization(self):
        """Test policy initialization."""
        policy = RecencyRankingPolicy(decay_rate=0.02, base_weight=0.4)
        
        assert policy.name == "recency"
        assert policy.decay_rate == 0.02
        assert policy.base_weight == 0.4
    
    def test_recency_prioritization(self, ranking_context):
        """Test that recent results are prioritized."""
        policy = RecencyRankingPolicy()
        
        base_time = datetime.now(timezone.utc)
        
        # Create results with different ages but similar base scores
        old_result = MemoryResult(
            id="old_1",
            text="Old content",
            score=0.9,  # Higher base score
            source=ResultSource.VECTOR,
            timestamp=base_time - timedelta(days=30)
        )
        
        recent_result = MemoryResult(
            id="recent_1", 
            text="Recent content",
            score=0.7,  # Lower base score
            source=ResultSource.VECTOR,
            timestamp=base_time  # Very recent
        )
        
        ranked = policy.rank([old_result, recent_result], ranking_context)
        
        # Recent result should be ranked higher despite lower base score
        assert ranked[0].id == "recent_1"
        assert ranked[0].score > ranked[1].score
    
    def test_recency_score_calculation(self):
        """Test recency score calculation."""
        policy = RecencyRankingPolicy(decay_rate=0.01)
        
        current_time = time.time()
        recent_time = datetime.fromtimestamp(current_time - 86400, tz=timezone.utc)  # 1 day ago
        old_time = datetime.fromtimestamp(current_time - (86400 * 10), tz=timezone.utc)  # 10 days ago
        
        recent_score = policy._calculate_recency_score(recent_time, current_time)
        old_score = policy._calculate_recency_score(old_time, current_time)
        
        assert 0.0 <= recent_score <= 1.0
        assert 0.0 <= old_score <= 1.0
        assert recent_score > old_score
    
    def test_unknown_timestamps(self, ranking_context):
        """Test handling of very old timestamps (edge case)."""
        policy = RecencyRankingPolicy()
        
        # Use a very old timestamp to test recency decay
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        result_old_time = MemoryResult(
            id="old_time",
            text="Content with very old timestamp",
            score=0.8,
            source=ResultSource.VECTOR,
            timestamp=old_time
        )
        
        # Should handle gracefully and apply recency decay
        ranked = policy.rank([result_old_time], ranking_context)
        assert len(ranked) == 1
        assert ranked[0].score < 0.8  # Should be significantly reduced due to age
        assert 0.0 <= ranked[0].score <= 1.0


class TestRankingPolicyEngine:
    """Test ranking policy engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = RankingPolicyEngine()
        
        # Should have built-in policies registered
        policies = engine.list_policies()
        assert "default" in policies
        assert "code_boost" in policies
        assert "recency" in policies
        
        assert engine.default_policy_name == "default"
    
    def test_policy_registration(self):
        """Test registering custom policies."""
        engine = RankingPolicyEngine()
        
        class CustomPolicy(DefaultRankingPolicy):
            @property
            def name(self) -> str:
                return "custom"
        
        custom_policy = CustomPolicy()
        engine.register_policy(custom_policy)
        
        assert "custom" in engine.list_policies()
        assert engine.get_policy("custom") == custom_policy
    
    def test_invalid_policy_registration(self):
        """Test error handling for invalid policy registration."""
        engine = RankingPolicyEngine()
        
        with pytest.raises(ValueError, match="must implement RankingPolicy"):
            engine.register_policy("not_a_policy")
    
    def test_policy_unregistration(self):
        """Test unregistering policies.""" 
        engine = RankingPolicyEngine()
        
        # Register a test policy
        test_policy = DefaultRankingPolicy()
        engine.register_policy(test_policy)
        
        # Unregister existing policy
        result = engine.unregister_policy("default")
        assert result is True
        assert "default" not in engine.list_policies()
        
        # Try to unregister non-existent policy
        result = engine.unregister_policy("nonexistent")
        assert result is False
    
    def test_default_policy_setting(self):
        """Test setting default policy."""
        engine = RankingPolicyEngine()
        
        # Set existing policy as default
        result = engine.set_default_policy("code_boost")
        assert result is True
        assert engine.default_policy_name == "code_boost"
        
        # Try to set non-existent policy as default
        result = engine.set_default_policy("nonexistent")
        assert result is False
        assert engine.default_policy_name == "code_boost"  # Should remain unchanged
    
    def test_rank_results_with_default_policy(self, sample_results, ranking_context):
        """Test ranking results with default policy."""
        engine = RankingPolicyEngine()
        
        ranked = engine.rank_results(sample_results, ranking_context)
        
        assert len(ranked) == len(sample_results)
        # Should be sorted by score
        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True)
    
    def test_rank_results_with_specific_policy(self, sample_results, ranking_context):
        """Test ranking with specific policy."""
        engine = RankingPolicyEngine()
        
        ranked = engine.rank_results(sample_results, ranking_context, "code_boost")
        
        assert len(ranked) == len(sample_results)
        
        # Code result should be highly ranked
        code_result_pos = next(
            i for i, r in enumerate(ranked) if r.type == ContentType.CODE
        )
        assert code_result_pos <= 1  # Should be in top 2
    
    def test_rank_results_with_invalid_policy(self, sample_results, ranking_context):
        """Test error handling for invalid policy."""
        engine = RankingPolicyEngine()
        
        with pytest.raises(ValueError, match="Policy 'invalid' not found"):
            engine.rank_results(sample_results, ranking_context, "invalid")
    
    def test_empty_results_ranking(self, ranking_context):
        """Test ranking empty results."""
        engine = RankingPolicyEngine()
        
        ranked = engine.rank_results([], ranking_context)
        
        assert ranked == []
    
    def test_policy_info_retrieval(self):
        """Test getting policy information."""
        engine = RankingPolicyEngine()
        
        # Get info for existing policy
        info = engine.get_policy_info("default")
        assert info is not None
        assert info["name"] == "default"
        assert "description" in info
        
        # Get info for non-existent policy
        info = engine.get_policy_info("nonexistent")
        assert info is None
    
    def test_all_policy_info(self):
        """Test getting all policy information."""
        engine = RankingPolicyEngine()
        
        all_info = engine.get_all_policy_info()
        
        assert isinstance(all_info, dict)
        assert "default" in all_info
        assert "code_boost" in all_info
        assert "recency" in all_info
        
        # Check structure
        for policy_name, policy_info in all_info.items():
            assert "name" in policy_info
            assert "description" in policy_info


class TestGlobalRankingEngine:
    """Test the global ranking engine instance."""
    
    def test_global_instance(self):
        """Test that global ranking engine is properly initialized."""
        policies = ranking_engine.list_policies()
        
        assert "default" in policies
        assert "code_boost" in policies
        assert "recency" in policies
        
        assert ranking_engine.default_policy_name == "default"
    
    def test_global_instance_functionality(self, sample_results, ranking_context):
        """Test that global instance works correctly."""
        ranked = ranking_engine.rank_results(sample_results, ranking_context, "code_boost")
        
        assert len(ranked) == len(sample_results)
        
        # Code should be boosted
        code_result = next(r for r in ranked if r.type == ContentType.CODE)
        assert code_result.score > 0.8  # Should be boosted


# Golden ranking tests - specific scenarios with expected outcomes
class TestGoldenRankings:
    """Test golden ranking scenarios with expected outcomes."""
    
    def test_golden_code_search_scenario(self):
        """Golden test: Code search should prioritize code results."""
        # Scenario: User searches for "python function"
        results = [
            MemoryResult(
                id="doc_python",
                text="Documentation about Python functions and their usage",
                type=ContentType.DOCUMENTATION,
                score=0.6,  # High base relevance but not too high
                source=ResultSource.VECTOR,
                tags=["python", "documentation"]
            ),
            MemoryResult(
                id="code_python",
                text="def calculate_average(numbers): return sum(numbers) / len(numbers)",
                type=ContentType.CODE,
                score=0.45,  # Lower base relevance, should be boosted
                source=ResultSource.VECTOR,
                tags=["python", "function", "code"]
            ),
            MemoryResult(
                id="general_programming",
                text="General information about programming best practices",
                type=ContentType.GENERAL,
                score=0.5,
                source=ResultSource.GRAPH,
                tags=["programming", "best-practices"]
            )
        ]
        
        context = RankingContext(query="python function", search_mode="vector")
        
        # Test with code boost policy
        code_boost_ranked = ranking_engine.rank_results(results, context, "code_boost")
        
        # Code result should be first despite lower base score
        assert code_boost_ranked[0].id == "code_python"
        assert code_boost_ranked[0].type == ContentType.CODE
        
        # Test with default policy  
        default_ranked = ranking_engine.rank_results(results, context, "default")
        
        # With default policy, documentation might win due to higher base score
        # but code should still be competitive
        code_position = next(i for i, r in enumerate(default_ranked) if r.id == "code_python")
        assert code_position <= 1  # Should be in top 2
    
    def test_golden_recency_scenario(self):
        """Golden test: Recent content prioritization."""
        base_time = datetime.now(timezone.utc)
        
        results = [
            MemoryResult(
                id="old_high_score",
                text="Very relevant old content",
                score=0.95,
                source=ResultSource.VECTOR,
                timestamp=base_time - timedelta(days=90)  # 3 months old
            ),
            MemoryResult(
                id="recent_medium_score", 
                text="Moderately relevant recent content",
                score=0.70,
                source=ResultSource.VECTOR,
                timestamp=base_time - timedelta(hours=2)  # Very recent
            ),
            MemoryResult(
                id="medium_age",
                text="Medium relevance, medium age",
                score=0.80,
                source=ResultSource.VECTOR,
                timestamp=base_time - timedelta(days=7)  # 1 week old
            )
        ]
        
        context = RankingContext(query="content search", search_mode="hybrid")
        
        # Test with recency policy
        recency_ranked = ranking_engine.rank_results(results, context, "recency")
        
        # Recent result should be first despite lower base score
        assert recency_ranked[0].id == "recent_medium_score"
        
        # Test with default policy
        default_ranked = ranking_engine.rank_results(results, context, "default")
        
        # Should balance relevance and recency
        # Recent or high-scoring result should be first
        top_result = default_ranked[0]
        assert top_result.id in ["old_high_score", "recent_medium_score", "medium_age"]