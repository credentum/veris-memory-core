#!/usr/bin/env python3
"""
Ranking policy engine for customizable result scoring.

This module provides a pluggable ranking system that allows different scoring
strategies to be applied to search results based on context and requirements.
"""

import math
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from ..interfaces.memory_result import MemoryResult, ContentType, ResultSource
from ..utils.logging_middleware import ranking_logger


class RankingFeature(str, Enum):
    """Available features for ranking calculations."""
    BASE_SCORE = "base_score"  # Original backend relevance score
    CONTENT_TYPE = "content_type"  # Type-based scoring boost
    RECENCY = "recency"  # Time-based decay scoring
    SOURCE_WEIGHT = "source_weight"  # Backend source weighting
    TAG_RELEVANCE = "tag_relevance"  # Tag matching relevance
    TEXT_LENGTH = "text_length"  # Content length consideration
    NAMESPACE_BOOST = "namespace_boost"  # Namespace-specific boosting
    CODE_BOOST = "code_boost"  # Code content boosting


@dataclass
class RankingContext:
    """Context information for ranking calculations."""
    query: str
    search_mode: str
    user_preferences: Dict[str, Any] = None
    namespace_weights: Dict[str, float] = None
    custom_features: Dict[str, Any] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.namespace_weights is None:
            self.namespace_weights = {}
        if self.custom_features is None:
            self.custom_features = {}


@dataclass
class FeatureWeights:
    """Weights for different ranking features."""
    base_score: float = 1.0
    content_type: float = 1.0
    recency: float = 1.0
    source_weight: float = 1.0
    tag_relevance: float = 1.0
    text_length: float = 1.0
    namespace_boost: float = 1.0
    code_boost: float = 1.0
    
    def get_weight(self, feature: RankingFeature) -> float:
        """Get weight for a specific feature."""
        return getattr(self, feature.value, 1.0)
    
    def set_weight(self, feature: RankingFeature, weight: float) -> None:
        """Set weight for a specific feature."""
        setattr(self, feature.value, weight)


class RankingPolicy(ABC):
    """
    Abstract base class for ranking policies.
    
    Ranking policies define how search results are scored and ordered
    based on various features and contextual information.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this ranking policy."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of this ranking policy."""
        pass
    
    @abstractmethod
    def rank(
        self,
        results: List[MemoryResult],
        context: RankingContext
    ) -> List[MemoryResult]:
        """
        Apply ranking policy to results.
        
        Args:
            results: List of search results to rank
            context: Ranking context with query and preferences
            
        Returns:
            Ranked list of results (highest score first)
        """
        pass
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current policy configuration for debugging."""
        return {
            "name": self.name,
            "description": self.description
        }


class DefaultRankingPolicy(RankingPolicy):
    """
    Default ranking policy with balanced scoring.
    
    Uses original backend scores with light content type and recency adjustments.
    """
    
    def __init__(self, weights: Optional[FeatureWeights] = None):
        self.weights = weights or FeatureWeights()
    
    @property
    def name(self) -> str:
        return "default"
    
    @property
    def description(self) -> str:
        return "Balanced ranking with original scores, light type boosting, and recency decay"
    
    def rank(self, results: List[MemoryResult], context: RankingContext) -> List[MemoryResult]:
        """Apply default ranking algorithm."""
        if not results:
            return results
        
        ranked_results = []
        
        for result in results:
            # Start with base score
            score = result.score * self.weights.get_weight(RankingFeature.BASE_SCORE)
            
            # Content type boost
            type_boost = self._calculate_content_type_boost(result.type)
            score *= (1.0 + type_boost * self.weights.get_weight(RankingFeature.CONTENT_TYPE))
            
            # Recency decay
            recency_factor = self._calculate_recency_factor(result.timestamp, context.timestamp)
            score *= (recency_factor * self.weights.get_weight(RankingFeature.RECENCY) + 
                     (1.0 - self.weights.get_weight(RankingFeature.RECENCY)))
            
            # Source weight adjustment
            source_weight = self._get_source_weight(result.source)
            score *= source_weight * self.weights.get_weight(RankingFeature.SOURCE_WEIGHT)
            
            # Create new result with adjusted score
            new_result = result.model_copy()
            new_result.score = min(1.0, max(0.0, score))  # Clamp to valid range
            ranked_results.append(new_result)
        
        # Sort by final score
        return sorted(ranked_results, key=lambda x: x.score, reverse=True)
    
    def _calculate_content_type_boost(self, content_type: ContentType) -> float:
        """Calculate boost based on content type."""
        boosts = {
            ContentType.CODE: 0.1,       # Slightly favor code
            ContentType.FACT: 0.05,      # Facts are valuable
            ContentType.DOCUMENTATION: 0.02,  # Docs are helpful
            ContentType.GENERAL: 0.0,    # Neutral
            ContentType.CONVERSATION: -0.05  # De-prioritize chatter
        }
        return boosts.get(content_type, 0.0)
    
    def _calculate_recency_factor(self, result_time, current_time) -> float:
        """Calculate recency factor (1.0 = newest, decays over time)."""
        if not result_time or not current_time:
            return 1.0
        
        # Convert to timestamps if needed
        if hasattr(result_time, 'timestamp'):
            result_timestamp = result_time.timestamp()
        else:
            result_timestamp = float(result_time)
        
        age_seconds = current_time - result_timestamp
        age_days = age_seconds / 86400  # Convert to days
        
        # Exponential decay: 90% relevance after 30 days, 50% after 180 days
        decay_rate = 0.005  # Adjust for desired decay speed
        return math.exp(-decay_rate * age_days)
    
    def _get_source_weight(self, source: ResultSource) -> float:
        """Get weight multiplier based on result source."""
        weights = {
            ResultSource.VECTOR: 1.0,    # Semantic search is primary
            ResultSource.GRAPH: 0.95,    # Graph relationships slightly lower
            ResultSource.KV: 0.9,        # Direct lookups lower priority
            ResultSource.HYBRID: 1.0     # Hybrid results neutral
        }
        return weights.get(source, 1.0)
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            **super().get_configuration(),
            "weights": self.weights.__dict__
        }


class CodeBoostRankingPolicy(RankingPolicy):
    """
    Ranking policy that strongly favors code-related content.
    
    Applies significant boosts to code content while maintaining
    reasonable scoring for other content types.
    """
    
    def __init__(self, code_boost_factor: float = 2.0, weights: Optional[FeatureWeights] = None):
        self.code_boost_factor = code_boost_factor
        self.weights = weights or FeatureWeights()
        # Increase content type weight for code boosting
        self.weights.content_type = 2.0
    
    @property
    def name(self) -> str:
        return "code_boost"
    
    @property
    def description(self) -> str:
        return f"Strongly favors code content with {self.code_boost_factor}x boost multiplier"
    
    def rank(self, results: List[MemoryResult], context: RankingContext) -> List[MemoryResult]:
        """Apply code-boosting ranking algorithm."""
        if not results:
            return results
        
        ranked_results = []
        
        for result in results:
            # Start with base score
            score = result.score
            
            # Strong code boost
            if result.type == ContentType.CODE:
                score *= self.code_boost_factor
            
            # Tag-based code detection (fallback)
            elif any(tag.lower() in ['code', 'python', 'javascript', 'sql', 'function'] 
                    for tag in result.tags):
                score *= (self.code_boost_factor * 0.7)  # 70% of full boost
            
            # Text-based code detection (heuristic)
            elif self._contains_code_patterns(result.text):
                score *= (self.code_boost_factor * 0.5)  # 50% of full boost
            
            # Apply other factors with reduced weight
            recency_factor = self._calculate_recency_factor(result.timestamp, context.timestamp)
            score *= (0.9 + 0.1 * recency_factor)  # Light recency influence
            
            # Create new result with adjusted score
            new_result = result.model_copy()
            new_result.score = min(1.0, max(0.0, score))  # Clamp to valid range
            ranked_results.append(new_result)
        
        return sorted(ranked_results, key=lambda x: x.score, reverse=True)
    
    def _contains_code_patterns(self, text: str) -> bool:
        """Heuristically detect if text contains code."""
        code_indicators = [
            'def ', 'function ', 'class ', 'import ', 'from ',
            '{', '}', '()', '[]', 'return', 'if __name__',
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE'
        ]
        text_lower = text.lower()
        return any(indicator.lower() in text_lower for indicator in code_indicators)
    
    def _calculate_recency_factor(self, result_time, current_time) -> float:
        """Simplified recency calculation."""
        if not result_time or not current_time:
            return 1.0
        
        if hasattr(result_time, 'timestamp'):
            result_timestamp = result_time.timestamp()
        else:
            result_timestamp = float(result_time)
        
        age_seconds = current_time - result_timestamp
        age_days = age_seconds / 86400
        
        # Slower decay for code content
        return math.exp(-0.002 * age_days)
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            **super().get_configuration(),
            "code_boost_factor": self.code_boost_factor,
            "weights": self.weights.__dict__
        }


class RecencyRankingPolicy(RankingPolicy):
    """
    Ranking policy that prioritizes recent content.
    
    Applies strong time-based decay to favor newer results
    while maintaining some consideration for relevance scores.
    """
    
    def __init__(self, decay_rate: float = 0.01, base_weight: float = 0.3):
        self.decay_rate = decay_rate
        self.base_weight = base_weight  # Weight for original score vs recency
    
    @property
    def name(self) -> str:
        return "recency"
    
    @property
    def description(self) -> str:
        return f"Prioritizes recent content with {self.decay_rate} daily decay rate"
    
    def rank(self, results: List[MemoryResult], context: RankingContext) -> List[MemoryResult]:
        """Apply recency-focused ranking algorithm."""
        if not results:
            return results
        
        ranked_results = []
        
        for result in results:
            # Calculate recency score
            recency_score = self._calculate_recency_score(result.timestamp, context.timestamp)
            
            # Blend original score with recency
            blended_score = (self.base_weight * result.score + 
                           (1.0 - self.base_weight) * recency_score)
            
            # Create new result with adjusted score
            new_result = result.model_copy()
            new_result.score = min(1.0, max(0.0, blended_score))
            ranked_results.append(new_result)
        
        return sorted(ranked_results, key=lambda x: x.score, reverse=True)
    
    def _calculate_recency_score(self, result_time, current_time) -> float:
        """Calculate recency-based score."""
        if not result_time or not current_time:
            return 0.5  # Neutral score for unknown times
        
        if hasattr(result_time, 'timestamp'):
            result_timestamp = result_time.timestamp()
        else:
            result_timestamp = float(result_time)
        
        age_seconds = current_time - result_timestamp
        age_days = age_seconds / 86400
        
        # Exponential decay based on configuration
        return math.exp(-self.decay_rate * age_days)


class RankingPolicyEngine:
    """
    Central engine for managing and applying ranking policies.
    
    Provides policy registration, selection, and execution capabilities
    with performance monitoring and debugging support.
    """
    
    def __init__(self):
        self.policies: Dict[str, RankingPolicy] = {}
        self.default_policy_name = "default"
        
        # Register built-in policies
        self._register_builtin_policies()
    
    def _register_builtin_policies(self):
        """Register the built-in ranking policies."""
        self.register_policy(DefaultRankingPolicy())
        self.register_policy(CodeBoostRankingPolicy())
        self.register_policy(RecencyRankingPolicy())
    
    def register_policy(self, policy: RankingPolicy) -> None:
        """
        Register a ranking policy.
        
        Args:
            policy: RankingPolicy instance to register
        """
        if not isinstance(policy, RankingPolicy):
            raise ValueError("Policy must implement RankingPolicy interface")
        
        self.policies[policy.name] = policy
        ranking_logger.info(f"Registered ranking policy: {policy.name}")
    
    def unregister_policy(self, name: str) -> bool:
        """
        Unregister a ranking policy.
        
        Args:
            name: Name of policy to remove
            
        Returns:
            True if policy was removed, False if not found
        """
        if name in self.policies:
            if name == self.default_policy_name:
                ranking_logger.warning(f"Unregistering default policy: {name}")
            
            del self.policies[name]
            ranking_logger.info(f"Unregistered ranking policy: {name}")
            return True
        return False
    
    def list_policies(self) -> List[str]:
        """List all registered policy names."""
        return list(self.policies.keys())
    
    def get_policy(self, name: str) -> Optional[RankingPolicy]:
        """Get a policy by name."""
        return self.policies.get(name)
    
    def set_default_policy(self, name: str) -> bool:
        """
        Set the default policy.
        
        Args:
            name: Name of policy to set as default
            
        Returns:
            True if successful, False if policy doesn't exist
        """
        if name in self.policies:
            self.default_policy_name = name
            ranking_logger.info(f"Set default ranking policy: {name}")
            return True
        return False
    
    def rank_results(
        self,
        results: List[MemoryResult],
        context: RankingContext,
        policy_name: Optional[str] = None
    ) -> List[MemoryResult]:
        """
        Rank results using specified or default policy.
        
        Args:
            results: Results to rank
            context: Ranking context
            policy_name: Policy to use (defaults to default policy)
            
        Returns:
            Ranked results
            
        Raises:
            ValueError: If specified policy doesn't exist
        """
        if not results:
            return results
        
        # Select policy
        policy_name = policy_name or self.default_policy_name
        policy = self.policies.get(policy_name)
        
        if not policy:
            available_policies = ", ".join(self.list_policies())
            raise ValueError(f"Policy '{policy_name}' not found. Available: {available_policies}")
        
        # Apply ranking with performance tracking
        start_time = time.time()
        try:
            ranked_results = policy.rank(results, context)
            duration_ms = (time.time() - start_time) * 1000
            
            ranking_logger.info(
                f"Ranking completed",
                policy=policy_name,
                input_count=len(results),
                output_count=len(ranked_results),
                duration_ms=duration_ms,
                trace_id=getattr(context, 'trace_id', None)
            )
            
            return ranked_results
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            ranking_logger.error(
                f"Ranking failed",
                policy=policy_name,
                error=str(e),
                duration_ms=duration_ms,
                trace_id=getattr(context, 'trace_id', None)
            )
            raise
    
    def get_policy_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a policy."""
        policy = self.policies.get(name)
        if policy:
            return policy.get_configuration()
        return None
    
    def get_all_policy_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered policies."""
        return {name: policy.get_configuration() 
                for name, policy in self.policies.items()}


# Global policy engine instance
ranking_engine = RankingPolicyEngine()