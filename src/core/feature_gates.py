"""
Feature gates and rollout infrastructure for fact retrieval system.

This module provides sophisticated feature flagging, A/B testing, and gradual
rollout capabilities to safely deploy and monitor new fact recall features.
"""

import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class FeatureState(Enum):
    """Feature gate states."""
    DISABLED = "disabled"      # Feature completely off
    TESTING = "testing"        # Enabled for testing users only
    ROLLOUT = "rollout"        # Gradual rollout to percentage of users
    ENABLED = "enabled"        # Feature fully enabled
    DEPRECATED = "deprecated"  # Feature deprecated, will be removed


class RolloutStrategy(Enum):
    """Rollout strategies for feature deployment."""
    PERCENTAGE = "percentage"      # Simple percentage rollout
    USER_COHORT = "user_cohort"   # Based on user cohorts/segments
    GEOGRAPHIC = "geographic"     # Geographic rollout
    TIME_BASED = "time_based"     # Time-windowed rollout
    CUSTOM = "custom"             # Custom logic-based rollout


@dataclass
class FeatureConfig:
    """Configuration for a feature gate."""
    feature_name: str
    state: FeatureState
    rollout_percentage: float
    rollout_strategy: RolloutStrategy
    enabled_cohorts: Set[str]
    disabled_cohorts: Set[str]
    geographic_regions: Set[str]
    start_time: Optional[float]
    end_time: Optional[float]
    custom_logic: Optional[str]
    metadata: Dict[str, Any]
    created_at: float
    updated_at: float


@dataclass
class FeatureEvaluation:
    """Result of feature gate evaluation."""
    feature_name: str
    enabled: bool
    user_id: str
    reason: str
    evaluation_time: float
    metadata: Dict[str, Any]


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    test_name: str
    feature_name: str
    variants: Dict[str, float]  # variant_name -> traffic_percentage
    control_variant: str
    target_metric: str
    minimum_sample_size: int
    confidence_level: float
    start_time: float
    end_time: Optional[float]
    status: str
    metadata: Dict[str, Any]


class FeatureGateManager:
    """
    Comprehensive feature gate and rollout management system.
    
    Provides safe feature deployment with A/B testing, gradual rollout,
    cohort-based targeting, and detailed telemetry.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Feature configurations
        self.feature_configs: Dict[str, FeatureConfig] = {}
        self.ab_tests: Dict[str, ABTestConfig] = {}
        
        # Evaluation cache and telemetry
        self.evaluation_cache: Dict[str, FeatureEvaluation] = {}
        self.cache_ttl = self.config.get('cache_ttl_seconds', 300)  # 5 minutes
        self.evaluations_log: List[FeatureEvaluation] = []
        self.max_log_size = self.config.get('max_evaluations_log', 10000)
        
        # Thread safety
        self.config_lock = threading.RLock()
        self.evaluation_lock = threading.RLock()
        
        # Default cohorts for testing
        self.testing_cohorts = set(self.config.get('testing_cohorts', ['dev', 'qa', 'beta']))
        
        # Initialize default feature configurations
        self._initialize_default_features()
        
        logger.info("FeatureGateManager initialized with default features")
    
    def _initialize_default_features(self) -> None:
        """Initialize default feature gates for fact retrieval components."""
        default_features = {
            # Phase 3 features
            'graph_entity_extraction': {
                'state': FeatureState.ENABLED,
                'rollout_percentage': 100.0,
                'description': 'Entity extraction and graph linking'
            },
            'hybrid_scoring': {
                'state': FeatureState.ENABLED,
                'rollout_percentage': 100.0,
                'description': 'Multi-component hybrid scoring system'
            },
            'query_expansion': {
                'state': FeatureState.ENABLED,
                'rollout_percentage': 100.0,
                'description': 'Graph-enhanced query expansion'
            },
            'graph_fact_storage': {
                'state': FeatureState.ENABLED,
                'rollout_percentage': 100.0,
                'description': 'Dual Redis+Neo4j fact storage'
            },
            
            # Phase 4 features
            'fact_telemetry': {
                'state': FeatureState.ENABLED,
                'rollout_percentage': 100.0,
                'description': 'Comprehensive fact operation telemetry'
            },
            'privacy_controls': {
                'state': FeatureState.ENABLED,
                'rollout_percentage': 100.0,
                'description': 'Privacy-aware fact processing'
            },
            'advanced_ranking': {
                'state': FeatureState.ROLLOUT,
                'rollout_percentage': 50.0,
                'description': 'Advanced ML-based fact ranking'
            },
            'semantic_caching': {
                'state': FeatureState.TESTING,
                'rollout_percentage': 10.0,
                'description': 'Semantic similarity-based result caching'
            },
            'fact_verification': {
                'state': FeatureState.DISABLED,
                'rollout_percentage': 0.0,
                'description': 'Automated fact verification and confidence scoring'
            },

            # Semantic Search Improvement Features (S3 Paraphrase Robustness)
            # Enabled at 100% after successful PR #393 testing
            'semantic_cache_keys': {
                'state': FeatureState.ENABLED,
                'rollout_percentage': 100.0,
                'description': 'Embedding-based cache key generation for semantic consistency'
            },
            'multi_query_expansion': {
                'state': FeatureState.ENABLED,
                'rollout_percentage': 100.0,
                'description': 'Multi-query expansion (MQE) for paraphrase robustness'
            },
            'search_enhancements_in_retrieval': {
                'state': FeatureState.ENABLED,
                'rollout_percentage': 100.0,
                'description': 'Apply search enhancements in retrieval core'
            },
            'query_normalization': {
                'state': FeatureState.ENABLED,
                'rollout_percentage': 100.0,
                'description': 'Semantic query normalization for paraphrase consistency'
            },
            'hyde_query_expansion': {
                'state': FeatureState.TESTING,
                'rollout_percentage': 10.0,
                'description': 'HyDE (Hypothetical Document Embeddings) for improved paraphrase robustness'
            }
        }
        
        current_time = time.time()
        
        for feature_name, config in default_features.items():
            self.feature_configs[feature_name] = FeatureConfig(
                feature_name=feature_name,
                state=config['state'],
                rollout_percentage=config['rollout_percentage'],
                rollout_strategy=RolloutStrategy.PERCENTAGE,
                enabled_cohorts=set(),
                disabled_cohorts=set(),
                geographic_regions=set(),
                start_time=None,
                end_time=None,
                custom_logic=None,
                metadata={'description': config['description']},
                created_at=current_time,
                updated_at=current_time
            )
    
    def create_feature(self, feature_name: str, state: FeatureState = FeatureState.DISABLED,
                      rollout_percentage: float = 0.0,
                      rollout_strategy: RolloutStrategy = RolloutStrategy.PERCENTAGE,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Create a new feature gate."""
        if not feature_name:
            raise ValueError("feature_name is required")
        
        if rollout_percentage < 0.0 or rollout_percentage > 100.0:
            raise ValueError("rollout_percentage must be between 0.0 and 100.0")
        
        current_time = time.time()
        
        with self.config_lock:
            if feature_name in self.feature_configs:
                raise ValueError(f"Feature '{feature_name}' already exists")
            
            self.feature_configs[feature_name] = FeatureConfig(
                feature_name=feature_name,
                state=state,
                rollout_percentage=rollout_percentage,
                rollout_strategy=rollout_strategy,
                enabled_cohorts=set(),
                disabled_cohorts=set(),
                geographic_regions=set(),
                start_time=None,
                end_time=None,
                custom_logic=None,
                metadata=metadata or {},
                created_at=current_time,
                updated_at=current_time
            )
        
        logger.info(f"Created feature gate: {feature_name} ({state.value}, {rollout_percentage}%)")
    
    def update_feature(self, feature_name: str, **updates) -> None:
        """Update an existing feature gate configuration."""
        with self.config_lock:
            if feature_name not in self.feature_configs:
                raise ValueError(f"Feature '{feature_name}' does not exist")
            
            config = self.feature_configs[feature_name]
            
            # Update allowed fields
            if 'state' in updates:
                config.state = FeatureState(updates['state'])
            if 'rollout_percentage' in updates:
                percentage = updates['rollout_percentage']
                if percentage < 0.0 or percentage > 100.0:
                    raise ValueError("rollout_percentage must be between 0.0 and 100.0")
                config.rollout_percentage = percentage
            if 'rollout_strategy' in updates:
                config.rollout_strategy = RolloutStrategy(updates['rollout_strategy'])
            if 'enabled_cohorts' in updates:
                config.enabled_cohorts = set(updates['enabled_cohorts'])
            if 'disabled_cohorts' in updates:
                config.disabled_cohorts = set(updates['disabled_cohorts'])
            if 'geographic_regions' in updates:
                config.geographic_regions = set(updates['geographic_regions'])
            if 'start_time' in updates:
                config.start_time = updates['start_time']
            if 'end_time' in updates:
                config.end_time = updates['end_time']
            if 'custom_logic' in updates:
                config.custom_logic = updates['custom_logic']
            if 'metadata' in updates:
                config.metadata.update(updates['metadata'])
            
            config.updated_at = time.time()
        
        # Clear cache for this feature
        self._clear_feature_cache(feature_name)
        
        logger.info(f"Updated feature gate: {feature_name}")
    
    def is_enabled(self, feature_name: str, user_id: str,
                   user_cohort: Optional[str] = None,
                   geographic_region: Optional[str] = None,
                   custom_attributes: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a feature is enabled for a user."""
        evaluation = self.evaluate_feature(
            feature_name, user_id, user_cohort, geographic_region, custom_attributes
        )
        return evaluation.enabled
    
    def evaluate_feature(self, feature_name: str, user_id: str,
                        user_cohort: Optional[str] = None,
                        geographic_region: Optional[str] = None,
                        custom_attributes: Optional[Dict[str, Any]] = None) -> FeatureEvaluation:
        """Evaluate a feature gate with detailed reasoning."""
        current_time = time.time()
        
        # Check cache first
        cache_key = f"{feature_name}:{user_id}:{user_cohort}:{geographic_region}"
        with self.evaluation_lock:
            if cache_key in self.evaluation_cache:
                cached_eval = self.evaluation_cache[cache_key]
                if current_time - cached_eval.evaluation_time < self.cache_ttl:
                    return cached_eval
        
        # Get feature configuration
        with self.config_lock:
            if feature_name not in self.feature_configs:
                evaluation = FeatureEvaluation(
                    feature_name=feature_name,
                    enabled=False,
                    user_id=user_id,
                    reason="Feature not found",
                    evaluation_time=current_time,
                    metadata={}
                )
                self._cache_evaluation(cache_key, evaluation)
                return evaluation
            
            config = self.feature_configs[feature_name]
        
        # Evaluate feature state
        enabled, reason = self._evaluate_feature_logic(
            config, user_id, user_cohort, geographic_region, custom_attributes
        )
        
        evaluation = FeatureEvaluation(
            feature_name=feature_name,
            enabled=enabled,
            user_id=user_id,
            reason=reason,
            evaluation_time=current_time,
            metadata={
                'user_cohort': user_cohort,
                'geographic_region': geographic_region,
                'rollout_percentage': config.rollout_percentage,
                'state': config.state.value
            }
        )
        
        # Cache and log evaluation
        self._cache_evaluation(cache_key, evaluation)
        self._log_evaluation(evaluation)
        
        return evaluation
    
    def _evaluate_feature_logic(self, config: FeatureConfig, user_id: str,
                               user_cohort: Optional[str] = None,
                               geographic_region: Optional[str] = None,
                               custom_attributes: Optional[Dict[str, Any]] = None) -> tuple[bool, str]:
        """Core logic for feature evaluation."""
        try:
            current_time = time.time()
            
            # Check if feature is completely disabled
            if config.state == FeatureState.DISABLED:
                return False, "Feature disabled"
            
            # Check if feature is deprecated
            if config.state == FeatureState.DEPRECATED:
                return False, "Feature deprecated"
            
            # Check time windows
            try:
                if config.start_time and current_time < config.start_time:
                    return False, "Before start time"
                
                if config.end_time and current_time > config.end_time:
                    return False, "After end time"
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid time configuration for feature {config.feature_name}: {e}")
                # Continue with evaluation, ignoring time restrictions
            
            # Check disabled cohorts first (takes precedence)
            try:
                if user_cohort and user_cohort in config.disabled_cohorts:
                    return False, f"User cohort '{user_cohort}' explicitly disabled"
                
                # Check enabled cohorts
                if config.enabled_cohorts and user_cohort:
                    if user_cohort in config.enabled_cohorts:
                        return True, f"User cohort '{user_cohort}' explicitly enabled"
            except (TypeError, AttributeError) as e:
                logger.warning(f"Cohort evaluation error for feature {config.feature_name}: {e}")
                # Continue without cohort-based evaluation
            
            # Check testing cohorts for testing state
            try:
                if config.state == FeatureState.TESTING:
                    if user_cohort and user_cohort in self.testing_cohorts:
                        return True, f"Testing enabled for cohort '{user_cohort}'"
                    else:
                        return False, "Feature in testing, user not in testing cohort"
            except (TypeError, AttributeError) as e:
                logger.warning(f"Testing cohort evaluation error for feature {config.feature_name}: {e}")
                return False, "Feature in testing, cohort evaluation failed"
            
            # Check geographic restrictions
            try:
                if config.geographic_regions and geographic_region:
                    if geographic_region not in config.geographic_regions:
                        return False, f"Geographic region '{geographic_region}' not in allowed regions"
            except (TypeError, AttributeError) as e:
                logger.warning(f"Geographic evaluation error for feature {config.feature_name}: {e}")
                # Continue without geographic restrictions
            
            # For fully enabled features
            if config.state == FeatureState.ENABLED:
                return True, "Feature fully enabled"
            
            # For rollout features, use rollout strategy
            if config.state == FeatureState.ROLLOUT:
                try:
                    return self._evaluate_rollout_strategy(config, user_id, custom_attributes)
                except Exception as e:
                    logger.error(f"Rollout strategy evaluation failed for feature {config.feature_name}: {e}")
                    return False, f"Rollout evaluation failed: {str(e)}"
            
            return False, "Unknown feature state"
            
        except Exception as e:
            logger.error(f"Feature evaluation failed for {config.feature_name}: {e}")
            return False, f"Feature evaluation error: {str(e)}"
    
    def _evaluate_rollout_strategy(self, config: FeatureConfig, user_id: str,
                                  custom_attributes: Optional[Dict[str, Any]] = None) -> tuple[bool, str]:
        """Evaluate rollout strategy for percentage-based rollout."""
        if config.rollout_strategy == RolloutStrategy.PERCENTAGE:
            # Use consistent hashing for stable user assignment
            user_hash = hashlib.md5(f"{config.feature_name}:{user_id}".encode()).hexdigest()
            user_percentage = int(user_hash[:8], 16) % 10000 / 100.0  # 0.00 to 99.99
            
            if user_percentage < config.rollout_percentage:
                return True, f"User in rollout ({user_percentage:.2f}% < {config.rollout_percentage}%)"
            else:
                return False, f"User not in rollout ({user_percentage:.2f}% >= {config.rollout_percentage}%)"
        
        elif config.rollout_strategy == RolloutStrategy.CUSTOM:
            # Custom logic evaluation would go here
            if config.custom_logic:
                # In a real implementation, this could evaluate custom Python expressions
                # For now, just return based on percentage
                return self._evaluate_rollout_strategy(
                    FeatureConfig(**{**asdict(config), 'rollout_strategy': RolloutStrategy.PERCENTAGE}),
                    user_id,
                    custom_attributes
                )
        
        # Default to percentage strategy
        return self._evaluate_rollout_strategy(
            FeatureConfig(**{**asdict(config), 'rollout_strategy': RolloutStrategy.PERCENTAGE}),
            user_id,
            custom_attributes
        )
    
    def create_ab_test(self, test_name: str, feature_name: str,
                      variants: Dict[str, float], control_variant: str,
                      target_metric: str, minimum_sample_size: int = 1000,
                      confidence_level: float = 0.95,
                      duration_hours: Optional[float] = None) -> None:
        """Create an A/B test configuration."""
        if not test_name or not feature_name:
            raise ValueError("test_name and feature_name are required")
        
        if control_variant not in variants:
            raise ValueError("control_variant must be in variants")
        
        if abs(sum(variants.values()) - 100.0) > 0.01:
            raise ValueError("Variant percentages must sum to 100.0")
        
        current_time = time.time()
        end_time = current_time + (duration_hours * 3600) if duration_hours else None
        
        self.ab_tests[test_name] = ABTestConfig(
            test_name=test_name,
            feature_name=feature_name,
            variants=variants,
            control_variant=control_variant,
            target_metric=target_metric,
            minimum_sample_size=minimum_sample_size,
            confidence_level=confidence_level,
            start_time=current_time,
            end_time=end_time,
            status="running",
            metadata={}
        )
        
        logger.info(f"Created A/B test: {test_name} for feature {feature_name}")
    
    def get_ab_test_variant(self, test_name: str, user_id: str) -> Optional[str]:
        """Get the A/B test variant for a user."""
        if test_name not in self.ab_tests:
            return None
        
        test_config = self.ab_tests[test_name]
        
        # Check if test is still running
        current_time = time.time()
        if test_config.end_time and current_time > test_config.end_time:
            return None
        
        # Use consistent hashing to assign user to variant
        user_hash = hashlib.md5(f"{test_name}:{user_id}".encode()).hexdigest()
        user_percentage = int(user_hash[:8], 16) % 10000 / 100.0
        
        cumulative_percentage = 0.0
        for variant, percentage in test_config.variants.items():
            cumulative_percentage += percentage
            if user_percentage < cumulative_percentage:
                return variant
        
        return test_config.control_variant  # Fallback
    
    def get_feature_stats(self, feature_name: Optional[str] = None,
                         hours: int = 24) -> Dict[str, Any]:
        """Get feature gate evaluation statistics."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.evaluation_lock:
            recent_evaluations = [
                eval for eval in self.evaluations_log
                if eval.evaluation_time >= cutoff_time and
                (feature_name is None or eval.feature_name == feature_name)
            ]
        
        if not recent_evaluations:
            return {"message": "No evaluations in time period"}
        
        # Aggregate statistics
        total_evaluations = len(recent_evaluations)
        enabled_evaluations = sum(1 for eval in recent_evaluations if eval.enabled)
        
        # Group by feature
        by_feature = defaultdict(lambda: {"enabled": 0, "disabled": 0})
        for eval in recent_evaluations:
            if eval.enabled:
                by_feature[eval.feature_name]["enabled"] += 1
            else:
                by_feature[eval.feature_name]["disabled"] += 1
        
        # Group by reason
        by_reason = defaultdict(int)
        for eval in recent_evaluations:
            by_reason[eval.reason] += 1
        
        return {
            "time_range_hours": hours,
            "total_evaluations": total_evaluations,
            "enabled_evaluations": enabled_evaluations,
            "disabled_evaluations": total_evaluations - enabled_evaluations,
            "enable_rate": enabled_evaluations / total_evaluations if total_evaluations > 0 else 0,
            "by_feature": dict(by_feature),
            "by_reason": dict(by_reason),
            "active_features": len(self.feature_configs),
            "active_ab_tests": len([t for t in self.ab_tests.values() if t.status == "running"])
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export feature gate configuration for backup/transfer."""
        with self.config_lock:
            return {
                "feature_configs": {
                    name: asdict(config) for name, config in self.feature_configs.items()
                },
                "ab_tests": {
                    name: asdict(test) for name, test in self.ab_tests.items()
                },
                "metadata": {
                    "exported_at": time.time(),
                    "testing_cohorts": list(self.testing_cohorts)
                }
            }
    
    def import_configuration(self, config_data: Dict[str, Any]) -> None:
        """Import feature gate configuration from backup/transfer."""
        with self.config_lock:
            # Import feature configs
            if "feature_configs" in config_data:
                for name, config_dict in config_data["feature_configs"].items():
                    # Convert sets and enums
                    config_dict["state"] = FeatureState(config_dict["state"])
                    config_dict["rollout_strategy"] = RolloutStrategy(config_dict["rollout_strategy"])
                    config_dict["enabled_cohorts"] = set(config_dict["enabled_cohorts"])
                    config_dict["disabled_cohorts"] = set(config_dict["disabled_cohorts"])
                    config_dict["geographic_regions"] = set(config_dict["geographic_regions"])
                    
                    self.feature_configs[name] = FeatureConfig(**config_dict)
            
            # Import A/B tests
            if "ab_tests" in config_data:
                for name, test_dict in config_data["ab_tests"].items():
                    self.ab_tests[name] = ABTestConfig(**test_dict)
        
        # Clear all caches
        self._clear_all_caches()
        
        logger.info("Imported feature gate configuration")
    
    def _cache_evaluation(self, cache_key: str, evaluation: FeatureEvaluation) -> None:
        """Cache a feature evaluation."""
        with self.evaluation_lock:
            self.evaluation_cache[cache_key] = evaluation
    
    def _log_evaluation(self, evaluation: FeatureEvaluation) -> None:
        """Log a feature evaluation for analytics."""
        with self.evaluation_lock:
            self.evaluations_log.append(evaluation)
            
            # Maintain size limit
            if len(self.evaluations_log) > self.max_log_size:
                self.evaluations_log = self.evaluations_log[-self.max_log_size//2:]
    
    def _clear_feature_cache(self, feature_name: str) -> None:
        """Clear cache entries for a specific feature."""
        with self.evaluation_lock:
            keys_to_remove = [
                key for key in self.evaluation_cache.keys()
                if key.startswith(f"{feature_name}:")
            ]
            for key in keys_to_remove:
                del self.evaluation_cache[key]
    
    def _clear_all_caches(self) -> None:
        """Clear all cached evaluations."""
        with self.evaluation_lock:
            self.evaluation_cache.clear()


# Global feature gate manager instance
_feature_gate_manager: Optional[FeatureGateManager] = None


def get_feature_gate_manager() -> FeatureGateManager:
    """Get global feature gate manager instance."""
    global _feature_gate_manager
    if _feature_gate_manager is None:
        _feature_gate_manager = FeatureGateManager()
    return _feature_gate_manager


def initialize_feature_gates(config: Dict[str, Any]) -> FeatureGateManager:
    """Initialize global feature gate manager with configuration."""
    global _feature_gate_manager
    _feature_gate_manager = FeatureGateManager(config)
    return _feature_gate_manager


# Convenience functions for common operations
def is_feature_enabled(feature_name: str, user_id: str, **kwargs) -> bool:
    """Check if a feature is enabled for a user."""
    return get_feature_gate_manager().is_enabled(feature_name, user_id, **kwargs)


def get_feature_variant(ab_test_name: str, user_id: str) -> Optional[str]:
    """Get A/B test variant for a user."""
    return get_feature_gate_manager().get_ab_test_variant(ab_test_name, user_id)


# Context manager for feature gate evaluation
class feature_gate:
    """Context manager for feature gate-controlled code execution."""
    
    def __init__(self, feature_name: str, user_id: str, 
                 fallback_enabled: bool = False, **kwargs):
        self.feature_name = feature_name
        self.user_id = user_id
        self.fallback_enabled = fallback_enabled
        self.kwargs = kwargs
        self.enabled = None
    
    def __enter__(self):
        try:
            self.enabled = is_feature_enabled(self.feature_name, self.user_id, **self.kwargs)
        except Exception as e:
            logger.warning(f"Feature gate evaluation failed for {self.feature_name}: {e}")
            self.enabled = self.fallback_enabled
        
        return self.enabled
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # No cleanup needed