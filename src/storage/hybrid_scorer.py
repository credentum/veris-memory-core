"""
Hybrid scoring system combining vector, lexical, and graph signals.

This module implements the three-component scoring system for enhanced fact recall
with configurable weights and feature gates for graph signals.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import math

# Try absolute imports first, fall back to relative
try:
    from .fact_ranker import FactAwareRanker, RankingResult
    from .graph_enhancer import GraphSignalEnhancer, GraphSignal
    from ..core.intent_classifier import IntentType
except ImportError:
    try:
        from storage.fact_ranker import FactAwareRanker, RankingResult
        from storage.graph_enhancer import GraphSignalEnhancer, GraphSignal
        from core.intent_classifier import IntentType
    except ImportError:
        from fact_ranker import FactAwareRanker, RankingResult
        from graph_enhancer import GraphSignalEnhancer, GraphSignal
        from intent_classifier import IntentType

logger = logging.getLogger(__name__)


class ScoringMode(Enum):
    """Scoring modes for different query types."""
    FACT_OPTIMIZED = "fact_optimized"        # Heavy graph + fact patterns
    GENERAL_SEARCH = "general_search"        # Balanced vector + lexical
    SEMANTIC_HEAVY = "semantic_heavy"        # Vector-dominant scoring
    GRAPH_ENHANCED = "graph_enhanced"        # All three components


@dataclass
class ScoringWeights:
    """Weights for hybrid scoring components."""
    alpha_dense: float      # Vector/semantic similarity weight
    beta_lexical: float     # Lexical/keyword matching weight  
    gamma_graph: float      # Graph signal weight
    fact_boost: float       # Additional fact pattern boost
    
    def __post_init__(self) -> None:
        """Validate weights sum to reasonable range."""
        total = self.alpha_dense + self.beta_lexical + self.gamma_graph
        if not (0.8 <= total <= 1.2):
            logger.warning(f"Scoring weights sum to {total:.3f}, may cause score drift")


@dataclass
class HybridScore:
    """Result of hybrid scoring with component breakdown."""
    final_score: float
    vector_score: float
    lexical_score: float
    graph_score: float
    fact_pattern_score: float
    combined_score: float
    explanation: str
    metadata: Dict[str, Any]


class HybridScorer:
    """
    Combines vector, lexical, and graph signals for enhanced fact recall.
    
    Implements configurable scoring modes with feature gates for gradual
    rollout of graph signals and fact-aware ranking patterns.
    """
    
    def __init__(self, fact_ranker: Optional[FactAwareRanker] = None,
                 graph_enhancer: Optional[GraphSignalEnhancer] = None,
                 config: Optional[Dict[str, Any]] = None):
        
        self.fact_ranker = fact_ranker or FactAwareRanker()
        self.graph_enhancer = graph_enhancer
        self.config = config or {}
        
        # Feature gates
        self.enable_graph_signals = self.config.get('enable_graph_signals', True)
        self.enable_fact_patterns = self.config.get('enable_fact_patterns', True)
        self.enable_score_explanation = self.config.get('enable_score_explanation', True)
        
        # Scoring mode configurations
        self.scoring_modes = {
            ScoringMode.FACT_OPTIMIZED: ScoringWeights(
                alpha_dense=0.4,    # Reduced vector weight
                beta_lexical=0.3,   # Moderate lexical
                gamma_graph=0.3,    # High graph weight
                fact_boost=0.15     # Strong fact boost
            ),
            ScoringMode.GENERAL_SEARCH: ScoringWeights(
                alpha_dense=0.6,    # High vector weight
                beta_lexical=0.4,   # Moderate lexical
                gamma_graph=0.0,    # No graph for general queries
                fact_boost=0.05     # Minimal fact boost
            ),
            ScoringMode.SEMANTIC_HEAVY: ScoringWeights(
                alpha_dense=0.8,    # Dominant vector weight
                beta_lexical=0.2,   # Low lexical
                gamma_graph=0.0,    # No graph
                fact_boost=0.0      # No fact boost
            ),
            ScoringMode.GRAPH_ENHANCED: ScoringWeights(
                alpha_dense=0.5,    # Balanced vector
                beta_lexical=0.3,   # Moderate lexical
                gamma_graph=0.2,    # Moderate graph
                fact_boost=0.1      # Moderate fact boost
            )
        }
        
        # Score normalization parameters
        self.score_ceiling = 1.0
        self.score_floor = 0.0
        
    def compute_hybrid_score(self, query: str, results: List[Dict[str, Any]], 
                           intent: Optional[IntentType] = None,
                           scoring_mode: Optional[ScoringMode] = None) -> List[HybridScore]:
        """
        Compute hybrid scores for search results.
        
        Args:
            query: User query text
            results: Search results with vector scores
            intent: Detected query intent (for mode selection)
            scoring_mode: Override scoring mode
            
        Returns:
            List of HybridScore objects with component breakdowns
        """
        # Determine scoring mode
        if scoring_mode is None:
            scoring_mode = self._select_scoring_mode(query, intent)
        
        weights = self.scoring_modes[scoring_mode]
        
        # Extract vector scores from results
        vector_scores = {}
        lexical_scores = {}
        for i, result in enumerate(results):
            result_id = result.get('id', str(i))
            vector_scores[result_id] = result.get('score', 0.0)
            lexical_scores[result_id] = result.get('lexical_score', 0.0)
        
        # Compute fact pattern scores
        fact_scores = {}
        if self.enable_fact_patterns and weights.fact_boost > 0:
            fact_ranking_results = self.fact_ranker.apply_fact_ranking(results, query)
            for ranking_result in fact_ranking_results:
                result_id = self._get_result_id(ranking_result, results)
                if result_id:
                    fact_scores[result_id] = ranking_result.fact_boost
        
        # Compute graph scores
        graph_scores = {}
        if (self.enable_graph_signals and self.graph_enhancer and 
            weights.gamma_graph > 0 and scoring_mode in [ScoringMode.FACT_OPTIMIZED, ScoringMode.GRAPH_ENHANCED]):
            
            graph_signals = self.graph_enhancer.compute_graph_score(query, results)
            for result_id, signal in graph_signals.items():
                graph_scores[result_id] = signal.total_score
        
        # Compute hybrid scores
        hybrid_scores = []
        for i, result in enumerate(results):
            result_id = result.get('id', str(i))
            
            # Get component scores
            vector_score = vector_scores.get(result_id, 0.0)
            lexical_score = lexical_scores.get(result_id, 0.0)
            graph_score = graph_scores.get(result_id, 0.0)
            fact_pattern_score = fact_scores.get(result_id, 0.0)
            
            # Compute weighted combination
            combined_score = (
                weights.alpha_dense * vector_score +
                weights.beta_lexical * lexical_score +
                weights.gamma_graph * graph_score
            )
            
            # Apply fact pattern boost
            final_score = combined_score + (weights.fact_boost * fact_pattern_score)
            
            # Normalize to [0, 1] range
            final_score = max(self.score_floor, min(self.score_ceiling, final_score))
            
            # Generate explanation
            explanation = self._generate_score_explanation(
                vector_score, lexical_score, graph_score, fact_pattern_score,
                weights, scoring_mode
            ) if self.enable_score_explanation else ""
            
            hybrid_score = HybridScore(
                final_score=final_score,
                vector_score=vector_score,
                lexical_score=lexical_score,
                graph_score=graph_score,
                fact_pattern_score=fact_pattern_score,
                combined_score=combined_score,
                explanation=explanation,
                metadata={
                    'scoring_mode': scoring_mode.value,
                    'weights': {
                        'alpha_dense': weights.alpha_dense,
                        'beta_lexical': weights.beta_lexical,
                        'gamma_graph': weights.gamma_graph,
                        'fact_boost': weights.fact_boost
                    },
                    'result_id': result_id
                }
            )
            
            hybrid_scores.append(hybrid_score)
        
        # Sort by final score (descending)
        hybrid_scores.sort(key=lambda x: x.final_score, reverse=True)
        
        logger.debug(f"Computed hybrid scores for {len(results)} results using {scoring_mode.value} mode")
        return hybrid_scores
    
    def _select_scoring_mode(self, query: str, intent: Optional[IntentType]) -> ScoringMode:
        """Select appropriate scoring mode based on query characteristics."""
        # Intent-based mode selection
        if intent and intent in [IntentType.FACT_LOOKUP, IntentType.STORE_FACT, IntentType.UPDATE_FACT]:
            return ScoringMode.FACT_OPTIMIZED
        
        # Query pattern analysis
        query_lower = query.lower()
        
        # Fact-like questions
        fact_indicators = [
            'what is my', 'what\'s my', 'who am i', 'where do i',
            'how old am i', 'my name', 'my email', 'my phone'
        ]
        
        if any(indicator in query_lower for indicator in fact_indicators):
            return ScoringMode.FACT_OPTIMIZED
        
        # Entity-rich queries (potential for graph signals)
        entity_indicators = ['work at', 'live in', 'friend', 'colleague', 'team']
        if any(indicator in query_lower for indicator in entity_indicators):
            return ScoringMode.GRAPH_ENHANCED
        
        # Semantic search indicators
        semantic_indicators = ['similar to', 'like', 'related', 'about', 'regarding']
        if any(indicator in query_lower for indicator in semantic_indicators):
            return ScoringMode.SEMANTIC_HEAVY
        
        # Default to general search
        return ScoringMode.GENERAL_SEARCH
    
    def _get_result_id(self, ranking_result: RankingResult, original_results: List[Dict[str, Any]]) -> Optional[str]:
        """Get result ID by matching content."""
        for i, result in enumerate(original_results):
            if result.get('content') == ranking_result.content:
                return result.get('id', str(i))
        return None
    
    def _generate_score_explanation(self, vector_score: float, lexical_score: float,
                                  graph_score: float, fact_pattern_score: float,
                                  weights: ScoringWeights, mode: ScoringMode) -> str:
        """Generate human-readable explanation of scoring."""
        components = []
        
        if weights.alpha_dense > 0:
            components.append(f"Vector: {vector_score:.3f}×{weights.alpha_dense:.2f}")
        
        if weights.beta_lexical > 0:
            components.append(f"Lexical: {lexical_score:.3f}×{weights.beta_lexical:.2f}")
        
        if weights.gamma_graph > 0 and graph_score > 0:
            components.append(f"Graph: {graph_score:.3f}×{weights.gamma_graph:.2f}")
        
        if weights.fact_boost > 0 and fact_pattern_score != 0:
            boost_sign = "+" if fact_pattern_score > 0 else ""
            components.append(f"FactBoost: {boost_sign}{fact_pattern_score:.3f}×{weights.fact_boost:.2f}")
        
        explanation = f"{mode.value.replace('_', ' ').title()}: {' + '.join(components)}"
        return explanation
    
    def explain_scoring_decision(self, query: str, intent: Optional[IntentType] = None) -> Dict[str, Any]:
        """Provide detailed explanation of scoring mode selection."""
        selected_mode = self._select_scoring_mode(query, intent)
        weights = self.scoring_modes[selected_mode]
        
        return {
            'query': query,
            'detected_intent': intent.value if intent else None,
            'selected_mode': selected_mode.value,
            'mode_rationale': self._get_mode_rationale(query, intent, selected_mode),
            'scoring_weights': {
                'alpha_dense': weights.alpha_dense,
                'beta_lexical': weights.beta_lexical,
                'gamma_graph': weights.gamma_graph,
                'fact_boost': weights.fact_boost
            },
            'feature_gates': {
                'graph_signals_enabled': self.enable_graph_signals,
                'fact_patterns_enabled': self.enable_fact_patterns,
                'score_explanation_enabled': self.enable_score_explanation
            },
            'available_modes': [mode.value for mode in ScoringMode]
        }
    
    def _get_mode_rationale(self, query: str, intent: Optional[IntentType], mode: ScoringMode) -> str:
        """Get human-readable rationale for mode selection."""
        if intent and intent in [IntentType.FACT_LOOKUP, IntentType.STORE_FACT, IntentType.UPDATE_FACT]:
            return f"Intent-based selection: {intent.value} intent detected"
        
        query_lower = query.lower()
        
        if mode == ScoringMode.FACT_OPTIMIZED:
            return "Fact-like question patterns detected (what's my, who am i, etc.)"
        elif mode == ScoringMode.GRAPH_ENHANCED:
            return "Entity-rich query detected (work at, live in, relationships)"
        elif mode == ScoringMode.SEMANTIC_HEAVY:
            return "Semantic search indicators detected (similar to, like, related)"
        else:
            return "Default general search mode"
    
    def update_scoring_weights(self, mode: ScoringMode, weights: ScoringWeights) -> None:
        """Update scoring weights for a specific mode."""
        self.scoring_modes[mode] = weights
        logger.info(f"Updated scoring weights for {mode.value}: {weights}")
    
    def get_scoring_statistics(self) -> Dict[str, Any]:
        """Get statistics about scoring configuration and usage."""
        return {
            'scoring_modes': {mode.value: {
                'alpha_dense': weights.alpha_dense,
                'beta_lexical': weights.beta_lexical,
                'gamma_graph': weights.gamma_graph,
                'fact_boost': weights.fact_boost
            } for mode, weights in self.scoring_modes.items()},
            'feature_gates': {
                'graph_signals_enabled': self.enable_graph_signals,
                'fact_patterns_enabled': self.enable_fact_patterns,
                'score_explanation_enabled': self.enable_score_explanation
            },
            'components_available': {
                'fact_ranker': self.fact_ranker is not None,
                'graph_enhancer': self.graph_enhancer is not None
            },
            'score_bounds': {
                'ceiling': self.score_ceiling,
                'floor': self.score_floor
            }
        }
    
    def benchmark_scoring_modes(self, test_queries: List[str], 
                              test_results: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Benchmark different scoring modes on test data."""
        if len(test_queries) != len(test_results):
            raise ValueError("Number of queries must match number of result sets")
        
        benchmark_results = {}
        
        for mode in ScoringMode:
            mode_scores = []
            
            for query, results in zip(test_queries, test_results):
                hybrid_scores = self.compute_hybrid_score(query, results, scoring_mode=mode)
                if hybrid_scores:
                    mode_scores.append({
                        'query': query,
                        'top_score': hybrid_scores[0].final_score,
                        'score_range': hybrid_scores[0].final_score - hybrid_scores[-1].final_score,
                        'component_breakdown': {
                            'vector_weight': hybrid_scores[0].vector_score,
                            'lexical_weight': hybrid_scores[0].lexical_score,
                            'graph_weight': hybrid_scores[0].graph_score,
                            'fact_pattern_weight': hybrid_scores[0].fact_pattern_score
                        }
                    })
            
            if mode_scores:
                avg_top_score = sum(s['top_score'] for s in mode_scores) / len(mode_scores)
                avg_range = sum(s['score_range'] for s in mode_scores) / len(mode_scores)
                
                benchmark_results[mode.value] = {
                    'average_top_score': avg_top_score,
                    'average_score_range': avg_range,
                    'num_queries_tested': len(mode_scores),
                    'sample_breakdowns': mode_scores[:3]  # Sample of first 3
                }
        
        logger.info(f"Benchmarked {len(ScoringMode)} scoring modes on {len(test_queries)} queries")
        return benchmark_results
    
    def apply_score_tuning(self, target_metrics: Dict[str, float], 
                          tuning_data: List[Tuple[str, List[Dict[str, Any]], int]]) -> Dict[str, Any]:
        """
        Apply automated score tuning based on target metrics.
        
        Args:
            target_metrics: Target P@1, MRR, etc.
            tuning_data: List of (query, results, relevant_index) tuples
            
        Returns:
            Tuning results and updated weights
        """
        # Placeholder for future ML-based weight optimization
        # For now, return current configuration
        current_stats = self.get_scoring_statistics()
        
        tuning_results = {
            'status': 'not_implemented',
            'message': 'Automated tuning not yet implemented',
            'current_configuration': current_stats,
            'tuning_data_size': len(tuning_data),
            'target_metrics': target_metrics
        }
        
        logger.info("Score tuning requested but not yet implemented")
        return tuning_results