"""
Graph-enhanced query understanding and expansion.

This module provides query expansion capabilities using graph signals and
entity relationships to improve context retrieval for fact-rich queries.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import re

# Try absolute imports first, fall back to relative
try:
    from .intent_classifier import IntentClassifier, IntentType
    from .query_rewriter import FactQueryRewriter
    from ..storage.graph_enhancer import EntityExtractor, GraphSignalEnhancer
    from ..storage.neo4j_client import Neo4jInitializer
except ImportError:
    try:
        from intent_classifier import IntentClassifier, IntentType
        from query_rewriter import FactQueryRewriter
        from storage.graph_enhancer import EntityExtractor, GraphSignalEnhancer
        from storage.neo4j_client import Neo4jInitializer
    except ImportError:
        # For test scripts - import from full paths
        from core.intent_classifier import IntentClassifier, IntentType
        from core.query_rewriter import FactQueryRewriter
        from storage.graph_enhancer import EntityExtractor, GraphSignalEnhancer
        from storage.neo4j_client import Neo4jInitializer

logger = logging.getLogger(__name__)


class ExpansionStrategy(Enum):
    """Query expansion strategies."""
    ENTITY_RELATIONSHIPS = "entity_relationships"    # Expand based on graph relationships
    FACT_TEMPLATES = "fact_templates"               # Expand using fact templates
    SEMANTIC_NEIGHBORS = "semantic_neighbors"       # Expand using semantic similarity
    HYBRID_EXPANSION = "hybrid_expansion"           # Combine multiple strategies


@dataclass
class ExpandedQuery:
    """An expanded query with metadata."""
    query: str
    expansion_type: ExpansionStrategy
    confidence: float
    source_entities: List[str]
    added_concepts: List[str]
    reasoning: str


@dataclass
class QueryExpansionResult:
    """Result of query expansion."""
    original_query: str
    expanded_queries: List[ExpandedQuery]
    detected_intent: IntentType
    extracted_entities: List[str]
    expansion_metadata: Dict[str, Any]


class GraphQueryExpander:
    """
    Graph-enhanced query understanding and expansion.
    
    Uses entity extraction, graph traversal, and semantic relationships
    to expand queries for improved context retrieval.
    """
    
    def __init__(self, neo4j_client: Optional[Neo4jInitializer] = None,
                 intent_classifier: Optional[IntentClassifier] = None,
                 query_rewriter: Optional[FactQueryRewriter] = None,
                 config: Optional[Dict[str, Any]] = None):
        
        self.neo4j_client = neo4j_client
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.query_rewriter = query_rewriter or FactQueryRewriter()
        self.entity_extractor = EntityExtractor()
        self.config = config or {}
        
        # Expansion configuration
        self.max_expansions = self.config.get('max_expansions', 5)
        self.entity_hop_limit = self.config.get('entity_hop_limit', 2)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        
        # Relationship types for expansion
        self.expansion_relationships = {
            'PERSON_WORKS_AT': ['workplace', 'colleagues', 'team'],
            'PERSON_LIVES_IN': ['location', 'neighborhood', 'area'],
            'PERSON_LIKES': ['preferences', 'interests', 'hobbies'],
            'PERSON_HAS_CONTACT': ['communication', 'contact info'],
            'PERSON_HAS_ATTRIBUTE': ['characteristics', 'properties'],
            'HAS_FACT_RELATION': ['related facts', 'connected information']
        }
    
    def expand_query(self, query: str, strategy: Optional[ExpansionStrategy] = None) -> QueryExpansionResult:
        """
        Expand query using graph-enhanced understanding.
        
        Args:
            query: Original query text
            strategy: Expansion strategy to use (auto-selected if None)
            
        Returns:
            QueryExpansionResult with expanded queries and metadata
        """
        # Classify intent and extract entities
        intent_result = self.intent_classifier.classify(query)
        entities = self.entity_extractor.extract_entities(query)
        
        # Select expansion strategy if not provided
        if strategy is None:
            strategy = self._select_expansion_strategy(query, intent_result.intent, entities)
        
        # Generate expansions based on strategy
        expanded_queries = []
        
        if strategy == ExpansionStrategy.ENTITY_RELATIONSHIPS:
            expanded_queries.extend(self._expand_by_entity_relationships(query, entities))
        
        elif strategy == ExpansionStrategy.FACT_TEMPLATES:
            expanded_queries.extend(self._expand_by_fact_templates(query, intent_result))
        
        elif strategy == ExpansionStrategy.SEMANTIC_NEIGHBORS:
            expanded_queries.extend(self._expand_by_semantic_neighbors(query, entities))
        
        elif strategy == ExpansionStrategy.HYBRID_EXPANSION:
            # Combine multiple strategies
            expanded_queries.extend(self._expand_by_entity_relationships(query, entities))
            expanded_queries.extend(self._expand_by_fact_templates(query, intent_result))
            expanded_queries.extend(self._expand_by_semantic_neighbors(query, entities))
        
        # Filter and rank expansions
        filtered_expansions = self._filter_and_rank_expansions(expanded_queries, query)
        
        # Prepare metadata
        metadata = {
            'strategy_used': strategy.value,
            'entities_found': len(entities),
            'total_expansions_generated': len(expanded_queries),
            'filtered_expansions': len(filtered_expansions),
            'neo4j_available': self.neo4j_client is not None
        }
        
        return QueryExpansionResult(
            original_query=query,
            expanded_queries=filtered_expansions,
            detected_intent=intent_result.intent,
            extracted_entities=[e.normalized_form for e in entities],
            expansion_metadata=metadata
        )
    
    def _select_expansion_strategy(self, query: str, intent: IntentType, entities: List) -> ExpansionStrategy:
        """Select appropriate expansion strategy based on query characteristics."""
        query_lower = query.lower()
        
        # Fact-specific queries benefit from templates
        if intent in [IntentType.FACT_LOOKUP, IntentType.STORE_FACT, IntentType.UPDATE_FACT]:
            return ExpansionStrategy.FACT_TEMPLATES
        
        # Entity-rich queries benefit from relationship expansion
        if entities and len(entities) >= 2:
            return ExpansionStrategy.ENTITY_RELATIONSHIPS
        
        # Relationship indicators suggest entity expansion
        relationship_indicators = ['work with', 'live near', 'know about', 'related to', 'connected']
        if any(indicator in query_lower for indicator in relationship_indicators):
            return ExpansionStrategy.ENTITY_RELATIONSHIPS
        
        # Complex queries benefit from hybrid approach
        if len(query.split()) > 6:
            return ExpansionStrategy.HYBRID_EXPANSION
        
        # Default to semantic neighbors for general queries
        return ExpansionStrategy.SEMANTIC_NEIGHBORS
    
    def _expand_by_entity_relationships(self, query: str, entities: List) -> List[ExpandedQuery]:
        """Expand query based on entity relationships in the graph."""
        expansions = []
        
        if not self.neo4j_client or not entities:
            return expansions
        
        try:
            for entity in entities[:2]:  # Limit to first 2 entities
                # Get entity relationships from graph
                cypher = """
                MATCH (e {normalized_name: $entity_name})-[r]->(connected)
                WHERE any(label IN labels(e) WHERE label IN ['Person', 'Entity'])
                RETURN type(r) as rel_type, 
                       connected.normalized_name as connected_entity,
                       connected.entity_type as connected_type,
                       r.confidence as confidence
                ORDER BY r.confidence DESC
                LIMIT 10
                """
                
                results = self.neo4j_client.query(cypher, {'entity_name': entity.normalized_form})
                
                for record in results:
                    rel_type = record.get('rel_type', '')
                    connected_entity = record.get('connected_entity', '')
                    connected_type = record.get('connected_type', '')
                    confidence = record.get('confidence', 0.5)
                    
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Generate expansion based on relationship
                    expansion_concepts = self.expansion_relationships.get(rel_type, [rel_type.lower()])
                    
                    for concept in expansion_concepts[:2]:  # Limit concepts per relationship
                        # Create expanded query
                        expanded_query = f"{query} {concept} {connected_entity}"
                        
                        expansion = ExpandedQuery(
                            query=expanded_query,
                            expansion_type=ExpansionStrategy.ENTITY_RELATIONSHIPS,
                            confidence=confidence * 0.8,  # Slightly reduced confidence
                            source_entities=[entity.normalized_form],
                            added_concepts=[concept, connected_entity],
                            reasoning=f"Added {concept} concept through {rel_type} relationship to {connected_entity}"
                        )
                        
                        expansions.append(expansion)
                        
                        if len(expansions) >= self.max_expansions:
                            break
                    
                    if len(expansions) >= self.max_expansions:
                        break
                
                if len(expansions) >= self.max_expansions:
                    break
        
        except Exception as e:
            logger.warning(f"Entity relationship expansion failed: {e}")
        
        logger.debug(f"Generated {len(expansions)} entity relationship expansions")
        return expansions
    
    def _expand_by_fact_templates(self, query: str, intent_result) -> List[ExpandedQuery]:
        """Expand query using fact templates and rewriting."""
        expansions = []
        
        try:
            # Use query rewriter to generate fact-based expansions
            rewritten_queries = self.query_rewriter.rewrite_fact_query(query, intent_result.attribute)
            
            for rewrite in rewritten_queries:
                # Convert rewritten query to expansion
                expansion = ExpandedQuery(
                    query=rewrite.query,
                    expansion_type=ExpansionStrategy.FACT_TEMPLATES,
                    confidence=rewrite.confidence,
                    source_entities=[],
                    added_concepts=[rewrite.method.value],
                    reasoning=f"Template-based expansion using {rewrite.method.value}"
                )
                
                expansions.append(expansion)
                
                if len(expansions) >= self.max_expansions:
                    break
        
        except Exception as e:
            logger.warning(f"Fact template expansion failed: {e}")
        
        logger.debug(f"Generated {len(expansions)} fact template expansions")
        return expansions
    
    def _expand_by_semantic_neighbors(self, query: str, entities: List) -> List[ExpandedQuery]:
        """Expand query using semantic neighbors and synonyms."""
        expansions = []
        
        try:
            # Simple semantic expansion using word synonyms and related terms
            semantic_mappings = {
                'email': ['contact', 'address', 'communication'],
                'phone': ['number', 'contact', 'call'],
                'name': ['identity', 'called', 'known as'],
                'location': ['place', 'address', 'area'],
                'work': ['job', 'employment', 'profession'],
                'like': ['prefer', 'enjoy', 'love'],
                'favorite': ['preferred', 'best', 'top choice']
            }
            
            query_words = query.lower().split()
            
            for word in query_words:
                if word in semantic_mappings:
                    synonyms = semantic_mappings[word]
                    
                    for synonym in synonyms[:2]:  # Limit synonyms per word
                        # Replace word with synonym
                        expanded_words = [synonym if w == word else w for w in query_words]
                        expanded_query = ' '.join(expanded_words)
                        
                        expansion = ExpandedQuery(
                            query=expanded_query,
                            expansion_type=ExpansionStrategy.SEMANTIC_NEIGHBORS,
                            confidence=0.7,
                            source_entities=[],
                            added_concepts=[synonym],
                            reasoning=f"Semantic expansion: '{word}' -> '{synonym}'"
                        )
                        
                        expansions.append(expansion)
                        
                        if len(expansions) >= self.max_expansions:
                            break
                
                if len(expansions) >= self.max_expansions:
                    break
        
        except Exception as e:
            logger.warning(f"Semantic neighbor expansion failed: {e}")
        
        logger.debug(f"Generated {len(expansions)} semantic neighbor expansions")
        return expansions
    
    def _filter_and_rank_expansions(self, expansions: List[ExpandedQuery], original_query: str) -> List[ExpandedQuery]:
        """Filter and rank expansions by relevance and confidence."""
        if not expansions:
            return []
        
        # Remove duplicates
        seen_queries = {original_query.lower()}
        unique_expansions = []
        
        for expansion in expansions:
            if expansion.query.lower() not in seen_queries:
                seen_queries.add(expansion.query.lower())
                unique_expansions.append(expansion)
        
        # Sort by confidence (descending)
        unique_expansions.sort(key=lambda x: x.confidence, reverse=True)
        
        # Apply confidence threshold
        filtered_expansions = [
            exp for exp in unique_expansions 
            if exp.confidence >= self.confidence_threshold
        ]
        
        # Limit to max expansions
        return filtered_expansions[:self.max_expansions]
    
    def explain_expansion_decision(self, query: str) -> Dict[str, Any]:
        """Provide detailed explanation of expansion strategy selection."""
        intent_result = self.intent_classifier.classify(query)
        entities = self.entity_extractor.extract_entities(query)
        strategy = self._select_expansion_strategy(query, intent_result.intent, entities)
        
        return {
            'query': query,
            'detected_intent': intent_result.intent.value,
            'detected_attribute': intent_result.attribute,
            'extracted_entities': [
                {
                    'text': e.text,
                    'type': e.entity_type.value,
                    'normalized': e.normalized_form,
                    'confidence': e.confidence
                } for e in entities
            ],
            'selected_strategy': strategy.value,
            'strategy_rationale': self._get_strategy_rationale(query, intent_result.intent, entities, strategy),
            'configuration': {
                'max_expansions': self.max_expansions,
                'entity_hop_limit': self.entity_hop_limit,
                'confidence_threshold': self.confidence_threshold
            },
            'available_strategies': [s.value for s in ExpansionStrategy]
        }
    
    def _get_strategy_rationale(self, query: str, intent: IntentType, entities: List, strategy: ExpansionStrategy) -> str:
        """Get human-readable rationale for strategy selection."""
        if strategy == ExpansionStrategy.FACT_TEMPLATES:
            return f"Selected fact templates due to {intent.value} intent"
        elif strategy == ExpansionStrategy.ENTITY_RELATIONSHIPS:
            return f"Selected entity relationships due to {len(entities)} entities detected"
        elif strategy == ExpansionStrategy.SEMANTIC_NEIGHBORS:
            return "Selected semantic neighbors for general query enhancement"
        elif strategy == ExpansionStrategy.HYBRID_EXPANSION:
            return f"Selected hybrid approach for complex query with {len(query.split())} words"
        else:
            return "Default strategy selection"
    
    def get_expansion_statistics(self) -> Dict[str, Any]:
        """Get statistics about query expansion capabilities."""
        return {
            'expansion_strategies': [s.value for s in ExpansionStrategy],
            'supported_relationships': list(self.expansion_relationships.keys()),
            'configuration': {
                'max_expansions': self.max_expansions,
                'entity_hop_limit': self.entity_hop_limit,
                'confidence_threshold': self.confidence_threshold
            },
            'components_available': {
                'neo4j_client': self.neo4j_client is not None,
                'intent_classifier': self.intent_classifier is not None,
                'query_rewriter': self.query_rewriter is not None,
                'entity_extractor': True
            }
        }
    
    def benchmark_expansion_strategies(self, test_queries: List[str]) -> Dict[str, Any]:
        """Benchmark different expansion strategies on test queries."""
        results = {}
        
        for strategy in ExpansionStrategy:
            strategy_results = []
            
            for query in test_queries:
                try:
                    expansion_result = self.expand_query(query, strategy)
                    strategy_results.append({
                        'query': query,
                        'expansions_generated': len(expansion_result.expanded_queries),
                        'avg_confidence': sum(eq.confidence for eq in expansion_result.expanded_queries) / len(expansion_result.expanded_queries) if expansion_result.expanded_queries else 0,
                        'entities_found': len(expansion_result.extracted_entities),
                        'intent': expansion_result.detected_intent.value
                    })
                except Exception as e:
                    logger.warning(f"Benchmark failed for query '{query}' with strategy {strategy.value}: {e}")
            
            if strategy_results:
                avg_expansions = sum(r['expansions_generated'] for r in strategy_results) / len(strategy_results)
                avg_confidence = sum(r['avg_confidence'] for r in strategy_results) / len(strategy_results)
                
                results[strategy.value] = {
                    'queries_tested': len(strategy_results),
                    'avg_expansions_per_query': avg_expansions,
                    'avg_expansion_confidence': avg_confidence,
                    'sample_results': strategy_results[:3]  # Sample of first 3
                }
        
        logger.info(f"Benchmarked {len(ExpansionStrategy)} expansion strategies on {len(test_queries)} queries")
        return results