"""
Graph signal enhancement for fact-rich queries.

This module provides conditional graph signal contribution to improve fact retrieval
when entity relationships are present in the knowledge graph.
"""

import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging

# Try absolute imports first, fall back to relative
try:
    from .neo4j_client import Neo4jInitializer
    from ..core.intent_classifier import IntentType
except ImportError:
    try:
        from storage.neo4j_client import Neo4jInitializer
        from core.intent_classifier import IntentType
    except ImportError:
        from neo4j_client import Neo4jInitializer
        from intent_classifier import IntentType

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities that can be extracted and linked."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONTACT = "contact"
    PREFERENCE = "preference"
    ATTRIBUTE = "attribute"
    UNKNOWN = "unknown"


@dataclass
class EntityMention:
    """An entity mention with linking information."""
    text: str
    entity_type: EntityType
    confidence: float
    normalized_form: str
    start_pos: int
    end_pos: int


@dataclass
class GraphEdge:
    """A graph edge with relationship metadata."""
    source_node: str
    target_node: str
    relationship_type: str
    properties: Dict[str, Any]
    weight: float
    confidence: float


@dataclass
class GraphSignal:
    """Graph signal contribution to ranking."""
    entity_id: str
    relationship_score: float
    path_score: float
    neighborhood_score: float
    total_score: float
    explanation: str


class EntityExtractor:
    """
    Extracts and normalizes entities from text for graph linking.
    
    Uses pattern-based extraction with confidence scoring for
    high-precision entity recognition in personal fact domains.
    """
    
    def __init__(self) -> None:
        # Fact-specific entity patterns with high precision
        self.entity_patterns = {
            EntityType.PERSON: [
                # Name patterns
                (r"\b(?:my name is|i'?m (?:called )?|call me)\s+([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)", 0.95, "name_declaration"),
                (r"\b(?:i go by|people call me)\s+([A-Z][a-zA-Z]+)", 0.90, "name_alias"),
                (r"\b([A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+)\s+(?:is my name|told me)", 0.85, "name_mention"),
            ],
            EntityType.ORGANIZATION: [
                # Company/workplace patterns  
                (r"\bi work (?:at|for)\s+([A-Z][a-zA-Z]+(?: [A-Z&\w]+)*)", 0.90, "workplace"),
                (r"\b(?:my company|my employer) is\s+([A-Z][a-zA-Z]+(?: [A-Z&\w]+)*)", 0.95, "employer"),
                (r"\b([A-Z][a-zA-Z]+(?: (?:Inc|LLC|Corp|Ltd)\.?))\b", 0.80, "company_entity"),
            ],
            EntityType.LOCATION: [
                # Location patterns
                (r"\bi live in\s+([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*(?:, [A-Z]{2})?)", 0.95, "residence"),
                (r"\bi'?m from\s+([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)", 0.90, "origin"),
                (r"\b(?:my location|my address) is\s+([A-Z][a-zA-Z,\s]+)", 0.85, "location_statement"),
            ],
            EntityType.CONTACT: [
                # Contact information
                (r"\bmy email (?:address )?is\s+([\w._%+-]+@[\w.-]+\.\w+)", 0.98, "email"),
                (r"\bmy (?:phone|number) is\s+([\d\s\-\(\)\+]+)", 0.95, "phone"),
                (r"\b([\w._%+-]+@[\w.-]+\.\w+)\b", 0.80, "email_mention"),
            ],
            EntityType.PREFERENCE: [
                # Preferences and likes
                (r"\bi (?:like|love|enjoy)\s+([a-zA-Z\s]+?)(?:\s+(?:and|but|,|\.)|$)", 0.85, "preference"),
                (r"\bmy favorite\s+\w+\s+is\s+([a-zA-Z\s]+?)(?:\s+(?:and|but|,|\.)|$)", 0.90, "favorite"),
                (r"\bi prefer\s+([a-zA-Z\s]+?)(?:\s+(?:over|to|and|,|\.)|$)", 0.85, "preference_statement"),
            ]
        }
        
        # Normalization rules for entity linking
        self.normalization_rules = {
            EntityType.PERSON: self._normalize_person_name,
            EntityType.ORGANIZATION: self._normalize_organization,
            EntityType.LOCATION: self._normalize_location,
            EntityType.CONTACT: self._normalize_contact,
            EntityType.PREFERENCE: self._normalize_preference,
        }
    
    def extract_entities(self, text: str) -> List[EntityMention]:
        """
        Extract entity mentions from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of EntityMention objects with linking information
        """
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern, confidence, pattern_type in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity_text = match.group(1).strip()
                    if len(entity_text) < 2:  # Skip very short matches
                        continue
                    
                    # Normalize for linking
                    normalized = self.normalization_rules[entity_type](entity_text)
                    
                    entity = EntityMention(
                        text=entity_text,
                        entity_type=entity_type,
                        confidence=confidence,
                        normalized_form=normalized,
                        start_pos=match.start(1),
                        end_pos=match.end(1)
                    )
                    
                    entities.append(entity)
        
        # Remove overlapping entities (keep highest confidence)
        entities = self._resolve_overlaps(entities)
        
        logger.debug(f"Extracted {len(entities)} entities from text")
        return entities
    
    def _normalize_person_name(self, name: str) -> str:
        """Normalize person names for consistent linking."""
        # Convert to title case and remove extra spaces
        normalized = ' '.join(name.split()).title()
        return normalized
    
    def _normalize_organization(self, org: str) -> str:
        """Normalize organization names for consistent linking."""
        # Standardize common suffixes
        normalized = org.strip()
        suffixes = {'Inc.': 'Inc', 'LLC.': 'LLC', 'Corp.': 'Corp', 'Ltd.': 'Ltd'}
        for old, new in suffixes.items():
            normalized = normalized.replace(old, new)
        return normalized
    
    def _normalize_location(self, location: str) -> str:
        """Normalize location names for consistent linking."""
        # Remove extra commas and spaces, title case
        normalized = re.sub(r',\s*,', ',', location)
        normalized = ' '.join(normalized.split()).title()
        return normalized
    
    def _normalize_contact(self, contact: str) -> str:
        """Normalize contact information for consistent linking."""
        if '@' in contact:
            # Email normalization
            return contact.lower().strip()
        else:
            # Phone number normalization
            return re.sub(r'[^\d+]', '', contact)
    
    def _normalize_preference(self, preference: str) -> str:
        """Normalize preference text for consistent linking."""
        # Convert to lowercase and remove trailing punctuation
        normalized = preference.lower().strip()
        normalized = re.sub(r'[.,;!?]+$', '', normalized)
        return normalized
    
    def _resolve_overlaps(self, entities: List[EntityMention]) -> List[EntityMention]:
        """Resolve overlapping entity mentions by keeping highest confidence."""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda e: e.start_pos)
        
        resolved = []
        for entity in entities:
            # Check for overlap with previously added entities
            overlap = False
            for existing in resolved:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    # There's an overlap
                    if entity.confidence > existing.confidence:
                        # Remove the existing lower-confidence entity
                        resolved.remove(existing)
                        break
                    else:
                        # Skip this entity
                        overlap = True
                        break
            
            if not overlap:
                resolved.append(entity)
        
        return resolved


class GraphSignalEnhancer:
    """
    Conditional graph signal contribution for fact queries.
    
    Provides graph-based ranking boosts when entities have relationships
    in the knowledge graph, with graceful fallback when no graph data exists.
    """
    
    def __init__(self, neo4j_client: Optional[Neo4jInitializer] = None, 
                 config: Optional[Dict[str, Any]] = None):
        self.neo4j_client = neo4j_client
        self.config = config or {}
        self.entity_extractor = EntityExtractor()
        
        # Graph signal configuration
        self.max_hop_distance = self.config.get('max_hop_distance', 2)
        self.relationship_weights = self.config.get('relationship_weights', {
            'PERSON_WORKS_AT': 0.9,
            'PERSON_LIVES_IN': 0.8,
            'PERSON_LIKES': 0.7,
            'PERSON_HAS_CONTACT': 0.95,
            'PERSON_HAS_ATTRIBUTE': 0.85,
            'DEFAULT': 0.5
        })
        
        # Cache for entity neighborhoods
        self._entity_cache = {}
        
    def compute_graph_score(self, query: str, context_results: List[Dict[str, Any]]) -> Dict[str, GraphSignal]:
        """
        Compute graph signal contributions for search results.
        
        Args:
            query: User query text
            context_results: Search results to enhance
            
        Returns:
            Dictionary mapping result IDs to GraphSignal objects
        """
        if not self.neo4j_client:
            logger.debug("No Neo4j client available - skipping graph signals")
            return {}
        
        # Extract entities from query
        query_entities = self.entity_extractor.extract_entities(query)
        if not query_entities:
            logger.debug("No entities found in query - skipping graph signals")
            return {}
        
        graph_signals = {}
        
        for result in context_results:
            result_id = result.get('id', 'unknown')
            content = result.get('content', '')
            
            # Extract entities from result content
            content_entities = self.entity_extractor.extract_entities(content)
            
            # Compute graph signals for entity pairs
            signal = self._compute_entity_graph_signal(query_entities, content_entities)
            
            if signal and signal.total_score > 0:
                graph_signals[result_id] = signal
        
        logger.debug(f"Computed graph signals for {len(graph_signals)} results")
        return graph_signals
    
    def get_entity_edges(self, entity: EntityMention) -> List[GraphEdge]:
        """
        Get graph edges for an entity from Neo4j.
        
        Args:
            entity: Entity mention to look up
            
        Returns:
            List of GraphEdge objects representing relationships
        """
        if not self.neo4j_client:
            return []
        
        # Check cache first
        cache_key = f"{entity.entity_type.value}:{entity.normalized_form}"
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]
        
        try:
            # Query Neo4j for entity relationships
            cypher = """
            MATCH (e {normalized_name: $entity_name})
            WHERE any(label IN labels(e) WHERE label IN $entity_types)
            MATCH (e)-[r]-(connected)
            RETURN e.id as source_id, 
                   connected.id as target_id,
                   type(r) as rel_type,
                   properties(r) as rel_props,
                   r.weight as weight,
                   r.confidence as confidence
            LIMIT 50
            """
            
            parameters = {
                'entity_name': entity.normalized_form,
                'entity_types': [entity.entity_type.value.upper(), 'ENTITY']
            }
            
            results = self.neo4j_client.query(cypher, parameters)
            
            edges = []
            for record in results:
                edge = GraphEdge(
                    source_node=record.get('source_id', ''),
                    target_node=record.get('target_id', ''),
                    relationship_type=record.get('rel_type', 'UNKNOWN'),
                    properties=record.get('rel_props', {}),
                    weight=record.get('weight', 0.5),
                    confidence=record.get('confidence', 0.5)
                )
                edges.append(edge)
            
            # Cache the results
            self._entity_cache[cache_key] = edges
            
            logger.debug(f"Found {len(edges)} edges for entity {entity.normalized_form}")
            return edges
            
        except Exception as e:
            logger.warning(f"Failed to query graph for entity {entity.normalized_form}: {e}")
            return []
    
    def _compute_entity_graph_signal(self, query_entities: List[EntityMention], 
                                   content_entities: List[EntityMention]) -> Optional[GraphSignal]:
        """Compute graph signal based on entity relationships."""
        if not query_entities or not content_entities:
            return None
        
        max_score = 0.0
        best_explanation = ""
        best_entity = ""
        
        for q_entity in query_entities:
            for c_entity in content_entities:
                # Get edges for both entities
                q_edges = self.get_entity_edges(q_entity)
                c_edges = self.get_entity_edges(c_entity)
                
                # Compute relationship score
                rel_score = self._compute_relationship_score(q_entity, c_entity, q_edges, c_edges)
                
                # Compute path score (entity similarity)
                path_score = self._compute_path_score(q_entity, c_entity)
                
                # Compute neighborhood score
                neighborhood_score = self._compute_neighborhood_score(q_edges, c_edges)
                
                # Combined score with weights
                total_score = (
                    0.4 * rel_score +
                    0.3 * path_score +
                    0.3 * neighborhood_score
                )
                
                if total_score > max_score:
                    max_score = total_score
                    best_entity = f"{q_entity.normalized_form}->{c_entity.normalized_form}"
                    best_explanation = f"Graph connection between {q_entity.text} and {c_entity.text} (score: {total_score:.3f})"
        
        if max_score > 0:
            return GraphSignal(
                entity_id=best_entity,
                relationship_score=0.0,  # Will be computed in detailed scoring
                path_score=0.0,
                neighborhood_score=0.0,
                total_score=max_score,
                explanation=best_explanation
            )
        
        return None
    
    def _compute_relationship_score(self, entity1: EntityMention, entity2: EntityMention,
                                  edges1: List[GraphEdge], edges2: List[GraphEdge]) -> float:
        """Compute direct relationship score between entities."""
        # Check for direct relationships
        for edge1 in edges1:
            for edge2 in edges2:
                if (edge1.target_node == edge2.source_node or 
                    edge1.source_node == edge2.target_node):
                    # Direct connection found
                    weight = self.relationship_weights.get(edge1.relationship_type, 
                                                         self.relationship_weights['DEFAULT'])
                    return min(edge1.confidence * edge2.confidence * weight, 1.0)
        
        return 0.0
    
    def _compute_path_score(self, entity1: EntityMention, entity2: EntityMention) -> float:
        """Compute entity similarity/path score."""
        # Simple entity type and text similarity
        type_bonus = 0.2 if entity1.entity_type == entity2.entity_type else 0.0
        
        # Text similarity (simple word overlap)
        words1 = set(entity1.normalized_form.lower().split())
        words2 = set(entity2.normalized_form.lower().split())
        
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            text_sim = overlap / union if union > 0 else 0.0
        else:
            text_sim = 0.0
        
        return min(type_bonus + text_sim * 0.8, 1.0)
    
    def _compute_neighborhood_score(self, edges1: List[GraphEdge], edges2: List[GraphEdge]) -> float:
        """Compute neighborhood overlap score."""
        if not edges1 or not edges2:
            return 0.0
        
        # Get neighboring nodes
        neighbors1 = set(edge.target_node for edge in edges1)
        neighbors2 = set(edge.target_node for edge in edges2)
        
        # Compute Jaccard similarity
        intersection = len(neighbors1.intersection(neighbors2))
        union = len(neighbors1.union(neighbors2))
        
        return intersection / union if union > 0 else 0.0
    
    def apply_gamma_graph(self, scores: Dict[str, float], graph_signals: Dict[str, GraphSignal], 
                         gamma: float = 0.1) -> Dict[str, float]:
        """
        Apply graph signal boost to ranking scores.
        
        Args:
            scores: Original ranking scores by result ID
            graph_signals: Graph signals by result ID
            gamma: Weight for graph signal contribution (0.0-1.0)
            
        Returns:
            Enhanced scores with graph signal contributions
        """
        if gamma <= 0 or not graph_signals:
            return scores
        
        enhanced_scores = scores.copy()
        
        for result_id, signal in graph_signals.items():
            if result_id in enhanced_scores:
                original_score = enhanced_scores[result_id]
                graph_boost = gamma * signal.total_score
                enhanced_score = min(original_score + graph_boost, 1.0)
                
                enhanced_scores[result_id] = enhanced_score
                
                logger.debug(f"Applied graph boost to {result_id}: {original_score:.3f} -> {enhanced_score:.3f} (+{graph_boost:.3f})")
        
        return enhanced_scores
    
    def explain_graph_contribution(self, result_id: str, graph_signals: Dict[str, GraphSignal]) -> str:
        """Generate human-readable explanation of graph contribution."""
        if result_id not in graph_signals:
            return "No graph signal available"
        
        signal = graph_signals[result_id]
        return signal.explanation
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about graph signal usage."""
        return {
            'entity_cache_size': len(self._entity_cache),
            'supported_entity_types': [et.value for et in EntityType],
            'relationship_weights': self.relationship_weights,
            'max_hop_distance': self.max_hop_distance,
            'neo4j_available': self.neo4j_client is not None
        }
    
    def clear_cache(self) -> None:
        """Clear the entity relationship cache."""
        self._entity_cache.clear()
        logger.info("Graph signal cache cleared")