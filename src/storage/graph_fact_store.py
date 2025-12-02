"""
Graph-integrated fact storage for entity relationship modeling.

This module extends the basic fact store with graph storage capabilities,
creating entity nodes and relationships when facts are stored.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import logging
import json
import time

# Try absolute imports first, fall back to relative
try:
    from .fact_store import FactStore
    from .graph_enhancer import EntityExtractor, EntityMention, EntityType
    from .neo4j_client import Neo4jInitializer
except ImportError:
    try:
        from storage.fact_store import FactStore
        from storage.graph_enhancer import EntityExtractor, EntityMention, EntityType
        from storage.neo4j_client import Neo4jInitializer
    except ImportError:
        from fact_store import FactStore
        from graph_enhancer import EntityExtractor, EntityMention, EntityType
        from neo4j_client import Neo4jInitializer

# Import monitoring for Neo4j fallback metrics
try:
    from ..monitoring.fact_monitoring import record_custom_metric
except ImportError:
    try:
        from monitoring.fact_monitoring import record_custom_metric
    except ImportError:
        # Fallback if monitoring not available
        def record_custom_metric(metric_name: str, value: float, **labels) -> None:
            pass

logger = logging.getLogger(__name__)


@dataclass
class EntityNode:
    """Represents an entity node in the graph."""
    id: str
    entity_type: EntityType
    name: str
    normalized_name: str
    properties: Dict[str, Any]
    confidence: float


@dataclass
class FactRelationship:
    """Represents a fact-based relationship in the graph."""
    source_entity: str
    target_entity: str
    relationship_type: str
    fact_attribute: str
    fact_value: Any
    confidence: float
    metadata: Dict[str, Any]


class GraphFactStore:
    """
    Extended fact store with graph storage capabilities.
    
    Stores facts in both Redis (for fast lookup) and Neo4j (for relationships),
    creating entity nodes and modeling fact relationships in the graph.
    """
    
    def __init__(self, fact_store: FactStore, neo4j_client: Optional[Neo4jInitializer] = None):
        self.fact_store = fact_store
        self.neo4j_client = neo4j_client
        self.entity_extractor = EntityExtractor()
        
        # Entity relationship mapping
        self.fact_to_relationship_map = {
            'name': 'PERSON_HAS_NAME',
            'email': 'PERSON_HAS_EMAIL',
            'phone': 'PERSON_HAS_PHONE',
            'location': 'PERSON_LIVES_IN',
            'job': 'PERSON_WORKS_AT',
            'age': 'PERSON_HAS_AGE',
            'birthday': 'PERSON_BORN_ON',
            'preferences.food': 'PERSON_LIKES_FOOD',
            'preferences.music': 'PERSON_LIKES_MUSIC',
        }
        
    def store_fact(self, namespace: str, user_id: str, attribute: str, value: Any, 
                   confidence: float = 1.0, source_turn_id: str = "", 
                   provenance: str = "user_input") -> None:
        """
        Store fact in both Redis and graph with entity modeling.
        
        Args:
            namespace: Fact namespace
            user_id: User ID for scoping
            attribute: Fact attribute name
            value: Fact value
            confidence: Confidence score (0.0-1.0)
            source_turn_id: Source conversation turn ID
            provenance: Fact provenance information
        """
        start_time = time.time()
        
        # Store in Redis first (fast lookup)
        self.fact_store.store_fact(namespace, user_id, attribute, value, confidence, 
                                 source_turn_id, provenance)
        
        # Track Redis storage success
        record_custom_metric("graph_fact_store_redis_operations", 1.0, 
                           operation="store", status="success", namespace=namespace)
        
        # Store in graph if available
        if self.neo4j_client:
            try:
                graph_start_time = time.time()
                self._store_fact_in_graph(namespace, user_id, attribute, value, confidence, 
                                        source_turn_id, provenance)
                
                # Track successful graph storage
                graph_duration = (time.time() - graph_start_time) * 1000
                record_custom_metric("graph_fact_store_neo4j_operations", 1.0,
                                   operation="store", status="success", namespace=namespace)
                record_custom_metric("graph_fact_store_neo4j_duration_ms", graph_duration,
                                   operation="store", namespace=namespace)
                
            except Exception as e:
                logger.warning(f"Failed to store fact in graph: {e}")
                
                # Track graph storage failure and fallback
                record_custom_metric("graph_fact_store_neo4j_operations", 1.0,
                                   operation="store", status="error", namespace=namespace)
                record_custom_metric("graph_fact_store_fallback_occurrences", 1.0,
                                   operation="store", reason="neo4j_error", namespace=namespace)
        else:
            # Track fallback due to no Neo4j client
            record_custom_metric("graph_fact_store_fallback_occurrences", 1.0,
                               operation="store", reason="no_neo4j_client", namespace=namespace)
        
        # Track overall operation duration
        total_duration = (time.time() - start_time) * 1000
        record_custom_metric("graph_fact_store_total_duration_ms", total_duration,
                           operation="store", namespace=namespace)
    
    def _store_fact_in_graph(self, namespace: str, user_id: str, attribute: str, 
                           value: Any, confidence: float, source_turn_id: str, 
                           provenance: str) -> None:
        """Store fact relationships in Neo4j graph."""
        if not self.neo4j_client:
            return
        
        # Create/update user entity node
        user_node_id = self._ensure_user_entity(namespace, user_id)
        
        # Extract entities from the fact value
        fact_text = f"My {attribute} is {value}"
        entities = self.entity_extractor.extract_entities(fact_text)
        
        # Create entity nodes and relationships
        for entity in entities:
            entity_node_id = self._ensure_entity_node(entity)
            self._create_fact_relationship(user_node_id, entity_node_id, attribute, 
                                         value, confidence, source_turn_id, provenance)
        
        # For simple attributes without complex entities, create attribute nodes
        if not entities and attribute in self.fact_to_relationship_map:
            attr_node_id = self._ensure_attribute_node(attribute, value)
            self._create_fact_relationship(user_node_id, attr_node_id, attribute,
                                         value, confidence, source_turn_id, provenance)
    
    def _ensure_user_entity(self, namespace: str, user_id: str) -> str:
        """Ensure user entity exists in graph."""
        user_key = f"{namespace}:{user_id}"
        
        try:
            # Check if user entity exists
            cypher = """
            MERGE (u:User:Person {user_id: $user_id, namespace: $namespace})
            ON CREATE SET u.id = $user_key,
                         u.created_at = datetime(),
                         u.fact_count = 0
            ON MATCH SET u.last_updated = datetime(),
                        u.fact_count = u.fact_count + 1
            RETURN u.id as user_id
            """
            
            result = self.neo4j_client.query(cypher, {
                'user_id': user_id,
                'namespace': namespace,
                'user_key': user_key
            })
            
            if result:
                return result[0]['user_id']
            else:
                raise RuntimeError("Failed to create/retrieve user entity")
                
        except Exception as e:
            logger.error(f"Failed to ensure user entity: {e}")
            raise
    
    def _ensure_entity_node(self, entity: EntityMention) -> str:
        """Ensure entity node exists in graph."""
        entity_id = f"{entity.entity_type.value}:{entity.normalized_form}"
        
        try:
            # Determine entity labels based on type
            labels = ['Entity']
            if entity.entity_type == EntityType.PERSON:
                labels.append('Person')
            elif entity.entity_type == EntityType.ORGANIZATION:
                labels.append('Organization')
            elif entity.entity_type == EntityType.LOCATION:
                labels.append('Location')
            elif entity.entity_type == EntityType.CONTACT:
                labels.append('Contact')
            elif entity.entity_type == EntityType.PREFERENCE:
                labels.append('Preference')
            
            # Create entity node
            cypher = f"""
            MERGE (e:{':'.join(labels)} {{normalized_name: $normalized_name}})
            ON CREATE SET e.id = $entity_id,
                         e.original_text = $original_text,
                         e.entity_type = $entity_type,
                         e.confidence = $confidence,
                         e.created_at = datetime(),
                         e.mention_count = 1
            ON MATCH SET e.last_mentioned = datetime(),
                        e.mention_count = e.mention_count + 1,
                        e.confidence = CASE 
                            WHEN $confidence > e.confidence THEN $confidence 
                            ELSE e.confidence 
                        END
            RETURN e.id as entity_id
            """
            
            result = self.neo4j_client.query(cypher, {
                'entity_id': entity_id,
                'normalized_name': entity.normalized_form,
                'original_text': entity.text,
                'entity_type': entity.entity_type.value,
                'confidence': entity.confidence
            })
            
            if result:
                return result[0]['entity_id']
            else:
                raise RuntimeError(f"Failed to create entity node for {entity.text}")
                
        except Exception as e:
            logger.error(f"Failed to ensure entity node: {e}")
            raise
    
    def _ensure_attribute_node(self, attribute: str, value: Any) -> str:
        """Ensure attribute value node exists in graph."""
        attr_id = f"attr:{attribute}:{str(value)}"
        
        try:
            cypher = """
            MERGE (a:Attribute {attribute_name: $attribute, value: $value})
            ON CREATE SET a.id = $attr_id,
                         a.created_at = datetime(),
                         a.reference_count = 1
            ON MATCH SET a.last_referenced = datetime(),
                        a.reference_count = a.reference_count + 1
            RETURN a.id as attr_id
            """
            
            result = self.neo4j_client.query(cypher, {
                'attr_id': attr_id,
                'attribute': attribute,
                'value': str(value)
            })
            
            if result:
                return result[0]['attr_id']
            else:
                raise RuntimeError(f"Failed to create attribute node for {attribute}={value}")
                
        except Exception as e:
            logger.error(f"Failed to ensure attribute node: {e}")
            raise
    
    def _create_fact_relationship(self, source_id: str, target_id: str, attribute: str,
                                value: Any, confidence: float, source_turn_id: str,
                                provenance: str) -> None:
        """Create fact-based relationship between entities."""
        if not self.neo4j_client:
            return
        
        # Get relationship type
        rel_type = self.fact_to_relationship_map.get(attribute, 'HAS_FACT_ABOUT')
        
        try:
            cypher = """
            MATCH (source {id: $source_id})
            MATCH (target {id: $target_id})
            MERGE (source)-[r:HAS_FACT_RELATION]->(target)
            SET r.relationship_type = $rel_type,
                r.fact_attribute = $attribute,
                r.fact_value = $value,
                r.confidence = $confidence,
                r.source_turn_id = $source_turn_id,
                r.provenance = $provenance,
                r.created_at = datetime(),
                r.weight = $confidence
            RETURN r
            """
            
            self.neo4j_client.query(cypher, {
                'source_id': source_id,
                'target_id': target_id,
                'rel_type': rel_type,
                'attribute': attribute,
                'value': str(value),
                'confidence': confidence,
                'source_turn_id': source_turn_id,
                'provenance': provenance
            })
            
            logger.debug(f"Created fact relationship: {source_id} -[{rel_type}]-> {target_id}")
            
        except Exception as e:
            logger.error(f"Failed to create fact relationship: {e}")
            raise
    
    def get_fact(self, namespace: str, user_id: str, attribute: str) -> Optional[Any]:
        """Get fact value (delegates to Redis store)."""
        start_time = time.time()
        
        try:
            result = self.fact_store.get_fact(namespace, user_id, attribute)
            
            # Track successful retrieval
            record_custom_metric("graph_fact_store_redis_operations", 1.0,
                               operation="get", status="success", namespace=namespace)
            
            if result is not None:
                record_custom_metric("graph_fact_store_retrieval_hits", 1.0,
                                   namespace=namespace, attribute=attribute)
            else:
                record_custom_metric("graph_fact_store_retrieval_misses", 1.0,
                                   namespace=namespace, attribute=attribute)
            
            return result
            
        except Exception as e:
            # Track retrieval failure
            record_custom_metric("graph_fact_store_redis_operations", 1.0,
                               operation="get", status="error", namespace=namespace)
            raise
        finally:
            # Track retrieval duration
            duration = (time.time() - start_time) * 1000
            record_custom_metric("graph_fact_store_total_duration_ms", duration,
                               operation="get", namespace=namespace)
    
    def get_user_facts(self, namespace: str, user_id: str) -> Dict[str, Any]:
        """Get all facts for user (delegates to Redis store)."""
        return self.fact_store.get_user_facts(namespace, user_id)
    
    def delete_user_facts(self, namespace: str, user_id: str) -> int:
        """Delete all facts for user from both Redis and graph."""
        # Delete from Redis
        deleted_count = self.fact_store.delete_user_facts(namespace, user_id)
        
        # Delete from graph
        if self.neo4j_client:
            try:
                self._delete_user_facts_from_graph(namespace, user_id)
            except Exception as e:
                logger.warning(f"Failed to delete user facts from graph: {e}")
        
        return deleted_count
    
    def _delete_user_facts_from_graph(self, namespace: str, user_id: str) -> None:
        """Delete user facts from Neo4j graph."""
        if not self.neo4j_client:
            return
        
        try:
            # Delete user's fact relationships
            cypher = """
            MATCH (u:User {user_id: $user_id, namespace: $namespace})-[r:HAS_FACT_RELATION]-()
            DELETE r
            """
            
            self.neo4j_client.query(cypher, {
                'user_id': user_id,
                'namespace': namespace
            })
            
            # Optionally delete the user node if no other relationships
            cypher = """
            MATCH (u:User {user_id: $user_id, namespace: $namespace})
            WHERE NOT (u)-[]-()
            DELETE u
            """
            
            self.neo4j_client.query(cypher, {
                'user_id': user_id,
                'namespace': namespace
            })
            
            logger.debug(f"Deleted user facts from graph: {namespace}:{user_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete user facts from graph: {e}")
            raise
    
    def get_entity_relationships(self, namespace: str, user_id: str) -> List[FactRelationship]:
        """Get all fact relationships for a user from the graph."""
        if not self.neo4j_client:
            return []
        
        try:
            cypher = """
            MATCH (u:User {user_id: $user_id, namespace: $namespace})-[r:HAS_FACT_RELATION]->(target)
            RETURN u.id as source_entity,
                   target.id as target_entity,
                   r.relationship_type as relationship_type,
                   r.fact_attribute as fact_attribute,
                   r.fact_value as fact_value,
                   r.confidence as confidence,
                   r.source_turn_id as source_turn_id,
                   r.provenance as provenance,
                   r.created_at as created_at
            ORDER BY r.created_at DESC
            """
            
            results = self.neo4j_client.query(cypher, {
                'user_id': user_id,
                'namespace': namespace
            })
            
            relationships = []
            for record in results:
                rel = FactRelationship(
                    source_entity=record.get('source_entity', ''),
                    target_entity=record.get('target_entity', ''),
                    relationship_type=record.get('relationship_type', ''),
                    fact_attribute=record.get('fact_attribute', ''),
                    fact_value=record.get('fact_value', ''),
                    confidence=record.get('confidence', 0.0),
                    metadata={
                        'source_turn_id': record.get('source_turn_id', ''),
                        'provenance': record.get('provenance', ''),
                        'created_at': str(record.get('created_at', ''))
                    }
                )
                relationships.append(rel)
            
            logger.debug(f"Retrieved {len(relationships)} fact relationships for {namespace}:{user_id}")
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to get entity relationships: {e}")
            return []
    
    def query_related_facts(self, namespace: str, user_id: str, entity_type: Optional[str] = None,
                          relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query related facts based on entity or relationship type."""
        if not self.neo4j_client:
            return []
        
        try:
            # Build dynamic query based on filters
            where_clauses = ["u.user_id = $user_id", "u.namespace = $namespace"]
            params = {'user_id': user_id, 'namespace': namespace}
            
            if entity_type:
                where_clauses.append("target.entity_type = $entity_type")
                params['entity_type'] = entity_type
            
            if relationship_type:
                where_clauses.append("r.relationship_type = $relationship_type")
                params['relationship_type'] = relationship_type
            
            where_clause = " AND ".join(where_clauses)
            
            cypher = f"""
            MATCH (u:User)-[r:HAS_FACT_RELATION]->(target)
            WHERE {where_clause}
            RETURN u.id as user_entity,
                   target.id as target_entity,
                   target.normalized_name as target_name,
                   target.entity_type as target_type,
                   r.relationship_type as relationship_type,
                   r.fact_attribute as fact_attribute,
                   r.fact_value as fact_value,
                   r.confidence as confidence
            ORDER BY r.confidence DESC, r.created_at DESC
            """
            
            results = self.neo4j_client.query(cypher, params)
            
            logger.debug(f"Found {len(results)} related facts for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to query related facts: {e}")
            return []
    
    def get_graph_statistics(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about the fact graph."""
        if not self.neo4j_client:
            return {'error': 'Neo4j not available'}
        
        try:
            stats = {}
            
            # User count
            user_cypher = "MATCH (u:User)"
            if namespace:
                user_cypher += " WHERE u.namespace = $namespace"
            user_cypher += " RETURN count(u) as user_count"
            
            user_result = self.neo4j_client.query(user_cypher, 
                                                {'namespace': namespace} if namespace else {})
            stats['user_count'] = user_result[0]['user_count'] if user_result else 0
            
            # Entity count by type
            entity_cypher = """
            MATCH (e:Entity)
            RETURN e.entity_type as entity_type, count(e) as count
            ORDER BY count DESC
            """
            
            entity_results = self.neo4j_client.query(entity_cypher)
            stats['entity_counts'] = {r['entity_type']: r['count'] for r in entity_results}
            
            # Relationship count by type
            rel_cypher = """
            MATCH ()-[r:HAS_FACT_RELATION]->()
            RETURN r.relationship_type as rel_type, count(r) as count
            ORDER BY count DESC
            """
            
            rel_results = self.neo4j_client.query(rel_cypher)
            stats['relationship_counts'] = {r['rel_type']: r['count'] for r in rel_results}
            
            # Total fact relationships
            total_cypher = "MATCH ()-[r:HAS_FACT_RELATION]->() RETURN count(r) as total"
            total_result = self.neo4j_client.query(total_cypher)
            stats['total_fact_relationships'] = total_result[0]['total'] if total_result else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {'error': str(e)}