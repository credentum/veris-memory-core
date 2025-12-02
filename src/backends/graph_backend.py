#!/usr/bin/env python3
"""
Graph backend adapter for Neo4j graph database.

This module implements the BackendSearchInterface for Neo4j graph operations,
providing graph traversal and relationship-based search capabilities.
"""

import time
from typing import List, Dict, Any, Optional

from ..interfaces.backend_interface import BackendSearchInterface, SearchOptions, BackendHealthStatus, BackendSearchError
from ..interfaces.memory_result import MemoryResult, ResultSource, ContentType
from ..utils.logging_middleware import backend_logger, log_backend_timing


class GraphBackend(BackendSearchInterface):
    """
    Graph search backend implementation using Neo4j.
    
    Provides relationship-based search capabilities and graph traversal
    for complex context queries.
    """
    
    def __init__(self, neo4j_client, cypher_validator=None):
        """
        Initialize graph backend.
        
        Args:
            neo4j_client: Neo4j client instance for graph operations
            cypher_validator: Optional validator for Cypher queries
        """
        self.client = neo4j_client
        self.cypher_validator = cypher_validator
        self._node_label = "Context"  # Default node label
    
    @property
    def backend_name(self) -> str:
        """Return the name of this backend."""
        return "graph"
    
    async def search(self, query: str, options: SearchOptions) -> List[MemoryResult]:
        """
        Search graph database using Cypher queries and text matching.
        
        Args:
            query: Search query text
            options: Search configuration options
            
        Returns:
            List of MemoryResult objects from graph nodes
            
        Raises:
            BackendSearchError: If graph search operation fails
        """
        async with log_backend_timing(self.backend_name, "search", backend_logger) as metadata:
            try:
                # Build Cypher query based on search parameters
                cypher_query, parameters = self._build_search_query(query, options)
                
                metadata["cypher_query_length"] = len(cypher_query)
                metadata["parameter_count"] = len(parameters)
                
                # Execute query
                search_start = time.time()
                raw_results = self._execute_query(cypher_query, parameters)
                search_time = (time.time() - search_start) * 1000
                
                metadata["search_time_ms"] = search_time
                metadata["raw_result_count"] = len(raw_results)
                
                # Convert to normalized format
                results = self._convert_to_memory_results(raw_results)
                
                # Apply additional filtering
                filtered_results = self._apply_filters(results, options)
                
                metadata["result_count"] = len(filtered_results)
                metadata["top_score"] = filtered_results[0].score if filtered_results else 0.0
                
                backend_logger.info(
                    f"Graph search completed",
                    query_type="cypher_text_match",
                    **metadata
                )
                
                return filtered_results
                
            except Exception as e:
                error_msg = f"Graph search failed: {str(e)}"
                backend_logger.error(error_msg, error=str(e))
                raise BackendSearchError(self.backend_name, error_msg, e)
    
    async def health_check(self) -> BackendHealthStatus:
        """
        Check the health of the graph backend.
        
        Returns:
            BackendHealthStatus with current health information
        """
        start_time = time.time()
        
        try:
            # Test basic connectivity with simple query
            test_query = "RETURN 1 as test"
            result = self._execute_query(test_query, {})
            response_time = (time.time() - start_time) * 1000
            
            # Check if we got expected result
            connectivity_ok = len(result) == 1 and result[0].get('test') == 1
            
            # Check node count in our collection
            count_query = f"MATCH (n:{self._node_label}) RETURN count(n) as node_count"
            try:
                count_result = self._execute_query(count_query, {})
                node_count = count_result[0].get('node_count', 0) if count_result else 0
            except Exception:
                node_count = -1  # Indicates count query failed
            
            # Determine overall status
            if connectivity_ok and node_count >= 0:
                status = "healthy"
            elif connectivity_ok:
                status = "degraded"  # Connected but can't count nodes
            else:
                status = "unhealthy"
            
            return BackendHealthStatus(
                status=status,
                response_time_ms=response_time,
                metadata={
                    "connectivity_test_passed": connectivity_ok,
                    "node_label": self._node_label,
                    "node_count": node_count,
                    "cypher_validator_available": self.cypher_validator is not None
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return BackendHealthStatus(
                status="unhealthy",
                response_time_ms=response_time,
                error_message=str(e),
                metadata={
                    "connection_error": True,
                    "error_type": type(e).__name__
                }
            )
    
    async def initialize(self) -> None:
        """Initialize the graph backend."""
        backend_logger.info("Initializing graph backend")
        
        try:
            # Test connectivity
            await self.health_check()
            
            # Optionally create indexes for better performance
            self._create_indexes()
            
            backend_logger.info("Graph backend initialized successfully")
            
        except Exception as e:
            backend_logger.error(f"Graph backend initialization failed: {e}")
            raise BackendSearchError(self.backend_name, f"Initialization failed: {e}", e)
    
    async def cleanup(self) -> None:
        """Clean up graph backend resources."""
        backend_logger.info("Cleaning up graph backend")
        # Neo4j client cleanup is handled by the client itself
    
    # Public method for custom Cypher queries
    async def execute_cypher(self, cypher_query: str, parameters: Dict[str, Any] = None) -> List[MemoryResult]:
        """
        Execute a custom Cypher query and return normalized results.
        
        Args:
            cypher_query: The Cypher query to execute
            parameters: Query parameters
            
        Returns:
            List of MemoryResult objects
            
        Raises:
            BackendSearchError: If query execution fails or is invalid
        """
        # Validate query if validator is available
        if self.cypher_validator:
            is_valid, error_msg = self.cypher_validator.validate_query(cypher_query)
            if not is_valid:
                raise BackendSearchError(
                    self.backend_name, 
                    f"Invalid Cypher query: {error_msg}",
                    ValueError(error_msg)
                )
        
        async with log_backend_timing(self.backend_name, "custom_cypher", backend_logger) as metadata:
            try:
                metadata["query_length"] = len(cypher_query)
                metadata["has_parameters"] = bool(parameters)
                
                raw_results = self._execute_query(cypher_query, parameters or {})
                results = self._convert_to_memory_results(raw_results)
                
                metadata["result_count"] = len(results)
                
                return results
                
            except Exception as e:
                error_msg = f"Custom Cypher execution failed: {str(e)}"
                backend_logger.error(error_msg, query=cypher_query[:100])
                raise BackendSearchError(self.backend_name, error_msg, e)
    
    # Private helper methods
    
    def _build_search_query(self, query: str, options: SearchOptions) -> tuple[str, Dict[str, Any]]:
        """Build Cypher query for text search with multi-word support."""
        parameters = {
            "limit": options.limit
        }

        # Base query with text matching
        where_conditions = []

        # Text search - search across all actual Context node fields
        # Fix for retrieval issue: Context nodes have title, description, keyword, user_input, bot_response
        # NOT the generic 'text' or 'content' fields that were being searched
        # PR #340: Added searchable_text field for custom property search
        text_fields = [
            "n.title",           # Manual contexts
            "n.description",     # Manual contexts
            "n.keyword",         # Manual contexts
            "n.user_input",      # Voice bot contexts
            "n.bot_response",    # Voice bot contexts
            "n.searchable_text"  # PR #340: Unified searchable field (standard + custom properties)
        ]

        # Enhanced multi-word search: split query into words
        # Each word must appear in at least one field (AND logic across words, OR across fields)
        query_words = [word.strip() for word in query.split() if word.strip()]

        if not query_words:
            # Empty query - match nothing
            where_conditions.append("false")
        elif len(query_words) == 1:
            # Single word - use simple CONTAINS (faster)
            parameters["search_text"] = query_words[0]
            text_search_conditions = " OR ".join([
                f"({field} IS NOT NULL AND toLower({field}) CONTAINS toLower($search_text))"
                for field in text_fields
            ])
            where_conditions.append(f"({text_search_conditions})")
        else:
            # Multi-word search - each word must appear in at least one field
            word_conditions = []
            for idx, word in enumerate(query_words):
                param_name = f"word_{idx}"
                parameters[param_name] = word
                # This word must appear in at least one field
                field_conditions = " OR ".join([
                    f"({field} IS NOT NULL AND toLower({field}) CONTAINS toLower(${param_name}))"
                    for field in text_fields
                ])
                word_conditions.append(f"({field_conditions})")

            # All words must be found (AND across words)
            combined_condition = " AND ".join(word_conditions)
            where_conditions.append(f"({combined_condition})")
        
        # Apply namespace filter
        if options.namespace:
            where_conditions.append("n.namespace = $namespace")
            parameters["namespace"] = options.namespace
        
        # Apply type filter
        if options.filters.get("type"):
            where_conditions.append("n.type = $type_filter")
            parameters["type_filter"] = options.filters["type"]
        
        # Apply user_id filter
        if options.filters.get("user_id"):
            where_conditions.append("n.user_id = $user_id_filter")
            parameters["user_id_filter"] = options.filters["user_id"]
        
        # Combine conditions
        where_clause = " AND ".join(where_conditions) if where_conditions else "true"
        
        cypher_query = f"""
        MATCH (n:{self._node_label})
        WHERE {where_clause}
        RETURN n
        ORDER BY n.timestamp DESC
        LIMIT $limit
        """
        
        return cypher_query.strip(), parameters
    
    def _execute_query(self, cypher_query: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute Cypher query and return results."""
        try:
            return self.client.query(cypher_query, parameters=parameters)
        except Exception as e:
            backend_logger.error(f"Cypher query execution failed: {e}", query=cypher_query[:200])
            raise
    
    def _convert_to_memory_results(self, raw_results: List[Dict[str, Any]]) -> List[MemoryResult]:
        """Convert Neo4j results to normalized MemoryResult format."""
        results = []
        
        for result in raw_results:
            try:
                # Handle different result formats
                if isinstance(result, dict) and 'n' in result:
                    # Standard query result format: {'n': {...}}
                    node_data = result['n']
                elif isinstance(result, dict):
                    # Direct node data
                    node_data = result
                else:
                    # Skip unknown formats
                    backend_logger.warning(f"Unknown result format: {type(result)}")
                    continue
                
                # Extract required fields with fallbacks
                result_id = str(node_data.get('id', node_data.get('_id', id(node_data))))

                # Extract text from actual Context node fields
                # Try all searchable fields in priority order
                text = (
                    node_data.get('title') or
                    node_data.get('description') or
                    node_data.get('user_input') or
                    node_data.get('bot_response') or
                    node_data.get('keyword') or
                    node_data.get('user_message') or  # Legacy field
                    node_data.get('message') or        # Legacy field
                    ''
                )

                if not text:
                    backend_logger.warning(f"Node missing text content: {result_id}")
                    continue
                
                # Determine content type
                content_type = ContentType.GENERAL
                if node_data.get('type'):
                    try:
                        content_type = ContentType(node_data['type'])
                    except ValueError:
                        pass
                
                # Calculate relevance score (graph results don't have similarity scores)
                score = 1.0
                if node_data.get('relevance_score'):
                    score = float(node_data['relevance_score'])
                elif 'created_at' in node_data or 'timestamp' in node_data:
                    # Simple time-based scoring - more recent = higher score
                    import datetime
                    timestamp_field = node_data.get('timestamp', node_data.get('created_at'))
                    if timestamp_field:
                        try:
                            # Assume recent content is more relevant
                            if isinstance(timestamp_field, str):
                                timestamp = datetime.datetime.fromisoformat(timestamp_field.replace('Z', '+00:00'))
                            else:
                                timestamp = timestamp_field
                            
                            now = datetime.datetime.now(datetime.timezone.utc)
                            days_old = (now - timestamp).days
                            score = max(0.1, 1.0 - (days_old / 365))  # Decay over a year
                        except Exception:
                            score = 0.5  # Default for unparseable timestamps
                
                # Extract tags
                tags = []
                if node_data.get('tags'):
                    if isinstance(node_data['tags'], list):
                        tags = node_data['tags']
                    else:
                        tags = [str(node_data['tags'])]
                
                # Create normalized result
                memory_result = MemoryResult(
                    id=result_id,
                    text=text,
                    type=content_type,
                    score=min(max(score, 0.0), 1.0),  # Clamp to valid range
                    source=ResultSource.GRAPH,
                    tags=tags,
                    metadata={
                        **node_data,
                        'graph_search': True,
                        'node_label': self._node_label
                    },
                    namespace=node_data.get('namespace'),
                    title=node_data.get('title'),
                    user_id=node_data.get('user_id')
                )
                
                results.append(memory_result)
                
            except Exception as e:
                backend_logger.warning(f"Failed to convert graph result: {e}", result_data=str(result)[:200])
                continue
        
        return results
    
    def _apply_filters(self, results: List[MemoryResult], options: SearchOptions) -> List[MemoryResult]:
        """Apply additional filtering to results."""
        filtered = results
        
        # Apply tag filter
        if options.filters.get("tags"):
            filter_tags = options.filters["tags"]
            if not isinstance(filter_tags, list):
                filter_tags = [filter_tags]
            filtered = [r for r in filtered if any(tag in r.tags for tag in filter_tags)]
        
        # Apply score threshold
        if options.score_threshold > 0:
            filtered = [r for r in filtered if r.score >= options.score_threshold]
        
        # Sort by score (highest first)
        filtered.sort(key=lambda x: x.score, reverse=True)
        
        return filtered[:options.limit]
    
    def _create_indexes(self):
        """Create performance indexes for common queries."""
        try:
            # Index on actual searchable fields for faster text search
            # Updated to match fields used in _build_search_query
            index_queries = [
                # Text search fields
                f"CREATE INDEX IF NOT EXISTS FOR (n:{self._node_label}) ON (n.title)",
                f"CREATE INDEX IF NOT EXISTS FOR (n:{self._node_label}) ON (n.description)",
                f"CREATE INDEX IF NOT EXISTS FOR (n:{self._node_label}) ON (n.keyword)",
                f"CREATE INDEX IF NOT EXISTS FOR (n:{self._node_label}) ON (n.user_input)",
                f"CREATE INDEX IF NOT EXISTS FOR (n:{self._node_label}) ON (n.bot_response)",
                f"CREATE INDEX IF NOT EXISTS FOR (n:{self._node_label}) ON (n.searchable_text)",  # PR #340
                # Filter fields
                f"CREATE INDEX IF NOT EXISTS FOR (n:{self._node_label}) ON (n.timestamp)",
                f"CREATE INDEX IF NOT EXISTS FOR (n:{self._node_label}) ON (n.namespace)",
                f"CREATE INDEX IF NOT EXISTS FOR (n:{self._node_label}) ON (n.type)"
            ]
            
            for index_query in index_queries:
                try:
                    self._execute_query(index_query, {})
                    backend_logger.debug(f"Created index: {index_query}")
                except Exception as e:
                    backend_logger.warning(f"Failed to create index: {e}", query=index_query)
                    # Continue with other indexes even if one fails
        
        except Exception as e:
            backend_logger.warning(f"Index creation failed: {e}")
            # Don't raise error as indexes are optional for functionality