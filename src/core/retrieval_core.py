#!/usr/bin/env python3
"""
Unified retrieval core for both API and MCP interfaces.

This module provides a single point of truth for context retrieval,
ensuring identical behavior across API and MCP endpoints while
properly tracking backend timings and utilizing the hybrid search architecture.

S3 Paraphrase Robustness Improvements:
- Phase 2: Multi-Query Expansion (MQE) integration
- Phase 3: Search enhancements integration
- Phase 4: Query normalization integration
"""

import logging
import os
from typing import Dict, List, Any, Optional

from ..interfaces.backend_interface import SearchOptions
from ..interfaces.memory_result import SearchResultResponse, MemoryResult
from ..core.query_dispatcher import QueryDispatcher, SearchMode
from ..utils.logging_middleware import search_logger

# Import MQE wrapper (Phase 2)
try:
    from .mqe_wrapper import get_mqe_wrapper, MQESearchResult

    MQE_AVAILABLE = True
except ImportError:
    MQE_AVAILABLE = False
    get_mqe_wrapper = None

# Import search enhancements (Phase 3)
try:
    from ..mcp_server.search_enhancements import apply_search_enhancements, is_technical_query

    SEARCH_ENHANCEMENTS_AVAILABLE = True
except ImportError:
    SEARCH_ENHANCEMENTS_AVAILABLE = False
    apply_search_enhancements = None
    is_technical_query = None

# Import query normalizer (Phase 4)
try:
    from .query_normalizer import get_query_normalizer

    QUERY_NORMALIZER_AVAILABLE = True
except ImportError:
    QUERY_NORMALIZER_AVAILABLE = False
    get_query_normalizer = None

# Import HyDE generator (Phase 5)
try:
    from .hyde_generator import get_hyde_generator, HyDEResult

    HYDE_AVAILABLE = True
except ImportError:
    HYDE_AVAILABLE = False
    get_hyde_generator = None
    HyDEResult = None


logger = logging.getLogger(__name__)

# Configuration from environment
ENABLE_MQE = os.getenv("MQE_ENABLED", "true").lower() == "true"
ENABLE_SEARCH_ENHANCEMENTS = os.getenv("SEARCH_ENHANCEMENTS_ENABLED", "true").lower() == "true"
ENABLE_QUERY_NORMALIZATION = os.getenv("QUERY_NORMALIZATION_ENABLED", "true").lower() == "true"
ENABLE_HYDE = os.getenv("HYDE_ENABLED", "false").lower() == "true"  # Disabled by default


class RetrievalCore:
    """
    Unified retrieval engine for both API and MCP interfaces.
    
    This ensures both API and MCP return identical results for identical inputs
    while properly utilizing the backend timing infrastructure.
    """
    
    def __init__(self, query_dispatcher: QueryDispatcher):
        """
        Initialize the unified retrieval core.
        
        Args:
            query_dispatcher: The configured query dispatcher with all backends
        """
        self.dispatcher = query_dispatcher
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        search_mode: str = "hybrid",
        context_type: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0
    ) -> SearchResultResponse:
        """
        Execute unified search across all configured backends.

        Includes S3 Paraphrase Robustness improvements:
        - Phase 2: Multi-Query Expansion (MQE) for paraphrase consistency
        - Phase 3: Search enhancements for result boosting
        - Phase 4: Query normalization for semantic consistency

        Args:
            query: Search query text
            limit: Maximum number of results to return
            search_mode: Search mode ("vector", "graph", "kv", "hybrid", "auto")
            context_type: Optional context type filter
            metadata_filters: Optional metadata filters
            score_threshold: Minimum score threshold for results

        Returns:
            SearchResultResponse with results and backend timing information

        Raises:
            Exception: If search fails across all backends
        """
        try:
            # PHASE 4: Query Normalization
            effective_query = query
            query_normalization_applied = False

            if (
                ENABLE_QUERY_NORMALIZATION
                and QUERY_NORMALIZER_AVAILABLE
                and get_query_normalizer is not None
            ):
                try:
                    normalizer = get_query_normalizer()
                    normalized = normalizer.normalize(query)
                    if normalized.confidence > 0.5 and normalized.normalized != query:
                        effective_query = normalized.normalized
                        query_normalization_applied = True
                        logger.info(
                            f"Query normalized: '{query[:30]}...' -> '{effective_query[:30]}...' "
                            f"(confidence={normalized.confidence:.2f}, intent={normalized.intent.value})"
                        )
                except Exception as norm_error:
                    logger.warning(f"Query normalization failed: {norm_error}")

            # Convert search_mode string to SearchMode enum
            if search_mode not in SearchMode.__members__:
                logger.warning(f"Invalid search_mode '{search_mode}', defaulting to 'hybrid'")
                search_mode = "hybrid"

            search_mode_enum = SearchMode(search_mode)

            # Build search options
            search_options = SearchOptions(
                limit=limit,
                score_threshold=score_threshold,
                namespace=None,  # Could be derived from context_type if needed
                filters=metadata_filters or {}
            )

            # Add context_type to filters if specified
            if context_type:
                search_options.filters["type"] = context_type

            logger.info(
                f"RetrievalCore executing search: query_length={len(effective_query)}, "
                f"mode={search_mode}, limit={limit}, score_threshold={score_threshold}, "
                f"hyde_enabled={ENABLE_HYDE and HYDE_AVAILABLE}, "
                f"mqe_enabled={ENABLE_MQE and MQE_AVAILABLE}"
            )

            # PHASE 5: HyDE (Hypothetical Document Embeddings)
            # If enabled, generate hypothetical doc and search with its embedding
            hyde_used = False
            search_response = None
            if ENABLE_HYDE and HYDE_AVAILABLE and get_hyde_generator is not None:
                try:
                    hyde_generator = get_hyde_generator()
                    if hyde_generator.config.enabled:
                        hyde_result = await hyde_generator.generate_hyde_embedding(effective_query)

                        if hyde_result.embedding and not hyde_result.error:
                            # Search using hypothetical doc embedding
                            search_response = await self.dispatcher.search_by_embedding(
                                embedding=hyde_result.embedding,
                                options=search_options,
                                search_mode=search_mode_enum
                            )
                            hyde_used = True
                            logger.info(
                                f"HyDE search completed: {len(search_response.results)} results, "
                                f"cache_hit={hyde_result.cache_hit}, "
                                f"time={hyde_result.generation_time_ms:.2f}ms"
                            )
                        elif hyde_result.error:
                            logger.warning(f"HyDE generation failed: {hyde_result.error}")
                except Exception as hyde_error:
                    logger.warning(f"HyDE search failed, falling back to MQE: {hyde_error}")

            # PHASE 2: Multi-Query Expansion (fallback if HyDE not used)
            mqe_used = False
            if not hyde_used and ENABLE_MQE and MQE_AVAILABLE and get_mqe_wrapper is not None:
                try:
                    mqe_wrapper = get_mqe_wrapper()
                    if mqe_wrapper.is_available and mqe_wrapper.config.enabled:
                        # Define search function for MQE to wrap
                        async def single_search(q: str, lim: int) -> List[Dict[str, Any]]:
                            response = await self.dispatcher.search(
                                query=q,
                                options=SearchOptions(
                                    limit=lim,
                                    score_threshold=score_threshold,
                                    filters=search_options.filters
                                ),
                                search_mode=search_mode_enum
                            )
                            # Convert MemoryResult to dict
                            return [self._memory_result_to_dict(r) for r in response.results]

                        # Execute MQE search
                        mqe_result = await mqe_wrapper.search_with_expansion(
                            query=effective_query,
                            search_func=single_search,
                            limit=limit
                        )

                        # Convert results back to MemoryResult
                        results = [self._dict_to_memory_result(d) for d in mqe_result.results]

                        mqe_used = True
                        logger.info(
                            f"MQE search completed: {len(results)} results from "
                            f"{len(mqe_result.paraphrases_used)} paraphrases in "
                            f"{mqe_result.search_time_ms:.2f}ms"
                        )

                        # Create response with MQE metadata
                        search_response = SearchResultResponse(
                            success=True,
                            results=results,
                            total_count=mqe_result.unique_docs_found,
                            search_mode_used=search_mode,
                            backends_used=["vector"],  # MQE primarily uses vector
                            backend_timings={"mqe": mqe_result.search_time_ms},
                            message=f"MQE search with {len(mqe_result.paraphrases_used)} paraphrases"
                        )
                except Exception as mqe_error:
                    logger.warning(f"MQE search failed, falling back to standard: {mqe_error}")

            # Standard search if neither HyDE nor MQE used
            if not hyde_used and not mqe_used:
                search_response = await self.dispatcher.search(
                    query=effective_query,
                    options=search_options,
                    search_mode=search_mode_enum
                )

            # PHASE 3: Apply Search Enhancements
            if (
                ENABLE_SEARCH_ENHANCEMENTS
                and SEARCH_ENHANCEMENTS_AVAILABLE
                and apply_search_enhancements is not None
                and search_response.results
            ):
                try:
                    technical_query = (
                        is_technical_query(effective_query)
                        if is_technical_query is not None
                        else False
                    )

                    # Convert results to dict format for enhancements
                    results_as_dicts = [
                        self._memory_result_to_enhancement_dict(r)
                        for r in search_response.results
                    ]

                    enhanced_dicts = apply_search_enhancements(
                        results=results_as_dicts,
                        query=effective_query,
                        enable_exact_match=True,
                        enable_type_weighting=True,
                        enable_recency_decay=True,
                        enable_technical_boost=technical_query
                    )

                    # Convert back to MemoryResult
                    search_response.results = [
                        self._enhancement_dict_to_memory_result(d, original)
                        for d, original in zip(enhanced_dicts, search_response.results)
                    ]

                    logger.debug(f"Applied search enhancements to {len(enhanced_dicts)} results")

                except Exception as enhance_error:
                    logger.warning(f"Search enhancements failed: {enhance_error}")

            logger.info(
                f"RetrievalCore search completed: results={len(search_response.results)}, "
                f"backends_used={search_response.backends_used}, "
                f"hyde_used={hyde_used}, mqe_used={mqe_used}, query_normalized={query_normalization_applied}"
            )

            return search_response

        except Exception as e:
            logger.error(f"RetrievalCore search failed: {e}")
            raise

    def _memory_result_to_dict(self, result: MemoryResult) -> Dict[str, Any]:
        """Convert MemoryResult to dict for MQE processing."""
        return {
            "id": result.id,
            "score": result.score,
            "text": result.text,
            "type": result.type.value if hasattr(result.type, 'value') else str(result.type),
            "metadata": result.metadata,
            "title": getattr(result, 'title', ''),
            "tags": getattr(result, 'tags', []),
            "source": result.source,
        }

    def _dict_to_memory_result(self, d: Dict[str, Any]) -> MemoryResult:
        """Convert dict back to MemoryResult after MQE processing."""
        from ..interfaces.memory_result import ContextType

        # Handle type conversion
        type_value = d.get("type", "general")
        try:
            context_type = ContextType(type_value) if isinstance(type_value, str) else type_value
        except (ValueError, KeyError):
            context_type = ContextType.GENERAL

        return MemoryResult(
            id=d.get("id", ""),
            score=d.get("score", 0.0),
            text=d.get("text", ""),
            type=context_type,
            metadata={
                **d.get("metadata", {}),
                "mqe_scores": d.get("mqe_scores", []),
                "mqe_queries": d.get("mqe_queries", []),
                "source_query": d.get("source_query", ""),
            },
            source=d.get("source", "vector"),
        )

    def _memory_result_to_enhancement_dict(self, result: MemoryResult) -> Dict[str, Any]:
        """Convert MemoryResult to dict format expected by search enhancements."""
        return {
            "id": result.id,
            "score": result.score,
            "payload": {
                "content": result.text,
                "type": result.type.value if hasattr(result.type, 'value') else str(result.type),
                "metadata": result.metadata,
                "title": getattr(result, 'title', ''),
                "tags": getattr(result, 'tags', []),
            }
        }

    def _enhancement_dict_to_memory_result(
        self, d: Dict[str, Any], original: MemoryResult
    ) -> MemoryResult:
        """Convert enhanced dict back to MemoryResult, preserving original data."""
        # Update score from enhancement
        enhanced_score = d.get("enhanced_score", d.get("score", original.score))

        # Create new result with enhanced score and boost metadata
        return MemoryResult(
            id=original.id,
            score=enhanced_score,
            text=original.text,
            type=original.type,
            metadata={
                **original.metadata,
                "original_score": original.score,
                "score_boosts": d.get("score_boosts", {}),
            },
            source=original.source,
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of all registered backends.
        
        Returns:
            Dict containing health status of all backends
        """
        try:
            backend_health = {}
            
            for backend_name, backend in self.dispatcher.backends.items():
                try:
                    health_status = await backend.health_check()
                    backend_health[backend_name] = {
                        "status": health_status.status,
                        "response_time_ms": health_status.response_time_ms,
                        "error_message": health_status.error_message,
                        "metadata": health_status.metadata
                    }
                except Exception as e:
                    backend_health[backend_name] = {
                        "status": "unhealthy",
                        "response_time_ms": -1,
                        "error_message": str(e),
                        "metadata": {"health_check_failed": True}
                    }
            
            # Determine overall health
            healthy_backends = sum(1 for health in backend_health.values() if health["status"] == "healthy")
            total_backends = len(backend_health)
            
            overall_status = "healthy" if healthy_backends == total_backends else \
                           "degraded" if healthy_backends > 0 else "unhealthy"
            
            return {
                "overall_status": overall_status,
                "backends": backend_health,
                "healthy_backends": healthy_backends,
                "total_backends": total_backends
            }
            
        except Exception as e:
            logger.error(f"RetrievalCore health check failed: {e}")
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "backends": {},
                "healthy_backends": 0,
                "total_backends": 0
            }


# Global instance to be shared between API and MCP
_retrieval_core_instance: Optional[RetrievalCore] = None


def get_retrieval_core() -> Optional[RetrievalCore]:
    """Get the global retrieval core instance."""
    return _retrieval_core_instance


def set_retrieval_core(retrieval_core: RetrievalCore) -> None:
    """Set the global retrieval core instance."""
    global _retrieval_core_instance
    _retrieval_core_instance = retrieval_core
    logger.info("RetrievalCore instance set globally")


def initialize_retrieval_core(query_dispatcher: QueryDispatcher) -> RetrievalCore:
    """
    Initialize and set the global retrieval core instance.
    
    Args:
        query_dispatcher: Configured query dispatcher with all backends
        
    Returns:
        The initialized RetrievalCore instance
    """
    retrieval_core = RetrievalCore(query_dispatcher)
    set_retrieval_core(retrieval_core)
    logger.info("RetrievalCore initialized and set globally")
    return retrieval_core