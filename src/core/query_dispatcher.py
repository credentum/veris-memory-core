#!/usr/bin/env python3
"""
Query dispatcher for routing searches across multiple backends.

This module provides centralized routing and orchestration of search queries
across vector, graph, and key-value backends with intelligent result merging.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Set, Callable
from enum import Enum

from ..interfaces.backend_interface import BackendSearchInterface, SearchOptions, BackendSearchError
from ..interfaces.memory_result import MemoryResult, SearchResultResponse, merge_results, sort_results_by_score
from ..utils.logging_middleware import search_logger, log_backend_timing, TimingCollector
from ..ranking.policy_engine import RankingPolicyEngine, RankingContext, ranking_engine
from ..filters.pre_filter import PreFilterEngine, TimeWindowFilter, FilterCriteria, FilterOperator, pre_filter_engine


class SearchMode(str, Enum):
    """Available search modes for query dispatch."""
    VECTOR = "vector"
    GRAPH = "graph"
    KV = "kv"
    TEXT = "text"  # BM25 text search
    HYBRID = "hybrid"
    AUTO = "auto"


class DispatchPolicy(str, Enum):
    """Policies for backend selection and result merging."""
    PARALLEL = "parallel"  # Run all backends in parallel
    SEQUENTIAL = "sequential"  # Run backends in sequence, stop early if enough results
    FALLBACK = "fallback"  # Try preferred backend first, fallback to others if needed
    SMART = "smart"  # Intelligent selection based on query analysis


class QueryDispatcher:
    """
    Central dispatcher for search queries across multiple backends.
    
    Handles backend registration, query routing, result merging, and
    performance optimization across the Veris Memory storage systems.
    """
    
    def __init__(self):
        """Initialize the query dispatcher."""
        self.backends: Dict[str, BackendSearchInterface] = {}
        self.backend_priorities = {
            "vector": 1,  # Highest priority for semantic search
            "text": 2,    # High priority for keyword/BM25 search
            "graph": 3,   # Medium priority for relationship queries  
            "kv": 4       # Lowest priority, used for direct lookups
        }
        self.default_policy = DispatchPolicy.PARALLEL
        self.timing_collector = TimingCollector()
    
    def register_backend(self, name: str, backend: BackendSearchInterface) -> None:
        """
        Register a search backend.
        
        Args:
            name: Backend name (should match backend.backend_name)
            backend: Backend implementation
        """
        if not isinstance(backend, BackendSearchInterface):
            raise ValueError(f"Backend must implement BackendSearchInterface")
        
        if name != backend.backend_name:
            search_logger.warning(f"Backend name mismatch: {name} != {backend.backend_name}")
        
        self.backends[name] = backend
        search_logger.info(f"Registered backend: {name}")
    
    def unregister_backend(self, name: str) -> bool:
        """
        Unregister a search backend.
        
        Args:
            name: Backend name to remove
            
        Returns:
            True if backend was removed, False if not found
        """
        if name in self.backends:
            del self.backends[name]
            search_logger.info(f"Unregistered backend: {name}")
            return True
        return False
    
    def list_backends(self) -> List[str]:
        """List all registered backend names."""
        return list(self.backends.keys())
    
    def get_backend(self, name: str) -> Optional[BackendSearchInterface]:
        """Get a specific backend by name."""
        return self.backends.get(name)
    
    async def search(
        self,
        query: str,
        options: SearchOptions,
        search_mode: SearchMode
    ) -> SearchResultResponse:
        """
        Simplified search interface for RetrievalCore compatibility.
        
        Args:
            query: Search query string
            options: Search configuration options
            search_mode: Which backends to use
            
        Returns:
            SearchResultResponse with merged and ranked results
        """
        return await self.dispatch_query(
            query=query,
            search_mode=search_mode,
            options=options
        )

    async def search_by_embedding(
        self,
        embedding: List[float],
        options: SearchOptions,
        search_mode: SearchMode
    ) -> SearchResultResponse:
        """
        Search using a pre-computed embedding vector.

        This method is used by HyDE (Hypothetical Document Embeddings) to search
        using the embedding of a hypothetical document rather than generating
        an embedding from the query text.

        Args:
            embedding: Pre-computed embedding vector
            options: Search configuration options
            search_mode: Which backends to use (primarily vector)

        Returns:
            SearchResultResponse with results
        """
        start_time = time.time()

        # For HyDE, we primarily use the vector backend
        vector_backend = self.backends.get("vector")
        if not vector_backend:
            return SearchResultResponse(
                success=False,
                results=[],
                total_count=0,
                search_mode_used=search_mode.value,
                backends_used=[],
                backend_timings={},
                message="Vector backend not available for HyDE search"
            )

        try:
            # Use the vector backend's search_by_embedding method
            results = await vector_backend.search_by_embedding(embedding, options)

            total_time = (time.time() - start_time) * 1000

            return SearchResultResponse(
                success=True,
                results=results,
                total_count=len(results),
                search_mode_used="hyde",
                backends_used=["vector"],
                backend_timings={"vector_hyde": total_time},
                message=f"HyDE search completed with {len(results)} results"
            )

        except Exception as e:
            search_logger.error(f"HyDE search failed: {e}")
            return SearchResultResponse(
                success=False,
                results=[],
                total_count=0,
                search_mode_used=search_mode.value,
                backends_used=[],
                backend_timings={},
                message=f"HyDE search failed: {str(e)}"
            )

    async def dispatch_query(
        self,
        query: str,
        search_mode: SearchMode = SearchMode.HYBRID,
        options: Optional[SearchOptions] = None,
        dispatch_policy: DispatchPolicy = None,
        ranking_policy: Optional[str] = None,
        pre_filters: Optional[List[FilterCriteria]] = None,
        time_window: Optional[TimeWindowFilter] = None
    ) -> SearchResultResponse:
        """
        Dispatch a search query to appropriate backends.
        
        Args:
            query: Search query string
            search_mode: Which backends to use
            options: Search configuration options
            dispatch_policy: How to execute the search across backends
            ranking_policy: Name of ranking policy to use
            pre_filters: List of pre-filtering criteria
            time_window: Time window filter configuration
            
        Returns:
            SearchResultResponse with merged and ranked results
            
        Raises:
            BackendSearchError: If all backends fail
        """
        if not options:
            options = SearchOptions()
        
        if not dispatch_policy:
            dispatch_policy = self.default_policy
        
        start_time = time.time()
        trace_id = f"dispatch_{int(time.time() * 1000)}"
        
        search_logger.info(
            f"Dispatching query",
            query_length=len(query),
            search_mode=search_mode.value,
            dispatch_policy=dispatch_policy.value,
            limit=options.limit,
            trace_id=trace_id
        )
        
        try:
            # Determine which backends to use
            target_backends = self._select_backends(search_mode, query, options)
            
            if not target_backends:
                return SearchResultResponse(
                    success=False,
                    results=[],
                    total_count=0,
                    search_mode_used=search_mode.value,
                    message="No backends available for search",
                    trace_id=trace_id
                )
            
            # Execute search across backends
            all_results, backend_timings, backends_used = await self._execute_search(
                query, options, target_backends, dispatch_policy, trace_id
            )
            
            # Merge results from all backends
            merged_results = merge_results(*all_results.values()) if all_results else []
            
            # Apply pre-filtering if specified
            if pre_filters:
                merged_results = pre_filter_engine.apply_criteria_filter(merged_results, pre_filters)
                search_logger.debug(f"Applied {len(pre_filters)} pre-filters", results_after_filter=len(merged_results))
            
            # Apply time window filtering if specified
            if time_window:
                merged_results = pre_filter_engine.apply_time_window(merged_results, time_window)
                search_logger.debug("Applied time window filter", results_after_filter=len(merged_results))
            
            # Apply ranking policy
            ranked_results = self._rank_results(merged_results, query, search_mode, ranking_policy, trace_id)
            
            # Apply final limit
            final_results = ranked_results[:options.limit]

            # Calculate source breakdown (Issue #311: visibility into hybrid search composition)
            source_breakdown = {}
            for result in final_results:
                source = result.source
                source_breakdown[source] = source_breakdown.get(source, 0) + 1

            total_time = (time.time() - start_time) * 1000

            search_logger.info(
                f"Query dispatch completed",
                total_time_ms=total_time,
                backends_used=list(backends_used),
                total_results=len(merged_results),
                final_results=len(final_results),
                source_breakdown=source_breakdown,
                trace_id=trace_id
            )

            return SearchResultResponse(
                success=True,
                results=final_results,
                total_count=len(merged_results),
                search_mode_used=search_mode.value,
                message=f"Found {len(merged_results)} matching contexts",
                response_time_ms=total_time,
                trace_id=trace_id,
                backend_timings=backend_timings,
                backends_used=list(backends_used),
                source_breakdown=source_breakdown
            )
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            error_msg = f"Query dispatch failed: {str(e)}"
            
            search_logger.error(
                error_msg,
                error=str(e),
                query_length=len(query),
                search_mode=search_mode.value,
                total_time_ms=total_time,
                trace_id=trace_id
            )
            
            return SearchResultResponse(
                success=False,
                results=[],
                total_count=0,
                search_mode_used=search_mode.value,
                message=error_msg,
                response_time_ms=total_time,
                trace_id=trace_id
            )
    
    async def health_check_all_backends(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform health checks on all registered backends.
        
        Returns:
            Dictionary mapping backend names to health status
        """
        health_results = {}
        
        for name, backend in self.backends.items():
            try:
                health = await backend.health_check()
                health_results[name] = {
                    "status": health.status,
                    "response_time_ms": health.response_time_ms,
                    "error_message": health.error_message,
                    "metadata": health.metadata
                }
            except Exception as e:
                health_results[name] = {
                    "status": "error",
                    "response_time_ms": None,
                    "error_message": str(e),
                    "metadata": {"exception_type": type(e).__name__}
                }
        
        return health_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from the timing collector."""
        return {
            "timing_summary": self.timing_collector.get_summary(),
            "registered_backends": self.list_backends(),
            "backend_priorities": self.backend_priorities,
            "default_policy": self.default_policy.value,
            "ranking_policies": ranking_engine.list_policies(),
            "default_ranking_policy": ranking_engine.default_policy_name
        }
    
    def get_available_ranking_policies(self) -> List[str]:
        """Get list of available ranking policies."""
        return ranking_engine.list_policies()
    
    def get_ranking_policy_info(self, policy_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific ranking policy."""
        return ranking_engine.get_policy_info(policy_name)
    
    def set_default_ranking_policy(self, policy_name: str) -> bool:
        """Set the default ranking policy."""
        return ranking_engine.set_default_policy(policy_name)
    
    def register_custom_filter(self, name: str, filter_func: Callable) -> None:
        """Register a custom pre-filter function."""
        pre_filter_engine.register_custom_filter(name, filter_func)
        search_logger.info(f"Registered custom filter in dispatcher: {name}")
    
    def get_filter_capabilities(self) -> Dict[str, Any]:
        """Get information about filtering capabilities."""
        return {
            "time_window_filtering": True,
            "tag_filtering": True,
            "content_type_filtering": True,
            "source_filtering": True,
            "score_filtering": True,
            "namespace_filtering": True,
            "text_filtering": True,
            "custom_filters": list(pre_filter_engine.custom_filters.keys()),
            "supported_operators": [op.value for op in FilterOperator.__members__.values()]
        }
    
    # Private methods
    
    def _select_backends(self, search_mode: SearchMode, query: str, options: SearchOptions) -> List[str]:
        """Select which backends to use based on search mode and query analysis."""
        available_backends = set(self.backends.keys())
        
        if search_mode == SearchMode.VECTOR:
            return ["vector"] if "vector" in available_backends else []
        elif search_mode == SearchMode.GRAPH:
            return ["graph"] if "graph" in available_backends else []
        elif search_mode == SearchMode.KV:
            return ["kv"] if "kv" in available_backends else []
        elif search_mode == SearchMode.TEXT:
            return ["text"] if "text" in available_backends else []
        elif search_mode == SearchMode.HYBRID:
            # Use all available backends
            return sorted(available_backends, key=lambda x: self.backend_priorities.get(x, 999))
        elif search_mode == SearchMode.AUTO:
            # Intelligent backend selection based on query analysis
            return self._analyze_query_for_backends(query, options, available_backends)
        else:
            return []
    
    def _analyze_query_for_backends(self, query: str, options: SearchOptions, available: Set[str]) -> List[str]:
        """Analyze query to determine optimal backend selection."""
        selected = []
        
        # Key-based queries favor KV
        if ":" in query or any(query.startswith(prefix) for prefix in ["state:", "scratchpad:", "cache:"]):
            if "kv" in available:
                selected.append("kv")
        
        # Relationship queries favor graph
        if any(word in query.lower() for word in ["related", "connected", "linked", "relationship", "depends"]):
            if "graph" in available:
                selected.append("graph")
        
        # Exact keyword/phrase queries favor text search
        if any(phrase in query for phrase in ['"', "'", "exact:"]) or \
           any(keyword in query.lower() for keyword in ["find", "search", "contains", "keyword"]):
            if "text" in available:
                selected.append("text")
        
        # Semantic queries favor vector
        if len(query.split()) > 1 and not selected:  # Multi-word queries are often semantic
            if "vector" in available:
                selected.append("vector")
        
        # If no specific preference, use hybrid approach
        if not selected:
            selected = sorted(available, key=lambda x: self.backend_priorities.get(x, 999))
        
        return selected
    
    async def _execute_search(
        self,
        query: str,
        options: SearchOptions,
        target_backends: List[str],
        policy: DispatchPolicy,
        trace_id: str
    ) -> tuple[Dict[str, List[MemoryResult]], Dict[str, float], Set[str]]:
        """Execute search across selected backends according to policy."""
        all_results = {}
        backend_timings = {}
        backends_used = set()
        
        if policy == DispatchPolicy.PARALLEL:
            # Run all backends in parallel
            tasks = []
            for backend_name in target_backends:
                if backend_name in self.backends:
                    task = self._search_single_backend(
                        backend_name, query, options, trace_id
                    )
                    tasks.append((backend_name, task))
            
            # Wait for all results
            for backend_name, task in tasks:
                try:
                    results, timing = await task
                    all_results[backend_name] = results
                    backend_timings[backend_name] = timing
                    backends_used.add(backend_name)
                    # Issue #311: Log backend results for debugging hybrid search
                    search_logger.info(
                        "Backend '%s' returned %d results in %.1fms",
                        backend_name,
                        len(results),
                        timing,
                        extra={
                            "backend": backend_name,
                            "result_count": len(results),
                            "timing_ms": timing,
                            "trace_id": trace_id
                        }
                    )
                except Exception as e:
                    search_logger.warning(f"Backend {backend_name} failed: {e}")
                    backend_timings[backend_name] = 0.0
        
        elif policy == DispatchPolicy.SEQUENTIAL:
            # Run backends in sequence, stop early if we have enough results
            total_results = 0
            for backend_name in target_backends:
                if backend_name in self.backends:
                    try:
                        results, timing = await self._search_single_backend(
                            backend_name, query, options, trace_id
                        )
                        all_results[backend_name] = results
                        backend_timings[backend_name] = timing
                        backends_used.add(backend_name)
                        total_results += len(results)
                        
                        # Stop early if we have enough results
                        if total_results >= options.limit:
                            break
                            
                    except Exception as e:
                        search_logger.warning(f"Backend {backend_name} failed: {e}")
                        backend_timings[backend_name] = 0.0
        
        elif policy == DispatchPolicy.FALLBACK:
            # Try backends in priority order, stop at first success
            for backend_name in target_backends:
                if backend_name in self.backends:
                    try:
                        results, timing = await self._search_single_backend(
                            backend_name, query, options, trace_id
                        )
                        all_results[backend_name] = results
                        backend_timings[backend_name] = timing
                        backends_used.add(backend_name)
                        
                        # Stop at first successful backend
                        if results:
                            break
                            
                    except Exception as e:
                        search_logger.warning(f"Backend {backend_name} failed, trying next: {e}")
                        backend_timings[backend_name] = 0.0
        
        elif policy == DispatchPolicy.SMART:
            # Implement smart policy (for now, same as parallel)
            return await self._execute_search(query, options, target_backends, DispatchPolicy.PARALLEL, trace_id)
        
        return all_results, backend_timings, backends_used
    
    async def _search_single_backend(
        self,
        backend_name: str,
        query: str,
        options: SearchOptions,
        trace_id: str
    ) -> tuple[List[MemoryResult], float]:
        """Search a single backend and return results with timing."""
        backend = self.backends[backend_name]
        start_time = time.time()
        
        try:
            results = await backend.search(query, options)
            timing = (time.time() - start_time) * 1000
            
            self.timing_collector.record_timing(
                f"backend_{backend_name}",
                timing,
                query_length=len(query),
                result_count=len(results),
                trace_id=trace_id
            )
            
            return results, timing
            
        except Exception as e:
            timing = (time.time() - start_time) * 1000
            search_logger.error(f"Backend {backend_name} search failed: {e}", trace_id=trace_id)
            raise BackendSearchError(backend_name, str(e), e)
    
    def _rank_results(
        self,
        results: List[MemoryResult],
        query: str,
        search_mode: SearchMode,
        ranking_policy: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> List[MemoryResult]:
        """Apply ranking to merged results using specified or default policy."""
        if not results:
            return results
        
        # Create ranking context
        ranking_context = RankingContext(
            query=query,
            search_mode=search_mode.value,
            timestamp=time.time(),
            custom_features={
                "trace_id": trace_id
            }
        )
        
        try:
            # Use ranking engine to apply policy
            ranked_results = ranking_engine.rank_results(
                results, 
                ranking_context, 
                ranking_policy
            )
            
            search_logger.debug(
                f"Applied ranking policy",
                policy=ranking_policy or "default",
                input_count=len(results),
                output_count=len(ranked_results),
                trace_id=trace_id
            )
            
            return ranked_results
            
        except Exception as e:
            search_logger.warning(
                f"Ranking failed, using fallback",
                error=str(e),
                policy=ranking_policy,
                trace_id=trace_id
            )
            
            # Fallback to simple score-based ranking
            return sort_results_by_score(results, descending=True)