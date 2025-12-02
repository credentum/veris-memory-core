"""
Multi-Query Expansion (MQE) wrapper for retrieval pipeline integration.

This module wraps the existing MultiQueryExpander to integrate it into
the retrieval pipeline, enabling paraphrase robustness by searching
multiple query variations and aggregating results.

This addresses the S3-Paraphrase-Robustness issue where different
phrasings of the same query would return completely different results.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Awaitable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MQEConfig:
    """Configuration for Multi-Query Expansion."""

    enabled: bool = True
    num_paraphrases: int = 2
    apply_field_boosts: bool = True
    parallel_search: bool = True
    max_concurrent_searches: int = 3
    aggregation_strategy: str = "max_score"  # "max_score", "average", "weighted"
    metrics_enabled: bool = True


@dataclass
class MQESearchResult:
    """Result of MQE search with metadata."""

    results: List[Dict[str, Any]]
    paraphrases_used: List[str]
    search_time_ms: float
    unique_docs_found: int
    aggregation_strategy: str
    fallback_used: bool = False
    error_message: Optional[str] = None


class MQERetrievalWrapper:
    """
    Wraps Multi-Query Expansion for use in the retrieval pipeline.

    Instead of searching with a single query, this wrapper:
    1. Generates paraphrases of the original query
    2. Executes searches for each paraphrase (optionally in parallel)
    3. Aggregates results, keeping the max score for duplicate documents
    4. Optionally applies field boosts for lexical relevance

    This improves paraphrase robustness by ensuring that semantically
    equivalent queries return consistent results.
    """

    def __init__(self, config: Optional[MQEConfig] = None):
        """
        Initialize the MQE retrieval wrapper.

        Args:
            config: Configuration options. If None, uses defaults from environment.
        """
        if config is None:
            config = MQEConfig(
                enabled=os.getenv("MQE_ENABLED", "true").lower() == "true",
                num_paraphrases=int(os.getenv("MQE_NUM_PARAPHRASES", "2")),
                apply_field_boosts=os.getenv("MQE_APPLY_FIELD_BOOSTS", "true").lower() == "true",
            )

        self.config = config
        self._expander = None
        self._field_processor = None
        self._metrics = {
            "total_searches": 0,
            "mq_searches": 0,
            "single_query_fallbacks": 0,
            "average_paraphrases_used": 0.0,
            "average_unique_docs": 0.0,
            "average_search_time_ms": 0.0,
            "aggregation_merges": 0,
        }

        self._initialize_components()

        logger.info(
            f"MQERetrievalWrapper initialized: enabled={config.enabled}, "
            f"num_paraphrases={config.num_paraphrases}, field_boosts={config.apply_field_boosts}"
        )

    def _initialize_components(self) -> None:
        """Initialize MQE components with lazy loading."""
        try:
            from ..storage.query_expansion import MultiQueryExpander, FieldBoostProcessor

            self._expander = MultiQueryExpander()
            if self.config.apply_field_boosts:
                self._field_processor = FieldBoostProcessor()
            logger.info("MQE components initialized successfully")
        except ImportError as e:
            logger.warning(f"MQE components not available: {e}")
            self._expander = None
            self._field_processor = None

    @property
    def is_available(self) -> bool:
        """Check if MQE components are available."""
        return self._expander is not None

    async def search_with_expansion(
        self,
        query: str,
        search_func: Callable[[str, int], Awaitable[List[Dict[str, Any]]]],
        limit: int = 10,
    ) -> MQESearchResult:
        """
        Execute search with multi-query expansion.

        Args:
            query: Original search query
            search_func: Async function to perform a single search (query, limit) -> results
            limit: Maximum number of results to return

        Returns:
            MQESearchResult with aggregated results and metadata
        """
        start_time = time.time()
        self._metrics["total_searches"] += 1

        # Check if MQE should be used
        if not self.config.enabled or not self.is_available:
            return await self._fallback_search(query, search_func, limit, start_time)

        try:
            # Generate paraphrases
            paraphrases = self._expander.generate_paraphrases(
                query, num_paraphrases=self.config.num_paraphrases
            )

            logger.debug(
                f"Generated {len(paraphrases)} paraphrases for query: {query[:50]}..."
            )

            # Execute searches (parallel or sequential)
            if self.config.parallel_search:
                all_results = await self._parallel_search(paraphrases, search_func, limit)
            else:
                all_results = await self._sequential_search(paraphrases, search_func, limit)

            # Aggregate results
            aggregated = self._aggregate_results(all_results, paraphrases)

            # Apply field boosts if enabled
            if self.config.apply_field_boosts and self._field_processor:
                aggregated = self._field_processor.process_results(aggregated)

            # Sort by score and limit
            aggregated.sort(key=lambda x: x.get("score", 0), reverse=True)
            final_results = aggregated[:limit]

            search_time_ms = (time.time() - start_time) * 1000

            # Update metrics
            self._update_metrics(
                paraphrases_count=len(paraphrases),
                unique_docs=len(aggregated),
                search_time_ms=search_time_ms,
                is_mq=True,
            )

            logger.info(
                f"MQE search completed: {len(final_results)} results from "
                f"{len(paraphrases)} paraphrases in {search_time_ms:.2f}ms"
            )

            return MQESearchResult(
                results=final_results,
                paraphrases_used=paraphrases,
                search_time_ms=search_time_ms,
                unique_docs_found=len(aggregated),
                aggregation_strategy=self.config.aggregation_strategy,
            )

        except Exception as e:
            logger.error(f"MQE search failed, falling back to single query: {e}")
            return await self._fallback_search(query, search_func, limit, start_time, str(e))

    async def _parallel_search(
        self,
        queries: List[str],
        search_func: Callable[[str, int], Awaitable[List[Dict[str, Any]]]],
        limit: int,
    ) -> List[List[Dict[str, Any]]]:
        """Execute searches in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_searches)

        async def limited_search(q: str) -> List[Dict[str, Any]]:
            async with semaphore:
                try:
                    return await search_func(q, limit)
                except Exception as e:
                    logger.warning(f"Search failed for paraphrase '{q[:30]}...': {e}")
                    return []

        tasks = [limited_search(q) for q in queries]
        return await asyncio.gather(*tasks)

    async def _sequential_search(
        self,
        queries: List[str],
        search_func: Callable[[str, int], Awaitable[List[Dict[str, Any]]]],
        limit: int,
    ) -> List[List[Dict[str, Any]]]:
        """Execute searches sequentially."""
        results = []
        for q in queries:
            try:
                result = await search_func(q, limit)
                results.append(result)
            except Exception as e:
                logger.warning(f"Search failed for paraphrase '{q[:30]}...': {e}")
                results.append([])
        return results

    def _aggregate_results(
        self,
        all_results: List[List[Dict[str, Any]]],
        paraphrases: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Aggregate results from multiple queries.

        Uses max-score aggregation: for each unique document, keep the
        highest score across all queries. This ensures that if a document
        is highly relevant to any paraphrase, it will be ranked highly.
        """
        # Track unique documents by ID with their best score
        doc_map: Dict[str, Dict[str, Any]] = {}

        for query_idx, results in enumerate(all_results):
            source_query = paraphrases[query_idx] if query_idx < len(paraphrases) else "unknown"

            for result in results:
                doc_id = result.get("id")
                if not doc_id:
                    continue

                current_score = result.get("score", 0.0)

                if doc_id not in doc_map:
                    # First time seeing this document
                    doc_map[doc_id] = {
                        **result,
                        "source_query": source_query,
                        "mqe_scores": [current_score],
                        "mqe_queries": [source_query],
                    }
                else:
                    # Document already seen - update based on aggregation strategy
                    existing = doc_map[doc_id]
                    existing["mqe_scores"].append(current_score)
                    existing["mqe_queries"].append(source_query)

                    if self.config.aggregation_strategy == "max_score":
                        if current_score > existing.get("score", 0):
                            # Update score and source query
                            existing["score"] = current_score
                            existing["source_query"] = source_query
                    elif self.config.aggregation_strategy == "average":
                        existing["score"] = sum(existing["mqe_scores"]) / len(
                            existing["mqe_scores"]
                        )

                    self._metrics["aggregation_merges"] += 1

        return list(doc_map.values())

    async def _fallback_search(
        self,
        query: str,
        search_func: Callable[[str, int], Awaitable[List[Dict[str, Any]]]],
        limit: int,
        start_time: float,
        error_message: Optional[str] = None,
    ) -> MQESearchResult:
        """Execute fallback single-query search."""
        try:
            results = await search_func(query, limit)

            # Add MQE metadata to results for consistency
            for result in results:
                result["source_query"] = query
                result["mqe_scores"] = [result.get("score", 0)]
                result["mqe_queries"] = [query]

            search_time_ms = (time.time() - start_time) * 1000

            self._update_metrics(
                paraphrases_count=1,
                unique_docs=len(results),
                search_time_ms=search_time_ms,
                is_mq=False,
            )

            return MQESearchResult(
                results=results,
                paraphrases_used=[query],
                search_time_ms=search_time_ms,
                unique_docs_found=len(results),
                aggregation_strategy="single_query",
                fallback_used=True,
                error_message=error_message,
            )

        except Exception as e:
            search_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Fallback search also failed: {e}")

            return MQESearchResult(
                results=[],
                paraphrases_used=[query],
                search_time_ms=search_time_ms,
                unique_docs_found=0,
                aggregation_strategy="failed",
                fallback_used=True,
                error_message=str(e),
            )

    def _update_metrics(
        self,
        paraphrases_count: int,
        unique_docs: int,
        search_time_ms: float,
        is_mq: bool,
    ) -> None:
        """Update running metrics."""
        total = self._metrics["total_searches"]

        if is_mq:
            self._metrics["mq_searches"] += 1
        else:
            self._metrics["single_query_fallbacks"] += 1

        # Running averages
        def update_avg(key: str, new_value: float) -> None:
            current = self._metrics[key]
            self._metrics[key] = (current * (total - 1) + new_value) / total if total > 0 else new_value

        update_avg("average_paraphrases_used", paraphrases_count)
        update_avg("average_unique_docs", unique_docs)
        update_avg("average_search_time_ms", search_time_ms)

    def get_metrics(self) -> Dict[str, Any]:
        """Get MQE metrics."""
        total = self._metrics["total_searches"]
        return {
            **self._metrics,
            "mq_search_rate": self._metrics["mq_searches"] / total if total > 0 else 0.0,
            "fallback_rate": self._metrics["single_query_fallbacks"] / total if total > 0 else 0.0,
            "config": {
                "enabled": self.config.enabled,
                "num_paraphrases": self.config.num_paraphrases,
                "apply_field_boosts": self.config.apply_field_boosts,
                "parallel_search": self.config.parallel_search,
                "aggregation_strategy": self.config.aggregation_strategy,
            },
            "components_available": self.is_available,
        }

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._metrics = {
            "total_searches": 0,
            "mq_searches": 0,
            "single_query_fallbacks": 0,
            "average_paraphrases_used": 0.0,
            "average_unique_docs": 0.0,
            "average_search_time_ms": 0.0,
            "aggregation_merges": 0,
        }


# Global instance
_mqe_wrapper: Optional[MQERetrievalWrapper] = None


def get_mqe_wrapper() -> MQERetrievalWrapper:
    """
    Get or create the global MQE wrapper.

    Returns:
        Global MQERetrievalWrapper instance
    """
    global _mqe_wrapper
    if _mqe_wrapper is None:
        _mqe_wrapper = MQERetrievalWrapper()
    return _mqe_wrapper


def reset_mqe_wrapper() -> None:
    """Reset the global MQE wrapper (useful for testing)."""
    global _mqe_wrapper
    _mqe_wrapper = None
