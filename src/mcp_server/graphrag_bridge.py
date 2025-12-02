#!/usr/bin/env python3
"""
GraphRAG Bridge for MCP Server Integration.

This module provides a simple bridge connecting the existing GraphRAG
implementation to the MCP server tools, enabling enhanced graph reasoning
capabilities without requiring architectural changes.
"""

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# Use relative import instead of sys.path manipulation

try:
    # Import from the correct relative path
    import sys

    # Only add to path if GraphRAG module not found in current environment
    try:
        from integrations.graphrag_integration import GraphRAGIntegration, GraphRAGResult
    except ImportError:
        # Fallback: try adding parent src directory (only as last resort)
        graphrag_path = Path(__file__).parent.parent.parent.parent / "src"
        if graphrag_path.exists() and str(graphrag_path) not in sys.path:
            sys.path.insert(0, str(graphrag_path))
            from integrations.graphrag_integration import GraphRAGIntegration, GraphRAGResult

            # Remove from path after import to avoid pollution
            sys.path.remove(str(graphrag_path))
except ImportError as e:
    logging.warning(f"GraphRAG integration not available: {e}")
    GraphRAGIntegration = None
    GraphRAGResult = None

logger = logging.getLogger(__name__)

# Thread pool size constant - configurable via environment variable
MAX_THREAD_POOL_SIZE = int(os.getenv("GRAPHRAG_THREAD_POOL_SIZE", "4"))

# Thread lock for singleton initialization
_bridge_lock = threading.Lock()


class GraphRAGBridge:
    """Bridge class for integrating GraphRAG capabilities with MCP server."""

    def __init__(self, config_path: str = ".ctxrc.yaml", verbose: bool = False):
        """Initialize GraphRAG bridge.

        Args:
            config_path: Path to GraphRAG configuration file
            verbose: Enable verbose logging
        """
        self.config_path = config_path
        self.verbose = verbose
        self._graphrag: Optional[GraphRAGIntegration] = None
        self._initialized = False
        self._init_lock = threading.Lock()

    @property
    def graphrag(self) -> Optional[GraphRAGIntegration]:
        """Lazy-loaded GraphRAG integration instance."""
        if not self._initialized:
            with self._init_lock:
                # Double-check locking pattern
                if not self._initialized:
                    self._initialize_graphrag()
        return self._graphrag

    def _initialize_graphrag(self) -> None:
        """Initialize GraphRAG integration instance with proper error handling."""
        if GraphRAGIntegration is None:
            logger.warning("GraphRAG integration class not available - features will be disabled")
            self._initialized = True
            return

        try:
            logger.info(f"Initializing GraphRAG with config: {self.config_path}")
            self._graphrag = GraphRAGIntegration(config_path=self.config_path, verbose=self.verbose)

            # Validate GraphRAG instance has required methods
            required_methods = ["connect", "_graph_neighborhood", "_extract_reasoning_path"]
            missing_methods = [
                method for method in required_methods if not hasattr(self._graphrag, method)
            ]

            if missing_methods:
                logger.warning(
                    f"GraphRAG instance missing methods: {missing_methods} - "
                    "some features will be limited"
                )

            # Test connection if available
            if hasattr(self._graphrag, "connect"):
                try:
                    success = self._graphrag.connect()
                    if not success:
                        logger.warning("GraphRAG connection test failed - using fallback mode")
                        self._graphrag = None
                except Exception as conn_e:
                    logger.error(
                        f"GraphRAG connection error: {conn_e} - disabling GraphRAG features"
                    )
                    self._graphrag = None

            self._initialized = True
            if self._graphrag:
                logger.info("GraphRAG bridge initialized successfully")
            else:
                logger.info("GraphRAG bridge initialized in fallback mode")

        except ImportError as e:
            logger.error(f"GraphRAG import failed: {e} - GraphRAG features disabled")
            self._graphrag = None
            self._initialized = True
        except ValueError as e:
            logger.error(f"GraphRAG configuration error: {e} - GraphRAG features disabled")
            self._graphrag = None
            self._initialized = True
        except Exception as e:
            logger.error(
                f"Unexpected error initializing GraphRAG: {e} - GraphRAG features disabled"
            )
            self._graphrag = None
            self._initialized = True

    async def enhance_search_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        max_hops: int = 2,
        include_reasoning: bool = True,
    ) -> Dict[str, Any]:
        """Enhance search results with GraphRAG reasoning.

        Args:
            query: Original search query
            results: Initial search results to enhance
            max_hops: Maximum hops for graph traversal
            include_reasoning: Include reasoning paths in results

        Returns:
            Enhanced results with GraphRAG analysis
        """
        if not self.graphrag:
            logger.warning("GraphRAG not available - returning original results")
            return {
                "enhanced_results": results,
                "graphrag_analysis": None,
                "reasoning_paths": [],
                "enhancement_applied": False,
            }

        try:
            # Run GraphRAG enhancement in thread pool to avoid blocking
            import concurrent.futures

            loop = asyncio.get_event_loop()

            # Use a ThreadPoolExecutor with limited size
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_THREAD_POOL_SIZE
            ) as executor:
                enhanced_data = await loop.run_in_executor(
                    executor,
                    self._enhance_results_sync,
                    query,
                    results,
                    max_hops,
                    include_reasoning,
                )

            return enhanced_data

        except Exception as e:
            logger.error(f"GraphRAG enhancement failed: {e}")
            return {
                "enhanced_results": results,
                "graphrag_analysis": None,
                "reasoning_paths": [],
                "enhancement_applied": False,
                "error": str(e),
            }

    def _enhance_results_sync(
        self, query: str, results: List[Dict[str, Any]], max_hops: int, include_reasoning: bool
    ) -> Dict[str, Any]:
        """Synchronous GraphRAG enhancement (runs in thread pool)."""
        if not self.graphrag:
            return {"enhanced_results": results, "enhancement_applied": False}

        # Extract document IDs from results for graph analysis
        doc_ids = []
        for result in results:
            if isinstance(result, dict):
                doc_id = result.get("id") or result.get("document_id")
                if doc_id:
                    doc_ids.append(str(doc_id))

        if not doc_ids:
            logger.warning("No document IDs found in results for GraphRAG enhancement")
            return {"enhanced_results": results, "enhancement_applied": False}

        try:
            # Use GraphRAG to analyze document relationships
            if hasattr(self.graphrag, "_graph_neighborhood"):
                neighborhood = self.graphrag._graph_neighborhood(doc_ids, max_hops=max_hops)

                # Extract reasoning paths if requested
                reasoning_paths = []
                if include_reasoning and hasattr(self.graphrag, "_extract_reasoning_path"):
                    reasoning_paths = self.graphrag._extract_reasoning_path(neighborhood)

                # Enhance results with graph context
                enhanced_results = self._add_graph_context(results, neighborhood)

                return {
                    "enhanced_results": enhanced_results,
                    "graphrag_analysis": {
                        "neighborhood_size": len(neighborhood.get("nodes", {})),
                        "relationships_found": len(neighborhood.get("relationships", [])),
                        "max_hops_used": max_hops,
                    },
                    "reasoning_paths": reasoning_paths,
                    "enhancement_applied": True,
                }

            else:
                logger.warning("GraphRAG methods not available")
                return {"enhanced_results": results, "enhancement_applied": False}

        except Exception as e:
            logger.error(f"GraphRAG sync enhancement failed: {e}")
            return {"enhanced_results": results, "enhancement_applied": False, "error": str(e)}

    def _add_graph_context(
        self, results: List[Dict[str, Any]], neighborhood: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Add graph context information to search results."""
        enhanced_results = []

        for result in results:
            enhanced_result = result.copy()

            # Add graph context if available
            doc_id = result.get("id") or result.get("document_id")
            if doc_id and str(doc_id) in neighborhood.get("nodes", {}):
                node_info = neighborhood["nodes"][str(doc_id)]
                enhanced_result["graph_context"] = {
                    "connections": len(
                        [
                            r
                            for r in neighborhood.get("relationships", [])
                            if r.get("source") == str(doc_id) or r.get("target") == str(doc_id)
                        ]
                    ),
                    "node_type": node_info.get("document_type", "unknown"),
                    "centrality_score": node_info.get("centrality", 0.0),
                }

            enhanced_results.append(enhanced_result)

        return enhanced_results

    async def detect_communities(
        self, algorithm: str = "louvain", min_community_size: int = 3, resolution: float = 1.0
    ) -> Dict[str, Any]:
        """Detect communities in the knowledge graph.

        Args:
            algorithm: Community detection algorithm to use
            min_community_size: Minimum size for communities
            resolution: Resolution parameter for algorithm

        Returns:
            Community detection results
        """
        if not self.graphrag:
            return {
                "success": False,
                "communities": [],
                "message": "GraphRAG not available for community detection",
            }

        try:
            import concurrent.futures

            loop = asyncio.get_event_loop()

            # Use a ThreadPoolExecutor with limited size
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_THREAD_POOL_SIZE
            ) as executor:
                communities = await loop.run_in_executor(
                    executor,
                    self._detect_communities_sync,
                    algorithm,
                    min_community_size,
                    resolution,
                )

            return {
                "success": True,
                "communities": communities,
                "algorithm_used": algorithm,
                "min_community_size": min_community_size,
                "resolution": resolution,
                "total_communities": len(communities),
            }

        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return {
                "success": False,
                "communities": [],
                "message": f"Community detection failed: {str(e)}",
            }

    def _detect_communities_sync(
        self, algorithm: str, min_community_size: int, resolution: float
    ) -> List[Dict[str, Any]]:
        """Synchronous community detection (runs in thread pool)."""
        if not self.graphrag:
            return []

        logger.info(f"Attempting {algorithm} community detection with resolution {resolution}")

        try:
            # Check if GraphRAG has community detection capabilities
            if hasattr(self.graphrag, "detect_communities"):
                communities = self.graphrag.detect_communities(
                    algorithm=algorithm,
                    min_community_size=min_community_size,
                    resolution=resolution,
                )
                return communities
            elif hasattr(self.graphrag, "_get_graph_data"):
                # Try to get graph data and use networkx for community detection
                try:
                    from networkx.algorithms import community

                    graph_data = self.graphrag._get_graph_data()
                    G = self._build_networkx_graph(graph_data)

                    if algorithm == "louvain":
                        communities_partition = community.louvain_communities(
                            G, resolution=resolution
                        )
                    elif algorithm == "leiden":
                        # Leiden requires python-igraph, fallback to louvain
                        logger.warning("Leiden algorithm not available, using Louvain")
                        communities_partition = community.louvain_communities(
                            G, resolution=resolution
                        )
                    elif algorithm == "modularity":
                        communities_partition = community.greedy_modularity_communities(
                            G, resolution=resolution
                        )
                    else:
                        logger.error(f"Unsupported algorithm: {algorithm}")
                        return []

                    # Convert networkx communities to our format
                    communities = []
                    for i, comm in enumerate(communities_partition):
                        if len(comm) >= min_community_size:
                            communities.append(
                                {
                                    "id": f"community_{i+1}",
                                    "size": len(comm),
                                    "members": list(comm),
                                    "topic": f"Community {i+1}",  # Would need topic modeling
                                    "centrality": self._calculate_community_centrality(G, comm),
                                }
                            )

                    return communities

                except ImportError:
                    logger.error("NetworkX not available for community detection")
                    return []
                except Exception as e:
                    logger.error(f"NetworkX community detection failed: {e}")
                    return []
            else:
                logger.warning("GraphRAG does not support community detection - feature disabled")
                return []

        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return []

    def _build_networkx_graph(self, graph_data: Dict[str, Any]):
        """Build NetworkX graph from GraphRAG data."""
        import networkx as nx

        G = nx.Graph()

        # Add nodes
        nodes = graph_data.get("nodes", {})
        for node_id, node_data in nodes.items():
            G.add_node(node_id, **node_data)

        # Add edges
        relationships = graph_data.get("relationships", [])
        for rel in relationships:
            source = rel.get("source")
            target = rel.get("target")
            if source and target:
                G.add_edge(
                    source,
                    target,
                    **{k: v for k, v in rel.items() if k not in ["source", "target"]},
                )

        return G

    def _calculate_community_centrality(self, G, community) -> float:
        """Calculate centrality score for a community."""
        try:
            import networkx as nx

            # Calculate average betweenness centrality for community members
            centrality = nx.betweenness_centrality(G)
            community_centrality = sum(centrality.get(node, 0) for node in community) / len(
                community
            )
            return round(community_centrality, 3)
        except Exception:
            return 0.0

    def is_available(self) -> bool:
        """Check if GraphRAG integration is available and functional."""
        return self.graphrag is not None

    def get_status(self) -> Dict[str, Any]:
        """Get GraphRAG bridge status information."""
        return {
            "initialized": self._initialized,
            "graphrag_available": self.graphrag is not None,
            "config_path": self.config_path,
            "verbose": self.verbose,
        }


# Global bridge instance (singleton pattern)
_bridge_instance: Optional[GraphRAGBridge] = None


def get_graphrag_bridge() -> GraphRAGBridge:
    """Get the global GraphRAG bridge instance (thread-safe singleton)."""
    global _bridge_instance
    if _bridge_instance is None:
        with _bridge_lock:
            # Double-check locking pattern
            if _bridge_instance is None:
                _bridge_instance = GraphRAGBridge(verbose=True)
    return _bridge_instance


def reset_graphrag_bridge() -> None:
    """Reset the global GraphRAG bridge instance (useful for testing)."""
    global _bridge_instance
    with _bridge_lock:
        _bridge_instance = None
