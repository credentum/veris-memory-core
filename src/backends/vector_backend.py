#!/usr/bin/env python3
"""
Vector backend adapter for Qdrant vector database.

This module implements the BackendSearchInterface for Qdrant vector operations,
providing semantic search capabilities with embedding generation.
"""

import asyncio
import os
import time
from typing import Any, List

from qdrant_client.http import models as qdrant_models

from ..interfaces.backend_interface import (
    BackendHealthStatus,
    BackendSearchError,
    BackendSearchInterface,
    SearchOptions,
)
from ..interfaces.memory_result import ContentType, MemoryResult, ResultSource
from ..utils.logging_middleware import backend_logger, log_backend_timing

# Import sparse embedding support for hybrid search
try:
    from ..embedding.sparse_service import get_sparse_embedding_service, SPARSE_ENABLED
    from ..storage.qdrant_client import SPARSE_EMBEDDINGS_ENABLED
    HYBRID_SEARCH_AVAILABLE = SPARSE_ENABLED and SPARSE_EMBEDDINGS_ENABLED
    backend_logger.info(f"🔍 Hybrid search config: SPARSE_ENABLED={SPARSE_ENABLED}, SPARSE_EMBEDDINGS_ENABLED={SPARSE_EMBEDDINGS_ENABLED}, HYBRID_AVAILABLE={HYBRID_SEARCH_AVAILABLE}")
except ImportError as e:
    HYBRID_SEARCH_AVAILABLE = False
    get_sparse_embedding_service = None
    SPARSE_ENABLED = False
    SPARSE_EMBEDDINGS_ENABLED = False
    backend_logger.warning(f"⚠️ Hybrid search disabled - import failed: {e}")


class VectorBackend(BackendSearchInterface):
    """
    Vector search backend implementation using Qdrant.

    Provides semantic search capabilities by generating embeddings and
    performing similarity search in vector space.
    """

    def __init__(self, qdrant_client, embedding_generator):
        """
        Initialize vector backend.

        Args:
            qdrant_client: VectorDBInitializer instance for Qdrant operations
            embedding_generator: Service for generating embeddings from text
        """
        self.client = qdrant_client
        self.embedding_generator = embedding_generator
        # Use environment variable for collection name to ensure consistency
        self._collection_name = os.getenv("QDRANT_COLLECTION_NAME", "context_embeddings")

    @property
    def backend_name(self) -> str:
        """Return the name of this backend."""
        return "vector"

    async def search(self, query: str, options: SearchOptions) -> List[MemoryResult]:
        """
        Search vector database using semantic similarity with optional hybrid search.

        When sparse embeddings are enabled, performs hybrid search combining
        dense (semantic) and sparse (keyword/BM25) vectors for improved retrieval.

        Args:
            query: Search query text
            options: Search configuration options

        Returns:
            List of MemoryResult objects sorted by relevance score

        Raises:
            BackendSearchError: If vector search operation fails
        """
        async with log_backend_timing(self.backend_name, "search", backend_logger) as metadata:
            try:
                # Generate dense query embedding
                embed_start = time.time()
                query_vector = await self._generate_query_embedding(query)
                embed_time = (time.time() - embed_start) * 1000

                metadata["embedding_time_ms"] = embed_time
                metadata["embedding_dimensions"] = len(query_vector) if query_vector else 0

                # Generate sparse query embedding for hybrid search
                sparse_vector = None
                if HYBRID_SEARCH_AVAILABLE and get_sparse_embedding_service:
                    try:
                        sparse_start = time.time()
                        sparse_service = get_sparse_embedding_service()
                        if not sparse_service.is_available():
                            await sparse_service.initialize()

                        if sparse_service.is_available():
                            sparse_result = sparse_service.generate_sparse_embedding(query)
                            if sparse_result:
                                sparse_vector = sparse_result.to_dict()
                                sparse_time = (time.time() - sparse_start) * 1000
                                metadata["sparse_embedding_time_ms"] = sparse_time
                                metadata["sparse_dimensions"] = len(sparse_vector.get("indices", []))
                    except Exception as e:
                        backend_logger.warning(f"Sparse embedding generation failed: {e}")

                # Perform vector search (hybrid if sparse available)
                search_start = time.time()
                raw_results = await self._perform_vector_search_with_sparse(
                    query_vector, sparse_vector, options
                )
                search_time = (time.time() - search_start) * 1000

                metadata["search_time_ms"] = search_time
                metadata["raw_result_count"] = len(raw_results)
                metadata["hybrid_search"] = sparse_vector is not None

                # Convert to normalized format
                results = self._convert_to_memory_results(raw_results)

                # Apply additional filtering if needed
                filtered_results = self._apply_filters(results, options)

                metadata["result_count"] = len(filtered_results)
                metadata["top_score"] = filtered_results[0].score if filtered_results else 0.0

                backend_logger.info(
                    "Vector search completed",
                    query_length=len(query),
                    hybrid=sparse_vector is not None,
                    **metadata
                )

                return filtered_results

            except Exception as e:
                error_msg = f"Vector search failed: {str(e)}"
                backend_logger.error(error_msg, error=str(e))
                raise BackendSearchError(self.backend_name, error_msg, e)

    async def search_by_embedding(
        self, embedding: List[float], options: SearchOptions, original_query: str = None
    ) -> List[MemoryResult]:
        """
        Search vector database using a pre-computed embedding with optional hybrid search.

        This method is used by HyDE (Hypothetical Document Embeddings) to search
        using the embedding of a hypothetical document rather than generating
        an embedding from the query text.

        When original_query is provided and hybrid search is available, generates
        sparse embeddings from the original query to enable BM25/keyword matching
        alongside the dense HyDE embedding.

        Args:
            embedding: Pre-computed embedding vector (dense, from HyDE)
            options: Search configuration options
            original_query: Original query text for sparse embedding generation.
                           Enables hybrid search when provided.

        Returns:
            List of MemoryResult objects sorted by relevance score

        Raises:
            BackendSearchError: If vector search operation fails
        """
        async with log_backend_timing(self.backend_name, "search_by_embedding", backend_logger) as metadata:
            try:
                metadata["embedding_dimensions"] = len(embedding) if embedding else 0
                metadata["search_type"] = "hyde"

                # Generate sparse embedding from original query for hybrid search
                sparse_vector = None
                if original_query and HYBRID_SEARCH_AVAILABLE and get_sparse_embedding_service:
                    try:
                        sparse_start = time.time()
                        sparse_service = get_sparse_embedding_service()
                        if not sparse_service.is_available():
                            await sparse_service.initialize()

                        if sparse_service.is_available():
                            sparse_result = sparse_service.generate_sparse_embedding(original_query)
                            if sparse_result:
                                sparse_vector = sparse_result.to_dict()
                                sparse_time = (time.time() - sparse_start) * 1000
                                metadata["sparse_embedding_time_ms"] = sparse_time
                                metadata["sparse_dimensions"] = len(sparse_vector.get("indices", []))
                                backend_logger.debug(
                                    f"HyDE+Hybrid: Generated sparse embedding from original query "
                                    f"({len(sparse_vector.get('indices', []))} dims)"
                                )
                    except Exception as e:
                        backend_logger.warning(f"Sparse embedding generation for HyDE failed: {e}")

                # Perform vector search (hybrid if sparse available)
                search_start = time.time()
                if sparse_vector:
                    # Use hybrid search with HyDE dense + original query sparse
                    raw_results = await self._perform_vector_search_with_sparse(
                        embedding, sparse_vector, options
                    )
                    metadata["hybrid_search"] = True
                else:
                    # Fall back to dense-only search
                    raw_results = await self._perform_vector_search(embedding, options)
                    metadata["hybrid_search"] = False
                search_time = (time.time() - search_start) * 1000

                metadata["search_time_ms"] = search_time
                metadata["raw_result_count"] = len(raw_results)

                # Convert to normalized format
                results = self._convert_to_memory_results(raw_results)

                # Apply additional filtering if needed
                filtered_results = self._apply_filters(results, options)

                metadata["result_count"] = len(filtered_results)
                metadata["top_score"] = filtered_results[0].score if filtered_results else 0.0

                backend_logger.info(
                    "Vector search by embedding completed",
                    embedding_dims=len(embedding),
                    hybrid=sparse_vector is not None,
                    original_query_provided=original_query is not None,
                    **metadata
                )

                return filtered_results

            except Exception as e:
                error_msg = f"Vector search by embedding failed: {str(e)}"
                backend_logger.error(error_msg, error=str(e))
                raise BackendSearchError(self.backend_name, error_msg, e)

    async def health_check(self) -> BackendHealthStatus:
        """
        Check the health of the vector backend.

        Returns:
            BackendHealthStatus with current health information
        """
        start_time = time.time()

        try:
            # Test basic connectivity
            collections = self.client.get_collections()
            response_time = (time.time() - start_time) * 1000

            # Check if our collection exists
            collection_exists = (
                any(col.name == self._collection_name for col in collections.collections)
                if hasattr(collections, "collections")
                else False
            )

            # Test embedding generation
            embed_test_success = True
            embed_error = None
            try:
                test_embedding = await self._generate_query_embedding("health test")
                embed_dimensions = len(test_embedding) if test_embedding else 0
            except Exception as e:
                embed_test_success = False
                embed_error = str(e)
                embed_dimensions = 0

            # Determine overall status
            if collection_exists and embed_test_success:
                status = "healthy"
            elif embed_test_success:
                status = "degraded"  # Can generate embeddings but collection missing
            else:
                status = "unhealthy"

            return BackendHealthStatus(
                status=status,
                response_time_ms=response_time,
                metadata={
                    "collection_exists": collection_exists,
                    "collection_name": self._collection_name,
                    "embedding_service_healthy": embed_test_success,
                    "embedding_dimensions": embed_dimensions,
                    "embedding_error": embed_error,
                    "total_collections": (
                        len(collections.collections) if hasattr(collections, "collections") else 0
                    ),
                },
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return BackendHealthStatus(
                status="unhealthy",
                response_time_ms=response_time,
                error_message=str(e),
                metadata={"connection_error": True, "error_type": type(e).__name__},
            )

    async def initialize(self) -> None:
        """Initialize the vector backend."""
        backend_logger.info("Initializing vector backend")

        try:
            # Ensure collection exists
            if hasattr(self.client, "create_collection"):
                self.client.create_collection(force=False)  # Don't overwrite existing

            # Test embedding service
            await self._generate_query_embedding("initialization test")

            backend_logger.info("Vector backend initialized successfully")

        except Exception as e:
            backend_logger.error(f"Vector backend initialization failed: {e}")
            raise BackendSearchError(self.backend_name, f"Initialization failed: {e}", e)

    async def cleanup(self) -> None:
        """Clean up vector backend resources."""
        backend_logger.info("Cleaning up vector backend")
        # No explicit cleanup needed for Qdrant client

    # Private helper methods

    async def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query."""
        try:
            if self.embedding_generator:
                # Use the existing embedding service
                if asyncio.iscoroutinefunction(self.embedding_generator.generate_embedding):
                    return await self.embedding_generator.generate_embedding(
                        query, adjust_dimensions=True
                    )
                else:
                    # Handle sync embedding generators
                    return self.embedding_generator.generate_embedding(
                        query, adjust_dimensions=True
                    )
            else:
                raise ValueError("No embedding generator available")

        except Exception as e:
            backend_logger.error(f"Embedding generation failed: {e}")
            raise

    def _build_qdrant_filter(self, options: SearchOptions) -> qdrant_models.Filter | None:
        """
        Convert SearchOptions.filters to Qdrant filter format.

        Handles metadata fields like author, tags, type, etc. by building
        Qdrant FieldCondition objects that filter at the database level.

        Args:
            options: Search options containing filters dict

        Returns:
            Qdrant Filter object or None if no filters specified
        """
        if not options.filters:
            return None

        filter_conditions = []

        for key, value in options.filters.items():
            # Skip special keys that are handled separately
            if key in ("include_shared", "sort_by", "sort_order"):
                continue

            # Handle different value types
            if value is None:
                continue

            if isinstance(value, list):
                # For list values (like tags), use MatchAny
                if value:  # Only add if non-empty
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchAny(any=value)
                        )
                    )
            elif isinstance(value, bool):
                filter_conditions.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value)
                    )
                )
            elif isinstance(value, (str, int, float)):
                filter_conditions.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value)
                    )
                )

        if not filter_conditions:
            return None

        backend_logger.debug(
            f"Built Qdrant filter with {len(filter_conditions)} conditions: "
            f"{[c.key for c in filter_conditions]}"
        )

        return qdrant_models.Filter(must=filter_conditions)

    async def _perform_vector_search(
        self, query_vector: List[float], options: SearchOptions
    ) -> List[Any]:
        """Perform dense-only vector search (used by search_by_embedding for HyDE)."""
        try:
            # Build Qdrant filter from options.filters (Issue #102 fix)
            qdrant_filter = self._build_qdrant_filter(options)

            results = self.client.search(
                query_vector=query_vector,
                limit=options.limit,
                filter_dict=qdrant_filter,
            )

            # Apply score threshold manually
            if options.score_threshold > 0:
                results = [r for r in results if r.get("score", 1.0) >= options.score_threshold]

            return results

        except Exception as e:
            backend_logger.error(f"Qdrant search failed: {e}")
            raise

    async def _perform_vector_search_with_sparse(
        self,
        query_vector: List[float],
        sparse_vector: dict,
        options: SearchOptions
    ) -> List[Any]:
        """Perform hybrid search combining dense and sparse vectors when available."""
        try:
            # Build Qdrant filter from options.filters (Issue #102 fix)
            qdrant_filter = self._build_qdrant_filter(options)

            # Use hybrid_search if we have sparse vector and client supports it
            if sparse_vector and hasattr(self.client, 'hybrid_search'):
                backend_logger.debug(
                    f"Performing hybrid search with {len(sparse_vector.get('indices', []))} sparse dims"
                )
                results = self.client.hybrid_search(
                    dense_vector=query_vector,
                    sparse_vector=sparse_vector,
                    limit=options.limit,
                    filter_dict=qdrant_filter,
                )

                # Apply score threshold
                if options.score_threshold > 0:
                    results = [r for r in results if r.get("score", 1.0) >= options.score_threshold]

                return results

            # Fallback to dense-only search
            backend_logger.debug("Falling back to dense-only search (no sparse vector or hybrid not supported)")
            return self.client.search(
                query_vector=query_vector,
                limit=options.limit,
                filter_dict=qdrant_filter,
            )

        except Exception as e:
            backend_logger.warning(f"Hybrid search failed, falling back to dense: {e}")
            # Fallback to dense-only search
            qdrant_filter = self._build_qdrant_filter(options)
            return self.client.search(
                query_vector=query_vector,
                limit=options.limit,
                filter_dict=qdrant_filter,
            )

    def _convert_to_memory_results(self, raw_results: List[Any]) -> List[MemoryResult]:
        """Convert Qdrant results to normalized MemoryResult format."""
        results = []

        for result in raw_results:
            try:
                # Extract data from Qdrant result structure
                # Handle both dict results (from qdrant_client.search()) and object results (ScoredPoint)
                if isinstance(result, dict):
                    # Dict format from VectorDBInitializer.search()
                    result_id = str(result.get("id", ""))
                    payload = result.get("payload", {})
                    score = float(result.get("score", 1.0))
                else:
                    # Object format (ScoredPoint) - fallback for direct Qdrant client usage
                    result_id = str(result.id) if hasattr(result, "id") else str(id(result))
                    payload = result.payload if hasattr(result, "payload") else {}
                    score = float(result.score) if hasattr(result, "score") else 1.0

                # Extract text content with robust fallback handling
                text = self._extract_text_content(payload)

                # Ensure text is never empty (validation requirement)
                if not text:
                    text = f"Document {result_id}"  # Fallback to prevent validation errors

                # Determine content type
                content_type = ContentType.GENERAL
                if payload.get("type"):
                    try:
                        content_type = ContentType(payload["type"])
                    except ValueError:
                        # Keep as GENERAL if unknown type
                        pass
                elif "code" in text.lower():
                    content_type = ContentType.CODE
                elif any(word in text.lower() for word in ["document", "guide", "manual"]):
                    content_type = ContentType.DOCUMENTATION

                # Extract tags
                tags = []
                if payload.get("tags"):
                    tags = (
                        payload["tags"] if isinstance(payload["tags"], list) else [payload["tags"]]
                    )

                # Create normalized result
                memory_result = MemoryResult(
                    id=result_id,
                    text=text,
                    type=content_type,
                    score=min(max(score, 0.0), 1.0),  # Clamp to valid range
                    source=ResultSource.VECTOR,
                    tags=tags,
                    metadata={
                        **payload,
                        "qdrant_score": score,  # Preserve original score
                        "vector_search": True,
                    },
                    namespace=payload.get("namespace"),
                    title=payload.get("title"),
                    user_id=payload.get("user_id") or payload.get("author"),
                    # Cross-team sharing (Issue #2)
                    shared=payload.get("shared", False),
                )

                results.append(memory_result)

            except Exception as e:
                # Handle both dict and object formats for logging
                if isinstance(result, dict):
                    log_id = str(result.get("id", "unknown"))
                else:
                    log_id = str(getattr(result, "id", "unknown"))
                backend_logger.warning(
                    f"Failed to convert vector result: {e}",
                    result_id=log_id,
                )
                continue

        return results

    def _apply_filters(
        self, results: List[MemoryResult], options: SearchOptions
    ) -> List[MemoryResult]:
        """Apply additional filtering to results."""
        filtered = results

        # Apply namespace filter if specified
        if options.namespace:
            filtered = [r for r in filtered if r.namespace == options.namespace]

        # Apply custom filters from options
        if options.filters:
            for filter_key, filter_value in options.filters.items():
                if filter_key == "type":
                    filtered = [r for r in filtered if r.type == filter_value]
                elif filter_key == "tags":
                    filter_tags = filter_value if isinstance(filter_value, list) else [filter_value]
                    filtered = [r for r in filtered if any(tag in r.tags for tag in filter_tags)]
                elif filter_key == "user_id":
                    # Cross-team sharing (Issue #2): Include shared contexts
                    include_shared = options.filters.get("include_shared", True)
                    if include_shared:
                        filtered = [r for r in filtered if r.user_id == filter_value or r.shared]
                    else:
                        filtered = [r for r in filtered if r.user_id == filter_value]

        # Apply score threshold
        if options.score_threshold > 0:
            filtered = [r for r in filtered if r.score >= options.score_threshold]

        return filtered[: options.limit]  # Ensure we don't exceed limit

    def _extract_text_content(self, payload: dict) -> str:
        """
        Extract text content from complex payload structures.

        Tries multiple extraction strategies to handle different data formats.
        """
        # Strategy 1: Direct text fields
        text = payload.get("text", "")
        if text and isinstance(text, str):
            return text.strip()

        # Strategy 2: Content field
        content = payload.get("content", "")
        if content:
            if isinstance(content, str):
                return content.strip()
            elif isinstance(content, dict):
                # Handle nested content structures
                return content.get("text", content.get("title", content.get("description", "")))

        # Strategy 3: Legacy compatibility
        if "user_message" in payload:
            return str(payload["user_message"]).strip()

        # Strategy 4: Title or description
        title = payload.get("title", "")
        if title:
            return str(title).strip()

        description = payload.get("description", "")
        if description:
            return str(description).strip()

        # Strategy 5: First string value in payload
        for key, value in payload.items():
            if isinstance(value, str) and value.strip() and key not in ["id", "type", "namespace"]:
                return value.strip()

        # No text content found
        return ""
