#!/usr/bin/env python3
"""
Enhanced storage orchestrator that coordinates multi-backend storage operations.

This module provides a unified interface for storing data across all backends
including the new text search backend, ensuring data consistency and proper
indexing across vector, graph, key-value, and text search systems.
"""

import asyncio
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..backends.text_backend import TextSearchBackend, get_text_backend
from ..storage.kv_store import ContextKV
from ..storage.neo4j_client import Neo4jInitializer
from ..storage.qdrant_client import VectorDBInitializer

logger = logging.getLogger(__name__)


@dataclass
class StorageResult:
    """Result of a storage operation in a single backend."""

    backend: str
    success: bool
    result_id: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        """Initialize the metadata dictionary if not provided."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StorageRequest:
    """Request for multi-backend storage operation."""

    content: Any  # Content to store (dict, string, etc.)
    content_type: str = "text"
    context_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    relationships: Optional[List[Dict[str, Any]]] = None
    backends: Optional[List[str]] = None  # Specific backends to use, None = all
    tags: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Initialize default values for optional fields."""
        if self.context_id is None:
            self.context_id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}
        if self.relationships is None:
            self.relationships = []
        if self.tags is None:
            self.tags = []


@dataclass
class StorageResponse:
    """Response from multi-backend storage operation."""

    success: bool
    context_id: str
    results: List[StorageResult]
    total_time_ms: float
    error_message: Optional[str] = None

    @property
    def successful_backends(self) -> List[str]:
        """Get list of backends that succeeded."""
        return [r.backend for r in self.results if r.success]

    @property
    def failed_backends(self) -> List[str]:
        """Get list of backends that failed."""
        return [r.backend for r in self.results if not r.success]

    @property
    def partial_success(self) -> bool:
        """Check if some but not all backends succeeded."""
        success_count = sum(1 for r in self.results if r.success)
        return 0 < success_count < len(self.results)


class EnhancedStorageOrchestrator:
    """
    Orchestrator for coordinated storage operations across multiple backends.

    This class ensures that data is properly stored and indexed across:
    - Vector database (Qdrant) for semantic search
    - Graph database (Neo4j) for relationship queries
    - Key-value store (Redis) for fast lookups
    - Text search backend (BM25) for keyword search
    """

    def __init__(
        self,
        qdrant_client: Optional[VectorDBInitializer] = None,
        neo4j_client: Optional[Neo4jInitializer] = None,
        kv_store: Optional[ContextKV] = None,
        text_backend: Optional[TextSearchBackend] = None,
        embedding_generator: Optional[Any] = None,
        max_concurrent_operations: int = 10,
        rate_limit_per_second: float = 5.0,
        retry_max_attempts: int = 3,
    ):
        """
        Initialize the storage orchestrator.

        Args:
            qdrant_client: Vector database client
            neo4j_client: Graph database client
            kv_store: Key-value store client
            text_backend: Text search backend
            embedding_generator: Service for generating embeddings
            max_concurrent_operations: Maximum concurrent storage operations
            rate_limit_per_second: Rate limit for operations per second
            retry_max_attempts: Maximum retry attempts for failed operations
        """
        self.qdrant_client = qdrant_client
        self.neo4j_client = neo4j_client
        self.kv_store = kv_store
        self.text_backend = text_backend or get_text_backend()
        self.embedding_generator = embedding_generator

        # Rate limiting and retry configuration
        self.max_concurrent_operations = max_concurrent_operations
        self.rate_limit_per_second = rate_limit_per_second
        self.retry_max_attempts = retry_max_attempts
        self._rate_limit_semaphore = asyncio.Semaphore(max_concurrent_operations)
        self._last_operation_time = 0.0

        # Dynamic rate limiting based on backend response times
        self._response_times: Dict[str, List[float]] = {}
        self._adaptive_rate_limit = rate_limit_per_second
        self._min_rate_limit = 1.0
        self._max_rate_limit = rate_limit_per_second * 2

        # Track available backends
        self.available_backends = {
            "vector": self.qdrant_client is not None,
            "graph": self.neo4j_client is not None,
            "kv": self.kv_store is not None,
            "text": self.text_backend is not None,
        }

        logger.info(f"Storage orchestrator initialized with backends: {self.available_backends}")

    def _update_adaptive_rate_limit(self, backend: str, response_time: float) -> None:
        """Update adaptive rate limiting based on backend response times."""
        if backend not in self._response_times:
            self._response_times[backend] = []

        # Keep only last 10 response times per backend
        self._response_times[backend].append(response_time)
        if len(self._response_times[backend]) > 10:
            self._response_times[backend].pop(0)

        # Calculate average response time across all backends
        all_response_times = []
        for times in self._response_times.values():
            all_response_times.extend(times)

        if all_response_times:
            avg_response_time = sum(all_response_times) / len(all_response_times)

            # Adjust rate limit based on response time
            if avg_response_time > 2.0:  # Slow responses, reduce rate
                self._adaptive_rate_limit = max(
                    self._min_rate_limit, self._adaptive_rate_limit * 0.8
                )
            elif avg_response_time < 0.5:  # Fast responses, increase rate
                self._adaptive_rate_limit = min(
                    self._max_rate_limit, self._adaptive_rate_limit * 1.2
                )

    async def _apply_rate_limit(self) -> None:
        """Apply adaptive rate limiting to prevent overwhelming backends."""
        current_time = time.time()
        time_since_last = current_time - self._last_operation_time
        min_interval = 1.0 / self._adaptive_rate_limit

        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self._last_operation_time = time.time()

    async def _retry_operation(self, operation_func, *args, **kwargs) -> Any:
        """Retry an operation with exponential backoff and response time tracking."""
        last_exception = None
        backend_name = (
            getattr(operation_func, "__name__", "unknown")
            .replace("_store_in_", "")
            .replace("_backend", "")
        )

        for attempt in range(self.retry_max_attempts):
            try:
                async with self._rate_limit_semaphore:
                    await self._apply_rate_limit()

                    start_time = time.time()
                    result = await operation_func(*args, **kwargs)
                    response_time = time.time() - start_time

                    # Update adaptive rate limiting based on response time
                    self._update_adaptive_rate_limit(backend_name, response_time)

                    return result
            except Exception as e:
                last_exception = e
                if attempt < self.retry_max_attempts - 1:
                    # Exponential backoff with jitter
                    delay = (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{self.retry_max_attempts}), retrying in {delay:.2f}s: {str(e)[:100]}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Operation failed after {self.retry_max_attempts} attempts: {str(e)[:100]}"
                    )

        raise last_exception or Exception("Operation failed after all retries")

    async def store_context(self, request: StorageRequest) -> StorageResponse:
        """
        Store context data across multiple backends.

        Args:
            request: Storage request with content and configuration

        Returns:
            Storage response with results from all backends
        """
        start_time = time.time()
        results = []

        # Determine which backends to use
        target_backends = request.backends or list(self.available_backends.keys())
        active_backends = [b for b in target_backends if self.available_backends.get(b, False)]

        logger.info(f"Storing context {request.context_id} to backends: {active_backends}")

        # Extract text content for indexing
        text_content = self._extract_text_content(request.content)

        # Store in each backend concurrently
        storage_tasks = []

        if "vector" in active_backends and self.qdrant_client:
            task = self._retry_operation(self._store_in_vector_db, request, text_content)
            storage_tasks.append(task)

        if "graph" in active_backends and self.neo4j_client:
            task = self._retry_operation(self._store_in_graph_db, request, text_content)
            storage_tasks.append(task)

        if "kv" in active_backends and self.kv_store:
            task = self._retry_operation(self._store_in_kv_store, request, text_content)
            storage_tasks.append(task)

        if "text" in active_backends and self.text_backend:
            task = self._retry_operation(self._store_in_text_backend, request, text_content)
            storage_tasks.append(task)

        # Execute all storage operations concurrently
        if storage_tasks:
            storage_results = await asyncio.gather(*storage_tasks, return_exceptions=True)

            # Process results
            for result in storage_results:
                if isinstance(result, Exception):
                    results.append(
                        StorageResult(
                            backend="unknown",
                            success=False,
                            error_message=f"Storage task failed: {str(result)}",
                        )
                    )
                elif isinstance(result, StorageResult):
                    results.append(result)

        # Calculate overall success
        total_time = (time.time() - start_time) * 1000
        success_count = sum(1 for r in results if r.success)
        overall_success = success_count > 0  # At least one backend succeeded

        # Build response
        response = StorageResponse(
            success=overall_success,
            context_id=request.context_id,
            results=results,
            total_time_ms=total_time,
        )

        if not overall_success:
            response.error_message = "All storage backends failed"
        elif response.partial_success:
            response.error_message = (
                f"Partial success: {len(response.failed_backends)} backends failed"
            )

        logger.info(
            f"Storage operation completed: {success_count}/{len(results)} backends successful, "
            f"total_time={total_time:.2f}ms"
        )

        return response

    def _extract_text_content(self, content: Any) -> str:
        """Extract text content for indexing from various content types."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Try common text fields
            text_fields = ["text", "content", "body", "description", "title", "summary"]
            for field in text_fields:
                if field in content and isinstance(content[field], str):
                    return content[field]

            # Combine title and description/content
            title = content.get("title", "")
            description = content.get("description", content.get("content", ""))
            if title or description:
                return f"{title} {description}".strip()

            # Fallback: join all string values
            text_parts = []
            for key, value in content.items():
                if isinstance(value, str) and value.strip():
                    text_parts.append(value)
            return " ".join(text_parts) if text_parts else str(content)
        else:
            return str(content)

    async def _store_in_vector_db(
        self, request: StorageRequest, text_content: str
    ) -> StorageResult:
        """Store content in vector database."""
        start_time = time.time()

        try:
            logger.debug(f"Storing in vector database: {request.context_id}")

            # Generate embedding
            embedding = None
            if self.embedding_generator:
                try:
                    # Use provided embedding generator
                    if hasattr(self.embedding_generator, "generate_embedding"):
                        embedding = await self.embedding_generator.generate_embedding(text_content)
                    else:
                        # Fallback for callable embedding generators
                        embedding = await self.embedding_generator(text_content)
                except Exception as e:
                    logger.warning(f"Embedding generation failed: {e}")
                    # Continue without embedding - will use fallback in store_vector

            # Store vector
            vector_id = self.qdrant_client.store_vector(
                vector_id=request.context_id,
                embedding=embedding,
                metadata={
                    "content": request.content,
                    "type": request.content_type,
                    "metadata": request.metadata,
                    "tags": request.tags,
                    "stored_at": datetime.now().isoformat(),
                },
            )

            processing_time = (time.time() - start_time) * 1000

            return StorageResult(
                backend="vector",
                success=True,
                result_id=vector_id,
                processing_time_ms=processing_time,
                metadata={"embedding_dimensions": len(embedding) if embedding else None},
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Vector storage failed: {e}")

            return StorageResult(
                backend="vector",
                success=False,
                error_message=f"Vector storage error: {str(e)}",
                processing_time_ms=processing_time,
            )

    async def _store_in_graph_db(self, request: StorageRequest, text_content: str) -> StorageResult:
        """Store content in graph database."""
        start_time = time.time()

        try:
            logger.debug(f"Storing in graph database: {request.context_id}")

            # Prepare properties for Neo4j
            properties = {
                "id": request.context_id,
                "type": request.content_type,
                "created_at": datetime.now().isoformat(),
                "text_content": text_content[:1000],  # Limit text length
            }

            # Handle complex content by flattening
            if isinstance(request.content, dict):
                for key, value in request.content.items():
                    if isinstance(value, (dict, list)):
                        properties[f"{key}_json"] = json.dumps(value)
                    elif isinstance(value, str) and len(value) <= 500:  # Limit string length
                        properties[key] = value

            # Add metadata
            if request.metadata:
                for key, value in request.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        properties[f"meta_{key}"] = value

            # Add tags
            if request.tags:
                properties["tags"] = ",".join(request.tags)

            # Create node
            graph_id = self.neo4j_client.create_node(
                labels=["Context", request.content_type.title()], properties=properties
            )

            # Create relationships if specified
            relationships_created = 0
            if request.relationships:
                for rel in request.relationships:
                    try:
                        self.neo4j_client.create_relationship(
                            from_id=graph_id,
                            to_id=rel.get("target"),
                            rel_type=rel.get("type", "RELATES_TO"),
                        )
                        relationships_created += 1
                    except Exception as rel_error:
                        logger.warning(f"Failed to create relationship: {rel_error}")

            processing_time = (time.time() - start_time) * 1000

            return StorageResult(
                backend="graph",
                success=True,
                result_id=graph_id,
                processing_time_ms=processing_time,
                metadata={
                    "relationships_created": relationships_created,
                    "properties_stored": len(properties),
                },
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Graph storage failed: {e}")

            return StorageResult(
                backend="graph",
                success=False,
                error_message=f"Graph storage error: {str(e)}",
                processing_time_ms=processing_time,
            )

    async def _store_in_kv_store(self, request: StorageRequest, text_content: str) -> StorageResult:
        """Store content in key-value store."""
        start_time = time.time()

        try:
            logger.debug(f"Storing in KV store: {request.context_id}")

            # Create KV entry with metadata
            kv_data = {
                "id": request.context_id,
                "type": request.content_type,
                "content": request.content,
                "metadata": request.metadata,
                "tags": request.tags,
                "text_content": text_content[:500],  # Store truncated text for quick access
                "stored_at": datetime.now().isoformat(),
            }

            # Store with expiration if configured
            key = f"context:{request.context_id}"
            ttl = request.metadata.get("ttl_hours") if request.metadata else None
            ttl_seconds = int(ttl * 3600) if ttl else None

            self.kv_store.set(key, json.dumps(kv_data), ex=ttl_seconds)

            # Also store by type for quick lookups
            type_key = f"type:{request.content_type}:{request.context_id}"
            self.kv_store.set(type_key, request.context_id, ex=ttl_seconds)

            processing_time = (time.time() - start_time) * 1000

            return StorageResult(
                backend="kv",
                success=True,
                result_id=key,
                processing_time_ms=processing_time,
                metadata={"ttl_seconds": ttl_seconds, "data_size_bytes": len(json.dumps(kv_data))},
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"KV storage failed: {e}")

            return StorageResult(
                backend="kv",
                success=False,
                error_message=f"KV storage error: {str(e)}",
                processing_time_ms=processing_time,
            )

    async def _store_in_text_backend(
        self, request: StorageRequest, text_content: str
    ) -> StorageResult:
        """Store content in text search backend."""
        start_time = time.time()

        try:
            logger.debug(f"Storing in text backend: {request.context_id}")

            if not text_content.strip():
                return StorageResult(
                    backend="text",
                    success=False,
                    error_message="No text content available for indexing",
                )

            # Prepare metadata for text indexing
            text_metadata = {
                "content_type": request.content_type,
                "tags": request.tags,
                "stored_at": datetime.now().isoformat(),
                "source": "enhanced_storage",
                **request.metadata,
            }

            # Index document in text backend
            await self.text_backend.index_document(
                doc_id=request.context_id,
                text=text_content,
                content_type=request.content_type,
                metadata=text_metadata,
            )

            processing_time = (time.time() - start_time) * 1000

            return StorageResult(
                backend="text",
                success=True,
                result_id=request.context_id,
                processing_time_ms=processing_time,
                metadata={
                    "text_length": len(text_content),
                    "token_count": len(self.text_backend.tokenize(text_content)),
                },
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Text backend storage failed: {e}")

            return StorageResult(
                backend="text",
                success=False,
                error_message=f"Text storage error: {str(e)}",
                processing_time_ms=processing_time,
            )

    async def retrieve_context(
        self, context_id: str, backends: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve context from multiple backends.

        Args:
            context_id: ID of context to retrieve
            backends: Specific backends to query, None = all

        Returns:
            Combined results from all backends
        """
        target_backends = backends or list(self.available_backends.keys())
        active_backends = [b for b in target_backends if self.available_backends.get(b, False)]

        results = {}

        # Query each backend
        retrieval_tasks = []

        if "vector" in active_backends and self.qdrant_client:
            retrieval_tasks.append(("vector", self._retrieve_from_vector_db(context_id)))

        if "graph" in active_backends and self.neo4j_client:
            retrieval_tasks.append(("graph", self._retrieve_from_graph_db(context_id)))

        if "kv" in active_backends and self.kv_store:
            retrieval_tasks.append(("kv", self._retrieve_from_kv_store(context_id)))

        if "text" in active_backends and self.text_backend:
            retrieval_tasks.append(("text", self._retrieve_from_text_backend(context_id)))

        # Execute retrievals
        if retrieval_tasks:
            backend_names = [name for name, _ in retrieval_tasks]
            tasks = [task for _, task in retrieval_tasks]

            retrieval_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for backend_name, result in zip(backend_names, retrieval_results):
                if isinstance(result, Exception):
                    results[backend_name] = {"error": str(result)}
                else:
                    results[backend_name] = result

        return {
            "context_id": context_id,
            "backends": results,
            "available_in": [k for k, v in results.items() if v.get("found", False)],
        }

    async def _retrieve_from_vector_db(self, context_id: str) -> Dict[str, Any]:
        """Retrieve context from vector database."""
        try:
            # Query by ID (this would need to be implemented in qdrant_client)
            # For now, return placeholder
            return {"found": False, "reason": "Vector retrieval by ID not implemented"}
        except Exception as e:
            return {"found": False, "error": str(e)}

    async def _retrieve_from_graph_db(self, context_id: str) -> Dict[str, Any]:
        """Retrieve context from graph database."""
        try:
            results = self.neo4j_client.query(
                "MATCH (n:Context {id: $context_id}) RETURN n", {"context_id": context_id}
            )

            if results:
                node_data = results[0].get("n", {})
                return {"found": True, "data": node_data}
            else:
                return {"found": False, "reason": "Context not found in graph"}
        except Exception as e:
            return {"found": False, "error": str(e)}

    async def _retrieve_from_kv_store(self, context_id: str) -> Dict[str, Any]:
        """Retrieve context from key-value store."""
        try:
            key = f"context:{context_id}"
            data = self.kv_store.get(key)

            if data:
                return {"found": True, "data": json.loads(data)}
            else:
                return {"found": False, "reason": "Context not found in KV store"}
        except Exception as e:
            return {"found": False, "error": str(e)}

    async def _retrieve_from_text_backend(self, context_id: str) -> Dict[str, Any]:
        """Retrieve context from text backend."""
        try:
            # Check if document exists in text backend
            if context_id in self.text_backend.documents:
                doc = self.text_backend.documents[context_id]
                return {
                    "found": True,
                    "data": {
                        "text": doc.text,
                        "metadata": doc.metadata,
                        "token_count": len(doc.tokens),
                    },
                }
            else:
                return {"found": False, "reason": "Context not found in text backend"}
        except Exception as e:
            return {"found": False, "error": str(e)}

    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get statistics about storage backends."""
        stats = {
            "backends": self.available_backends,
            "text_backend": None,
            "last_updated": datetime.now().isoformat(),
        }

        # Text backend statistics
        if self.text_backend:
            try:
                text_stats = self.text_backend.get_index_statistics()
                stats["text_backend"] = text_stats
            except Exception as e:
                stats["text_backend"] = {"error": str(e)}

        return stats


# Global orchestrator instance
_storage_orchestrator: Optional[EnhancedStorageOrchestrator] = None


def get_storage_orchestrator() -> Optional[EnhancedStorageOrchestrator]:
    """Get the global storage orchestrator instance."""
    return _storage_orchestrator


def initialize_storage_orchestrator(
    qdrant_client: Optional[VectorDBInitializer] = None,
    neo4j_client: Optional[Neo4jInitializer] = None,
    kv_store: Optional[ContextKV] = None,
    text_backend: Optional[TextSearchBackend] = None,
    embedding_generator: Optional[Any] = None,
) -> EnhancedStorageOrchestrator:
    """Initialize the global storage orchestrator."""
    global _storage_orchestrator
    _storage_orchestrator = EnhancedStorageOrchestrator(
        qdrant_client=qdrant_client,
        neo4j_client=neo4j_client,
        kv_store=kv_store,
        text_backend=text_backend,
        embedding_generator=embedding_generator,
    )
    logger.info("Enhanced storage orchestrator initialized")
    return _storage_orchestrator
