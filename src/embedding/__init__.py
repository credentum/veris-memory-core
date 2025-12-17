"""
Embedding module for Veris Memory.

Provides robust embedding generation with automatic dimension adjustment,
retry logic, caching, and health monitoring.

Also includes sparse embedding support for hybrid search (BM25-style).
"""

from .service import (
    EmbeddingService,
    EmbeddingConfig,
    EmbeddingModel,
    EmbeddingError,
    ModelLoadError,
    DimensionMismatchError,
    get_embedding_service,
    generate_embedding
)

from .sparse_service import (
    SparseVector,
    SparseEmbeddingService,
    get_sparse_embedding_service,
    generate_sparse_embedding,
    SPARSE_ENABLED,
)

__all__ = [
    # Dense embeddings
    "EmbeddingService",
    "EmbeddingConfig",
    "EmbeddingModel",
    "EmbeddingError",
    "ModelLoadError",
    "DimensionMismatchError",
    "get_embedding_service",
    "generate_embedding",
    # Sparse embeddings (for hybrid search)
    "SparseVector",
    "SparseEmbeddingService",
    "get_sparse_embedding_service",
    "generate_sparse_embedding",
    "SPARSE_ENABLED",
]