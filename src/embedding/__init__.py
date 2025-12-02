"""
Embedding module for Veris Memory.

Provides robust embedding generation with automatic dimension adjustment,
retry logic, caching, and health monitoring.
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

__all__ = [
    "EmbeddingService",
    "EmbeddingConfig", 
    "EmbeddingModel",
    "EmbeddingError",
    "ModelLoadError",
    "DimensionMismatchError",
    "get_embedding_service",
    "generate_embedding"
]