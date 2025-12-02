#!/usr/bin/env python3
"""
reranker.py: Cross-encoder reranker for improving retrieval precision

This module implements a cross-encoder based reranker that takes the top-k 
results from vector/hybrid search and reorders them based on semantic 
relevance to improve precision@1.

Features:
- Cross-encoder models (ms-marco-MiniLM-L-6-v2, bge-reranker-v2-m3)
- Configurable via environment variables
- Optional reranking (can be disabled)
- Caching for model loading
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not available - reranker disabled")


class ContextReranker:
    """Cross-encoder based reranker for context retrieval results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the reranker with configuration.
        
        Args:
            config: Optional configuration dict with reranker settings
        """
        self.config = config or {}
        
        # Configuration with environment variable fallbacks
        self.enabled = self._get_bool_config("RERANKER_ENABLED", "reranker.enabled", True)
        self.model_name = self._get_str_config(
            "RERANKER_MODEL", 
            "reranker.model", 
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.top_k = self._get_int_config("RERANK_TOP_K", "reranker.top_k", 20)
        self.return_k = self._get_int_config("RERANK_RETURN_K", "reranker.return_k", 5)
        self.score_threshold = self._get_float_config(
            "RERANKER_SCORE_THRESHOLD", 
            "reranker.score_threshold", 
            -1000.0  # Very permissive threshold
        )
        
        self.model: Optional[CrossEncoder] = None
        self.model_loaded = False
        
        if self.enabled and not CROSS_ENCODER_AVAILABLE:
            logger.error("Reranker enabled but sentence-transformers not available")
            self.enabled = False
    
    def _get_bool_config(self, env_key: str, config_key: str, default: bool) -> bool:
        """Get boolean config from environment or config dict."""
        env_val = os.getenv(env_key)
        if env_val is not None:
            return env_val.lower() in ('true', '1', 'yes', 'on')
        
        # Navigate nested config keys like "reranker.enabled"
        value = self.config
        for key in config_key.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return bool(value) if value is not None else default
    
    def _get_str_config(self, env_key: str, config_key: str, default: str) -> str:
        """Get string config from environment or config dict."""
        env_val = os.getenv(env_key)
        if env_val is not None:
            return env_val
        
        value = self.config
        for key in config_key.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return str(value) if value is not None else default
    
    def _get_int_config(self, env_key: str, config_key: str, default: int) -> int:
        """Get integer config from environment or config dict."""
        env_val = os.getenv(env_key)
        if env_val is not None:
            try:
                return int(env_val)
            except ValueError:
                logger.warning(f"Invalid integer value for {env_key}: {env_val}")
                return default
        
        value = self.config
        for key in config_key.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        try:
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _get_float_config(self, env_key: str, config_key: str, default: float) -> float:
        """Get float config from environment or config dict."""
        env_val = os.getenv(env_key)
        if env_val is not None:
            try:
                return float(env_val)
            except ValueError:
                logger.warning(f"Invalid float value for {env_key}: {env_val}")
                return default
        
        value = self.config
        for key in config_key.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _load_model(self) -> bool:
        """Load the cross-encoder model."""
        if self.model_loaded:
            return self.model is not None
        
        if not self.enabled or not CROSS_ENCODER_AVAILABLE:
            self.model_loaded = True
            return False
        
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name, max_length=512)
            self.model_loaded = True
            logger.info("Cross-encoder model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model {self.model_name}: {e}")
            self.model = None
            self.model_loaded = True
            return False
    
    def _extract_text_content(self, result: Dict[str, Any]) -> str:
        """Extract text content from a search result for reranking.
        
        Args:
            result: Search result dictionary
            
        Returns:
            Extracted text content
        """
        # Try different possible content locations
        if "payload" in result:
            payload = result["payload"]
            if isinstance(payload, dict):
                # Try content.text first (common format)
                if "content" in payload:
                    content = payload["content"]
                    if isinstance(content, dict) and "text" in content:
                        return str(content["text"])
                    elif isinstance(content, str):
                        return content
                
                # Try metadata.content
                if "metadata" in payload:
                    metadata = payload["metadata"]
                    if isinstance(metadata, dict) and "content" in metadata:
                        return str(metadata["content"])
                
                # Try direct text fields
                for field in ["text", "description", "summary", "title"]:
                    if field in payload:
                        return str(payload[field])
        
        # Fallback to converting the whole result to string
        content_str = str(result)
        # Truncate very long content
        return content_str[:2000] if len(content_str) > 2000 else content_str
    
    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank search results using cross-encoder.
        
        Args:
            query: The search query
            results: List of search results to rerank
            
        Returns:
            Reranked list of results (limited to return_k)
        """
        if not self.enabled:
            logger.debug("Reranker disabled, returning original results")
            return results[:self.return_k]
        
        if not results:
            return results
        
        if not self._load_model():
            logger.warning("Reranker model not available, returning original results")
            return results[:self.return_k]
        
        try:
            # Take only top_k results for reranking to manage computational cost
            candidates = results[:self.top_k]
            
            if len(candidates) <= 1:
                return candidates
            
            # Prepare query-document pairs for cross-encoder
            pairs = []
            for result in candidates:
                doc_text = self._extract_text_content(result)
                pairs.append([query, doc_text])
            
            # Get cross-encoder scores
            logger.debug(f"Reranking {len(pairs)} results for query: {query[:100]}...")
            scores = self.model.predict(pairs)
            
            # Combine results with new scores
            scored_results = []
            for i, result in enumerate(candidates):
                new_result = result.copy()
                new_result["rerank_score"] = float(scores[i])
                new_result["original_score"] = result.get("score", 0.0)
                new_result["source"] = f"{result.get('source', 'unknown')}_reranked"
                scored_results.append(new_result)
            
            # Sort by rerank score (higher is better for cross-encoders)
            scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            # Filter by score threshold and return top return_k
            filtered_results = [
                r for r in scored_results 
                if r["rerank_score"] >= self.score_threshold
            ]
            
            final_results = filtered_results[:self.return_k]
            
            logger.info(
                f"Reranked {len(candidates)} → {len(final_results)} results. "
                f"Top score: {final_results[0]['rerank_score']:.4f}" 
                if final_results else "No results after filtering"
            )
            
            return final_results
        
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback to original results
            return results[:self.return_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics and configuration."""
        return {
            "enabled": self.enabled,
            "model_name": self.model_name,
            "model_loaded": self.model_loaded,
            "top_k": self.top_k,
            "return_k": self.return_k,
            "score_threshold": self.score_threshold,
            "cross_encoder_available": CROSS_ENCODER_AVAILABLE
        }


# Global reranker instance for reuse across requests
_global_reranker: Optional[ContextReranker] = None


def get_reranker(config: Optional[Dict[str, Any]] = None) -> ContextReranker:
    """Get or create global reranker instance.
    
    Args:
        config: Optional configuration dict
        
    Returns:
        ContextReranker instance
    """
    global _global_reranker
    
    if _global_reranker is None:
        _global_reranker = ContextReranker(config)
    
    return _global_reranker


def rerank_results(
    query: str, 
    results: List[Dict[str, Any]], 
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Convenience function to rerank results.
    
    Args:
        query: Search query
        results: List of search results
        config: Optional configuration
        
    Returns:
        Reranked results
    """
    reranker = get_reranker(config)
    return reranker.rerank(query, results)