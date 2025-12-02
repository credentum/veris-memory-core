#!/usr/bin/env python3
"""
Bulletproof reranker with hardened text extraction and fail-safes
Fixes the classic "all rerank_score = 0.000" bug from empty text inputs
"""

import logging
import time
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class RerankerMetrics:
    """Simple metrics tracker"""
    def __init__(self) -> None:
        self.counters: Dict[str, int] = {}
        self.timers: List[float] = []
    
    def counter(self, name: str) -> 'RerankerMetrics':
        if name not in self.counters:
            self.counters[name] = 0
        return self
    
    def inc(self) -> None:
        # Get last counter name and increment
        if self.counters:
            last_key = list(self.counters.keys())[-1]
            self.counters[last_key] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "counters": self.counters.copy(),
            "avg_latency_ms": np.mean(self.timers) if self.timers else 0.0
        }

# Global metrics instance
metrics = RerankerMetrics()

def extract_chunk_text(payload: Dict[str, Any], debug: bool = False) -> str:
    """
    Bulletproof text extraction - handles multiple payload shapes
    
    Try common shapes:
    - {"text": "..."}
    - {"content": {"text": "..."}}
    - {"content": "..."}  # plain string
    - {"content": [{"type":"text","text":"..."}]}  # tool-style
    - {"payload": {"content": ...}}  # nested
    """
    if not payload:
        if debug:
            logger.debug("extract_chunk_text: empty payload")
        return ""

    # Direct text field
    if isinstance(payload.get("text"), str):
        text = payload["text"]
        if debug:
            logger.debug(f"extract_chunk_text: found direct text, len={len(text)}")
        return text

    # Content variations
    content = payload.get("content")
    if isinstance(content, str):
        if debug:
            logger.debug(f"extract_chunk_text: found content string, len={len(content)}")
        return content
    
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        text = content["text"]
        if debug:
            logger.debug(f"extract_chunk_text: found content.text, len={len(text)}")
        return text
    
    if isinstance(content, list):
        # Tool-style content array
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif "content" in item and isinstance(item["content"], str):
                    parts.append(item["content"])
        
        if parts:
            result = "\n".join(parts)
            if debug:
                logger.debug(f"extract_chunk_text: joined {len(parts)} parts, total len={len(result)}")
            return result

    # Nested payload (common in MCP responses)
    if "payload" in payload and isinstance(payload["payload"], dict):
        nested_result = extract_chunk_text(payload["payload"], debug)
        if nested_result:
            if debug:
                logger.debug(f"extract_chunk_text: found nested payload text, len={len(nested_result)}")
            return nested_result

    # Alternative field names
    for field in ("body", "markdown", "raw", "description", "summary", "title"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            if debug:
                logger.debug(f"extract_chunk_text: found {field}, len={len(value)}")
            return value

    # Last resort: convert entire payload to string
    if payload:
        fallback = str(payload)
        if debug:
            logger.debug(f"extract_chunk_text: fallback to str(payload), len={len(fallback)}")
        return fallback

    if debug:
        logger.debug("extract_chunk_text: no text found, returning empty")
    return ""

def clamp_for_rerank(txt: str, max_chars: int = 4000) -> str:
    """
    Clamp text for reranker input (~512 tokens ≈ 2k–4k chars)
    Also strips boilerplate and empty lines
    """
    text = (txt or "").strip()
    
    # Strip obvious boilerplate and very short lines
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 2:  # Skip very short lines
            lines.append(line)
    
    text = "\n".join(lines)
    
    # Clamp to max chars
    if len(text) > max_chars:
        text = text[:max_chars]
    
    return text

class BulletproofReranker:
    """Bulletproof reranker with fail-safes and debugging"""
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 debug_mode: bool = False,
                 auto_disable_threshold: int = 5):
        self.model_name = model_name
        self.debug_mode = debug_mode
        self.auto_disable_threshold = auto_disable_threshold
        self.model: Optional[CrossEncoder] = None
        self.enabled = True
        self.request_count = 0
        self._load_model()
    
    def _load_model(self) -> None:
        """Load cross-encoder model with error handling"""
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name, max_length=512)
            # Make deterministic
            self.model.model.eval()
            logger.info("✅ Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load cross-encoder model: {e}")
            self.model = None
            self.enabled = False
    
    def _debug_log_candidates(self, query: str, candidates: List[Dict], texts: List[str], scores: Optional[List[float]] = None) -> None:
        """Debug logging for first few requests per boot"""
        if not self.debug_mode or self.request_count > 10:
            return
        
        logger.info(f"🔍 RERANKER DEBUG - Request #{self.request_count}")
        logger.info(f"Query: '{query[:100]}...' (len={len(query)})")
        logger.info(f"Candidates: {len(candidates)}")
        
        for i, (candidate, text) in enumerate(zip(candidates[:5], texts[:5])):
            candidate_id = candidate.get("id", candidate.get("context_id", f"candidate_{i}"))
            dense_score = candidate.get("score", candidate.get("dense_score", 0.0))
            text_len = len(text) if text else 0
            rerank_score = scores[i] if scores and i < len(scores) else "N/A"
            
            logger.info(f"  #{i+1}: {candidate_id} | dense={dense_score:.3f} | text_len={text_len} | rerank={rerank_score}")
            
            if text_len == 0:
                logger.warning(f"    ❌ EMPTY TEXT for {candidate_id} - payload keys: {list(candidate.keys())}")
            elif text_len < 10:
                logger.warning(f"    ⚠️  VERY SHORT TEXT for {candidate_id}: '{text}'")
    
    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Bulletproof reranking with fail-safes
        
        Args:
            query: Search query
            candidates: List of candidates with payload/content
            
        Returns:
            Reranked candidates (fail-safe to original order if issues)
        """
        start_time = time.time()
        self.request_count += 1
        
        if not self.enabled or not self.model or not candidates:
            return candidates
        
        try:
            # Extract texts with robust extraction
            texts = []
            for candidate in candidates:
                # Try multiple payload locations
                payload = candidate.get("payload", candidate)
                text = extract_chunk_text(payload, debug=self.debug_mode)
                clamped_text = clamp_for_rerank(text)
                texts.append(clamped_text)
            
            # Hard fail-safe: if all empty, abort rerank
            empties = sum(1 for t in texts if not t.strip())
            if empties == len(texts):
                metrics.counter("reranker_all_empty").inc()
                logger.error(f"❌ RERANKER: All {len(texts)} candidates have empty text - aborting rerank")
                self._debug_log_candidates(query, candidates, texts)
                return candidates
            
            if empties > len(texts) * 0.5:
                logger.warning(f"⚠️  RERANKER: {empties}/{len(texts)} candidates have empty text")
            
            # Build pairs and score
            pairs = []
            for text in texts:
                # Use space for completely empty texts (model needs something)
                pair_text = text if text.strip() else " "
                pairs.append([query, pair_text])
            
            # Get cross-encoder scores with deterministic inference
            with torch.inference_mode():
                scores = self.model.predict(pairs)
            
            # Convert to list if numpy array
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            
            # Sanity check: no all-zeros after model call
            if all(abs(s) < 1e-9 for s in scores):
                metrics.counter("reranker_all_zero_scores").inc()
                logger.error("❌ RERANKER: All scores are zero - model prediction failed")
                self._debug_log_candidates(query, candidates, texts, scores)
                
                # Auto-disable if too many zero-score failures
                if metrics.counters.get("reranker_all_zero_scores", 0) >= self.auto_disable_threshold:
                    logger.error(f"❌ RERANKER: Auto-disabling after {self.auto_disable_threshold} zero-score failures")
                    self.enabled = False
                
                return candidates
            
            # Debug logging
            self._debug_log_candidates(query, candidates, texts, scores)
            
            # Create scored candidates
            scored_candidates = []
            for i, candidate in enumerate(candidates):
                scored = candidate.copy()
                scored["rerank_score"] = float(scores[i])
                scored["original_score"] = candidate.get("score", 0.0)
                scored["text_length"] = len(texts[i]) if i < len(texts) else 0
                scored_candidates.append(scored)
            
            # Sort by rerank score (higher = better, descending)
            order = np.argsort(-np.asarray(scores))
            reranked = [scored_candidates[i] for i in order]
            
            # Assert scores are descending
            rerank_scores = [reranked[i]["rerank_score"] for i in range(len(reranked))]
            if len(rerank_scores) > 1:
                for i in range(len(rerank_scores) - 1):
                    if rerank_scores[i] < rerank_scores[i + 1]:
                        logger.warning(f"⚠️  Score ordering issue at positions {i}, {i+1}: {rerank_scores[i]} < {rerank_scores[i+1]}")
                        break
            
            # Track metrics
            latency_ms = (time.time() - start_time) * 1000
            metrics.timers.append(latency_ms)
            metrics.counter("reranker_invocations").inc()
            
            logger.info(f"✅ RERANKER: Scored {len(candidates)} candidates in {latency_ms:.1f}ms")
            logger.info(f"   Top score: {rerank_scores[0]:.4f}, Bottom: {rerank_scores[-1]:.4f}")
            
            return reranked
            
        except Exception as e:
            logger.error(f"❌ RERANKER: Exception during reranking: {e}")
            metrics.counter("reranker_exceptions").inc()
            return candidates
    
    def debug_rerank(self, query: str, candidate_payloads: List[Dict]) -> List[Dict[str, Any]]:
        """
        Debug endpoint to inspect exactly what reranker sees
        Returns: [{"id": ..., "text_len": ..., "dense_score": ..., "rerank_score": ...}]
        """
        results = []
        
        for i, payload in enumerate(candidate_payloads):
            candidate_id = payload.get("id", payload.get("context_id", f"candidate_{i}"))
            
            # Extract text
            text = extract_chunk_text(payload, debug=True)
            clamped_text = clamp_for_rerank(text)
            
            # Get scores if model available
            rerank_score = None
            if self.model and self.enabled:
                try:
                    with torch.inference_mode():
                        score = self.model.predict([[query, clamped_text if clamped_text else " "]])
                        rerank_score = float(score[0]) if hasattr(score, '__getitem__') else float(score)
                except Exception as e:
                    rerank_score = f"ERROR: {str(e)}"
            
            results.append({
                "id": candidate_id,
                "text_len": len(text),
                "clamped_len": len(clamped_text),
                "dense_score": payload.get("score", 0.0),
                "rerank_score": rerank_score,
                "text_preview": (clamped_text[:100] + "...") if len(clamped_text) > 100 else clamped_text,
                "payload_keys": list(payload.keys())
            })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics and health"""
        return {
            "enabled": self.enabled,
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "request_count": self.request_count,
            "debug_mode": self.debug_mode,
            "auto_disable_threshold": self.auto_disable_threshold,
            "metrics": metrics.get_stats()
        }

# Global instance
_bulletproof_reranker: Optional[BulletproofReranker] = None

def get_bulletproof_reranker(debug_mode: bool = False) -> BulletproofReranker:
    """Get or create global bulletproof reranker"""
    global _bulletproof_reranker
    
    if _bulletproof_reranker is None:
        _bulletproof_reranker = BulletproofReranker(debug_mode=debug_mode)
    
    return _bulletproof_reranker

def rerank_bulletproof(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convenience function for bulletproof reranking"""
    reranker = get_bulletproof_reranker()
    return reranker.rerank(query, candidates)

if __name__ == "__main__":
    # Test the bulletproof reranker
    print("🔧 Testing Bulletproof Reranker...")
    
    reranker = BulletproofReranker(debug_mode=True)
    
    # Test data with problematic payloads
    query = "What are microservices benefits?"
    
    test_candidates = [
        {"id": "empty_payload", "payload": {}},
        {"id": "nested_content", "payload": {"content": {"text": "Microservices provide scalability..."}}},
        {"id": "direct_text", "text": "Database indexing improves performance..."},
        {"id": "tool_style", "payload": {"content": [{"type": "text", "text": "OAuth provides security..."}]}},
        {"id": "string_content", "payload": {"content": "Docker enables containerization..."}},
    ]
    
    results = reranker.rerank(query, test_candidates)
    
    print(f"\n✅ Reranked {len(results)} candidates:")
    for i, result in enumerate(results):
        print(f"  #{i+1}: {result['id']} (score: {result.get('rerank_score', 'N/A'):.3f})")
    
    print(f"\n📊 Stats: {reranker.get_stats()}")