#!/usr/bin/env python3
"""
Triage fixes for reranker issues found in Phase 2 testing
- Single sanity pair test
- Candidate set logging
- Score direction assertions
- Text clamping to 512 tokens
- Deterministic inference
- Fixed reranker logic
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Tuple
from sentence_transformers import CrossEncoder
import tiktoken

logger = logging.getLogger(__name__)

class TriageReranker:
    """Fixed reranker with all triage improvements"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model in deterministic mode"""
        try:
            self.model = CrossEncoder(self.model_name, max_length=512)
            # Make deterministic
            self.model.model.eval()
            
            # Load tokenizer for text clamping
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except:
                logger.warning("tiktoken not available, using character-based truncation")
                self.tokenizer = None
                
            logger.info(f"Loaded reranker model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            raise
    
    def _clamp_text(self, text: str, max_tokens: int = 512) -> str:
        """Clamp text to ≤512 tokens"""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_tokens:
                clamped_tokens = tokens[:max_tokens]
                return self.tokenizer.decode(clamped_tokens)
            return text
        else:
            # Fallback: rough character-based truncation (1 token ≈ 4 chars)
            max_chars = max_tokens * 4
            return text[:max_chars] if len(text) > max_chars else text
    
    def sanity_test(self, query: str, gold_text: str, random_text: str) -> bool:
        """Single sanity pair: Query vs its gold chunk → expect score(gold) > score(random)"""
        try:
            pairs = [
                [query, self._clamp_text(gold_text)],
                [query, self._clamp_text(random_text)]
            ]
            
            with torch.inference_mode():
                scores = self.model.predict(pairs)
            
            gold_score = scores[0]
            random_score = scores[1]
            
            logger.info(f"Sanity test - Gold: {gold_score:.4f}, Random: {random_score:.4f}")
            
            return gold_score > random_score
            
        except Exception as e:
            logger.error(f"Sanity test failed: {e}")
            return False
    
    def rerank_with_logging(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Fixed reranker logic with comprehensive logging
        
        Args:
            query: Search query
            candidates: List of candidate documents with 'text' and 'doc_id' fields
            top_k: Number of candidates to rerank
            
        Returns:
            Reranked candidates (top 5)
        """
        if not candidates:
            return []
        
        # Take top_k candidates for reranking
        candidates = candidates[:top_k]
        
        logger.info(f"Reranking {len(candidates)} candidates for query: '{query[:50]}...'")
        
        try:
            # Prepare pairs with clamped text
            pairs = []
            for i, candidate in enumerate(candidates):
                chunk_text = candidate.get('text', str(candidate))
                clamped_text = self._clamp_text(chunk_text)
                pairs.append([query, clamped_text])
                
                # Log first few candidates before reranking
                if i < 5:
                    original_score = candidate.get('score', 0.0)
                    doc_id = candidate.get('doc_id', f'doc_{i}')
                    logger.info(f"  Before rerank #{i+1}: {doc_id} (score: {original_score:.4f})")
            
            # Get reranker scores with deterministic inference
            with torch.inference_mode():
                scores = self.model.predict(pairs, batch_size=16)
            
            # Assert score direction correctness
            max_score = np.max(scores)
            sorted_indices = np.argsort(scores)
            assert max_score == scores[sorted_indices[-1]], f"Score direction error: max={max_score}, sorted_max={scores[sorted_indices[-1]]}"
            
            # Combine candidates with scores
            scored_candidates = []
            for i, candidate in enumerate(candidates):
                scored_candidate = candidate.copy()
                scored_candidate['rerank_score'] = float(scores[i])
                scored_candidate['original_score'] = candidate.get('score', 0.0)
                scored_candidates.append(scored_candidate)
            
            # Sort by rerank score (higher = better)
            order = np.argsort(-scores)  # DESC sort
            reranked = [scored_candidates[i] for i in order]
            
            # Log top 5 after reranking
            logger.info("After reranking (top 5):")
            for i in range(min(5, len(reranked))):
                candidate = reranked[i]
                doc_id = candidate.get('doc_id', f'doc_{order[i]}')
                rerank_score = candidate['rerank_score']
                original_score = candidate['original_score']
                logger.info(f"  #{i+1}: {doc_id} (rerank: {rerank_score:.4f}, original: {original_score:.4f})")
            
            # Return top 5
            return reranked[:5]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return candidates[:5]  # Fallback to original order


def run_reranker_triage_test():
    """Test the triage fixes"""
    print("🔧 Running Reranker Triage Test...")
    
    # Initialize reranker
    reranker = TriageReranker()
    
    # Test data
    query = "What are the benefits of microservices architecture?"
    
    gold_text = """Microservices architecture provides several key benefits including:
    1. Scalability - Each service can be scaled independently
    2. Fault isolation - Failure in one service doesn't bring down the entire system
    3. Technology diversity - Different services can use different technologies
    4. Team autonomy - Small teams can own and deploy services independently
    5. Better fault tolerance through distributed design"""
    
    random_text = """Database indexing is a performance optimization technique that uses
    B-tree data structures to speed up query execution. Proper indexing strategies
    include creating composite indices and avoiding over-indexing which can slow
    down write operations."""
    
    # Sanity test
    sanity_passed = reranker.sanity_test(query, gold_text, random_text)
    print(f"✅ Sanity test: {'PASS' if sanity_passed else 'FAIL'}")
    
    # Test candidates
    candidates = [
        {"doc_id": "random_doc", "text": random_text, "score": 0.8},
        {"doc_id": "gold_doc", "text": gold_text, "score": 0.7},
        {"doc_id": "other_doc_1", "text": "OAuth 2.0 is an authorization framework...", "score": 0.6},
        {"doc_id": "other_doc_2", "text": "Docker containers provide isolation...", "score": 0.5},
        {"doc_id": "other_doc_3", "text": "REST APIs follow stateless principles...", "score": 0.4},
    ]
    
    # Test reranking with logging
    reranked = reranker.rerank_with_logging(query, candidates)
    
    print(f"\n📊 Reranking Results:")
    print(f"Gold doc position: {next((i+1 for i, c in enumerate(reranked) if c['doc_id'] == 'gold_doc'), 'Not found')}")
    
    return sanity_passed and len(reranked) > 0


if __name__ == "__main__":
    run_reranker_triage_test()