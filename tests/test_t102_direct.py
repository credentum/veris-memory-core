#!/usr/bin/env python3
"""
Direct T-102 Test - Testing bulletproof reranker with proper candidate format
"""

import sys
import time
sys.path.insert(0, 'src')

from src.storage.reranker_bulletproof import BulletproofReranker

def test_t102_direct():
    """Test T-102 fix with proper candidate format"""
    print("🔧 Direct T-102 Test - Bulletproof Reranker")
    print("=" * 50)
    
    reranker = BulletproofReranker(debug_mode=True)
    
    # Test case that would trigger T-102 - minimal content with scores
    query = "test query"
    candidates = [
        {
            "id": "doc_1", 
            "score": 0.5,  # Original retrieval score
            "content": "a"
        },
        {
            "id": "doc_2", 
            "score": 0.6,
            "content": "b"  
        },
        {
            "id": "doc_3",
            "score": 0.7,
            "content": "test content that matches query"
        }
    ]
    
    print(f"Input candidates:")
    for i, cand in enumerate(candidates):
        print(f"  #{i+1}: {cand['id']} (score: {cand['score']}, content: '{cand['content']}')")
    
    # Run reranking
    start_time = time.time()
    result = reranker.rerank(query, candidates)
    end_time = time.time()
    
    print(f"\nOutput after reranking:")
    for i, cand in enumerate(result):
        rerank_score = cand.get('rerank_score', 'N/A')
        orig_score = cand.get('original_score', 'N/A')
        print(f"  #{i+1}: {cand['id']} (rerank: {rerank_score}, orig: {orig_score})")
    
    # Check for T-102 bug
    rerank_scores = [cand.get('rerank_score', 0) for cand in result]
    all_zeros = all(abs(score) < 1e-9 for score in rerank_scores if score is not None)
    
    print(f"\nResults:")
    print(f"  Processing time: {(end_time - start_time) * 1000:.1f}ms")
    print(f"  Rerank scores: {rerank_scores}")
    print(f"  All zeros bug: {'❌ DETECTED' if all_zeros else '✅ FIXED'}")
    
    if all_zeros:
        print("\n❌ T-102 BUG STILL PRESENT!")
        print("   The bulletproof reranker is still returning all zero scores.")
        return False
    else:
        print("\n✅ T-102 BUG FIXED!")
        print("   The bulletproof reranker is working correctly.")
        return True

if __name__ == "__main__":
    success = test_t102_direct()
    sys.exit(0 if success else 1)