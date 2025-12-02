#!/usr/bin/env python3
"""
Simulate T-102 failure to generate pre/post candidate logs for debugging
Shows the exact format you'd see in a real reranker failure
"""

import numpy as np
from typing import List, Dict, Any

def simulate_t102_failure_logs():
    """Generate the type of logs that would appear in a T-102 failure"""
    
    print("🔍 T-102 Reranker Failure - Candidate Logs")
    print("=" * 50)
    
    # Simulate typical candidates from dense search
    query = "What are the benefits of microservices architecture?"
    
    # Mock candidates (what would come from dense/hybrid search)
    candidates = [
        {"doc_id": "doc_23", "text": "Database indexing improves query performance...", "score": 0.73},
        {"doc_id": "doc_91", "text": "OAuth 2.0 provides secure authentication...", "score": 0.71}, 
        {"doc_id": "doc_45", "text": "Microservices architecture offers scalability and fault isolation...", "score": 0.69},  # This should be #1
        {"doc_id": "doc_12", "text": "Docker containers enable application isolation...", "score": 0.67},
        {"doc_id": "doc_78", "text": "REST APIs follow stateless design principles...", "score": 0.65},
        {"doc_id": "doc_33", "text": "Machine learning requires large datasets...", "score": 0.63},
        {"doc_id": "doc_56", "text": "Cloud computing provides on-demand resources...", "score": 0.61},
        {"doc_id": "doc_89", "text": "Kubernetes orchestrates containerized applications...", "score": 0.59},
        {"doc_id": "doc_15", "text": "GraphQL enables efficient data fetching...", "score": 0.57},
        {"doc_id": "doc_67", "text": "Redis provides in-memory data caching...", "score": 0.55},
    ]
    
    print(f"Query: '{query}'")
    print(f"Candidate set size: {len(candidates)}")
    print()
    
    print("📥 PRE-RERANK (Top 5 from dense search):")
    for i, candidate in enumerate(candidates[:5]):
        print(f"  #{i+1}: {candidate['doc_id']} (dense_score: {candidate['score']:.3f})")
        print(f"      Text: {candidate['text'][:60]}...")
    print()
    
    # Simulate broken reranker behavior (common failure patterns)
    failure_scenarios = {
        "all_zero_scores": "Reranker returns all 0.0 scores",
        "nan_scores": "Reranker returns NaN values", 
        "wrong_order": "Reranker scores in wrong direction (ascending vs descending)",
        "text_extraction_fail": "Reranker gets empty/malformed text input"
    }
    
    # Show most common failure: all zero scores
    print("❌ POST-RERANK (FAILURE - All Zero Scores):")
    rerank_scores = [0.0] * len(candidates)  # Common failure pattern
    
    # Sort by rerank score (all zeros = no change)
    order = list(range(len(candidates)))  # No reordering due to zero scores
    
    for i in range(5):
        idx = order[i]
        candidate = candidates[idx]
        print(f"  #{i+1}: {candidate['doc_id']} (rerank_score: {rerank_scores[idx]:.3f}, original: {candidate['score']:.3f})")
        print(f"      Text: {candidate['text'][:60]}...")
    print()
    
    print("🐛 FAILURE ANALYSIS:")
    print("  • All rerank scores = 0.0 (should be in range [-10, +10])")  
    print("  • No reordering occurred (dense order preserved)")
    print("  • Gold document (doc_45) stayed at position #3")
    print("  • Expected: Gold document should move to position #1")
    print()
    
    print("🔧 LIKELY ROOT CAUSES:")
    print("  1. Text extraction failed - reranker got empty/malformed input")
    print("  2. Model loading failed - using dummy/fallback model")  
    print("  3. Cross-encoder predict() returned wrong format")
    print("  4. Score normalization/scaling bug")
    print("  5. Torch device mismatch (GPU/CPU)")
    
    # Show what successful reranking should look like
    print()
    print("✅ EXPECTED POST-RERANK (Successful):")
    # Mock successful reranker scores
    successful_rerank_scores = [
        -2.1,  # doc_23 (database) - not relevant  
        -1.8,  # doc_91 (oauth) - not relevant
        4.2,   # doc_45 (microservices) - highly relevant!
        0.3,   # doc_12 (docker) - somewhat relevant
        -0.9,  # doc_78 (REST) - not very relevant  
        -3.2,  # doc_33 (ML) - not relevant
        1.1,   # doc_56 (cloud) - somewhat relevant 
        0.7,   # doc_89 (k8s) - somewhat relevant
        -1.5,  # doc_15 (graphql) - not relevant
        -2.8   # doc_67 (redis) - not relevant
    ]
    
    # Sort by rerank score (descending)  
    successful_order = np.argsort(-np.array(successful_rerank_scores))
    
    for i in range(5):
        idx = successful_order[i]
        candidate = candidates[idx]
        print(f"  #{i+1}: {candidate['doc_id']} (rerank_score: {successful_rerank_scores[idx]:.3f}, original: {candidate['score']:.3f})")
        print(f"      Text: {candidate['text'][:60]}...")
    
    print()
    print("✅ SUCCESS METRICS:")
    print("  • Gold document (doc_45) moved to position #1")
    print("  • Rerank scores span reasonable range [-3.2, +4.2]") 
    print("  • Scores strictly descending: 4.2 > 1.1 > 0.7 > 0.3 > -0.9")
    print("  • Precision@1 = 1.0 (would be 0.0 in failure case)")

if __name__ == "__main__":
    simulate_t102_failure_logs()