#!/usr/bin/env python3
"""
Debug endpoints for reranker inspection and troubleshooting
SECURITY: Internal-only endpoints with authentication for production safety
"""

import json
import os
import hashlib
import time
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
from ..storage.reranker_bulletproof import get_bulletproof_reranker

# Security configuration - Debug endpoints disabled by default for production security
DEBUG_ENABLED = os.getenv("DEBUG_ENDPOINTS_ENABLED", "false").lower() == "true"
DEBUG_TOKEN = os.getenv("DEBUG_TOKEN")  # Required in production for security
REQUIRE_AUTH = os.getenv("REQUIRE_DEBUG_AUTH", "true").lower() == "true"

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("DEBUG_RATE_LIMIT_REQUESTS", "10"))  # requests per window
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("DEBUG_RATE_LIMIT_WINDOW", "60"))  # 1 minute window
RATE_LIMIT_ENABLED = os.getenv("DEBUG_RATE_LIMIT_ENABLED", "true").lower() == "true"

# Rate limiting storage (in production, use Redis or similar)
rate_limit_requests: Dict[str, deque] = defaultdict(deque)

def check_rate_limit(client_id: str) -> tuple[bool, Dict[str, Any]]:
    """
    Check if client is within rate limits
    
    Args:
        client_id: Unique identifier for the client (IP, token hash, etc.)
        
    Returns:
        Tuple of (allowed: bool, rate_limit_info: dict)
    """
    if not RATE_LIMIT_ENABLED:
        return True, {"rate_limit_enabled": False}
    
    current_time = time.time()
    window_start = current_time - RATE_LIMIT_WINDOW_SECONDS
    
    # Clean old requests outside the window
    client_requests = rate_limit_requests[client_id]
    while client_requests and client_requests[0] < window_start:
        client_requests.popleft()
    
    # Check if within limit
    requests_in_window = len(client_requests)
    remaining_requests = max(0, RATE_LIMIT_REQUESTS - requests_in_window)
    
    if requests_in_window >= RATE_LIMIT_REQUESTS:
        # Rate limited
        reset_time = client_requests[0] + RATE_LIMIT_WINDOW_SECONDS
        return False, {
            "rate_limit_enabled": True,
            "requests_in_window": requests_in_window,
            "limit": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
            "remaining": 0,
            "reset_time": reset_time,
            "retry_after_seconds": int(reset_time - current_time)
        }
    
    # Add this request to the window
    client_requests.append(current_time)
    
    return True, {
        "rate_limit_enabled": True,
        "requests_in_window": requests_in_window + 1,
        "limit": RATE_LIMIT_REQUESTS,
        "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
        "remaining": remaining_requests - 1
    }

def get_client_id(params: Dict[str, Any]) -> str:
    """
    Generate a client ID for rate limiting
    
    Uses token hash if available, otherwise fallback to generic client
    """
    debug_token = params.get("debug_token") or params.get("token")
    if debug_token:
        # Use hash of token for privacy
        return hashlib.sha256(str(debug_token).encode()).hexdigest()[:16]
    
    # Fallback to generic client (all unauthenticated requests share same limit)
    return "anonymous"

def verify_debug_access(params: Dict[str, Any]) -> bool:
    """
    Verify debug endpoint access with security checks
    
    Args:
        params: Request parameters that may contain auth token
        
    Returns:
        True if access is allowed, False otherwise
    """
    # Check if debug endpoints are enabled
    if not DEBUG_ENABLED:
        return False
    
    # Skip auth check if explicitly disabled (development only)
    if not REQUIRE_AUTH:
        return True
    
    # Require token in production
    if not DEBUG_TOKEN:
        return False  # No token configured
    
    # Check provided token
    provided_token = params.get("debug_token") or params.get("token")
    if not provided_token:
        return False
    
    # Secure token comparison (constant time)
    expected_hash = hashlib.sha256(DEBUG_TOKEN.encode()).hexdigest()
    provided_hash = hashlib.sha256(str(provided_token).encode()).hexdigest()
    
    return expected_hash == provided_hash

def require_debug_auth(func):
    """Decorator to require debug authentication and enforce rate limiting"""
    async def wrapper(params: Dict[str, Any]) -> Dict[str, Any]:
        # Check authentication first
        if not verify_debug_access(params):
            return {
                "error": "Debug endpoints disabled or authentication required",
                "hint": "Set DEBUG_ENDPOINTS_ENABLED=true and provide debug_token",
                "timestamp": time.time()
            }
        
        # Check rate limits
        client_id = get_client_id(params)
        rate_allowed, rate_info = check_rate_limit(client_id)
        
        if not rate_allowed:
            return {
                "error": "Rate limit exceeded",
                "rate_limit": rate_info,
                "hint": f"Try again in {rate_info.get('retry_after_seconds', 60)} seconds",
                "timestamp": time.time()
            }
        
        # Execute the function and include rate limit info in response
        result = await func(params)
        if isinstance(result, dict):
            result["rate_limit"] = rate_info
        
        return result
    return wrapper

@require_debug_auth
async def debug_rerank_endpoint(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Debug endpoint: /debug/rerank
    
    Input: {
        "query": "...", 
        "candidate_ids": ["doc_23","doc_91","doc_45"],
        "candidate_payloads": [{"content": "..."), ...]  # optional, for testing extraction
    }
    
    Returns: [{
        "id": "doc_23",
        "text_len": 0,           # ← KEY DIAGNOSTIC
        "clamped_len": 0,
        "dense_score": 0.73,
        "rerank_score": 0.0,     # ← Shows if scoring works
        "text_preview": "...",
        "payload_keys": ["content", "metadata"]
    }]
    """
    query = params.get("query", "")
    candidate_ids = params.get("candidate_ids", [])
    candidate_payloads = params.get("candidate_payloads", [])
    
    if not query:
        return {"error": "query parameter required"}
    
    # If payloads provided directly, use those
    if candidate_payloads:
        payloads = candidate_payloads
    else:
        # Otherwise, create mock payloads for the IDs
        # In real implementation, this would fetch from the vector store
        payloads = []
        for candidate_id in candidate_ids:
            # Mock payload - in real system, fetch from storage
            payloads.append({
                "id": candidate_id,
                "content": f"Mock content for {candidate_id}",
                "score": 0.5
            })
    
    # Get bulletproof reranker and run debug
    reranker = get_bulletproof_reranker(debug_mode=True)
    debug_results = reranker.debug_rerank(query, payloads)
    
    return {
        "query": query,
        "candidate_count": len(debug_results),
        "results": debug_results,
        "reranker_stats": reranker.get_stats()
    }

@require_debug_auth
async def debug_text_extraction_endpoint(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Debug endpoint: /debug/text_extraction
    
    Test text extraction on various payload shapes
    """
    from ..storage.reranker_bulletproof import extract_chunk_text, clamp_for_rerank
    
    # Test payloads with different shapes
    test_payloads = [
        {"name": "direct_text", "payload": {"text": "Direct text field"}},
        {"name": "nested_content", "payload": {"content": {"text": "Nested content.text"}}},
        {"name": "string_content", "payload": {"content": "Plain string content"}},
        {"name": "tool_array", "payload": {"content": [{"type": "text", "text": "Tool-style array"}]}},
        {"name": "empty_payload", "payload": {}},
        {"name": "nested_payload", "payload": {"payload": {"content": "Double nested"}}},
        {"name": "alternative_fields", "payload": {"body": "Body field content"}},
    ]
    
    # Add any user-provided payloads
    user_payloads = params.get("test_payloads", [])
    for i, payload in enumerate(user_payloads):
        test_payloads.append({"name": f"user_{i}", "payload": payload})
    
    results = []
    for test in test_payloads:
        try:
            extracted = extract_chunk_text(test["payload"], debug=True)
            clamped = clamp_for_rerank(extracted)
            
            results.append({
                "name": test["name"],
                "payload_keys": list(test["payload"].keys()),
                "extracted_text": extracted[:100] + "..." if len(extracted) > 100 else extracted,
                "extracted_len": len(extracted),
                "clamped_len": len(clamped),
                "success": len(extracted) > 0
            })
        except Exception as e:
            results.append({
                "name": test["name"],
                "error": str(e),
                "success": False
            })
    
    return {
        "test_count": len(results),
        "results": results,
        "success_rate": sum(1 for r in results if r.get("success", False)) / len(results) if results else 0.0
    }

@require_debug_auth
async def debug_reranker_health_endpoint(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Debug endpoint: /debug/reranker_health
    
    Check reranker component health and metrics
    """
    reranker = get_bulletproof_reranker()
    stats = reranker.get_stats()
    
    # Health checks
    health = {
        "overall": "healthy",
        "issues": []
    }
    
    # Check for common issues
    metrics_data = stats.get("metrics", {})
    counters = metrics_data.get("counters", {})
    
    if counters.get("reranker_all_empty", 0) > 0:
        health["issues"].append(f"Text extraction failures: {counters['reranker_all_empty']}")
        health["overall"] = "degraded"
    
    if counters.get("reranker_all_zero_scores", 0) > 0:
        health["issues"].append(f"Zero-score failures: {counters['reranker_all_zero_scores']}")
        health["overall"] = "degraded"
    
    if counters.get("reranker_exceptions", 0) > 0:
        health["issues"].append(f"Exceptions: {counters['reranker_exceptions']}")
        health["overall"] = "degraded"
    
    if not stats.get("enabled", False):
        health["issues"].append("Reranker is disabled")
        health["overall"] = "unhealthy"
    
    if not stats.get("model_loaded", False):
        health["issues"].append("Model not loaded")
        health["overall"] = "unhealthy"
    
    return {
        "health": health,
        "stats": stats,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# Debug endpoint registry
DEBUG_ENDPOINTS = {
    "debug_rerank": debug_rerank_endpoint,
    "debug_text_extraction": debug_text_extraction_endpoint,
    "debug_reranker_health": debug_reranker_health_endpoint,
}

async def handle_debug_endpoint(endpoint_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle debug endpoint requests"""
    if endpoint_name not in DEBUG_ENDPOINTS:
        return {"error": f"Unknown debug endpoint: {endpoint_name}"}
    
    try:
        return await DEBUG_ENDPOINTS[endpoint_name](params)
    except Exception as e:
        return {"error": f"Debug endpoint error: {str(e)}"}

if __name__ == "__main__":
    # Test debug endpoints
    import asyncio
    
    async def test_debug_endpoints():
        print("🔍 Testing Debug Endpoints...")
        
        # Test text extraction
        print("\n1. Text Extraction Test:")
        result = await debug_text_extraction_endpoint({})
        print(f"   Success rate: {result['success_rate']:.1%}")
        
        # Test reranker health
        print("\n2. Reranker Health:")
        health = await debug_reranker_health_endpoint({})
        print(f"   Overall health: {health['health']['overall']}")
        
        # Test rerank debug
        print("\n3. Rerank Debug:")
        debug_result = await debug_rerank_endpoint({
            "query": "What are microservices benefits?",
            "candidate_payloads": [
                {"content": "Microservices provide scalability..."},
                {"text": "Database indexing improves performance..."},
                {}  # Empty payload to test extraction
            ]
        })
        print(f"   Processed {debug_result['candidate_count']} candidates")
        for result in debug_result['results']:
            print(f"     {result['id']}: text_len={result['text_len']}, score={result['rerank_score']}")
    
    asyncio.run(test_debug_endpoints())