#!/usr/bin/env python3
"""
Test script to reproduce GitHub issue 35 problems.

This script tests the exact scenarios reported in the issue:
1. Store context with name "Matt"
2. Store context with favorite color "green" 
3. Query for name and color
4. Verify semantic search works correctly
"""

import asyncio
import aiohttp
import json
import time
import sys

# Test server URL
BASE_URL = "http://localhost:8000"

async def test_issue_35_reproduction():
    """Reproduce the exact scenarios from GitHub issue 35."""
    
    print("🔍 Testing GitHub Issue 35 - Vector Search and Storage Issues")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Check system health
        print("\n1️⃣ Testing system health...")
        async with session.get(f"{BASE_URL}/health") as resp:
            health = await resp.json()
            print(f"Health status: {health.get('status', 'unknown')}")
            print(f"Services: {health.get('services', {})}")
            
            if health.get('status') != 'healthy':
                print("⚠️ System not healthy - continuing with degraded testing")
        
        # Test 2: Store context - "My name is Matt"
        print("\n2️⃣ Storing context: 'My name is Matt'...")
        matt_payload = {
            "content": {
                "user_message": "My name is Matt",
                "assistant_response": "Nice to meet you, Matt!"
            },
            "type": "log",
            "metadata": {"user_id": "test_user", "timestamp": time.time()}
        }
        
        async with session.post(f"{BASE_URL}/tools/store_context", json=matt_payload) as resp:
            matt_result = await resp.json()
            print(f"Store result: {matt_result}")
            
            if matt_result.get('success'):
                print(f"✅ Stored successfully - ID: {matt_result.get('id')}")
                print(f"   Vector ID: {matt_result.get('vector_id')}")
                print(f"   Graph ID: {matt_result.get('graph_id')}")
                
                if matt_result.get('vector_id') is None:
                    print("❌ ISSUE CONFIRMED: vector_id is null!")
            else:
                print(f"❌ Storage failed: {matt_result}")
        
        # Test 3: Store context - "My favorite color is green"
        print("\n3️⃣ Storing context: 'My favorite color is green'...")
        color_payload = {
            "content": {
                "user_message": "My favorite color is green",
                "assistant_response": "Green is a great color!"
            },
            "type": "log", 
            "metadata": {"user_id": "test_user", "timestamp": time.time()}
        }
        
        async with session.post(f"{BASE_URL}/tools/store_context", json=color_payload) as resp:
            color_result = await resp.json()
            print(f"Store result: {color_result}")
            
            if color_result.get('success'):
                print(f"✅ Stored successfully - ID: {color_result.get('id')}")
                print(f"   Vector ID: {color_result.get('vector_id')}")
                print(f"   Graph ID: {color_result.get('graph_id')}")
                
                if color_result.get('vector_id') is None:
                    print("❌ ISSUE CONFIRMED: vector_id is null!")
            else:
                print(f"❌ Storage failed: {color_result}")
        
        # Wait a moment for indexing
        print("\n⏳ Waiting 2 seconds for indexing...")
        await asyncio.sleep(2)
        
        # Test 4: Query for name
        print("\n4️⃣ Querying: 'What's my name?'...")
        name_query = {
            "query": "What's my name?",
            "search_mode": "hybrid",
            "limit": 5
        }
        
        async with session.post(f"{BASE_URL}/tools/retrieve_context", json=name_query) as resp:
            name_results = await resp.json()
            print(f"Query results: {name_results}")
            
            if name_results.get('success'):
                results = name_results.get('results', [])
                print(f"✅ Found {len(results)} results")
                
                # Check if Matt context is found
                matt_found = False
                for result in results:
                    content = result.get('content', {})
                    if 'Matt' in str(content):
                        matt_found = True
                        print(f"✅ CORRECT: Found Matt context with score {result.get('score', 0)}")
                        break
                
                if not matt_found:
                    print("❌ ISSUE CONFIRMED: Semantic search not finding 'Matt' for name query!")
                    
            else:
                print(f"❌ Query failed: {name_results}")
        
        # Test 5: Query for favorite color  
        print("\n5️⃣ Querying: 'What's my favorite color?'...")
        color_query = {
            "query": "What's my favorite color?",
            "search_mode": "hybrid", 
            "limit": 5
        }
        
        async with session.post(f"{BASE_URL}/tools/retrieve_context", json=color_query) as resp:
            color_results = await resp.json()
            print(f"Query results: {color_results}")
            
            if color_results.get('success'):
                results = color_results.get('results', [])
                print(f"✅ Found {len(results)} results")
                
                # Check if green context is found
                green_found = False
                for result in results:
                    content = result.get('content', {})
                    if 'green' in str(content).lower():
                        green_found = True
                        print(f"✅ CORRECT: Found green context with score {result.get('score', 0)}")
                        break
                
                if not green_found:
                    print("❌ ISSUE CONFIRMED: Semantic search not finding 'green' for color query!")
                    
            else:
                print(f"❌ Query failed: {color_results}")
        
        # Test 6: Generic query that should NOT match
        print("\n6️⃣ Testing irrelevant query: 'Hi'...")
        hi_query = {
            "query": "Hi",
            "search_mode": "hybrid",
            "limit": 5
        }
        
        async with session.post(f"{BASE_URL}/tools/retrieve_context", json=hi_query) as resp:
            hi_results = await resp.json()
            print(f"Query results: {hi_results}")
            
            if hi_results.get('success'):
                results = hi_results.get('results', [])
                print(f"Found {len(results)} results for generic 'Hi' query")
                
                # This should have low relevance scores
                for result in results:
                    score = result.get('score', 0)
                    print(f"   Result score: {score}")
    
    print("\n" + "=" * 60)
    print("🔍 Issue 35 reproduction test complete!")
    print("\nIf you see:")
    print("❌ vector_id is null - Qdrant collection creation failed")
    print("❌ Semantic search not finding correct contexts - Hash-based embeddings issue")
    print("✅ CORRECT results - Fixes are working!")

if __name__ == "__main__":
    print("Starting Issue 35 reproduction test...")
    print("Make sure the MCP server is running on localhost:8000")
    print()
    
    try:
        asyncio.run(test_issue_35_reproduction())
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        sys.exit(1)