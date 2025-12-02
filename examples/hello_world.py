#!/usr/bin/env python3
"""
Context Store MCP - Hello World Example

This example demonstrates basic usage of the Context Store MCP server:
1. Store a context entry
2. Retrieve context by search
3. Update agent scratchpad
4. Query the graph database

Prerequisites:
- Context Store running locally (docker-compose up -d)
- Python 3.8+ with requests installed
"""

import json
import time
from typing import Any, Dict

import requests


class ContextStoreMCP:
    """Simple MCP client for Context Store."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url
        self.session = requests.Session()

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool."""
        payload = {"method": "call_tool", "params": {"name": name, "arguments": arguments}}

        response = self.session.post(
            f"{self.base_url}/mcp/call_tool",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API call failed: {response.status_code} - {response.text}")


def main() -> None:
    """Run the Hello World example."""
    print("🚀 Context Store MCP - Hello World Example")
    print("=" * 50)

    # Initialize client
    client = ContextStoreMCP()

    # Check health
    try:
        health = requests.get("http://localhost:8000/health", timeout=5)
        if health.status_code == 200:
            print("✅ Context Store is healthy")
        else:
            print("❌ Context Store health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to Context Store: {e}")
        print("💡 Make sure to run: docker-compose up -d")
        return

    print()

    # Step 1: Store Context
    print("📝 Step 1: Store Context")
    print("-" * 30)

    store_result = client.call_tool(
        "store_context",
        {
            "type": "design",
            "content": {
                "title": "Hello World API",
                "description": "A simple REST API that returns greeting messages",
                "endpoints": [
                    {"path": "/hello", "method": "GET", "description": "Returns hello message"},
                    {
                        "path": "/hello/{name}",
                        "method": "GET",
                        "description": "Returns personalized greeting",
                    },
                ],
            },
            "metadata": {
                "author": "hello-world-example",
                "version": "1.0.0",
                "tags": ["api", "greeting", "example"],
            },
        },
    )

    print(f"Context stored with ID: {store_result.get('id', 'unknown')}")
    print(f"Vector storage: {store_result.get('vector_id', 'failed')}")
    print(f"Graph storage: {store_result.get('graph_id', 'failed')}")
    print()

    # Step 2: Retrieve Context
    print("🔍 Step 2: Retrieve Context")
    print("-" * 30)

    retrieve_result = client.call_tool(
        "retrieve_context",
        {"query": "Hello World API greeting", "search_mode": "hybrid", "limit": 5},
    )

    results = retrieve_result.get("results", [])
    print(f"Found {len(results)} matching contexts:")
    for i, result in enumerate(results, 1):
        content = result.get("payload", {}).get("content", result.get("content", {}))
        title = content.get("title", "Untitled")
        print(f"  {i}. {title}")
    print()

    # Step 3: Update Scratchpad
    print("📋 Step 3: Update Agent Scratchpad")
    print("-" * 35)

    scratchpad_result = client.call_tool(
        "update_scratchpad",
        {
            "agent_id": "hello-world-agent",
            "field": "progress",
            "content": (
                f"Successfully completed Hello World example at "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            ),
            "ttl": 3600,  # 1 hour
        },
    )

    if scratchpad_result.get("success"):
        print(f"✅ Scratchpad updated: {scratchpad_result.get('message')}")
    else:
        print(f"❌ Scratchpad update failed: {scratchpad_result.get('message')}")
    print()

    # Step 4: Query Graph
    print("🕸️  Step 4: Query Graph Database")
    print("-" * 35)

    graph_result = client.call_tool(
        "query_graph",
        {
            "query": "MATCH (n:Context) WHERE n.type = 'design' RETURN n.id, n.content LIMIT 3",
            "limit": 3,
        },
    )

    if graph_result.get("success"):
        graph_results = graph_result.get("results", [])
        print(f"Found {len(graph_results)} design contexts in graph:")
        for result in graph_results:
            node_id = result.get("n.id", "unknown")
            content = json.loads(result.get("n.content", "{}"))
            title = content.get("title", "Untitled")
            print(f"  - {node_id}: {title}")
    else:
        print(f"❌ Graph query failed: {graph_result.get('error', 'unknown error')}")
    print()

    # Step 5: Get Agent State
    print("🤖 Step 5: Get Agent State")
    print("-" * 30)

    state_result = client.call_tool(
        "get_agent_state", {"agent_id": "hello-world-agent", "prefix": "scratchpad"}
    )

    if state_result.get("success"):
        data = state_result.get("data", {})
        field_names = state_result.get("keys", [])
        print(f"Agent has {len(field_names)} scratchpad entries:")
        for field_name in field_names:
            content = data.get(field_name, "")
            print(f"  - {field_name}: {content[:50]}{'...' if len(content) > 50 else ''}")
    else:
        print(f"❌ Failed to get agent state: {state_result.get('message')}")

    print()
    print("🎉 Hello World example completed successfully!")
    print("💡 Next steps:")
    print("   - Explore the MCP Tools documentation")
    print("   - Try building your own context-aware application")
    print("   - Check out more examples in the examples/ directory")


if __name__ == "__main__":
    main()
