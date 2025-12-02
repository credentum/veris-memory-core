#!/bin/bash
# Context Store MCP - Hello World Example (Shell Script)
#
# This script demonstrates basic MCP tool usage via curl commands.
# Prerequisites: Context Store running locally (docker-compose up -d)

set -e

BASE_URL="http://localhost:8000"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "🚀 Context Store MCP - Hello World Example"
echo "================================================="

# Check health
echo "🏥 Checking Context Store health..."
if curl -s -f "$BASE_URL/health" > /dev/null; then
    echo "✅ Context Store is healthy"
else
    echo "❌ Context Store is not accessible"
    echo "💡 Make sure to run: docker-compose up -d"
    exit 1
fi

echo

# Step 1: Store Context
echo "📝 Step 1: Store Context"
echo "-------------------------"

STORE_RESULT=$(curl -s -X POST "$BASE_URL/mcp/call_tool" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "call_tool",
    "params": {
      "name": "store_context",
      "arguments": {
        "type": "design",
        "content": {
          "title": "Hello World API",
          "description": "A simple REST API that returns greeting messages",
          "endpoints": [
            {"path": "/hello", "method": "GET", "description": "Returns hello message"},
            {"path": "/hello/{name}", "method": "GET", "description": "Returns personalized greeting"}
          ]
        },
        "metadata": {
          "author": "hello-world-shell-example",
          "version": "1.0.0",
          "tags": ["api", "greeting", "example", "shell"]
        }
      }
    }
  }')

CONTEXT_ID=$(echo "$STORE_RESULT" | jq -r '.id // "unknown"')
echo "Context stored with ID: $CONTEXT_ID"
echo

# Step 2: Retrieve Context
echo "🔍 Step 2: Retrieve Context"
echo "----------------------------"

RETRIEVE_RESULT=$(curl -s -X POST "$BASE_URL/mcp/call_tool" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "call_tool",
    "params": {
      "name": "retrieve_context",
      "arguments": {
        "query": "Hello World API greeting",
        "search_mode": "hybrid",
        "limit": 3
      }
    }
  }')

RESULTS_COUNT=$(echo "$RETRIEVE_RESULT" | jq -r '.results | length')
echo "Found $RESULTS_COUNT matching contexts:"

echo "$RETRIEVE_RESULT" | jq -r '.results[] | "  - " + (.payload.content.title // .content.title // "Untitled")'
echo

# Step 3: Update Scratchpad
echo "📋 Step 3: Update Agent Scratchpad"
echo "-----------------------------------"

SCRATCHPAD_RESULT=$(curl -s -X POST "$BASE_URL/mcp/call_tool" \
  -H "Content-Type: application/json" \
  -d "{
    \"method\": \"call_tool\",
    \"params\": {
      \"name\": \"update_scratchpad\",
      \"arguments\": {
        \"agent_id\": \"hello-world-shell-agent\",
        \"key\": \"progress\",
        \"content\": \"Successfully completed Hello World shell example at $TIMESTAMP\",
        \"ttl\": 3600
      }
    }
  }")

SCRATCHPAD_SUCCESS=$(echo "$SCRATCHPAD_RESULT" | jq -r '.success')
SCRATCHPAD_MESSAGE=$(echo "$SCRATCHPAD_RESULT" | jq -r '.message')

if [ "$SCRATCHPAD_SUCCESS" = "true" ]; then
    echo "✅ Scratchpad updated: $SCRATCHPAD_MESSAGE"
else
    echo "❌ Scratchpad update failed: $SCRATCHPAD_MESSAGE"
fi
echo

# Step 4: Query Graph
echo "🕸️  Step 4: Query Graph Database"
echo "--------------------------------"

GRAPH_RESULT=$(curl -s -X POST "$BASE_URL/mcp/call_tool" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "call_tool",
    "params": {
      "name": "query_graph",
      "arguments": {
        "query": "MATCH (n:Context) WHERE n.type = '\''design'\'' RETURN n.id, n.content LIMIT 3",
        "limit": 3
      }
    }
  }')

GRAPH_SUCCESS=$(echo "$GRAPH_RESULT" | jq -r '.success')

if [ "$GRAPH_SUCCESS" = "true" ]; then
    GRAPH_COUNT=$(echo "$GRAPH_RESULT" | jq -r '.results | length')
    echo "Found $GRAPH_COUNT design contexts in graph:"
    echo "$GRAPH_RESULT" | jq -r '.results[] | "  - " + (.["n.id"] // "unknown") + ": " + ((.["n.content"] | fromjson).title // "Untitled")'
else
    GRAPH_ERROR=$(echo "$GRAPH_RESULT" | jq -r '.error // "unknown error"')
    echo "❌ Graph query failed: $GRAPH_ERROR"
fi
echo

# Step 5: Get Agent State
echo "🤖 Step 5: Get Agent State"
echo "---------------------------"

STATE_RESULT=$(curl -s -X POST "$BASE_URL/mcp/call_tool" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "call_tool",
    "params": {
      "name": "get_agent_state",
      "arguments": {
        "agent_id": "hello-world-shell-agent",
        "prefix": "scratchpad"
      }
    }
  }')

STATE_SUCCESS=$(echo "$STATE_RESULT" | jq -r '.success')

if [ "$STATE_SUCCESS" = "true" ]; then
    KEYS_COUNT=$(echo "$STATE_RESULT" | jq -r '.keys | length')
    echo "Agent has $KEYS_COUNT scratchpad entries:"
    echo "$STATE_RESULT" | jq -r '.keys[] | "  - " + . + ": " + (.data[.] // "empty")'
else
    STATE_MESSAGE=$(echo "$STATE_RESULT" | jq -r '.message')
    echo "❌ Failed to get agent state: $STATE_MESSAGE"
fi

echo
echo "🎉 Hello World shell example completed successfully!"
echo "💡 Next steps:"
echo "   - Explore the MCP Tools documentation"
echo "   - Try the Python example: python examples/hello_world.py"
echo "   - Check out the web interface at http://localhost:7474 (Neo4j)"
echo "   - Explore vector data at http://localhost:6333/dashboard (Qdrant)"
