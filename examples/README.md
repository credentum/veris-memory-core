# Context Store Examples

This directory contains practical examples demonstrating how to use the Context Store MCP server.

## Quick Start Examples

### [hello_world.py](hello_world.py)

Complete Python example demonstrating all 5 MCP tools:

```bash
# Prerequisites: Context Store running locally
docker-compose up -d

# Install dependencies
pip install requests

# Run example
python examples/hello_world.py
```

**Features demonstrated**:

- ✅ Health check verification
- ✅ Store context with metadata and relationships
- ✅ Retrieve context using hybrid search
- ✅ Update agent scratchpad with TTL
- ✅ Query graph database with Cypher
- ✅ Get agent state with namespace isolation

### [hello_world.sh](hello_world.sh)

Shell script version using curl commands:

```bash
# Prerequisites: Context Store running + jq installed
docker-compose up -d
sudo apt-get install jq  # or brew install jq

# Run example
chmod +x examples/hello_world.sh
./examples/hello_world.sh
```

**Features demonstrated**:

- ✅ All MCP tools via HTTP API
- ✅ JSON parsing and error handling
- ✅ Portable across different environments
- ✅ Easy integration into CI/CD pipelines

## Usage Patterns

### Basic Context Storage

```python
# Store design context
result = await client.call_tool("store_context", {
    "type": "design",
    "content": {
        "title": "Authentication System",
        "description": "JWT-based authentication with refresh tokens",
        "components": ["auth-service", "token-validator", "user-store"]
    },
    "metadata": {
        "author": "dev-team",
        "priority": "high",
        "epic": "security-v2"
    },
    "relationships": [
        {"type": "implements", "target": "req-auth-001"},
        {"type": "depends_on", "target": "ctx_user_management"}
    ]
})
```

### Context Retrieval

```python
# Semantic search
results = await client.call_tool("retrieve_context", {
    "query": "authentication JWT security tokens",
    "search_mode": "hybrid",  # vector + graph search
    "limit": 10,
    "type": "design"  # filter by context type
})

for context in results["results"]:
    print(f"Found: {context['payload']['content']['title']}")
    print(f"Score: {context['score']}")
```

### Graph Analysis

```python
# Find related contexts
graph_result = await client.call_tool("query_graph", {
    "query": """
        MATCH (auth:Context {type: 'design'})-[:IMPLEMENTS]->(req:Context)
        WHERE auth.content CONTAINS 'authentication'
        RETURN auth.id, auth.content, req.id
        LIMIT 5
    """,
    "parameters": {}
})

for row in graph_result["results"]:
    print(f"Design {row['auth.id']} implements {row['req.id']}")
```

### Agent Memory Management

```python
# Store working memory
await client.call_tool("update_scratchpad", {
    "agent_id": "auth-dev-agent",
    "key": "current_task",
    "content": "Implementing JWT refresh token rotation",
    "ttl": 3600  # 1 hour
})

# Retrieve agent state
state = await client.call_tool("get_agent_state", {
    "agent_id": "auth-dev-agent",
    "prefix": "scratchpad"
})

print(f"Agent has {len(state['keys'])} scratchpad entries")
```

## Integration Examples

### Web Application Integration

```javascript
// client-side integration
import { MCPClient } from "@modelcontextprotocol/client";

class ContextService {
  constructor() {
    this.client = new MCPClient({
      serverUrl: "http://localhost:8000/mcp",
    });
  }

  async storeUserAction(action) {
    return await this.client.callTool("store_context", {
      type: "trace",
      content: {
        action: action.type,
        user: action.userId,
        timestamp: action.timestamp,
        metadata: action.data,
      },
      metadata: {
        source: "web-app",
        session: action.sessionId,
      },
    });
  }

  async searchContext(query) {
    const result = await this.client.callTool("retrieve_context", {
      query: query,
      search_mode: "hybrid",
      limit: 20,
    });

    return result.results.map((r) => r.payload.content);
  }
}
```

### CLI Tool Integration

```python
#!/usr/bin/env python3
"""CLI tool for context management."""

import click
import asyncio
from mcp.client import MCPClient

@click.group()
def cli():
    """Context Store CLI tool."""
    pass

@cli.command()
@click.argument('query')
@click.option('--limit', default=10, help='Maximum results')
@click.option('--type', help='Context type filter')
async def search(query, limit, type):
    """Search for contexts."""
    async with MCPClient("http://localhost:8000/mcp") as client:
        result = await client.call_tool("retrieve_context", {
            "query": query,
            "limit": limit,
            "type": type or "all"
        })

        for context in result["results"]:
            content = context["payload"]["content"]
            click.echo(f"• {content.get('title', 'Untitled')} (score: {context['score']:.2f})")

@cli.command()
@click.option('--title', required=True, help='Context title')
@click.option('--description', help='Context description')
@click.option('--type', default='design', help='Context type')
async def store(title, description, type):
    """Store a new context."""
    async with MCPClient("http://localhost:8000/mcp") as client:
        result = await client.call_tool("store_context", {
            "content": {
                "title": title,
                "description": description or ""
            },
            "type": type,
            "metadata": {
                "source": "cli-tool"
            }
        })

        if result["success"]:
            click.echo(f"✅ Stored context: {result['id']}")
        else:
            click.echo(f"❌ Failed: {result['message']}")

if __name__ == '__main__':
    asyncio.run(cli())
```

### CI/CD Pipeline Integration

```yaml
# .github/workflows/context-tracking.yml
name: Track Development Context

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  track-context:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Store commit context
        run: |
          curl -X POST http://context-store:8000/mcp/call_tool \
            -H "Content-Type: application/json" \
            -d '{
              "method": "call_tool",
              "params": {
                "name": "store_context",
                "arguments": {
                  "type": "trace",
                  "content": {
                    "commit": "'$GITHUB_SHA'",
                    "message": "'$(git log -1 --pretty=%B)'",
                    "author": "'$(git log -1 --pretty=%an)'",
                    "files": "'$(git diff --name-only HEAD~1)'",
                    "branch": "'$GITHUB_REF_NAME'"
                  },
                  "metadata": {
                    "source": "github-actions",
                    "repository": "'$GITHUB_REPOSITORY'",
                    "workflow": "'$GITHUB_WORKFLOW'"
                  }
                }
              }
            }'
```

## Error Handling Examples

### Robust Client Implementation

```python
import asyncio
import logging
from mcp.client import MCPClient
from typing import Dict, Any, Optional

class RobustContextClient:
    def __init__(self, server_url: str, max_retries: int = 3):
        self.server_url = server_url
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

    async def call_tool_with_retry(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Call MCP tool with exponential backoff retry logic."""

        for attempt in range(self.max_retries):
            try:
                async with MCPClient(self.server_url) as client:
                    result = await client.call_tool(tool_name, arguments)

                    if result.get("success"):
                        return result

                    error_type = result.get("error_type")

                    # Handle specific error types
                    if error_type == "rate_limit":
                        wait_time = (2 ** attempt) * 1.0
                        self.logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    elif error_type in ["storage_unavailable", "connection_timeout"]:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(1)
                            continue

                    elif error_type in ["validation_error", "forbidden_operation"]:
                        # Don't retry validation errors
                        self.logger.error(f"Validation error: {result.get('message')}")
                        return result

                    # For other errors, retry once
                    if attempt == 0:
                        await asyncio.sleep(0.5)
                        continue

                    return result

            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

        return {"success": False, "message": "Max retries exceeded"}

# Usage
client = RobustContextClient("http://localhost:8000/mcp")
result = await client.call_tool_with_retry("store_context", {
    "content": {"title": "Test"},
    "type": "design"
})
```

## Testing Examples

### Unit Tests

```python
import pytest
from mcp.client import MCPClient

@pytest.fixture
async def mcp_client():
    """MCP client fixture for testing."""
    client = MCPClient("http://localhost:8000/mcp")
    await client.connect()
    yield client
    await client.close()

@pytest.mark.asyncio
async def test_store_and_retrieve_context(mcp_client):
    """Test basic store and retrieve functionality."""

    # Store a test context
    store_result = await mcp_client.call_tool("store_context", {
        "content": {
            "title": "Test Context",
            "description": "Integration test context"
        },
        "type": "design",
        "metadata": {
            "test": True,
            "timestamp": "2024-01-15T10:00:00Z"
        }
    })

    assert store_result["success"] is True
    context_id = store_result["id"]

    # Retrieve the context
    retrieve_result = await mcp_client.call_tool("retrieve_context", {
        "query": "Test Context integration",
        "limit": 5
    })

    assert retrieve_result["success"] is True
    assert len(retrieve_result["results"]) > 0

    # Verify the stored context is found
    found_context = None
    for result in retrieve_result["results"]:
        if result["id"] == context_id:
            found_context = result
            break

    assert found_context is not None
    assert found_context["payload"]["content"]["title"] == "Test Context"

@pytest.mark.asyncio
async def test_scratchpad_functionality(mcp_client):
    """Test agent scratchpad operations."""

    agent_id = "test-agent-123"
    test_content = "Test scratchpad content"

    # Update scratchpad
    update_result = await mcp_client.call_tool("update_scratchpad", {
        "agent_id": agent_id,
        "key": "test_key",
        "content": test_content,
        "ttl": 3600
    })

    assert update_result["success"] is True

    # Retrieve agent state
    state_result = await mcp_client.call_tool("get_agent_state", {
        "agent_id": agent_id,
        "key": "test_key",
        "prefix": "scratchpad"
    })

    assert state_result["success"] is True
    assert state_result["data"]["content"] == test_content
```

## Performance Examples

### Batch Operations

```python
async def batch_store_contexts(client: MCPClient, contexts: List[Dict]):
    """Store multiple contexts efficiently."""

    # Use semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(10)

    async def store_single_context(context_data):
        async with semaphore:
            return await client.call_tool("store_context", context_data)

    # Execute all operations concurrently
    tasks = [store_single_context(ctx) for ctx in contexts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    successful = 0
    failed = 0

    for result in results:
        if isinstance(result, Exception):
            failed += 1
        elif result.get("success"):
            successful += 1
        else:
            failed += 1

    return {"successful": successful, "failed": failed, "total": len(contexts)}
```

## Production Examples

See the [Troubleshooting Guide](../docs/TROUBLESHOOTING.md) for production deployment patterns and monitoring examples.

## Contributing Examples

Have a useful example? Please contribute:

1. Create your example file in this directory
2. Add appropriate documentation and comments
3. Test with the current Context Store version
4. Submit a pull request

Example contributions we'd love to see:

- Language-specific integrations (Go, Rust, Java)
- Framework integrations (FastAPI, Express, Django)
- Monitoring and observability examples
- Advanced query patterns
- Performance optimization examples
