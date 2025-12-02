# ◎ Veris Memory
> **memory with covenant**
> Truthful memory for agents. For those who remember.

Veris is memory that persists through change. For agents who carry weight. For those who remember what others forget.

**🎉 Production-Proven**: Successfully deployed in production with multiple concurrent users and persistent memory. Documentation improved based on real-world integration feedback.

## Agent-First Schema

Veris Memory implements the **Agent-First Schema Protocol** - a structured approach to memory management designed specifically for AI agents:

```json
{
  "name": "veris_memory",
  "label": "◎ Veris Memory",
  "subtitle": "memory with covenant",
  "type": "tool",
  "category": "memory_store",
  "tags": [
    "memory",
    "semantic",
    "agent-aligned",
    "truth-preserving",
    "long-term",
    "context"
  ]
}
```

## Core Capabilities

- **🎯 Semantic Retrieval**: Vector similarity search using Qdrant
- **🕸️ Graph Traversal**: Complex relationship queries via Neo4j
- **⚡ Fast Lookup**: Key-value storage with Redis
- **🤝 MCP Protocol**: Full Model Context Protocol v1.0 implementation
- **🛡️ Schema Validation**: Comprehensive YAML validation
- **🚀 Deploy Ready**: Docker + cloud deployment ready

## Architecture

### System Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Agents     │    │  Applications   │    │   Claude CLI    │
│                 │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │              MCP Protocol                   │
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Context Store MCP     │
                    │       Server           │
                    │   (FastAPI + MCP SDK)  │
                    └────────────┬────────────┘
                                 │
               ┌─────────────────┼─────────────────┐
               │                 │                 │
    ┌──────────▼──────────┐ ┌────▼──────┐ ┌──────▼────────┐
    │     Qdrant          │ │   Neo4j   │ │    Redis      │
    │  Vector Database    │ │   Graph   │ │  Key-Value    │
    │  (Embeddings)       │ │ Database  │ │    Store      │
    └─────────────────────┘ └───────────┘ └───────────────┘
```

## MCP Tools

**Core Memory Operations:**

1. `store_context` - Store structured context with covenant metadata
2. `retrieve_context` - Semantic similarity search with agent filters
3. `query_graph` - Traverse memory relationships and connections

**Agent State Management:**

4. `update_scratchpad` - Manage agent working memory
5. `get_agent_state` - Retrieve current agent state

**Memory Management (Sprint 13):**

6. `delete_context` - Hard delete contexts (human-only, requires admin role)
7. `forget_context` - Soft delete with retention period (human-only, requires admin role)

### Tool Coverage

| Tool                | Vector | Graph | KV Store | Auth Required | Purpose                               |
| ------------------- | ------ | ----- | -------- | ------------- | ------------------------------------- |
| `store_context`     | ✅     | ✅    | ➖       | Optional      | Store with embeddings + relationships |
| `retrieve_context`  | ✅     | ✅    | ➖       | Optional      | Hybrid semantic + graph search        |
| `query_graph`       | ➖     | ✅    | ➖       | Optional      | Advanced Cypher queries               |
| `update_scratchpad` | ➖     | ➖    | ✅       | Optional      | Transient agent memory                |
| `get_agent_state`   | ➖     | ➖    | ✅       | Optional      | Agent state retrieval                 |
| `delete_context`    | ✅     | ✅    | ✅       | **Human-only** | Hard delete context (audit logged)   |
| `forget_context`    | ➖     | ✅    | ✅       | **Human-only** | Soft delete with retention period    |

### Directory Structure

```
context-store/
├── src/
│   ├── storage/          # Database clients and operations
│   ├── validators/       # Schema validation and data integrity
│   ├── mcp_server/       # MCP protocol server implementation
│   └── core/            # Shared utilities and base classes
├── schemas/             # YAML schemas for context validation
├── contracts/           # MCP tool contracts and specifications
├── docs/               # Documentation (quickstart, tools, errors)
├── examples/           # Usage examples (Python, shell)
├── tests/              # Test suite
└── docker-compose.yml  # Docker deployment configuration
```

## Architecture

Built for agents, by agents:

- **Python 3.8+** with FastAPI core
- **Qdrant** for semantic embeddings
- **Neo4j** for memory graphs
- **Redis** for fast recall
- **Pydantic** for data integrity

## Quick Start

### ⚡ One-Line Docker Deployment

```bash
# Deploy and test in under 5 minutes
git clone https://github.com/credentum/veris-memory.git && cd veris-memory && docker-compose up -d && sleep 30 && curl http://localhost:8000/health
```

### Using Docker (Recommended)

```bash
# Clone and start all services
git clone https://github.com/credentum/veris-memory.git
cd veris-memory
docker-compose up -d

# Verify services
curl http://localhost:8000/health
```

### Manual Installation

```bash
# Python dependencies
pip install -r requirements.txt

# Start MCP server
python -m uvicorn src.mcp_server.main:app --host 0.0.0.0 --port 8000
```

## MCP Tools

The context store provides these MCP tools:

**Core Operations:**
- `store_context`: Store context data with vector embeddings and graph relationships
- `retrieve_context`: Hybrid retrieval using vector similarity and graph traversal
- `query_graph`: Direct Cypher queries for advanced graph operations
- `update_scratchpad`: Transient key-value storage with TTL
- `get_agent_state`: Retrieve persistent agent memory and state

**Memory Management (Sprint 13):**
- `delete_context`: Hard delete contexts (human-only, requires admin role)
- `forget_context`: Soft delete with retention period (human-only, requires admin role)

## Configuration

Configure via environment variables:

```bash
# Database connections
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# MCP server
MCP_SERVER_PORT=8000
MCP_LOG_LEVEL=info

# Storage settings
VECTOR_COLLECTION=project_context  # Fixed: Use correct collection name
GRAPH_DATABASE=context_graph

# API Key Authentication (Sprint 13)
# Format: API_KEY_{NAME}=key:user_id:role:is_agent
API_KEY_ADMIN=vmk_admin_secret:admin_user:admin:false
API_KEY_AGENT=vmk_agent_key:ai_assistant:writer:true
API_KEY_READER=vmk_readonly:reader_user:reader:false
```

## Authentication & Authorization

### API Key Setup

Veris Memory supports optional API key authentication with role-based access control (Sprint 13).

**Configure API keys** via environment variables:

```bash
# Format: API_KEY_{NAME}=key:user_id:role:is_agent
API_KEY_ADMIN=vmk_admin_secret:admin_user:admin:false
API_KEY_AGENT=vmk_agent_key:ai_assistant:writer:true
API_KEY_READER=vmk_readonly:reader_user:reader:false
```

### Authentication Methods

**Option 1: X-API-Key Header**
```bash
curl -X POST http://localhost:8000/tools/store_context \
  -H "X-API-Key: vmk_admin_secret" \
  -H "Content-Type: application/json" \
  -d '{"type": "log", "content": {"text": "Example"}}'
```

**Option 2: Authorization Bearer**
```bash
curl -X POST http://localhost:8000/tools/store_context \
  -H "Authorization: Bearer vmk_admin_secret" \
  -H "Content-Type: application/json" \
  -d '{"type": "log", "content": {"text": "Example"}}'
```

### Role-Based Access Control

| Role | Capabilities | Use Case |
|------|-------------|----------|
| `admin` | all (including delete) | System administrators |
| `writer` | read, write, query, cache | AI agents |
| `reader` | read, query | Read-only applications |
| `guest` | read | Public access |

### Human-Only Operations

**Critical Safety Feature**: The following operations require `is_agent=false` (human authentication):

- `delete_context` - Hard delete operation (prevents accidental data loss by agents)
- `forget_context` - Soft delete with retention period (requires human approval)

**Why this matters**: AI agents cannot accidentally or maliciously delete data. Only human-authenticated API keys can perform deletions.

**Example - Human API Key**:
```bash
API_KEY_ADMIN=vmk_admin_secret:admin_user:admin:false
                                              ^^^^^ is_agent=false (human)
```

**Example - Agent API Key**:
```bash
API_KEY_AGENT=vmk_agent_key:ai_assistant:writer:true
                                               ^^^^ is_agent=true (agent, cannot delete)
```

For complete API documentation, see [Sprint 13 API Documentation](docs/SPRINT_13_API_DOCUMENTATION.md).

## Development

### Prerequisites

- Python 3.8+
- Node.js 18+
- Docker and Docker Compose
- Qdrant v1.15.1+ (upgraded for improved performance)
- Neo4j Community Edition

### Setup

```bash
# Development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Start development services
docker-compose -f docker-compose.dev.yml up -d
```

### Testing

```bash
# Run all tests
pytest --cov=src --cov-report=html

# Run integration tests
pytest tests/integration/

# Run MCP tool tests
npm test
```

## 🚀 Production Deployment

*Based on real production deployments and lessons learned.*

### Quick Production Checklist

**Before deploying:**
- ✅ Set all required environment variables (see Configuration section)
- ✅ Configure persistent storage volumes for databases
- ✅ Set up monitoring for health endpoints
- ✅ Test with your integration code using staging environment
- ✅ Verify backup/restore procedures

### Local Development

```bash
docker-compose up -d
```

### Production Deployment

```bash
# Using production configuration with persistent storage
docker-compose -f docker-compose.prod.yml up -d

# Scale MCP servers for high availability
docker-compose up --scale mcp-server=3
```

### Production Considerations

**Persistent Storage:**
```yaml
# In docker-compose.prod.yml
volumes:
  qdrant_data:
    driver: local
  neo4j_data: 
    driver: local
  redis_data:
    driver: local
```

**Monitoring Integration:**
```bash
# Key metrics to monitor
curl http://localhost:8000/health          # Basic uptime
curl http://localhost:8000/status          # Detailed service status
curl -X POST http://localhost:8000/tools/verify_readiness  # Full diagnostics
```

**Connection Pooling:**
- Redis: Configure connection pool size based on concurrent users
- Neo4j: Use connection pooling for graph queries  
- Qdrant: Vector operations are optimized automatically

**Backup Strategies:**
- **Redis**: Use Redis persistence (RDB + AOF)
- **Neo4j**: Regular database backups via neo4j-admin
- **Qdrant**: Vector index snapshots

### Cloud Deployment

See `docs/deployment/` for cloud-specific deployment guides:

- AWS ECS/Fargate
- Google Cloud Run  
- Azure Container Instances
- Kubernetes

### Performance Tuning

**For high-throughput applications:**
```bash
# Environment variables for production
REDIS_MAX_CONNECTIONS=100
NEO4J_POOL_SIZE=50
QDRANT_TIMEOUT=30
MCP_SERVER_WORKERS=4
```

**Scaling Indicators:**
- **Scale up**: Response times > 100ms consistently
- **Scale out**: CPU usage > 70% or memory > 80%
- **Database scaling**: Query timeouts or connection pool exhaustion

## 📚 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get running in under 5 minutes
- **[MCP Tools Reference](docs/MCP_TOOLS.md)** - Complete API documentation for all 5 tools
- **[Error Codes Reference](docs/ERROR_CODES.md)** - Comprehensive error handling guide
- **[Migration Guide](docs/MIGRATION.md)** - Migrate from direct storage to MCP
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## 🚀 Examples

- **[Hello World (Python)](examples/hello_world.py)** - Complete example using all MCP tools
- **[Hello World (Shell)](examples/hello_world.sh)** - Same example using curl commands
- **[MCP Client Integration](docs/MCP_TOOLS.md#examples)** - Integration patterns for different languages

## API Documentation

### Health Check

```bash
GET /health
```

Returns service status and dependency health.

### System Status

```bash
GET /status
```

Returns comprehensive system information for agent orchestration:

```json
{
  "label": "◎ Veris Memory",
  "version": "1.0.0",
  "protocol": "MCP-1.0",
  "agent_ready": true,
  "tools": [
    "store_context",
    "retrieve_context", 
    "query_graph",
    "update_scratchpad",
    "get_agent_state"
  ],
  "dependencies": {
    "qdrant": "healthy",
    "neo4j": "healthy",
    "redis": "healthy"
  }
}
```

### Agent Readiness

```bash
POST /tools/verify_readiness
```

Provides diagnostic information for agents to verify system readiness:

```json
{
  "ready": true,
  "readiness_level": "FULL",
  "readiness_score": 100,
  "tools_available": 5,
  "service_status": {
    "core_services": {
      "redis": "healthy",
      "status": "healthy"
    },
    "enhanced_services": {
      "qdrant": "healthy",
      "neo4j": "healthy",
      "status": "healthy" 
    }
  },
  "recommended_actions": [
    "✓ Core functionality operational"
  ]
}
```

**Readiness Levels:**
- `BASIC`: Core tools available, minimal functionality
- `STANDARD`: Redis healthy, all 5 MCP tools operational  
- `FULL`: All services healthy, enhanced search available

## 📋 Response Format Examples

*Based on production integration feedback - these examples show actual response structures you'll receive.*

### Store Context Input

```json
{
  "content": {
    "text": "My name is Matt",
    "type": "decision",
    "title": "User Name",
    "fact_type": "personal_info"
  },
  "type": "log",
  "metadata": {"source": "telegram_bot", "timestamp": 1234567890}
}
```

### Retrieve Context Response

All search results (vector, graph, hybrid) use this **consistent format**:

```json
{
  "success": true,
  "results": [
    {
      "id": "uuid-string",
      "content": {
        "text": "My name is Matt",
        "type": "decision",
        "title": "User Name", 
        "fact_type": "personal_info"
      },
      "score": 0.95,
      "source": "vector"
    }
  ],
  "total_count": 1,
  "search_mode_used": "hybrid",
  "message": "Found 1 matching contexts"
}
```

**Key Points:**
- ✅ **Consistent structure**: All results use `{id, content, score, source}` format
- ✅ **No nested wrappers**: Clean, predictable structure
- ✅ **Single parsing path**: Same code handles vector and graph results

## 🔧 Integration Examples

*Production-tested integration patterns from real deployments.*

### Python Integration (Recommended)

```python
import aiohttp
import asyncio

class VerisMemoryClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    async def store_user_fact(self, text: str, fact_type: str = "personal_info"):
        """Store a user fact with proper format."""
        payload = {
            "content": {
                "text": text,
                "type": "decision",
                "title": text[:50] + "..." if len(text) > 50 else text,
                "fact_type": fact_type
            },
            "type": "log",
            "metadata": {"source": "your_app", "timestamp": time.time()}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/tools/store_context", json=payload) as resp:
                return await resp.json()
    
    async def search_memories(self, query: str, limit: int = 10):
        """Search stored memories with clean response handling."""
        payload = {
            "query": query,
            "search_mode": "hybrid",
            "limit": limit
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/tools/retrieve_context", json=payload) as resp:
                result = await resp.json()
                
                if result.get('success'):
                    # Clean, consistent parsing
                    memories = []
                    for item in result.get('results', []):
                        content = item['content']  # Always present
                        memories.append({
                            'text': content['text'],
                            'type': content['type'],
                            'score': item['score']
                        })
                    return memories
                else:
                    raise Exception(f"Search failed: {result.get('message', 'Unknown error')}")

# Usage example
async def main():
    client = VerisMemoryClient()
    
    # Store user information
    await client.store_user_fact("My name is Matt", "personal_info")
    await client.store_user_fact("I prefer green color", "preference")
    
    # Search for user info
    memories = await client.search_memories("What's my name?")
    for memory in memories:
        print(f"Found: {memory['text']} (confidence: {memory['score']:.2f})")

asyncio.run(main())
```

### TypeScript Integration

```typescript
interface MemoryContent {
  text: string;
  type: string;
  title: string;
  fact_type: string;
}

interface MemoryResult {
  id: string;
  content: MemoryContent;
  score: number;
  source: string;
}

interface MemoryResponse {
  success: boolean;
  results: MemoryResult[];
  total_count: number;
  search_mode_used: string;
  message: string;
}

class VerisMemoryClient {
  constructor(private baseUrl: string = "http://localhost:8000") {}

  async storeMemory(text: string, type: string = "decision"): Promise<any> {
    const payload = {
      content: {
        text,
        type,
        title: text.length > 50 ? text.substring(0, 47) + "..." : text,
        fact_type: "general"
      },
      type: "log",
      metadata: { source: "typescript_app", timestamp: Date.now() }
    };

    const response = await fetch(`${this.baseUrl}/tools/store_context`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    return await response.json();
  }

  async searchMemories(query: string): Promise<MemoryResult[]> {
    const payload = {
      query,
      search_mode: "hybrid",
      limit: 10
    };

    const response = await fetch(`${this.baseUrl}/tools/retrieve_context`, {
      method: 'POST', 
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const result: MemoryResponse = await response.json();
    
    if (result.success) {
      return result.results; // Clean, typed results
    } else {
      throw new Error(`Search failed: ${result.message}`);
    }
  }
}
```

## 🆕 Recent Improvements

**Sprint 13: Authentication & Memory Management:**
- ✅ API key authentication with role-based access control
- ✅ `delete_context` endpoint - Hard delete (human-only, audit logged)
- ✅ `forget_context` endpoint - Soft delete with retention period (human-only)
- ✅ Audit logging for all deletion operations (DeleteAuditLogger)
- ✅ Embedding pipeline visibility (status in responses)
- ✅ Namespace management and relationship auto-detection
- ✅ Human-only operation restrictions (prevents agent deletion)

**Bug Fixes (PR #329):**
- ✅ Fixed redis_client parameter bug in delete/forget endpoints
- ✅ Added comprehensive unit tests for redis_client parameter passing

**Search System Fixes (PR #159):**
- ✅ Fixed vector backend collection mismatch (now uses `project_context`)
- ✅ Enhanced text extraction with robust fallback strategies
- ✅ Added Redis compatibility layer for KV backend operations
- ✅ Improved error handling and validation across all backends

## 🔍 Troubleshooting Guide

*Common issues and solutions from production deployments.*

### Health Check Interpretation

| Endpoint | Purpose | When to Use |
|----------|---------|-------------|
| `GET /health` | Quick up/down status | Basic monitoring, load balancer health checks |
| `GET /status` | Detailed system info | Performance debugging, service overview |
| `POST /tools/verify_readiness` | Comprehensive diagnostics | Integration troubleshooting, readiness verification |

### Common Warnings and What They Mean

**"Vector collection mismatch"**
- ❌ **Status**: Vector searches returning empty results
- 🔧 **Action**: Ensure `VECTOR_COLLECTION=project_context` in environment
- 📊 **Impact**: Fixed in v0.9.0 - collection name now correctly configured

**"Enhanced features unavailable - Qdrant not ready"**
- ✅ **Status**: System operational with basic functionality
- 🔧 **Action**: Check Qdrant connection, vector search unavailable but core features work
- 📊 **Impact**: Graph and KV operations continue normally

**"Readiness level: STANDARD"**  
- ✅ **Status**: Normal operation, all core tools available
- 🔧 **Action**: None required, enhanced features may be degraded
- 📊 **Impact**: Perfect for most use cases

**"Redis connection timeout"**
- ❌ **Status**: Critical - core functionality affected
- 🔧 **Action**: Check Redis connectivity immediately
- 📊 **Impact**: Agent state and scratchpad operations fail

### API Endpoint Notes

**MCP Server vs REST API:**
- MCP Server endpoints: `/tools/*` (e.g., `/tools/retrieve_context` for search)
- REST API endpoints: `/api/v1/*` (not deployed by default, advanced features)
- Note: `search_context` only exists in REST API, use `retrieve_context` for MCP

### Migration from Previous Versions

If you're upgrading from earlier versions that used different field names:

| Old Field | New Field | Notes |
|-----------|-----------|-------|
| `user_message` | `text` | Unified text field |
| `assistant_response` | `text` | Unified text field |
| `exchange_type` | `type` | Must be one of: design, decision, trace, sprint, log |
| `context_type` | `type` | Field name changed in store_context |

### MCP Protocol

The server implements the MCP specification. Connect using any MCP-compatible client:

```typescript
import { MCPClient } from '@modelcontextprotocol/client';

const client = new MCPClient({
  serverUrl: 'http://localhost:8000/mcp'
});

await client.connect();
const result = await client.callTool('store_context', {
  content: {
    text: "Important information",
    type: "decision", 
    title: "Important information",
    fact_type: "general"
  },
  type: "log",
  metadata: {}
});
```

### 🛠️ MCP Tools Overview

| Tool                | Purpose                                         | Auth Required | Documentation                                          |
| ------------------- | ----------------------------------------------- | ------------- | ------------------------------------------------------ |
| `store_context`     | Store context with embeddings and relationships | Optional      | [API Reference](docs/MCP_TOOLS.md#1-store_context)     |
| `retrieve_context`  | Hybrid search across stored contexts            | Optional      | [API Reference](docs/MCP_TOOLS.md#2-retrieve_context)  |
| `query_graph`       | Execute read-only Cypher queries                | Optional      | [API Reference](docs/MCP_TOOLS.md#3-query_graph)       |
| `update_scratchpad` | Transient storage with TTL                      | Optional      | [API Reference](docs/MCP_TOOLS.md#4-update_scratchpad) |
| `get_agent_state`   | Retrieve agent state with isolation             | Optional      | [API Reference](docs/MCP_TOOLS.md#5-get_agent_state)   |
| `delete_context`    | Hard delete contexts (audit logged)             | **Human-only** | [API Reference](docs/SPRINT_13_API_DOCUMENTATION.md)   |
| `forget_context`    | Soft delete with retention period               | **Human-only** | [API Reference](docs/SPRINT_13_API_DOCUMENTATION.md)   |

## Performance

- **Response Time**: <50ms for typical MCP tool calls
- **Throughput**: 1000+ concurrent connections supported
- **Storage**: Scales with Qdrant and Neo4j limits
- **Memory**: ~100MB base memory usage

## Security

- **Authentication**: API key authentication with role-based access control (Sprint 13)
  - Support for X-API-Key and Authorization Bearer headers
  - Format: `API_KEY_{NAME}=key:user_id:role:is_agent`
  - See [Authentication & Authorization](#authentication--authorization) section
- **Authorization**: Role-based access control (admin, writer, reader, guest)
  - Human-only operations for delete/forget endpoints
  - Prevents AI agents from accidentally deleting data
- **Audit Logging**: All deletion operations logged to Redis with retention
- **Input Validation**: Comprehensive schema validation
- **Network**: TLS encryption for all external connections

For security documentation, see [docs/SECURITY.md](docs/SECURITY.md) and [docs/SPRINT_13_API_DOCUMENTATION.md](docs/SPRINT_13_API_DOCUMENTATION.md).

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make changes and add tests
4. Run the test suite: `pytest`
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/credentum/context-store/issues)
- **Discussions**: [GitHub Discussions](https://github.com/credentum/context-store/discussions)

## Related Projects

- [agent-context-template](https://github.com/credentum/agent-context-template) - Reference implementation using context-store
- [MCP Specification](https://github.com/modelcontextprotocol/specification) - Model Context Protocol documentationtrigger CI
