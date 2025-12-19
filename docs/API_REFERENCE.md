# Veris Memory Core - API Reference

Complete reference for all HTTP endpoints and MCP tools.

## Base URL

- **Docker**: `http://context-store:8000`
- **Host Machine**: `http://172.17.0.1:8000`

## Authentication

Include API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: vmk_your_key_here" http://172.17.0.1:8000/health
```

---

## Health & Status

### `GET /health`

Basic health check.

**Response:**
```json
{"status": "healthy", "uptime_seconds": 123, "message": "Server is running"}
```

### `GET /readiness`

Kubernetes readiness probe.

**Response:**
```json
{"ready": true, "checks": {"qdrant": true, "neo4j": true, "redis": true}}
```

### `GET /audit/stats`

Audit system statistics.

**Response:**
```json
{
  "total_entries": 1234,
  "wal_size_bytes": 56789,
  "last_entry_timestamp": "2025-12-19T10:00:00Z"
}
```

---

## MCP Tools

### `POST /tools/store_context`

Store a new context in memory.

**Request:**
```json
{
  "type": "decision",
  "content": {
    "title": "Architecture Decision",
    "description": "Details of the decision"
  },
  "author": "agent-name",
  "author_type": "agent",
  "metadata": {"tags": ["architecture"]},
  "authority": 7
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Context type: decision, design, log, trace, fact, precedent |
| `content` | object | Yes | The content to store |
| `author` | string | Yes | Author identifier |
| `author_type` | string | Yes | "agent" or "human" |
| `metadata` | object | No | Additional metadata |
| `authority` | int | No | 1-10, used by Covenant Mediator (default: 5) |

**Response:**
```json
{
  "context_id": "uuid-here",
  "stored": true,
  "gated": false
}
```

### `POST /tools/retrieve_context`

Retrieve contexts by semantic similarity.

**Request:**
```json
{
  "query": "architecture decisions about memory",
  "limit": 10,
  "type_filter": ["decision", "design"],
  "hours_ago": 24
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Search query |
| `limit` | int | No | Max results (default: 10) |
| `type_filter` | list | No | Filter by context type |
| `hours_ago` | int | No | Only contexts from last N hours |

**Response:**
```json
{
  "results": [
    {
      "id": "uuid-here",
      "content": {...},
      "score": 0.92,
      "type": "decision",
      "created_at": "2025-12-19T10:00:00Z"
    }
  ],
  "total": 1
}
```

### `POST /tools/query_graph`

Execute a Cypher query against the graph database.

**Request:**
```json
{
  "query": "MATCH (c:Context {type: 'decision'}) RETURN c LIMIT 5"
}
```

**Response:**
```json
{
  "results": [...],
  "query": "MATCH..."
}
```

### `POST /tools/update_scratchpad`

Update transient key-value storage.

**Request:**
```json
{
  "key": "current_task",
  "value": {"status": "in_progress", "step": 3},
  "ttl_seconds": 3600
}
```

### `POST /tools/get_agent_state`

Get current agent state from scratchpad.

**Request:**
```json
{
  "agent_id": "agent-123"
}
```

### `POST /tools/list_conflicts`

List pending covenant conflicts.

**Request:**
```json
{}
```

**Response:**
```json
{
  "conflicts": [
    {
      "conflict_id": "uuid",
      "existing_id": "existing-context-uuid",
      "new_content": {...},
      "similarity_score": 0.85,
      "created_at": "2025-12-19T10:00:00Z"
    }
  ]
}
```

### `POST /tools/resolve_conflict`

Resolve a pending conflict.

**Request:**
```json
{
  "conflict_id": "uuid-here",
  "resolution": "accept_new",
  "reason": "New information supersedes old"
}
```

**Resolution Options:**
- `accept_new` - Replace existing with new content
- `keep_existing` - Discard new content
- `merge` - Combine both (requires manual review)

---

## Search API

### `POST /search`

Multi-backend search with filtering and ranking.

**Request:**
```json
{
  "query": "search terms",
  "limit": 10,
  "backends": ["vector", "graph", "kv"],
  "filters": {
    "type": ["decision"],
    "hours_ago": 24
  },
  "ranking": "hybrid"
}
```

**Response:**
```json
{
  "results": [...],
  "search_time_ms": 45,
  "backends_used": ["vector", "graph"]
}
```

### `GET /search/modes`

Available search modes.

### `GET /search/policies`

Dispatch policies.

### `GET /search/ranking`

Ranking policies with details.

### `GET /search/backends`

Available backends.

### `GET /search/system-info`

Complete system information.

---

## Trajectory Logging

### `POST /trajectories/log`

Log an agent execution trajectory.

**Request:**
```json
{
  "task_id": "task-uuid",
  "outcome": "failure",
  "error_type": "TypeError",
  "error_message": "Cannot read property...",
  "stack_trace": "...",
  "context": {
    "file": "src/main.py",
    "function": "process_data"
  },
  "recovery_action": "Added null check",
  "succeeded_after_recovery": true
}
```

### `POST /trajectories/search`

Search trajectories by similarity.

**Request:**
```json
{
  "query": "TypeError in data processing",
  "limit": 5,
  "outcome_filter": "failure"
}
```

---

## Audit & Provenance

### `POST /audit/trace/{artifact_id}`

Trace lineage back to origin.

**Response:**
```json
{
  "lineage": [
    {"id": "uuid-1", "type": "source", "created_at": "..."},
    {"id": "uuid-2", "type": "derived", "derived_from": "uuid-1"}
  ]
}
```

### `POST /audit/descendants/{artifact_id}`

Find items derived from artifact.

### `POST /audit/retention/process`

Process retention policies (SCAR).

---

## Learning System

### `POST /learning/precedents/store`

Store a new precedent.

**Request:**
```json
{
  "precedent_type": "FAILURE",
  "error_pattern": "NullPointerException in UserService",
  "resolution": "Added null check before accessing user.email",
  "confidence": 0.9,
  "source_trajectory_id": "trajectory-uuid"
}
```

**Precedent Types:**
- `FAILURE` - Learning from errors
- `SUCCESS` - Successful patterns
- `DECISION` - Architectural decisions
- `SKILL` - Learned capabilities
- `PATTERN` - Recurring patterns

### `POST /learning/precedents/query`

Query precedents by similarity.

**Request:**
```json
{
  "query": "null pointer exception handling",
  "limit": 5,
  "type_filter": ["FAILURE"]
}
```

### `GET /learning/stats`

Learning system statistics.

**Response:**
```json
{
  "total_precedents": 45,
  "by_type": {
    "FAILURE": 30,
    "SUCCESS": 10,
    "DECISION": 5
  },
  "last_extracted": "2025-12-19T10:00:00Z"
}
```

---

## Metrics

### `GET /metrics`

Prometheus metrics endpoint.

---

## Error Responses

All endpoints return standard error format:

```json
{
  "error": "Error type",
  "message": "Detailed error message",
  "status_code": 400
}
```

**Common Status Codes:**

| Code | Meaning |
|------|---------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid or missing API key |
| 403 | Forbidden - Insufficient permissions (e.g., human-only operation) |
| 404 | Not Found - Resource doesn't exist |
| 422 | Validation Error - Schema validation failed |
| 500 | Internal Server Error |

---

## Rate Limiting

Default rate limits:
- 100 requests/minute per API key
- 1000 requests/hour per API key

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1702987200
```

---

## Related Documentation

- [ARCHITECTURE_DECISIONS.md](./ARCHITECTURE_DECISIONS.md) - System architecture decisions
- [.env.example](../.env.example) - Environment configuration
- [VMC-ADR-010: Retrieval Enhancements](./ARCHITECTURE_DECISIONS.md#2025-12-19-vmc-adr-010---retrieval-enhancements-implemented)
- [VMC-ADR-011: Security Infrastructure](./ARCHITECTURE_DECISIONS.md#2025-12-19-vmc-adr-011---security-infrastructure-implemented)
