---
name: observability
version: "1.1"
trigger_phrases:
  - "search trajectories"
  - "find errors"
  - "what failed"
  - "show recent errors"
  - "search error logs"
  - "find similar failures"
  - "trajectory search"
  - "debug errors"
  - "trace errors"
  - "playback trajectory"
---

# Observability Skill

Query agent execution trajectories and structured errors from the Veris platform's observability endpoints.

## Activation Signals

Use this skill when the user:
- Asks to search or find trajectories
- Asks about errors or failures in the system
- Wants to debug or trace issues
- Asks "what failed?" or "show errors"
- Mentions searching logs or observability data

## Time-Based Filtering

The `hours_ago` parameter filters results to records created within the last N hours.

**How it works:**
- Records store `timestamp_unix` (Unix timestamp) for efficient range queries
- Filter uses Qdrant Range query: `timestamp_unix >= (now - hours_ago)`
- Only records logged after this feature was deployed have `timestamp_unix`
- Older records without `timestamp_unix` are excluded from time-filtered searches

**Example:** Find failures in the last 2 hours:
```bash
curl -X POST http://172.17.0.1:8000/api/v1/trajectories/search \
  -H "Content-Type: application/json" \
  -d '{"outcome": "failure", "hours_ago": 2}'
```

## Endpoints

Base URL: `http://172.17.0.1:8000` (or `http://context-store:8000` from Docker network)

### POST /api/v1/trajectories/search

Search agent execution trajectories.

**Request:**
```json
{
  "query": "optional semantic search text",
  "agent": "optional agent name filter",
  "outcome": "success|failure|partial",
  "task_id": "optional task ID filter",
  "hours_ago": 24,
  "limit": 20
}
```

**Response:**
```json
{
  "success": true,
  "trajectories": [
    {
      "trajectory_id": "traj_abc123",
      "task_id": "task_xyz",
      "agent": "architect",
      "outcome": "failure",
      "error": "Connection timeout",
      "duration_ms": 5432.1,
      "cost_usd": 0.05,
      "trace_id": "mcp_123456",
      "timestamp": "2025-12-15T10:30:00",
      "metadata": {},
      "score": 0.95
    }
  ],
  "count": 1,
  "trace_id": "traj_search_123"
}
```

### POST /api/v1/errors/search

Search structured error logs.

**Request:**
```json
{
  "query": "optional semantic search text",
  "service": "optional service name filter",
  "error_type": "optional error type filter",
  "trace_id": "optional trace ID to find",
  "hours_ago": 24,
  "limit": 20
}
```

**Response:**
```json
{
  "success": true,
  "errors": [
    {
      "error_id": "err_def456",
      "trace_id": "mcp_123456",
      "task_id": "task_xyz",
      "service": "orchestrator",
      "error_type": "ValidationError",
      "error_message": "Missing required field: name",
      "context": {"request_id": "req_789"},
      "timestamp": "2025-12-15T10:30:00",
      "score": 0.92
    }
  ],
  "count": 1,
  "trace_id": "err_search_123"
}
```

## Few-Shot Examples

### Example 1: Find Recent Failures

**User:** "What failed in the last hour?"

**Action:**
```bash
curl -X POST http://172.17.0.1:8000/api/v1/trajectories/search \
  -H "Content-Type: application/json" \
  -d '{"outcome": "failure", "hours_ago": 1, "limit": 10}'
```

### Example 2: Search Errors by Service

**User:** "Show errors from the orchestrator"

**Action:**
```bash
curl -X POST http://172.17.0.1:8000/api/v1/errors/search \
  -H "Content-Type: application/json" \
  -d '{"service": "orchestrator", "limit": 20}'
```

### Example 3: Semantic Search for Similar Errors

**User:** "Find errors like 'connection refused'"

**Action:**
```bash
curl -X POST http://172.17.0.1:8000/api/v1/errors/search \
  -H "Content-Type: application/json" \
  -d '{"query": "connection refused", "limit": 10}'
```

### Example 4: Trace a Specific Error

**User:** "Find all errors with trace ID mcp_1702500000_abc123"

**Action:**
```bash
curl -X POST http://172.17.0.1:8000/api/v1/errors/search \
  -H "Content-Type: application/json" \
  -d '{"trace_id": "mcp_1702500000_abc123"}'
```

### Example 5: Find Failed Trajectories by Agent

**User:** "Show failed architect agent runs"

**Action:**
```bash
curl -X POST http://172.17.0.1:8000/api/v1/trajectories/search \
  -H "Content-Type: application/json" \
  -d '{"agent": "architect", "outcome": "failure", "limit": 20}'
```

### Example 6: Semantic Trajectory Search

**User:** "Find trajectories similar to 'timeout during API call'"

**Action:**
```bash
curl -X POST http://172.17.0.1:8000/api/v1/trajectories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "timeout during API call", "limit": 10}'
```

## Python Usage

```python
import httpx

async def search_trajectories(query=None, agent=None, outcome=None, hours_ago=None, limit=20):
    """Search trajectories with optional filters."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://172.17.0.1:8000/api/v1/trajectories/search",
            json={
                "query": query,
                "agent": agent,
                "outcome": outcome,
                "hours_ago": hours_ago,
                "limit": limit
            }
        )
        return response.json()

async def search_errors(query=None, service=None, error_type=None, trace_id=None, hours_ago=None, limit=20):
    """Search errors with optional filters."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://172.17.0.1:8000/api/v1/errors/search",
            json={
                "query": query,
                "service": service,
                "error_type": error_type,
                "trace_id": trace_id,
                "hours_ago": hours_ago,
                "limit": limit
            }
        )
        return response.json()

# Usage examples
failures = await search_trajectories(outcome="failure", hours_ago=24)
errors = await search_errors(service="orchestrator", limit=10)
similar = await search_errors(query="connection timeout")
```

## Related Endpoints

These endpoints are also part of the observability infrastructure:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/trajectories/log` | POST | Log a new trajectory |
| `/api/v1/errors/log` | POST | Log a structured error |
| `/api/v1/telemetry/snapshot` | GET | Get system telemetry |
| `/api/v1/packets/{id}/replay` | POST | Replay a failed packet |

## PROHIBITED

- Do NOT use these endpoints to write or modify data (they are search-only)
- Do NOT expose sensitive error context in user-facing responses
- Do NOT make excessive queries (use filters to narrow results)
- Do NOT assume all errors are critical (check error_type and context)
