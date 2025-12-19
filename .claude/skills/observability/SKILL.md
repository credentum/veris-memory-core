---
name: observability
version: "2.0"
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
  - "stuck packets"
  - "active work"
  - "packet events"
  - "what's running"
  - "packet timeline"
---

# Observability Skill

Query agent execution trajectories, structured errors, and real-time pipeline state from the Veris platform's observability infrastructure.

## Activation Signals

Use this skill when the user:
- Asks to search or find trajectories
- Asks about errors or failures in the system
- Wants to debug or trace issues
- Asks "what failed?" or "show errors"
- Mentions searching logs or observability data
- Asks about stuck or abandoned work packets
- Wants to see packet lifecycle events
- Asks "what's running now?" or "which agent has this packet?"

## Observability Architecture

The Veris platform uses **two complementary observability systems**:

### 1. **Redis-Based Real-Time Observability** (NEW - PR #80)
- **Active Work Tracking**: Know which agent claimed which packet and when
- **Event Stream**: Queryable timeline of packet lifecycle (work_started, coder_completed, error, etc.)
- **Stuck Packet Detection**: Find packets claimed but never completed (agent crashes)
- **Storage**: Redis with TTLs (10 min for active work, 24h for events)
- **Use Case**: Real-time monitoring, debugging in-flight work, detecting silent failures

### 2. **Neo4j/Qdrant Trajectory & Error Search**
- **Trajectory Search**: Historical agent execution records with semantic search
- **Error Search**: Structured error logs with filters and correlation
- **Storage**: Neo4j (graph) + Qdrant (vector embeddings)
- **Use Case**: Historical analysis, pattern detection, learning from past failures

---

## Redis-Based Observability (Real-Time)

Base URL: `http://172.17.0.1:8000` (or `http://context-store:8000` from Docker network)

### GET /tools/stuck_packets

Find packets that were claimed by agents but never completed (likely crashed/stuck).

**Query Parameters:**
- `user_id` (required): Team/user ID (e.g., "dev_team")
- `threshold` (optional): Minimum age in seconds (default: 300, min: 60)

**Request:**
```bash
curl -s "http://172.17.0.1:8000/tools/stuck_packets?user_id=dev_team&threshold=300"
```

**Response:**
```json
{
  "stuck_packets": [
    {
      "packet_id": "wp-001",
      "agent_id": "agent-007",
      "claimed_at": "2025-12-19T10:00:00+00:00",
      "trace_id": "trace-abc123",
      "parent_packet_id": "parent-001",
      "user_id": "dev_team",
      "task_title": "Implement feature X",
      "age_seconds": 450
    }
  ],
  "count": 1,
  "threshold_seconds": 300
}
```

**Use Cases:**
- Health monitoring dashboards
- Detecting silent agent crashes
- Finding work that needs manual intervention
- Alerting on long-running tasks

### POST /tools/get_packet_events

Get the event timeline for a specific packet (lifecycle events logged by agents).

**Request:**
```json
{
  "packet_id": "wp-001",
  "limit": 50
}
```

**Example:**
```bash
curl -s -X POST "http://172.17.0.1:8000/tools/get_packet_events" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: vmk_mcp_903e1bcb70d704da4fbf207722c471ba" \
  -d '{"packet_id": "wp-001", "limit": 50}'
```

**Response:**
```json
{
  "packet_id": "wp-001",
  "events": [
    {
      "event_type": "work_started",
      "agent_id": "coding_agent",
      "timestamp": "2025-12-19T12:19:10.781743+00:00",
      "trace_id": "a7f5c6dc",
      "metadata": {"parent_packet_id": "parent-001"},
      "score": 1766146750.781
    },
    {
      "event_type": "coder_completed",
      "agent_id": "coding_agent",
      "timestamp": "2025-12-19T12:20:15.234567+00:00",
      "trace_id": "a7f5c6dc",
      "metadata": {"status": "COMPLETE", "files_modified": ["src/main.py"]},
      "score": 1766146815.234
    },
    {
      "event_type": "error",
      "agent_id": "coding_agent",
      "timestamp": "2025-12-19T12:19:10.820325+00:00",
      "trace_id": "a7f5c6dc",
      "metadata": {
        "error_type": "OSError",
        "error_message": "[Errno 7] Argument list too long"
      },
      "score": 1766146750.820
    }
  ],
  "total": 3
}
```

**Event Types:**
- `work_started` - Agent claimed packet from queue
- `coder_started` - Coding phase began
- `coder_completed` - Coding phase finished
- `reviewer_started` - Review phase began
- `reviewer_completed` - Review phase finished
- `error` - Error occurred during execution

**Use Cases:**
- Debugging "why did this fail?"
- Understanding execution timeline
- Correlating events across packets (via trace_id)
- Replay analysis for learning

### POST /tools/log_execution_event

**Note:** This endpoint is for agents to log events. Search queries use `get_packet_events` above.

**Request:**
```json
{
  "packet_id": "wp-001",
  "event_type": "coder_completed",
  "agent_id": "coding_agent",
  "trace_id": "a7f5c6dc",
  "metadata": {"status": "COMPLETE", "files_modified": ["src/main.py"]}
}
```

---

## Trajectory & Error Search (Historical)

### POST /api/v1/trajectories/search

Search agent execution trajectories with semantic search and filters.

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

**Time-Based Filtering:**
- `hours_ago` parameter filters records created within last N hours
- Uses `timestamp_unix` for efficient range queries
- Only records with `timestamp_unix` are included in time-filtered searches

### POST /api/v1/errors/search

Search structured error logs with semantic search and filters.

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

---

## Debugging Workflow

### Scenario 1: "Work packet disappeared"

**Problem:** Packet published to queue but no trajectory logged.

**Steps:**
1. Check if packet was claimed:
```bash
curl -s "http://172.17.0.1:8000/tools/stuck_packets?user_id=dev_team&threshold=60"
```

2. Check event timeline:
```bash
curl -s -X POST "http://172.17.0.1:8000/tools/get_packet_events" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: vmk_mcp_903e1bcb70d704da4fbf207722c471ba" \
  -d '{"packet_id": "wp-001", "limit": 50}'
```

3. Check Redis completion record:
```python
import redis, json, os
from dotenv import load_dotenv
load_dotenv("/claude-workspace/.env")

r = redis.Redis(
    host="redis", port=6379,
    password=os.environ["REDIS_PASSWORD"],
    decode_responses=True
)

completion = r.get("dev_team:completions:wp-001")
if completion:
    print(json.dumps(json.loads(completion), indent=2))
```

4. Search trajectories for similar failures:
```bash
curl -X POST http://172.17.0.1:8000/api/v1/trajectories/search \
  -d '{"task_id": "wp-001", "limit": 50}'
```

### Scenario 2: "Agent crashed silently"

**Problem:** Agent started work but never completed.

**Steps:**
1. Find stuck packets (> 5 min old):
```bash
curl -s "http://172.17.0.1:8000/tools/stuck_packets?user_id=dev_team&threshold=300"
```

2. Check event timeline for last known state:
```bash
curl -s -X POST "http://172.17.0.1:8000/tools/get_packet_events" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: vmk_mcp_903e1bcb70d704da4fbf207722c471ba" \
  -d '{"packet_id": "wp-123", "limit": 50}'
```

3. Look for error events in timeline or search error logs:
```bash
curl -X POST http://172.17.0.1:8000/api/v1/errors/search \
  -d '{"trace_id": "trace-abc", "limit": 20}'
```

### Scenario 3: "Same error keeps happening"

**Problem:** Pattern of failures, need to find root cause.

**Steps:**
1. Search errors semantically:
```bash
curl -X POST http://172.17.0.1:8000/api/v1/errors/search \
  -d '{"query": "connection refused redis", "hours_ago": 24, "limit": 10}'
```

2. Search failed trajectories:
```bash
curl -X POST http://172.17.0.1:8000/api/v1/trajectories/search \
  -d '{"outcome": "failure", "agent": "coding_agent", "hours_ago": 24}'
```

3. Check if packets are getting stuck at same phase:
```bash
curl -s "http://172.17.0.1:8000/tools/stuck_packets?user_id=dev_team&threshold=300"
```

---

## Python Helper Functions

```python
import httpx
import redis
import json
import os
from dotenv import load_dotenv

# Load environment
load_dotenv("/claude-workspace/.env")

# Redis client for real-time observability
def get_redis():
    return redis.Redis(
        host="redis",
        port=6379,
        password=os.environ["REDIS_PASSWORD"],
        decode_responses=True
    )

# Real-time observability
async def get_stuck_packets(user_id="dev_team", threshold=300):
    """Find packets claimed but not completed."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://172.17.0.1:8000/tools/stuck_packets",
            params={"user_id": user_id, "threshold": threshold}
        )
        return response.json()

async def get_packet_events(packet_id, limit=50):
    """Get event timeline for a packet."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://172.17.0.1:8000/tools/get_packet_events",
            headers={"X-API-Key": os.environ.get("VERIS_API_KEY_MCP")},
            json={"packet_id": packet_id, "limit": limit}
        )
        return response.json()

def check_completion_status(packet_id, user_id="dev_team"):
    """Check Redis completion record."""
    r = get_redis()
    key = f"{user_id}:completions:{packet_id}"
    data = r.get(key)
    return json.loads(data) if data else None

def check_active_work(user_id="dev_team"):
    """List all active work for a team."""
    r = get_redis()
    keys = r.keys(f"{user_id}:active_work:*")
    active = []
    for key in keys:
        data = json.loads(r.get(key))
        data['ttl'] = r.ttl(key)
        active.append(data)
    return active

# Historical search
async def search_trajectories(query=None, agent=None, outcome=None, task_id=None, hours_ago=None, limit=20):
    """Search trajectories with optional filters."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://172.17.0.1:8000/api/v1/trajectories/search",
            json={
                "query": query,
                "agent": agent,
                "outcome": outcome,
                "task_id": task_id,
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
stuck = await get_stuck_packets(threshold=300)
events = await get_packet_events("wp-001")
completion = check_completion_status("wp-001")
active = check_active_work()

failures = await search_trajectories(outcome="failure", hours_ago=24)
errors = await search_errors(service="orchestrator", limit=10)
```

---

## Quick Reference

### Real-Time (Redis)
| Endpoint | Purpose | TTL |
|----------|---------|-----|
| `/tools/stuck_packets` | Find abandoned work | Active work: 10min |
| `/tools/get_packet_events` | Packet event timeline | Events: 24h |
| `/tools/log_execution_event` | Log lifecycle event | Events: 24h |

### Historical (Neo4j/Qdrant)
| Endpoint | Purpose | Storage |
|----------|---------|---------|
| `/api/v1/trajectories/search` | Find agent executions | Permanent |
| `/api/v1/errors/search` | Find structured errors | Permanent |
| `/api/v1/trajectories/log` | Log trajectory | Permanent |
| `/api/v1/errors/log` | Log error | Permanent |

---

## PROHIBITED

- Do NOT use these endpoints to write or modify data (search/read only)
- Do NOT expose sensitive error context in user-facing responses
- Do NOT make excessive queries (use filters to narrow results)
- Do NOT assume all errors are critical (check error_type and context)
- Do NOT rely solely on Redis data for historical analysis (use trajectory search)
- Do NOT expect events older than 24h in Redis (use trajectory search instead)
