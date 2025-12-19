# Architecture Decision Records

Index of architecture decisions stored in Veris Memory for the veris-memory-core repository.

## How to Use

Each decision is stored in Veris Memory with full context. Use the recovery query to retrieve details:

```bash
curl -X POST http://172.17.0.1:8000/tools/retrieve_context \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $VERIS_API_KEY" \
  -d '{"query": "<recovery_query>", "limit": 5}'
```

Or query by context ID:

```bash
curl -X POST http://172.17.0.1:8000/tools/query_graph \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $VERIS_API_KEY" \
  -d '{"query": "MATCH (c:Context {id: \"<context_id>\"}) RETURN c"}'
```

---

## Decisions

| Date | Title | Status | Context ID | Recovery Query |
|------|-------|--------|------------|----------------|
| 2025-12-19 | VMC-ADR-008: Episodic Memory System | Deferred | `b05fed38-eb86-46b0-8731-e62dcfdc7fc8` | `VMC-ADR-008 episodic memory episode boundaries replay summarization deferred` |
| 2025-12-19 | VMC-ADR-007: Vault HSM Integration | Deferred | `7f5c1c8d-e4af-46ed-b361-03a1c72ffa4d` | `VMC-ADR-007 Vault HSM signing Ed25519 compliance deferred` |
| 2025-12-19 | VMC-ADR-006: Cold Storage Archival | Deferred | `4804c7dd-9281-4af7-9816-10805fb8610f` | `VMC-ADR-006 cold storage archival SCAR S3 deferred` |
| 2025-12-19 | VMC-ADR-005: Covenant Conflict Resolution | Deferred | `8998e997-e185-4376-affe-bb2e5dd4dad0` | `ADR-005 conflict resolution deferred automation trigger threshold` |
| 2025-12-19 | Phase 4: Covenant Mediator | Deployed | `93a0dfb7-50b8-45b0-8509-8bbe82452d3d` | `covenant mediator phase 4 memory gating` |
| 2025-12-19 | System Architecture Map | Active | `2867afb3-aae2-4755-8540-cb0b90c75f37` | `veris system architecture map layers status what is built` |
| 2025-12-18 | Veris Memory Two-Tier Pattern | Documented | `f1b0faf7-a310-47bb-bd36-1ef8a3d7ba68` | `veris memory two-tier pattern semantic operational` |
| 2025-12-02 | S3-Paraphrase-Robustness Fix | Implemented | `2a76cda2-7c8d-4d7b-8a45-bd71427cee7b` | `S3 paraphrase robustness semantic search fix` |

---

## Decision Details

### 2025-12-19: VMC-ADR-008 - Episodic Memory System (Deferred)

**Status:** Deferred (P3)
**GitHub Issue:** [#72](https://github.com/credentum/veris-memory-core/issues/72)

**Context:** System has trajectory memory (find similar failures) but lacks episodic memory (what happened during session X, summarize this week). Current episodic capability scores 4/10.

**Current Capabilities:**
- Raw events captured via trajectory logging
- Causation tracked via PRECEDED_BY relationships
- Semantic search over trajectories
- hours_ago time filtering (limited)

**Missing Capabilities:**
- Episode boundaries (what is an episode?)
- Date range queries ("what happened on December 15th?")
- Episode replay ("show task-xyz from start to finish")
- Episode summarization ("summarize this week's learnings")
- Pattern extraction across episodes

**Decision:** `DEFER_WITH_DESIGN`

**Rationale:** Trajectory system serves immediate need (learning from failures). Episodic memory is enriching but not essential today. Estimated effort: 2-3 sprints.

**Episode Schema (Preserved Design):**

```python
@dataclass
class Episode:
    episode_id: UUID
    episode_type: Literal["task", "session", "incident", "sprint"]
    start_time: datetime
    end_time: datetime
    trigger_event: str
    outcome: Literal["success", "failure", "partial", "abandoned"]
    trajectories: List[trajectory_id]
    summary: Optional[str]  # LLM-generated
    learnings: List[str]
    tags: List[str]
```

**Future Endpoints:**
- `GET /episodes?date=YYYY-MM-DD`
- `GET /episodes?start=X&end=Y`
- `GET /episodes/recent?limit=N`
- `GET /episodes/{id}/replay`
- `POST /episodes/{id}/summarize`

**Activation Triggers:**

| Condition | Metric | Threshold |
|-----------|--------|-----------|
| 1 | `trajectory_count` | 500 |
| 2 | `incident_replay_request` | 1 |
| 3 | `meta_precedent_extraction_needed` | boolean |
| 4 | `positive_learning_initiative` | boolean |

**Council Dissent:** The Confessor notes the system is pessimistically biased—learns from failures, not successes. Episodic memory would address this philosophical gap.

**Agent Instructions:**
> Do not implement episodic memory until activation triggers are met. Current trajectory system is sufficient for learning from failures.

---

### 2025-12-19: VMC-ADR-007 - Vault HSM Integration (Deferred)

**Status:** Deferred
**GitHub Issue:** [#71](https://github.com/credentum/veris-memory-core/issues/71)

**Context:** Audit entries are signed with Ed25519 but keys are in stub mode (in-memory). No HSM protection.

**Current State:** `crypto.py` has Vault path designed but uses stub signer. Keys generated at startup, lost on restart.

**Decision:** `DEFER_IMPLEMENTATION`

**Rationale:** No compliance requirement yet. Stub mode sufficient for development and testing.

**Future Interface:**
- Provider: HashiCorp Vault Transit Secrets Engine
- Key Type: ed25519
- Operations: sign, verify
- Rotation: Automatic via Vault policies

**Activation Triggers:**

| Condition | Metric | Threshold |
|-----------|--------|-----------|
| 1 | `compliance_requirement` | SOC2 or equivalent |
| 2 | `production_deployment` | customer_facing |
| 3 | `audit_tampering_attempt` | 1 |

**Agent Instructions:**
> Do not implement Vault integration until activation triggers are met. Stub mode is acceptable for internal use.

---

### 2025-12-19: VMC-ADR-006 - Cold Storage Archival (Deferred)

**Status:** Deferred
**GitHub Issue:** [#70](https://github.com/credentum/veris-memory-core/issues/70)

**Context:** SCAR retention entries are marked for archival after 90 days but remain in Qdrant indefinitely. No cold storage tier exists.

**Current State:** `retention.py` marks SCAR entries as 'archived' but doesn't move them anywhere.

**Decision:** `DEFER_IMPLEMENTATION`

**Rationale:** Storage costs are negligible at current scale. Premature optimization.

**Future Interface:**
- Destination: S3 or similar object storage
- Format: Parquet or JSON-LD for queryability
- Retrieval: On-demand rehydration to Qdrant

**Activation Triggers:**

| Condition | Metric | Threshold |
|-----------|--------|-----------|
| 1 | `qdrant_storage_gb` | 50 |
| 2 | `monthly_storage_cost_usd` | $100 |
| 3 | `scar_entry_count` | 100,000 |

**Agent Instructions:**
> Do not implement cold storage until activation triggers are met. Current archival marking is sufficient.

---

### 2025-12-19: VMC-ADR-005 - Covenant Conflict Resolution (Deferred)

**Status:** Deferred
**GitHub Issue:** [#66](https://github.com/credentum/veris-memory-core/issues/66)

**Context:** Phase 4 (Covenant Mediator) identifies memory contradictions but lacks a protocol to resolve them autonomously.

**Decision:** `DEFER_IMPLEMENTATION`

**Rationale:** Adherence to "Just-in-Time" infrastructure. Manual overhead is currently < 5% of operational time.

**Activation Triggers:**

| Condition | Metric | Threshold |
|-----------|--------|-----------|
| 1 | `blocking_packet_events` | 1 |
| 2 | `unresolved_conflict_count` | 10 |
| 3 | `pattern_emergence` | 3 repeats |

**Agent Instructions:**
> Do not attempt to build or suggest resolution code until one of the activation_triggers is met.

---

### 2025-12-19: Phase 4 - Covenant Mediator

**Status:** Deployed (feature-flagged)
**PR:** [#65](https://github.com/credentum/veris-memory-core/pull/65)

**Context:** Implement Titans-inspired memory gating (arXiv:2412.00341). Only information with sufficient novelty and authority gets committed to long-term memory.

**Core Formulas:**

```
Surprise = (1 - max_cosine_similarity) × (1 + token_novelty_bonus)
Weight = Surprise × (Authority / 10) × (1 + 0.5 × Sparsity)
```

**Thresholds by Type:**

| Type | Threshold |
|------|-----------|
| decision | 0.4 |
| design | 0.35 |
| log | 0.2 |
| trace | 0.15 |
| default | 0.3 |

**New MCP Tools:**
- `store_context` - Enhanced with `authority` parameter
- `list_conflicts` - List pending CovenantConflict nodes
- `resolve_conflict` - Resolve with accept_new, keep_existing, or merge

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `COVENANT_MEDIATOR_ENABLED` | false | Feature flag |
| `HIGH_AUTHORITY_BYPASS` | 8 | Authority threshold for bypass |
| `COVENANT_WEIGHT_THRESHOLD` | 0.3 | Default weight threshold |

**Related Contexts:**
- Feature Explanation: `607f5982-a304-4404-9100-1a4bf3da49fd`

---

### 2025-12-19: System Architecture Map

**Status:** Active
**GitHub Issue:** [#68](https://github.com/credentum/veris-memory-core/issues/68)

**Purpose:** Single source of truth for what's built across all Veris repos.

**Architecture Layers:**

| Layer | Name | Status | Repo |
|-------|------|--------|------|
| 1 | Covenant Mediator (Gating) | Implemented | veris-memory-core |
| 2 | Audit Log (WAL + Crypto) | Implemented | veris-memory-core |
| 3 | Provenance Graph | Implemented | veris-memory-core |
| 4 | Retention & Learning | Implemented | veris-memory-core |
| 5 | Storage Layer | Implemented | veris-memory-core |
| 6 | Hybrid Retrieval | Implemented | veris-memory-core |
| 7 | Safety Valve | Implemented | agent-dev |

**Cross-Repo Coordination:**
- agent-dev ADRs: https://github.com/credentum/agent-dev/blob/main/docs/ARCHITECTURE_DECISIONS.md

---

### 2025-12-18: Veris Memory Two-Tier Pattern

**Status:** Documented

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│  VERIS MEMORY (Semantic)                                        │
│  Store: Qdrant + Neo4j                                          │
│  Purpose: Knowledge that should be REMEMBERED                   │
│  Contains: decisions, designs, precedents, facts, learnings     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  OPERATIONAL STORE (Time-series)                                │
│  Store: Redis Streams                                           │
│  Purpose: Data for DEBUGGING and OBSERVABILITY                  │
│  Contains: logs, traces, metrics, errors                        │
└─────────────────────────────────────────────────────────────────┘
```

---

### 2025-12-02: S3-Paraphrase-Robustness Fix

**Status:** Implemented
**Test Results:** 121 tests passing (100%)

**Problem:** Semantically equivalent queries returned different results. Cache keys generated from raw query text instead of embeddings.

**Solution:** 4-phase fix with feature flags:
1. Semantic Cache Keys
2. MultiQueryExpander Integration
3. Search Enhancements Integration
4. Query Normalization

---

## Deferred (With Activation Triggers)

- [ ] **VMC-ADR-008:** Episodic Memory System ([#72](https://github.com/credentum/veris-memory-core/issues/72)) - P3
- [ ] **VMC-ADR-007:** Vault HSM Integration ([#71](https://github.com/credentum/veris-memory-core/issues/71))
- [ ] **VMC-ADR-006:** Cold Storage Archival ([#70](https://github.com/credentum/veris-memory-core/issues/70))
- [ ] **VMC-ADR-005:** Covenant Conflict Resolution Automation ([#66](https://github.com/credentum/veris-memory-core/issues/66))

## Not Yet Built (No ADR)

- [ ] Forgetting/decay mechanism
- [ ] Memory consolidation (sleep cycle)
- [ ] Cross-agent theory of mind

---

## Template for New Decisions

```markdown
### YYYY-MM-DD: VMC-ADR-XXX - Title

**Status:** Planning | Accepted | Implemented | Deferred | Superseded

**Context:** Why is this decision needed?

**Problem:** What problem are we solving?

**Decision:** What did we decide?

**Alternatives Considered:**
- Option A: ...
- Option B: ...

**Related PRs:**
- `#number` - Description

**Veris Memory:**
- Context ID: `uuid`
- Recovery Query: `query terms`
```

---

## Related Resources

- [System Architecture Map (Issue #68)](https://github.com/credentum/veris-memory-core/issues/68)
- [agent-dev Architecture Decisions](https://github.com/credentum/agent-dev/blob/main/docs/ARCHITECTURE_DECISIONS.md)
- [VMC-ADR-005: Covenant Conflict Resolution (Issue #66)](https://github.com/credentum/veris-memory-core/issues/66)
- [VMC-ADR-006: Cold Storage Archival (Issue #70)](https://github.com/credentum/veris-memory-core/issues/70)
- [VMC-ADR-007: Vault HSM Integration (Issue #71)](https://github.com/credentum/veris-memory-core/issues/71)
- [VMC-ADR-008: Episodic Memory System (Issue #72)](https://github.com/credentum/veris-memory-core/issues/72)
