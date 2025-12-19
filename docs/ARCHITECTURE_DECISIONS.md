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
| 2025-12-19 | VMC-ADR-015: Saga Failure Recovery | Proposed | `30d134cc-e2d0-44f7-b63b-9138ef93202e` | `VMC-ADR-015 saga failure recovery compensation pattern` |
| 2025-12-19 | VMC-ADR-014: Cross-Agent Trust Protocol | Proposed | `b4954378-ba21-4680-927e-0adb2d217fcb` | `VMC-ADR-014 cross-agent trust protocol attestation` |
| 2025-12-19 | VMC-ADR-013: Success Pattern Learning | Proposed | `6ede8763-1870-49a7-bcc2-ee2aad35a064` | `VMC-ADR-013 success pattern learning positive precedents` |
| 2025-12-19 | VMC-ADR-012: Forgetting/Decay Mechanism | Proposed | `ebdac157-7938-4301-b580-900b00c0882e` | `VMC-ADR-012 forgetting decay mechanism memory pruning` |
| 2025-12-19 | VMC-ADR-011: Security Infrastructure | Implemented | `e20a97e7-2ec1-4266-9e6a-112a8a8397b5` | `VMC-ADR-011 security WAF RBAC secrets compliance` |
| 2025-12-19 | VMC-ADR-010: Retrieval Enhancements | Implemented | `53ef660f-0707-46bb-a714-61123d547467` | `VMC-ADR-010 HyDE cross-encoder sparse embeddings hybrid search` |
| 2025-12-19 | VMC-ADR-009: Audit Log HTTP Endpoints | Deferred | `d3c2dd9e-cfaf-4ef7-955e-11a769089545` | `VMC-ADR-009 audit log endpoints WAL verify hash chain` |
| 2025-12-19 | VMC-ADR-008: Episodic Memory System | Deferred | `b05fed38-eb86-46b0-8731-e62dcfdc7fc8` | `VMC-ADR-008 episodic memory episode boundaries replay summarization deferred` |
| 2025-12-19 | VMC-ADR-007: Vault HSM Integration | Deferred | `7f5c1c8d-e4af-46ed-b361-03a1c72ffa4d` | `VMC-ADR-007 Vault HSM signing Ed25519 compliance deferred` |
| 2025-12-19 | VMC-ADR-006: Cold Storage Archival | Deferred | `4804c7dd-9281-4af7-9816-10805fb8610f` | `VMC-ADR-006 cold storage archival SCAR S3 deferred` |
| 2025-12-19 | VMC-ADR-005: Conflict Resolution | Implemented | `8998e997-e185-4376-affe-bb2e5dd4dad0` | `ADR-005 conflict resolution list resolve MCP tools` |
| 2025-12-19 | Phase 4: Covenant Mediator | Deployed | `93a0dfb7-50b8-45b0-8509-8bbe82452d3d` | `covenant mediator phase 4 memory gating` |
| 2025-12-19 | System Architecture Map | Active | `2867afb3-aae2-4755-8540-cb0b90c75f37` | `veris system architecture map layers status what is built` |
| 2025-12-18 | Veris Memory Two-Tier Pattern | Documented | `f1b0faf7-a310-47bb-bd36-1ef8a3d7ba68` | `veris memory two-tier pattern semantic operational` |
| 2025-12-02 | S3-Paraphrase-Robustness Fix | Implemented | `2a76cda2-7c8d-4d7b-8a45-bd71427cee7b` | `S3 paraphrase robustness semantic search fix` |

---

## Decision Details

### 2025-12-19: VMC-ADR-015 - Saga Failure Recovery (Proposed)

**Status:** Proposed (P2)
**Target:** Q2 2026
**GitHub Issue:** [#79](https://github.com/credentum/veris-memory-core/issues/79)

**Problem:** When multi-step agent workflow fails at step 7 of 12, no compensation pattern exists. The system cannot recover gracefully.

**Proposed Solution:** Saga pattern with compensation actions for complex agent workflows.

**Design Considerations:**

| Aspect | Approach |
|--------|----------|
| Checkpoint Storage | Persist saga state at each step |
| Compensation Registry | Define rollback action per step type |
| Rollback Strategy | Partial vs full rollback based on failure type |
| Dead Letter Queue | Unrecoverable sagas for manual review |

**Integration Points:**
- Trajectory logging (capture saga state)
- Agent-dev pipeline (workflow orchestration)
- Audit trail (compensation actions logged)

**Activation Triggers:**

| Condition | Metric | Threshold |
|-----------|--------|-----------|
| 1 | `multi_step_task_count` | 100 |
| 2 | `mid_saga_failure_rate` | 5% |
| 3 | `manual_recovery_requests` | 10 |

**Council Rationale:** "What happens when this fails halfway through the saga?" - The Orchestrator

---

### 2025-12-19: VMC-ADR-014 - Cross-Agent Trust Protocol (Proposed)

**Status:** Proposed (P2)
**Target:** Q2 2026
**GitHub Issue:** [#78](https://github.com/credentum/veris-memory-core/issues/78)

**Problem:** When Agent A tells Agent B a fact, B cannot verify chain of custody. Multi-agent systems need trust protocols.

**Proposed Solution:** Attestation chain for multi-agent fact passing with cryptographic verification.

**Design Considerations:**

| Aspect | Approach |
|--------|----------|
| Attestation Format | Signed claims with provenance chain |
| Trust Levels | verified, attested, unverified |
| Propagation Rules | Trust decay across hops (e.g., -10% per hop) |
| Conflict Resolution | Weighted voting based on trust level |

**Integration Points:**
- `store_context` authority field
- Provenance graph (ATTESTED_BY relationships)
- Conflict resolution (trust-weighted decisions)

**Activation Triggers:**

| Condition | Metric | Threshold |
|-----------|--------|-----------|
| 1 | `multi_agent_task_count` | 50 |
| 2 | `cross_agent_fact_transfers` | 100 |
| 3 | `fact_conflict_rate` | 3% |

**Council Rationale:** "Multi-agent systems in 2026 will need trust protocols we haven't built." - The Auditor

---

### 2025-12-19: VMC-ADR-013 - Success Pattern Learning (Proposed)

**Status:** Proposed (P1)
**Target:** Q1 2026
**GitHub Issue:** [#77](https://github.com/credentum/veris-memory-core/issues/77)

**Problem:** Current system only learns from failures (pessimistic bias). Missing patterns from successful trajectories leads to incomplete learning.

**Proposed Solution:** Extract SUCCESS precedents from completed tasks, not just FAILURE precedents.

**Design Considerations:**

| Aspect | Approach |
|--------|----------|
| Success Criteria | Define per task type (e.g., tests pass, PR merged) |
| Confidence Scoring | Higher confidence for repeated success patterns |
| Overfitting Prevention | Require N occurrences before extracting pattern |
| Balance | Weight success vs failure precedents (e.g., 40/60) |

**New Precedent Types:**

| Type | Description |
|------|-------------|
| `SUCCESS` | Task completed successfully |
| `SKILL` | Learned capability (e.g., "knows how to deploy") |
| `OPTIMIZATION` | Efficiency improvement pattern |

**Current vs Proposed:**

```
Current:  FAILURE → Extract → Precedent → Avoid
Proposed: SUCCESS → Extract → Precedent → Replicate
          FAILURE → Extract → Precedent → Avoid
```

**Activation Triggers:**

| Condition | Metric | Threshold |
|-----------|--------|-----------|
| 1 | `trajectory_count` | 200 |
| 2 | `success_trajectory_count` | 50 |
| 3 | `learning_request_for_success` | 1 |

**Council Rationale:** "An immune system that only remembers diseases, never health, will be outcompeted." - The Immune System Designer

---

### 2025-12-19: VMC-ADR-012 - Forgetting/Decay Mechanism (Proposed)

**Status:** Proposed (P1)
**Target:** Q1 2026
**GitHub Issue:** [#76](https://github.com/credentum/veris-memory-core/issues/76)

**Problem:** Memory without forgetting degrades retrieval quality at scale. Vector spaces get noisy, graph queries slow. At 100,000 entries, the system will struggle.

**Proposed Solution:** Time-weighted relevance decay with configurable half-life per context type.

**Decay Function Options:**

| Function | Formula | Use Case |
|----------|---------|----------|
| Exponential | `score × e^(-λt)` | Natural memory decay |
| Linear | `score × (1 - t/T)` | Predictable deprecation |
| Step | `score if t < T else 0` | Hard expiration |

**Type-Specific Half-Lives (Proposed):**

| Context Type | Half-Life | Rationale |
|--------------|-----------|-----------|
| decision | 365 days | Architectural decisions persist |
| design | 180 days | Designs evolve |
| precedent | 90 days | Patterns may become stale |
| log | 30 days | Operational data expires |
| trace | 14 days | Debug data short-lived |

**Consolidation Strategy:**
- When similar memories decay, merge into single consolidated memory
- Preserve highest-authority version
- Link to original sources in provenance

**Resurrection:**
- If decayed memory is accessed, boost relevance
- "Use it or lose it" principle

**Activation Triggers:**

| Condition | Metric | Threshold |
|-----------|--------|-----------|
| 1 | `qdrant_entry_count` | 50,000 |
| 2 | `retrieval_precision_drop` | 10% |
| 3 | `graph_query_latency_p99_ms` | 500 |

**Council Rationale:** "Without forgetting, memory degrades at scale. Every memory system in nature prunes." - The Archivist

---

### 2025-12-19: VMC-ADR-011 - Security Infrastructure (Implemented)

**Status:** Implemented
**GitHub Issue:** [#74](https://github.com/credentum/veris-memory-core/issues/74)

**Context:** Document the comprehensive security infrastructure that has been built but was not previously recorded as an ADR.

**Components:**

| Component | File | Purpose | Status |
|-----------|------|---------|--------|
| API Key Auth | `src/middleware/api_key_auth.py` | RBAC with human-only operations | ✅ Sprint 13 |
| Scope Validator | `src/middleware/scope_validator.py` | Scope-based authorization | ✅ Complete |
| WAF | `src/security/waf.py` | Web Application Firewall | ✅ Complete |
| Secrets Manager | `src/security/secrets_manager.py` | Credential protection | ✅ Complete |
| Compliance Reporter | `src/security/compliance_reporter.py` | Security scanning | ✅ Complete |
| Cypher Validator | `src/security/cypher_validator.py` | Query injection prevention | ✅ Complete |
| Fact Privacy | `src/security/fact_privacy.py` | Sensitive data masking | ✅ Complete |
| Port Filter | `src/security/port_filter.py` | Network security | ✅ Complete |
| Security Scanner | `src/security/security_scanner.py` | Vulnerability detection | ✅ Complete |

**WAF Rules (16 default rules):**
- SQL Injection detection
- XSS prevention
- Path traversal blocking
- Command injection detection
- LDAP injection prevention
- XML/XXE attack blocking
- And more...

**RBAC Roles:**

| Role | Permissions |
|------|-------------|
| `admin` | Full access |
| `writer` | Create, update, delete |
| `reader` | Read-only |
| `guest` | Limited read |

**Human-Only Operations (Sprint 13):**
- `delete_context` - Hard delete
- `forget_context` - Soft delete with retention

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_REQUIRED` | false | Enable API key authentication |
| `VERIS_API_KEY_*` | - | API keys with metadata |

**Agent Instructions:**
> Security is always on. WAF rules apply to all requests. Respect human-only operation restrictions.

---

### 2025-12-19: VMC-ADR-010 - Retrieval Enhancements (Implemented)

**Status:** Implemented
**GitHub Issue:** [#75](https://github.com/credentum/veris-memory-core/issues/75)

**Context:** Document the advanced retrieval features that improve search quality beyond basic semantic search.

**Components:**

| Feature | File | Purpose | Default |
|---------|------|---------|---------|
| HyDE Generator | `src/core/hyde_generator.py` | Hypothetical Document Embeddings | Enabled |
| Cross-Encoder Reranker | `src/core/retrieval_core.py` | Re-rank results for precision | Enabled |
| Sparse Embeddings | `src/embedding/sparse_service.py` | BM25 hybrid search | Enabled |
| Query Dispatcher | `src/core/query_dispatcher.py` | Multi-backend routing | Always |
| Semantic Cache | `src/core/semantic_cache.py` | Cache with semantic keys | Enabled |
| Query Normalizer | `src/core/query_normalizer.py` | Query optimization | Enabled |
| Intent Classifier | `src/core/intent_classifier.py` | Query intent analysis | Always |

**HyDE (Hypothetical Document Embeddings):**

Generates hypothetical answer documents from queries for better semantic matching.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYDE_ENABLED` | true | Enable HyDE generation |
| `HYDE_API_PROVIDER` | openrouter | LLM provider |
| `HYDE_MODEL` | meta-llama/llama-2-7b | Model for generation |
| `HYDE_BASE_URL` | https://openrouter.ai/api/v1 | API endpoint |
| `HYDE_MAX_TOKENS` | 150 | Max tokens for hypothetical doc |
| `HYDE_TEMPERATURE` | 0.7 | Generation temperature |
| `HYDE_CACHE_ENABLED` | true | Cache hypothetical docs |
| `HYDE_LLM_TIMEOUT` | 30.0 | Timeout in seconds |
| `OPENROUTER_API_KEY` | - | Required for HyDE |

**Cross-Encoder Reranker:**

Re-ranks initial retrieval results using a cross-encoder model for better relevance.

| Variable | Default | Description |
|----------|---------|-------------|
| `CROSS_ENCODER_RERANKER_ENABLED` | true | Enable reranking |
| `CROSS_ENCODER_TOP_K` | 50 | Candidates to re-rank |
| `CROSS_ENCODER_RETURN_K` | 10 | Results to return |
| `CROSS_ENCODER_FALLBACK_THRESHOLD` | -5.0 | Score threshold |

**Sparse Embeddings (Hybrid Search):**

Combines dense (semantic) with sparse (BM25) vectors for keyword matching.

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARSE_EMBEDDINGS_ENABLED` | true | Enable sparse vectors |
| `SPARSE_EMBEDDING_MODEL` | Qdrant/bm25 | Sparse model |

**Search Enhancements:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MQE_ENABLED` | true | Multi-Query Expansion |
| `MQE_NUM_PARAPHRASES` | 2 | Number of query variants |
| `SEARCH_ENHANCEMENTS_ENABLED` | true | Enable all enhancements |
| `QUERY_NORMALIZATION_ENABLED` | true | Normalize queries |

**Retrieval Pipeline:**

```
Query → Intent Classifier → Query Normalizer → HyDE Generator
    → Multi-Query Expander → Semantic Cache Check
    → [Cache Miss] → Sparse + Dense Search → Cross-Encoder Rerank
    → Results
```

**Agent Instructions:**
> Retrieval enhancements are enabled by default. Disable only for debugging. HyDE requires OPENROUTER_API_KEY.

---

### 2025-12-19: VMC-ADR-009 - Audit Log HTTP Endpoints (Deferred)

**Status:** Deferred
**GitHub Issue:** [#73](https://github.com/credentum/veris-memory-core/issues/73)

**Context:** Audit infrastructure code is built (WAL, crypto, provenance) but some HTTP endpoints are not exposed. The code works internally but lacks direct API access.

**Current State:**

| Phase | Code | Endpoints |
|-------|------|-----------|
| Phase 1: Audit Log | ✅ Built | ❌ Not exposed |
| Phase 2: Provenance | ✅ Built | ⚠️ Partial |
| Phase 3: Retention | ✅ Built | ✅ Complete |

**Missing Endpoints:**

| Endpoint | Purpose | File |
|----------|---------|------|
| `POST /audit/log` | Write audit entries directly | `src/audit/wal.py` |
| `GET /audit/log/{id}` | Retrieve specific audit entry | `src/audit/wal.py` |
| `POST /audit/verify` | Verify hash chain integrity | `src/audit/crypto.py` |
| `POST /audit/provenance/link` | Explicitly link artifacts | `src/audit/provenance.py` |

**Working Endpoints:**

| Endpoint | Status |
|----------|--------|
| `POST /audit/trace/{id}` | ✅ Works |
| `POST /audit/descendants/{id}` | ✅ Works |
| `POST /audit/retention/process` | ✅ Works |
| `POST /learning/precedents/store` | ✅ Works |
| `POST /learning/precedents/query` | ✅ Works |
| `GET /learning/stats` | ✅ Works |

**Decision:** `DEFER_IMPLEMENTATION`

**Rationale:** Internal usage is sufficient for now. Endpoints would be needed for external audit tools or compliance dashboards.

**Activation Triggers:**

| Condition | Metric | Threshold |
|-----------|--------|-----------|
| 1 | `external_audit_tool_integration` | 1 |
| 2 | `compliance_dashboard_request` | 1 |
| 3 | `hash_chain_verification_needed` | 1 |

**Agent Instructions:**
> Do not expose audit log endpoints until activation triggers are met. Internal audit functionality works via other operations.

---

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

### 2025-12-19: VMC-ADR-005 - Conflict Resolution (Implemented)

**Status:** Implemented
**GitHub Issue:** [#66](https://github.com/credentum/veris-memory-core/issues/66)

**Context:** Phase 4 (Covenant Mediator) identifies memory contradictions. This ADR documents the implemented resolution mechanism.

**Implementation:**

| Component | File | Status |
|-----------|------|--------|
| Conflict Store | `src/storage/conflict_store.py` | ✅ Complete |
| List Conflicts | `POST /tools/list_conflicts` | ✅ Works |
| Resolve Conflict | `POST /tools/resolve_conflict` | ✅ Works |

**Resolution Modes:**

| Mode | Behavior |
|------|----------|
| `accept_new` | Replace existing context with new conflicting content |
| `keep_existing` | Discard the new content, keep existing |
| `merge` | Combine both contexts (requires manual review) |

**MCP Tools:**
```python
# List pending conflicts
list_conflicts() -> List[CovenantConflict]

# Resolve a specific conflict
resolve_conflict(conflict_id: str, resolution: str, reason: Optional[str])
```

**What's NOT Automated (Future Work):**
- Autonomous resolution without human/agent intervention
- Pattern-based automatic resolution rules
- Conflict prediction/prevention

**Agent Instructions:**
> Use `list_conflicts` to check for pending contradictions. Use `resolve_conflict` with explicit reasoning. Prefer `keep_existing` when uncertain.

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

## 2026 Roadmap (Covenant Council Review)

**Review Date:** 2025-12-19
**Verdict:** REQUIRES AMENDMENT
**Context ID:** `d5e1f35b-43ee-4a09-8fab-40e97399731e`

### Q1 2026 (P1 - Critical)

| Priority | ADR | Title | Rationale |
|----------|-----|-------|-----------|
| 1 | VMC-ADR-012 | Forgetting/Decay Mechanism | Without it, memory degrades at scale |
| 2 | VMC-ADR-013 | Success Pattern Learning | Breaks pessimistic bias |
| 3 | - | Metrics/Validation Dashboard | Must prove value before scaling |
| 4 | VMC-ADR-008 | Episodic Memory (Activate) | Temporal queries are table stakes |

### Q2 2026 (P2 - Important)

| Priority | ADR | Title | Rationale |
|----------|-----|-------|-----------|
| 5 | - | Real Production Workload | Validate with agent-dev at scale |
| 6 | VMC-ADR-014 | Cross-Agent Trust Protocol | Required for multi-agent future |
| 7 | VMC-ADR-015 | Saga Failure Recovery | Multi-step tasks need compensation |

### Q3 2026 (P3 - When Ready)

| Priority | ADR | Title | Rationale |
|----------|-----|-------|-----------|
| 8 | VMC-ADR-007 | HSM Integration (Activate) | Before external deployment |

---

## Proposed (2026)

- [ ] **VMC-ADR-012:** Forgetting/Decay Mechanism ([#76](https://github.com/credentum/veris-memory-core/issues/76)) - Q1 P1
- [ ] **VMC-ADR-013:** Success Pattern Learning ([#77](https://github.com/credentum/veris-memory-core/issues/77)) - Q1 P1
- [ ] **VMC-ADR-014:** Cross-Agent Trust Protocol ([#78](https://github.com/credentum/veris-memory-core/issues/78)) - Q2 P2
- [ ] **VMC-ADR-015:** Saga Failure Recovery ([#79](https://github.com/credentum/veris-memory-core/issues/79)) - Q2 P2

## Implemented

- [x] **VMC-ADR-011:** Security Infrastructure ([#74](https://github.com/credentum/veris-memory-core/issues/74)) - WAF, RBAC, Secrets
- [x] **VMC-ADR-010:** Retrieval Enhancements ([#75](https://github.com/credentum/veris-memory-core/issues/75)) - HyDE, Cross-Encoder, Sparse
- [x] **VMC-ADR-005:** Conflict Resolution ([#66](https://github.com/credentum/veris-memory-core/issues/66)) - MCP tools for list/resolve

## Deferred (With Activation Triggers)

- [ ] **VMC-ADR-009:** Audit Log HTTP Endpoints ([#73](https://github.com/credentum/veris-memory-core/issues/73))
- [ ] **VMC-ADR-008:** Episodic Memory System ([#72](https://github.com/credentum/veris-memory-core/issues/72)) - **Activate Q1 2026**
- [ ] **VMC-ADR-007:** Vault HSM Integration ([#71](https://github.com/credentum/veris-memory-core/issues/71)) - **Activate Q3 2026**
- [ ] **VMC-ADR-006:** Cold Storage Archival ([#70](https://github.com/credentum/veris-memory-core/issues/70))

## Not Yet Built (No ADR)

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
- [VMC-ADR-005: Conflict Resolution (Issue #66)](https://github.com/credentum/veris-memory-core/issues/66)
- [VMC-ADR-006: Cold Storage Archival (Issue #70)](https://github.com/credentum/veris-memory-core/issues/70)
- [VMC-ADR-007: Vault HSM Integration (Issue #71)](https://github.com/credentum/veris-memory-core/issues/71)
- [VMC-ADR-008: Episodic Memory System (Issue #72)](https://github.com/credentum/veris-memory-core/issues/72)
- [VMC-ADR-009: Audit Log HTTP Endpoints (Issue #73)](https://github.com/credentum/veris-memory-core/issues/73)
- [VMC-ADR-010: Retrieval Enhancements (Issue #75)](https://github.com/credentum/veris-memory-core/issues/75)
- [VMC-ADR-011: Security Infrastructure (Issue #74)](https://github.com/credentum/veris-memory-core/issues/74)
- [VMC-ADR-012: Forgetting/Decay Mechanism (Issue #76)](https://github.com/credentum/veris-memory-core/issues/76)
- [VMC-ADR-013: Success Pattern Learning (Issue #77)](https://github.com/credentum/veris-memory-core/issues/77)
- [VMC-ADR-014: Cross-Agent Trust Protocol (Issue #78)](https://github.com/credentum/veris-memory-core/issues/78)
- [VMC-ADR-015: Saga Failure Recovery (Issue #79)](https://github.com/credentum/veris-memory-core/issues/79)
