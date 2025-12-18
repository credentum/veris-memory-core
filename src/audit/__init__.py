"""
Veris Audit Infrastructure

"Truth, remembered — especially when it wounds."

This module implements the audit infrastructure that enables:
- TRUTH: Honest failures, logged with full context
- GOVERNANCE: Accountable decisions with full provenance
- LEARNING: Trajectories that feed precedent and skill evolution
- MEMORY: Persistence that cannot be rewritten into fiction

Phase 1 - Append-Only Foundation:
- Append-only storage (composite keys prevent overwrite)
- Hash-chained entries (each references previous hash)
- Signed entries (Ed25519, stub until Vault/HSM)
- Schema-validated (JSONL format)

Phase 2 - Provenance Graph:
- Artifact tracking in Neo4j
- Lineage relationships (DERIVED_FROM, SUPERSEDES)
- Signature chains (SIGNED, VERIFIED_BY)
- Path queries for tracing origins
"""

from .models import AuditEntry, AuditAction, RetentionClass
from .service import AuditService
from .crypto import AuditSigner
from .wal import WriteAheadLog
from .provenance import ProvenanceGraph

__all__ = [
    "AuditEntry",
    "AuditAction",
    "RetentionClass",
    "AuditService",
    "AuditSigner",
    "WriteAheadLog",
    "ProvenanceGraph",
]
