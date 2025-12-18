"""
Veris Audit Infrastructure - Phase 1: Append-Only Foundation

"Truth, remembered — especially when it wounds."

This module implements the audit infrastructure that enables:
- TRUTH: Honest failures, logged with full context
- GOVERNANCE: Accountable decisions with full provenance
- LEARNING: Trajectories that feed precedent and skill evolution
- MEMORY: Persistence that cannot be rewritten into fiction

All audit entries are:
- Append-only (composite keys prevent overwrite)
- Hash-chained (each entry references previous hash)
- Signed (Ed25519, stub until Vault/HSM integration)
- Schema-validated (JSONL format)
"""

from .models import AuditEntry, AuditAction, RetentionClass
from .service import AuditService
from .crypto import AuditSigner
from .wal import WriteAheadLog

__all__ = [
    "AuditEntry",
    "AuditAction",
    "RetentionClass",
    "AuditService",
    "AuditSigner",
    "WriteAheadLog",
]
