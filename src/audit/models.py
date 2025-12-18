"""
Audit Entry Models

Defines the schema for audit entries with:
- Composite keys (UUID + timestamp) for append-only semantics
- Hash chain linking for tamper evidence
- Retention classification for tiered storage
- Actor provenance for accountability
"""

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class AuditAction(str, Enum):
    """Types of auditable actions."""

    # Memory mutations
    STORE_CONTEXT = "store_context"
    UPDATE_CONTEXT = "update_context"
    DELETE_CONTEXT = "delete_context"

    # Scratchpad operations
    UPDATE_SCRATCHPAD = "update_scratchpad"
    CLEAR_SCRATCHPAD = "clear_scratchpad"

    # Read operations (shadow reads)
    RETRIEVE_CONTEXT = "retrieve_context"
    QUERY_GRAPH = "query_graph"
    GET_SCRATCHPAD = "get_scratchpad"

    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_CHANGE = "config_change"

    # Error events
    VALIDATION_ERROR = "validation_error"
    RECOVERY_ATTEMPT = "recovery_attempt"
    CRYPTO_FAILURE = "crypto_failure"

    # Provenance events
    SIGNATURE_CREATED = "signature_created"
    SIGNATURE_VERIFIED = "signature_verified"
    SIGNATURE_FAILED = "signature_failed"


class RetentionClass(str, Enum):
    """Retention tier classification.

    Declared at write time by the actor, not inferred later.
    This prevents accidental lobotomy during compression.
    """

    # Hot tier: 7 days, high-fidelity
    EPHEMERAL = "ephemeral"

    # Warm tier: 30-90 days, compressed
    TRACE = "trace"

    # Cold tier: forever, signed digest
    SCAR = "scar"


class AuditEntry(BaseModel):
    """
    A single audit log entry.

    Append-only semantics enforced by composite key:
    - Primary key: {id}_{timestamp_epoch_ns}
    - This prevents any overwrite via upsert

    Hash chain linking:
    - prev_hash: SHA256 of previous entry's canonical JSON
    - entry_hash: SHA256 of this entry (computed after signing)

    Signature:
    - signature: Ed25519 signature of entry_hash
    - signer_id: ID of the signing key (for Vault lookup)
    """

    # Identity (composite key components)
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Action metadata
    action: AuditAction
    actor_id: str = Field(..., description="Who initiated this action")
    actor_type: str = Field(..., description="human | agent | system")

    # Target metadata
    target_id: Optional[str] = Field(None, description="ID of affected resource")
    target_type: Optional[str] = Field(None, description="context | scratchpad | config")

    # Payload
    input_snapshot: Optional[Dict[str, Any]] = Field(
        None, description="Sanitized input that triggered this action"
    )
    output_snapshot: Optional[Dict[str, Any]] = Field(
        None, description="Result or error details"
    )
    delta: Optional[Dict[str, Any]] = Field(
        None, description="What changed (for mutations)"
    )

    # Error tracking
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    recovery_metadata: Optional[Dict[str, Any]] = None

    # Retention
    retention_class: RetentionClass = Field(
        default=RetentionClass.TRACE,
        description="Declared by actor at write time"
    )
    compression_exempt: bool = Field(
        default=False,
        description="If True, never downsample this entry"
    )

    # Hash chain
    prev_hash: Optional[str] = Field(
        None, description="SHA256 of previous entry's canonical form"
    )
    entry_hash: Optional[str] = Field(
        None, description="SHA256 of this entry (set after creation)"
    )

    # Signature (Ed25519)
    signature: Optional[str] = Field(
        None, description="Base64-encoded Ed25519 signature"
    )
    signer_id: Optional[str] = Field(
        None, description="Key ID for Vault lookup"
    )
    signature_algorithm: str = Field(
        default="ed25519-stub",
        description="Algorithm used for signing"
    )

    # Indexing hints
    tags: List[str] = Field(default_factory=list)
    log_level: str = Field(
        default="system",
        description="system | redteam | user | scar | crypto"
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v):
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        return v

    @property
    def composite_key(self) -> str:
        """Generate composite key for append-only storage."""
        epoch_ns = int(self.timestamp.timestamp() * 1_000_000_000)
        return f"{self.id}_{epoch_ns}"

    @property
    def hash_prefix(self) -> str:
        """First 8 chars of entry_hash for indexing."""
        if self.entry_hash:
            return self.entry_hash[:8]
        return ""

    def canonical_form(self, include_signature: bool = False) -> str:
        """
        Generate canonical JSON for hashing/signing.

        Excludes entry_hash and signature (those are computed from this).
        Keys are sorted for determinism.
        """
        data = self.model_dump(
            exclude={"entry_hash", "signature"} if not include_signature else {"entry_hash"},
            mode="json"
        )
        # Convert UUID and datetime to strings
        data["id"] = str(data["id"])
        data["timestamp"] = data["timestamp"]
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Compute SHA256 hash of canonical form."""
        canonical = self.canonical_form()
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant-compatible payload."""
        return {
            "id": str(self.id),
            "composite_key": self.composite_key,
            "timestamp": self.timestamp.isoformat(),
            "timestamp_epoch": int(self.timestamp.timestamp()),
            "action": self.action.value,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "retention_class": self.retention_class.value,
            "compression_exempt": self.compression_exempt,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
            "hash_prefix": self.hash_prefix,
            "signature": self.signature,
            "signer_id": self.signer_id,
            "signature_algorithm": self.signature_algorithm,
            "tags": self.tags,
            "log_level": self.log_level,
            "error_code": self.error_code,
            # Store complex fields as JSON strings
            "input_snapshot_json": json.dumps(self.input_snapshot) if self.input_snapshot else None,
            "output_snapshot_json": json.dumps(self.output_snapshot) if self.output_snapshot else None,
            "delta_json": json.dumps(self.delta) if self.delta else None,
            "error_message": self.error_message,
            "recovery_metadata_json": json.dumps(self.recovery_metadata) if self.recovery_metadata else None,
        }


class AuditChainHead(BaseModel):
    """Tracks the current head of the audit chain."""

    chain_id: str = Field(default="main", description="Chain identifier")
    head_entry_id: str
    head_hash: str
    head_timestamp: datetime
    entry_count: int

    def to_redis_dict(self) -> Dict[str, str]:
        """Convert to Redis hash-compatible dict."""
        return {
            "chain_id": self.chain_id,
            "head_entry_id": self.head_entry_id,
            "head_hash": self.head_hash,
            "head_timestamp": self.head_timestamp.isoformat(),
            "entry_count": str(self.entry_count),
        }

    @classmethod
    def from_redis_dict(cls, data: Dict[str, str]) -> "AuditChainHead":
        """Create from Redis hash."""
        return cls(
            chain_id=data["chain_id"],
            head_entry_id=data["head_entry_id"],
            head_hash=data["head_hash"],
            head_timestamp=datetime.fromisoformat(data["head_timestamp"]),
            entry_count=int(data["entry_count"]),
        )
