"""
Audit Cryptographic Signing

Ed25519 signature implementation with:
- Stub mode for development (logs but doesn't enforce)
- Vault integration path for production
- HSM roadmap placeholder

"Keys are the system's memory of choice.
 Store them where no one person can rewrite history alone."
"""

import base64
import hashlib
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Tuple

from loguru import logger


class SignerBackend(ABC):
    """Abstract base for signing backends."""

    @abstractmethod
    def sign(self, data: bytes) -> Tuple[bytes, str]:
        """Sign data, return (signature, key_id)."""
        pass

    @abstractmethod
    def verify(self, data: bytes, signature: bytes, key_id: str) -> bool:
        """Verify a signature."""
        pass

    @abstractmethod
    def get_key_id(self) -> str:
        """Get current signing key ID."""
        pass


class StubSigner(SignerBackend):
    """
    Stub signer for development.

    Does NOT provide real security - signs with a deterministic stub.
    All operations are logged for audit trail.

    NEVER use in production.
    """

    def __init__(self, stub_key_id: str = "stub-dev-key-001"):
        self.key_id = stub_key_id
        self._stub_secret = b"STUB_SECRET_NOT_FOR_PRODUCTION"
        logger.warning(
            "AuditSigner initialized with STUB backend. "
            "This provides NO security. Use Vault in production."
        )

    def sign(self, data: bytes) -> Tuple[bytes, str]:
        """Create stub signature (HMAC-SHA256, not Ed25519)."""
        import hmac

        signature = hmac.new(self._stub_secret, data, hashlib.sha256).digest()
        logger.debug(
            f"STUB signature created",
            key_id=self.key_id,
            data_hash=hashlib.sha256(data).hexdigest()[:16],
        )
        return signature, self.key_id

    def verify(self, data: bytes, signature: bytes, key_id: str) -> bool:
        """Verify stub signature."""
        import hmac

        if key_id != self.key_id:
            logger.warning(f"Key ID mismatch: expected {self.key_id}, got {key_id}")
            return False

        expected = hmac.new(self._stub_secret, data, hashlib.sha256).digest()
        is_valid = hmac.compare_digest(signature, expected)

        if not is_valid:
            logger.warning("STUB signature verification FAILED")

        return is_valid

    def get_key_id(self) -> str:
        return self.key_id


class VaultSigner(SignerBackend):
    """
    HashiCorp Vault Transit backend for Ed25519 signing.

    Requires:
    - VAULT_ADDR environment variable
    - VAULT_TOKEN or VAULT_ROLE_ID + VAULT_SECRET_ID
    - Transit secrets engine enabled
    - Ed25519 key created at transit/keys/{key_name}

    This is the production path until HSM integration.
    """

    def __init__(
        self,
        key_name: str = "veris-audit-signing-key",
        vault_addr: Optional[str] = None,
        vault_token: Optional[str] = None,
    ):
        self.key_name = key_name
        self.vault_addr = vault_addr or os.environ.get("VAULT_ADDR")
        self.vault_token = vault_token or os.environ.get("VAULT_TOKEN")

        if not self.vault_addr:
            raise ValueError(
                "VAULT_ADDR must be set for VaultSigner. "
                "Use StubSigner for development."
            )

        # Import hvac only when actually using Vault
        try:
            import hvac

            self._hvac = hvac
        except ImportError:
            raise ImportError(
                "hvac package required for Vault integration. "
                "Install with: pip install hvac"
            )

        self._client: Optional["hvac.Client"] = None
        logger.info(
            f"VaultSigner initialized",
            vault_addr=self.vault_addr,
            key_name=self.key_name,
        )

    @property
    def client(self):
        """Lazy-initialize Vault client."""
        if self._client is None:
            self._client = self._hvac.Client(
                url=self.vault_addr,
                token=self.vault_token,
            )
            if not self._client.is_authenticated():
                raise RuntimeError("Vault authentication failed")
        return self._client

    def sign(self, data: bytes) -> Tuple[bytes, str]:
        """Sign using Vault Transit."""
        # Vault expects base64-encoded input
        b64_data = base64.b64encode(data).decode()

        response = self.client.secrets.transit.sign_data(
            name=self.key_name,
            hash_input=b64_data,
            signature_algorithm="ed25519",
        )

        # Response format: vault:v1:base64signature
        signature_str = response["data"]["signature"]
        # Extract just the signature part
        sig_b64 = signature_str.split(":")[-1]
        signature = base64.b64decode(sig_b64)

        key_version = response["data"].get("key_version", "1")
        key_id = f"{self.key_name}:v{key_version}"

        logger.debug(
            f"Vault signature created",
            key_id=key_id,
            data_hash=hashlib.sha256(data).hexdigest()[:16],
        )

        return signature, key_id

    def verify(self, data: bytes, signature: bytes, key_id: str) -> bool:
        """Verify using Vault Transit."""
        b64_data = base64.b64encode(data).decode()
        b64_sig = base64.b64encode(signature).decode()

        # Reconstruct Vault signature format
        vault_sig = f"vault:v1:{b64_sig}"

        try:
            response = self.client.secrets.transit.verify_signed_data(
                name=self.key_name,
                hash_input=b64_data,
                signature=vault_sig,
                signature_algorithm="ed25519",
            )
            return response["data"]["valid"]
        except Exception as e:
            logger.warning(f"Vault signature verification failed: {e}")
            return False

    def get_key_id(self) -> str:
        """Get current key version from Vault."""
        response = self.client.secrets.transit.read_key(name=self.key_name)
        latest_version = response["data"]["latest_version"]
        return f"{self.key_name}:v{latest_version}"


class AuditSigner:
    """
    Main audit signing interface.

    Automatically selects backend based on environment:
    - If VAULT_ADDR is set: VaultSigner (production)
    - Otherwise: StubSigner (development)

    Usage:
        signer = AuditSigner()
        signature, key_id = signer.sign(entry_hash_bytes)
        is_valid = signer.verify(entry_hash_bytes, signature, key_id)
    """

    def __init__(
        self,
        backend: Optional[SignerBackend] = None,
        force_stub: bool = False,
    ):
        if backend:
            self._backend = backend
        elif force_stub or not os.environ.get("VAULT_ADDR"):
            self._backend = StubSigner()
        else:
            self._backend = VaultSigner()

        self._sign_count = 0
        self._verify_count = 0
        self._started_at = datetime.now(timezone.utc)

    @property
    def backend_type(self) -> str:
        """Return the type of backend in use."""
        return self._backend.__class__.__name__

    @property
    def is_stub(self) -> bool:
        """Check if using stub (insecure) backend."""
        return isinstance(self._backend, StubSigner)

    def sign(self, data: bytes) -> Tuple[str, str]:
        """
        Sign data and return (base64_signature, key_id).

        The signature is over the raw bytes (typically a hash).
        """
        signature_bytes, key_id = self._backend.sign(data)
        self._sign_count += 1

        return base64.b64encode(signature_bytes).decode(), key_id

    def sign_hash(self, hash_hex: str) -> Tuple[str, str]:
        """
        Sign a hex-encoded hash string.

        Convenience method for signing entry_hash values.
        """
        hash_bytes = bytes.fromhex(hash_hex)
        return self.sign(hash_bytes)

    def verify(self, data: bytes, signature_b64: str, key_id: str) -> bool:
        """Verify a base64-encoded signature."""
        try:
            signature_bytes = base64.b64decode(signature_b64)
            is_valid = self._backend.verify(data, signature_bytes, key_id)
            self._verify_count += 1
            return is_valid
        except Exception as e:
            logger.warning(f"Signature verification error: {e}")
            return False

    def verify_hash(self, hash_hex: str, signature_b64: str, key_id: str) -> bool:
        """Verify signature over a hex-encoded hash."""
        hash_bytes = bytes.fromhex(hash_hex)
        return self.verify(hash_bytes, signature_b64, key_id)

    def get_stats(self) -> dict:
        """Get signing statistics."""
        return {
            "backend_type": self.backend_type,
            "is_stub": self.is_stub,
            "key_id": self._backend.get_key_id(),
            "sign_count": self._sign_count,
            "verify_count": self._verify_count,
            "uptime_seconds": (datetime.now(timezone.utc) - self._started_at).total_seconds(),
        }
