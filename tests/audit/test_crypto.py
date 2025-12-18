"""
Tests for audit cryptographic signing.

Tests:
- StubSigner functionality
- Signature creation and verification
- Key ID management
- Stats tracking
"""

import pytest

from src.audit.crypto import AuditSigner, StubSigner


class TestStubSigner:
    """Tests for StubSigner backend."""

    def test_create_stub_signer(self):
        """Test creating a stub signer."""
        signer = StubSigner()

        assert signer.key_id == "stub-dev-key-001"

    def test_custom_key_id(self):
        """Test custom key ID."""
        signer = StubSigner(stub_key_id="custom-key")

        assert signer.key_id == "custom-key"

    def test_sign_returns_signature_and_key_id(self):
        """Test signing returns tuple of (signature, key_id)."""
        signer = StubSigner()
        data = b"test data to sign"

        signature, key_id = signer.sign(data)

        assert isinstance(signature, bytes)
        assert len(signature) == 32  # SHA256 output
        assert key_id == "stub-dev-key-001"

    def test_verify_valid_signature(self):
        """Test verifying a valid signature."""
        signer = StubSigner()
        data = b"test data to sign"

        signature, key_id = signer.sign(data)
        is_valid = signer.verify(data, signature, key_id)

        assert is_valid is True

    def test_verify_invalid_signature(self):
        """Test verifying an invalid signature."""
        signer = StubSigner()
        data = b"test data to sign"

        signature, key_id = signer.sign(data)
        # Tamper with signature
        tampered = b"\x00" * len(signature)

        is_valid = signer.verify(data, tampered, key_id)

        assert is_valid is False

    def test_verify_wrong_key_id(self):
        """Test verifying with wrong key ID."""
        signer = StubSigner()
        data = b"test data to sign"

        signature, _ = signer.sign(data)
        is_valid = signer.verify(data, signature, "wrong-key-id")

        assert is_valid is False

    def test_signature_determinism(self):
        """Test same data produces same signature."""
        signer = StubSigner()
        data = b"test data to sign"

        sig1, _ = signer.sign(data)
        sig2, _ = signer.sign(data)

        assert sig1 == sig2

    def test_different_data_different_signature(self):
        """Test different data produces different signatures."""
        signer = StubSigner()

        sig1, _ = signer.sign(b"data one")
        sig2, _ = signer.sign(b"data two")

        assert sig1 != sig2


class TestAuditSigner:
    """Tests for AuditSigner wrapper."""

    def test_default_uses_stub(self):
        """Test default initialization uses stub backend."""
        signer = AuditSigner(force_stub=True)

        assert signer.is_stub is True
        assert signer.backend_type == "StubSigner"

    def test_sign_returns_base64(self):
        """Test sign returns base64-encoded signature."""
        signer = AuditSigner(force_stub=True)
        data = b"test data"

        signature_b64, key_id = signer.sign(data)

        # Should be valid base64
        import base64

        decoded = base64.b64decode(signature_b64)
        assert len(decoded) == 32

    def test_sign_hash_convenience(self):
        """Test sign_hash convenience method."""
        signer = AuditSigner(force_stub=True)
        hash_hex = "abc123def456" * 5 + "abcd"  # 64 chars

        signature_b64, key_id = signer.sign_hash(hash_hex)

        assert signature_b64
        assert key_id

    def test_verify_hash_convenience(self):
        """Test verify_hash convenience method."""
        signer = AuditSigner(force_stub=True)
        hash_hex = "abc123def456" * 5 + "abcd"  # 64 chars

        signature_b64, key_id = signer.sign_hash(hash_hex)
        is_valid = signer.verify_hash(hash_hex, signature_b64, key_id)

        assert is_valid is True

    def test_stats_tracking(self):
        """Test statistics are tracked."""
        signer = AuditSigner(force_stub=True)

        # Sign a few times
        for _ in range(3):
            signer.sign(b"test")

        # Verify once
        sig, key_id = signer.sign(b"verify me")
        signer.verify(b"verify me", sig, key_id)

        stats = signer.get_stats()

        assert stats["sign_count"] == 4
        assert stats["verify_count"] == 1
        assert stats["is_stub"] is True
        assert "uptime_seconds" in stats

    def test_verify_invalid_base64(self):
        """Test verify handles invalid base64 gracefully."""
        signer = AuditSigner(force_stub=True)

        is_valid = signer.verify(b"data", "not-valid-base64!!!", "key-id")

        assert is_valid is False
