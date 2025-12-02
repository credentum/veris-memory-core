"""
Secrets Management Tests
Sprint 10 Phase 3 - Issue 007: SEC-107
Tests the enterprise secrets management system
"""

import pytest
import os
import tempfile
import shutil
import json
import hashlib
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.security.secrets_manager import (
    SecretsManager,
    SecretType,
    SecretStatus,
    SecretDetector,
    SecretEncryption,
    SecretStorage,
    SecretMetadata,
    SecretValue,
    SecretAccessLog
)


class TestSecretDetector:
    """Test ID: SEC-107-A - Secret Detection"""
    
    def test_api_key_detection(self):
        """Test detection of API keys in content"""
        detector = SecretDetector()
        
        content = '''
        export API_KEY="sk-1234567890abcdef1234567890abcdef12345678"
        const apikey = "AIzaSyC1234567890abcdef1234567890abcdef123"
        api_key: "abc123def456ghi789jkl012mno345pqr678stu901"
        '''
        
        secrets = detector.detect_secrets(content)
        
        assert len(secrets) >= 2
        api_key_secrets = [s for s in secrets if s[0] == SecretType.API_KEY]
        assert len(api_key_secrets) >= 2
    
    def test_jwt_secret_detection(self):
        """Test detection of JWT secrets"""
        detector = SecretDetector()
        
        content = '''
        JWT_SECRET="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        jwt_secret: "dGhpcyBpcyBhIHNlY3JldCBqd3Qgc2lnbmluZyBrZXk"
        '''
        
        secrets = detector.detect_secrets(content)
        
        jwt_secrets = [s for s in secrets if s[0] == SecretType.JWT_SECRET]
        assert len(jwt_secrets) >= 1
    
    def test_database_password_detection(self):
        """Test detection of database passwords"""
        detector = SecretDetector()
        
        content = '''
        password="MyVerySecretPassword123!"
        db_pass: "SuperSecretDBPass456@"
        '''
        
        secrets = detector.detect_secrets(content)
        
        db_secrets = [s for s in secrets if s[0] == SecretType.DATABASE_PASSWORD]
        assert len(db_secrets) >= 1
    
    def test_private_key_detection(self):
        """Test detection of private keys"""
        detector = SecretDetector()
        
        content = '''
        -----BEGIN RSA PRIVATE KEY-----
        MIIEpAIBAAKCAQEA1234567890abcdef...
        -----END RSA PRIVATE KEY-----
        '''
        
        secrets = detector.detect_secrets(content)
        
        key_secrets = [s for s in secrets if s[0] == SecretType.PRIVATE_KEY]
        assert len(key_secrets) >= 1
    
    def test_entropy_based_detection(self):
        """Test entropy-based secret detection"""
        detector = SecretDetector()
        
        # High entropy values
        high_entropy_values = [
            ("Kj8Nm2Pq9Rs4Tv6Ux1Yz3Ab5Cd7Ef0Gh2Ik4Jl6", "api_key"),
            ("dGhpcyBpcyBhIHNlY3JldCB0b2tlbiBzdHJpbmc", "token"), 
            ("9f8e7d6c5b4a3928f7e6d5c4b3a29187f6e5d4c3", "secret"),
        ]
        
        # Low entropy values (should not be detected)
        low_entropy_values = [
            ("password123", "password"),
            ("12345678", "key"),
            ("aaaaaaaaaa", "secret"),
        ]
        
        for value, context in high_entropy_values:
            is_secret, secret_type = detector.is_likely_secret(value, context)
            assert is_secret, f"Should detect {value} as secret"
            assert secret_type is not None
        
        for value, context in low_entropy_values:
            is_secret, _ = detector.is_likely_secret(value, context)
            assert not is_secret, f"Should not detect {value} as secret"
    
    def test_context_based_detection(self):
        """Test context-aware secret detection"""
        detector = SecretDetector()
        
        # Same value, different contexts
        secret_value = "Kj8Nm2Pq9Rs4Tv6Ux1Yz3Ab5Cd7Ef0Gh2"
        
        contexts = [
            ("api_key", SecretType.API_KEY),
            ("jwt_secret", SecretType.JWT_SECRET), 
            ("password", SecretType.DATABASE_PASSWORD),
            ("access_token", SecretType.ACCESS_TOKEN),
        ]
        
        for context, expected_type in contexts:
            is_secret, secret_type = detector.is_likely_secret(secret_value, context)
            assert is_secret
            assert secret_type == expected_type
    
    def test_false_positive_avoidance(self):
        """Test that common non-secrets are not detected"""
        detector = SecretDetector()
        
        non_secrets = [
            "user123",
            "localhost",
            "example.com",
            "test_value",
            "configuration_key",
            "development_mode",
        ]
        
        for value in non_secrets:
            is_secret, _ = detector.is_likely_secret(value)
            assert not is_secret, f"Should not detect {value} as secret"


class TestSecretEncryption:
    """Test ID: SEC-107-B - Secret Encryption"""
    
    def test_basic_encryption_decryption(self):
        """Test basic encrypt/decrypt functionality"""
        encryption = SecretEncryption()
        
        original = "this_is_a_secret_value"
        encrypted = encryption.encrypt(original)
        decrypted = encryption.decrypt(encrypted)
        
        assert decrypted == original
        assert encrypted != original.encode()  # Should be encrypted
    
    def test_password_based_encryption(self):
        """Test password-based key derivation"""
        password = "secure_master_password_123"
        
        encryption1 = SecretEncryption.from_password(password)
        encryption2 = SecretEncryption.from_password(password)
        
        # Should generate same key from same password
        original = "secret_data"
        encrypted1 = encryption1.encrypt(original)
        decrypted2 = encryption2.decrypt(encrypted1)
        
        assert decrypted2 == original
    
    def test_key_rotation(self):
        """Test encryption key rotation"""
        encryption = SecretEncryption()
        
        original = "secret_before_rotation"
        encrypted_old = encryption.encrypt(original)
        
        # Rotate key
        old_key = encryption.rotate_key()
        
        # Old encrypted data should not decrypt with new key
        with pytest.raises(Exception):
            encryption.decrypt(encrypted_old)
        
        # New data should encrypt/decrypt with new key
        encrypted_new = encryption.encrypt(original)
        decrypted_new = encryption.decrypt(encrypted_new)
        
        assert decrypted_new == original
        assert old_key != encryption.master_key
    
    def test_encryption_determinism(self):
        """Test that encryption is non-deterministic"""
        encryption = SecretEncryption()
        
        original = "same_secret_value"
        encrypted1 = encryption.encrypt(original)
        encrypted2 = encryption.encrypt(original)
        
        # Should produce different ciphertext each time (due to nonce)
        assert encrypted1 != encrypted2
        
        # But both should decrypt to same value
        assert encryption.decrypt(encrypted1) == original
        assert encryption.decrypt(encrypted2) == original


class TestSecretStorage:
    """Test ID: SEC-107-C - Secret Storage"""
    
    def setUp(self):
        """Set up test storage directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.encryption = SecretEncryption()
        self.storage = SecretStorage(self.temp_dir, self.encryption)
    
    def tearDown(self):
        """Clean up test storage"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_store_retrieve_secret(self):
        """Test storing and retrieving secrets"""
        self.setUp()
        
        try:
            # Create test secret
            metadata = SecretMetadata(
                secret_id="test_secret_1",
                secret_type=SecretType.API_KEY,
                status=SecretStatus.ACTIVE,
                created_at=datetime.utcnow()
            )
            
            encrypted_value = self.encryption.encrypt("secret_api_key_value")
            checksum = hashlib.sha256(encrypted_value).hexdigest()  # Proper checksum
            
            secret = SecretValue(
                encrypted_value=encrypted_value,
                metadata=metadata,
                checksum=checksum
            )
            
            # Store secret
            success = self.storage.store_secret(secret)
            assert success
            
            # Retrieve secret
            retrieved = self.storage.retrieve_secret("test_secret_1")
            assert retrieved is not None
            assert retrieved.metadata.secret_id == "test_secret_1"
            assert retrieved.metadata.secret_type == SecretType.API_KEY
            
            # Decrypt and verify
            decrypted = self.encryption.decrypt(retrieved.encrypted_value)
            assert decrypted == "secret_api_key_value"
            
        finally:
            self.tearDown()
    
    def test_secret_not_found(self):
        """Test retrieving non-existent secret"""
        self.setUp()
        
        try:
            retrieved = self.storage.retrieve_secret("non_existent_secret")
            assert retrieved is None
            
        finally:
            self.tearDown()
    
    def test_delete_secret(self):
        """Test secret deletion"""
        self.setUp()
        
        try:
            # Create and store secret
            metadata = SecretMetadata(
                secret_id="delete_test",
                secret_type=SecretType.DATABASE_PASSWORD,
                status=SecretStatus.ACTIVE,
                created_at=datetime.utcnow()
            )
            
            encrypted_value = self.encryption.encrypt("password_to_delete")
            secret = SecretValue(
                encrypted_value=encrypted_value,
                metadata=metadata,
                checksum=hashlib.sha256(encrypted_value).hexdigest()
            )
            
            self.storage.store_secret(secret)
            
            # Verify it exists
            assert self.storage.retrieve_secret("delete_test") is not None
            
            # Delete it
            success = self.storage.delete_secret("delete_test")
            assert success
            
            # Verify it's gone
            assert self.storage.retrieve_secret("delete_test") is None
            
        finally:
            self.tearDown()
    
    def test_list_secrets(self):
        """Test listing all secrets"""
        self.setUp()
        
        try:
            # Create multiple secrets
            for i in range(3):
                metadata = SecretMetadata(
                    secret_id=f"list_test_{i}",
                    secret_type=SecretType.API_KEY,
                    status=SecretStatus.ACTIVE,
                    created_at=datetime.utcnow()
                )
                
                encrypted_value = self.encryption.encrypt(f"secret_value_{i}")
                secret = SecretValue(
                    encrypted_value=encrypted_value,
                    metadata=metadata,
                    checksum=hashlib.sha256(encrypted_value).hexdigest()
                )
                
                self.storage.store_secret(secret)
            
            # List secrets
            secrets = self.storage.list_secrets()
            assert len(secrets) == 3
            
            secret_ids = [s.secret_id for s in secrets]
            assert "list_test_0" in secret_ids
            assert "list_test_1" in secret_ids
            assert "list_test_2" in secret_ids
            
        finally:
            self.tearDown()
    
    def test_storage_permissions(self):
        """Test that storage directory has secure permissions"""
        self.setUp()
        
        try:
            # Check directory permissions
            stat = os.stat(self.temp_dir)
            permissions = oct(stat.st_mode)[-3:]
            
            # Should be readable/writable only by owner
            assert permissions == "700"
            
        finally:
            self.tearDown()


class TestSecretsManager:
    """Test ID: SEC-107-D - Complete Secrets Management"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = SecretsManager(storage_path=self.temp_dir, master_password="test_password")
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_create_and_retrieve_secret(self):
        """Test creating and retrieving secrets"""
        self.setUp()
        
        try:
            # Create secret
            success = self.manager.create_secret(
                secret_id="test_api_key",
                secret_value="sk-1234567890abcdef1234567890abcdef12345678",
                secret_type=SecretType.API_KEY,
                rotation_frequency_days=30
            )
            assert success
            
            # Retrieve secret
            value = self.manager.get_secret("test_api_key", "test_user")
            assert value == "sk-1234567890abcdef1234567890abcdef12345678"
            
            # Check access was logged
            logs = self.manager.get_access_logs("test_api_key")
            assert len(logs) >= 2  # create + read
            assert logs[0].access_type == "read"
            assert logs[0].accessed_by == "test_user"
            
        finally:
            self.tearDown()
    
    def test_secret_expiration(self):
        """Test secret expiration handling"""
        import time
        self.setUp()
        
        try:
            # Create secret that expires very soon (use negative days to ensure it's expired)
            success = self.manager.create_secret(
                secret_id="expiring_secret",
                secret_value="expires_soon",
                secret_type=SecretType.DATABASE_PASSWORD,
                expires_in_days=-1  # Expired 1 day ago
            )
            assert success
            
            # Should not be able to retrieve expired secret
            value = self.manager.get_secret("expiring_secret", "test_user")
            assert value is None
            
            # Check error was logged
            logs = self.manager.get_access_logs("expiring_secret")
            failed_logs = [log for log in logs if not log.success]
            assert len(failed_logs) > 0
            assert "expired" in failed_logs[0].error_message.lower()
            
        finally:
            self.tearDown()
    
    def test_secret_rotation(self):
        """Test secret rotation"""
        self.setUp()
        
        try:
            # Create secret
            self.manager.create_secret(
                secret_id="rotation_test",
                secret_value="original_value",
                secret_type=SecretType.JWT_SECRET
            )
            
            original_value = self.manager.get_secret("rotation_test", "system")
            assert original_value == "original_value"
            
            # Rotate with new value
            success = self.manager.rotate_secret(
                secret_id="rotation_test",
                new_value="rotated_value",
                rotated_by="admin"
            )
            assert success
            
            # Get rotated value
            new_value = self.manager.get_secret("rotation_test", "system")
            assert new_value == "rotated_value"
            assert new_value != original_value
            
            # Check rotation was logged
            logs = self.manager.get_access_logs("rotation_test")
            rotation_logs = [log for log in logs if log.access_type == "rotate"]
            assert len(rotation_logs) >= 1
            assert rotation_logs[0].accessed_by == "admin"
            
        finally:
            self.tearDown()
    
    def test_automatic_rotation_detection(self):
        """Test detection of secrets needing rotation"""
        self.setUp()
        
        try:
            # Create secret with short rotation frequency
            self.manager.create_secret(
                secret_id="auto_rotate_test",
                secret_value="needs_rotation",
                secret_type=SecretType.API_KEY,
                rotation_frequency_days=1  # 1 day frequency
            )
            
            # Initially should not need rotation
            needing_rotation = self.manager.get_secrets_requiring_rotation()
            assert len(needing_rotation) == 0
            
            # Simulate time passing by backdating creation
            secrets = self.manager.storage.list_secrets()
            for secret in secrets:
                if secret.secret_id == "auto_rotate_test":
                    # Backdate creation to 2 days ago
                    secret.created_at = datetime.utcnow() - timedelta(days=2)
                    
                    # Store updated metadata
                    secret_value = self.manager.storage.retrieve_secret(secret.secret_id)
                    secret_value.metadata = secret
                    self.manager.storage.store_secret(secret_value)
            
            # Now should need rotation
            needing_rotation = self.manager.get_secrets_requiring_rotation()
            assert len(needing_rotation) == 1
            assert needing_rotation[0].secret_id == "auto_rotate_test"
            
        finally:
            self.tearDown()
    
    def test_secret_revocation(self):
        """Test secret revocation"""
        self.setUp()
        
        try:
            # Create secret
            self.manager.create_secret(
                secret_id="revoke_test",
                secret_value="to_be_revoked",
                secret_type=SecretType.ACCESS_TOKEN
            )
            
            # Verify can retrieve
            value = self.manager.get_secret("revoke_test", "user")
            assert value == "to_be_revoked"
            
            # Revoke secret
            success = self.manager.revoke_secret("revoke_test", "admin")
            assert success
            
            # Should not be able to retrieve revoked secret
            value = self.manager.get_secret("revoke_test", "user")
            assert value is None
            
            # Check revocation was logged
            logs = self.manager.get_access_logs("revoke_test")
            revoke_logs = [log for log in logs if log.access_type == "revoke"]
            assert len(revoke_logs) >= 1
            
        finally:
            self.tearDown()
    
    def test_secret_listing_with_filters(self):
        """Test listing secrets with status filters"""
        self.setUp()
        
        try:
            # Create multiple secrets with different statuses
            self.manager.create_secret("active_1", "value1", SecretType.API_KEY)
            self.manager.create_secret("active_2", "value2", SecretType.DATABASE_PASSWORD)
            self.manager.create_secret("to_revoke", "value3", SecretType.JWT_SECRET)
            
            # Revoke one secret
            self.manager.revoke_secret("to_revoke", "admin")
            
            # List all secrets (excluding revoked)
            active_secrets = self.manager.list_secrets(include_revoked=False)
            assert len(active_secrets) == 2
            
            secret_ids = [s.secret_id for s in active_secrets]
            assert "active_1" in secret_ids
            assert "active_2" in secret_ids
            assert "to_revoke" not in secret_ids
            
            # List including revoked
            all_secrets = self.manager.list_secrets(include_revoked=True)
            assert len(all_secrets) == 3
            
        finally:
            self.tearDown()
    
    def test_content_scanning(self):
        """Test scanning content for embedded secrets"""
        self.setUp()
        
        try:
            content = '''
            # Configuration file
            API_KEY=sk-1234567890abcdef1234567890abcdef12345678
            DATABASE_URL=postgresql://user:SecretPass123@localhost:5432/db
            JWT_SECRET=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9
            '''
            
            detected_secrets = self.manager.scan_content_for_secrets(content)
            
            assert len(detected_secrets) >= 2  # Should find API key and JWT secret
            
            secret_types = [s[0] for s in detected_secrets]
            assert SecretType.API_KEY in secret_types
            
        finally:
            self.tearDown()
    
    def test_access_tracking(self):
        """Test comprehensive access tracking"""
        self.setUp()
        
        try:
            # Create secret
            self.manager.create_secret("tracking_test", "tracked_value", SecretType.API_KEY)
            
            # Access multiple times by different users
            for i in range(5):
                self.manager.get_secret("tracking_test", f"user_{i}")
            
            # Get access logs
            logs = self.manager.get_access_logs("tracking_test")
            
            # Should have create log + 5 read logs
            assert len(logs) >= 6
            
            read_logs = [log for log in logs if log.access_type == "read"]
            assert len(read_logs) == 5
            
            # Check different users are recorded
            users = [log.accessed_by for log in read_logs]
            assert len(set(users)) == 5  # 5 different users
            
        finally:
            self.tearDown()
    
    def test_duplicate_secret_prevention(self):
        """Test prevention of duplicate secret IDs"""
        self.setUp()
        
        try:
            # Create first secret
            success1 = self.manager.create_secret(
                "duplicate_test", "value1", SecretType.API_KEY
            )
            assert success1
            
            # Try to create duplicate
            success2 = self.manager.create_secret(
                "duplicate_test", "value2", SecretType.DATABASE_PASSWORD
            )
            assert not success2  # Should fail
            
            # Original secret should be unchanged
            value = self.manager.get_secret("duplicate_test", "user")
            assert value == "value1"
            
        finally:
            self.tearDown()
    
    def test_concurrent_access(self):
        """Test thread-safe concurrent access"""
        import threading
        
        self.setUp()
        
        try:
            # Create secret
            self.manager.create_secret("concurrent_test", "concurrent_value", SecretType.API_KEY)
            
            results = []
            errors = []
            
            def access_secret(thread_id):
                try:
                    for _ in range(10):
                        value = self.manager.get_secret("concurrent_test", f"thread_{thread_id}")
                        if value:
                            results.append(value)
                except Exception as e:
                    errors.append(e)
            
            # Start multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=access_secret, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Should have no errors and consistent results
            assert len(errors) == 0
            assert len(results) == 50  # 5 threads * 10 accesses each
            assert all(r == "concurrent_value" for r in results)
            
        finally:
            self.tearDown()


class TestIntegration:
    """Test ID: SEC-107-E - Integration Tests"""
    
    def test_end_to_end_secret_lifecycle(self):
        """Test complete secret lifecycle"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            manager = SecretsManager(storage_path=temp_dir, master_password="integration_test")
            
            # 1. Create secret
            success = manager.create_secret(
                secret_id="lifecycle_test",
                secret_value="initial_secret_value",
                secret_type=SecretType.DATABASE_PASSWORD,
                expires_in_days=30,
                rotation_frequency_days=7,
                tags={"environment": "test", "service": "database"}
            )
            assert success
            
            # 2. Access secret multiple times
            for i in range(3):
                value = manager.get_secret("lifecycle_test", f"service_{i}")
                assert value == "initial_secret_value"
            
            # 3. Rotate secret
            success = manager.rotate_secret("lifecycle_test", "rotated_secret_value", "admin")
            assert success
            
            # 4. Verify new value
            new_value = manager.get_secret("lifecycle_test", "validator")
            assert new_value == "rotated_secret_value"
            
            # 5. Check metadata
            secrets = manager.list_secrets()
            lifecycle_secret = next(s for s in secrets if s.secret_id == "lifecycle_test")
            
            assert lifecycle_secret.secret_type == SecretType.DATABASE_PASSWORD
            assert lifecycle_secret.status == SecretStatus.ACTIVE
            assert lifecycle_secret.access_count >= 4  # 3 initial + 1 after rotation
            assert lifecycle_secret.last_rotated is not None
            assert lifecycle_secret.tags["environment"] == "test"
            
            # 6. Revoke secret
            success = manager.revoke_secret("lifecycle_test", "admin")
            assert success
            
            # 7. Verify cannot access revoked secret
            value = manager.get_secret("lifecycle_test", "final_check")
            assert value is None
            
            # 8. Verify comprehensive audit trail
            logs = manager.get_access_logs("lifecycle_test")
            
            log_types = [log.access_type for log in logs]
            assert "create" in log_types
            assert "read" in log_types
            assert "rotate" in log_types
            assert "revoke" in log_types
            
            # Should have detailed access tracking
            assert len(logs) >= 7  # create + 3 reads + rotate + 1 read + revoke
            
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def test_secret_manager_persistence(self):
        """Test that secrets persist across manager instances"""
        temp_dir = tempfile.mkdtemp()
        master_password = "persistence_test_password"
        
        try:
            # Create manager and secret
            manager1 = SecretsManager(storage_path=temp_dir, master_password=master_password)
            manager1.create_secret("persistent_test", "persistent_value", SecretType.API_KEY)
            
            # Create new manager instance
            manager2 = SecretsManager(storage_path=temp_dir, master_password=master_password)
            
            # Should be able to access secret created by first manager
            value = manager2.get_secret("persistent_test", "new_instance")
            assert value == "persistent_value"
            
            # Modify with second manager
            manager2.rotate_secret("persistent_test", "updated_by_second", "manager2")
            
            # Create third manager instance
            manager3 = SecretsManager(storage_path=temp_dir, master_password=master_password)
            
            # Should see the updated value
            value = manager3.get_secret("persistent_test", "third_instance")
            assert value == "updated_by_second"
            
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def test_wrong_master_password(self):
        """Test behavior with wrong master password"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create secret with one password
            manager1 = SecretsManager(storage_path=temp_dir, master_password="correct_password")
            manager1.create_secret("password_test", "secret_data", SecretType.JWT_SECRET)
            
            # Try to access with wrong password
            manager2 = SecretsManager(storage_path=temp_dir, master_password="wrong_password")
            
            # Should not be able to decrypt existing secret
            value = manager2.get_secret("password_test", "wrong_pass_user")
            assert value is None
            
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run secrets management tests
    pytest.main([__file__, "-v", "-s"])