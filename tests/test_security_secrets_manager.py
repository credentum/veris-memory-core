#!/usr/bin/env python3
"""
Comprehensive tests for Security Secrets Manager - Phase 7 Coverage

This test module provides comprehensive coverage for the secrets management system
including encryption, storage, rotation, detection, and access control.
"""
import pytest
import tempfile
import os
import json
import base64
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, MagicMock, mock_open
from typing import Dict, Any, List

# Import secrets manager components
try:
    from src.security.secrets_manager import (
        SecretType, SecretStatus, SecretMetadata, SecretValue, SecretAccessLog,
        SecretDetector, SecretEncryption, SecretStorage, SecretsManager
    )
    from cryptography.fernet import Fernet
    SECRETS_MANAGER_AVAILABLE = True
except ImportError:
    SECRETS_MANAGER_AVAILABLE = False


@pytest.mark.skipif(not SECRETS_MANAGER_AVAILABLE, reason="Secrets manager not available")
class TestSecretsManagerEnums:
    """Test secrets manager enums and constants"""
    
    def test_secret_type_enum(self):
        """Test SecretType enum values"""
        assert SecretType.DATABASE_PASSWORD.value == "database_password"
        assert SecretType.API_KEY.value == "api_key"
        assert SecretType.JWT_SECRET.value == "jwt_secret"
        assert SecretType.ENCRYPTION_KEY.value == "encryption_key"
        assert SecretType.CERTIFICATE.value == "certificate"
        assert SecretType.PRIVATE_KEY.value == "private_key"
        assert SecretType.ACCESS_TOKEN.value == "access_token"
        assert SecretType.WEBHOOK_SECRET.value == "webhook_secret"
        assert SecretType.SERVICE_ACCOUNT.value == "service_account"
    
    def test_secret_status_enum(self):
        """Test SecretStatus enum values"""
        assert SecretStatus.ACTIVE.value == "active"
        assert SecretStatus.EXPIRED.value == "expired"
        assert SecretStatus.REVOKED.value == "revoked"
        assert SecretStatus.PENDING_ROTATION.value == "pending_rotation"
        assert SecretStatus.COMPROMISED.value == "compromised"


@pytest.mark.skipif(not SECRETS_MANAGER_AVAILABLE, reason="Secrets manager not available")
class TestSecretsManagerDataModels:
    """Test secrets manager data models"""
    
    def test_secret_metadata_creation(self):
        """Test SecretMetadata dataclass creation"""
        now = datetime.utcnow()
        expires_at = now + timedelta(days=90)
        
        metadata = SecretMetadata(
            secret_id="secret-123",
            secret_type=SecretType.API_KEY,
            status=SecretStatus.ACTIVE,
            created_at=now,
            expires_at=expires_at,
            last_rotated=now,
            rotation_frequency_days=30,
            tags={"environment": "production", "service": "api"},
            access_count=5,
            last_accessed=now
        )
        
        assert metadata.secret_id == "secret-123"
        assert metadata.secret_type == SecretType.API_KEY
        assert metadata.status == SecretStatus.ACTIVE
        assert metadata.created_at == now
        assert metadata.expires_at == expires_at
        assert metadata.last_rotated == now
        assert metadata.rotation_frequency_days == 30
        assert metadata.tags == {"environment": "production", "service": "api"}
        assert metadata.access_count == 5
        assert metadata.last_accessed == now
    
    def test_secret_metadata_defaults(self):
        """Test SecretMetadata default values"""
        now = datetime.utcnow()
        metadata = SecretMetadata(
            secret_id="test-secret",
            secret_type=SecretType.DATABASE_PASSWORD,
            status=SecretStatus.ACTIVE,
            created_at=now
        )
        
        assert metadata.expires_at is None
        assert metadata.last_rotated is None
        assert metadata.rotation_frequency_days == 90
        assert metadata.tags == {}
        assert metadata.access_count == 0
        assert metadata.last_accessed is None
    
    def test_secret_value_creation(self):
        """Test SecretValue dataclass creation"""
        now = datetime.utcnow()
        metadata = SecretMetadata(
            secret_id="val-test",
            secret_type=SecretType.JWT_SECRET,
            status=SecretStatus.ACTIVE,
            created_at=now
        )
        
        encrypted_value = b"encrypted_secret_data"
        checksum = "abc123def456"
        
        secret_value = SecretValue(
            encrypted_value=encrypted_value,
            metadata=metadata,
            checksum=checksum
        )
        
        assert secret_value.encrypted_value == encrypted_value
        assert secret_value.metadata == metadata
        assert secret_value.checksum == checksum
    
    def test_secret_value_integrity_verification(self):
        """Test SecretValue integrity verification"""
        now = datetime.utcnow()
        metadata = SecretMetadata(
            secret_id="integrity-test",
            secret_type=SecretType.ENCRYPTION_KEY,
            status=SecretStatus.ACTIVE,
            created_at=now
        )
        
        encrypted_value = b"test_encrypted_data"
        
        # Calculate correct checksum
        import hashlib
        correct_checksum = hashlib.sha256(encrypted_value).hexdigest()
        
        # Test with correct checksum
        secret_value = SecretValue(
            encrypted_value=encrypted_value,
            metadata=metadata,
            checksum=correct_checksum
        )
        
        assert secret_value.verify_integrity() is True
        
        # Test with incorrect checksum
        secret_value_bad = SecretValue(
            encrypted_value=encrypted_value,
            metadata=metadata,
            checksum="wrong_checksum"
        )
        
        assert secret_value_bad.verify_integrity() is False
    
    def test_secret_access_log_creation(self):
        """Test SecretAccessLog dataclass creation"""
        now = datetime.utcnow()
        
        log_entry = SecretAccessLog(
            secret_id="log-test-secret",
            accessed_by="user@example.com",
            access_type="read",
            timestamp=now,
            client_ip="192.168.1.100",
            user_agent="SecretsClient/1.0",
            success=True,
            error_message=None
        )
        
        assert log_entry.secret_id == "log-test-secret"
        assert log_entry.accessed_by == "user@example.com"
        assert log_entry.access_type == "read"
        assert log_entry.timestamp == now
        assert log_entry.client_ip == "192.168.1.100"
        assert log_entry.user_agent == "SecretsClient/1.0"
        assert log_entry.success is True
        assert log_entry.error_message is None
    
    def test_secret_access_log_defaults(self):
        """Test SecretAccessLog default values"""
        now = datetime.utcnow()
        
        log_entry = SecretAccessLog(
            secret_id="default-test",
            accessed_by="test_user",
            access_type="write",
            timestamp=now
        )
        
        assert log_entry.client_ip is None
        assert log_entry.user_agent is None
        assert log_entry.success is True
        assert log_entry.error_message is None


@pytest.mark.skipif(not SECRETS_MANAGER_AVAILABLE, reason="Secrets manager not available")
class TestSecretDetector:
    """Test secret detector functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.detector = SecretDetector()
    
    def test_detect_api_keys(self):
        """Test API key detection"""
        test_content = '''
        # Configuration file
        API_KEY = "sk-1234567890abcdefghijklmnopqrstuvwxyz"
        apikey: "ak_live_1234567890abcdefghijklmnopqr"
        api-key = "key_1234567890abcdefghijklmnopqrstuvwxyz"
        '''
        
        detections = self.detector.detect_secrets(test_content)
        
        api_key_detections = [d for d in detections if d["type"] == SecretType.API_KEY]
        assert len(api_key_detections) > 0
        
        for detection in api_key_detections:
            assert "confidence" in detection
            assert detection["confidence"] > 0.5
            assert "value" in detection
            assert "line" in detection
    
    def test_detect_database_passwords(self):
        """Test database password detection"""
        test_content = '''
        DATABASE_URL = "postgresql://user:password123@localhost/db"
        DB_PASSWORD = "super_secret_db_pass"
        password: "MyStrongPassword123!"
        '''
        
        detections = self.detector.detect_secrets(test_content)
        
        password_detections = [d for d in detections 
                             if d["type"] == SecretType.DATABASE_PASSWORD]
        assert len(password_detections) > 0
        
        for detection in password_detections:
            assert "confidence" in detection
            assert "value" in detection
            assert len(detection["value"]) > 5  # Should capture actual password
    
    def test_detect_jwt_secrets(self):
        """Test JWT secret detection"""
        test_content = '''
        JWT_SECRET = "your-256-bit-secret-key-here"
        jwt_secret_key: "super_secret_jwt_signing_key"
        TOKEN_SECRET = "jwt_signing_secret_123456789"
        '''
        
        detections = self.detector.detect_secrets(test_content)
        
        jwt_detections = [d for d in detections if d["type"] == SecretType.JWT_SECRET]
        assert len(jwt_detections) > 0
        
        for detection in jwt_detections:
            assert detection["confidence"] > 0.6
            assert len(detection["value"]) >= 16  # JWT secrets should be reasonably long
    
    def test_detect_encryption_keys(self):
        """Test encryption key detection"""
        test_content = '''
        ENCRYPTION_KEY = "fernet_key_abcdef1234567890abcdef1234567890"
        SECRET_KEY = "encryption_secret_key_very_long_and_secure"
        MASTER_KEY = "AES256_encryption_master_key_32_bytes"
        '''
        
        detections = self.detector.detect_secrets(test_content)
        
        encryption_detections = [d for d in detections 
                               if d["type"] == SecretType.ENCRYPTION_KEY]
        assert len(encryption_detections) > 0
    
    def test_detect_private_keys(self):
        """Test private key detection"""
        test_content = '''
        -----BEGIN PRIVATE KEY-----
        MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC1234567890
        -----END PRIVATE KEY-----
        
        -----BEGIN RSA PRIVATE KEY-----
        MIIEpAIBAAKCAQEA1234567890abcdefghijklmnopqrstuvwxyz
        -----END RSA PRIVATE KEY-----
        '''
        
        detections = self.detector.detect_secrets(test_content)
        
        private_key_detections = [d for d in detections 
                                if d["type"] == SecretType.PRIVATE_KEY]
        assert len(private_key_detections) > 0
        
        for detection in private_key_detections:
            assert "BEGIN" in detection["value"] and "END" in detection["value"]
            assert detection["confidence"] > 0.9  # Private key patterns are very specific
    
    def test_detect_certificates(self):
        """Test certificate detection"""
        test_content = '''
        -----BEGIN CERTIFICATE-----
        MIIDXTCCAkWgAwIBAgIJAKoK/YJ9i6J7MA0GCSqGSIb3DQEBCwUAMEUxCzAJBgNV
        BAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX
        -----END CERTIFICATE-----
        '''
        
        detections = self.detector.detect_secrets(test_content)
        
        cert_detections = [d for d in detections if d["type"] == SecretType.CERTIFICATE]
        assert len(cert_detections) > 0
        
        for detection in cert_detections:
            assert "CERTIFICATE" in detection["value"]
            assert detection["confidence"] > 0.9
    
    def test_detect_access_tokens(self):
        """Test access token detection"""
        test_content = '''
        GITHUB_TOKEN = "ghp_1234567890abcdefghijklmnopqrstuvwxyz12"
        ACCESS_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        OAUTH_TOKEN = "oauth_token_1234567890abcdefghijklmnopqr"
        '''
        
        detections = self.detector.detect_secrets(test_content)
        
        token_detections = [d for d in detections if d["type"] == SecretType.ACCESS_TOKEN]
        assert len(token_detections) > 0
    
    def test_detect_webhook_secrets(self):
        """Test webhook secret detection"""
        test_content = '''
        WEBHOOK_SECRET = "whsec_1234567890abcdefghijklmnopqrstuvwxyz"
        GITHUB_WEBHOOK_SECRET = "webhook_secret_for_github_integration"
        SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
        '''
        
        detections = self.detector.detect_secrets(test_content)
        
        webhook_detections = [d for d in detections 
                            if d["type"] == SecretType.WEBHOOK_SECRET]
        assert len(webhook_detections) > 0
    
    def test_no_false_positives_on_clean_content(self):
        """Test that clean content doesn't trigger false positives"""
        clean_content = '''
        # Configuration file
        DEBUG = True
        PORT = 8080
        HOST = "localhost"
        LOG_LEVEL = "INFO"
        MAX_CONNECTIONS = 100
        TIMEOUT = 30
        '''
        
        detections = self.detector.detect_secrets(clean_content)
        
        # Clean content should have no or very few detections
        assert len(detections) <= 1  # Allow for some edge cases
    
    def test_entropy_based_detection(self):
        """Test entropy-based secret detection"""
        high_entropy_content = '''
        SECRET_VALUE = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYzEXAMPLEKEY"
        RANDOM_KEY = "fb4b2b2c0a3a6b8e1f0d9c8a7b5e3d2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e"
        BASE64_SECRET = "VGhpcyBpcyBhIGxvbmcgYW5kIHJhbmRvbSBzdHJpbmcgd2l0aCBoaWdoIGVudHJvcHk="
        '''
        
        detections = self.detector.detect_secrets(high_entropy_content)
        
        # High entropy strings should be detected
        assert len(detections) > 0
        
        # Check that at least some detections have high confidence
        high_confidence_detections = [d for d in detections if d["confidence"] > 0.7]
        assert len(high_confidence_detections) > 0


@pytest.mark.skipif(not SECRETS_MANAGER_AVAILABLE, reason="Secrets manager not available")
class TestSecretEncryption:
    """Test secret encryption functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.encryption = SecretEncryption()
    
    def test_key_generation(self):
        """Test encryption key generation"""
        key = self.encryption.generate_key()
        
        assert isinstance(key, bytes)
        assert len(key) > 0
        
        # Should be able to create Fernet with generated key
        fernet = Fernet(key)
        assert fernet is not None
    
    def test_key_derivation_from_password(self):
        """Test key derivation from password"""
        password = "test_password_123"
        salt = b"test_salt_16byte"
        
        key = self.encryption.derive_key_from_password(password, salt)
        
        assert isinstance(key, bytes)
        assert len(key) > 0
        
        # Same password and salt should generate same key
        key2 = self.encryption.derive_key_from_password(password, salt)
        assert key == key2
        
        # Different salt should generate different key
        key3 = self.encryption.derive_key_from_password(password, b"different_salt16")
        assert key != key3
    
    def test_encrypt_decrypt_cycle(self):
        """Test encryption and decryption cycle"""
        plaintext = "super_secret_password_123"
        key = self.encryption.generate_key()
        
        # Encrypt
        encrypted = self.encryption.encrypt(plaintext, key)
        assert isinstance(encrypted, bytes)
        assert encrypted != plaintext.encode()
        
        # Decrypt
        decrypted = self.encryption.decrypt(encrypted, key)
        assert decrypted == plaintext
    
    def test_encrypt_decrypt_with_different_keys(self):
        """Test that decryption fails with wrong key"""
        plaintext = "secret_data"
        key1 = self.encryption.generate_key()
        key2 = self.encryption.generate_key()
        
        encrypted = self.encryption.encrypt(plaintext, key1)
        
        # Should fail to decrypt with wrong key
        with pytest.raises(Exception):  # Fernet raises InvalidToken
            self.encryption.decrypt(encrypted, key2)
    
    def test_encrypt_empty_string(self):
        """Test encrypting empty string"""
        plaintext = ""
        key = self.encryption.generate_key()
        
        encrypted = self.encryption.encrypt(plaintext, key)
        decrypted = self.encryption.decrypt(encrypted, key)
        
        assert decrypted == plaintext
    
    def test_encrypt_unicode_content(self):
        """Test encrypting unicode content"""
        plaintext = "🔐 Secret with unicode: αβγδε"
        key = self.encryption.generate_key()
        
        encrypted = self.encryption.encrypt(plaintext, key)
        decrypted = self.encryption.decrypt(encrypted, key)
        
        assert decrypted == plaintext
    
    def test_encrypt_large_content(self):
        """Test encrypting large content"""
        plaintext = "A" * 10000  # 10KB of data
        key = self.encryption.generate_key()
        
        encrypted = self.encryption.encrypt(plaintext, key)
        decrypted = self.encryption.decrypt(encrypted, key)
        
        assert decrypted == plaintext
        assert len(encrypted) > len(plaintext.encode())  # Encrypted should be larger
    
    def test_key_rotation(self):
        """Test key rotation functionality"""
        plaintext = "secret_to_rotate"
        old_key = self.encryption.generate_key()
        new_key = self.encryption.generate_key()
        
        # Encrypt with old key
        encrypted_old = self.encryption.encrypt(plaintext, old_key)
        
        # Rotate to new key
        encrypted_new = self.encryption.rotate_key(encrypted_old, old_key, new_key)
        
        # Should be able to decrypt with new key
        decrypted = self.encryption.decrypt(encrypted_new, new_key)
        assert decrypted == plaintext
        
        # Should not be able to decrypt new version with old key
        with pytest.raises(Exception):
            self.encryption.decrypt(encrypted_new, old_key)


@pytest.mark.skipif(not SECRETS_MANAGER_AVAILABLE, reason="Secrets manager not available")
class TestSecretStorage:
    """Test secret storage functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.storage_path = temp_dir
            self.storage = SecretStorage(self.storage_path)
    
    def test_storage_initialization(self):
        """Test storage initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SecretStorage(temp_dir)
            assert storage.storage_path == temp_dir
            assert os.path.exists(temp_dir)
    
    @patch('src.security.secrets_manager.open', new_callable=mock_open)
    @patch('src.security.secrets_manager.os.path.exists')
    def test_store_secret(self, mock_exists, mock_file):
        """Test storing a secret"""
        mock_exists.return_value = True
        
        now = datetime.utcnow()
        metadata = SecretMetadata(
            secret_id="test-store",
            secret_type=SecretType.API_KEY,
            status=SecretStatus.ACTIVE,
            created_at=now
        )
        
        encrypted_value = b"encrypted_secret_data"
        checksum = "test_checksum"
        
        secret_value = SecretValue(
            encrypted_value=encrypted_value,
            metadata=metadata,
            checksum=checksum
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SecretStorage(temp_dir)
            result = storage.store_secret(secret_value)
            
            assert result is True
    
    @patch('src.security.secrets_manager.open', new_callable=mock_open)
    @patch('src.security.secrets_manager.os.path.exists')
    def test_retrieve_secret(self, mock_exists, mock_file):
        """Test retrieving a secret"""
        mock_exists.return_value = True
        
        # Mock file content
        secret_data = {
            "encrypted_value": base64.b64encode(b"encrypted_data").decode(),
            "metadata": {
                "secret_id": "test-retrieve",
                "secret_type": "api_key",
                "status": "active",
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": None,
                "last_rotated": None,
                "rotation_frequency_days": 90,
                "tags": {},
                "access_count": 0,
                "last_accessed": None
            },
            "checksum": "test_checksum"
        }
        
        mock_file.return_value.read.return_value = json.dumps(secret_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SecretStorage(temp_dir)
            secret_value = storage.retrieve_secret("test-retrieve")
            
            assert secret_value is not None
            assert secret_value.metadata.secret_id == "test-retrieve"
    
    @patch('src.security.secrets_manager.os.path.exists')
    def test_retrieve_nonexistent_secret(self, mock_exists):
        """Test retrieving a non-existent secret"""
        mock_exists.return_value = False
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SecretStorage(temp_dir)
            secret_value = storage.retrieve_secret("nonexistent")
            
            assert secret_value is None
    
    @patch('src.security.secrets_manager.os.remove')
    @patch('src.security.secrets_manager.os.path.exists')
    def test_delete_secret(self, mock_exists, mock_remove):
        """Test deleting a secret"""
        mock_exists.return_value = True
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SecretStorage(temp_dir)
            result = storage.delete_secret("test-delete")
            
            assert result is True
            mock_remove.assert_called_once()
    
    @patch('src.security.secrets_manager.os.listdir')
    @patch('src.security.secrets_manager.os.path.exists')
    def test_list_secrets(self, mock_exists, mock_listdir):
        """Test listing all secrets"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["secret1.json", "secret2.json", "not_json.txt"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SecretStorage(temp_dir)
            secret_ids = storage.list_secrets()
            
            assert "secret1" in secret_ids
            assert "secret2" in secret_ids
            assert "not_json" not in secret_ids  # Should filter non-JSON files
    
    def test_storage_path_creation(self):
        """Test storage path creation for non-existent directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_path = os.path.join(temp_dir, "new_storage")
            storage = SecretStorage(nonexistent_path)
            
            # Should create the directory
            assert os.path.exists(nonexistent_path)


@pytest.mark.skipif(not SECRETS_MANAGER_AVAILABLE, reason="Secrets manager not available")
class TestSecretsManager:
    """Test main secrets manager orchestration"""
    
    def setup_method(self):
        """Setup test environment"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.manager = SecretsManager(storage_path=temp_dir)
    
    def test_manager_initialization(self):
        """Test secrets manager initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            assert manager is not None
            assert hasattr(manager, 'storage')
            assert hasattr(manager, 'encryption')
            assert hasattr(manager, 'detector')
    
    def test_create_secret(self):
        """Test creating a new secret"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            secret_id = manager.create_secret(
                secret_type=SecretType.API_KEY,
                value="test_api_key_12345",
                tags={"environment": "test"}
            )
            
            assert secret_id is not None
            assert isinstance(secret_id, str)
            assert len(secret_id) > 0
    
    def test_get_secret(self):
        """Test retrieving a secret"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # Create a secret first
            secret_id = manager.create_secret(
                secret_type=SecretType.DATABASE_PASSWORD,
                value="super_secret_password"
            )
            
            # Retrieve the secret
            retrieved_value = manager.get_secret(secret_id, accessed_by="test_user")
            
            assert retrieved_value == "super_secret_password"
    
    def test_get_nonexistent_secret(self):
        """Test retrieving a non-existent secret"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            retrieved_value = manager.get_secret("nonexistent", accessed_by="test_user")
            
            assert retrieved_value is None
    
    def test_update_secret(self):
        """Test updating an existing secret"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # Create a secret
            secret_id = manager.create_secret(
                secret_type=SecretType.JWT_SECRET,
                value="original_jwt_secret"
            )
            
            # Update the secret
            result = manager.update_secret(
                secret_id,
                new_value="updated_jwt_secret",
                updated_by="admin_user"
            )
            
            assert result is True
            
            # Verify the update
            retrieved_value = manager.get_secret(secret_id, accessed_by="test_user")
            assert retrieved_value == "updated_jwt_secret"
    
    def test_delete_secret(self):
        """Test deleting a secret"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # Create a secret
            secret_id = manager.create_secret(
                secret_type=SecretType.ENCRYPTION_KEY,
                value="encryption_key_to_delete"
            )
            
            # Delete the secret
            result = manager.delete_secret(secret_id, deleted_by="admin_user")
            
            assert result is True
            
            # Verify deletion
            retrieved_value = manager.get_secret(secret_id, accessed_by="test_user")
            assert retrieved_value is None
    
    def test_rotate_secret(self):
        """Test secret rotation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # Create a secret
            secret_id = manager.create_secret(
                secret_type=SecretType.ACCESS_TOKEN,
                value="original_access_token"
            )
            
            # Rotate the secret
            new_secret_id = manager.rotate_secret(
                secret_id,
                rotated_by="system_admin"
            )
            
            assert new_secret_id is not None
            assert new_secret_id != secret_id  # Should be a new ID
            
            # Old secret should be revoked
            old_metadata = manager.get_secret_metadata(secret_id)
            assert old_metadata.status == SecretStatus.REVOKED
            
            # New secret should be active
            new_metadata = manager.get_secret_metadata(new_secret_id)
            assert new_metadata.status == SecretStatus.ACTIVE
    
    def test_list_secrets_by_type(self):
        """Test listing secrets by type"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # Create secrets of different types
            api_key_id = manager.create_secret(
                secret_type=SecretType.API_KEY,
                value="api_key_value"
            )
            
            db_password_id = manager.create_secret(
                secret_type=SecretType.DATABASE_PASSWORD,
                value="db_password_value"
            )
            
            # List API keys
            api_keys = manager.list_secrets(secret_type=SecretType.API_KEY)
            assert api_key_id in api_keys
            assert db_password_id not in api_keys
            
            # List database passwords
            db_passwords = manager.list_secrets(secret_type=SecretType.DATABASE_PASSWORD)
            assert db_password_id in db_passwords
            assert api_key_id not in db_passwords
    
    def test_list_secrets_by_status(self):
        """Test listing secrets by status"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # Create and revoke a secret
            secret_id = manager.create_secret(
                secret_type=SecretType.WEBHOOK_SECRET,
                value="webhook_secret_value"
            )
            
            manager.revoke_secret(secret_id, revoked_by="admin")
            
            # List active secrets (should not include revoked)
            active_secrets = manager.list_secrets(status=SecretStatus.ACTIVE)
            assert secret_id not in active_secrets
            
            # List revoked secrets
            revoked_secrets = manager.list_secrets(status=SecretStatus.REVOKED)
            assert secret_id in revoked_secrets
    
    def test_get_secret_metadata(self):
        """Test retrieving secret metadata"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # Create a secret with tags
            secret_id = manager.create_secret(
                secret_type=SecretType.CERTIFICATE,
                value="certificate_content",
                tags={"domain": "example.com", "environment": "production"}
            )
            
            metadata = manager.get_secret_metadata(secret_id)
            
            assert metadata is not None
            assert metadata.secret_id == secret_id
            assert metadata.secret_type == SecretType.CERTIFICATE
            assert metadata.status == SecretStatus.ACTIVE
            assert metadata.tags["domain"] == "example.com"
            assert metadata.tags["environment"] == "production"
    
    def test_search_secrets_by_tags(self):
        """Test searching secrets by tags"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # Create secrets with different tags
            prod_secret = manager.create_secret(
                secret_type=SecretType.API_KEY,
                value="prod_api_key",
                tags={"environment": "production", "service": "api"}
            )
            
            dev_secret = manager.create_secret(
                secret_type=SecretType.API_KEY,
                value="dev_api_key",
                tags={"environment": "development", "service": "api"}
            )
            
            test_secret = manager.create_secret(
                secret_type=SecretType.DATABASE_PASSWORD,
                value="test_db_pass",
                tags={"environment": "test", "service": "database"}
            )
            
            # Search by environment
            prod_secrets = manager.search_secrets(tags={"environment": "production"})
            assert prod_secret in prod_secrets
            assert dev_secret not in prod_secrets
            assert test_secret not in prod_secrets
            
            # Search by service
            api_secrets = manager.search_secrets(tags={"service": "api"})
            assert prod_secret in api_secrets
            assert dev_secret in api_secrets
            assert test_secret not in api_secrets
    
    def test_audit_log_tracking(self):
        """Test audit log tracking for secret access"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # Create a secret
            secret_id = manager.create_secret(
                secret_type=SecretType.PRIVATE_KEY,
                value="private_key_content"
            )
            
            # Access the secret multiple times
            manager.get_secret(secret_id, accessed_by="user1")
            manager.get_secret(secret_id, accessed_by="user2")
            manager.update_secret(secret_id, new_value="updated_key", updated_by="admin")
            
            # Get audit logs
            logs = manager.get_audit_logs(secret_id)
            
            assert len(logs) >= 3  # At least create, read, read, update operations
            
            # Verify log entries
            access_types = [log.access_type for log in logs]
            assert "read" in access_types
            assert "write" in access_types or "update" in access_types
            
            # Verify different users
            users = [log.accessed_by for log in logs]
            assert "user1" in users
            assert "user2" in users
    
    def test_secret_expiration_check(self):
        """Test checking for expired secrets"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # Create a secret that expires soon
            secret_id = manager.create_secret(
                secret_type=SecretType.ACCESS_TOKEN,
                value="expiring_token",
                expires_in_days=1
            )
            
            # Get secrets expiring soon
            expiring_secrets = manager.get_expiring_secrets(days_ahead=7)
            
            assert secret_id in expiring_secrets
    
    def test_bulk_secret_operations(self):
        """Test bulk operations on secrets"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # Create multiple secrets
            secret_data = [
                {
                    "secret_type": SecretType.API_KEY,
                    "value": f"api_key_{i}",
                    "tags": {"batch": "test_batch", "index": str(i)}
                }
                for i in range(5)
            ]
            
            created_ids = manager.create_secrets_bulk(secret_data)
            
            assert len(created_ids) == 5
            assert all(isinstance(sid, str) for sid in created_ids)
            
            # Verify all were created
            batch_secrets = manager.search_secrets(tags={"batch": "test_batch"})
            assert len(batch_secrets) == 5
    
    def test_secret_validation(self):
        """Test secret validation during creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # Test empty secret value
            with pytest.raises(ValueError):
                manager.create_secret(
                    secret_type=SecretType.API_KEY,
                    value=""
                )
            
            # Test None secret value
            with pytest.raises(ValueError):
                manager.create_secret(
                    secret_type=SecretType.API_KEY,
                    value=None
                )
            
            # Test very long secret value (should work)
            long_value = "A" * 10000
            secret_id = manager.create_secret(
                secret_type=SecretType.ENCRYPTION_KEY,
                value=long_value
            )
            
            assert secret_id is not None
            retrieved = manager.get_secret(secret_id, accessed_by="test")
            assert retrieved == long_value


@pytest.mark.skipif(not SECRETS_MANAGER_AVAILABLE, reason="Secrets manager not available")
class TestSecretsManagerIntegrationScenarios:
    """Test integrated secrets management scenarios"""
    
    def test_complete_secret_lifecycle(self):
        """Test complete secret lifecycle from creation to deletion"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # 1. Create secret
            secret_id = manager.create_secret(
                secret_type=SecretType.DATABASE_PASSWORD,
                value="initial_password_123",
                tags={"environment": "production", "database": "main"}
            )
            
            # 2. Access secret
            retrieved_value = manager.get_secret(secret_id, accessed_by="app_service")
            assert retrieved_value == "initial_password_123"
            
            # 3. Update secret
            manager.update_secret(
                secret_id,
                new_value="updated_password_456",
                updated_by="admin"
            )
            
            # 4. Verify update
            updated_value = manager.get_secret(secret_id, accessed_by="app_service")
            assert updated_value == "updated_password_456"
            
            # 5. Rotate secret
            new_secret_id = manager.rotate_secret(secret_id, rotated_by="automated_system")
            
            # 6. Verify rotation
            assert new_secret_id != secret_id
            old_metadata = manager.get_secret_metadata(secret_id)
            assert old_metadata.status == SecretStatus.REVOKED
            
            new_metadata = manager.get_secret_metadata(new_secret_id)
            assert new_metadata.status == SecretStatus.ACTIVE
            
            # 7. Delete old secret
            manager.delete_secret(secret_id, deleted_by="cleanup_service")
            
            # 8. Verify deletion
            deleted_value = manager.get_secret(secret_id, accessed_by="test")
            assert deleted_value is None
    
    def test_multi_environment_secret_management(self):
        """Test managing secrets across multiple environments"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            environments = ["development", "staging", "production"]
            secret_ids = {}
            
            # Create same secret type for different environments
            for env in environments:
                secret_id = manager.create_secret(
                    secret_type=SecretType.API_KEY,
                    value=f"api_key_for_{env}",
                    tags={"environment": env, "service": "payment_api"}
                )
                secret_ids[env] = secret_id
            
            # Verify environment isolation
            for env in environments:
                env_secrets = manager.search_secrets(tags={"environment": env})
                assert secret_ids[env] in env_secrets
                
                # Other environments should not be included
                other_envs = [e for e in environments if e != env]
                for other_env in other_envs:
                    assert secret_ids[other_env] not in env_secrets
    
    def test_secret_rotation_automation(self):
        """Test automated secret rotation scenarios"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # Create secrets with different rotation frequencies
            short_rotation_id = manager.create_secret(
                secret_type=SecretType.ACCESS_TOKEN,
                value="short_lived_token",
                rotation_frequency_days=7
            )
            
            long_rotation_id = manager.create_secret(
                secret_type=SecretType.CERTIFICATE,
                value="long_lived_cert",
                rotation_frequency_days=365
            )
            
            # Simulate time passing and check rotation needs
            secrets_needing_rotation = manager.get_secrets_needing_rotation(days_ahead=30)
            
            # Short rotation secret should need rotation within 30 days
            assert short_rotation_id in secrets_needing_rotation
            # Long rotation secret should not need rotation within 30 days
            assert long_rotation_id not in secrets_needing_rotation
    
    def test_secret_access_patterns(self):
        """Test different secret access patterns and tracking"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # Create a high-value secret
            secret_id = manager.create_secret(
                secret_type=SecretType.PRIVATE_KEY,
                value="high_value_private_key",
                tags={"classification": "confidential"}
            )
            
            # Simulate different access patterns
            access_patterns = [
                ("automated_service", "192.168.1.100"),
                ("human_admin", "10.0.0.50"),
                ("backup_system", "192.168.1.200"),
                ("monitoring_service", "172.16.0.10")
            ]
            
            for user, ip in access_patterns:
                manager.get_secret(
                    secret_id,
                    accessed_by=user,
                    client_ip=ip
                )
            
            # Analyze access logs
            logs = manager.get_audit_logs(secret_id)
            access_logs = [log for log in logs if log.access_type == "read"]
            
            assert len(access_logs) >= len(access_patterns)
            
            # Verify different IPs were tracked
            logged_ips = [log.client_ip for log in access_logs if log.client_ip]
            unique_ips = set(logged_ips)
            assert len(unique_ips) >= 2  # Should have multiple unique IPs
    
    def test_secret_compromise_response(self):
        """Test secret compromise detection and response"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # Create a secret
            secret_id = manager.create_secret(
                secret_type=SecretType.WEBHOOK_SECRET,
                value="webhook_secret_value"
            )
            
            # Mark as compromised
            manager.mark_secret_compromised(
                secret_id,
                compromised_by="security_team",
                reason="Detected in public repository"
            )
            
            # Verify compromise status
            metadata = manager.get_secret_metadata(secret_id)
            assert metadata.status == SecretStatus.COMPROMISED
            
            # Compromised secret should not be accessible
            retrieved_value = manager.get_secret(secret_id, accessed_by="test_user")
            assert retrieved_value is None  # Should not return compromised secrets
            
            # Should appear in compromised secrets list
            compromised_secrets = manager.list_secrets(status=SecretStatus.COMPROMISED)
            assert secret_id in compromised_secrets
    
    def test_secret_backup_and_recovery(self):
        """Test secret backup and recovery scenarios"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecretsManager(storage_path=temp_dir)
            
            # Create multiple secrets
            secrets_data = []
            for i in range(3):
                secret_id = manager.create_secret(
                    secret_type=SecretType.API_KEY,
                    value=f"backup_test_key_{i}",
                    tags={"backup_group": "test_group"}
                )
                secrets_data.append((secret_id, f"backup_test_key_{i}"))
            
            # Export secrets for backup
            backup_data = manager.export_secrets(tags={"backup_group": "test_group"})
            
            assert isinstance(backup_data, (str, dict))  # Should be serializable
            
            # Simulate deletion of secrets
            for secret_id, _ in secrets_data:
                manager.delete_secret(secret_id, deleted_by="test_cleanup")
            
            # Verify deletion
            remaining_secrets = manager.search_secrets(tags={"backup_group": "test_group"})
            assert len(remaining_secrets) == 0
            
            # Restore from backup (would need import functionality)
            # This tests the export functionality which is part of backup strategy
            assert backup_data is not None