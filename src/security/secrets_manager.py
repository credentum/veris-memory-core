"""
Secrets Management System
Sprint 10 Phase 3 - Issue 007: SEC-107
Provides enterprise-grade secrets management and rotation
"""

import os
import json
import base64
import secrets
import hashlib
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

logger = logging.getLogger(__name__)


class SecretType(Enum):
    """Types of secrets"""
    DATABASE_PASSWORD = "database_password"
    API_KEY = "api_key"
    JWT_SECRET = "jwt_secret"
    ENCRYPTION_KEY = "encryption_key"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"
    ACCESS_TOKEN = "access_token"
    WEBHOOK_SECRET = "webhook_secret"
    SERVICE_ACCOUNT = "service_account"


class SecretStatus(Enum):
    """Secret lifecycle status"""
    ACTIVE = "active"
    EXPIRED = "expired" 
    REVOKED = "revoked"
    PENDING_ROTATION = "pending_rotation"
    COMPROMISED = "compromised"


@dataclass
class SecretMetadata:
    """Metadata for a secret"""
    secret_id: str
    secret_type: SecretType
    status: SecretStatus
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_rotated: Optional[datetime] = None
    rotation_frequency_days: int = 90
    tags: Dict[str, str] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class SecretValue:
    """Encrypted secret value with metadata"""
    encrypted_value: bytes
    metadata: SecretMetadata
    checksum: str
    
    def verify_integrity(self) -> bool:
        """Verify secret value hasn't been tampered with"""
        expected_checksum = hashlib.sha256(self.encrypted_value).hexdigest()
        return self.checksum == expected_checksum


@dataclass
class SecretAccessLog:
    """Log entry for secret access"""
    secret_id: str
    accessed_by: str
    access_type: str  # read, write, rotate, revoke
    timestamp: datetime
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


class SecretDetector:
    """Detects secrets in text content"""
    
    def __init__(self):
        """Initialize secret detection patterns"""
        self.patterns = {
            SecretType.API_KEY: [
                r'api[_-]?key["\s]*[:=]["\s]*([a-zA-Z0-9_\-]{20,})',
                r'apikey["\s]*[:=]["\s]*([a-zA-Z0-9_\-]{20,})',
                r'sk-[a-zA-Z0-9]{48}',  # OpenAI style
            ],
            SecretType.JWT_SECRET: [
                r'jwt[_-]?secret["\s]*[:=]["\s]*([a-zA-Z0-9_\-+/=]{32,})',
            ],
            SecretType.DATABASE_PASSWORD: [
                r'password["\s]*[:=]["\s]*([^\s"\']{8,})',
                r'db[_-]?pass["\s]*[:=]["\s]*([^\s"\']{8,})',
            ],
            SecretType.ACCESS_TOKEN: [
                r'access[_-]?token["\s]*[:=]["\s]*([a-zA-Z0-9_\-+/=]{32,})',
                r'bearer["\s]+([a-zA-Z0-9_\-+/=]{32,})',
            ],
            SecretType.PRIVATE_KEY: [
                r'-----BEGIN [A-Z ]+ PRIVATE KEY-----',
                r'-----BEGIN RSA PRIVATE KEY-----',
            ]
        }
    
    def detect_secrets(self, content: str) -> List[Tuple[SecretType, str, int]]:
        """
        Detect secrets in content.
        
        Returns:
            List of (secret_type, matched_value, position) tuples
        """
        import re
        
        detected = []
        
        for secret_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        secret_value = match.group(1)
                    else:
                        secret_value = match.group(0)
                    
                    detected.append((secret_type, secret_value, match.start()))
        
        return detected
    
    def is_likely_secret(self, value: str, context: str = "") -> Tuple[bool, Optional[SecretType]]:
        """
        Check if a value is likely a secret based on patterns and context.
        
        Args:
            value: The value to check
            context: Surrounding context (variable name, etc.)
            
        Returns:
            (is_secret, secret_type) tuple
        """
        # Check for high entropy
        entropy = self._calculate_entropy(value)
        
        # Check length and character composition
        has_mixed_case = any(c.islower() for c in value) and any(c.isupper() for c in value)
        has_numbers = any(c.isdigit() for c in value)
        has_special = any(c in '_-+/=' for c in value)
        
        # Context-based detection
        context_lower = context.lower()
        secret_keywords = ['secret', 'key', 'token', 'password', 'pass', 'auth', 'api']
        
        has_secret_context = any(keyword in context_lower for keyword in secret_keywords)
        
        # Combine factors for decision
        if len(value) >= 16 and entropy > 3.5:
            if has_mixed_case or has_numbers or has_special:
                if has_secret_context:
                    # Determine most likely secret type based on context
                    if 'api' in context_lower or 'key' in context_lower:
                        return True, SecretType.API_KEY
                    elif 'jwt' in context_lower:
                        return True, SecretType.JWT_SECRET
                    elif 'pass' in context_lower:
                        return True, SecretType.DATABASE_PASSWORD
                    elif 'token' in context_lower:
                        return True, SecretType.ACCESS_TOKEN
                    else:
                        return True, SecretType.API_KEY  # Default assumption
                elif entropy > 4.0 and len(value) >= 32:
                    return True, SecretType.API_KEY  # High entropy, likely secret
        
        return False, None
    
    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of a string"""
        import math
        
        if not data:
            return 0
        
        # Calculate frequency of each character
        frequency = {}
        for char in data:
            frequency[char] = frequency.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0
        length = len(data)
        for count in frequency.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy


class SecretEncryption:
    """Handles secret encryption and decryption"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        """Initialize with master key or generate new one"""
        if master_key:
            self.master_key = master_key
        else:
            self.master_key = Fernet.generate_key()
        
        self.fernet = Fernet(self.master_key)
    
    @classmethod
    def from_password(cls, password: str, salt: Optional[bytes] = None) -> 'SecretEncryption':
        """Create encryption instance from password"""
        if salt is None:
            # Use deterministic salt based on password for consistent key generation
            salt = hashlib.sha256(password.encode()).digest()[:16]
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return cls(key)
    
    def encrypt(self, data: str) -> bytes:
        """Encrypt string data"""
        return self.fernet.encrypt(data.encode())
    
    def decrypt(self, encrypted_data: bytes) -> str:
        """Decrypt to string"""
        return self.fernet.decrypt(encrypted_data).decode()
    
    def rotate_key(self) -> bytes:
        """Generate new master key"""
        old_key = self.master_key
        self.master_key = Fernet.generate_key()
        self.fernet = Fernet(self.master_key)
        return old_key


class SecretStorage:
    """Local file-based secret storage"""
    
    def __init__(self, storage_path: str, encryption: SecretEncryption):
        """Initialize storage with path and encryption"""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, mode=0o700)  # Secure permissions
        self.encryption = encryption
        self._lock = threading.RLock()
    
    def store_secret(self, secret_value: SecretValue) -> bool:
        """Store encrypted secret"""
        with self._lock:
            try:
                secret_file = self.storage_path / f"{secret_value.metadata.secret_id}.json"
                
                data = {
                    "encrypted_value": base64.b64encode(secret_value.encrypted_value).decode(),
                    "metadata": {
                        "secret_id": secret_value.metadata.secret_id,
                        "secret_type": secret_value.metadata.secret_type.value,
                        "status": secret_value.metadata.status.value,
                        "created_at": secret_value.metadata.created_at.isoformat(),
                        "expires_at": secret_value.metadata.expires_at.isoformat() if secret_value.metadata.expires_at else None,
                        "last_rotated": secret_value.metadata.last_rotated.isoformat() if secret_value.metadata.last_rotated else None,
                        "rotation_frequency_days": secret_value.metadata.rotation_frequency_days,
                        "tags": secret_value.metadata.tags,
                        "access_count": secret_value.metadata.access_count,
                        "last_accessed": secret_value.metadata.last_accessed.isoformat() if secret_value.metadata.last_accessed else None,
                    },
                    "checksum": secret_value.checksum
                }
                
                # Write with secure permissions
                with open(secret_file, 'w', opener=lambda p, f: os.open(p, f, mode=0o600)) as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"Stored secret: {secret_value.metadata.secret_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to store secret {secret_value.metadata.secret_id}: {e}")
                return False
    
    def retrieve_secret(self, secret_id: str) -> Optional[SecretValue]:
        """Retrieve secret by ID"""
        with self._lock:
            try:
                secret_file = self.storage_path / f"{secret_id}.json"
                
                if not secret_file.exists():
                    return None
                
                with open(secret_file, 'r') as f:
                    data = json.load(f)
                
                metadata = SecretMetadata(
                    secret_id=data["metadata"]["secret_id"],
                    secret_type=SecretType(data["metadata"]["secret_type"]),
                    status=SecretStatus(data["metadata"]["status"]),
                    created_at=datetime.fromisoformat(data["metadata"]["created_at"]),
                    expires_at=datetime.fromisoformat(data["metadata"]["expires_at"]) if data["metadata"]["expires_at"] else None,
                    last_rotated=datetime.fromisoformat(data["metadata"]["last_rotated"]) if data["metadata"]["last_rotated"] else None,
                    rotation_frequency_days=data["metadata"]["rotation_frequency_days"],
                    tags=data["metadata"]["tags"],
                    access_count=data["metadata"]["access_count"],
                    last_accessed=datetime.fromisoformat(data["metadata"]["last_accessed"]) if data["metadata"]["last_accessed"] else None,
                )
                
                secret_value = SecretValue(
                    encrypted_value=base64.b64decode(data["encrypted_value"]),
                    metadata=metadata,
                    checksum=data["checksum"]
                )
                
                # Verify integrity
                if not secret_value.verify_integrity():
                    logger.error(f"Secret integrity check failed: {secret_id}")
                    return None
                
                return secret_value
                
            except Exception as e:
                logger.error(f"Failed to retrieve secret {secret_id}: {e}")
                return None
    
    def delete_secret(self, secret_id: str) -> bool:
        """Delete secret from storage"""
        with self._lock:
            try:
                secret_file = self.storage_path / f"{secret_id}.json"
                
                if secret_file.exists():
                    secret_file.unlink()
                    logger.info(f"Deleted secret: {secret_id}")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete secret {secret_id}: {e}")
                return False
    
    def list_secrets(self) -> List[SecretMetadata]:
        """List all secret metadata"""
        with self._lock:
            secrets = []
            
            for secret_file in self.storage_path.glob("*.json"):
                try:
                    with open(secret_file, 'r') as f:
                        data = json.load(f)
                    
                    metadata = SecretMetadata(
                        secret_id=data["metadata"]["secret_id"],
                        secret_type=SecretType(data["metadata"]["secret_type"]),
                        status=SecretStatus(data["metadata"]["status"]),
                        created_at=datetime.fromisoformat(data["metadata"]["created_at"]),
                        expires_at=datetime.fromisoformat(data["metadata"]["expires_at"]) if data["metadata"]["expires_at"] else None,
                        last_rotated=datetime.fromisoformat(data["metadata"]["last_rotated"]) if data["metadata"]["last_rotated"] else None,
                        rotation_frequency_days=data["metadata"]["rotation_frequency_days"],
                        tags=data["metadata"]["tags"],
                        access_count=data["metadata"]["access_count"],
                        last_accessed=datetime.fromisoformat(data["metadata"]["last_accessed"]) if data["metadata"]["last_accessed"] else None,
                    )
                    
                    secrets.append(metadata)
                    
                except Exception as e:
                    logger.error(f"Failed to read secret metadata from {secret_file}: {e}")
            
            return secrets


class SecretsManager:
    """Main secrets management system"""
    
    def __init__(self, storage_path: str = "./secrets", master_password: Optional[str] = None):
        """Initialize secrets manager"""
        self.storage_path = storage_path
        
        # Initialize encryption
        if master_password:
            self.encryption = SecretEncryption.from_password(master_password)
        else:
            self.encryption = SecretEncryption()
        
        # Initialize components
        self.detector = SecretDetector()
        self.storage = SecretStorage(storage_path, self.encryption)
        self.access_logs: List[SecretAccessLog] = []
        self._lock = threading.RLock()
        
        # Rotation tracking
        self._rotation_schedule: Dict[str, datetime] = {}
    
    def create_secret(self, 
                     secret_id: str,
                     secret_value: str,
                     secret_type: SecretType,
                     expires_in_days: Optional[int] = None,
                     rotation_frequency_days: int = 90,
                     tags: Optional[Dict[str, str]] = None) -> bool:
        """Create a new secret"""
        
        with self._lock:
            try:
                # Check if secret already exists
                if self.storage.retrieve_secret(secret_id):
                    logger.error(f"Secret {secret_id} already exists")
                    return False
                
                # Create metadata
                now = datetime.utcnow()
                expires_at = now + timedelta(days=expires_in_days) if expires_in_days else None
                
                metadata = SecretMetadata(
                    secret_id=secret_id,
                    secret_type=secret_type,
                    status=SecretStatus.ACTIVE,
                    created_at=now,
                    expires_at=expires_at,
                    rotation_frequency_days=rotation_frequency_days,
                    tags=tags or {}
                )
                
                # Encrypt secret
                encrypted_value = self.encryption.encrypt(secret_value)
                checksum = hashlib.sha256(encrypted_value).hexdigest()
                
                secret = SecretValue(
                    encrypted_value=encrypted_value,
                    metadata=metadata,
                    checksum=checksum
                )
                
                # Store secret
                if self.storage.store_secret(secret):
                    self._log_access(secret_id, "system", "create")
                    self._schedule_rotation(secret_id, rotation_frequency_days)
                    return True
                
                return False
                
            except Exception as e:
                logger.error(f"Failed to create secret {secret_id}: {e}")
                return False
    
    def get_secret(self, secret_id: str, accessed_by: str = "system") -> Optional[str]:
        """Retrieve and decrypt a secret"""
        
        with self._lock:
            try:
                secret = self.storage.retrieve_secret(secret_id)
                
                if not secret:
                    self._log_access(secret_id, accessed_by, "read", success=False, 
                                   error="Secret not found")
                    return None
                
                # Check if secret is active
                if secret.metadata.status != SecretStatus.ACTIVE:
                    self._log_access(secret_id, accessed_by, "read", success=False,
                                   error=f"Secret status: {secret.metadata.status.value}")
                    return None
                
                # Check expiration
                if secret.metadata.expires_at and secret.metadata.expires_at <= datetime.utcnow():
                    # Mark as expired
                    secret.metadata.status = SecretStatus.EXPIRED
                    self.storage.store_secret(secret)
                    
                    self._log_access(secret_id, accessed_by, "read", success=False,
                                   error="Secret expired")
                    return None
                
                # Decrypt secret
                decrypted_value = self.encryption.decrypt(secret.encrypted_value)
                
                # Update access tracking
                secret.metadata.access_count += 1
                secret.metadata.last_accessed = datetime.utcnow()
                self.storage.store_secret(secret)
                
                self._log_access(secret_id, accessed_by, "read")
                
                return decrypted_value
                
            except Exception as e:
                logger.error(f"Failed to retrieve secret {secret_id}: {e}")
                self._log_access(secret_id, accessed_by, "read", success=False,
                               error=str(e))
                return None
    
    def rotate_secret(self, secret_id: str, new_value: Optional[str] = None, 
                     rotated_by: str = "system") -> bool:
        """Rotate a secret with new value"""
        
        with self._lock:
            try:
                secret = self.storage.retrieve_secret(secret_id)
                
                if not secret:
                    self._log_access(secret_id, rotated_by, "rotate", success=False,
                                   error="Secret not found")
                    return False
                
                # Generate new value if not provided
                if new_value is None:
                    new_value = self._generate_secret_value(secret.metadata.secret_type)
                
                # Update secret
                encrypted_value = self.encryption.encrypt(new_value)
                secret.encrypted_value = encrypted_value
                secret.checksum = hashlib.sha256(encrypted_value).hexdigest()
                secret.metadata.last_rotated = datetime.utcnow()
                secret.metadata.status = SecretStatus.ACTIVE
                
                # Reset expiration if applicable
                if secret.metadata.expires_at:
                    secret.metadata.expires_at = datetime.utcnow() + timedelta(days=90)
                
                # Store updated secret
                if self.storage.store_secret(secret):
                    self._log_access(secret_id, rotated_by, "rotate")
                    self._schedule_rotation(secret_id, secret.metadata.rotation_frequency_days)
                    logger.info(f"Rotated secret: {secret_id}")
                    return True
                
                return False
                
            except Exception as e:
                logger.error(f"Failed to rotate secret {secret_id}: {e}")
                self._log_access(secret_id, rotated_by, "rotate", success=False,
                               error=str(e))
                return False
    
    def revoke_secret(self, secret_id: str, revoked_by: str = "system") -> bool:
        """Mark secret as revoked"""
        
        with self._lock:
            try:
                secret = self.storage.retrieve_secret(secret_id)
                
                if not secret:
                    self._log_access(secret_id, revoked_by, "revoke", success=False,
                                   error="Secret not found")
                    return False
                
                secret.metadata.status = SecretStatus.REVOKED
                
                if self.storage.store_secret(secret):
                    self._log_access(secret_id, revoked_by, "revoke")
                    logger.info(f"Revoked secret: {secret_id}")
                    return True
                
                return False
                
            except Exception as e:
                logger.error(f"Failed to revoke secret {secret_id}: {e}")
                self._log_access(secret_id, revoked_by, "revoke", success=False,
                               error=str(e))
                return False
    
    def list_secrets(self, include_revoked: bool = False) -> List[SecretMetadata]:
        """List all secrets"""
        
        secrets = self.storage.list_secrets()
        
        if not include_revoked:
            secrets = [s for s in secrets if s.status != SecretStatus.REVOKED]
        
        return secrets
    
    def get_secrets_requiring_rotation(self) -> List[SecretMetadata]:
        """Get secrets that need rotation"""
        
        secrets = self.storage.list_secrets()
        requiring_rotation = []
        now = datetime.utcnow()
        
        for secret in secrets:
            if secret.status != SecretStatus.ACTIVE:
                continue
            
            # Check if rotation is overdue
            if secret.last_rotated:
                next_rotation = secret.last_rotated + timedelta(days=secret.rotation_frequency_days)
                if now >= next_rotation:
                    requiring_rotation.append(secret)
            else:
                # Never rotated, check based on creation date
                next_rotation = secret.created_at + timedelta(days=secret.rotation_frequency_days)
                if now >= next_rotation:
                    requiring_rotation.append(secret)
        
        return requiring_rotation
    
    def scan_content_for_secrets(self, content: str) -> List[Tuple[SecretType, str, int]]:
        """Scan content for potential secrets"""
        return self.detector.detect_secrets(content)
    
    def get_access_logs(self, secret_id: Optional[str] = None, 
                       limit: int = 100) -> List[SecretAccessLog]:
        """Get access logs"""
        
        logs = self.access_logs
        
        if secret_id:
            logs = [log for log in logs if log.secret_id == secret_id]
        
        # Return most recent logs first
        return sorted(logs, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def _generate_secret_value(self, secret_type: SecretType) -> str:
        """Generate a new secret value based on type"""
        
        if secret_type == SecretType.DATABASE_PASSWORD:
            # Generate strong password
            length = 32
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
            return ''.join(secrets.choice(chars) for _ in range(length))
        
        elif secret_type == SecretType.API_KEY:
            # Generate API key format
            return f"sk-{secrets.token_urlsafe(32)}"
        
        elif secret_type == SecretType.JWT_SECRET:
            # Generate JWT secret
            return secrets.token_urlsafe(64)
        
        elif secret_type == SecretType.ENCRYPTION_KEY:
            # Generate encryption key
            return base64.b64encode(secrets.token_bytes(32)).decode()
        
        else:
            # Default: random token
            return secrets.token_urlsafe(32)
    
    def _log_access(self, secret_id: str, accessed_by: str, access_type: str,
                   client_ip: Optional[str] = None, success: bool = True,
                   error: Optional[str] = None):
        """Log secret access"""
        
        log_entry = SecretAccessLog(
            secret_id=secret_id,
            accessed_by=accessed_by,
            access_type=access_type,
            timestamp=datetime.utcnow(),
            client_ip=client_ip,
            success=success,
            error_message=error
        )
        
        self.access_logs.append(log_entry)
        
        # Keep only recent logs (prevent memory growth)
        if len(self.access_logs) > 10000:
            self.access_logs = self.access_logs[-5000:]
    
    def _schedule_rotation(self, secret_id: str, frequency_days: int):
        """Schedule next rotation for secret"""
        next_rotation = datetime.utcnow() + timedelta(days=frequency_days)
        self._rotation_schedule[secret_id] = next_rotation


# Export main components
__all__ = [
    "SecretsManager",
    "SecretType", 
    "SecretStatus",
    "SecretMetadata",
    "SecretValue",
    "SecretAccessLog",
    "SecretDetector",
    "SecretEncryption",
    "SecretStorage",
]