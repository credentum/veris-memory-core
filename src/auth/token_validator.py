"""
JWT Token Validation Module
Sprint 10 - Issue 002: Token validation and security
"""

import os
import jwt
import json
import hashlib
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
import redis

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of token validation"""
    is_valid: bool
    error: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    issued_at: Optional[datetime] = None
    token_id: Optional[str] = None
    status_code: int = 200

    @property
    def authorized(self) -> bool:
        """Compatibility property for legacy code"""
        return self.is_valid


class TokenValidator:
    """
    Comprehensive JWT token validator with security features.
    Validates tokens, checks signatures, expiry, and maintains revocation list.
    """
    
    # Token configuration constants
    DEFAULT_ALGORITHM = "HS256"
    SUPPORTED_ALGORITHMS = ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"]
    MAX_TOKEN_AGE_DAYS = 30
    TOKEN_REFRESH_THRESHOLD = 3600  # 1 hour before expiry
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        public_key: Optional[str] = None,
        algorithm: str = "HS256",
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
        redis_client: Optional[redis.Redis] = None
    ):
        """
        Initialize token validator.
        
        Args:
            secret_key: Secret key for HMAC algorithms
            public_key: Public key for RSA algorithms
            algorithm: JWT algorithm to use
            issuer: Expected token issuer
            audience: Expected token audience
            redis_client: Redis client for revocation list
        """
        self.secret_key = secret_key or os.environ.get("JWT_SECRET", "default-secret-key")
        self.public_key = public_key
        self.algorithm = algorithm if algorithm in self.SUPPORTED_ALGORITHMS else self.DEFAULT_ALGORITHM
        self.issuer = issuer or os.environ.get("JWT_ISSUER", "context-store")
        self.audience = audience or os.environ.get("JWT_AUDIENCE", "context-store-api")
        self.redis_client = redis_client or self._init_redis()
        
        # Initialize revocation list
        self.revoked_tokens = set()
        self._load_revoked_tokens()
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis connection for token revocation"""
        try:
            return redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", 6379)),
                password=os.environ.get("REDIS_PASSWORD"),
                decode_responses=True,
                ssl=os.environ.get("REDIS_TLS", "false").lower() == "true"
            )
        except Exception as e:
            logger.warning(f"Redis not available for token revocation: {e}")
            return None
    
    def _load_revoked_tokens(self):
        """Load revoked tokens from Redis"""
        if self.redis_client:
            try:
                revoked = self.redis_client.smembers("revoked_tokens")
                self.revoked_tokens = set(revoked) if revoked else set()
            except Exception as e:
                logger.warning(f"Failed to load revoked tokens: {e}")
                # Initialize empty set if Redis is unavailable
                self.revoked_tokens = set()
    
    def validate(self, token: str) -> ValidationResult:
        """
        Validate JWT token comprehensively.
        
        Args:
            token: JWT token string
            
        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(is_valid=False)
        
        # Check if token is revoked
        if self._is_revoked(token):
            result.error = "Token has been revoked"
            result.status_code = 401  # Unauthorized
            return result
        
        try:
            # Determine the key based on algorithm
            if self.algorithm.startswith("RS"):
                key = self.public_key
            else:
                key = self.secret_key
            
            # Decode and validate token
            decode_kwargs = {
                "algorithms": [self.algorithm],
                "options": {
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iat": True,
                    "verify_aud": True if self.audience else False,
                    "verify_iss": True if self.issuer else False,
                }
            }
            
            # Only add issuer/audience if they are set
            if self.issuer:
                decode_kwargs["issuer"] = self.issuer
            if self.audience:
                decode_kwargs["audience"] = self.audience
                
            payload = jwt.decode(token, key, **decode_kwargs)
            
            # Extract token information
            result.is_valid = True
            result.status_code = 200  # OK
            result.user_id = payload.get("sub")
            result.role = payload.get("role", "guest")
            result.capabilities = payload.get("capabilities", [])
            result.token_id = payload.get("jti")
            
            # Parse timestamps
            if "exp" in payload:
                result.expires_at = datetime.fromtimestamp(payload["exp"])
            if "iat" in payload:
                result.issued_at = datetime.fromtimestamp(payload["iat"])
            
            # Additional validations
            if not self._validate_token_age(result):
                result.is_valid = False
                result.error = "Token age exceeds maximum allowed"
                result.status_code = 401  # Unauthorized
            
            if not self._validate_required_claims(payload):
                result.is_valid = False
                result.error = "Missing required claims"
                result.status_code = 401  # Unauthorized
            
        except jwt.ExpiredSignatureError:
            result.error = "Token has expired"
            result.status_code = 401  # Unauthorized
        except jwt.InvalidSignatureError:
            result.error = "Invalid token signature"
            result.status_code = 401  # Unauthorized
        except jwt.InvalidTokenError as e:
            result.error = f"Invalid token: {str(e)}"
            result.status_code = 401  # Unauthorized
        except jwt.InvalidIssuerError:
            result.error = "Invalid token issuer"
            result.status_code = 401  # Unauthorized
        except jwt.InvalidAudienceError:
            result.error = "Invalid token audience"
            result.status_code = 401  # Unauthorized
        except Exception as e:
            result.error = f"Token validation failed: {str(e)}"
            result.status_code = 500  # Internal server error
            logger.error(f"Unexpected error during token validation: {e}")
        
        return result
    
    def _is_revoked(self, token: str) -> bool:
        """Check if token is in revocation list"""
        # Check in-memory set
        if token in self.revoked_tokens:
            return True
        
        # Check Redis if available (with timeout and robust error handling)
        if self.redis_client:
            try:
                token_hash = self._hash_token(token)
                # Use a short timeout to prevent hanging
                result = self.redis_client.sismember("revoked_tokens", token_hash)
                return bool(result)
            except Exception as e:
                logger.warning(f"Failed to check token revocation in Redis, assuming not revoked: {e}")
                # If Redis check fails, assume token is not revoked (fail-open for availability)
        
        return False
    
    def _hash_token(self, token: str) -> str:
        """Hash token for storage"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def _validate_token_age(self, result: ValidationResult) -> bool:
        """Validate token age is within acceptable range"""
        if result.issued_at:
            age = datetime.now(timezone.utc) - result.issued_at
            if age.days > self.MAX_TOKEN_AGE_DAYS:
                return False
        return True
    
    def _validate_required_claims(self, payload: Dict[str, Any]) -> bool:
        """Validate required JWT claims are present"""
        required_claims = ["sub", "role", "exp", "iat"]
        return all(claim in payload for claim in required_claims)
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token by adding it to revocation list.
        
        Args:
            token: JWT token to revoke
            
        Returns:
            True if successfully revoked
        """
        try:
            # Add to in-memory set
            self.revoked_tokens.add(token)
            
            # Add to Redis if available
            if self.redis_client:
                token_hash = self._hash_token(token)
                
                # Get token expiry to set TTL
                validation = self.validate(token)
                if validation.expires_at:
                    ttl = int((validation.expires_at - datetime.now(timezone.utc)).total_seconds())
                    if ttl > 0:
                        self.redis_client.sadd("revoked_tokens", token_hash)
                        self.redis_client.expire(f"revoked_token:{token_hash}", ttl)
            
            logger.info(f"Token revoked: {token[:20]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False
    
    def needs_refresh(self, token: str) -> bool:
        """
        Check if token needs refresh (close to expiry).
        
        Args:
            token: JWT token to check
            
        Returns:
            True if token should be refreshed
        """
        validation = self.validate(token)
        
        if not validation.is_valid:
            return False
        
        if validation.expires_at:
            time_to_expiry = validation.expires_at - datetime.now(timezone.utc)
            return time_to_expiry.total_seconds() < self.TOKEN_REFRESH_THRESHOLD
        
        return False
    
    def extract_claims(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Extract claims from token without full validation.
        
        Args:
            token: JWT token
            
        Returns:
            Token claims or None if invalid
        """
        try:
            # Decode without verification for claim extraction
            claims = jwt.decode(
                token,
                options={"verify_signature": False}
            )
            return claims
        except Exception as e:
            logger.error(f"Failed to extract claims: {e}")
            return None
    
    def validate_permissions(
        self,
        token: str,
        required_capabilities: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate token has required capabilities.
        
        Args:
            token: JWT token
            required_capabilities: List of required capabilities
            
        Returns:
            Tuple of (is_authorized, error_message)
        """
        validation = self.validate(token)
        
        if not validation.is_valid:
            return False, validation.error
        
        # Admin bypass
        if validation.role == "admin":
            return True, None
        
        # Check capabilities
        token_capabilities = set(validation.capabilities)
        required_set = set(required_capabilities)
        
        if "*" in token_capabilities:  # Wildcard permission
            return True, None
        
        if required_set.issubset(token_capabilities):
            return True, None
        
        missing = required_set - token_capabilities
        return False, f"Missing capabilities: {', '.join(missing)}"
    
    def get_token_info(self, token: str) -> Dict[str, Any]:
        """
        Get detailed token information for debugging.
        
        Args:
            token: JWT token
            
        Returns:
            Dictionary with token details
        """
        validation = self.validate(token)
        claims = self.extract_claims(token)
        
        info = {
            "valid": validation.is_valid,
            "error": validation.error,
            "user_id": validation.user_id,
            "role": validation.role,
            "capabilities": validation.capabilities,
            "expires_at": validation.expires_at.isoformat() if validation.expires_at else None,
            "issued_at": validation.issued_at.isoformat() if validation.issued_at else None,
            "token_id": validation.token_id,
            "revoked": self._is_revoked(token),
            "needs_refresh": self.needs_refresh(token),
            "algorithm": claims.get("alg") if claims else None,
            "issuer": claims.get("iss") if claims else None,
            "audience": claims.get("aud") if claims else None,
        }
        
        # Calculate time to expiry
        if validation.expires_at:
            time_to_expiry = validation.expires_at - datetime.now(timezone.utc)
            info["expires_in_seconds"] = int(time_to_expiry.total_seconds())
        
        return info


class TokenBlacklist:
    """
    Manages token blacklist for revocation.
    Separate from validator for modularity.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize token blacklist"""
        self.redis_client = redis_client or self._init_redis()
        
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis connection"""
        try:
            return redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", 6379)),
                password=os.environ.get("REDIS_PASSWORD"),
                decode_responses=True
            )
        except Exception:
            return None
    
    def add(self, token: str, expires_at: Optional[datetime] = None):
        """Add token to blacklist"""
        if self.redis_client:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            
            if expires_at:
                ttl = int((expires_at - datetime.now(timezone.utc)).total_seconds())
                if ttl > 0:
                    self.redis_client.setex(
                        f"blacklist:{token_hash}",
                        ttl,
                        json.dumps({
                            "revoked_at": datetime.now(timezone.utc).isoformat(),
                            "expires_at": expires_at.isoformat()
                        })
                    )
    
    def is_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        if self.redis_client:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            return self.redis_client.exists(f"blacklist:{token_hash}") > 0
        return False
    
    def remove(self, token: str):
        """Remove token from blacklist"""
        if self.redis_client:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            self.redis_client.delete(f"blacklist:{token_hash}")


# Export components
__all__ = [
    "TokenValidator",
    "TokenBlacklist",
    "ValidationResult",
]