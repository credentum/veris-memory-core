"""
Security Headers Management
Sprint 10 - Issue 003: WAF & Port Allowlisting
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CORSPolicy:
    """CORS policy configuration"""
    allowed_origins: List[str]
    allowed_methods: List[str]
    allowed_headers: List[str]
    expose_headers: List[str]
    max_age: int = 3600
    allow_credentials: bool = True


class SecurityHeadersMiddleware:
    """Security headers middleware for HTTP responses"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize security headers middleware"""
        self.config = config or {}
        self.headers = self._build_default_headers()
    
    def _build_default_headers(self) -> Dict[str, str]:
        """Build default security headers"""
        headers = {
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            
            # Enable XSS protection (legacy browsers)
            "X-XSS-Protection": "1; mode=block",
            
            # Content Security Policy
            "Content-Security-Policy": self._build_csp(),
            
            # Force HTTPS
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            
            # Control referrer information
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions Policy (formerly Feature Policy)
            "Permissions-Policy": self._build_permissions_policy(),
            
            # Prevent caching of sensitive data
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            
            # Additional security headers
            "X-Permitted-Cross-Domain-Policies": "none",
            "X-Download-Options": "noopen",
        }
        
        # Add custom headers from config
        if "custom_headers" in self.config:
            headers.update(self.config["custom_headers"])
        
        return headers
    
    def _build_csp(self) -> str:
        """Build Content Security Policy"""
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # Consider removing unsafe-* in production
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self' data:",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "upgrade-insecure-requests",
        ]
        
        # Add custom CSP from config
        if "csp_directives" in self.config:
            csp_directives.extend(self.config["csp_directives"])
        
        return "; ".join(csp_directives)
    
    def _build_permissions_policy(self) -> str:
        """Build Permissions Policy"""
        policies = [
            "accelerometer=()",
            "camera=()",
            "geolocation=()",
            "gyroscope=()",
            "magnetometer=()",
            "microphone=()",
            "payment=()",
            "usb=()",
        ]
        
        return ", ".join(policies)
    
    def get_headers(self) -> Dict[str, str]:
        """Get security headers"""
        return self.headers.copy()
    
    def apply_headers(self, response: Dict) -> Dict:
        """
        Apply security headers to response.
        
        Args:
            response: Response dictionary
            
        Returns:
            Response with security headers added
        """
        if "headers" not in response:
            response["headers"] = {}
        
        response["headers"].update(self.headers)
        return response
    
    def update_header(self, name: str, value: str):
        """Update a specific header"""
        self.headers[name] = value
        logger.info(f"Updated security header: {name}")
    
    def remove_header(self, name: str):
        """Remove a specific header"""
        if name in self.headers:
            del self.headers[name]
            logger.info(f"Removed security header: {name}")


class CORSConfig:
    """CORS configuration manager"""
    
    def __init__(self, policy: Optional[CORSPolicy] = None):
        """Initialize CORS configuration"""
        if policy:
            self.policy = policy
        else:
            # Default restrictive CORS policy
            self.policy = CORSPolicy(
                allowed_origins=["https://app.example.com"],
                allowed_methods=["GET", "POST", "OPTIONS"],
                allowed_headers=["Content-Type", "Authorization"],
                expose_headers=["X-Request-ID"],
                max_age=3600,
                allow_credentials=True
            )
    
    def is_origin_allowed(self, origin: str) -> bool:
        """
        Check if origin is allowed.
        
        Args:
            origin: Origin to check
            
        Returns:
            True if allowed, False otherwise
        """
        # Check for wildcard
        if "*" in self.policy.allowed_origins:
            return True
        
        # Check exact match
        if origin in self.policy.allowed_origins:
            return True
        
        # Check pattern match (e.g., https://*.example.com)
        for allowed in self.policy.allowed_origins:
            if self._match_origin_pattern(origin, allowed):
                return True
        
        return False
    
    def _match_origin_pattern(self, origin: str, pattern: str) -> bool:
        """Match origin against pattern with wildcards"""
        import re
        
        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace(".", r"\.")
        regex_pattern = regex_pattern.replace("*", r"[^.]+")
        regex_pattern = f"^{regex_pattern}$"
        
        return bool(re.match(regex_pattern, origin))
    
    def credentials_required(self) -> bool:
        """Check if credentials are required"""
        return self.policy.allow_credentials
    
    def get_allowed_methods(self) -> List[str]:
        """Get allowed HTTP methods"""
        return self.policy.allowed_methods.copy()
    
    def get_allowed_headers(self) -> List[str]:
        """Get allowed request headers"""
        return self.policy.allowed_headers.copy()
    
    def get_cors_headers(self, origin: str, method: str) -> Dict[str, str]:
        """
        Get CORS headers for response.
        
        Args:
            origin: Request origin
            method: Request method
            
        Returns:
            Dictionary of CORS headers
        """
        headers = {}
        
        # Check if origin is allowed
        if not self.is_origin_allowed(origin):
            return headers
        
        # Set origin header
        headers["Access-Control-Allow-Origin"] = origin
        
        # Set credentials header if required
        if self.policy.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
        
        # For preflight requests
        if method == "OPTIONS":
            headers["Access-Control-Allow-Methods"] = ", ".join(self.policy.allowed_methods)
            headers["Access-Control-Allow-Headers"] = ", ".join(self.policy.allowed_headers)
            headers["Access-Control-Max-Age"] = str(self.policy.max_age)
        
        # Expose headers
        if self.policy.expose_headers:
            headers["Access-Control-Expose-Headers"] = ", ".join(self.policy.expose_headers)
        
        return headers
    
    def add_allowed_origin(self, origin: str):
        """Add an allowed origin"""
        if origin not in self.policy.allowed_origins:
            self.policy.allowed_origins.append(origin)
            logger.info(f"Added allowed origin: {origin}")
    
    def remove_allowed_origin(self, origin: str):
        """Remove an allowed origin"""
        if origin in self.policy.allowed_origins:
            self.policy.allowed_origins.remove(origin)
            logger.info(f"Removed allowed origin: {origin}")


class ResponseSecurityFilter:
    """Filter responses for security"""
    
    def __init__(self):
        """Initialize response security filter"""
        self.sensitive_headers = {
            "Server",
            "X-Powered-By",
            "X-AspNet-Version",
            "X-AspNetMvc-Version",
            "X-Version",
        }
    
    def filter_response_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Filter sensitive headers from response.
        
        Args:
            headers: Response headers
            
        Returns:
            Filtered headers
        """
        filtered = headers.copy()
        
        # Remove sensitive headers
        for header in self.sensitive_headers:
            if header in filtered:
                del filtered[header]
                logger.debug(f"Removed sensitive header: {header}")
        
        # Sanitize remaining headers
        for key, value in filtered.items():
            # Remove newlines to prevent header injection
            if isinstance(value, str):
                filtered[key] = value.replace('\r', '').replace('\n', '')
        
        return filtered
    
    def add_request_id(self, headers: Dict[str, str], request_id: str):
        """Add request ID header for tracing"""
        headers["X-Request-ID"] = request_id
        return headers


class ContentTypeManager:
    """Manage content type headers"""
    
    def __init__(self):
        """Initialize content type manager"""
        self.type_mappings = {
            ".json": "application/json",
            ".xml": "application/xml",
            ".html": "text/html",
            ".txt": "text/plain",
            ".css": "text/css",
            ".js": "application/javascript",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".pdf": "application/pdf",
        }
    
    def get_content_type(self, file_extension: str) -> str:
        """
        Get content type for file extension.
        
        Args:
            file_extension: File extension (with or without dot)
            
        Returns:
            Content type string
        """
        if not file_extension.startswith('.'):
            file_extension = '.' + file_extension
        
        return self.type_mappings.get(
            file_extension.lower(),
            "application/octet-stream"
        )
    
    def set_content_type_header(
        self,
        headers: Dict[str, str],
        content_type: str,
        charset: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Set content type header.
        
        Args:
            headers: Headers dictionary
            content_type: Content type
            charset: Optional character set
            
        Returns:
            Updated headers
        """
        if charset and content_type.startswith("text/"):
            headers["Content-Type"] = f"{content_type}; charset={charset}"
        else:
            headers["Content-Type"] = content_type
        
        return headers


# Export main components
__all__ = [
    "SecurityHeadersMiddleware",
    "CORSConfig",
    "CORSPolicy",
    "ResponseSecurityFilter",
    "ContentTypeManager",
]