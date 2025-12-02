#!/usr/bin/env python3
"""
Data models and configuration for Veris Sentinel.

Contains the core data structures used throughout the sentinel system.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional


@dataclass
class CheckResult:
    """Result of a single check execution."""
    check_id: str
    timestamp: datetime
    status: str  # "pass", "warn", "fail"
    latency_ms: float
    message: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckResult':
        """Create from dictionary format."""
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class SentinelConfig:
    """
    Configuration for Sentinel monitoring.

    Uses TARGET_BASE_URL environment variable for Docker deployments.
    Falls back to localhost for local development.
    """
    target_base_url: Optional[str] = None
    check_interval_seconds: int = 60
    alert_threshold_failures: int = 3
    webhook_url: Optional[str] = None
    github_token: Optional[str] = None
    github_repo: Optional[str] = None
    enabled_checks: List[str] = None

    def __post_init__(self):
        """Set defaults from environment variables if not specified."""
        import os

        # Set target_base_url from environment (Docker) or use localhost (local dev)
        # Note: Default port is 8000 to match context-store default port
        if self.target_base_url is None:
            self.target_base_url = os.getenv('TARGET_BASE_URL', 'http://localhost:8000')

        # Set default enabled checks if not specified
        # PR #397: Only runtime checks enabled by default
        # CI/CD-only checks (S3, S4, S7, S8, S9) run via GitHub Actions on deploy
        if self.enabled_checks is None:
            self.enabled_checks = [
                "S1-probes",           # Runtime: health probes (1 min)
                "S2-golden-fact-recall", # Spot-check: data quality (hourly)
                # S3-paraphrase-robustness - CI/CD only
                # S4-metrics-wiring - CI/CD only
                "S5-security-negatives", # Spot-check: security (hourly)
                "S6-backup-restore",     # Spot-check: backup validation (6 hours)
                # S7-config-parity - CI/CD only
                # S8-capacity-smoke - CI/CD only
                # S9-graph-intent - CI/CD only
                "S10-content-pipeline",  # Spot-check: pipeline health (hourly)
                "S11-firewall-status"    # Runtime: firewall security (5 min)
            ]
    
    def is_check_enabled(self, check_id: str) -> bool:
        """Check if a specific check is enabled."""
        return check_id in self.enabled_checks
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value like a dictionary.
        
        This method allows the config to be used like a dictionary
        for backward compatibility with checks.
        """
        # Map common keys to actual attributes
        if key == 'veris_memory_url':
            return self.target_base_url
        elif key == 'api_url':
            # Return the configured target_base_url (already set from environment)
            return self.target_base_url
        elif key == 'qdrant_url':
            import os
            return os.getenv('QDRANT_URL', 'http://localhost:6333')
        elif key == 'neo4j_url':
            import os
            return os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        elif key == 'redis_url':
            import os
            return os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        # Try to get from object attributes
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SentinelConfig':
        """Create from dictionary format."""
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> 'SentinelConfig':
        """Create configuration from environment variables.

        Reads Sentinel-specific environment variables and creates a SentinelConfig instance.
        TARGET_BASE_URL and enabled_checks are handled by __post_init__() to avoid duplication.

        Environment variables:
            SENTINEL_CHECK_INTERVAL: Check interval in seconds (default: 60)
            SENTINEL_ALERT_THRESHOLD: Number of failures before alerting (default: 3)
            SENTINEL_WEBHOOK_URL: Webhook URL for alerts
            GITHUB_TOKEN: GitHub token for creating issues
            SENTINEL_GITHUB_REPO: GitHub repository for issue creation
            SENTINEL_ENABLED_CHECKS: Comma-separated list of enabled checks
            TARGET_BASE_URL: Base URL for Veris Memory (handled by __post_init__)
        """
        import os

        # Let __post_init__() handle target_base_url and enabled_checks to avoid duplication
        # Only read Sentinel-specific env vars here
        return cls(
            # target_base_url=None handled by __post_init__() reading TARGET_BASE_URL
            check_interval_seconds=int(os.getenv('SENTINEL_CHECK_INTERVAL', '60')),
            alert_threshold_failures=int(os.getenv('SENTINEL_ALERT_THRESHOLD', '3')),
            webhook_url=os.getenv('SENTINEL_WEBHOOK_URL'),
            github_token=os.getenv('GITHUB_TOKEN'),
            github_repo=os.getenv('SENTINEL_GITHUB_REPO'),
            # enabled_checks=None handled by __post_init__() with SENTINEL_ENABLED_CHECKS fallback
            enabled_checks=os.getenv('SENTINEL_ENABLED_CHECKS', '').split(',') if os.getenv('SENTINEL_ENABLED_CHECKS') else None
        )