#!/usr/bin/env python3
"""
qdrant_index_config.py: HNSW Index Parameter Management for Sprint 11

This module provides centralized management and validation of Qdrant HNSW parameters
ensuring they meet Sprint 11 v1.0 specification requirements.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SearchIntent(str, Enum):
    """Search intent types for parameter optimization"""
    DEFAULT = "default"
    INTERACTIVE = "interactive"    # Lower latency for real-time queries
    HIGH_STAKES = "high_stakes"    # Higher accuracy for critical operations


@dataclass
class HNSWConfig:
    """HNSW configuration parameters"""
    m: int                    # Number of bi-directional links for every new element during construction
    ef_construct: int         # Size of dynamic candidate list during construction
    ef_search: int           # Size of dynamic candidate list during search
    max_connections: int      # Maximum number of connections for each element
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for Qdrant client"""
        return {
            "m": self.m,
            "ef_construct": self.ef_construct,
            "ef_search": self.ef_search,
            "max_connections": self.max_connections
        }


class QdrantIndexConfigManager:
    """Manages Qdrant HNSW index parameters for Sprint 11 v1.0 compliance"""
    
    # Sprint 11 v1.0 specification requirements
    REQUIRED_M = 32
    REQUIRED_EF_SEARCH_DEFAULT = 256
    REQUIRED_EF_CONSTRUCT = 128
    REQUIRED_MAX_CONNECTIONS = 64
    
    # Intent-specific overrides as per Sprint 11 spec
    INTENT_OVERRIDES = {
        SearchIntent.INTERACTIVE: {"ef_search": 128},     # Lower latency
        SearchIntent.HIGH_STAKES: {"ef_search": 384},     # Higher accuracy
    }
    
    def __init__(self):
        """Initialize index config manager"""
        self.default_config = self._create_default_config()
        self.intent_configs = self._create_intent_configs()
    
    def _create_default_config(self) -> HNSWConfig:
        """Create default HNSW configuration per Sprint 11 spec"""
        return HNSWConfig(
            m=self.REQUIRED_M,
            ef_construct=self.REQUIRED_EF_CONSTRUCT,
            ef_search=self.REQUIRED_EF_SEARCH_DEFAULT,
            max_connections=self.REQUIRED_MAX_CONNECTIONS
        )
    
    def _create_intent_configs(self) -> Dict[SearchIntent, HNSWConfig]:
        """Create intent-specific HNSW configurations"""
        configs = {}
        
        for intent, overrides in self.INTENT_OVERRIDES.items():
            config = HNSWConfig(
                m=self.REQUIRED_M,
                ef_construct=self.REQUIRED_EF_CONSTRUCT,
                ef_search=overrides.get("ef_search", self.REQUIRED_EF_SEARCH_DEFAULT),
                max_connections=self.REQUIRED_MAX_CONNECTIONS
            )
            configs[intent] = config
            
        return configs
    
    def get_config(self, intent: SearchIntent = SearchIntent.DEFAULT) -> HNSWConfig:
        """Get HNSW config for specified search intent
        
        Args:
            intent: Search intent type
            
        Returns:
            Appropriate HNSW configuration
        """
        if intent in self.intent_configs:
            return self.intent_configs[intent]
        return self.default_config
    
    def get_config_dict(self, intent: SearchIntent = SearchIntent.DEFAULT) -> Dict[str, int]:
        """Get HNSW config as dictionary for Qdrant client"""
        return self.get_config(intent).to_dict()
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate HNSW configuration against Sprint 11 requirements
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation results with any errors
        """
        errors = []
        warnings = []
        
        # Required parameter validation
        if config.get("m") != self.REQUIRED_M:
            errors.append(f"Invalid M parameter: expected {self.REQUIRED_M}, got {config.get('m')}")
        
        ef_search = config.get("ef_search")
        if ef_search not in [128, 256, 384]:
            warnings.append(f"Non-standard ef_search value: {ef_search} (expected 128, 256, or 384)")
        
        if config.get("ef_construct") != self.REQUIRED_EF_CONSTRUCT:
            warnings.append(f"Non-standard ef_construct: expected {self.REQUIRED_EF_CONSTRUCT}, got {config.get('ef_construct')}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "compliance_status": "PASSED" if len(errors) == 0 else "FAILED"
        }
    
    def get_admin_info(self) -> Dict[str, Any]:
        """Get comprehensive index configuration info for admin endpoint"""
        return {
            "default_config": self.default_config.to_dict(),
            "intent_overrides": {
                intent.value: config.to_dict() 
                for intent, config in self.intent_configs.items()
            },
            "sprint11_requirements": {
                "m": self.REQUIRED_M,
                "ef_search_default": self.REQUIRED_EF_SEARCH_DEFAULT,
                "ef_construct": self.REQUIRED_EF_CONSTRUCT,
                "max_connections": self.REQUIRED_MAX_CONNECTIONS
            },
            "available_intents": [intent.value for intent in SearchIntent],
            "compliance_notes": [
                "M=32 is required for Sprint 11 v1.0 compliance",
                "ef_search overrides available for interactive (128) and high_stakes (384)",
                "Default ef_search=256 balances performance and accuracy"
            ]
        }


# Global instance for easy import
index_config_manager = QdrantIndexConfigManager()

# Export main components
__all__ = [
    "SearchIntent",
    "HNSWConfig", 
    "QdrantIndexConfigManager",
    "index_config_manager"
]