#!/usr/bin/env python3
"""
test_index_config.py: Unit tests for Sprint 11 HNSW parameter validation

Tests Sprint 11 Phase 2 Task 2 requirements:
- HNSW M=32, ef_search=256 default
- Intent overrides: interactive=128, high_stakes=384
- Parameter validation and admin endpoint functionality
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Import components to test
from src.storage.qdrant_index_config import (
    QdrantIndexConfigManager,
    SearchIntent,
    HNSWConfig,
    index_config_manager
)
from src.api.routes.admin import admin_router
from src.core.config import Config


class TestHNSWConfig:
    """Test HNSW configuration dataclass"""
    
    def test_hnsw_config_creation(self):
        """Test HNSW config creation and dict conversion"""
        config = HNSWConfig(
            m=32,
            ef_construct=128,
            ef_search=256,
            max_connections=64
        )
        
        assert config.m == 32
        assert config.ef_search == 256
        
        config_dict = config.to_dict()
        assert config_dict["m"] == 32
        assert config_dict["ef_search"] == 256


class TestQdrantIndexConfigManager:
    """Test Sprint 11 HNSW parameter management"""
    
    def test_sprint11_requirements(self):
        """Test that Sprint 11 v1.0 requirements are enforced"""
        manager = QdrantIndexConfigManager()
        
        # Verify Sprint 11 specification constants
        assert manager.REQUIRED_M == 32
        assert manager.REQUIRED_EF_SEARCH_DEFAULT == 256
        assert manager.REQUIRED_EF_CONSTRUCT == 128
        assert manager.REQUIRED_MAX_CONNECTIONS == 64
    
    def test_default_config(self):
        """Test default configuration matches Sprint 11 spec"""
        manager = QdrantIndexConfigManager()
        config = manager.get_config()
        
        assert config.m == 32                    # Sprint 11 requirement
        assert config.ef_search == 256           # Sprint 11 default
        assert config.ef_construct == 128        # Sprint 11 requirement
        assert config.max_connections == 64      # Sprint 11 requirement
    
    def test_interactive_intent_override(self):
        """Test interactive intent lowers ef_search to 128"""
        manager = QdrantIndexConfigManager()
        config = manager.get_config(SearchIntent.INTERACTIVE)
        
        assert config.m == 32                    # Still required M
        assert config.ef_search == 128           # Lower for interactivity
        assert config.ef_construct == 128        # Unchanged
    
    def test_high_stakes_intent_override(self):
        """Test high_stakes intent increases ef_search to 384"""
        manager = QdrantIndexConfigManager()
        config = manager.get_config(SearchIntent.HIGH_STAKES)
        
        assert config.m == 32                    # Still required M
        assert config.ef_search == 384           # Higher for accuracy
        assert config.ef_construct == 128        # Unchanged
    
    def test_config_validation_success(self):
        """Test validation passes for compliant config"""
        manager = QdrantIndexConfigManager()
        
        valid_config = {
            "m": 32,
            "ef_search": 256,
            "ef_construct": 128,
            "max_connections": 64
        }
        
        result = manager.validate_config(valid_config)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["compliance_status"] == "PASSED"
    
    def test_config_validation_failure(self):
        """Test validation fails for non-compliant config"""
        manager = QdrantIndexConfigManager()
        
        invalid_config = {
            "m": 16,  # Wrong M value (should be 32)
            "ef_search": 100,
            "ef_construct": 64
        }
        
        result = manager.validate_config(invalid_config)
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "Invalid M parameter" in result["errors"][0]
        assert result["compliance_status"] == "FAILED"
    
    def test_admin_info_structure(self):
        """Test admin info contains all required Sprint 11 data"""
        manager = QdrantIndexConfigManager()
        admin_info = manager.get_admin_info()
        
        # Verify structure
        assert "default_config" in admin_info
        assert "intent_overrides" in admin_info  
        assert "sprint11_requirements" in admin_info
        assert "available_intents" in admin_info
        assert "compliance_notes" in admin_info
        
        # Verify Sprint 11 requirements are documented
        reqs = admin_info["sprint11_requirements"]
        assert reqs["m"] == 32
        assert reqs["ef_search_default"] == 256
        
        # Verify intent overrides exist
        overrides = admin_info["intent_overrides"]
        assert "interactive" in overrides
        assert "high_stakes" in overrides
        assert overrides["interactive"]["ef_search"] == 128
        assert overrides["high_stakes"]["ef_search"] == 384


class TestGlobalInstance:
    """Test global index_config_manager instance"""
    
    def test_global_instance_available(self):
        """Test global instance is properly initialized"""
        assert index_config_manager is not None
        assert isinstance(index_config_manager, QdrantIndexConfigManager)
    
    def test_global_instance_sprint11_compliant(self):
        """Test global instance meets Sprint 11 requirements"""
        config = index_config_manager.get_config()
        assert config.m == 32
        assert config.ef_search == 256


@pytest.mark.asyncio
class TestAdminAPI:
    """Test admin API endpoints for Sprint 11 compliance"""
    
    def test_search_intent_enum(self):
        """Test SearchIntent enum values"""
        assert SearchIntent.DEFAULT.value == "default"
        assert SearchIntent.INTERACTIVE.value == "interactive"  
        assert SearchIntent.HIGH_STAKES.value == "high_stakes"
    
    @patch('src.core.config.Config.EMBEDDING_DIMENSIONS', 384)
    async def test_index_config_endpoint_success(self):
        """Test /admin/index_config endpoint returns correct data"""
        # Mock the admin access verification
        with patch('src.api.routes.admin.verify_admin_access', return_value=True):
            from fastapi.testclient import TestClient
            from fastapi import FastAPI
            
            app = FastAPI()
            app.include_router(admin_router)
            client = TestClient(app)
            
            response = client.get("/admin/index_config")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["dimensions"] == 384
            assert data["hnsw_config"]["m"] == 32
            assert data["hnsw_config"]["ef_search"] == 256
            assert data["sprint11_compliance"]["overall_status"] == "COMPLIANT"
    
    @patch('src.core.config.Config.EMBEDDING_DIMENSIONS', 1536)  # Wrong dimension
    async def test_index_config_endpoint_non_compliant(self):
        """Test endpoint detects non-compliance"""
        with patch('src.api.routes.admin.verify_admin_access', return_value=True):
            from fastapi.testclient import TestClient
            from fastapi import FastAPI
            
            app = FastAPI()
            app.include_router(admin_router)
            client = TestClient(app)
            
            response = client.get("/admin/index_config")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["sprint11_compliance"]["overall_status"] == "NON_COMPLIANT"
            assert "FAILED" in data["sprint11_compliance"]["dimension_requirement"]
    
    async def test_intent_parameter_overrides(self):
        """Test intent parameter correctly overrides ef_search"""
        with patch('src.api.routes.admin.verify_admin_access', return_value=True):
            from fastapi.testclient import TestClient
            from fastapi import FastAPI
            
            app = FastAPI()
            app.include_router(admin_router)
            client = TestClient(app)
            
            # Test interactive intent
            response = client.get("/admin/index_config?intent=interactive")
            data = response.json()
            assert data["hnsw_config"]["ef_search"] == 128
            
            # Test high_stakes intent  
            response = client.get("/admin/index_config?intent=high_stakes")
            data = response.json()
            assert data["hnsw_config"]["ef_search"] == 384


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])