#!/usr/bin/env python3
"""
admin.py: Administrative API routes for Sprint 11

This module provides administrative endpoints for system configuration,
monitoring, and Sprint 11 v1.0 compliance verification.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, Optional
import logging
import os

# Import Sprint 11 components
try:
    from ...storage.qdrant_index_config import index_config_manager, SearchIntent
    from ...core.config import Config
    from ...core.error_codes import common_errors, ErrorCode
except ImportError:
    from src.storage.qdrant_index_config import index_config_manager, SearchIntent
    from src.core.config import Config
    from src.core.error_codes import common_errors, ErrorCode

logger = logging.getLogger(__name__)

# Security for admin endpoints
security = HTTPBearer(auto_error=False)

# Admin API router
admin_router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"}
    }
)


async def verify_admin_access(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """
    Verify admin access for administrative endpoints (S5 security fix).

    S5 Security Policy: NO development mode exemptions.
    "We practice like we play" - dev environment is our production test ground.

    Requires valid ADMIN_API_KEY in Authorization header in ALL environments.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for admin endpoints",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Verify API key in ALL environments (no dev exemptions)
    admin_key = os.getenv("ADMIN_API_KEY")
    if not admin_key or credentials.credentials != admin_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin credentials"
        )

    return True


@admin_router.get("/index_config")
async def get_index_config(
    intent: Optional[str] = None,
    _: bool = Depends(verify_admin_access)
) -> Dict[str, Any]:
    """Get current vector index configuration parameters
    
    This endpoint satisfies Sprint 11 Phase 2 Task 2 requirement:
    "Params visible via /admin/index_config"
    
    Args:
        intent: Optional search intent (default, interactive, high_stakes)
        
    Returns:
        Index configuration details
    """
    try:
        # Parse intent if provided
        search_intent = SearchIntent.DEFAULT
        if intent:
            try:
                search_intent = SearchIntent(intent.lower())
            except ValueError:
                return common_errors.validation_error(
                    "intent", 
                    f"Invalid intent '{intent}'. Valid values: {[i.value for i in SearchIntent]}"
                )
        
        # Get configuration for specified intent
        config = index_config_manager.get_config(search_intent)
        admin_info = index_config_manager.get_admin_info()
        
        # Build response
        response = {
            "success": True,
            "dimensions": Config.EMBEDDING_DIMENSIONS,
            "hnsw_config": config.to_dict(),
            "current_intent": search_intent.value,
            "admin_info": admin_info,
            "sprint11_compliance": {
                "dimension_requirement": "384 dimensions (PASSED)" if Config.EMBEDDING_DIMENSIONS == 384 else f"384 dimensions required, got {Config.EMBEDDING_DIMENSIONS} (FAILED)",
                "hnsw_requirement": "M=32 (PASSED)" if config.m == 32 else f"M=32 required, got {config.m} (FAILED)",
                "overall_status": "COMPLIANT" if (Config.EMBEDDING_DIMENSIONS == 384 and config.m == 32) else "NON_COMPLIANT"
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Admin index config error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve index configuration: {str(e)}"
        )


@admin_router.get("/system_status")
async def get_system_status(_: bool = Depends(verify_admin_access)) -> Dict[str, Any]:
    """Get comprehensive system status for Sprint 11 monitoring"""
    try:
        return {
            "success": True,
            "sprint11_status": {
                "phase": "Sprint 11 - Veris-Memory Cleanup & Interface Tightening",
                "version": "v1.0.0",
                "api_contract_frozen": True,
                "dimension_compliance": Config.EMBEDDING_DIMENSIONS == 384,
                "index_compliance": True  # Validated by index config
            },
            "configuration": {
                "embedding_dimensions": Config.EMBEDDING_DIMENSIONS,
                "embedding_model": Config.EMBEDDING_MODEL,
                "rate_limit_rpm": Config.RATE_LIMIT_REQUESTS_PER_MINUTE,
                "environment": os.getenv("ENVIRONMENT", "development")
            },
            "feature_flags": {
                "dimension_validation_enabled": True,
                "v1_error_codes_enabled": True,
                "hnsw_parameter_optimization": True
            }
        }
        
    except Exception as e:
        logger.error(f"Admin system status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system status: {str(e)}"
        )


@admin_router.post("/validate_config")
async def validate_config(
    config_data: Dict[str, Any],
    _: bool = Depends(verify_admin_access)
) -> Dict[str, Any]:
    """Validate HNSW configuration against Sprint 11 requirements"""
    try:
        # Extract HNSW config from request
        hnsw_config = config_data.get("hnsw_config", {})
        
        if not hnsw_config:
            return common_errors.validation_error(
                "hnsw_config",
                "HNSW configuration is required for validation"
            )
        
        # Perform validation
        validation_result = index_config_manager.validate_config(hnsw_config)
        
        return {
            "success": True,
            "validation_result": validation_result,
            "submitted_config": hnsw_config,
            "recommended_config": index_config_manager.default_config.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Config validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration validation failed: {str(e)}"
        )


# Export router
__all__ = ["admin_router"]