#!/usr/bin/env python3
"""
Dashboard API Endpoints

WebSocket streaming and REST endpoints for the Veris Memory dashboard system.
Provides both JSON and ASCII formatted output for different client types.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Import monitoring components with comprehensive error handling
_dashboard_module = None
_streaming_module = None
_rate_limiter_module = None

try:
    # Try relative imports first (normal package structure)
    from .dashboard import UnifiedDashboard
    _dashboard_module = "relative"
    logger = logging.getLogger(__name__)
    logger.debug("Loaded dashboard via relative import")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Relative dashboard import failed: {e}")

try:
    from .streaming import MetricsStreamer
    _streaming_module = "relative"
    logger.debug("Loaded streaming via relative import")
except ImportError as e:
    logger.warning(f"Relative streaming import failed: {e}")

try:
    from ..core.rate_limiter import get_rate_limiter, MCPRateLimiter
    _rate_limiter_module = "relative"
    logger.debug("Loaded rate_limiter via relative import")
except ImportError as e:
    logger.warning(f"Relative rate_limiter import failed: {e}")

# Fallback to absolute imports if relative imports failed
if not all([_dashboard_module, _streaming_module, _rate_limiter_module]):
    try:
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        if not _dashboard_module:
            from src.monitoring.dashboard import UnifiedDashboard
            _dashboard_module = "absolute"
            logger.info("Loaded dashboard via absolute import fallback")
        
        if not _streaming_module:
            from src.monitoring.streaming import MetricsStreamer
            _streaming_module = "absolute"
            logger.info("Loaded streaming via absolute import fallback")
        
        if not _rate_limiter_module:
            from src.core.rate_limiter import get_rate_limiter, MCPRateLimiter
            _rate_limiter_module = "absolute"
            logger.info("Loaded rate_limiter via absolute import fallback")
            
    except ImportError as e:
        logger.error(f"Failed to import monitoring components: {e}")
        logger.error("Dashboard API may not function correctly without these dependencies")
        
        # Create fallback implementations to prevent crashes
        class UnifiedDashboard:
            def __init__(self, *args, **kwargs):
                self.last_update = None
                logger.error("Using fallback UnifiedDashboard - functionality limited")
            
            async def collect_all_metrics(self, *args, **kwargs):
                return {"error": "Dashboard module not available"}
            
            def generate_ascii_dashboard(self, *args, **kwargs):
                return "Dashboard module not available"
            
            async def shutdown(self):
                pass
        
        class MetricsStreamer:
            def __init__(self, *args, **kwargs):
                logger.error("Using fallback MetricsStreamer - functionality limited")
        
        def get_rate_limiter():
            logger.error("Rate limiter not available - rate limiting disabled")
            class MockRateLimiter:
                def get_client_id(self, *args): return "unknown"
                async def async_check_rate_limit(self, *args): return True, None
                async def async_check_burst_protection(self, *args): return True, None
                endpoint_limits = {}
            return MockRateLimiter()
        
        class MCPRateLimiter:
            pass

logger = logging.getLogger(__name__)


class MonitoringRateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for monitoring endpoints."""
    
    def __init__(self, app, config: Optional[Dict[str, Any]] = None):
        super().__init__(app)
        self.config = config or {}
        self.rate_limiter = get_rate_limiter()
        
        # Configure monitoring-specific rate limits
        self.monitoring_limits = {
            "/api/dashboard": {"rpm": 60, "burst": 10},  # 1 req/sec, burst 10
            "/api/dashboard/ascii": {"rpm": 30, "burst": 5},  # 0.5 req/sec, burst 5
            "/api/dashboard/system": {"rpm": 120, "burst": 20},  # 2 req/sec, burst 20
            "/api/dashboard/services": {"rpm": 120, "burst": 20},  # 2 req/sec, burst 20
            "/api/dashboard/security": {"rpm": 60, "burst": 10},  # 1 req/sec, burst 10
            "/api/dashboard/refresh": {"rpm": 12, "burst": 3},  # 0.2 req/sec, burst 3
            "/api/dashboard/health": {"rpm": 300, "burst": 50},  # 5 req/sec, burst 50
            "/api/dashboard/connections": {"rpm": 60, "burst": 10},  # 1 req/sec, burst 10
        }
        
        # Update rate limiter with monitoring endpoint limits using thread-safe method
        monitoring_endpoint_limits = {}
        for endpoint, limits in self.monitoring_limits.items():
            endpoint_key = f"monitoring{endpoint.replace('/api/dashboard', '')}"
            monitoring_endpoint_limits[endpoint_key] = limits
        
        self.rate_limiter.register_endpoint_limits(monitoring_endpoint_limits)
    
    async def dispatch(self, request: Request, call_next):
        """Check rate limits for monitoring endpoints."""
        # Only apply rate limiting to monitoring API endpoints
        if not request.url.path.startswith("/api/dashboard"):
            return await call_next(request)
        
        # Extract client information
        client_info = {
            "remote_addr": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", ""),
            "client_id": request.headers.get("x-client-id", "")
        }
        
        client_id = self.rate_limiter.get_client_id(client_info)
        endpoint_path = request.url.path
        
        # Map endpoint path to monitoring key
        endpoint_key = f"monitoring{endpoint_path.replace('/api/dashboard', '')}"
        if not endpoint_key.replace("monitoring", ""):
            endpoint_key = "monitoring"  # Root dashboard endpoint
        
        # Check rate limits
        try:
            allowed, error_msg = await self.rate_limiter.async_check_rate_limit(
                endpoint_key, client_id, 1
            )
            
            if not allowed:
                logger.warning(f"Rate limit exceeded for {client_id} on {endpoint_path}: {error_msg}")
                return Response(
                    content=json.dumps({
                        "error": "Rate limit exceeded",
                        "message": error_msg,
                        "endpoint": endpoint_path,
                        "timestamp": datetime.utcnow().isoformat()
                    }),
                    status_code=429,
                    headers={
                        "Content-Type": "application/json",
                        "Retry-After": "60",  # Standard retry header
                        "X-RateLimit-Endpoint": endpoint_key
                    }
                )
            
            # Check burst protection
            burst_ok, burst_msg = await self.rate_limiter.async_check_burst_protection(client_id)
            if not burst_ok:
                logger.warning(f"Burst protection triggered for {client_id} on {endpoint_path}: {burst_msg}")
                return Response(
                    content=json.dumps({
                        "error": "Too many requests",
                        "message": burst_msg,
                        "endpoint": endpoint_path,
                        "timestamp": datetime.utcnow().isoformat()
                    }),
                    status_code=429,
                    headers={
                        "Content-Type": "application/json",
                        "Retry-After": "10",  # Shorter retry for burst protection
                        "X-RateLimit-Type": "burst"
                    }
                )
            
            # Log successful rate limit check
            logger.debug(f"Rate limit check passed for {client_id} on {endpoint_path}")
            
        except Exception as e:
            logger.error(f"Rate limiting error for {endpoint_path}: {e}")
            # Continue to endpoint on rate limiting errors
        
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Endpoint"] = endpoint_key
        response.headers["X-RateLimit-Client"] = client_id[:8]  # Truncated for privacy
        
        return response


class DashboardAPI:
    """
    API endpoints for the unified dashboard system.
    
    Provides WebSocket streaming for real-time updates and REST endpoints
    for on-demand dashboard data in both JSON and ASCII formats.
    """

    def __init__(self, app: FastAPI, config: Optional[Dict[str, Any]] = None):
        """Initialize dashboard API with FastAPI app."""
        self.app = app
        self.config = config or self._get_default_config()
        self.dashboard = UnifiedDashboard(self.config.get('dashboard', {}))
        self.streamer = MetricsStreamer(self.dashboard, self.config.get('streaming', {}))
        
        # WebSocket connections
        self.websocket_connections: Set[WebSocket] = set()
        
        # Setup routes
        self._setup_routes()
        self._setup_middleware()
        
        logger.info("📊 DashboardAPI initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default API configuration."""
        return {
            'streaming': {
                'update_interval_seconds': 5,
                'max_connections': 100,
                'heartbeat_interval_seconds': 30
            },
            'cors': {
                'allow_origins': ["*"],
                'allow_methods': ["*"],
                'allow_headers': ["*"]
            },
            'rate_limiting': {
                'enabled': True,
                'requests_per_minute': 60
            }
        }

    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        # Rate limiting middleware (added first to be applied early)
        rate_limit_config = self.config.get('rate_limiting', {})
        if rate_limit_config.get('enabled', True):
            self.app.add_middleware(
                MonitoringRateLimitMiddleware,
                config=rate_limit_config
            )
            logger.info("✅ Monitoring rate limiting enabled")
        
        # CORS middleware for cross-origin requests
        cors_config = self.config.get('cors', {})
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config.get('allow_origins', ["*"]),
            allow_credentials=True,
            allow_methods=cors_config.get('allow_methods', ["*"]),
            allow_headers=cors_config.get('allow_headers', ["*"])
        )

    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.websocket("/ws/dashboard")
        async def websocket_dashboard(websocket: WebSocket):
            """WebSocket endpoint for real-time dashboard streaming."""
            await self._handle_websocket_connection(websocket)

        @self.app.get("/api/dashboard")
        async def get_dashboard_json():
            """Get complete dashboard data in JSON format."""
            return await self._get_dashboard_json()

        @self.app.get("/api/dashboard/ascii", response_class=PlainTextResponse)
        async def get_dashboard_ascii():
            """Get dashboard in ASCII format for human reading."""
            return await self._get_dashboard_ascii()

        @self.app.get("/api/dashboard/system")
        async def get_system_metrics():
            """Get system metrics only."""
            return await self._get_system_metrics()

        @self.app.get("/api/dashboard/services")
        async def get_service_metrics():
            """Get service health metrics only."""
            return await self._get_service_metrics()

        @self.app.get("/api/dashboard/security")
        async def get_security_metrics():
            """Get security metrics only."""
            return await self._get_security_metrics()


        @self.app.post("/api/dashboard/refresh")
        async def refresh_dashboard():
            """Force refresh of dashboard metrics."""
            return await self._force_refresh_dashboard()

        @self.app.get("/api/dashboard/health")
        async def dashboard_health():
            """Dashboard API health check."""
            return await self._get_api_health()

        @self.app.get("/api/dashboard/connections")
        async def get_websocket_connections():
            """Get active WebSocket connection count."""
            return {"active_connections": len(self.websocket_connections)}

        @self.app.get("/api/dashboard/rate-limit-status")
        async def get_rate_limit_status(request: Request):
            """Get rate limit status for debugging."""
            return await self._get_rate_limit_status(request)

    async def _handle_websocket_connection(self, websocket: WebSocket):
        """Handle WebSocket connection for real-time dashboard updates."""
        try:
            await websocket.accept()
            self.websocket_connections.add(websocket)
            
            connection_count = len(self.websocket_connections)
            max_connections = self.config['streaming']['max_connections']
            
            logger.info(f"📡 WebSocket connected ({connection_count}/{max_connections})")
            
            if connection_count > max_connections:
                await websocket.close(code=1008, reason="Max connections exceeded")
                return

            # Send initial dashboard data
            metrics = await self.dashboard.collect_all_metrics(force_refresh=True)
            await websocket.send_json({
                "type": "initial_data",
                "timestamp": datetime.utcnow().isoformat(),
                "data": metrics
            })

            # Start streaming updates
            await self._stream_dashboard_updates(websocket)

        except WebSocketDisconnect:
            logger.info("📡 WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.close(code=1011, reason="Internal error")
        finally:
            self.websocket_connections.discard(websocket)

    async def _stream_dashboard_updates(self, websocket: WebSocket):
        """Stream dashboard updates to WebSocket client."""
        update_interval = self.config['streaming']['update_interval_seconds']
        heartbeat_interval = self.config['streaming']['heartbeat_interval_seconds']
        
        last_heartbeat = datetime.utcnow()
        
        try:
            while True:
                # Collect current metrics
                metrics = await self.dashboard.collect_all_metrics()
                
                # Send update
                update_message = {
                    "type": "dashboard_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": metrics
                }
                
                await websocket.send_json(update_message)
                
                # Send heartbeat if needed
                now = datetime.utcnow()
                if (now - last_heartbeat).total_seconds() >= heartbeat_interval:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": now.isoformat()
                    })
                    last_heartbeat = now
                
                # Wait for next update
                await asyncio.sleep(update_interval)
                
        except WebSocketDisconnect:
            logger.info("📡 WebSocket client disconnected during streaming")
        except Exception as e:
            logger.error(f"Streaming error: {e}")

    async def _get_dashboard_json(self) -> Dict[str, Any]:
        """Get complete dashboard data in JSON format."""
        try:
            metrics = await self.dashboard.collect_all_metrics()
            return {
                "success": True,
                "format": "json",
                "timestamp": datetime.utcnow().isoformat(),
                "data": metrics
            }
        except Exception as e:
            logger.error(f"Failed to get dashboard JSON: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_dashboard_ascii(self) -> str:
        """Get dashboard in ASCII format."""
        try:
            metrics = await self.dashboard.collect_all_metrics()
            ascii_output = self.dashboard.generate_ascii_dashboard(metrics)
            return ascii_output
        except Exception as e:
            logger.error(f"Failed to get dashboard ASCII: {e}")
            return f"Dashboard Error: {str(e)}"

    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics only."""
        try:
            metrics = await self.dashboard.collect_all_metrics()
            return {
                "success": True,
                "type": "system_metrics",
                "timestamp": datetime.utcnow().isoformat(),
                "data": metrics.get("system", {})
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_service_metrics(self) -> Dict[str, Any]:
        """Get service health metrics only."""
        try:
            metrics = await self.dashboard.collect_all_metrics()
            return {
                "success": True,
                "type": "service_metrics",
                "timestamp": datetime.utcnow().isoformat(),
                "data": metrics.get("services", [])
            }
        except Exception as e:
            logger.error(f"Failed to get service metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics only."""
        try:
            metrics = await self.dashboard.collect_all_metrics()
            return {
                "success": True,
                "type": "security_metrics",
                "timestamp": datetime.utcnow().isoformat(),
                "data": metrics.get("security", {})
            }
        except Exception as e:
            logger.error(f"Failed to get security metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))


    async def _force_refresh_dashboard(self) -> Dict[str, Any]:
        """Force refresh of dashboard metrics."""
        try:
            metrics = await self.dashboard.collect_all_metrics(force_refresh=True)
            
            # Broadcast update to all WebSocket connections
            await self._broadcast_to_websockets({
                "type": "force_refresh",
                "timestamp": datetime.utcnow().isoformat(),
                "data": metrics
            })
            
            return {
                "success": True,
                "message": "Dashboard metrics refreshed",
                "timestamp": datetime.utcnow().isoformat(),
                "websocket_notifications_sent": len(self.websocket_connections)
            }
        except Exception as e:
            logger.error(f"Failed to refresh dashboard: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_api_health(self) -> Dict[str, Any]:
        """Get dashboard API health status."""
        try:
            # Check dashboard health
            dashboard_healthy = self.dashboard.last_update is not None
            
            # Check WebSocket health
            websocket_healthy = len(self.websocket_connections) <= self.config['streaming']['max_connections']
            
            # Overall health
            overall_healthy = dashboard_healthy and websocket_healthy
            
            return {
                "success": True,
                "healthy": overall_healthy,
                "timestamp": datetime.utcnow().isoformat(),
                "components": {
                    "dashboard": {
                        "healthy": dashboard_healthy,
                        "last_update": self.dashboard.last_update.isoformat() if self.dashboard.last_update else None
                    },
                    "websockets": {
                        "healthy": websocket_healthy,
                        "active_connections": len(self.websocket_connections),
                        "max_connections": self.config['streaming']['max_connections']
                    },
                    "streaming": {
                        "enabled": True,
                        "update_interval_seconds": self.config['streaming']['update_interval_seconds']
                    }
                }
            }
        except Exception as e:
            logger.error(f"Failed to get API health: {e}")
            return {
                "success": False,
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _get_rate_limit_status(self, request: Request) -> Dict[str, Any]:
        """Get rate limit status for the requesting client."""
        try:
            # Extract client information
            client_info = {
                "remote_addr": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", ""),
                "client_id": request.headers.get("x-client-id", "")
            }
            
            rate_limiter = get_rate_limiter()
            client_id = rate_limiter.get_client_id(client_info)
            
            # Get status for all monitoring endpoints
            endpoint_statuses = {}
            for endpoint_path in ["/api/dashboard", "/api/dashboard/ascii", "/api/dashboard/system",
                                "/api/dashboard/services", "/api/dashboard/security", 
                                "/api/dashboard/refresh", "/api/dashboard/health",
                                "/api/dashboard/connections"]:
                endpoint_key = f"monitoring{endpoint_path.replace('/api/dashboard', '')}"
                if not endpoint_key.replace("monitoring", ""):
                    endpoint_key = "monitoring"
                
                if endpoint_key in rate_limiter.endpoint_limits:
                    status = rate_limiter.get_rate_limit_info(endpoint_key, client_id)
                    endpoint_statuses[endpoint_path] = status
            
            return {
                "success": True,
                "client_id": client_id,
                "timestamp": datetime.utcnow().isoformat(),
                "endpoint_statuses": endpoint_statuses,
                "rate_limiting": {
                    "enabled": True,
                    "global_burst_protection": True
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to get rate limit status: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _broadcast_to_websockets(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSocket clients."""
        if not self.websocket_connections:
            return
        
        # Create list of connections to avoid modification during iteration
        connections = list(self.websocket_connections)
        
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                # Remove failed connection
                self.websocket_connections.discard(websocket)

    async def run_monitoring_updates_stream(self):
        """Stream monitoring updates to WebSocket clients."""
        try:
            # Collect current metrics
            metrics = await self.dashboard.collect_all_metrics(force_refresh=True)
            
            # Broadcast metrics to all connected clients
            await self._broadcast_to_websockets({
                "type": "monitoring_update",
                "timestamp": datetime.utcnow().isoformat(),
                "data": metrics
            })
            
            logger.info("✅ Monitoring update streamed to clients")
            
        except Exception as e:
            logger.error(f"Monitoring streaming failed: {e}")
            
            # Broadcast error
            await self._broadcast_to_websockets({
                "type": "monitoring_error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            })

    async def shutdown(self):
        """Clean shutdown of dashboard API."""
        logger.info("📊 Shutting down DashboardAPI")
        
        # Close all WebSocket connections
        connections = list(self.websocket_connections)
        for websocket in connections:
            try:
                await websocket.close(code=1001, reason="Server shutdown")
            except Exception:
                pass
        
        self.websocket_connections.clear()
        
        # Shutdown dashboard
        await self.dashboard.shutdown()
        
        logger.info("✅ DashboardAPI shutdown complete")


# Utility function to integrate with existing FastAPI app
def setup_dashboard_api(app: FastAPI, config: Optional[Dict[str, Any]] = None) -> DashboardAPI:
    """
    Setup dashboard API with existing FastAPI application.
    
    Args:
        app: FastAPI application instance
        config: Optional configuration dictionary
        
    Returns:
        DashboardAPI instance
    """
    dashboard_api = DashboardAPI(app, config)
    
    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_dashboard():
        logger.info("🚀 Dashboard API startup complete")
    
    @app.on_event("shutdown")
    async def shutdown_dashboard():
        await dashboard_api.shutdown()
    
    return dashboard_api


# Export main components
__all__ = ["DashboardAPI", "setup_dashboard_api"]