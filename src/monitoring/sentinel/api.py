#!/usr/bin/env python3
"""
Sentinel API - Web interface for Sentinel monitoring.

Provides HTTP endpoints for accessing Sentinel status, results,
and controlling the monitoring system.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import aiohttp_cors
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

from aiohttp import web
from aiohttp.web import Request, Response

from .models import SentinelConfig
from .runner import SentinelRunner

logger = logging.getLogger(__name__)


class SentinelAPI:
    """Web API for Sentinel monitoring system."""

    def __init__(self, runner: SentinelRunner, config: SentinelConfig):
        self.runner = runner
        self.config = config
        self.app = web.Application()
        # Storage for host-based check results (e.g., firewall status from host script)
        self._host_check_results: Dict[str, Any] = {}
        self._setup_routes()
        self._setup_cors()
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        self.app.router.add_get('/status', self.get_status)
        self.app.router.add_get('/checks', self.get_checks)
        self.app.router.add_get('/checks/{check_id}', self.get_check_detail)
        self.app.router.add_get('/checks/{check_id}/history', self.get_check_history)
        self.app.router.add_post('/checks/{check_id}/run', self.run_check)
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_post('/start', self.start_monitoring)
        self.app.router.add_post('/stop', self.stop_monitoring)
        # Host-based check result submission endpoint (for S11, etc.)
        self.app.router.add_post('/host-checks/{check_id}', self.submit_host_check_result)
    
    def _setup_cors(self) -> None:
        """Setup CORS for API access."""
        if not CORS_AVAILABLE:
            logger.warning("aiohttp_cors not available, CORS not enabled")
            return
            
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def get_status(self, request: Request) -> Response:
        """Get overall Sentinel status."""
        try:
            status = self.runner.get_status_summary()
            return web.json_response({
                'success': True,
                'data': status,
                'timestamp': datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Status endpoint error: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }, status=500)
    
    async def get_checks(self, request: Request) -> Response:
        """Get list of all checks and their status."""
        try:
            checks_info = {}
            
            for check_id, check_instance in self.runner.checks.items():
                stats = check_instance.get_statistics()
                checks_info[check_id] = {
                    'enabled': check_instance.is_enabled(),
                    'description': check_instance.description,
                    'statistics': stats
                }
            
            return web.json_response({
                'success': True,
                'data': {
                    'checks': checks_info,
                    'total_count': len(checks_info),
                    'enabled_count': sum(1 for info in checks_info.values() if info['enabled'])
                },
                'timestamp': datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Checks endpoint error: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }, status=500)
    
    async def get_check_detail(self, request: Request) -> Response:
        """Get detailed information about a specific check."""
        check_id = request.match_info['check_id']
        
        if check_id not in self.runner.checks:
            return web.json_response({
                'success': False,
                'error': f'Check {check_id} not found',
                'timestamp': datetime.utcnow().isoformat()
            }, status=404)
        
        try:
            check_instance = self.runner.checks[check_id]
            stats = check_instance.get_statistics()
            
            return web.json_response({
                'success': True,
                'data': {
                    'check_id': check_id,
                    'enabled': check_instance.is_enabled(),
                    'description': check_instance.description,
                    'statistics': stats,
                    'last_result': check_instance.last_result.to_dict() if check_instance.last_result else None
                },
                'timestamp': datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Check detail endpoint error: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }, status=500)
    
    async def get_check_history(self, request: Request) -> Response:
        """Get history for a specific check."""
        check_id = request.match_info['check_id']
        limit = int(request.query.get('limit', 50))
        
        if check_id not in self.runner.checks:
            return web.json_response({
                'success': False,
                'error': f'Check {check_id} not found',
                'timestamp': datetime.utcnow().isoformat()
            }, status=404)
        
        try:
            history = self.runner.get_check_history(check_id, limit)
            
            return web.json_response({
                'success': True,
                'data': {
                    'check_id': check_id,
                    'history': history,
                    'count': len(history)
                },
                'timestamp': datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Check history endpoint error: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }, status=500)
    
    async def run_check(self, request: Request) -> Response:
        """Manually run a specific check."""
        check_id = request.match_info['check_id']
        
        if check_id not in self.runner.checks:
            return web.json_response({
                'success': False,
                'error': f'Check {check_id} not found',
                'timestamp': datetime.utcnow().isoformat()
            }, status=404)
        
        try:
            check_instance = self.runner.checks[check_id]
            result = await check_instance.execute()
            
            # Process the result
            await self.runner._process_check_result(result)
            
            return web.json_response({
                'success': True,
                'data': {
                    'check_id': check_id,
                    'result': result.to_dict()
                },
                'timestamp': datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Run check endpoint error: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }, status=500)
    
    async def health_check(self, request: Request) -> Response:
        """Simple health check for the API itself."""
        return web.json_response({
            'success': True,
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'sentinel-api'
        })
    
    async def start_monitoring(self, request: Request) -> Response:
        """Start the Sentinel monitoring loop."""
        if self.runner.running:
            return web.json_response({
                'success': False,
                'error': 'Monitoring is already running',
                'timestamp': datetime.utcnow().isoformat()
            }, status=400)
        
        try:
            # Start monitoring in background task
            asyncio.create_task(self.runner.start())
            
            return web.json_response({
                'success': True,
                'message': 'Monitoring started',
                'timestamp': datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Start monitoring error: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }, status=500)
    
    async def stop_monitoring(self, request: Request) -> Response:
        """Stop the Sentinel monitoring loop."""
        if not self.runner.running:
            return web.json_response({
                'success': False,
                'error': 'Monitoring is not running',
                'timestamp': datetime.utcnow().isoformat()
            }, status=400)
        
        try:
            await self.runner.stop()
            
            return web.json_response({
                'success': True,
                'message': 'Monitoring stopped',
                'timestamp': datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Stop monitoring error: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }, status=500)
    
    def get_host_check_result(self, check_id: str) -> Optional[Any]:
        """
        Get the most recent host-based check result for a given check ID.

        Used by checks like S11 (firewall) that rely on host-based monitoring
        scripts to submit results via the API.

        Args:
            check_id: The check ID to retrieve results for (e.g., "S11-firewall-status")

        Returns:
            CheckResult object if available, None otherwise
        """
        return self._host_check_results.get(check_id)

    async def submit_host_check_result(self, request: Request) -> Response:
        """
        Submit a host-based check result from host monitoring script.

        Endpoint: POST /host-checks/{check_id}
        Body: {
            "status": "pass|warn|fail",
            "message": "Status message",
            "details": {...}
        }
        """
        check_id = request.match_info['check_id']

        try:
            data = await request.json()

            # Import CheckResult here to avoid circular imports
            from .models import CheckResult

            # Create CheckResult from submitted data
            result = CheckResult(
                check_id=check_id,
                timestamp=datetime.utcnow(),
                status=data.get('status', 'unknown'),
                latency_ms=0.0,  # Not applicable for host-based checks
                message=data.get('message', 'Host-based check result'),
                details=data.get('details', {})
            )

            # Store the result
            self._host_check_results[check_id] = result

            logger.info(f"Received host check result for {check_id}: {result.status}")

            return web.json_response({
                'success': True,
                'message': f'Host check result stored for {check_id}',
                'timestamp': datetime.utcnow().isoformat()
            })

        except Exception as e:
            logger.error(f"Submit host check error: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }, status=500)

    def get_app(self) -> web.Application:
        """Get the aiohttp application."""
        return self.app
    
    async def start_server(self, host: str = '0.0.0.0', port: int = 9090) -> None:
        """Start the API server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        logger.info(f"Sentinel API server started on {host}:{port}")
        
        return runner, site