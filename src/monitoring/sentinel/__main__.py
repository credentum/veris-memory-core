#!/usr/bin/env python3
"""
Sentinel Monitoring Service Entry Point

Run the Sentinel monitoring system as a standalone service.
"""

import asyncio
import os
import sys
import argparse
import logging

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.monitoring.sentinel.runner import SentinelRunner
from src.monitoring.sentinel.models import SentinelConfig
from src.monitoring.sentinel.api import SentinelAPI
from aiohttp import web


def setup_logging():
    """Configure logging for Sentinel."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


async def main():
    """Main entry point for Sentinel monitoring service."""
    parser = argparse.ArgumentParser(description="Veris Sentinel Monitoring Service")
    parser.add_argument("--standalone", action="store_true", help="Run in standalone mode")
    parser.add_argument("--api-port", type=int, default=9090, help="API server port")
    parser.add_argument("--no-api", action="store_true", help="Disable API server")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("🚀 Starting Veris Sentinel Monitoring Service")
    
    # Create configuration from environment
    config = SentinelConfig.from_env()
    
    # Create runner
    runner = SentinelRunner(config)
    
    try:
        # Start API server if not disabled
        if not args.no_api:
            api = SentinelAPI(runner, config)

            # Configure S11 with API instance for host-based monitoring
            # This solves the initialization order problem where checks are created
            # before the API exists (fixes issue #280)
            if "S11-firewall-status" in runner.checks:
                runner.checks["S11-firewall-status"].set_api_instance(api)
                logger.info("✅ S11 configured for host-based firewall monitoring")

            app_runner = web.AppRunner(api.app)
            await app_runner.setup()
            site = web.TCPSite(app_runner, '0.0.0.0', args.api_port)
            await site.start()
            logger.info(f"✅ API server started on port {args.api_port}")

        # Start monitoring
        logger.info("✅ Starting monitoring checks")
        await runner.start()
        
    except KeyboardInterrupt:
        logger.info("⚠️ Received shutdown signal")
    except Exception as e:
        logger.error(f"❌ Sentinel failed: {e}")
        sys.exit(1)
    finally:
        logger.info("👋 Sentinel shutting down")


if __name__ == "__main__":
    asyncio.run(main())