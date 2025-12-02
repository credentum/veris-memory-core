#!/usr/bin/env python3
"""
Dashboard API Server Entry Point

Run the monitoring dashboard as a standalone service.
"""

import os
import sys
import logging
import uvicorn
from fastapi import FastAPI

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .dashboard_api import setup_dashboard_api

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Veris Memory Monitoring Dashboard",
        description="Real-time monitoring dashboard for Veris Memory system",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Setup dashboard API
    setup_dashboard_api(app)
    
    return app

def main():
    """Main entry point for dashboard server."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🎯 Starting Veris Memory Monitoring Dashboard")
    
    # Get configuration from environment
    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", "8080"))
    workers = int(os.getenv("DASHBOARD_WORKERS", "1"))
    
    # Create app
    app = create_app()
    
    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True,
        loop="asyncio"
    )

if __name__ == "__main__":
    main()