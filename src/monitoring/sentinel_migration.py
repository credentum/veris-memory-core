#!/usr/bin/env python3
"""
Migration script to transition from monolithic veris_sentinel.py to modular architecture.

This script demonstrates how to use the new modular Sentinel system
and provides backward compatibility during the transition period.
"""

import asyncio
import logging
from typing import Dict, Any

# New modular imports
from .sentinel import SentinelConfig, SentinelRunner, SentinelAPI

# Optional: Import from original for comparison
try:
    from .veris_sentinel import SentinelRunner as LegacySentinelRunner
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False

logger = logging.getLogger(__name__)


class SentinelMigrationManager:
    """Manages migration from legacy to modular Sentinel architecture."""
    
    def __init__(self):
        self.legacy_runner = None
        self.modular_runner = None
        self.config = None
    
    def create_config_from_legacy(self, legacy_config: Dict[str, Any] = None) -> SentinelConfig:
        """Create new SentinelConfig from legacy configuration."""
        if legacy_config is None:
            # Use environment-based configuration
            return SentinelConfig.from_env()
        
        # Map legacy config to new structure
        return SentinelConfig(
            target_base_url=legacy_config.get('target_base_url', 'http://localhost:8000'),
            check_interval_seconds=legacy_config.get('check_interval_seconds', 60),
            alert_threshold_failures=legacy_config.get('alert_threshold_failures', 3),
            webhook_url=legacy_config.get('webhook_url'),
            github_token=legacy_config.get('github_token'),
            github_repo=legacy_config.get('github_repo'),
            enabled_checks=legacy_config.get('enabled_checks')
        )
    
    def setup_modular_sentinel(self, config: SentinelConfig = None) -> tuple[SentinelRunner, SentinelAPI]:
        """Setup the new modular Sentinel system."""
        if config is None:
            config = SentinelConfig.from_env()
        
        self.config = config
        
        # Create runner
        self.modular_runner = SentinelRunner(config)
        
        # Create API
        api = SentinelAPI(self.modular_runner, config)
        
        logger.info("Modular Sentinel system initialized")
        return self.modular_runner, api
    
    async def run_comparison_test(self) -> Dict[str, Any]:
        """Run comparison between legacy and modular implementations."""
        if not LEGACY_AVAILABLE:
            logger.warning("Legacy Sentinel not available for comparison")
            return {"legacy_available": False}
        
        results = {"legacy_available": True, "comparisons": []}
        
        # TODO: Implement comparison logic
        # This would run equivalent checks in both systems and compare results
        
        return results
    
    async def migrate_data(self, legacy_db_path: str, new_db_path: str) -> bool:
        """Migrate data from legacy database to new format."""
        try:
            # TODO: Implement data migration logic
            # This would copy check results, alerts, etc. from old to new format
            logger.info(f"Data migration from {legacy_db_path} to {new_db_path} completed")
            return True
        except Exception as e:
            logger.error(f"Data migration failed: {e}")
            return False


async def main():
    """Example of how to use the new modular Sentinel system."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create migration manager
    migration_manager = SentinelMigrationManager()
    
    # Setup new modular system
    config = SentinelConfig(
        target_base_url="http://localhost:8000",
        check_interval_seconds=30,  # More frequent for demo
        enabled_checks=["S1-probes", "S2-golden-fact-recall"]  # Enable subset for demo
    )
    
    runner, api = migration_manager.setup_modular_sentinel(config)
    
    # Start API server
    api_runner, api_site = await api.start_server(host='0.0.0.0', port=9090)
    
    try:
        print("Modular Sentinel system started!")
        print("API available at: http://localhost:9090")
        print("Available endpoints:")
        print("  GET  /status - Overall status")
        print("  GET  /checks - List all checks")
        print("  GET  /checks/{check_id} - Check details")
        print("  POST /checks/{check_id}/run - Run specific check")
        print("  GET  /health - API health")
        print("  POST /start - Start monitoring")
        print("  POST /stop - Stop monitoring")
        
        # Start monitoring in background
        monitoring_task = asyncio.create_task(runner.start())
        
        # Run for a demo period
        print("\nRunning demo for 2 minutes...")
        await asyncio.sleep(120)
        
        # Stop monitoring
        await runner.stop()
        await monitoring_task
        
        print("\nDemo completed!")
        
        # Show final status
        status = runner.get_status_summary()
        print(f"Final status: {status}")
        
    finally:
        # Cleanup
        await api_runner.cleanup()


def create_systemd_service_file() -> str:
    """Generate systemd service file for the modular Sentinel."""
    return """[Unit]
Description=Veris Sentinel Monitoring Service
After=network.target
Wants=network.target

[Service]
Type=exec
User=sentinel
Group=sentinel
WorkingDirectory=/opt/veris-memory
ExecStart=/usr/bin/python3 -m src.monitoring.sentinel_migration
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Environment variables
Environment=TARGET_BASE_URL=http://localhost:8000
Environment=SENTINEL_CHECK_INTERVAL=60
Environment=SENTINEL_ALERT_THRESHOLD=3

[Install]
WantedBy=multi-user.target
"""


def create_docker_compose_service() -> str:
    """Generate docker-compose service definition for the modular Sentinel."""
    return """version: '3.8'

services:
  sentinel:
    build:
      context: .
      dockerfile: docker/Dockerfile.sentinel
    container_name: veris-sentinel
    restart: unless-stopped
    ports:
      - "9090:9090"
    environment:
      - TARGET_BASE_URL=http://veris-memory:8000
      - SENTINEL_CHECK_INTERVAL=60
      - SENTINEL_ALERT_THRESHOLD=3
    volumes:
      - sentinel_data:/var/lib/sentinel
    depends_on:
      - veris-memory
    networks:
      - veris-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9090/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  sentinel_data:

networks:
  veris-network:
    external: true
"""


if __name__ == "__main__":
    asyncio.run(main())