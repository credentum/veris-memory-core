#!/usr/bin/env python3
"""
Metrics Streaming System

Handles real-time streaming of metrics data to WebSocket clients
with efficient updates and bandwidth optimization.
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, asdict

# Import dashboard with fallback handling
try:
    from .dashboard import UnifiedDashboard
except ImportError:
    # Fallback for testing
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.monitoring.dashboard import UnifiedDashboard

logger = logging.getLogger(__name__)


@dataclass
class StreamingMetrics:
    """Metrics about the streaming system itself."""
    active_connections: int
    messages_sent_per_second: float
    bandwidth_usage_kbps: float
    error_rate_percent: float
    last_update: datetime


class MetricsStreamer:
    """
    High-performance metrics streaming system for real-time dashboard updates.
    
    Features:
    - Efficient delta updates to reduce bandwidth
    - Adaptive update rates based on change frequency
    - Client-specific filtering and subscriptions
    - Streaming health monitoring
    """

    def __init__(self, dashboard: UnifiedDashboard, config: Optional[Dict[str, Any]] = None):
        """Initialize metrics streamer."""
        self.dashboard = dashboard
        self.config = config or self._get_default_config()
        
        # Streaming state
        self.is_streaming = False
        self.last_metrics_hash = None
        self.last_full_metrics = None
        self.update_count = 0
        self.start_time = datetime.utcnow()
        
        # Performance tracking
        self.messages_sent = 0
        self.bytes_sent = 0
        self.errors_count = 0
        
        # Client subscriptions (future feature)
        self.client_subscriptions: Dict[str, Set[str]] = {}
        
        logger.info("📡 MetricsStreamer initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default streaming configuration."""
        return {
            'update_interval_seconds': 5,
            'adaptive_updates': True,
            'delta_compression': True,
            'max_unchanged_updates': 3,
            'metrics_retention_minutes': 60,
            'bandwidth_limit_kbps': 1000,
            'client_filters': {
                'system_only': ['system'],
                'services_only': ['services'], 
                'security_only': ['security'],
                'monitoring_only': ['system', 'services']
            }
        }

    async def start_streaming(self):
        """Start the metrics streaming loop."""
        if self.is_streaming:
            logger.warning("Streaming already active")
            return
        
        self.is_streaming = True
        self.start_time = datetime.utcnow()
        logger.info("🚀 Starting metrics streaming")
        
        try:
            await self._streaming_loop()
        except Exception as e:
            logger.error(f"Streaming loop failed: {e}")
        finally:
            self.is_streaming = False

    async def stop_streaming(self):
        """Stop the metrics streaming loop."""
        self.is_streaming = False
        logger.info("⏹️ Stopped metrics streaming")

    async def _streaming_loop(self):
        """Main streaming loop with adaptive updates."""
        unchanged_count = 0
        base_interval = self.config['update_interval_seconds']
        
        while self.is_streaming:
            try:
                # Collect current metrics
                current_metrics = await self.dashboard.collect_all_metrics()
                
                # Calculate metrics hash for change detection
                metrics_json = json.dumps(current_metrics, sort_keys=True, default=str)
                current_hash = hashlib.md5(metrics_json.encode()).hexdigest()
                
                # Check if metrics changed
                metrics_changed = current_hash != self.last_metrics_hash
                
                if metrics_changed:
                    # Reset unchanged counter
                    unchanged_count = 0
                    
                    # Generate update message
                    update_message = self._create_update_message(current_metrics)
                    
                    # Stream to clients (would be handled by DashboardAPI)
                    self._track_message_sent(update_message)
                    
                    # Update state
                    self.last_metrics_hash = current_hash
                    self.last_full_metrics = current_metrics
                    
                    logger.debug(f"📡 Streamed metrics update #{self.update_count}")
                    
                else:
                    unchanged_count += 1
                    
                    # Send heartbeat if no changes for a while
                    if unchanged_count >= self.config['max_unchanged_updates']:
                        heartbeat_message = self._create_heartbeat_message()
                        self._track_message_sent(heartbeat_message)
                        unchanged_count = 0  # Reset after heartbeat

                # Adaptive update interval
                if self.config['adaptive_updates']:
                    interval = self._calculate_adaptive_interval(metrics_changed, unchanged_count)
                else:
                    interval = base_interval
                
                # Wait for next update
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.errors_count += 1
                logger.error(f"Streaming iteration failed: {e}")
                await asyncio.sleep(base_interval)  # Continue with base interval on error

    def _create_update_message(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create streaming update message."""
        self.update_count += 1
        
        message = {
            "type": "metrics_update",
            "timestamp": datetime.utcnow().isoformat(),
            "update_id": self.update_count,
            "data": metrics
        }
        
        # Add delta information if delta compression is enabled
        if self.config.get('delta_compression', False) and self.last_full_metrics:
            delta = self._calculate_metrics_delta(self.last_full_metrics, metrics)
            if delta:
                message["delta"] = delta
                message["compression"] = "delta"
        
        return message

    def _create_heartbeat_message(self) -> Dict[str, Any]:
        """Create heartbeat message."""
        return {
            "type": "heartbeat",
            "timestamp": datetime.utcnow().isoformat(),
            "streaming_stats": self._get_streaming_stats()
        }

    def _calculate_metrics_delta(self, old_metrics: Dict[str, Any], 
                                new_metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate delta between metric sets."""
        # This is a simplified delta calculation
        # In production, would use more sophisticated diffing
        
        delta = {}
        
        # Check system metrics changes
        if 'system' in old_metrics and 'system' in new_metrics:
            old_system = old_metrics['system']
            new_system = new_metrics['system']
            
            system_delta = {}
            for key in ['cpu_percent', 'memory_percent', 'disk_percent']:
                if key in old_system and key in new_system:
                    if abs(old_system[key] - new_system[key]) > 0.1:  # Threshold for changes
                        system_delta[key] = new_system[key]
            
            if system_delta:
                delta['system'] = system_delta
        
        # Check service status changes
        if 'services' in old_metrics and 'services' in new_metrics:
            old_services = {s['name']: s for s in old_metrics['services']}
            new_services = {s['name']: s for s in new_metrics['services']}
            
            service_deltas = []
            for name, new_service in new_services.items():
                if name in old_services:
                    old_service = old_services[name]
                    if old_service['status'] != new_service['status']:
                        service_deltas.append({
                            'name': name,
                            'status': new_service['status'],
                            'changed': True
                        })
            
            if service_deltas:
                delta['services'] = service_deltas
        
        return delta if delta else None

    def _calculate_adaptive_interval(self, metrics_changed: bool, unchanged_count: int) -> float:
        """Calculate adaptive update interval based on change frequency."""
        base_interval = self.config['update_interval_seconds']
        
        if metrics_changed:
            # Faster updates when metrics are changing
            return max(base_interval * 0.5, 1.0)
        elif unchanged_count > 5:
            # Slower updates when metrics are stable
            return min(base_interval * 2.0, 30.0)
        else:
            return base_interval

    def _track_message_sent(self, message: Dict[str, Any]):
        """Track message statistics for performance monitoring."""
        self.messages_sent += 1
        
        # Calculate message size
        message_json = json.dumps(message, default=str)
        message_bytes = len(message_json.encode('utf-8'))
        self.bytes_sent += message_bytes

    def _get_streaming_stats(self) -> Dict[str, Any]:
        """Get current streaming statistics."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Calculate rates
        messages_per_second = self.messages_sent / uptime if uptime > 0 else 0
        bandwidth_kbps = (self.bytes_sent / 1024) / uptime if uptime > 0 else 0
        error_rate = (self.errors_count / self.messages_sent * 100) if self.messages_sent > 0 else 0
        
        return {
            'uptime_seconds': uptime,
            'messages_sent': self.messages_sent,
            'bytes_sent': self.bytes_sent,
            'errors_count': self.errors_count,
            'messages_per_second': round(messages_per_second, 2),
            'bandwidth_kbps': round(bandwidth_kbps, 2),
            'error_rate_percent': round(error_rate, 2),
            'update_count': self.update_count
        }

    def get_streaming_metrics(self) -> StreamingMetrics:
        """Get streaming metrics as structured object."""
        stats = self._get_streaming_stats()
        
        return StreamingMetrics(
            active_connections=0,  # Would be provided by DashboardAPI
            messages_sent_per_second=stats['messages_per_second'],
            bandwidth_usage_kbps=stats['bandwidth_kbps'],
            error_rate_percent=stats['error_rate_percent'],
            last_update=datetime.utcnow()
        )

    def create_filtered_update(self, metrics: Dict[str, Any], 
                             filter_name: str) -> Optional[Dict[str, Any]]:
        """
        Create filtered update message for specific client subscriptions.
        
        Args:
            metrics: Complete metrics dictionary
            filter_name: Name of the filter to apply
            
        Returns:
            Filtered metrics update or None if filter doesn't exist
        """
        client_filters = self.config.get('client_filters', {})
        
        if filter_name not in client_filters:
            return None
        
        # Get allowed sections for this filter
        allowed_sections = client_filters[filter_name]
        
        # Filter metrics to only include allowed sections
        filtered_metrics = {}
        for section in allowed_sections:
            if section in metrics:
                filtered_metrics[section] = metrics[section]
        
        # Add metadata
        filtered_metrics['timestamp'] = metrics.get('timestamp')
        filtered_metrics['filter_applied'] = filter_name
        
        return {
            "type": "filtered_update",
            "timestamp": datetime.utcnow().isoformat(),
            "filter": filter_name,
            "data": filtered_metrics
        }

    async def stream_monitoring_update(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stream monitoring metrics update in real-time.
        
        Args:
            metrics: Current metrics dictionary
            
        Returns:
            Streaming message for monitoring update
        """
        message = {
            "type": "monitoring_stream",
            "timestamp": datetime.utcnow().isoformat(),
            "data": metrics
        }
        
        self._track_message_sent(message)
        
        logger.info("📡 Streamed monitoring update")
        
        return message

    def create_custom_dashboard(self, sections: List[str], 
                              format_type: str = "json") -> Dict[str, Any]:
        """
        Create custom dashboard with specific sections.
        
        Args:
            sections: List of metric sections to include
            format_type: Output format ('json' or 'summary')
            
        Returns:
            Custom dashboard message
        """
        # This would integrate with the main dashboard to create custom views
        custom_config = {
            'sections': sections,
            'format': format_type,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return {
            "type": "custom_dashboard",
            "config": custom_config,
            "message": "Custom dashboard configuration applied"
        }

    def reset_streaming_stats(self):
        """Reset streaming statistics (useful for long-running streams)."""
        self.messages_sent = 0
        self.bytes_sent = 0
        self.errors_count = 0
        self.update_count = 0
        self.start_time = datetime.utcnow()
        
        logger.info("📊 Streaming statistics reset")


class StreamingHealthMonitor:
    """Monitor streaming system health and performance."""
    
    def __init__(self, streamer: MetricsStreamer):
        """Initialize streaming health monitor."""
        self.streamer = streamer
        self.health_history: List[StreamingMetrics] = []
        self.max_history = 100
    
    def check_streaming_health(self) -> Dict[str, Any]:
        """Check overall streaming system health."""
        current_metrics = self.streamer.get_streaming_metrics()
        
        # Add to history
        self.health_history.append(current_metrics)
        if len(self.health_history) > self.max_history:
            self.health_history.pop(0)
        
        # Analyze health
        health_status = "healthy"
        warnings = []
        
        # Check error rate
        if current_metrics.error_rate_percent > 5.0:
            health_status = "degraded"
            warnings.append(f"High error rate: {current_metrics.error_rate_percent:.1f}%")
        
        # Check bandwidth usage
        if current_metrics.bandwidth_usage_kbps > 500:  # Arbitrary threshold
            health_status = "warning"
            warnings.append(f"High bandwidth usage: {current_metrics.bandwidth_usage_kbps:.1f} kbps")
        
        return {
            "health_status": health_status,
            "current_metrics": asdict(current_metrics),
            "warnings": warnings,
            "history_samples": len(self.health_history),
            "timestamp": datetime.utcnow().isoformat()
        }


# Export main components
__all__ = ["MetricsStreamer", "StreamingMetrics", "StreamingHealthMonitor"]