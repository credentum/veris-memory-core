"""
Metrics Collection System
Sprint 10 Phase 2 - Issue 006: SEC-106
Collects and aggregates security and performance metrics
"""

import time
import threading
import json
import os
import sys
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metrics"""
    name: str
    data_points: deque = field(default_factory=lambda: deque(maxlen=1000))
    tags: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: float, timestamp: Optional[datetime] = None):
        """Add a data point to the series"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.data_points.append(Metric(
            name=self.name,
            value=value,
            timestamp=timestamp,
            tags=self.tags.copy()
        ))
    
    def get_recent(self, minutes: int = 60) -> List[Metric]:
        """Get recent data points"""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [
            point for point in self.data_points
            if point.timestamp > cutoff
        ]
    
    def calculate_stats(self, minutes: int = 60) -> Dict[str, float]:
        """Calculate statistics for recent data"""
        recent_points = self.get_recent(minutes)
        
        if not recent_points:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "sum": 0}
        
        values = [point.value for point in recent_points]
        
        return {
            "count": len(values),
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
            "median": statistics.median(values),
            "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            "p99": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
        }


class MetricsCollector:
    """Collects and manages metrics"""
    
    def __init__(self, retention_minutes: int = 1440):  # 24 hours default
        """Initialize metrics collector"""
        self.retention_minutes = retention_minutes
        self.metrics = {}  # name -> MetricSeries
        self.counters = defaultdict(int)
        self.gauges = {}
        self.histograms = defaultdict(list)
        self.custom_collectors = {}  # name -> callable
        self.collection_thread = None
        self.running = False
        self.collection_interval = 60  # seconds
        
        # Built-in metric collectors
        self._setup_builtin_collectors()
    
    def _setup_builtin_collectors(self):
        """Setup built-in metric collectors"""
        self.add_custom_collector("system_cpu", self._collect_cpu_usage)
        self.add_custom_collector("system_memory", self._collect_memory_usage)
        self.add_custom_collector("system_disk", self._collect_disk_usage)
        self.add_custom_collector("request_rate", self._collect_request_rate)
        self.add_custom_collector("error_rate", self._collect_error_rate)
    
    def start_collection(self):
        """Start automatic metrics collection"""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop automatic metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                self._collect_all_metrics()
                self._cleanup_old_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
    
    def _collect_all_metrics(self):
        """Collect all registered metrics"""
        for name, collector in self.custom_collectors.items():
            try:
                value = collector()
                if value is not None:
                    self.record_gauge(name, value)
            except Exception as e:
                logger.error(f"Error collecting metric {name}: {e}")
    
    def _cleanup_old_metrics(self):
        """Remove old metric data points"""
        cutoff = datetime.utcnow() - timedelta(minutes=self.retention_minutes)
        
        for series in self.metrics.values():
            # Remove old points
            while series.data_points and series.data_points[0].timestamp < cutoff:
                series.data_points.popleft()
        
        # Clean up histograms
        for name, values in self.histograms.items():
            self.histograms[name] = [
                (timestamp, value) for timestamp, value in values
                if timestamp > cutoff
            ]
    
    def record_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Record a counter metric"""
        full_name = self._get_metric_name(name, tags)
        self.counters[full_name] += value
        
        # Also record as time series
        if full_name not in self.metrics:
            self.metrics[full_name] = MetricSeries(full_name, tags=tags or {})
        
        self.metrics[full_name].add_point(self.counters[full_name])
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric"""
        full_name = self._get_metric_name(name, tags)
        self.gauges[full_name] = value
        
        # Also record as time series
        if full_name not in self.metrics:
            self.metrics[full_name] = MetricSeries(full_name, tags=tags or {})
        
        self.metrics[full_name].add_point(value)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric"""
        full_name = self._get_metric_name(name, tags)
        timestamp = datetime.utcnow()
        self.histograms[full_name].append((timestamp, value))
        
        # Also record as time series
        if full_name not in self.metrics:
            self.metrics[full_name] = MetricSeries(full_name, tags=tags or {})
        
        self.metrics[full_name].add_point(value, timestamp)
    
    def _get_metric_name(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Generate full metric name with tags"""
        if not tags:
            return name
        
        tag_string = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_string}]"
    
    def get_metric_value(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current value of a metric"""
        full_name = self._get_metric_name(name, tags)
        
        # Try gauge first
        if full_name in self.gauges:
            return self.gauges[full_name]
        
        # Try counter
        if full_name in self.counters:
            return float(self.counters[full_name])
        
        # Try time series
        if full_name in self.metrics and self.metrics[full_name].data_points:
            return self.metrics[full_name].data_points[-1].value
        
        return None
    
    def get_metric_stats(self, name: str, minutes: int = 60, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get statistics for a metric"""
        full_name = self._get_metric_name(name, tags)
        
        if full_name in self.metrics:
            return self.metrics[full_name].calculate_stats(minutes)
        
        return {"count": 0, "avg": 0, "min": 0, "max": 0, "sum": 0}
    
    def get_histogram_percentiles(self, name: str, percentiles: List[float], minutes: int = 60, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Calculate percentiles for histogram data"""
        full_name = self._get_metric_name(name, tags)
        
        if full_name not in self.histograms:
            return {f"p{int(p*100)}": 0 for p in percentiles}
        
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent_values = [
            value for timestamp, value in self.histograms[full_name]
            if timestamp > cutoff
        ]
        
        if not recent_values:
            return {f"p{int(p*100)}": 0 for p in percentiles}
        
        recent_values.sort()
        result = {}
        
        for p in percentiles:
            index = int(len(recent_values) * p)
            if index >= len(recent_values):
                index = len(recent_values) - 1
            result[f"p{int(p*100)}"] = recent_values[index]
        
        return result
    
    def add_custom_collector(self, name: str, collector: Callable[[], Optional[float]]):
        """Add a custom metric collector"""
        self.custom_collectors[name] = collector
        logger.info(f"Added custom collector: {name}")
    
    def remove_custom_collector(self, name: str):
        """Remove a custom metric collector"""
        if name in self.custom_collectors:
            del self.custom_collectors[name]
            logger.info(f"Removed custom collector: {name}")
    
    def _collect_cpu_usage(self) -> Optional[float]:
        """Collect CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return None
    
    def _collect_memory_usage(self) -> Optional[float]:
        """Collect memory usage percentage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent
        except ImportError:
            return None
    
    def _collect_disk_usage(self) -> Optional[float]:
        """Collect disk usage percentage"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return (disk.used / disk.total) * 100
        except ImportError:
            return None
    
    def _collect_request_rate(self) -> Optional[float]:
        """Collect request rate (requests per minute)"""
        # This would typically integrate with web server metrics
        # For now, calculate from request counter
        if "http_requests" in self.metrics:
            stats = self.metrics["http_requests"].calculate_stats(1)  # 1 minute
            return stats.get("count", 0)
        return 0
    
    def _collect_error_rate(self) -> Optional[float]:
        """Collect error rate percentage"""
        # Calculate error rate from error and total request counters
        error_count = self.get_metric_stats("http_errors", 1).get("count", 0)
        total_count = self.get_metric_stats("http_requests", 1).get("count", 0)
        
        if total_count > 0:
            return (error_count / total_count) * 100
        return 0
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "metrics": {}
        }
        
        # Add time series data
        for name, series in self.metrics.items():
            data["metrics"][name] = {
                "current_value": series.data_points[-1].value if series.data_points else 0,
                "stats_1h": series.calculate_stats(60),
                "stats_24h": series.calculate_stats(1440),
                "tags": series.tags
            }
        
        if format == "json":
            return json.dumps(data, indent=2, default=str)
        elif format == "prometheus":
            return self._export_prometheus_format(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_prometheus_format(self, data: Dict[str, Any]) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Export counters
        for name, value in data["counters"].items():
            clean_name = name.replace("[", "_").replace("]", "_").replace("=", "_").replace(",", "_")
            lines.append(f"# TYPE {clean_name} counter")
            lines.append(f"{clean_name} {value}")
        
        # Export gauges
        for name, value in data["gauges"].items():
            clean_name = name.replace("[", "_").replace("]", "_").replace("=", "_").replace(",", "_")
            lines.append(f"# TYPE {clean_name} gauge")
            lines.append(f"{clean_name} {value}")
        
        return "\n".join(lines)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard display"""
        return {
            "overview": {
                "total_metrics": len(self.metrics),
                "total_counters": len(self.counters),
                "total_gauges": len(self.gauges),
                "collection_interval": self.collection_interval,
                "retention_minutes": self.retention_minutes
            },
            "system_metrics": {
                "cpu_usage": self.get_metric_value("system_cpu"),
                "memory_usage": self.get_metric_value("system_memory"),
                "disk_usage": self.get_metric_value("system_disk")
            },
            "application_metrics": {
                "request_rate": self.get_metric_value("request_rate"),
                "error_rate": self.get_metric_value("error_rate"),
                "active_connections": self.get_metric_value("active_connections", {"type": "http"})
            },
            "security_metrics": {
                "blocked_requests": self.get_metric_value("waf_blocked_requests"),
                "failed_logins": self.get_metric_value("auth_failures"),
                "rate_limited": self.get_metric_value("rate_limited_requests")
            }
        }


class HealthChecker:
    """Health check system"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize health checker"""
        self.metrics = metrics_collector
        self.checks = {}  # name -> check function
        self.health_status = {}  # name -> (status, message, timestamp)
        
        # Setup default health checks
        self._setup_default_checks()
    
    def _setup_default_checks(self):
        """Setup default health checks"""
        self.add_check("cpu_usage", self._check_cpu_usage)
        self.add_check("memory_usage", self._check_memory_usage)
        self.add_check("disk_usage", self._check_disk_usage)
        self.add_check("error_rate", self._check_error_rate)
    
    def add_check(self, name: str, check_func: Callable[[], tuple]):
        """Add a health check"""
        self.checks[name] = check_func
        logger.info(f"Added health check: {name}")
    
    def remove_check(self, name: str):
        """Remove a health check"""
        if name in self.checks:
            del self.checks[name]
            if name in self.health_status:
                del self.health_status[name]
            logger.info(f"Removed health check: {name}")
    
    def run_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all health checks"""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                status, message = check_func()
                self.health_status[name] = (status, message, datetime.utcnow())
                results[name] = {
                    "status": status,
                    "message": message,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                error_msg = f"Health check failed: {e}"
                self.health_status[name] = ("error", error_msg, datetime.utcnow())
                results[name] = {
                    "status": "error",
                    "message": error_msg,
                    "timestamp": datetime.utcnow().isoformat()
                }
                logger.error(f"Health check {name} failed: {e}")
        
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        if not self.health_status:
            return {"status": "unknown", "message": "No health checks configured"}
        
        statuses = [status for status, _, _ in self.health_status.values()]
        
        if "critical" in statuses:
            overall_status = "critical"
        elif "warning" in statuses:
            overall_status = "warning"
        elif "error" in statuses:
            overall_status = "error"
        elif all(s == "healthy" for s in statuses):
            overall_status = "healthy"
        else:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": len(self.checks),
            "details": {
                name: {"status": status, "message": message, "timestamp": timestamp.isoformat()}
                for name, (status, message, timestamp) in self.health_status.items()
            }
        }
    
    def _check_cpu_usage(self) -> tuple:
        """Check CPU usage"""
        cpu_usage = self.metrics.get_metric_value("system_cpu")
        
        if cpu_usage is None:
            return ("error", "CPU metrics not available")
        
        if cpu_usage > 90:
            return ("critical", f"CPU usage critical: {cpu_usage:.1f}%")
        elif cpu_usage > 80:
            return ("warning", f"CPU usage high: {cpu_usage:.1f}%")
        else:
            return ("healthy", f"CPU usage normal: {cpu_usage:.1f}%")
    
    def _check_memory_usage(self) -> tuple:
        """Check memory usage"""
        memory_usage = self.metrics.get_metric_value("system_memory")
        
        if memory_usage is None:
            return ("error", "Memory metrics not available")
        
        if memory_usage > 95:
            return ("critical", f"Memory usage critical: {memory_usage:.1f}%")
        elif memory_usage > 85:
            return ("warning", f"Memory usage high: {memory_usage:.1f}%")
        else:
            return ("healthy", f"Memory usage normal: {memory_usage:.1f}%")
    
    def _check_disk_usage(self) -> tuple:
        """Check disk usage"""
        disk_usage = self.metrics.get_metric_value("system_disk")
        
        if disk_usage is None:
            return ("error", "Disk metrics not available")
        
        if disk_usage > 95:
            return ("critical", f"Disk usage critical: {disk_usage:.1f}%")
        elif disk_usage > 85:
            return ("warning", f"Disk usage high: {disk_usage:.1f}%")
        else:
            return ("healthy", f"Disk usage normal: {disk_usage:.1f}%")
    
    def _check_error_rate(self) -> tuple:
        """Check application error rate"""
        error_rate = self.metrics.get_metric_value("error_rate")
        
        if error_rate is None:
            return ("error", "Error rate metrics not available")
        
        if error_rate > 10:
            return ("critical", f"Error rate critical: {error_rate:.1f}%")
        elif error_rate > 5:
            return ("warning", f"Error rate high: {error_rate:.1f}%")
        else:
            return ("healthy", f"Error rate normal: {error_rate:.1f}%")


# Export main components
__all__ = [
    "MetricsCollector",
    "MetricSeries",
    "Metric",
    "HealthChecker",
]