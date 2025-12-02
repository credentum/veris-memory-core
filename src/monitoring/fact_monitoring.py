"""
Comprehensive monitoring dashboards and alerting for fact retrieval system.

This module provides real-time monitoring, alerting, and health checks for all
fact-related operations with sophisticated alerting rules and dashboard generation.
"""

import time
import json
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import logging
import statistics
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics being monitored."""
    COUNTER = "counter"        # Monotonically increasing
    GAUGE = "gauge"           # Current value
    HISTOGRAM = "histogram"   # Distribution of values
    SUMMARY = "summary"       # Statistical summary


class AlertCondition(Enum):
    """Alert condition types."""
    THRESHOLD_ABOVE = "threshold_above"
    THRESHOLD_BELOW = "threshold_below"
    RATE_CHANGE = "rate_change"
    ANOMALY_DETECTION = "anomaly_detection"
    MISSING_DATA = "missing_data"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str]
    metadata: Dict[str, Any]


@dataclass
class AlertRule:
    """Configuration for monitoring alert."""
    rule_name: str
    metric_name: str
    condition: AlertCondition
    threshold: float
    severity: AlertSeverity
    window_minutes: int
    cooldown_minutes: int
    labels_filter: Dict[str, str]
    description: str
    runbook_url: str
    enabled: bool
    created_at: float


@dataclass
class Alert:
    """Active alert instance."""
    alert_id: str
    rule_name: str
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold: float
    description: str
    fired_at: float
    resolved_at: Optional[float]
    status: str
    labels: Dict[str, str]
    annotations: Dict[str, str]


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    widget_id: str
    widget_type: str
    title: str
    metric_query: str
    refresh_interval: int
    chart_config: Dict[str, Any]
    position: Dict[str, int]


@dataclass
class Dashboard:
    """Monitoring dashboard configuration."""
    dashboard_id: str
    title: str
    description: str
    widgets: List[DashboardWidget]
    refresh_interval: int
    created_at: float
    updated_at: float


class MetricCollector:
    """Thread-safe metric collection and storage."""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.retention_seconds = retention_hours * 3600
        
        # Metric storage: metric_name -> List[MetricPoint]
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Aggregated statistics cache
        self.stats_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 60  # 1 minute cache TTL
        self.last_cache_update = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background cleanup
        self.cleanup_interval = 3600  # 1 hour
        self.last_cleanup = time.time()
    
    def record_metric(self, metric_name: str, value: float,
                     labels: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a metric data point."""
        current_time = time.time()
        
        point = MetricPoint(
            timestamp=current_time,
            value=value,
            labels=labels or {},
            metadata=metadata or {}
        )
        
        with self.lock:
            self.metrics[metric_name].append(point)
            
            # Periodic cleanup
            if current_time - self.last_cleanup > self.cleanup_interval:
                self._cleanup_old_metrics()
                self.last_cleanup = current_time
        
        # Invalidate cache
        self.last_cache_update = 0
    
    def get_metric_points(self, metric_name: str, 
                         hours: float = 1.0,
                         labels_filter: Optional[Dict[str, str]] = None) -> List[MetricPoint]:
        """Get metric points for time range with optional label filtering."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            if metric_name not in self.metrics:
                return []
            
            points = []
            for point in self.metrics[metric_name]:
                if point.timestamp >= cutoff_time:
                    # Apply label filtering
                    if labels_filter:
                        if all(point.labels.get(k) == v for k, v in labels_filter.items()):
                            points.append(point)
                    else:
                        points.append(point)
            
            return points
    
    def get_metric_stats(self, metric_name: str, 
                        hours: float = 1.0,
                        labels_filter: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get statistical summary of metric."""
        cache_key = f"{metric_name}:{hours}:{json.dumps(labels_filter, sort_keys=True)}"
        
        # Check cache
        current_time = time.time()
        if (current_time - self.last_cache_update < self.cache_ttl and 
            cache_key in self.stats_cache):
            return self.stats_cache[cache_key]
        
        points = self.get_metric_points(metric_name, hours, labels_filter)
        
        if not points:
            return {"error": "No data points found"}
        
        values = [point.value for point in points]
        
        stats = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            "p99": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values),
            "first_timestamp": points[0].timestamp,
            "last_timestamp": points[-1].timestamp,
            "time_range_hours": hours
        }
        
        # Cache result
        self.stats_cache[cache_key] = stats
        self.last_cache_update = current_time
        
        return stats
    
    def get_all_metric_names(self) -> List[str]:
        """Get list of all metric names."""
        with self.lock:
            return list(self.metrics.keys())
    
    def _cleanup_old_metrics(self) -> None:
        """Remove old metric points to manage memory."""
        cutoff_time = time.time() - self.retention_seconds
        
        for metric_name in self.metrics:
            # Remove old points
            while (self.metrics[metric_name] and 
                   self.metrics[metric_name][0].timestamp < cutoff_time):
                self.metrics[metric_name].popleft()


class AlertManager:
    """Alert rule evaluation and notification management."""
    
    def __init__(self, metric_collector: MetricCollector):
        self.metric_collector = metric_collector
        
        # Alert rules and active alerts
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Alert evaluation state
        self.last_evaluation = 0
        self.evaluation_interval = 60  # 1 minute
        self.alert_cooldowns: Dict[str, float] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Alert handlers
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Initialize default alert rules
        self._initialize_default_alerts()
    
    def _initialize_default_alerts(self) -> None:
        """Initialize default alert rules for fact system monitoring."""
        default_alerts = [
            {
                'rule_name': 'high_error_rate',
                'metric_name': 'fact_operation_errors_total',
                'condition': AlertCondition.THRESHOLD_ABOVE,
                'threshold': 10.0,
                'severity': AlertSeverity.WARNING,
                'window_minutes': 5,
                'description': 'High error rate in fact operations'
            },
            {
                'rule_name': 'critical_error_rate',
                'metric_name': 'fact_operation_errors_total',
                'condition': AlertCondition.THRESHOLD_ABOVE,
                'threshold': 50.0,
                'severity': AlertSeverity.CRITICAL,
                'window_minutes': 5,
                'description': 'Critical error rate in fact operations'
            },
            {
                'rule_name': 'high_latency',
                'metric_name': 'fact_operation_duration_ms',
                'condition': AlertCondition.THRESHOLD_ABOVE,
                'threshold': 1000.0,
                'severity': AlertSeverity.WARNING,
                'window_minutes': 10,
                'description': 'High latency in fact operations (p95 > 1s)'
            },
            {
                'rule_name': 'low_success_rate',
                'metric_name': 'fact_operation_success_rate',
                'condition': AlertCondition.THRESHOLD_BELOW,
                'threshold': 0.95,
                'severity': AlertSeverity.WARNING,
                'window_minutes': 10,
                'description': 'Low success rate in fact operations'
            },
            {
                'rule_name': 'no_operations',
                'metric_name': 'fact_operations_total',
                'condition': AlertCondition.MISSING_DATA,
                'threshold': 0.0,
                'severity': AlertSeverity.WARNING,
                'window_minutes': 30,
                'description': 'No fact operations detected'
            }
        ]
        
        current_time = time.time()
        
        for alert_config in default_alerts:
            rule = AlertRule(
                rule_name=alert_config['rule_name'],
                metric_name=alert_config['metric_name'],
                condition=alert_config['condition'],
                threshold=alert_config['threshold'],
                severity=alert_config['severity'],
                window_minutes=alert_config['window_minutes'],
                cooldown_minutes=alert_config.get('cooldown_minutes', 15),
                labels_filter={},
                description=alert_config['description'],
                runbook_url="",
                enabled=True,
                created_at=current_time
            )
            self.alert_rules[rule.rule_name] = rule
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        with self.lock:
            self.alert_rules[rule.rule_name] = rule
        
        logger.info(f"Added alert rule: {rule.rule_name}")
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove an alert rule."""
        with self.lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                
                # Resolve any active alerts for this rule
                alerts_to_resolve = [
                    alert_id for alert_id, alert in self.active_alerts.items()
                    if alert.rule_name == rule_name
                ]
                for alert_id in alerts_to_resolve:
                    self._resolve_alert(alert_id, "Rule removed")
                
                return True
            return False
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert notification handler."""
        self.alert_handlers.append(handler)
    
    def evaluate_alerts(self) -> List[Alert]:
        """Evaluate all alert rules and fire/resolve alerts."""
        current_time = time.time()
        
        # Rate limit evaluation
        if current_time - self.last_evaluation < self.evaluation_interval:
            return []
        
        fired_alerts = []
        
        with self.lock:
            for rule_name, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                # Check cooldown
                if (rule_name in self.alert_cooldowns and 
                    current_time - self.alert_cooldowns[rule_name] < rule.cooldown_minutes * 60):
                    continue
                
                # Evaluate rule
                alert = self._evaluate_alert_rule(rule)
                if alert:
                    fired_alerts.append(alert)
                    self.alert_cooldowns[rule_name] = current_time
        
        self.last_evaluation = current_time
        return fired_alerts
    
    def _evaluate_alert_rule(self, rule: AlertRule) -> Optional[Alert]:
        """Evaluate a single alert rule."""
        window_hours = rule.window_minutes / 60.0
        
        if rule.condition == AlertCondition.MISSING_DATA:
            points = self.metric_collector.get_metric_points(
                rule.metric_name, window_hours, rule.labels_filter
            )
            
            if not points:
                return self._fire_alert(rule, 0.0, "No data points found")
            return None
        
        # Get metric statistics
        stats = self.metric_collector.get_metric_stats(
            rule.metric_name, window_hours, rule.labels_filter
        )
        
        if "error" in stats:
            return None
        
        # Choose metric value based on condition
        if rule.metric_name.endswith('_duration_ms'):
            current_value = stats.get('p95', 0.0)  # Use P95 for latency
        elif rule.metric_name.endswith('_rate'):
            current_value = stats.get('mean', 0.0)  # Use mean for rates
        else:
            current_value = stats.get('max', 0.0)  # Use max for counters
        
        # Evaluate condition
        should_fire = False
        
        if rule.condition == AlertCondition.THRESHOLD_ABOVE:
            should_fire = current_value > rule.threshold
        elif rule.condition == AlertCondition.THRESHOLD_BELOW:
            should_fire = current_value < rule.threshold
        elif rule.condition == AlertCondition.RATE_CHANGE:
            # Compare with previous window
            prev_stats = self.metric_collector.get_metric_stats(
                rule.metric_name, window_hours * 2, rule.labels_filter
            )
            if "error" not in prev_stats:
                prev_value = prev_stats.get('mean', 0.0)
                rate_change = abs(current_value - prev_value) / max(prev_value, 0.1)
                should_fire = rate_change > rule.threshold
        
        if should_fire:
            return self._fire_alert(rule, current_value, "Threshold exceeded")
        
        return None
    
    def _fire_alert(self, rule: AlertRule, current_value: float, reason: str) -> Alert:
        """Fire a new alert."""
        alert_id = f"{rule.rule_name}_{int(time.time())}"
        
        alert = Alert(
            alert_id=alert_id,
            rule_name=rule.rule_name,
            severity=rule.severity,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold=rule.threshold,
            description=rule.description,
            fired_at=time.time(),
            resolved_at=None,
            status="firing",
            labels=dict(rule.labels_filter),
            annotations={
                "reason": reason,
                "runbook_url": rule.runbook_url
            }
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(f"Alert fired: {rule.rule_name} - {rule.description}")
        return alert
    
    def _resolve_alert(self, alert_id: str, reason: str = "Condition resolved") -> None:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = time.time()
            alert.status = "resolved"
            alert.annotations["resolution_reason"] = reason
            
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert.rule_name} - {reason}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of currently active alerts."""
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            alert for alert in self.alert_history
            if alert.fired_at >= cutoff_time
        ]


class DashboardManager:
    """Management of monitoring dashboards and widgets."""
    
    def __init__(self, metric_collector: MetricCollector, alert_manager: AlertManager):
        self.metric_collector = metric_collector
        self.alert_manager = alert_manager
        
        # Dashboard storage
        self.dashboards: Dict[str, Dashboard] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize default dashboards
        self._initialize_default_dashboards()
    
    def _initialize_default_dashboards(self) -> None:
        """Initialize default monitoring dashboards."""
        current_time = time.time()
        
        # Main fact operations dashboard
        main_widgets = [
            DashboardWidget(
                widget_id="operations_rate",
                widget_type="time_series",
                title="Fact Operations Rate",
                metric_query="rate(fact_operations_total[5m])",
                refresh_interval=30,
                chart_config={"chart_type": "line", "unit": "ops/sec"},
                position={"x": 0, "y": 0, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="error_rate",
                widget_type="time_series",
                title="Error Rate",
                metric_query="rate(fact_operation_errors_total[5m])",
                refresh_interval=30,
                chart_config={"chart_type": "line", "unit": "errors/sec", "color": "red"},
                position={"x": 6, "y": 0, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="latency_p95",
                widget_type="time_series",
                title="P95 Latency",
                metric_query="histogram_quantile(0.95, fact_operation_duration_ms)",
                refresh_interval=30,
                chart_config={"chart_type": "line", "unit": "ms"},
                position={"x": 0, "y": 4, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="active_operations",
                widget_type="gauge",
                title="Active Operations",
                metric_query="fact_operations_active",
                refresh_interval=15,
                chart_config={"chart_type": "gauge", "min": 0, "max": 100},
                position={"x": 6, "y": 4, "width": 3, "height": 4}
            ),
            DashboardWidget(
                widget_id="success_rate",
                widget_type="gauge",
                title="Success Rate",
                metric_query="fact_operation_success_rate",
                refresh_interval=30,
                chart_config={"chart_type": "gauge", "min": 0, "max": 1, "unit": "%"},
                position={"x": 9, "y": 4, "width": 3, "height": 4}
            )
        ]
        
        main_dashboard = Dashboard(
            dashboard_id="fact_operations_main",
            title="Fact Operations - Main Dashboard",
            description="Primary monitoring dashboard for fact retrieval operations",
            widgets=main_widgets,
            refresh_interval=30,
            created_at=current_time,
            updated_at=current_time
        )
        
        # Phase 3 specific dashboard
        phase3_widgets = [
            DashboardWidget(
                widget_id="entity_extraction_rate",
                widget_type="time_series",
                title="Entity Extraction Rate",
                metric_query="rate(entity_extraction_operations[5m])",
                refresh_interval=30,
                chart_config={"chart_type": "line", "unit": "extractions/sec"},
                position={"x": 0, "y": 0, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="hybrid_scoring_latency",
                widget_type="time_series",
                title="Hybrid Scoring Latency",
                metric_query="histogram_quantile(0.95, hybrid_scoring_duration_ms)",
                refresh_interval=30,
                chart_config={"chart_type": "line", "unit": "ms"},
                position={"x": 6, "y": 0, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="query_expansion_coverage",
                widget_type="time_series",
                title="Query Expansion Coverage",
                metric_query="avg(query_expansion_variants_generated)",
                refresh_interval=30,
                chart_config={"chart_type": "line", "unit": "variants"},
                position={"x": 0, "y": 4, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="graph_operations",
                widget_type="time_series",
                title="Graph Operations",
                metric_query="rate(graph_operations_total[5m])",
                refresh_interval=30,
                chart_config={"chart_type": "line", "unit": "ops/sec"},
                position={"x": 6, "y": 4, "width": 6, "height": 4}
            )
        ]
        
        phase3_dashboard = Dashboard(
            dashboard_id="phase3_graph_integration",
            title="Phase 3 - Graph Integration",
            description="Monitoring dashboard for Phase 3 graph integration features",
            widgets=phase3_widgets,
            refresh_interval=30,
            created_at=current_time,
            updated_at=current_time
        )
        
        self.dashboards["fact_operations_main"] = main_dashboard
        self.dashboards["phase3_graph_integration"] = phase3_dashboard
    
    def create_dashboard(self, dashboard: Dashboard) -> None:
        """Create a new monitoring dashboard."""
        with self.lock:
            self.dashboards[dashboard.dashboard_id] = dashboard
        
        logger.info(f"Created dashboard: {dashboard.title}")
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get a dashboard by ID."""
        return self.dashboards.get(dashboard_id)
    
    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get dashboard data with current metric values."""
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return {"error": "Dashboard not found"}
        
        dashboard_data = {
            "dashboard": asdict(dashboard),
            "widgets_data": {},
            "generated_at": time.time()
        }
        
        # Get data for each widget
        for widget in dashboard.widgets:
            widget_data = self._get_widget_data(widget)
            dashboard_data["widgets_data"][widget.widget_id] = widget_data
        
        return dashboard_data
    
    def _get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for a specific widget."""
        # In a real implementation, this would parse the metric_query
        # and fetch appropriate data from the metric collector
        
        # For now, return mock data based on widget type
        if widget.widget_type == "time_series":
            # Generate time series data
            current_time = time.time()
            time_points = []
            
            for i in range(60):  # 60 data points
                timestamp = current_time - (59 - i) * 60  # 1-minute intervals
                # Mock data generation based on metric name
                if "error" in widget.metric_query:
                    value = max(0, 2 + (i % 10) - 5)  # Varying error rate
                elif "latency" in widget.metric_query:
                    value = 100 + (i % 20) * 10  # Varying latency
                else:
                    value = 50 + (i % 30)  # General varying metric
                
                time_points.append({"timestamp": timestamp, "value": value})
            
            return {
                "widget_type": "time_series",
                "data": time_points,
                "last_value": time_points[-1]["value"] if time_points else 0
            }
        
        elif widget.widget_type == "gauge":
            # Return single current value
            if "success_rate" in widget.metric_query:
                value = 0.95  # 95% success rate
            elif "active" in widget.metric_query:
                value = 12  # 12 active operations
            else:
                value = 75  # Generic gauge value
            
            return {
                "widget_type": "gauge",
                "current_value": value,
                "timestamp": time.time()
            }
        
        return {"error": "Unsupported widget type"}
    
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all available dashboards."""
        with self.lock:
            return [
                {
                    "dashboard_id": dashboard.dashboard_id,
                    "title": dashboard.title,
                    "description": dashboard.description,
                    "widget_count": len(dashboard.widgets),
                    "created_at": dashboard.created_at,
                    "updated_at": dashboard.updated_at
                }
                for dashboard in self.dashboards.values()
            ]


class FactMonitoringSystem:
    """Comprehensive monitoring system for fact retrieval pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.metric_collector = MetricCollector(
            retention_hours=self.config.get('retention_hours', 24)
        )
        self.alert_manager = AlertManager(self.metric_collector)
        self.dashboard_manager = DashboardManager(self.metric_collector, self.alert_manager)
        
        # Health check state
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.system_health = {"status": "healthy", "last_check": time.time()}
        
        # Background monitoring
        self.monitoring_enabled = True
        self.monitoring_thread = None
        
        logger.info("FactMonitoringSystem initialized")
    
    def start_monitoring(self) -> None:
        """Start background monitoring processes."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.monitoring_enabled = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Background monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring processes."""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Background monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_enabled:
            try:
                # Evaluate alerts
                fired_alerts = self.alert_manager.evaluate_alerts()
                
                # Run health checks
                self._run_health_checks()
                
                # Sleep for monitoring interval
                time.sleep(self.config.get('monitoring_interval', 60))
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)  # Brief pause on error
    
    def _run_health_checks(self) -> None:
        """Run all registered health checks."""
        overall_healthy = True
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = check_func()
                if not result:
                    overall_healthy = False
                    logger.warning(f"Health check failed: {check_name}")
            except Exception as e:
                overall_healthy = False
                logger.error(f"Health check error for {check_name}: {e}")
        
        self.system_health = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "last_check": time.time(),
            "individual_checks": {
                name: {"status": "unknown"} for name in self.health_checks
            }
        }
    
    def add_health_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Add a system health check."""
        self.health_checks[name] = check_func
        logger.info(f"Added health check: {name}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_time = time.time()
        
        # Get recent metrics summary
        all_metrics = self.metric_collector.get_all_metric_names()
        metrics_summary = {}
        
        for metric_name in all_metrics[:10]:  # Limit to first 10 metrics
            stats = self.metric_collector.get_metric_stats(metric_name, hours=1.0)
            if "error" not in stats:
                metrics_summary[metric_name] = {
                    "count": stats.get("count", 0),
                    "latest_value": stats.get("max", 0),
                    "avg_value": stats.get("mean", 0)
                }
        
        return {
            "system_health": self.system_health,
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "metrics_summary": metrics_summary,
            "total_metrics": len(all_metrics),
            "monitoring_enabled": self.monitoring_enabled,
            "uptime_seconds": current_time - self.metric_collector.last_cache_update,
            "timestamp": current_time
        }
    
    def export_monitoring_config(self) -> Dict[str, Any]:
        """Export monitoring configuration for backup."""
        return {
            "alert_rules": {
                name: asdict(rule) for name, rule in self.alert_manager.alert_rules.items()
            },
            "dashboards": {
                dashboard_id: asdict(dashboard) 
                for dashboard_id, dashboard in self.dashboard_manager.dashboards.items()
            },
            "config": self.config,
            "exported_at": time.time()
        }


# Global monitoring system instance
_monitoring_system: Optional[FactMonitoringSystem] = None


def get_monitoring_system() -> FactMonitoringSystem:
    """Get global monitoring system instance."""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = FactMonitoringSystem()
    return _monitoring_system


def initialize_monitoring(config: Dict[str, Any]) -> FactMonitoringSystem:
    """Initialize global monitoring system with configuration."""
    global _monitoring_system
    _monitoring_system = FactMonitoringSystem(config)
    return _monitoring_system


# Convenience functions for metric recording
def record_operation_metric(operation: str, duration_ms: float, success: bool, **labels) -> None:
    """Record a fact operation metric."""
    monitoring = get_monitoring_system()
    
    # Record operation count
    monitoring.metric_collector.record_metric(
        "fact_operations_total",
        1.0,
        labels={"operation": operation, "status": "success" if success else "error", **labels}
    )
    
    # Record duration
    monitoring.metric_collector.record_metric(
        "fact_operation_duration_ms",
        duration_ms,
        labels={"operation": operation, **labels}
    )
    
    # Record error if applicable
    if not success:
        monitoring.metric_collector.record_metric(
            "fact_operation_errors_total",
            1.0,
            labels={"operation": operation, **labels}
        )


def record_custom_metric(metric_name: str, value: float, **labels) -> None:
    """Record a custom metric."""
    get_monitoring_system().metric_collector.record_metric(metric_name, value, labels)