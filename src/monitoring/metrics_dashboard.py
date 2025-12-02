#!/usr/bin/env python3
"""
metrics_dashboard.py: p95/p99 Latency & Error Budget Dashboard for Sprint 11 Phase 5

Implements comprehensive observability dashboard with:
- Real-time p95/p99 latency tracking
- Error budget monitoring and alerting  
- SLA compliance tracking
- Performance degradation detection
- Service health scoring
"""

import asyncio
import time
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from src.storage.circuit_breaker import get_mcp_service_health
from src.core.error_codes import ErrorCode

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SLOStatus(str, Enum):
    """SLO compliance status"""
    HEALTHY = "healthy"           # Within error budget
    WARNING = "warning"           # Approaching budget burn
    CRITICAL = "critical"         # Exceeding error budget
    EXHAUSTED = "exhausted"       # Error budget depleted


@dataclass
class LatencyMetrics:
    """Latency performance metrics"""
    p50: float  # 50th percentile (median)
    p95: float  # 95th percentile
    p99: float  # 99th percentile
    p99_9: float  # 99.9th percentile
    mean: float
    max: float
    min: float
    count: int
    window_start: datetime
    window_end: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat()
        }


@dataclass
class ErrorBudgetMetrics:
    """Error budget tracking metrics"""
    total_requests: int
    failed_requests: int
    error_rate: float
    error_budget: float          # Target error budget (e.g., 0.1% = 0.001)
    error_budget_remaining: float  # How much budget is left
    budget_burn_rate: float      # Rate of budget consumption
    slo_status: SLOStatus
    time_to_exhaustion: Optional[timedelta]  # When budget will be exhausted
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["slo_status"] = self.slo_status.value
        if self.time_to_exhaustion:
            result["time_to_exhaustion"] = str(self.time_to_exhaustion)
        return result


@dataclass 
class ServiceHealthMetrics:
    """Overall service health metrics"""
    health_score: float          # 0.0 - 1.0
    availability: float          # Uptime percentage
    latency_score: float         # Based on p95/p99 targets
    error_rate_score: float      # Based on error budget
    circuit_breaker_score: float # Based on CB state
    last_incident: Optional[datetime]
    incidents_24h: int
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.last_incident:
            result["last_incident"] = self.last_incident.isoformat()
        return result


@dataclass
class Alert:
    """Performance or error budget alert"""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    metric_type: str
    current_value: float
    threshold: float
    created_at: datetime
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["severity"] = self.severity.value
        result["created_at"] = self.created_at.isoformat()
        if self.resolved_at:
            result["resolved_at"] = self.resolved_at.isoformat()
        return result


class MetricWindow:
    """Sliding window for metric collection"""
    
    def __init__(self, window_size: int = 1000):
        """Initialize metric window
        
        Args:
            window_size: Maximum number of samples to keep
        """
        self.window_size = window_size
        self.samples: deque = deque(maxlen=window_size)
        self.lock = threading.Lock()
    
    def add_sample(self, value: float, timestamp: Optional[datetime] = None):
        """Add a sample to the window"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        with self.lock:
            self.samples.append((value, timestamp))
    
    def get_percentiles(self, window_minutes: int = 5) -> Optional[LatencyMetrics]:
        """Calculate percentiles for recent samples"""
        if not self.samples:
            return None
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        with self.lock:
            recent_samples = [
                value for value, timestamp in self.samples 
                if timestamp > cutoff_time
            ]
        
        if not recent_samples:
            return None
        
        # Calculate percentiles
        sorted_samples = sorted(recent_samples)
        count = len(sorted_samples)
        
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = k - f
            if f >= len(data) - 1:
                return data[-1]
            return data[f] + c * (data[f + 1] - data[f])
        
        window_start = min(timestamp for _, timestamp in self.samples if timestamp > cutoff_time)
        window_end = datetime.utcnow()
        
        return LatencyMetrics(
            p50=percentile(sorted_samples, 50),
            p95=percentile(sorted_samples, 95),
            p99=percentile(sorted_samples, 99),
            p99_9=percentile(sorted_samples, 99.9),
            mean=statistics.mean(sorted_samples),
            max=max(sorted_samples),
            min=min(sorted_samples),
            count=count,
            window_start=window_start,
            window_end=window_end
        )


class ErrorBudgetTracker:
    """Tracks error budget consumption and SLO compliance"""
    
    def __init__(self, error_budget_percent: float = 0.1):
        """Initialize error budget tracker
        
        Args:
            error_budget_percent: Error budget as percentage (0.1 = 0.1%)
        """
        self.error_budget = error_budget_percent / 100.0  # Convert to decimal
        self.requests: deque = deque(maxlen=10000)  # Keep last 10k requests
        self.lock = threading.Lock()
    
    def record_request(self, success: bool, timestamp: Optional[datetime] = None):
        """Record a request result"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        with self.lock:
            self.requests.append((success, timestamp))
    
    def get_error_budget_metrics(self, window_hours: int = 24) -> ErrorBudgetMetrics:
        """Calculate error budget metrics for specified window"""
        cutoff_time = datetime.utcnow() - timedelta(hours=window_hours)
        
        with self.lock:
            recent_requests = [
                success for success, timestamp in self.requests
                if timestamp > cutoff_time
            ]
        
        if not recent_requests:
            return ErrorBudgetMetrics(
                total_requests=0,
                failed_requests=0,
                error_rate=0.0,
                error_budget=self.error_budget,
                error_budget_remaining=1.0,
                budget_burn_rate=0.0,
                slo_status=SLOStatus.HEALTHY,
                time_to_exhaustion=None
            )
        
        total_requests = len(recent_requests)
        failed_requests = sum(1 for success in recent_requests if not success)
        error_rate = failed_requests / total_requests if total_requests > 0 else 0.0
        
        # Calculate error budget consumption
        error_budget_consumed = error_rate / self.error_budget if self.error_budget > 0 else 0.0
        error_budget_remaining = max(0.0, 1.0 - error_budget_consumed)
        
        # Calculate burn rate (errors per hour)
        burn_rate = (failed_requests / window_hours) if window_hours > 0 else 0.0
        
        # Estimate time to exhaustion
        time_to_exhaustion = None
        if burn_rate > 0 and error_budget_remaining > 0:
            remaining_budget_errors = self.error_budget * total_requests * (error_budget_remaining / error_budget_consumed) if error_budget_consumed > 0 else float('inf')
            hours_remaining = remaining_budget_errors / burn_rate
            if hours_remaining < 168:  # Less than a week
                time_to_exhaustion = timedelta(hours=hours_remaining)
        
        # Determine SLO status
        if error_budget_remaining <= 0:
            slo_status = SLOStatus.EXHAUSTED
        elif error_budget_remaining < 0.2:  # Less than 20% remaining
            slo_status = SLOStatus.CRITICAL
        elif error_budget_remaining < 0.5:  # Less than 50% remaining  
            slo_status = SLOStatus.WARNING
        else:
            slo_status = SLOStatus.HEALTHY
        
        return ErrorBudgetMetrics(
            total_requests=total_requests,
            failed_requests=failed_requests,
            error_rate=error_rate,
            error_budget=self.error_budget,
            error_budget_remaining=error_budget_remaining,
            budget_burn_rate=burn_rate,
            slo_status=slo_status,
            time_to_exhaustion=time_to_exhaustion
        )


class MetricsDashboard:
    """Comprehensive metrics dashboard for Sprint 11"""
    
    def __init__(self):
        """Initialize metrics dashboard"""
        
        # Metric collection windows
        self.latency_windows = {
            "store_context": MetricWindow(),
            "retrieve_context": MetricWindow(),
            "query_graph": MetricWindow(),
            "overall": MetricWindow()
        }
        
        # Error budget trackers
        self.error_budgets = {
            "store_context": ErrorBudgetTracker(error_budget_percent=0.1),
            "retrieve_context": ErrorBudgetTracker(error_budget_percent=0.05),
            "query_graph": ErrorBudgetTracker(error_budget_percent=0.2),
            "overall": ErrorBudgetTracker(error_budget_percent=0.1)
        }
        
        # Alert management
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # SLA targets
        self.sla_targets = {
            "store_context": {"p95_ms": 1000, "p99_ms": 2000},
            "retrieve_context": {"p95_ms": 500, "p99_ms": 1000},
            "query_graph": {"p95_ms": 2000, "p99_ms": 5000},
            "overall": {"p95_ms": 1000, "p99_ms": 2000}
        }
        
        # Service health tracking
        self.service_start_time = datetime.utcnow()
        self.last_health_check = datetime.utcnow()
    
    def record_request_latency(self, endpoint: str, latency_ms: float, success: bool):
        """Record request latency and success/failure"""
        timestamp = datetime.utcnow()
        
        # Record latency
        if endpoint in self.latency_windows:
            self.latency_windows[endpoint].add_sample(latency_ms, timestamp)
        self.latency_windows["overall"].add_sample(latency_ms, timestamp)
        
        # Record success/failure for error budget
        if endpoint in self.error_budgets:
            self.error_budgets[endpoint].record_request(success, timestamp)
        self.error_budgets["overall"].record_request(success, timestamp)
        
        # Check for alert conditions
        self._check_alert_conditions(endpoint, latency_ms, success)
    
    def get_latency_metrics(self, endpoint: str = "overall", window_minutes: int = 5) -> Optional[LatencyMetrics]:
        """Get latency metrics for an endpoint"""
        if endpoint not in self.latency_windows:
            return None
        
        return self.latency_windows[endpoint].get_percentiles(window_minutes)
    
    def get_error_budget_metrics(self, endpoint: str = "overall", window_hours: int = 24) -> ErrorBudgetMetrics:
        """Get error budget metrics for an endpoint"""
        if endpoint not in self.error_budgets:
            return ErrorBudgetMetrics(
                total_requests=0,
                failed_requests=0,
                error_rate=0.0,
                error_budget=0.1,
                error_budget_remaining=1.0,
                budget_burn_rate=0.0,
                slo_status=SLOStatus.HEALTHY,
                time_to_exhaustion=None
            )
        
        return self.error_budgets[endpoint].get_error_budget_metrics(window_hours)
    
    def get_service_health_metrics(self) -> ServiceHealthMetrics:
        """Calculate overall service health metrics"""
        
        # Calculate availability (uptime)
        uptime = (datetime.utcnow() - self.service_start_time).total_seconds()
        availability = 1.0  # Simplified - would track actual downtime
        
        # Get latency metrics
        overall_latency = self.get_latency_metrics("overall", window_minutes=15)
        latency_score = self._calculate_latency_score(overall_latency) if overall_latency else 1.0
        
        # Get error budget metrics
        error_budget = self.get_error_budget_metrics("overall", window_hours=24)
        error_rate_score = error_budget.error_budget_remaining
        
        # Get circuit breaker health
        cb_health = get_mcp_service_health()
        cb_score = 1.0 if cb_health["available"] else 0.3
        
        # Calculate overall health score (weighted average)
        health_score = (
            availability * 0.3 +
            latency_score * 0.3 + 
            error_rate_score * 0.3 +
            cb_score * 0.1
        )
        
        # Count recent incidents (alerts)
        incidents_24h = len([
            alert for alert in self.alert_history
            if alert.created_at > datetime.utcnow() - timedelta(hours=24)
        ])
        
        # Find last incident
        last_incident = None
        if self.alert_history:
            last_incident = max(alert.created_at for alert in self.alert_history)
        
        return ServiceHealthMetrics(
            health_score=health_score,
            availability=availability,
            latency_score=latency_score,
            error_rate_score=error_rate_score,
            circuit_breaker_score=cb_score,
            last_incident=last_incident,
            incidents_24h=incidents_24h
        )
    
    def _calculate_latency_score(self, latency_metrics: LatencyMetrics) -> float:
        """Calculate latency health score (0.0 - 1.0)"""
        targets = self.sla_targets.get("overall", {"p95_ms": 1000, "p99_ms": 2000})
        
        # Score based on how well we meet SLA targets
        p95_score = min(1.0, targets["p95_ms"] / max(latency_metrics.p95, 1))
        p99_score = min(1.0, targets["p99_ms"] / max(latency_metrics.p99, 1))
        
        # Weight p95 more heavily than p99
        return (p95_score * 0.7 + p99_score * 0.3)
    
    def _check_alert_conditions(self, endpoint: str, latency_ms: float, success: bool):
        """Check if alert conditions are met"""
        
        # Check latency SLA violations
        targets = self.sla_targets.get(endpoint)
        if targets:
            if latency_ms > targets["p99_ms"] * 2:  # 2x p99 target
                self._create_alert(
                    f"high_latency_{endpoint}",
                    AlertSeverity.CRITICAL,
                    f"Extreme latency detected for {endpoint}",
                    f"Request took {latency_ms:.2f}ms (target: {targets['p99_ms']}ms)",
                    "latency",
                    latency_ms,
                    targets["p99_ms"] * 2
                )
            elif latency_ms > targets["p95_ms"] * 3:  # 3x p95 target
                self._create_alert(
                    f"high_latency_{endpoint}",
                    AlertSeverity.WARNING,
                    f"High latency detected for {endpoint}",
                    f"Request took {latency_ms:.2f}ms (target: {targets['p95_ms']}ms)",
                    "latency", 
                    latency_ms,
                    targets["p95_ms"] * 3
                )
        
        # Check error budget burn rate
        if not success:
            error_budget = self.get_error_budget_metrics(endpoint, window_hours=1)  # Check hourly burn
            if error_budget.slo_status == SLOStatus.CRITICAL:
                self._create_alert(
                    f"error_budget_{endpoint}",
                    AlertSeverity.ERROR,
                    f"Error budget critical for {endpoint}",
                    f"Error budget remaining: {error_budget.error_budget_remaining:.1%}",
                    "error_budget",
                    error_budget.error_budget_remaining,
                    0.2  # 20% threshold
                )
    
    def _create_alert(self, alert_id: str, severity: AlertSeverity, title: str, 
                     message: str, metric_type: str, current_value: float, threshold: float):
        """Create or update an alert"""
        
        if alert_id in self.active_alerts:
            # Update existing alert
            self.active_alerts[alert_id].message = message
            self.active_alerts[alert_id].current_value = current_value
        else:
            # Create new alert
            alert = Alert(
                id=alert_id,
                severity=severity,
                title=title,
                message=message,
                metric_type=metric_type,
                current_value=current_value,
                threshold=threshold,
                created_at=datetime.utcnow()
            )
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            logger.warning(f"🚨 Alert created: {alert.title} - {alert.message}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.utcnow()
            del self.active_alerts[alert_id]
            
            logger.info(f"✅ Alert resolved: {alert.title}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service_health": self.get_service_health_metrics().to_dict(),
            "latency_metrics": {},
            "error_budget_metrics": {},
            "active_alerts": [alert.to_dict() for alert in self.active_alerts.values()],
            "recent_incidents": len([
                alert for alert in self.alert_history
                if alert.created_at > datetime.utcnow() - timedelta(hours=24)
            ]),
            "sla_targets": self.sla_targets
        }
        
        # Add latency metrics for each endpoint
        for endpoint in ["store_context", "retrieve_context", "query_graph", "overall"]:
            latency = self.get_latency_metrics(endpoint, window_minutes=15)
            if latency:
                dashboard_data["latency_metrics"][endpoint] = latency.to_dict()
        
        # Add error budget metrics for each endpoint
        for endpoint in ["store_context", "retrieve_context", "query_graph", "overall"]:
            error_budget = self.get_error_budget_metrics(endpoint, window_hours=24)
            dashboard_data["error_budget_metrics"][endpoint] = error_budget.to_dict()
        
        return dashboard_data
    
    def get_slo_compliance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate SLO compliance report"""
        
        report = {
            "report_period_hours": hours,
            "generated_at": datetime.utcnow().isoformat(),
            "overall_compliance": True,
            "endpoint_compliance": {}
        }
        
        overall_compliant = True
        
        for endpoint in ["store_context", "retrieve_context", "query_graph"]:
            # Get metrics for the period
            latency = self.get_latency_metrics(endpoint, window_minutes=hours*60)
            error_budget = self.get_error_budget_metrics(endpoint, window_hours=hours)
            targets = self.sla_targets.get(endpoint, {})
            
            # Check SLA compliance
            latency_compliant = True
            if latency and targets:
                latency_compliant = (
                    latency.p95 <= targets.get("p95_ms", float('inf')) and
                    latency.p99 <= targets.get("p99_ms", float('inf'))
                )
            
            error_budget_compliant = error_budget.slo_status in [SLOStatus.HEALTHY, SLOStatus.WARNING]
            
            endpoint_compliant = latency_compliant and error_budget_compliant
            overall_compliant = overall_compliant and endpoint_compliant
            
            report["endpoint_compliance"][endpoint] = {
                "compliant": endpoint_compliant,
                "latency_compliant": latency_compliant,
                "error_budget_compliant": error_budget_compliant,
                "latency_metrics": latency.to_dict() if latency else None,
                "error_budget_metrics": error_budget.to_dict(),
                "sla_targets": targets
            }
        
        report["overall_compliance"] = overall_compliant
        
        return report


# Global metrics dashboard instance
metrics_dashboard = MetricsDashboard()


def record_request_metrics(endpoint: str, latency_ms: float, success: bool):
    """Global function to record request metrics"""
    metrics_dashboard.record_request_latency(endpoint, latency_ms, success)


def get_dashboard_snapshot() -> Dict[str, Any]:
    """Get current dashboard snapshot"""
    return metrics_dashboard.get_dashboard_data()


def get_slo_status() -> Dict[str, str]:
    """Get current SLO status for all endpoints"""
    status = {}
    
    for endpoint in ["store_context", "retrieve_context", "query_graph", "overall"]:
        error_budget = metrics_dashboard.get_error_budget_metrics(endpoint)
        status[endpoint] = error_budget.slo_status.value
    
    return status