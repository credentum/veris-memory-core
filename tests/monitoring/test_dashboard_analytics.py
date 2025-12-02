#!/usr/bin/env python3
"""
Test suite for dashboard analytics features.

Tests cover:
- generate_json_dashboard_with_analytics method
- _generate_performance_insights method
- Real-time analytics integration
- Performance threshold analysis
- Time-series analytics output
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.monitoring.dashboard import UnifiedDashboard
from src.monitoring.request_metrics import RequestMetricsCollector


class TestDashboardAnalytics:
    """Test suite for dashboard analytics integration."""
    
    @pytest.fixture
    def dashboard(self):
        """Create a Dashboard instance for testing."""
        config = {
            'json': {
                'enabled': True,
                'include_trends': True,
                'include_insights': True,
                'pretty_print': True
            },
            'ascii': {
                'enabled': True,
                'colored': True
            },
            'thresholds': {
                'error_rate_warning_percent': 1.0,
                'error_rate_critical_percent': 5.0,
                'latency_warning_ms': 100,
                'latency_critical_ms': 500
            }
        }
        return UnifiedDashboard(config)
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Create a mock metrics collector."""
        collector = Mock(spec=RequestMetricsCollector)
        collector.get_trending_data = AsyncMock()
        collector.get_endpoint_stats = AsyncMock()
        collector.get_global_stats = AsyncMock()
        return collector
    
    @pytest.mark.asyncio
    async def test_generate_json_dashboard_with_analytics_basic(self, dashboard):
        """Test basic JSON dashboard generation with analytics."""
        # Mock base metrics
        base_metrics = {
            'system': {'cpu_percent': 45.2, 'memory_percent': 62.1},
            'services': [{'name': 'Redis', 'status': 'healthy'}]
        }
        
        with patch('src.monitoring.request_metrics.get_metrics_collector') as mock_get_collector:
            mock_collector = Mock(spec=RequestMetricsCollector)
            mock_collector.get_trending_data.return_value = []
            mock_collector.get_endpoint_stats.return_value = {}
            mock_collector.get_global_stats.return_value = {
                'error_rate_percent': 0.5,
                'avg_duration_ms': 50.0,
                'p99_duration_ms': 120.0
            }
            mock_get_collector.return_value = mock_collector
            
            result_json = await dashboard.generate_json_dashboard_with_analytics(
                metrics=base_metrics,
                include_trends=True,
                minutes=5
            )
            
            result = json.loads(result_json)
            
            # Should include base metrics
            assert 'system' in result
            assert 'services' in result
            
            # Should include analytics
            assert 'analytics' in result
            assert 'trending_data' in result['analytics']
            assert 'endpoint_statistics' in result['analytics']
            assert 'global_request_stats' in result['analytics']
            
            # Should include insights
            assert 'insights' in result
    
    @pytest.mark.asyncio
    async def test_generate_json_dashboard_with_trends_disabled(self, dashboard):
        """Test JSON dashboard generation with trends disabled."""
        base_metrics = {'system': {'cpu_percent': 45.2}}
        
        result_json = await dashboard.generate_json_dashboard_with_analytics(
            metrics=base_metrics,
            include_trends=False
        )
        
        result = json.loads(result_json)
        
        # Should include base metrics but not analytics
        assert 'system' in result
        assert 'analytics' not in result
    
    @pytest.mark.asyncio
    async def test_generate_json_dashboard_error_handling(self, dashboard):
        """Test JSON dashboard generation with import errors."""
        base_metrics = {'system': {'cpu_percent': 45.2}}
        
        # Mock import failure
        with patch('src.monitoring.request_metrics.get_metrics_collector', side_effect=ImportError("Module not found")):
            result_json = await dashboard.generate_json_dashboard_with_analytics(
                metrics=base_metrics,
                include_trends=True
            )
            
            result = json.loads(result_json)
            
            # Should still include base metrics
            assert 'system' in result
            # Analytics should be gracefully omitted
            assert 'analytics' not in result
    
    @pytest.mark.asyncio
    async def test_generate_json_dashboard_no_base_metrics(self, dashboard):
        """Test JSON dashboard generation when no base metrics provided."""
        with patch.object(dashboard, 'collect_all_metrics') as mock_collect:
            mock_collect.return_value = {'system': {'cpu_percent': 30.0}}
            
            with patch('src.monitoring.request_metrics.get_metrics_collector') as mock_get_collector:
                mock_collector = Mock(spec=RequestMetricsCollector)
                mock_collector.get_trending_data.return_value = []
                mock_collector.get_endpoint_stats.return_value = {}
                mock_collector.get_global_stats.return_value = {}
                mock_get_collector.return_value = mock_collector
                
                result_json = await dashboard.generate_json_dashboard_with_analytics()
                result = json.loads(result_json)
                
                # Should have collected base metrics
                assert 'system' in result
                assert result['system']['cpu_percent'] == 30.0
                
                # Should include analytics
                assert 'analytics' in result


class TestPerformanceInsights:
    """Test suite for performance insights generation."""
    
    @pytest.fixture
    def dashboard(self):
        """Create a Dashboard with test thresholds."""
        config = {
            'thresholds': {
                'error_rate_warning_percent': 1.0,
                'error_rate_critical_percent': 5.0,
                'latency_warning_ms': 100,
                'latency_critical_ms': 500
            }
        }
        return UnifiedDashboard(config)
    
    @pytest.mark.asyncio
    async def test_performance_insights_healthy_system(self, dashboard):
        """Test performance insights for healthy system."""
        global_stats = {
            'error_rate_percent': 0.2,
            'avg_duration_ms': 45.0,
            'p99_duration_ms': 85.0
        }
        endpoint_stats = {
            'GET /api/users': {
                'error_rate_percent': 0.0,
                'avg_duration_ms': 30.0,
                'p99_duration_ms': 60.0
            }
        }
        
        insights = await dashboard._generate_performance_insights(global_stats, endpoint_stats)
        
        assert insights['performance_status'] == 'good'
        assert len(insights['alerts']) == 0
        assert len(insights['recommendations']) == 0
        
        # Should include key metrics
        assert 'key_metrics' in insights
    
    @pytest.mark.asyncio
    async def test_performance_insights_warning_error_rate(self, dashboard):
        """Test performance insights with warning-level error rate."""
        global_stats = {
            'error_rate_percent': 2.5,  # Above 1% warning threshold
            'avg_duration_ms': 45.0,
            'p99_duration_ms': 85.0
        }
        endpoint_stats = {}
        
        insights = await dashboard._generate_performance_insights(global_stats, endpoint_stats)
        
        assert insights['performance_status'] == 'warning'
        assert len(insights['alerts']) == 1
        
        alert = insights['alerts'][0]
        assert alert['type'] == 'error_rate_warning'
        assert alert['severity'] == 'warning'
        assert '2.50%' in alert['message']
        assert '1%' in alert['message']
    
    @pytest.mark.asyncio
    async def test_performance_insights_critical_error_rate(self, dashboard):
        """Test performance insights with critical error rate."""
        global_stats = {
            'error_rate_percent': 7.5,  # Above 5% critical threshold
            'avg_duration_ms': 45.0,
            'p99_duration_ms': 85.0
        }
        endpoint_stats = {}
        
        insights = await dashboard._generate_performance_insights(global_stats, endpoint_stats)
        
        assert insights['performance_status'] == 'critical'
        assert len(insights['alerts']) == 1
        
        alert = insights['alerts'][0]
        assert alert['type'] == 'error_rate_critical'
        assert alert['severity'] == 'critical'
        assert 'Investigate failing requests immediately' in insights['recommendations']
    
    @pytest.mark.asyncio
    async def test_performance_insights_warning_latency(self, dashboard):
        """Test performance insights with warning-level latency."""
        global_stats = {
            'error_rate_percent': 0.5,
            'avg_duration_ms': 150.0,  # Above 100ms warning threshold
            'p99_duration_ms': 250.0
        }
        endpoint_stats = {}
        
        insights = await dashboard._generate_performance_insights(global_stats, endpoint_stats)
        
        assert insights['performance_status'] == 'warning'
        assert len(insights['alerts']) == 1
        
        alert = insights['alerts'][0]
        assert alert['type'] == 'latency_warning'
        assert '150.0ms' in alert['message']
    
    @pytest.mark.asyncio
    async def test_performance_insights_critical_latency(self, dashboard):
        """Test performance insights with critical P99 latency."""
        global_stats = {
            'error_rate_percent': 0.5,
            'avg_duration_ms': 200.0,
            'p99_duration_ms': 750.0  # Above 500ms critical threshold
        }
        endpoint_stats = {}
        
        insights = await dashboard._generate_performance_insights(global_stats, endpoint_stats)
        
        assert insights['performance_status'] == 'critical'
        assert len(insights['alerts']) == 1
        
        alert = insights['alerts'][0]
        assert alert['type'] == 'latency_critical'
        assert alert['severity'] == 'critical'
        assert 'Optimize slow endpoints and database queries' in insights['recommendations']
    
    @pytest.mark.asyncio
    async def test_performance_insights_multiple_issues(self, dashboard):
        """Test performance insights with multiple performance issues."""
        global_stats = {
            'error_rate_percent': 3.0,  # Warning level
            'avg_duration_ms': 200.0,   # Warning level
            'p99_duration_ms': 600.0    # Critical level
        }
        endpoint_stats = {}
        
        insights = await dashboard._generate_performance_insights(global_stats, endpoint_stats)
        
        # Should be critical due to P99 latency
        assert insights['performance_status'] == 'critical'
        
        # Should have multiple alerts
        assert len(insights['alerts']) >= 2
        
        alert_types = [alert['type'] for alert in insights['alerts']]
        assert 'error_rate_warning' in alert_types
        assert 'latency_critical' in alert_types
    
    @pytest.mark.asyncio
    async def test_performance_insights_slow_endpoints_analysis(self, dashboard):
        """Test performance insights with slow endpoints analysis."""
        global_stats = {
            'error_rate_percent': 0.5,
            'avg_duration_ms': 50.0,
            'p99_duration_ms': 85.0
        }
        endpoint_stats = {
            'GET /api/fast': {
                'avg_duration_ms': 25.0,
                'p99_duration_ms': 50.0,
                'error_rate_percent': 0.0
            },
            'GET /api/slow': {
                'avg_duration_ms': 250.0,  # Slow endpoint
                'p99_duration_ms': 450.0,
                'error_rate_percent': 1.0
            },
            'POST /api/batch': {
                'avg_duration_ms': 180.0,
                'p99_duration_ms': 380.0,
                'error_rate_percent': 0.5
            }
        }
        
        insights = await dashboard._generate_performance_insights(global_stats, endpoint_stats)
        
        # Should identify slow endpoints
        assert 'slow_endpoints' in insights['key_metrics']
        slow_endpoints = insights['key_metrics']['slow_endpoints']
        
        # Should identify the slowest endpoints
        assert len(slow_endpoints) > 0
        slowest_endpoint = slow_endpoints[0]
        assert slowest_endpoint['endpoint'] == 'GET /api/slow'
        assert slowest_endpoint['avg_duration_ms'] == 250.0
    
    @pytest.mark.asyncio
    async def test_performance_insights_error_prone_endpoints(self, dashboard):
        """Test performance insights with error-prone endpoints analysis."""
        global_stats = {
            'error_rate_percent': 1.5,
            'avg_duration_ms': 50.0,
            'p99_duration_ms': 85.0
        }
        endpoint_stats = {
            'GET /api/stable': {
                'error_rate_percent': 0.1,
                'avg_duration_ms': 45.0
            },
            'POST /api/unreliable': {
                'error_rate_percent': 5.5,  # High error rate
                'avg_duration_ms': 65.0
            },
            'PUT /api/flaky': {
                'error_rate_percent': 2.8,
                'avg_duration_ms': 55.0
            }
        }
        
        insights = await dashboard._generate_performance_insights(global_stats, endpoint_stats)
        
        # Should identify error-prone endpoints
        assert 'error_prone_endpoints' in insights['key_metrics']
        error_endpoints = insights['key_metrics']['error_prone_endpoints']
        
        # Should identify the most error-prone endpoints
        assert len(error_endpoints) > 0
        most_errors = error_endpoints[0]
        assert most_errors['endpoint'] == 'POST /api/unreliable'
        assert most_errors['error_rate_percent'] == 5.5
    
    @pytest.mark.asyncio
    async def test_performance_insights_default_thresholds(self):
        """Test performance insights with default thresholds."""
        # Dashboard without explicit thresholds should use defaults
        dashboard = UnifiedDashboard({
            'json': {'enabled': True, 'pretty_print': True},
            'ascii': {'enabled': True, 'colored': True}
        })
        
        global_stats = {
            'error_rate_percent': 1.5,  # Should trigger warning with default 1%
            'avg_duration_ms': 50.0,
            'p99_duration_ms': 85.0
        }
        endpoint_stats = {}
        
        insights = await dashboard._generate_performance_insights(global_stats, endpoint_stats)
        
        assert insights['performance_status'] == 'warning'
        assert len(insights['alerts']) == 1
        assert insights['alerts'][0]['type'] == 'error_rate_warning'
    
    @pytest.mark.asyncio
    async def test_performance_insights_recommendations_generation(self, dashboard):
        """Test that appropriate recommendations are generated."""
        global_stats = {
            'error_rate_percent': 6.0,  # Critical
            'avg_duration_ms': 300.0,   # Warning
            'p99_duration_ms': 800.0    # Critical
        }
        endpoint_stats = {}
        
        insights = await dashboard._generate_performance_insights(global_stats, endpoint_stats)
        
        recommendations = insights['recommendations']
        
        # Should have recommendations for both error rate and latency
        assert 'Investigate failing requests immediately' in recommendations
        assert 'Optimize slow endpoints and database queries' in recommendations
    
    @pytest.mark.asyncio
    async def test_performance_insights_key_metrics_structure(self, dashboard):
        """Test the structure of key metrics in insights."""
        global_stats = {
            'error_rate_percent': 0.5,
            'avg_duration_ms': 50.0,
            'p99_duration_ms': 85.0,
            'requests_per_minute': 120.0
        }
        endpoint_stats = {
            'GET /api/test': {
                'avg_duration_ms': 45.0,
                'error_rate_percent': 0.2,
                'request_count': 100
            }
        }
        
        insights = await dashboard._generate_performance_insights(global_stats, endpoint_stats)
        
        key_metrics = insights['key_metrics']
        
        # Should include summary statistics
        assert 'total_endpoints' in key_metrics
        assert key_metrics['total_endpoints'] == 1
        
        assert 'avg_endpoint_latency' in key_metrics
        assert key_metrics['avg_endpoint_latency'] == 45.0
        
        assert 'requests_per_minute' in key_metrics
        assert key_metrics['requests_per_minute'] == 120.0


class TestAnalyticsIntegration:
    """Test suite for analytics integration with existing dashboard."""
    
    @pytest.fixture
    def dashboard(self):
        """Create a Dashboard for integration testing."""
        return UnifiedDashboard({
            'json': {'enabled': True, 'include_trends': True, 'pretty_print': True},
            'ascii': {'enabled': True, 'colored': True},
            'collection_interval_seconds': 30
        })
    
    @pytest.mark.asyncio
    async def test_analytics_data_structure(self, dashboard):
        """Test the structure of analytics data in JSON output."""
        base_metrics = {'system': {'cpu_percent': 45.0}}
        
        with patch('src.monitoring.request_metrics.get_metrics_collector') as mock_get_collector:
            mock_collector = Mock(spec=RequestMetricsCollector)
            
            # Mock trending data
            mock_collector.get_trending_data.return_value = [
                {
                    'timestamp': '2023-01-01T12:00:00',
                    'request_count': 25,
                    'error_count': 1,
                    'avg_duration_ms': 65.0,
                    'error_rate_percent': 4.0
                },
                {
                    'timestamp': '2023-01-01T12:01:00',
                    'request_count': 30,
                    'error_count': 0,
                    'avg_duration_ms': 55.0,
                    'error_rate_percent': 0.0
                }
            ]
            
            # Mock endpoint stats
            mock_collector.get_endpoint_stats.return_value = {
                'GET /api/users': {
                    'request_count': 45,
                    'error_count': 0,
                    'avg_duration_ms': 35.0,
                    'error_rate_percent': 0.0
                }
            }
            
            # Mock global stats
            mock_collector.get_global_stats.return_value = {
                'total_requests': 55,
                'total_errors': 1,
                'error_rate_percent': 1.8,
                'avg_duration_ms': 60.0,
                'p99_duration_ms': 120.0,
                'requests_per_minute': 27.5
            }
            
            mock_get_collector.return_value = mock_collector
            
            result_json = await dashboard.generate_json_dashboard_with_analytics(
                metrics=base_metrics,
                minutes=2
            )
            
            result = json.loads(result_json)
            
            # Verify analytics structure
            analytics = result['analytics']
            assert 'trending_data' in analytics
            assert 'endpoint_stats' in analytics
            assert 'global_stats' in analytics
            
            # Verify trending data
            trending = analytics['trending_data']
            assert len(trending) == 2
            assert trending[0]['request_count'] == 25
            assert trending[1]['error_rate_percent'] == 0.0
            
            # Verify endpoint stats
            endpoint_stats = analytics['endpoint_stats']
            assert 'GET /api/users' in endpoint_stats
            assert endpoint_stats['GET /api/users']['avg_duration_ms'] == 35.0
            
            # Verify global stats
            global_stats = analytics['global_stats']
            assert global_stats['total_requests'] == 55
            assert global_stats['error_rate_percent'] == 1.8
            
            # Verify insights are included
            assert 'insights' in result
            assert 'performance_status' in result['insights']
    
    @pytest.mark.asyncio
    async def test_analytics_graceful_failure(self, dashboard):
        """Test that analytics failures don't break the dashboard."""
        base_metrics = {'system': {'cpu_percent': 45.0}}
        
        with patch('src.monitoring.request_metrics.get_metrics_collector') as mock_get_collector:
            mock_collector = Mock(spec=RequestMetricsCollector)
            
            # Make collector methods fail
            mock_collector.get_trending_data.side_effect = Exception("Connection failed")
            mock_collector.get_endpoint_stats.side_effect = Exception("Timeout")
            mock_collector.get_global_stats.side_effect = Exception("Service unavailable")
            
            mock_get_collector.return_value = mock_collector
            
            # Should not raise exception
            result_json = await dashboard.generate_json_dashboard_with_analytics(
                metrics=base_metrics
            )
            
            result = json.loads(result_json)
            
            # Base metrics should still be present
            assert 'system' in result
            assert result['system']['cpu_percent'] == 45.0
            
            # Analytics might be missing or empty, but dashboard should work
            # This tests graceful degradation
    
    @pytest.mark.asyncio
    async def test_analytics_time_window_parameter(self, dashboard):
        """Test analytics with different time windows."""
        base_metrics = {'system': {'cpu_percent': 45.0}}
        
        with patch('src.monitoring.request_metrics.get_metrics_collector') as mock_get_collector:
            mock_collector = Mock(spec=RequestMetricsCollector)
            mock_collector.get_trending_data.return_value = []
            mock_collector.get_endpoint_stats.return_value = {}
            mock_collector.get_global_stats.return_value = {}
            mock_get_collector.return_value = mock_collector
            
            # Test different time windows
            for minutes in [1, 5, 15, 60]:
                result_json = await dashboard.generate_json_dashboard_with_analytics(
                    metrics=base_metrics,
                    minutes=minutes
                )
                
                # Should call get_trending_data with correct parameter
                mock_collector.get_trending_data.assert_called_with(minutes)
                
                # Should produce valid JSON
                result = json.loads(result_json)
                assert 'system' in result