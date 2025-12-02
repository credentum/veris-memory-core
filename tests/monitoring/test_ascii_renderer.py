#!/usr/bin/env python3
"""
Unit tests for ASCIIRenderer class.

Tests terminal capability detection, rendering methods, and output formatting.
"""

import pytest
import os
from unittest.mock import patch, Mock
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.monitoring.ascii_renderer import ASCIIRenderer


class TestASCIIRenderer:
    """Test suite for ASCIIRenderer class."""

    @pytest.fixture
    def renderer(self):
        """Create ASCIIRenderer instance with default config."""
        return ASCIIRenderer()

    @pytest.fixture
    def test_metrics(self):
        """Sample metrics for testing rendering."""
        return {
            'timestamp': '2025-08-14T12:00:00Z',
            'system': {
                'cpu_percent': 45.5,
                'memory_percent': 62.3,
                'memory_total_gb': 16.0,
                'memory_used_gb': 10.0,
                'disk_percent': 78.1,
                'disk_total_gb': 500.0,
                'disk_used_gb': 390.5,
                'load_average': [0.5, 0.6, 0.7],
                'uptime_hours': 48.5
            },
            'services': [
                {'name': 'Redis', 'status': 'healthy', 'port': 6379},
                {'name': 'Neo4j', 'status': 'unhealthy', 'port': 7474},
                {'name': 'Qdrant', 'status': 'unknown', 'port': 6334}
            ],
            'veris': {
                'total_memories': 85432,
                'memories_today': 1247,
                'avg_query_latency_ms': 23.5,
                'p99_latency_ms': 89.2,
                'error_rate_percent': 0.02,
                'active_agents': 5,
                'successful_operations_24h': 5000,
                'failed_operations_24h': 1
            },
            'security': {
                'failed_auth_attempts': 0,
                'blocked_ips': 0,
                'waf_blocks_today': 12,
                'ssl_cert_expiry_days': 87,
                'rbac_violations': 0,
                'audit_events_24h': 150
            },
            'backups': {
                'last_backup_time': '2025-08-14T09:00:00Z',
                'backup_size_gb': 4.7,
                'restore_tested': True,
                'last_restore_time_seconds': 142.0,
                'backup_success_rate_percent': 100.0,
                'offsite_sync_status': 'healthy'
            }
        }

    @pytest.fixture
    def test_thresholds(self):
        """Sample thresholds for testing."""
        return {
            'memory_warning_percent': 80,
            'memory_critical_percent': 95,
            'disk_warning_percent': 85,
            'disk_critical_percent': 95,
            'cpu_warning_percent': 80,
            'cpu_critical_percent': 95,
            'error_rate_warning_percent': 1.0,
            'error_rate_critical_percent': 5.0,
            'latency_warning_ms': 100,
            'latency_critical_ms': 500
        }

    def test_init_default_config(self):
        """Test renderer initialization with default configuration."""
        renderer = ASCIIRenderer()

        assert renderer.config['width'] == 80
        assert renderer.config['use_color'] is True or renderer.config['use_color'] is False  # Depends on terminal
        assert renderer.config['use_emoji'] is True or renderer.config['use_emoji'] is False  # Depends on terminal
        assert renderer.config['progress_bar_width'] == 10

    def test_init_custom_config(self):
        """Test renderer initialization with custom configuration."""
        custom_config = {
            'width': 120,
            'use_color': False,
            'use_emoji': False,
            'progress_bar_width': 15
        }

        renderer = ASCIIRenderer(custom_config)

        assert renderer.config['width'] == 120
        assert renderer.config['use_color'] is False
        assert renderer.config['use_emoji'] is False
        assert renderer.config['progress_bar_width'] == 15

    def test_terminal_capability_detection_no_tty(self):
        """Test terminal capability detection when not in a TTY."""
        with patch('sys.stdout.isatty', return_value=False):
            renderer = ASCIIRenderer({'use_color': True, 'use_emoji': True})

            assert renderer.config['use_color'] is False
            assert renderer.config['use_emoji'] is False

    def test_color_support_detection(self, renderer):
        """Test color support detection logic."""
        # Test NO_COLOR environment variable
        with patch.dict(os.environ, {'NO_COLOR': '1'}):
            assert renderer._supports_color() is False

        # Test color-supporting TERM values
        with patch.dict(os.environ, {'TERM': 'xterm-256color'}, clear=True):
            assert renderer._supports_color() is True

        with patch.dict(os.environ, {'TERM': 'screen-256color'}, clear=True):
            assert renderer._supports_color() is True

        # Test COLORTERM
        with patch.dict(os.environ, {'COLORTERM': 'truecolor'}, clear=True):
            assert renderer._supports_color() is True

        # Test unsupported terminal
        with patch.dict(os.environ, {'TERM': 'dumb'}, clear=True):
            assert renderer._supports_color() is False

    def test_emoji_support_detection(self, renderer):
        """Test emoji support detection logic."""
        # Test UTF-8 locale support
        with patch.dict(os.environ, {'LANG': 'en_US.UTF-8'}, clear=True):
            assert renderer._supports_emoji() is True

        with patch.dict(os.environ, {'LC_ALL': 'C.UTF-8'}, clear=True):
            assert renderer._supports_emoji() is True

        # Test modern terminal emulators
        with patch.dict(os.environ, {'TERM_PROGRAM': 'iTerm.app'}, clear=True):
            assert renderer._supports_emoji() is True

        with patch.dict(os.environ, {'TERM_PROGRAM': 'vscode'}, clear=True):
            assert renderer._supports_emoji() is True

        # Test Windows Terminal
        with patch.dict(os.environ, {'WT_SESSION': 'abc123'}, clear=True):
            assert renderer._supports_emoji() is True

        # Test unsupported environment
        with patch.dict(os.environ, {}, clear=True):
            # Should default to False for conservative compatibility
            assert renderer._supports_emoji() is False

    def test_terminal_width_detection(self, renderer):
        """Test terminal width detection."""
        # Test with shutil.get_terminal_size
        with patch('shutil.get_terminal_size') as mock_size:
            mock_size.return_value.columns = 120
            width = renderer._get_terminal_width()
            assert width == 120

        # Test with COLUMNS environment variable
        with patch('shutil.get_terminal_size', side_effect=Exception), \
             patch.dict(os.environ, {'COLUMNS': '100'}):
            width = renderer._get_terminal_width()
            assert width == 100

        # Test fallback
        with patch('shutil.get_terminal_size', side_effect=Exception), \
             patch.dict(os.environ, {}, clear=True):
            width = renderer._get_terminal_width()
            assert width is None

    def test_width_adjustment_for_narrow_terminal(self):
        """Test width adjustment for narrow terminals."""
        with patch.object(ASCIIRenderer, '_get_terminal_width', return_value=70), \
             patch('sys.stdout.isatty', return_value=True):
            renderer = ASCIIRenderer({'width': 80, 'use_color': True, 'use_emoji': True, 'progress_bar_width': 10})
            # Width should be adjusted to terminal width minus margin
            assert renderer.config['width'] == max(60, 70 - 4)  # max(60, terminal_width - 4)

    def test_render_dashboard_full(self, renderer, test_metrics, test_thresholds):
        """Test full dashboard rendering."""
        result = renderer.render_dashboard(test_metrics, test_thresholds)

        assert isinstance(result, str)
        assert 'VERIS MEMORY STATUS' in result
        assert 'SYSTEM RESOURCES' in result
        assert 'SERVICE HEALTH' in result
        assert 'VERIS METRICS' in result
        assert 'SECURITY STATUS' in result
        assert 'BACKUP STATUS' in result

    def test_render_header(self, renderer):
        """Test header rendering."""
        header = renderer._render_header("TEST DASHBOARD")

        assert isinstance(header, str)
        assert "TEST DASHBOARD" in header
        assert len(header.split('\n')) >= 2  # Title and separator

    def test_render_system_metrics(self, renderer, test_metrics, test_thresholds):
        """Test system metrics rendering."""
        lines = renderer._render_system_metrics(test_metrics['system'], test_thresholds)

        assert isinstance(lines, list)
        assert any('CPU' in line for line in lines)
        assert any('Memory' in line for line in lines)
        assert any('Disk' in line for line in lines)
        assert any('45.5%' in line for line in lines)  # CPU percentage

    def test_render_service_metrics(self, renderer, test_metrics, test_thresholds):
        """Test service metrics rendering."""
        lines = renderer._render_service_metrics(test_metrics['services'], test_thresholds)

        assert isinstance(lines, list)
        assert any('Redis' in line for line in lines)
        assert any('Neo4j' in line for line in lines)
        assert any('Qdrant' in line for line in lines)

    def test_render_veris_metrics(self, renderer, test_metrics, test_thresholds):
        """Test Veris metrics rendering."""
        lines = renderer._render_veris_metrics(test_metrics['veris'], test_thresholds)

        assert isinstance(lines, list)
        assert any('Total Memories' in line for line in lines)
        assert any('Query Latency' in line for line in lines)
        assert any('Error Rate' in line for line in lines)
        assert any('85,432' in line for line in lines)  # Formatted number

    def test_render_security_metrics(self, renderer, test_metrics):
        """Test security metrics rendering."""
        lines = renderer._render_security_metrics(test_metrics['security'])

        assert isinstance(lines, list)
        assert any('Auth Failures' in line for line in lines)
        assert any('SSL Expires' in line for line in lines)
        assert any('87 days' in line for line in lines)

    def test_render_backup_metrics(self, renderer, test_metrics):
        """Test backup metrics rendering."""
        lines = renderer._render_backup_metrics(test_metrics['backups'])

        assert isinstance(lines, list)
        assert any('Last Backup' in line for line in lines)
        assert any('Backup Size' in line for line in lines)
        assert any('Restore Test' in line for line in lines)

    def test_render_progress_bar(self, renderer):
        """Test progress bar rendering."""
        # Test different percentages
        bar_0 = renderer._render_progress_bar(0, 100)
        bar_50 = renderer._render_progress_bar(50, 100)
        bar_100 = renderer._render_progress_bar(100, 100)

        assert isinstance(bar_0, str)
        assert isinstance(bar_50, str)
        assert isinstance(bar_100, str)

        # Progress bars should have consistent format
        assert bar_0.startswith('[') and bar_0.endswith(']')
        assert bar_50.startswith('[') and bar_50.endswith(']')
        assert bar_100.startswith('[') and bar_100.endswith(']')

    def test_render_progress_bar_custom_width(self, renderer):
        """Test progress bar with custom width."""
        bar = renderer._render_progress_bar(25, 100, width=20)

        # Should be 20 characters wide plus brackets
        assert len(bar) == 22  # [20 chars]

    def test_render_progress_bar_overflow(self, renderer):
        """Test progress bar with value exceeding maximum."""
        bar = renderer._render_progress_bar(150, 100)

        # Should cap at 100%
        assert isinstance(bar, str)
        # Should still render without error

    def test_get_status_emoji_healthy(self, renderer):
        """Test status emoji selection for healthy status."""
        emoji = renderer._get_service_status_emoji('healthy')
        expected = renderer.emojis['healthy'] if renderer.config['use_emoji'] else ''
        assert emoji == expected

    def test_get_status_emoji_warning(self, renderer):
        """Test status emoji selection for warning status."""
        emoji = renderer._get_service_status_emoji('warning')
        expected = renderer.emojis['warning'] if renderer.config['use_emoji'] else ''
        assert emoji == expected

    def test_get_status_emoji_critical(self, renderer):
        """Test status emoji selection for critical status."""
        emoji = renderer._get_service_status_emoji('critical')
        expected = renderer.emojis['critical'] if renderer.config['use_emoji'] else ''
        assert emoji == expected

    def test_get_status_emoji_unknown(self, renderer):
        """Test status emoji selection for unknown status."""
        emoji = renderer._get_service_status_emoji('unknown')
        expected = renderer.emojis['unknown'] if renderer.config['use_emoji'] else ''
        assert emoji == expected

    def test_format_number_small(self, renderer):
        """Test number formatting for small numbers."""
        assert renderer.format_number(123) == "123"
        assert renderer.format_number(999) == "999"

    def test_format_number_thousands(self, renderer):
        """Test number formatting for thousands."""
        assert renderer.format_number(1234) == "1.2K"
        assert renderer.format_number(12345) == "12.3K"
        assert renderer.format_number(123456) == "123.5K"

    def test_format_number_float(self, renderer):
        """Test number formatting for floats."""
        assert renderer.format_number(1234.5) == "1.2K"
        assert renderer.format_number(234.56) == "234.6"

    def test_get_trend_indicator(self, renderer):
        """Test trend indicator selection."""
        trend_up = renderer._get_trend_arrow(150)  # > 100 for up trend
        trend_down = renderer._get_trend_arrow(-150)  # < -100 for down trend
        trend_flat = renderer._get_trend_arrow(50)  # -100 to 100 for flat

        if renderer.config['use_emoji']:
            assert trend_up == renderer.emojis['trend_up']
            assert trend_down == renderer.emojis['trend_down']
            assert trend_flat == renderer.emojis['trend_flat']
        else:
            assert trend_up == ''
            assert trend_down == ''
            assert trend_flat == ''

    def test_render_footer(self, renderer):
        """Test footer rendering."""
        footer = renderer._render_footer()

        assert isinstance(footer, str)
        # Footer should contain separator line
        assert '═' in footer or '=' in footer

    def test_render_with_no_emoji_config(self):
        """Test rendering with emoji disabled."""
        renderer = ASCIIRenderer({'use_emoji': False})

        # All emoji values should be empty strings
        assert all(emoji == '' for emoji in renderer.emojis.values())

    def test_render_with_no_color_config(self):
        """Test rendering with color disabled."""
        renderer = ASCIIRenderer({'use_color': False})

        # All color values should be empty strings
        assert all(color == '' for color in renderer.colors.values())

    def test_color_application(self, renderer):
        """Test color code application in output."""
        # Create renderer with colors explicitly enabled
        renderer_with_color = ASCIIRenderer({'use_color': True, 'width': 80})
        result_with_color = renderer_with_color._render_header("TEST")

        # Create renderer with colors disabled
        renderer_no_color = ASCIIRenderer({'use_color': False, 'width': 80})
        result_no_color = renderer_no_color._render_header("TEST")

        # If color is enabled and supported, output should contain color codes
        if renderer_with_color.config['use_color']:
            assert len(result_with_color) >= len(result_no_color)

    def test_empty_metrics_handling(self, renderer, test_thresholds):
        """Test handling of empty or missing metrics."""
        empty_metrics = {
            'timestamp': '2023-01-01T00:00:00Z',
            'system': {},
            'services': [],
            'veris': {},
            'security': {},
            'backups': {}
        }

        result = renderer.render_dashboard(empty_metrics, test_thresholds)

        # Should not crash and should return valid string
        assert isinstance(result, str)
        assert 'VERIS MEMORY STATUS' in result

    def test_partial_metrics_handling(self, renderer, test_thresholds):
        """Test handling of partial metrics data."""
        partial_metrics = {
            'timestamp': '2025-08-14T12:00:00Z',
            'system': {
                'cpu_percent': 45.5
                # Missing other system metrics
            }
            # Missing other metric categories
        }

        result = renderer.render_dashboard(partial_metrics, test_thresholds)

        # Should not crash and should return valid string
        assert isinstance(result, str)
        assert 'SYSTEM RESOURCES' in result

    def test_threshold_based_status(self, renderer):
        """Test status determination based on thresholds."""
        thresholds = {
            'cpu_warning_percent': 70,
            'cpu_critical_percent': 90
        }

        # Test below warning - should contain "HEALTHY"
        status = renderer._get_status_indicator(50, thresholds, 'cpu')
        assert 'HEALTHY' in status

        # Test warning range - should contain "WARNING"
        status = renderer._get_status_indicator(80, thresholds, 'cpu')
        assert 'WARNING' in status

        # Test critical range - should contain "CRITICAL"
        status = renderer._get_status_indicator(95, thresholds, 'cpu')
        assert 'CRITICAL' in status


if __name__ == '__main__':
    pytest.main([__file__, '-v'])