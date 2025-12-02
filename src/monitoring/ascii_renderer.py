#!/usr/bin/env python3
"""
ASCII Renderer for Veris Memory Dashboard

Provides beautiful ASCII dashboard rendering with:
- Progress bars with dynamic scaling
- Emoji health indicators
- Trend arrows for metrics
- Color coding for status levels
"""

import math
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union


class ASCIIRenderer:
    """
    Renders dashboard metrics in beautiful ASCII format for human operators.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ASCII renderer with configuration."""
        default_config = {
            'width': 80,
            'use_color': True,
            'use_emoji': True,
            'progress_bar_width': 10
        }
        self.config = config or default_config
        
        # Detect terminal capabilities and adjust config
        self._detect_terminal_capabilities()
        
        # Color codes (if enabled)
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'magenta': '\033[95m'
        } if self.config['use_color'] else {k: '' for k in ['reset', 'bold', 'green', 'yellow', 'red', 'blue', 'cyan', 'magenta']}

        # Define emoji mappings
        emoji_mappings = {
            'target': '🎯',
            'memory': '💾',
            'disk': '💽',
            'cpu': '💻',
            'network': '🌐',
            'load': '⚡',
            'services': '🔧',
            'brain': '🧠',
            'security': '🛡️',
            'backup': '💾',
            'healthy': '✅',
            'warning': '⚠️',
            'critical': '❌',
            'unknown': '❓',
            'fast': '⚡',
            'excellent': '🎯',
            'robot': '🤖',
            'trend_up': '↗️',
            'trend_down': '↘️',
            'trend_flat': '→',
            'fire': '🚨',
            'shield': '🛡️',
            'lock': '🔒'
        }
        
        # Emoji indicators (if enabled)
        self.emojis = emoji_mappings if self.config['use_emoji'] else {k: '' for k in emoji_mappings}

    def render_dashboard(self, metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> str:
        """
        Render complete ASCII dashboard.
        
        Args:
            metrics: Complete metrics dictionary
            thresholds: Threshold configuration for status determination
            
        Returns:
            Complete ASCII dashboard as string
        """
        lines = []
        
        # Header
        timestamp = datetime.fromisoformat(metrics['timestamp'].replace('Z', '+00:00'))
        header = f"{self.emojis['target']} VERIS MEMORY STATUS - {timestamp.strftime('%a %b %d %H:%M UTC')}"
        lines.append(self._render_header(header))
        
        # System resources section
        lines.append("")
        lines.append(f"{self.colors['bold']}{self.emojis['memory']} SYSTEM RESOURCES{self.colors['reset']}")
        lines.extend(self._render_system_metrics(metrics.get('system', {}), thresholds))
        
        # Service health section
        lines.append("")
        lines.append(f"{self.colors['bold']}{self.emojis['services']} SERVICE HEALTH{self.colors['reset']}")
        lines.extend(self._render_service_metrics(metrics.get('services', []), thresholds))
        
        # Veris metrics section
        lines.append("")
        lines.append(f"{self.colors['bold']}{self.emojis['brain']} VERIS METRICS{self.colors['reset']}")
        lines.extend(self._render_veris_metrics(metrics.get('veris', {}), thresholds))
        
        # Security status section
        lines.append("")
        lines.append(f"{self.colors['bold']}{self.emojis['security']} SECURITY STATUS{self.colors['reset']}")
        lines.extend(self._render_security_metrics(metrics.get('security', {})))
        
        # Backup status section
        lines.append("")
        lines.append(f"{self.colors['bold']}{self.emojis['backup']} BACKUP STATUS{self.colors['reset']}")
        lines.extend(self._render_backup_metrics(metrics.get('backups', {})))
        
        # Footer
        lines.append("")
        lines.append(self._render_footer())
        
        return '\n'.join(lines)

    def _detect_terminal_capabilities(self) -> None:
        """Detect terminal capabilities and adjust configuration accordingly."""
        try:
            # Check if we're in a terminal
            if not sys.stdout.isatty():
                self.config['use_color'] = False
                self.config['use_emoji'] = False
                return
            
            # Check for color support
            if not self._supports_color():
                self.config['use_color'] = False
            
            # Check for emoji support
            if not self._supports_emoji():
                self.config['use_emoji'] = False
                
            # Adjust width based on terminal size
            terminal_width = self._get_terminal_width()
            if terminal_width and terminal_width < self.config['width']:
                self.config['width'] = max(60, terminal_width - 4)  # Leave some margin
                
        except Exception:
            # Fall back to safe defaults on any detection error
            self.config['use_color'] = False
            self.config['use_emoji'] = False

    def _supports_color(self) -> bool:
        """Check if terminal supports color codes."""
        try:
            # Check common environment variables for color support
            if os.environ.get('NO_COLOR'):
                return False
                
            term = os.environ.get('TERM', '').lower()
            if any(term.startswith(prefix) for prefix in ['xterm', 'screen', 'tmux', 'rxvt']):
                return True
                
            colorterm = os.environ.get('COLORTERM', '').lower()
            if colorterm in ['truecolor', '24bit', 'yes']:
                return True
                
            # Check if ANSI color codes work by testing capability
            if term and 'color' in term:
                return True
                
            return False
            
        except Exception:
            return False

    def _supports_emoji(self) -> bool:
        """Check if terminal supports emoji display."""
        try:
            # Check environment variables and locale
            lang = os.environ.get('LANG', '').lower()
            lc_all = os.environ.get('LC_ALL', '').lower()
            
            # UTF-8 environments generally support emoji
            if any('utf' in var for var in [lang, lc_all]):
                return True
                
            # Check terminal emulator
            term_program = os.environ.get('TERM_PROGRAM', '').lower()
            if term_program in ['iterm.app', 'hyper', 'vscode', 'terminus']:
                return True
                
            # Windows Terminal and modern terminals
            wt_session = os.environ.get('WT_SESSION')
            if wt_session:
                return True
                
            # Conservative fallback - assume emoji not supported
            return False
            
        except Exception:
            return False

    def _get_terminal_width(self) -> Optional[int]:
        """Get terminal width if available."""
        try:
            # Try to get terminal size
            import shutil
            return shutil.get_terminal_size().columns
        except Exception:
            try:
                # Fallback to environment variables
                columns = os.environ.get('COLUMNS')
                if columns and columns.isdigit():
                    return int(columns)
            except Exception:
                pass
            return None

    def _render_header(self, title: str) -> str:
        """Render dashboard header with border."""
        width = self.config['width']
        border = '═' * width
        
        # Center the title
        padding = (width - len(title)) // 2
        centered_title = ' ' * padding + title + ' ' * (width - padding - len(title))
        
        return f"{self.colors['bold']}{self.colors['cyan']}{centered_title}{self.colors['reset']}\n{border}"

    def _render_footer(self) -> str:
        """Render dashboard footer."""
        width = self.config['width']
        return '═' * width

    def _render_system_metrics(self, system: Dict[str, Any], thresholds: Dict[str, Any]) -> List[str]:
        """Render system resource metrics."""
        lines = []
        
        if not system:
            lines.append("No system metrics available")
            return lines
        
        # CPU
        cpu_percent = system.get('cpu_percent', 0)
        cpu_status = self._get_status_indicator(cpu_percent, thresholds, 'cpu')
        cpu_bar = self._render_progress_bar(cpu_percent, 100)
        lines.append(f"CPU    {cpu_bar} {cpu_percent}% {cpu_status}")
        
        # Memory
        memory_percent = system.get('memory_percent', 0)
        memory_used = system.get('memory_used_gb', 0)
        memory_total = system.get('memory_total_gb', 0)
        memory_status = self._get_status_indicator(memory_percent, thresholds, 'memory')
        memory_bar = self._render_progress_bar(memory_percent, 100)
        lines.append(f"Memory {memory_bar} {memory_percent}% ({memory_used:.1f}GB/{memory_total:.1f}GB) {memory_status}")
        
        # Disk
        disk_percent = system.get('disk_percent', 0)
        disk_used = system.get('disk_used_gb', 0)
        disk_total = system.get('disk_total_gb', 0)
        disk_status = self._get_status_indicator(disk_percent, thresholds, 'disk')
        disk_bar = self._render_progress_bar(disk_percent, 100)
        lines.append(f"Disk   {disk_bar} {disk_percent}% ({disk_used:.1f}GB/{disk_total:.1f}GB) {disk_status}")
        
        # Load average
        load_avg = system.get('load_average', [0, 0, 0])
        if load_avg and len(load_avg) >= 3:
            load_status = self._get_load_status_indicator(load_avg[0])
            lines.append(f"Load   {self.emojis['load']} {load_avg[0]:.2f} {load_avg[1]:.2f} {load_avg[2]:.2f} {load_status}")
        
        # Uptime
        uptime_hours = system.get('uptime_hours', 0)
        uptime_days = uptime_hours / 24
        if uptime_days >= 1:
            lines.append(f"Uptime {uptime_days:.1f} days")
        else:
            lines.append(f"Uptime {uptime_hours:.1f} hours")
        
        return lines

    def _render_service_metrics(self, services: List[Dict[str, Any]], thresholds: Dict[str, Any]) -> List[str]:
        """Render service health metrics."""
        lines = []
        
        if not services:
            lines.append("No service metrics available")
            return lines
        
        for service in services:
            name = service.get('name', 'Unknown')
            status = service.get('status', 'unknown')
            port = service.get('port')
            
            # Status indicator
            status_emoji = self._get_service_status_emoji(status)
            
            # Additional metrics
            metrics_parts = []
            if service.get('memory_mb'):
                metrics_parts.append(f"{service['memory_mb']:.1f}MB")
            if service.get('operations_per_sec'):
                metrics_parts.append(f"{service['operations_per_sec']} ops/s")
            if service.get('connections'):
                metrics_parts.append(f"{service['connections']} conn")
            
            metrics_str = " | ".join(metrics_parts)
            if metrics_str:
                metrics_str = " | " + metrics_str
            
            lines.append(f"{name:<12} {status_emoji} {status.title()}{metrics_str}")
        
        return lines

    def _render_veris_metrics(self, veris: Dict[str, Any], thresholds: Dict[str, Any]) -> List[str]:
        """Render Veris Memory specific metrics."""
        lines = []
        
        if not veris:
            lines.append("No Veris metrics available")
            return lines
        
        # Total memories with daily growth
        total_memories = veris.get('total_memories', 0)
        memories_today = veris.get('memories_today', 0)
        trend_arrow = self._get_trend_arrow(memories_today)
        lines.append(f"Total Memories: {total_memories:,} (+{memories_today:,} today {trend_arrow})")
        
        # Latency metrics
        avg_latency = veris.get('avg_query_latency_ms', 0)
        p99_latency = veris.get('p99_latency_ms', 0)
        latency_status = self._get_latency_status(avg_latency, thresholds)
        lines.append(f"Query Latency:  {avg_latency:.1f}ms avg | {p99_latency:.1f}ms p99 {latency_status}")
        
        # Error rate
        error_rate = veris.get('error_rate_percent', 0)
        error_status = self._get_error_rate_status(error_rate, thresholds)
        lines.append(f"Error Rate:     {error_rate:.3f}% {error_status}")
        
        # Active agents
        active_agents = veris.get('active_agents', 0)
        lines.append(f"Active Agents:  {active_agents} {self.emojis['robot']}")
        
        # Operation counts
        successful_ops = veris.get('successful_operations_24h', 0)
        failed_ops = veris.get('failed_operations_24h', 0)
        total_ops = successful_ops + failed_ops
        if total_ops > 0:
            success_rate = (successful_ops / total_ops) * 100
            lines.append(f"Operations 24h: {total_ops:,} ({success_rate:.1f}% success)")
        
        return lines

    def _render_security_metrics(self, security: Dict[str, Any]) -> List[str]:
        """Render security and compliance metrics."""
        lines = []
        
        if not security:
            lines.append("No security metrics available")
            return lines
        
        # Authentication failures
        auth_failures = security.get('failed_auth_attempts', 0)
        auth_status = self.emojis['healthy'] if auth_failures == 0 else self.emojis['warning']
        lines.append(f"Auth Failures:  {auth_failures} {auth_status}")
        
        # Blocked IPs
        blocked_ips = security.get('blocked_ips', 0)
        blocked_status = self.emojis['warning'] if blocked_ips > 0 else self.emojis['healthy']
        lines.append(f"Blocked IPs:    {blocked_ips} {blocked_status}")
        
        # WAF blocks
        waf_blocks = security.get('waf_blocks_today', 0)
        lines.append(f"WAF Blocks:     {waf_blocks} today")
        
        # SSL certificate expiry
        ssl_days = security.get('ssl_cert_expiry_days', 0)
        ssl_status = self._get_ssl_status(ssl_days)
        lines.append(f"SSL Expires:    {ssl_days} days {ssl_status}")
        
        # RBAC violations
        rbac_violations = security.get('rbac_violations', 0)
        rbac_status = self.emojis['healthy'] if rbac_violations == 0 else self.emojis['critical']
        lines.append(f"RBAC Violations: {rbac_violations} {rbac_status}")
        
        return lines

    def _render_backup_metrics(self, backups: Dict[str, Any]) -> List[str]:
        """Render backup and disaster recovery metrics."""
        lines = []
        
        if not backups:
            lines.append("No backup metrics available")
            return lines
        
        # Last backup
        if 'last_backup_time' in backups:
            last_backup = datetime.fromisoformat(str(backups['last_backup_time']).replace('Z', '+00:00'))
            hours_ago = (datetime.utcnow().replace(tzinfo=last_backup.tzinfo) - last_backup).total_seconds() / 3600
            backup_status = self.emojis['healthy'] if hours_ago < 25 else self.emojis['warning']
            lines.append(f"Last Backup:    {hours_ago:.0f}h ago {backup_status}")
        
        # Backup size
        backup_size = backups.get('backup_size_gb', 0)
        lines.append(f"Backup Size:    {backup_size:.1f} GB")
        
        # Restore testing
        restore_tested = backups.get('restore_tested', False)
        restore_time = backups.get('last_restore_time_seconds', 0)
        restore_status = self.emojis['healthy'] if restore_tested and restore_time < 300 else self.emojis['warning']
        if restore_tested:
            lines.append(f"Restore Test:   PASSED ({restore_time:.0f}s) {restore_status}")
        else:
            lines.append(f"Restore Test:   NOT TESTED {self.emojis['warning']}")
        
        # Offsite sync
        sync_status = backups.get('offsite_sync_status', 'unknown')
        sync_emoji = self.emojis['healthy'] if sync_status == 'healthy' else self.emojis['warning']
        lines.append(f"Offsite Sync:   {sync_status.upper()} {sync_emoji}")
        
        return lines


    def _render_progress_bar(self, value: float, max_value: float, width: Optional[int] = None) -> str:
        """
        Render a progress bar for the given value.
        
        Args:
            value: Current value
            max_value: Maximum value
            width: Width of progress bar (uses config default if not provided)
            
        Returns:
            ASCII progress bar string
        """
        if width is None:
            width = self.config['progress_bar_width']
        
        if max_value <= 0:
            percentage = 0
        else:
            percentage = min(value / max_value, 1.0)
        
        filled_width = int(percentage * width)
        empty_width = width - filled_width
        
        # Use different characters for different fill levels
        filled_char = '█'
        empty_char = '░'
        
        bar = f"[{filled_char * filled_width}{empty_char * empty_width}]"
        return bar

    def _get_status_indicator(self, value: float, thresholds: Dict[str, Any], metric_type: str) -> str:
        """Get status indicator emoji based on thresholds."""
        warning_threshold = thresholds.get(f'{metric_type}_warning_percent', 80)
        critical_threshold = thresholds.get(f'{metric_type}_critical_percent', 95)
        
        if value >= critical_threshold:
            return f"{self.colors['red']}{self.emojis['critical']} CRITICAL{self.colors['reset']}"
        elif value >= warning_threshold:
            return f"{self.colors['yellow']}{self.emojis['warning']} WARNING{self.colors['reset']}"
        else:
            return f"{self.colors['green']}{self.emojis['healthy']} HEALTHY{self.colors['reset']}"

    def _get_service_status_emoji(self, status: str) -> str:
        """Get emoji for service status."""
        status_map = {
            'healthy': self.emojis['healthy'],
            'warning': self.emojis['warning'],
            'critical': self.emojis['critical'],
            'error': self.emojis['critical'],
            'unknown': self.emojis['unknown']
        }
        return status_map.get(status.lower(), self.emojis['unknown'])

    def _get_load_status_indicator(self, load: float) -> str:
        """Get status indicator for system load."""
        if load > 4.0:
            return f"{self.colors['red']}{self.emojis['critical']} HIGH{self.colors['reset']}"
        elif load > 2.0:
            return f"{self.colors['yellow']}{self.emojis['warning']} ELEVATED{self.colors['reset']}"
        else:
            return f"{self.colors['green']}{self.emojis['healthy']} LOW{self.colors['reset']}"

    def _get_latency_status(self, latency_ms: float, thresholds: Dict[str, Any]) -> str:
        """Get status indicator for latency."""
        warning_ms = thresholds.get('latency_warning_ms', 100)
        critical_ms = thresholds.get('latency_critical_ms', 500)
        
        if latency_ms >= critical_ms:
            return f"{self.colors['red']}{self.emojis['critical']} SLOW{self.colors['reset']}"
        elif latency_ms >= warning_ms:
            return f"{self.colors['yellow']}{self.emojis['warning']} MODERATE{self.colors['reset']}"
        else:
            return f"{self.colors['green']}{self.emojis['fast']} FAST{self.colors['reset']}"

    def _get_error_rate_status(self, error_rate: float, thresholds: Dict[str, Any]) -> str:
        """Get status indicator for error rate."""
        warning_rate = thresholds.get('error_rate_warning_percent', 1.0)
        critical_rate = thresholds.get('error_rate_critical_percent', 5.0)
        
        if error_rate >= critical_rate:
            return f"{self.colors['red']}{self.emojis['critical']} HIGH{self.colors['reset']}"
        elif error_rate >= warning_rate:
            return f"{self.colors['yellow']}{self.emojis['warning']} ELEVATED{self.colors['reset']}"
        else:
            return f"{self.colors['green']}{self.emojis['excellent']} EXCELLENT{self.colors['reset']}"

    def _get_ssl_status(self, days_until_expiry: int) -> str:
        """Get status indicator for SSL certificate expiry."""
        if days_until_expiry < 7:
            return f"{self.colors['red']}{self.emojis['critical']} URGENT{self.colors['reset']}"
        elif days_until_expiry < 30:
            return f"{self.colors['yellow']}{self.emojis['warning']} SOON{self.colors['reset']}"
        else:
            return f"{self.colors['green']}{self.emojis['healthy']} OK{self.colors['reset']}"

    def _get_trend_arrow(self, value: Union[int, float]) -> str:
        """Get trend arrow for metrics."""
        if value > 100:
            return self.emojis['trend_up']
        elif value < -100:
            return self.emojis['trend_down']
        else:
            return self.emojis['trend_flat']

    def format_number(self, value: Union[int, float], unit: str = '') -> str:
        """Format numbers with appropriate units and separators."""
        if isinstance(value, float):
            if value >= 1000000:
                return f"{value/1000000:.1f}M{unit}"
            elif value >= 1000:
                return f"{value/1000:.1f}K{unit}"
            else:
                return f"{value:.1f}{unit}"
        else:
            if value >= 1000000:
                return f"{value/1000000:.1f}M{unit}"
            elif value >= 1000:
                return f"{value/1000:.1f}K{unit}"
            else:
                return f"{value:,}{unit}"


# Export main class
__all__ = ["ASCIIRenderer"]