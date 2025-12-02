"""
Veris Sentinel - Modular Monitoring System

This package contains the modular architecture for Veris Sentinel,
split from the original monolithic veris_sentinel.py into focused components.
"""

from .models import CheckResult, SentinelConfig
from .runner import SentinelRunner
from .api import SentinelAPI
from .checks import *

__all__ = [
    'CheckResult',
    'SentinelConfig', 
    'SentinelRunner',
    'SentinelAPI',
    # Check classes are exported via checks.__all__
]