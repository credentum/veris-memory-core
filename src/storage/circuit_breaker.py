#!/usr/bin/env python3
"""
circuit_breaker.py: Circuit Breaker Pattern for Sprint 11 Phase 4

Implements circuit breaker pattern to handle MCP server outages gracefully
with fail-closed behavior and automatic recovery detection.
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Optional, Any, Callable, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast, not calling service
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 30.0      # Seconds before trying recovery
    timeout: float = 5.0                # Request timeout in seconds
    success_threshold: int = 3          # Successes needed to close from half-open


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """Circuit breaker implementation for MCP service calls"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 30.0,
                 timeout: float = 5.0,
                 success_threshold: int = 3):
        """Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            timeout: Request timeout in seconds
            success_threshold: Consecutive successes needed to close circuit
        """
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            timeout=timeout,
            success_threshold=success_threshold
        )
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None
        
        # Metrics tracking
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_transitions: Dict[str, int] = {
            "closed_to_open": 0,
            "open_to_half_open": 0,
            "half_open_to_closed": 0,
            "half_open_to_open": 0
        }
    
    def _can_attempt_call(self) -> bool:
        """Check if we can attempt a call based on current state"""
        now = datetime.utcnow()
        
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (self.next_attempt_time and now >= self.next_attempt_time):
                self._transition_to_half_open()
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN"""
        logger.info("Circuit breaker transitioning from OPEN to HALF_OPEN")
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.state_transitions["open_to_half_open"] += 1
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        logger.warning(f"Circuit breaker OPENING after {self.failure_count} failures")
        self.state = CircuitState.OPEN
        self.last_failure_time = datetime.utcnow()
        self.next_attempt_time = self.last_failure_time + timedelta(
            seconds=self.config.recovery_timeout
        )
        
        if self.state == CircuitState.CLOSED:
            self.state_transitions["closed_to_open"] += 1
        else:
            self.state_transitions["half_open_to_open"] += 1
    
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        logger.info("Circuit breaker CLOSING - service recovered")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.state_transitions["half_open_to_closed"] += 1
    
    def _record_success(self):
        """Record a successful call"""
        self.total_requests += 1
        self.total_successes += 1
        self.failure_count = 0  # Reset failure count on success
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.info(f"Circuit breaker half-open success {self.success_count}/{self.config.success_threshold}")
            
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
    
    def _record_failure(self, exception: Exception):
        """Record a failed call"""
        self.total_requests += 1
        self.total_failures += 1
        self.failure_count += 1
        
        logger.warning(f"Circuit breaker failure {self.failure_count}/{self.config.failure_threshold}: {exception}")
        
        if self.state == CircuitState.CLOSED or self.state == CircuitState.HALF_OPEN:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
    
    async def __aenter__(self):
        """Async context manager entry"""
        if not self._can_attempt_call():
            raise CircuitBreakerError(
                f"Circuit breaker is {self.state.value}. "
                f"Next attempt at: {self.next_attempt_time}"
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if exc_type is None:
            # Success
            self._record_success()
        else:
            # Failure
            self._record_failure(exc_val)
        return False  # Don't suppress exceptions
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call a function through the circuit breaker"""
        async with self:
            # Apply timeout to the call
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError as e:
                logger.error(f"Circuit breaker timeout after {self.config.timeout}s")
                raise e
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "failure_rate": (
                self.total_failures / max(self.total_requests, 1)
            ),
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "next_attempt_time": (
                self.next_attempt_time.isoformat() if self.next_attempt_time else None
            ),
            "state_transitions": self.state_transitions.copy(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "timeout": self.config.timeout,
                "success_threshold": self.config.success_threshold
            }
        }
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        logger.info("Circuit breaker manually reset")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None


class MCPCircuitBreaker(CircuitBreaker):
    """Specialized circuit breaker for MCP server calls"""
    
    def __init__(self):
        # MCP-specific configuration
        super().__init__(
            failure_threshold=3,    # Fail fast for MCP
            recovery_timeout=10.0,  # Shorter recovery window
            timeout=5.0,           # 5 second timeout for MCP calls
            success_threshold=2    # Quick recovery for MCP
        )
        self.service_name = "MCP"
    
    def is_mcp_available(self) -> bool:
        """Quick check if MCP service is available"""
        return self.state == CircuitState.CLOSED or self._can_attempt_call()
    
    async def call_mcp_service(self, service_call: Callable, *args, **kwargs) -> Any:
        """Call MCP service through circuit breaker with MCP-specific handling"""
        try:
            return await self.call(service_call, *args, **kwargs)
        except CircuitBreakerError as e:
            logger.error(f"MCP service unavailable: {e}")
            # Re-raise with MCP-specific context
            raise ConnectionError(f"MCP service circuit breaker is {self.state.value}") from e
        except Exception as e:
            logger.error(f"MCP service call failed: {e}")
            raise


# Global MCP circuit breaker instance
mcp_circuit_breaker = MCPCircuitBreaker()


async def with_mcp_circuit_breaker(service_call: Callable, *args, **kwargs) -> Any:
    """Convenience function to call MCP services with circuit breaker protection"""
    return await mcp_circuit_breaker.call_mcp_service(service_call, *args, **kwargs)


def get_mcp_service_health() -> Dict[str, Any]:
    """Get MCP service health from circuit breaker perspective"""
    stats = mcp_circuit_breaker.get_stats()
    
    health_status = {
        "service": "MCP",
        "available": mcp_circuit_breaker.is_mcp_available(),
        "circuit_state": stats["state"],
        "failure_rate": stats["failure_rate"],
        "total_requests": stats["total_requests"],
        "recent_failures": stats["failure_count"],
        "last_failure": stats["last_failure_time"],
        "recovery_time": stats["next_attempt_time"] if stats["state"] == "open" else None
    }
    
    return health_status