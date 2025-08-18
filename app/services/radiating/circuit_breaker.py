"""
Circuit Breaker for LLM calls in Radiating Coverage System

Implements the circuit breaker pattern to prevent cascading failures
when LLM services are experiencing issues.
"""

import asyncio
import logging
from typing import Any, Callable, Optional, Dict
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

from app.core.timeout_settings_cache import get_radiating_timeout

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.
    
    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    """
    
    def __init__(
        self,
        name: str = "llm_circuit_breaker",
        failure_threshold: float = None,
        recovery_timeout: int = None,
        half_open_requests: int = 1
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name for logging
            failure_threshold: Failure rate to trigger open state (0.0-1.0)
            recovery_timeout: Seconds before attempting recovery
            half_open_requests: Number of test requests in half-open state
        """
        self.name = name
        
        # Load configuration from timeout settings or use defaults
        self.failure_threshold = failure_threshold or get_radiating_timeout(
            "circuit_breaker_threshold", 0.5
        )
        self.recovery_timeout = recovery_timeout or get_radiating_timeout(
            "circuit_breaker_cooldown", 60
        )
        self.half_open_requests = half_open_requests
        
        # State management
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.state_changed_at = datetime.now()
        self.half_open_successes = 0
        
        # Sliding window for failure rate calculation
        self.window_size = 10  # Last N requests
        self.request_results: list = []  # True for success, False for failure
        
        logger.info(f"[{self.name}] Initialized with threshold={self.failure_threshold}, "
                   f"recovery_timeout={self.recovery_timeout}s")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpen: If circuit is open
            Original exception: If function fails and circuit allows
        """
        # Check if we should transition from OPEN to HALF_OPEN
        self._check_recovery()
        
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpen(
                f"Circuit breaker {self.name} is OPEN. "
                f"Service unavailable for {self._time_until_recovery()}s"
            )
        
        if self.state == CircuitState.HALF_OPEN:
            # In half-open state, limit the number of test requests
            if self.half_open_successes >= self.half_open_requests:
                # We've had enough successful test requests, close the circuit
                self._close_circuit()
        
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            raise
    
    def _check_recovery(self):
        """Check if circuit should transition from OPEN to HALF_OPEN"""
        if self.state == CircuitState.OPEN:
            time_since_open = (datetime.now() - self.state_changed_at).total_seconds()
            if time_since_open >= self.recovery_timeout:
                logger.info(f"[{self.name}] Attempting recovery, transitioning to HALF_OPEN")
                self._transition_to_half_open()
    
    def _record_success(self):
        """Record a successful call"""
        self.stats.total_calls += 1
        self.stats.successful_calls += 1
        self.stats.last_success_time = datetime.now()
        self.stats.consecutive_successes += 1
        self.stats.consecutive_failures = 0
        
        # Update sliding window
        self.request_results.append(True)
        if len(self.request_results) > self.window_size:
            self.request_results.pop(0)
        
        # Handle state transitions
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.half_open_requests:
                self._close_circuit()
    
    def _record_failure(self):
        """Record a failed call"""
        self.stats.total_calls += 1
        self.stats.failed_calls += 1
        self.stats.last_failure_time = datetime.now()
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        
        # Update sliding window
        self.request_results.append(False)
        if len(self.request_results) > self.window_size:
            self.request_results.pop(0)
        
        # Check if we should open the circuit
        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state reopens the circuit
            self._open_circuit()
        elif self.state == CircuitState.CLOSED:
            # Check failure rate
            if self._should_open_circuit():
                self._open_circuit()
    
    def _should_open_circuit(self) -> bool:
        """Check if failure rate exceeds threshold"""
        if len(self.request_results) < self.window_size // 2:
            # Not enough data yet
            return False
        
        failure_count = sum(1 for r in self.request_results if not r)
        failure_rate = failure_count / len(self.request_results)
        
        return failure_rate >= self.failure_threshold
    
    def _open_circuit(self):
        """Transition to OPEN state"""
        self.state = CircuitState.OPEN
        self.state_changed_at = datetime.now()
        logger.warning(f"[{self.name}] Circuit OPENED due to high failure rate. "
                      f"Stats: {self.get_stats()}")
    
    def _close_circuit(self):
        """Transition to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.state_changed_at = datetime.now()
        self.half_open_successes = 0
        logger.info(f"[{self.name}] Circuit CLOSED, normal operation resumed")
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.state_changed_at = datetime.now()
        self.half_open_successes = 0
    
    def _time_until_recovery(self) -> int:
        """Calculate seconds until recovery attempt"""
        if self.state != CircuitState.OPEN:
            return 0
        
        time_since_open = (datetime.now() - self.state_changed_at).total_seconds()
        return max(0, int(self.recovery_timeout - time_since_open))
    
    def get_state(self) -> str:
        """Get current circuit state"""
        return self.state.value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "state": self.state.value,
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "failure_rate": self._calculate_failure_rate(),
            "consecutive_failures": self.stats.consecutive_failures,
            "consecutive_successes": self.stats.consecutive_successes,
            "time_until_recovery": self._time_until_recovery() if self.state == CircuitState.OPEN else None
        }
    
    def _calculate_failure_rate(self) -> float:
        """Calculate current failure rate"""
        if not self.request_results:
            return 0.0
        
        failure_count = sum(1 for r in self.request_results if not r)
        return failure_count / len(self.request_results)
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.state_changed_at = datetime.now()
        self.half_open_successes = 0
        self.request_results = []
        logger.info(f"[{self.name}] Circuit breaker reset")


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class LLMCircuitBreakerMixin:
    """
    Mixin class to add circuit breaker protection to LLM calls.
    
    Usage:
        class MyService(LLMCircuitBreakerMixin):
            async def call_llm(self, prompt):
                return await self.protected_llm_call(
                    self.llm_client.invoke,
                    prompt
                )
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._circuit_breaker = CircuitBreaker(
            name=f"{self.__class__.__name__}_circuit_breaker"
        )
    
    async def protected_llm_call(self, llm_func: Callable, *args, **kwargs) -> Any:
        """
        Execute LLM call with circuit breaker protection.
        
        Args:
            llm_func: LLM function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            LLM response or None if circuit is open
        """
        try:
            return await self._circuit_breaker.call(llm_func, *args, **kwargs)
        except CircuitBreakerOpen as e:
            logger.warning(f"LLM call rejected: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def get_circuit_breaker_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return self._circuit_breaker.get_stats()
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker"""
        self._circuit_breaker.reset()


# Global circuit breaker instance for shared use
_global_llm_circuit_breaker = None


def get_global_llm_circuit_breaker() -> CircuitBreaker:
    """Get or create global LLM circuit breaker"""
    global _global_llm_circuit_breaker
    if _global_llm_circuit_breaker is None:
        _global_llm_circuit_breaker = CircuitBreaker(
            name="global_llm_circuit_breaker"
        )
    return _global_llm_circuit_breaker


async def protected_llm_call(llm_func: Callable, *args, **kwargs) -> Any:
    """
    Convenience function for protected LLM calls using global circuit breaker.
    
    Args:
        llm_func: LLM function to call
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        LLM response or None if circuit is open
    """
    circuit_breaker = get_global_llm_circuit_breaker()
    try:
        return await circuit_breaker.call(llm_func, *args, **kwargs)
    except CircuitBreakerOpen as e:
        logger.warning(f"Global LLM circuit breaker open: {e}")
        return None
    except Exception as e:
        logger.error(f"Protected LLM call failed: {e}")
        raise