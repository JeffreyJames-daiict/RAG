"""Circuit breaker pattern protecting external providers (OpenRouter, Cohere, hosted embeddings) from cascading failures."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional

from rag.core.exceptions import CircuitBreakerOpenError

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Operational states of a circuit breaker."""

    CLOSED = "CLOSED"      # Normal operation: traffic passes through
    OPEN = "OPEN"          # Provider failing: fast-fail without calling external service
    HALF_OPEN = "HALF_OPEN"# Trial period: testing whether external service has recovered


class CircuitBreaker:
    """Thread-safe and async-safe circuit breaker with timeout recovery and fallback support."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_sec: float = 30.0,
        consecutive_success_threshold: int = 2,
        fallback: Optional[Callable[..., Any]] = None,
    ):
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_timeout_sec = max(0.1, recovery_timeout_sec)
        self.consecutive_success_threshold = max(1, consecutive_success_threshold)
        self.fallback = fallback

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._consecutive_successes = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._evaluate_state_transition()
            return self._state

    def _evaluate_state_transition(self) -> None:
        """Evaluate if an OPEN circuit has exceeded its recovery timeout and should transition to HALF_OPEN."""
        if self._state == CircuitState.OPEN:
            elapsed = time.perf_counter() - self._last_failure_time
            if elapsed >= self.recovery_timeout_sec:
                logger.info(
                    "Circuit breaker '%s' recovery timeout (%.1fs) elapsed. Transitioning from OPEN to HALF_OPEN.",
                    self.name,
                    self.recovery_timeout_sec,
                )
                self._state = CircuitState.HALF_OPEN
                self._consecutive_successes = 0

    def can_execute(self) -> bool:
        """Check whether execution is currently allowed through the circuit."""
        with self._lock:
            self._evaluate_state_transition()
            return self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        """Record a successful provider call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self.consecutive_success_threshold:
                    logger.info(
                        "Circuit breaker '%s' recorded %d consecutive successes. Transitioning from HALF_OPEN to CLOSED.",
                        self.name,
                        self._consecutive_successes,
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._consecutive_successes = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def record_failure(self, error: Exception) -> None:
        """Record an encountered failure."""
        with self._lock:
            self._last_failure_time = time.perf_counter()
            self._failure_count += 1

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    "Circuit breaker '%s' failed in HALF_OPEN state (%s). Reopening circuit.",
                    self.name,
                    error,
                )
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    logger.error(
                        "Circuit breaker '%s' reached failure threshold (%d/%d). Trip OPEN: fast-failing traffic for %.1fs.",
                        self.name,
                        self._failure_count,
                        self.failure_threshold,
                        self.recovery_timeout_sec,
                    )
                    self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Force reset circuit breaker to closed healthy state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._consecutive_successes = 0
            self._last_failure_time = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get diagnostic summary of the circuit breaker state."""
        with self._lock:
            self._evaluate_state_transition()
            remaining = 0.0
            if self._state == CircuitState.OPEN:
                elapsed = time.perf_counter() - self._last_failure_time
                remaining = max(0.0, self.recovery_timeout_sec - elapsed)

            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "consecutive_successes": self._consecutive_successes,
                "recovery_time_remaining_sec": round(remaining, 1),
            }

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a synchronous function through the circuit breaker."""
        if not self.can_execute():
            stats = self.get_stats()
            if self.fallback:
                logger.info("Circuit breaker '%s' is OPEN. Invoking registered fallback.", self.name)
                return self.fallback(*args, **kwargs)
            raise CircuitBreakerOpenError(
                provider_name=self.name,
                recovery_time_remaining_sec=stats["recovery_time_remaining_sec"],
            )

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as exc:
            self.record_failure(exc)
            if self.fallback:
                logger.info("Call failed on circuit '%s'. Invoking fallback handler: %s", self.name, exc)
                return self.fallback(*args, **kwargs)
            raise

    async def call_async(self, async_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute an asynchronous coroutine through the circuit breaker."""
        if not self.can_execute():
            stats = self.get_stats()
            if self.fallback:
                logger.info("Circuit breaker '%s' is OPEN. Invoking registered fallback.", self.name)
                if inspect.iscoroutinefunction(self.fallback):
                    return await self.fallback(*args, **kwargs)
                return self.fallback(*args, **kwargs)
            raise CircuitBreakerOpenError(
                provider_name=self.name,
                recovery_time_remaining_sec=stats["recovery_time_remaining_sec"],
            )

        try:
            result = await async_func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as exc:
            self.record_failure(exc)
            if self.fallback:
                logger.info("Async call failed on circuit '%s'. Invoking fallback handler: %s", self.name, exc)
                if inspect.iscoroutinefunction(self.fallback):
                    return await self.fallback(*args, **kwargs)
                return self.fallback(*args, **kwargs)
            raise


_REGISTRY: Dict[str, CircuitBreaker] = {}
_REGISTRY_LOCK = threading.Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout_sec: float = 30.0,
    fallback: Optional[Callable] = None,
) -> CircuitBreaker:
    """Get or create a named singleton circuit breaker."""
    with _REGISTRY_LOCK:
        if name not in _REGISTRY:
            _REGISTRY[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout_sec=recovery_timeout_sec,
                fallback=fallback,
            )
        return _REGISTRY[name]
