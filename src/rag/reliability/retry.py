"""Bounded retry policies with exponential backoff, jitter, and typed exception classification."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import random
import time
from typing import Any, Callable, Optional, Sequence, Tuple, Type, TypeVar

from rag.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    DatabaseError,
    EmbeddingError,
    LLMProviderError,
    RAGError,
    RateLimitError,
    RerankingError,
    SerializationError,
    TimeoutError,
    ValidationError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    TimeoutError,
    RateLimitError,
    EmbeddingError,
    RerankingError,
    LLMProviderError,
    DatabaseError,
    ConnectionError,
    TimeoutError,
)

DEFAULT_NON_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    SerializationError,
    ValueError,
    TypeError,
)


class RetryPolicy:
    """Configurable retry policy enforcing bounded attempts, exponential backoff, and full jitter."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay_sec: float = 0.5,
        max_delay_sec: float = 30.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Sequence[Type[Exception]]] = None,
        non_retryable_exceptions: Optional[Sequence[Type[Exception]]] = None,
    ):
        self.max_retries = max(0, max_retries)
        self.initial_delay_sec = max(0.0, initial_delay_sec)
        self.max_delay_sec = max(self.initial_delay_sec, max_delay_sec)
        self.backoff_multiplier = max(1.0, backoff_multiplier)
        self.jitter = jitter

        self.retryable_exceptions: Tuple[Type[Exception], ...] = (
            tuple(retryable_exceptions) if retryable_exceptions else DEFAULT_RETRYABLE_EXCEPTIONS
        )
        self.non_retryable_exceptions: Tuple[Type[Exception], ...] = (
            tuple(non_retryable_exceptions) if non_retryable_exceptions else DEFAULT_NON_RETRYABLE_EXCEPTIONS
        )

    def is_retryable(self, exc: Exception) -> bool:
        """Determine whether an encountered exception is eligible for retry."""
        # Explicitly check non-retryable first
        if isinstance(exc, self.non_retryable_exceptions):
            return False

        # If it's a structured RAGError, check its is_retryable flag
        if isinstance(exc, RAGError):
            return exc.is_retryable

        # Check explicit retryable types
        if isinstance(exc, self.retryable_exceptions):
            return True

        return False

    def compute_delay(self, attempt: int) -> float:
        """Compute bounded exponential backoff delay with optional full jitter."""
        base_delay = self.initial_delay_sec * (self.backoff_multiplier ** max(0, attempt - 1))
        capped_delay = min(self.max_delay_sec, base_delay)

        if self.jitter:
            # Full jitter: random duration between 0 and capped_delay
            return random.uniform(capped_delay * 0.5, capped_delay)

        return capped_delay


def retry_call(fn: Callable[..., T], *args: Any, policy: Optional[RetryPolicy] = None, **kwargs: Any) -> T:
    """Synchronously execute a callable with the specified retry policy."""
    pol = policy or RetryPolicy()
    attempt = 0

    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            attempt += 1
            if attempt > pol.max_retries or not pol.is_retryable(exc):
                logger.warning(
                    "Call '%s' failed irreversibly or exhausted %d retries: %s",
                    getattr(fn, "__name__", str(fn)),
                    pol.max_retries,
                    exc,
                )
                raise

            delay = pol.compute_delay(attempt)
            logger.info(
                "Call '%s' encountered %s (attempt %d/%d). Retrying in %.2fs...",
                getattr(fn, "__name__", str(fn)),
                type(exc).__name__,
                attempt,
                pol.max_retries,
                delay,
            )
            time.sleep(delay)


async def retry_async(async_fn: Callable[..., Any], *args: Any, policy: Optional[RetryPolicy] = None, **kwargs: Any) -> Any:
    """Asynchronously execute a coroutine function with the specified retry policy."""
    pol = policy or RetryPolicy()
    attempt = 0

    while True:
        try:
            return await async_fn(*args, **kwargs)
        except Exception as exc:
            attempt += 1
            if attempt > pol.max_retries or not pol.is_retryable(exc):
                logger.warning(
                    "Async call '%s' failed irreversibly or exhausted %d retries: %s",
                    getattr(async_fn, "__name__", str(async_fn)),
                    pol.max_retries,
                    exc,
                )
                raise

            delay = pol.compute_delay(attempt)
            logger.info(
                "Async call '%s' encountered %s (attempt %d/%d). Retrying in %.2fs...",
                getattr(async_fn, "__name__", str(async_fn)),
                type(exc).__name__,
                attempt,
                pol.max_retries,
                delay,
            )
            await asyncio.sleep(delay)


def with_retry(policy: Optional[RetryPolicy] = None) -> Callable:
    """Decorator applying retry policy to synchronous or asynchronous functions."""
    pol = policy or RetryPolicy()

    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await retry_async(func, *args, policy=pol, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return retry_call(func, *args, policy=pol, **kwargs)
            return sync_wrapper

    return decorator
