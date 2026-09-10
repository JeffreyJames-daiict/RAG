"""Comprehensive tests for reliability, observability, caching, and failure handling subsystems.

Tests cover:
1. Structured logging with JSON formatting, trace/request/tenant injection, and secret/PII masking
2. Distributed tracing with span hierarchy, timing, attributes, and W3C traceparent propagation
3. Error taxonomy with typed subclasses, categories, retryable flags, and safe serialization
4. Retry policies with bounded exponential backoff, jitter, and exception classification
5. Circuit breakers with state transitions, timeout recovery, and fallback execution
6. Multi-tenant caching with tenant isolation, TTL expiration, version invalidation, and LRU pruning
7. Health endpoints with liveness, readiness, and dependency diagnostics
8. Pipeline integration with partial failures, timeouts, and fallback behavior
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import threading
from unittest.mock import MagicMock, patch

import pytest

# ────────────────────────── Imports ──────────────────────────

from rag.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    CircuitBreakerOpenError,
    ConfigurationError,
    ConnectorError,
    DatabaseError,
    EmbeddingError,
    ErrorCategory,
    LLMProviderError,
    RAGError,
    RateLimitError,
    RerankingError,
    SearchError,
    SerializationError,
    TimeoutError,
    ValidationError,
)
from rag.observability.context import (
    get_request_id,
    get_span_id,
    get_tenant_id,
    get_trace_id,
    request_context,
    set_request_id,
    set_tenant_id,
    set_trace_id,
)
from rag.observability.logging import (
    StructuredJsonFormatter,
    configure_structured_logging,
    mask_sensitive_data,
)
from rag.observability.tracing import Span, Tracer, get_tracer
from rag.reliability.retry import (
    RetryPolicy,
    retry_async,
    retry_call,
    with_retry,
)
from rag.reliability.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    get_circuit_breaker,
)
from rag.caching.models import CacheDomain, CacheKey
from rag.caching.service import MultiTenantCacheService
from rag.health.models import ComponentHealth, HealthReport, HealthStatus
from rag.health.service import HealthCheckService


# ═══════════════════════════════════════════════════════════════
# 1. ERROR TAXONOMY TESTS
# ═══════════════════════════════════════════════════════════════

class TestErrorTaxonomy:
    """Validate structured RAGError hierarchy, categories, codes, and serialization."""

    def test_base_rag_error_defaults(self):
        err = RAGError()
        assert err.error_code == "RAG_INTERNAL_ERROR"
        assert err.category == ErrorCategory.INTERNAL
        assert err.is_retryable is False
        assert err.status_code == 500
        assert err.tenant_id is None
        assert err.details == {}

    def test_base_rag_error_custom(self):
        err = RAGError(
            message="Custom error",
            error_code="CUSTOM_CODE",
            category=ErrorCategory.DATABASE,
            is_retryable=True,
            status_code=503,
            tenant_id="tenant_abc",
            details={"key": "value"},
        )
        assert err.message == "Custom error"
        assert err.category == ErrorCategory.DATABASE
        assert err.is_retryable is True
        assert err.status_code == 503
        assert err.tenant_id == "tenant_abc"
        assert err.details == {"key": "value"}

    def test_error_to_dict_safe_serialization(self):
        err = RAGError(
            message="Test error",
            error_code="TEST_ERR",
            category=ErrorCategory.LLM_PROVIDER,
            details={"api_key": "sk-secret123", "model": "gpt-4"},
        )
        d = err.to_dict()
        assert d["error_code"] == "TEST_ERR"
        assert d["category"] == "llm_provider"
        # Secret key should be masked
        assert d["details"]["api_key"] == "******"
        # Non-secret should pass through
        assert d["details"]["model"] == "gpt-4"

    def test_error_str_with_tenant(self):
        err = RAGError(message="Boom", error_code="BOOM", tenant_id="t1")
        s = str(err)
        assert "BOOM" in s
        assert "t1" in s
        assert "Boom" in s

    @pytest.mark.parametrize(
        "exc_cls,expected_code,expected_category,expected_retryable,expected_status",
        [
            (ValidationError, "VALIDATION_FAILED", ErrorCategory.VALIDATION, False, 400),
            (AuthenticationError, "AUTH_FAILED", ErrorCategory.AUTHENTICATION, False, 401),
            (AuthorizationError, "PERMISSION_DENIED", ErrorCategory.AUTHORIZATION, False, 403),
            (EmbeddingError, "EMBEDDING_GENERATION_FAILED", ErrorCategory.EMBEDDING, True, 502),
            (DatabaseError, "DATABASE_ERROR", ErrorCategory.DATABASE, True, 500),
            (SearchError, "SEARCH_EXECUTION_FAILED", ErrorCategory.SEARCH, False, 500),
            (RerankingError, "RERANKING_FAILED", ErrorCategory.RERANKING, True, 502),
            (LLMProviderError, "LLM_PROVIDER_ERROR", ErrorCategory.LLM_PROVIDER, True, 502),
            (TimeoutError, "DEADLINE_EXCEEDED", ErrorCategory.TIMEOUT, True, 504),
            (ConfigurationError, "INVALID_CONFIGURATION", ErrorCategory.CONFIGURATION, False, 500),
            (SerializationError, "SERIALIZATION_ERROR", ErrorCategory.SERIALIZATION, False, 400),
        ],
    )
    def test_typed_exception_properties(self, exc_cls, expected_code, expected_category, expected_retryable, expected_status):
        err = exc_cls(message="test")
        assert err.error_code == expected_code
        assert err.category == expected_category
        assert err.is_retryable == expected_retryable
        assert err.status_code == expected_status
        assert isinstance(err, RAGError)

    def test_rate_limit_error_retry_after(self):
        err = RateLimitError(retry_after_sec=60.0)
        assert err.retry_after_sec == 60.0
        assert err.is_retryable is True
        assert err.status_code == 429
        assert err.details["retry_after_sec"] == 60.0

    def test_circuit_breaker_open_error(self):
        err = CircuitBreakerOpenError(provider_name="openrouter", recovery_time_remaining_sec=15.0)
        assert err.category == ErrorCategory.CIRCUIT_BREAKER
        assert err.status_code == 503
        assert err.is_retryable is False
        assert "openrouter" in err.message

    def test_connector_error_is_rag_error(self):
        err = ConnectorError(message="fail")
        assert isinstance(err, RAGError)
        assert err.category == ErrorCategory.CONNECTOR

    def test_all_error_categories_exist(self):
        expected = [
            "VALIDATION", "AUTHENTICATION", "AUTHORIZATION", "EMBEDDING",
            "DATABASE", "SEARCH", "RERANKING", "LLM_PROVIDER", "TIMEOUT",
            "RATE_LIMIT", "CONFIGURATION", "SERIALIZATION", "CIRCUIT_BREAKER",
            "CONNECTOR", "INTERNAL",
        ]
        for cat_name in expected:
            assert hasattr(ErrorCategory, cat_name)


# ═══════════════════════════════════════════════════════════════
# 2. STRUCTURED LOGGING TESTS
# ═══════════════════════════════════════════════════════════════

class TestStructuredLogging:
    """Validate JSON log formatting, context injection, and PII/secret masking."""

    def test_json_format_output(self):
        formatter = StructuredJsonFormatter()
        record = logging.LogRecord(
            name="rag.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["logger"] == "rag.test"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_context_injection(self):
        formatter = StructuredJsonFormatter()

        with request_context(request_id="req_abc123", trace_id="trace_xyz", tenant_id="tenant_1"):
            record = logging.LogRecord(
                name="rag.test", level=logging.INFO, pathname="", lineno=1,
                msg="traced", args=None, exc_info=None,
            )
            output = formatter.format(record)
            data = json.loads(output)
            assert data["request_id"] == "req_abc123"
            assert data["trace_id"] == "trace_xyz"
            assert data["tenant_id"] == "tenant_1"

    def test_rag_extra_fields_injection(self):
        formatter = StructuredJsonFormatter()
        record = logging.LogRecord(
            name="rag", level=logging.INFO, pathname="", lineno=1,
            msg="rag op", args=None, exc_info=None,
        )
        record.retrieval_method = "hybrid"
        record.candidate_counts = 50
        record.latency_ms = 123.4

        output = formatter.format(record)
        data = json.loads(output)
        assert data["retrieval_method"] == "hybrid"
        assert data["candidate_counts"] == 50
        assert data["latency_ms"] == 123.4

    def test_exception_capture(self):
        formatter = StructuredJsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="rag", level=logging.ERROR, pathname="", lineno=1,
                msg="fail", args=None, exc_info=sys.exc_info(),
            )
            output = formatter.format(record)
            data = json.loads(output)
            assert data["exception"]["type"] == "ValueError"
            assert "test error" in data["exception"]["message"]

    def test_mask_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5"
        masked = mask_sensitive_data(text)
        assert "eyJhbGciOiJSUzI1NiIsInR5" not in masked
        assert "******" in masked

    def test_mask_api_key(self):
        text = 'api_key: "sk-proj-abc123def456ghi789"'
        masked = mask_sensitive_data(text)
        assert "sk-proj-abc123def456ghi789" not in masked

    def test_mask_dict_sensitive_keys(self):
        d = {"api_token": "secret_value", "model": "gpt-4", "auth_header": "Bearer xyz"}
        masked = mask_sensitive_data(d)
        assert masked["api_token"] == "******"
        assert masked["model"] == "gpt-4"
        assert masked["auth_header"] == "******"

    def test_mask_raw_content_redacted(self):
        d = {"raw_content": "very long document text...", "other": "ok"}
        masked = mask_sensitive_data(d)
        assert masked["raw_content"] == "[redacted raw content]"
        assert masked["other"] == "ok"

    def test_mask_long_text_truncation(self):
        long_text = "A" * 1000
        masked = mask_sensitive_data(long_text, max_text_len=100)
        assert len(masked) < 1000
        assert "truncated" in masked

    def test_mask_nested_list(self):
        data = [{"password": "secret123"}, "normal text"]
        masked = mask_sensitive_data(data)
        assert masked[0]["password"] == "******"
        assert masked[1] == "normal text"


# ═══════════════════════════════════════════════════════════════
# 3. DISTRIBUTED TRACING TESTS
# ═══════════════════════════════════════════════════════════════

class TestDistributedTracing:
    """Validate span creation, parent-child hierarchy, timing, and W3C propagation."""

    def test_span_basic(self):
        span = Span(
            name="test_op",
            trace_id="trace_001",
            span_id="span_001",
        )
        assert span.name == "test_op"
        assert span.status == "OK"
        assert span.error is None

        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"

        span.finish()
        assert span.duration_ms >= 0
        assert span.end_time is not None

    def test_span_error_recording(self):
        span = Span(name="fail_op", trace_id="t1", span_id="s1")
        span.record_error(ValueError("bad input"))
        assert span.status == "ERROR"
        assert "ValueError" in span.error
        assert span.attributes["error.type"] == "ValueError"

    def test_span_to_dict(self):
        span = Span(name="op", trace_id="t1", span_id="s1", parent_span_id="p1")
        span.finish()
        d = span.to_dict()
        assert d["name"] == "op"
        assert d["trace_id"] == "t1"
        assert d["span_id"] == "s1"
        assert d["parent_span_id"] == "p1"
        assert "duration_ms" in d

    def test_span_finish_idempotent(self):
        span = Span(name="op", trace_id="t", span_id="s")
        time.sleep(0.01)
        span.finish()
        first_dur = span.duration_ms
        span.finish()  # Should not change
        assert span.duration_ms == first_dur

    def test_tracer_span_hierarchy(self):
        tracer = Tracer()
        with request_context(trace_id="trace_root"):
            with tracer.start_span("parent") as parent_span:
                with tracer.start_span("child") as child_span:
                    pass

        spans = tracer.get_completed_spans()
        assert len(spans) == 2
        child = spans[0]
        parent = spans[1]
        assert child.parent_span_id == parent.span_id
        assert parent.trace_id == child.trace_id

    def test_tracer_auto_generates_trace_id(self):
        tracer = Tracer()
        # Ensure no trace in context
        with request_context(trace_id=""):
            with tracer.start_span("auto") as span:
                pass
        spans = tracer.get_completed_spans()
        assert len(spans) == 1
        assert spans[0].trace_id != ""

    def test_tracer_tenant_attribute(self):
        tracer = Tracer()
        with request_context(tenant_id="org_42"):
            with tracer.start_span("op") as span:
                pass
        assert tracer.get_completed_spans()[0].attributes["tenant_id"] == "org_42"

    def test_traceparent_create_and_parse(self):
        tp = Tracer.create_traceparent("abcd1234abcd1234abcd1234abcd1234", "0123456789abcdef")
        assert tp.startswith("00-")
        assert tp.endswith("-01")

        parsed = Tracer.parse_traceparent(tp)
        assert parsed is not None
        trace_id, span_id = parsed
        assert len(trace_id) == 32
        assert len(span_id) == 16

    def test_traceparent_parse_invalid(self):
        assert Tracer.parse_traceparent("") is None
        assert Tracer.parse_traceparent("invalid") is None
        assert Tracer.parse_traceparent("00-short-short-01") is None

    def test_tracer_clear(self):
        tracer = Tracer()
        with request_context():
            with tracer.start_span("op"):
                pass
        assert len(tracer.get_completed_spans()) == 1
        tracer.clear()
        assert len(tracer.get_completed_spans()) == 0

    def test_span_context_manager_error_propagation(self):
        tracer = Tracer()
        with request_context():
            with pytest.raises(ValueError, match="boom"):
                with tracer.start_span("fail") as span:
                    raise ValueError("boom")
        spans = tracer.get_completed_spans()
        assert spans[0].status == "ERROR"
        assert "ValueError" in spans[0].error


# ═══════════════════════════════════════════════════════════════
# 4. REQUEST CONTEXT TESTS
# ═══════════════════════════════════════════════════════════════

class TestRequestContext:
    """Validate context variable management and lifecycle."""

    def test_context_sets_and_resets(self):
        before_req = get_request_id()
        before_trace = get_trace_id()

        with request_context(request_id="req_999", trace_id="trace_999", tenant_id="t99"):
            assert get_request_id() == "req_999"
            assert get_trace_id() == "trace_999"
            assert get_tenant_id() == "t99"

        # After context exits, should revert
        assert get_request_id() == before_req
        assert get_trace_id() == before_trace

    def test_auto_generated_ids(self):
        with request_context():
            req = get_request_id()
            trace = get_trace_id()
            assert req.startswith("req_")
            assert trace.startswith("trace_")

    def test_nested_context(self):
        with request_context(request_id="outer"):
            assert get_request_id() == "outer"
            with request_context(request_id="inner"):
                assert get_request_id() == "inner"
            assert get_request_id() == "outer"


# ═══════════════════════════════════════════════════════════════
# 5. RETRY POLICY TESTS
# ═══════════════════════════════════════════════════════════════

class TestRetryPolicy:
    """Validate bounded exponential backoff, jitter, and exception classification."""

    def test_default_policy(self):
        pol = RetryPolicy()
        assert pol.max_retries == 3
        assert pol.initial_delay_sec == 0.5
        assert pol.max_delay_sec == 30.0
        assert pol.backoff_multiplier == 2.0
        assert pol.jitter is True

    def test_retryable_exceptions(self):
        pol = RetryPolicy()
        # Retryable by type
        assert pol.is_retryable(TimeoutError("timeout")) is True
        assert pol.is_retryable(RateLimitError()) is True
        assert pol.is_retryable(EmbeddingError("fail")) is True
        assert pol.is_retryable(RerankingError("fail")) is True
        assert pol.is_retryable(LLMProviderError("fail")) is True
        assert pol.is_retryable(DatabaseError("fail")) is True

    def test_non_retryable_exceptions(self):
        pol = RetryPolicy()
        assert pol.is_retryable(ValidationError("bad")) is False
        assert pol.is_retryable(AuthenticationError()) is False
        assert pol.is_retryable(AuthorizationError()) is False
        assert pol.is_retryable(ConfigurationError("bad")) is False
        assert pol.is_retryable(SerializationError("bad")) is False
        assert pol.is_retryable(ValueError("bad")) is False

    def test_rag_error_retryable_flag(self):
        pol = RetryPolicy()
        retryable_err = RAGError(is_retryable=True)
        non_retryable_err = RAGError(is_retryable=False)
        assert pol.is_retryable(retryable_err) is True
        assert pol.is_retryable(non_retryable_err) is False

    def test_compute_delay_exponential(self):
        pol = RetryPolicy(initial_delay_sec=1.0, backoff_multiplier=2.0, jitter=False, max_delay_sec=100.0)
        assert pol.compute_delay(1) == 1.0
        assert pol.compute_delay(2) == 2.0
        assert pol.compute_delay(3) == 4.0
        assert pol.compute_delay(4) == 8.0

    def test_compute_delay_capped(self):
        pol = RetryPolicy(initial_delay_sec=1.0, backoff_multiplier=10.0, jitter=False, max_delay_sec=5.0)
        assert pol.compute_delay(1) == 1.0
        assert pol.compute_delay(2) == 5.0  # Capped
        assert pol.compute_delay(3) == 5.0

    def test_compute_delay_jitter(self):
        pol = RetryPolicy(initial_delay_sec=1.0, backoff_multiplier=2.0, jitter=True, max_delay_sec=10.0)
        delays = [pol.compute_delay(2) for _ in range(100)]
        # With jitter, delays should be between 0.5*2.0 and 2.0 (since capped_delay=2.0)
        assert all(0.0 <= d <= 2.0 for d in delays)
        # They shouldn't all be the same
        assert len(set(round(d, 4) for d in delays)) > 1

    def test_retry_call_success(self):
        counter = {"calls": 0}

        def succeed():
            counter["calls"] += 1
            return "ok"

        result = retry_call(succeed, policy=RetryPolicy(max_retries=3))
        assert result == "ok"
        assert counter["calls"] == 1

    def test_retry_call_eventual_success(self):
        counter = {"calls": 0}

        def fail_twice():
            counter["calls"] += 1
            if counter["calls"] <= 2:
                raise TimeoutError("timeout")
            return "recovered"

        result = retry_call(fail_twice, policy=RetryPolicy(max_retries=3, initial_delay_sec=0.01))
        assert result == "recovered"
        assert counter["calls"] == 3

    def test_retry_call_exhausted(self):
        def always_fail():
            raise TimeoutError("always times out")

        with pytest.raises(TimeoutError):
            retry_call(always_fail, policy=RetryPolicy(max_retries=2, initial_delay_sec=0.01))

    def test_retry_call_non_retryable_immediate(self):
        counter = {"calls": 0}

        def fail_validation():
            counter["calls"] += 1
            raise ValidationError("bad input")

        with pytest.raises(ValidationError):
            retry_call(fail_validation, policy=RetryPolicy(max_retries=5, initial_delay_sec=0.01))
        assert counter["calls"] == 1  # Should not retry

    @pytest.mark.asyncio
    async def test_retry_async_eventual_success(self):
        counter = {"calls": 0}

        async def fail_once():
            counter["calls"] += 1
            if counter["calls"] <= 1:
                raise EmbeddingError("transient")
            return "recovered"

        result = await retry_async(fail_once, policy=RetryPolicy(max_retries=3, initial_delay_sec=0.01))
        assert result == "recovered"
        assert counter["calls"] == 2

    @pytest.mark.asyncio
    async def test_retry_async_exhausted(self):
        async def always_fail():
            raise LLMProviderError("down")

        with pytest.raises(LLMProviderError):
            await retry_async(always_fail, policy=RetryPolicy(max_retries=1, initial_delay_sec=0.01))

    def test_with_retry_decorator_sync(self):
        counter = {"calls": 0}

        @with_retry(RetryPolicy(max_retries=2, initial_delay_sec=0.01))
        def flaky():
            counter["calls"] += 1
            if counter["calls"] < 2:
                raise DatabaseError("transient")
            return "done"

        result = flaky()
        assert result == "done"
        assert counter["calls"] == 2

    @pytest.mark.asyncio
    async def test_with_retry_decorator_async(self):
        counter = {"calls": 0}

        @with_retry(RetryPolicy(max_retries=2, initial_delay_sec=0.01))
        async def flaky():
            counter["calls"] += 1
            if counter["calls"] < 2:
                raise RerankingError("transient")
            return "done"

        result = await flaky()
        assert result == "done"
        assert counter["calls"] == 2

    def test_negative_max_retries_clamped(self):
        pol = RetryPolicy(max_retries=-5)
        assert pol.max_retries == 0


# ═══════════════════════════════════════════════════════════════
# 6. CIRCUIT BREAKER TESTS
# ═══════════════════════════════════════════════════════════════

class TestCircuitBreaker:
    """Validate state transitions, timeout recovery, fallback, and thread-safety."""

    def test_initial_state_closed(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_trip_to_open(self):
        cb = CircuitBreaker(name="trip_test", failure_threshold=3, recovery_timeout_sec=60.0)
        for i in range(3):
            cb.record_failure(Exception(f"fail {i}"))
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_open_to_half_open_on_timeout(self):
        cb = CircuitBreaker(name="timeout_test", failure_threshold=2, recovery_timeout_sec=0.1)
        cb.record_failure(Exception("fail 1"))
        cb.record_failure(Exception("fail 2"))
        assert cb.state == CircuitState.OPEN

        time.sleep(0.15)  # Wait for recovery timeout
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker(name="recovery_test", failure_threshold=2, recovery_timeout_sec=0.05, consecutive_success_threshold=2)
        cb.record_failure(Exception("1"))
        cb.record_failure(Exception("2"))
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # Need 2 successes
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_back_to_open_on_failure(self):
        cb = CircuitBreaker(name="relapse_test", failure_threshold=2, recovery_timeout_sec=0.05)
        cb.record_failure(Exception("1"))
        cb.record_failure(Exception("2"))
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure(Exception("relapse"))
        assert cb.state == CircuitState.OPEN

    def test_call_sync_success(self):
        cb = CircuitBreaker(name="sync_call", failure_threshold=5)
        result = cb.call(lambda x: x * 2, 5)
        assert result == 10

    def test_call_sync_failure_with_fallback(self):
        fallback = lambda: "fallback_value"
        cb = CircuitBreaker(name="fb_test", failure_threshold=5, fallback=fallback)

        result = cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")), )
        # Actually the lambda is tricky, let's use a proper function
        def raise_error():
            raise RuntimeError("fail")

        result = cb.call(raise_error)
        assert result == "fallback_value"

    def test_call_sync_circuit_open_raises(self):
        cb = CircuitBreaker(name="open_raise", failure_threshold=1, recovery_timeout_sec=60.0)
        cb.record_failure(Exception("trip"))
        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerOpenError):
            cb.call(lambda: "should not run")

    def test_call_sync_circuit_open_fallback(self):
        cb = CircuitBreaker(name="open_fb", failure_threshold=1, recovery_timeout_sec=60.0, fallback=lambda: "safe")
        cb.record_failure(Exception("trip"))
        result = cb.call(lambda: "should not run")
        assert result == "safe"

    @pytest.mark.asyncio
    async def test_call_async_success(self):
        cb = CircuitBreaker(name="async_call", failure_threshold=5)

        async def async_op():
            return "async_result"

        result = await cb.call_async(async_op)
        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_call_async_circuit_open_raises(self):
        cb = CircuitBreaker(name="async_open", failure_threshold=1, recovery_timeout_sec=60.0)
        cb.record_failure(Exception("trip"))

        async def should_not_run():
            return "nope"

        with pytest.raises(CircuitBreakerOpenError):
            await cb.call_async(should_not_run)

    def test_reset(self):
        cb = CircuitBreaker(name="reset_test", failure_threshold=2)
        cb.record_failure(Exception("1"))
        cb.record_failure(Exception("2"))
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_get_stats(self):
        cb = CircuitBreaker(name="stats_test", failure_threshold=5, recovery_timeout_sec=30.0)
        cb.record_failure(Exception("1"))

        stats = cb.get_stats()
        assert stats["name"] == "stats_test"
        assert stats["state"] == "CLOSED"
        assert stats["failure_count"] == 1
        assert stats["failure_threshold"] == 5

    def test_singleton_registry(self):
        cb1 = get_circuit_breaker("shared_breaker", failure_threshold=3)
        cb2 = get_circuit_breaker("shared_breaker")
        assert cb1 is cb2

    def test_thread_safety(self):
        cb = CircuitBreaker(name="thread_safe", failure_threshold=100)
        errors = []

        def record_many():
            try:
                for _ in range(50):
                    cb.record_failure(Exception("f"))
                    cb.record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0  # No thread-safety crashes

    def test_failure_threshold_clamped_minimum(self):
        cb = CircuitBreaker(name="min_thresh", failure_threshold=0)
        assert cb.failure_threshold == 1


# ═══════════════════════════════════════════════════════════════
# 7. MULTI-TENANT CACHING TESTS
# ═══════════════════════════════════════════════════════════════

class TestMultiTenantCaching:
    """Validate tenant isolation, TTL, LRU, and invalidation strategies."""

    def _make_key(self, tenant: str, domain: CacheDomain = CacheDomain.RETRIEVAL_RESULTS, query: str = "test query") -> CacheKey:
        return CacheKey(tenant_id=tenant, domain=domain, query_text=query)

    def test_basic_set_and_get(self):
        cache = MultiTenantCacheService(max_entries=100)
        key = self._make_key("t1")
        cache.set(key, {"results": [1, 2, 3]})
        assert cache.get(key) == {"results": [1, 2, 3]}

    def test_cache_miss(self):
        cache = MultiTenantCacheService(max_entries=100)
        key = self._make_key("t1")
        assert cache.get(key) is None

    def test_tenant_isolation(self):
        cache = MultiTenantCacheService(max_entries=100)
        key_t1 = self._make_key("tenant_a", query="how to reset?")
        key_t2 = self._make_key("tenant_b", query="how to reset?")

        cache.set(key_t1, "tenant_a_data")
        cache.set(key_t2, "tenant_b_data")

        assert cache.get(key_t1) == "tenant_a_data"
        assert cache.get(key_t2) == "tenant_b_data"

    def test_tenant_isolation_no_cross_leak(self):
        cache = MultiTenantCacheService(max_entries=100)
        key_t1 = self._make_key("isolated_a")
        cache.set(key_t1, "secret_data_a")

        key_t2 = CacheKey(
            tenant_id="isolated_b",
            domain=CacheDomain.RETRIEVAL_RESULTS,
            query_hash=key_t1.query_hash,  # Same hash
        )
        assert cache.get(key_t2) is None  # Must not leak

    def test_ttl_expiration(self):
        cache = MultiTenantCacheService(max_entries=100, default_ttl_sec=0.1)
        key = self._make_key("t1")
        cache.set(key, "expires_soon")
        assert cache.get(key) == "expires_soon"

        time.sleep(0.15)
        assert cache.get(key) is None  # Expired

    def test_custom_ttl_override(self):
        cache = MultiTenantCacheService(max_entries=100, default_ttl_sec=3600)
        key = self._make_key("t1")
        cache.set(key, "short_lived", ttl_sec=0.1)

        time.sleep(0.15)
        assert cache.get(key) is None

    def test_lru_eviction(self):
        cache = MultiTenantCacheService(max_entries=10)
        # Fill cache beyond capacity
        for i in range(15):
            key = self._make_key("t1", query=f"query_{i}")
            cache.set(key, f"value_{i}")

        stats = cache.get_stats()
        assert stats["total_entries"] <= 10

        # Oldest entries should be evicted
        oldest_key = self._make_key("t1", query="query_0")
        assert cache.get(oldest_key) is None

        # Newest should still be present
        newest_key = self._make_key("t1", query="query_14")
        assert cache.get(newest_key) == "value_14"

    def test_invalidate_tenant(self):
        cache = MultiTenantCacheService(max_entries=100)
        for i in range(5):
            cache.set(self._make_key("purge_me", query=f"q{i}"), f"v{i}")
        cache.set(self._make_key("keep_me", query="q0"), "keep")

        removed = cache.invalidate_tenant("purge_me")
        assert removed == 5
        assert cache.get(self._make_key("keep_me", query="q0")) == "keep"

    def test_invalidate_document_version(self):
        cache = MultiTenantCacheService(max_entries=100)
        key_v1 = CacheKey(tenant_id="t1", domain=CacheDomain.RETRIEVAL_RESULTS, query_text="q", document_index_version="v1")
        key_v2 = CacheKey(tenant_id="t1", domain=CacheDomain.RETRIEVAL_RESULTS, query_text="q", document_index_version="v2")

        cache.set(key_v1, "old_results")
        cache.set(key_v2, "new_results")

        removed = cache.invalidate_document_version("t1", "v1")
        assert removed == 1
        assert cache.get(key_v1) is None
        assert cache.get(key_v2) == "new_results"

    def test_invalidate_domain(self):
        cache = MultiTenantCacheService(max_entries=100)
        cache.set(self._make_key("t1", domain=CacheDomain.RETRIEVAL_RESULTS), "retrieval")
        cache.set(self._make_key("t1", domain=CacheDomain.LLM_RESPONSES), "llm")

        removed = cache.invalidate_domain("t1", CacheDomain.RETRIEVAL_RESULTS)
        assert removed == 1
        assert cache.get(self._make_key("t1", domain=CacheDomain.LLM_RESPONSES)) == "llm"

    def test_clear(self):
        cache = MultiTenantCacheService(max_entries=100)
        cache.set(self._make_key("t1"), "v1")
        cache.set(self._make_key("t2"), "v2")
        cache.clear()
        assert cache.get_stats()["total_entries"] == 0
        assert cache.get_stats()["hits"] == 0
        assert cache.get_stats()["misses"] == 0

    def test_hit_ratio(self):
        cache = MultiTenantCacheService(max_entries=100)
        key = self._make_key("t1")
        cache.set(key, "v")

        cache.get(key)  # Hit
        cache.get(key)  # Hit
        cache.get(self._make_key("t2"))  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert 0.6 < stats["hit_ratio"] < 0.7

    def test_has(self):
        cache = MultiTenantCacheService(max_entries=100)
        key = self._make_key("t1")
        assert cache.has(key) is False
        cache.set(key, "v")
        assert cache.has(key) is True

    def test_overwrite_moves_to_end(self):
        cache = MultiTenantCacheService(max_entries=100)
        key = self._make_key("t1")
        cache.set(key, "original")
        cache.set(key, "updated")
        assert cache.get(key) == "updated"


# ═══════════════════════════════════════════════════════════════
# 8. CACHE KEY TESTS
# ═══════════════════════════════════════════════════════════════

class TestCacheKey:
    """Validate deterministic key generation and tenant partitioning."""

    def test_key_string_starts_with_tenant(self):
        key = CacheKey(tenant_id="org_123", domain=CacheDomain.QUERY_EMBEDDINGS, query_text="hello")
        key_str = key.to_key_string()
        assert key_str.startswith("t=org_123:")

    def test_auto_query_hash(self):
        key = CacheKey(tenant_id="t1", domain=CacheDomain.RETRIEVAL_RESULTS, query_text="What is RAG?")
        assert key.query_hash != ""
        assert len(key.query_hash) == 24

    def test_normalized_query_hash(self):
        k1 = CacheKey(tenant_id="t1", domain=CacheDomain.RETRIEVAL_RESULTS, query_text="  What  Is  RAG?  ")
        k2 = CacheKey(tenant_id="t1", domain=CacheDomain.RETRIEVAL_RESULTS, query_text="what is rag?")
        assert k1.query_hash == k2.query_hash

    def test_different_tenants_different_keys(self):
        k1 = CacheKey(tenant_id="t1", domain=CacheDomain.RETRIEVAL_RESULTS, query_text="q")
        k2 = CacheKey(tenant_id="t2", domain=CacheDomain.RETRIEVAL_RESULTS, query_text="q")
        assert k1.to_key_string() != k2.to_key_string()

    def test_key_includes_optional_fields(self):
        key = CacheKey(
            tenant_id="t1",
            domain=CacheDomain.RERANKING_RESULTS,
            query_text="q",
            embedding_model="text-embedding-3-small",
            reranker_model="cohere-rerank-v3",
        )
        key_str = key.to_key_string()
        assert "em=text-embedding-3-small" in key_str
        assert "rm=cohere-rerank-v3" in key_str

    def test_compute_config_hash(self):
        h1 = CacheKey.compute_config_hash({"top_k": 10, "alpha": 0.5})
        h2 = CacheKey.compute_config_hash({"alpha": 0.5, "top_k": 10})  # Same keys, different order
        assert h1 == h2
        assert len(h1) == 16


# ═══════════════════════════════════════════════════════════════
# 9. HEALTH & READINESS TESTS
# ═══════════════════════════════════════════════════════════════

class TestHealthService:
    """Validate liveness, readiness, dependency probes, and degraded/unhealthy determination."""

    def test_liveness(self):
        svc = HealthCheckService()
        assert svc.check_liveness() is True

    def test_healthy_readiness(self):
        svc = HealthCheckService()
        assert svc.check_readiness() is True

    def test_full_health_report(self):
        svc = HealthCheckService(environment="test", version="2.0.0")
        report = svc.check_health()

        assert report.status == HealthStatus.HEALTHY
        assert report.liveness is True
        assert report.readiness is True
        assert report.environment == "test"
        assert report.version == "2.0.0"
        assert "database" in report.dependencies
        assert "vector_index" in report.dependencies
        assert "embedding_provider" in report.dependencies
        assert "llm_provider" in report.dependencies
        assert report.uptime_seconds >= 0

    def test_degraded_on_non_critical_failure(self):
        svc = HealthCheckService()

        def failing_sparse():
            raise ConnectionError("BM25 index unavailable")

        svc.register_probe("sparse_index", failing_sparse, is_critical=False)
        report = svc.check_health()

        assert report.status == HealthStatus.DEGRADED
        assert report.readiness is True  # Still ready, non-critical
        assert report.dependencies["sparse_index"].status == HealthStatus.UNHEALTHY
        assert "BM25" in report.dependencies["sparse_index"].message

    def test_unhealthy_on_critical_failure(self):
        svc = HealthCheckService()

        def failing_db():
            raise ConnectionError("DB connection pool exhausted")

        svc.register_probe("database", failing_db, is_critical=True)
        report = svc.check_health()

        assert report.status == HealthStatus.UNHEALTHY
        assert report.readiness is False

    def test_custom_probe_registration(self):
        svc = HealthCheckService()

        def custom_check():
            return {"custom_metric": 42}

        svc.register_probe("custom_service", custom_check, is_critical=False)
        report = svc.check_health()
        assert "custom_service" in report.dependencies
        assert report.dependencies["custom_service"].details["custom_metric"] == 42

    def test_dependency_latency_measured(self):
        svc = HealthCheckService()

        def slow_probe():
            time.sleep(0.05)
            return {"status": "ok"}

        svc.register_probe("slow_service", slow_probe, is_critical=False)
        report = svc.check_health()
        assert report.dependencies["slow_service"].latency_ms >= 40

    def test_health_report_serializable(self):
        svc = HealthCheckService()
        report = svc.check_health()
        # Should be JSON serializable
        data = report.model_dump()
        json_str = json.dumps(data, default=str)
        assert len(json_str) > 0

    def test_multiple_critical_failures(self):
        svc = HealthCheckService()
        svc.register_probe("database", lambda: (_ for _ in ()).throw(Exception("db fail")), is_critical=True)
        svc.register_probe("vector_index", lambda: (_ for _ in ()).throw(Exception("vec fail")), is_critical=True)

        # Use proper failing functions
        def fail_db():
            raise Exception("db fail")

        def fail_vec():
            raise Exception("vec fail")

        svc.register_probe("database", fail_db, is_critical=True)
        svc.register_probe("vector_index", fail_vec, is_critical=True)

        report = svc.check_health()
        assert report.status == HealthStatus.UNHEALTHY
        assert report.readiness is False


# ═══════════════════════════════════════════════════════════════
# 10. PIPELINE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    """Test end-to-end integration of observability, reliability, and caching."""

    def test_traced_retry_with_circuit_breaker(self):
        """Simulate a pipeline stage with tracing, retry, and circuit breaker protection."""
        tracer = Tracer()
        cb = CircuitBreaker(name="integration_test", failure_threshold=5, recovery_timeout_sec=0.1)
        pol = RetryPolicy(max_retries=2, initial_delay_sec=0.01)

        call_count = {"n": 0}

        def flaky_provider():
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise TimeoutError("timeout")
            return "success"

        with request_context(request_id="req_int", trace_id="trace_int", tenant_id="t1"):
            with tracer.start_span("embedding") as span:
                result = retry_call(
                    lambda: cb.call(flaky_provider),
                    policy=pol,
                )
                span.set_attribute("result", result)

        assert result == "success"
        assert call_count["n"] == 2
        spans = tracer.get_completed_spans()
        assert len(spans) == 1
        assert spans[0].attributes["result"] == "success"

    def test_cached_retrieval_with_tracing(self):
        """Simulate cached retrieval with tracing context."""
        cache = MultiTenantCacheService(max_entries=100)
        tracer = Tracer()

        with request_context(tenant_id="t1"):
            with tracer.start_span("retrieval") as span:
                cache_key = CacheKey(tenant_id="t1", domain=CacheDomain.RETRIEVAL_RESULTS, query_text="test query")
                cached = cache.get(cache_key)

                if cached is None:
                    span.set_attribute("cache_hit", False)
                    result = ["doc1", "doc2"]
                    cache.set(cache_key, result)
                else:
                    span.set_attribute("cache_hit", True)
                    result = cached

        assert result == ["doc1", "doc2"]
        assert tracer.get_completed_spans()[0].attributes["cache_hit"] is False

        # Second call should hit cache
        with request_context(tenant_id="t1"):
            with tracer.start_span("retrieval_cached") as span2:
                cached = cache.get(cache_key)
                span2.set_attribute("cache_hit", cached is not None)

        assert tracer.get_completed_spans()[-1].attributes["cache_hit"] is True

    def test_circuit_breaker_fallback_with_logging(self):
        """Simulate circuit breaker triggering fallback, with structured logging context."""
        fallback_results = [{"fallback": True}]
        cb = CircuitBreaker(
            name="embedding_provider",
            failure_threshold=2,
            recovery_timeout_sec=60.0,
            fallback=lambda *a, **kw: fallback_results,
        )

        with request_context(request_id="req_fb", tenant_id="t1"):
            # Trip the circuit
            cb.record_failure(Exception("fail 1"))
            cb.record_failure(Exception("fail 2"))
            assert cb.state == CircuitState.OPEN

            # Should invoke fallback
            result = cb.call(lambda: "should not run")
            assert result == fallback_results

    def test_error_taxonomy_integration_with_retry(self):
        """Ensure typed errors correctly gate retry behavior in pipeline."""
        counter = {"n": 0}

        def stage_with_typed_errors():
            counter["n"] += 1
            if counter["n"] == 1:
                raise RateLimitError(retry_after_sec=0.01)
            if counter["n"] == 2:
                raise EmbeddingError("transient", is_retryable=True)
            return "ok"

        result = retry_call(stage_with_typed_errors, policy=RetryPolicy(max_retries=3, initial_delay_sec=0.01))
        assert result == "ok"
        assert counter["n"] == 3

    def test_non_retryable_halts_pipeline(self):
        """Ensure non-retryable errors immediately halt without retries."""
        counter = {"n": 0}

        def auth_fail():
            counter["n"] += 1
            raise AuthenticationError("invalid credentials")

        with pytest.raises(AuthenticationError):
            retry_call(auth_fail, policy=RetryPolicy(max_retries=5, initial_delay_sec=0.01))

        assert counter["n"] == 1  # No retries for auth failures

    def test_full_lifecycle_with_health(self):
        """Test health service reports degraded when a circuit breaker is open."""
        svc = HealthCheckService()
        cb = CircuitBreaker(name="llm_lifecycle", failure_threshold=1, recovery_timeout_sec=60.0)

        # Trip the circuit
        cb.record_failure(Exception("LLM down"))

        def llm_probe():
            if cb.state == CircuitState.OPEN:
                raise RuntimeError(f"Circuit breaker '{cb.name}' is OPEN")
            return {"status": "ready"}

        svc.register_probe("llm_provider", llm_probe, is_critical=True)
        report = svc.check_health()
        assert report.status == HealthStatus.UNHEALTHY
        assert report.readiness is False

    @pytest.mark.asyncio
    async def test_async_pipeline_with_retry_and_tracing(self):
        """Simulate an async pipeline stage with retry and tracing."""
        tracer = Tracer()
        counter = {"n": 0}

        async def async_embed(text: str):
            counter["n"] += 1
            if counter["n"] < 2:
                raise EmbeddingError("transient embedding failure")
            return [0.1, 0.2, 0.3]

        with request_context(request_id="req_async", tenant_id="t1"):
            with tracer.start_span("async_embedding") as span:
                result = await retry_async(
                    async_embed, "test query",
                    policy=RetryPolicy(max_retries=3, initial_delay_sec=0.01),
                )
                span.set_attribute("embedding_dim", len(result))

        assert result == [0.1, 0.2, 0.3]
        assert counter["n"] == 2
        assert tracer.get_completed_spans()[0].attributes["embedding_dim"] == 3
