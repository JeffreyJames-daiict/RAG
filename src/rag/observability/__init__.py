"""Observability subsystem: structured logging, distributed tracing, and request context propagation."""

from rag.observability.context import (
    get_request_id,
    get_span_id,
    get_tenant_id,
    get_trace_id,
    request_context,
    set_request_id,
    set_span_id,
    set_tenant_id,
    set_trace_id,
)
from rag.observability.logging import (
    StructuredJsonFormatter,
    configure_structured_logging,
    mask_sensitive_data,
)
from rag.observability.tracing import Span, SpanContextManager, Tracer, get_tracer

__all__ = [
    "get_request_id",
    "set_request_id",
    "get_trace_id",
    "set_trace_id",
    "get_span_id",
    "set_span_id",
    "get_tenant_id",
    "set_tenant_id",
    "request_context",
    "StructuredJsonFormatter",
    "configure_structured_logging",
    "mask_sensitive_data",
    "Span",
    "SpanContextManager",
    "Tracer",
    "get_tracer",
]
