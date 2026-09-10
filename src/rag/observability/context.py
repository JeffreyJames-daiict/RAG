"""Context variables and request lifecycle tracking for structured observability."""

from __future__ import annotations

import contextvars
import uuid
from contextlib import contextmanager
from typing import Generator, Optional

_REQUEST_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("rag_request_id", default="")
_TRACE_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("rag_trace_id", default="")
_SPAN_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("rag_span_id", default="")
_TENANT_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("rag_tenant_id", default="default")


def get_request_id() -> str:
    """Get the active request ID or generate a fallback."""
    val = _REQUEST_ID_CTX.get()
    return val if val else ""


def set_request_id(request_id: str) -> contextvars.Token:
    """Set active request ID."""
    return _REQUEST_ID_CTX.set(request_id)


def get_trace_id() -> str:
    """Get active distributed trace ID."""
    val = _TRACE_ID_CTX.get()
    return val if val else ""


def set_trace_id(trace_id: str) -> contextvars.Token:
    """Set active distributed trace ID."""
    return _TRACE_ID_CTX.set(trace_id)


def get_span_id() -> str:
    """Get active span ID."""
    val = _SPAN_ID_CTX.get()
    return val if val else ""


def set_span_id(span_id: str) -> contextvars.Token:
    """Set active span ID."""
    return _SPAN_ID_CTX.set(span_id)


def get_tenant_id() -> str:
    """Get current tenant identifier."""
    return _TENANT_ID_CTX.get()


def set_tenant_id(tenant_id: str) -> contextvars.Token:
    """Set current tenant identifier."""
    return _TENANT_ID_CTX.set(tenant_id)


@contextmanager
def request_context(
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> Generator[None, None, None]:
    """Context manager setting request, trace, and tenant identifiers for the current task/thread."""
    r_id = request_id or f"req_{uuid.uuid4().hex[:12]}"
    t_id = trace_id or f"trace_{uuid.uuid4().hex[:16]}"
    ten_id = tenant_id or "default"

    tok_req = _REQUEST_ID_CTX.set(r_id)
    tok_trace = _TRACE_ID_CTX.set(t_id)
    tok_tenant = _TENANT_ID_CTX.set(ten_id)

    try:
        yield
    finally:
        _REQUEST_ID_CTX.reset(tok_req)
        _TRACE_ID_CTX.reset(tok_trace)
        _TENANT_ID_CTX.reset(tok_tenant)
