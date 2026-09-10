"""Distributed in-process tracing with OpenTelemetry-compatible span models and W3C traceparent propagation."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from rag.observability.context import (
    get_span_id,
    get_tenant_id,
    get_trace_id,
    set_span_id,
    set_trace_id,
)

logger = logging.getLogger(__name__)


class Span:
    """Represents a single timed unit of work in the RAG execution graph."""

    def __init__(
        self,
        name: str,
        trace_id: str,
        span_id: str,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.attributes: Dict[str, Any] = attributes or {}
        self.start_time = time.perf_counter()
        self.end_time: Optional[float] = None
        self.duration_ms: float = 0.0
        self.status: str = "OK"
        self.error: Optional[str] = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Add an operational attribute to this span."""
        self.attributes[key] = value

    def record_error(self, error: Exception) -> None:
        """Mark span as failed and capture exception summary."""
        self.status = "ERROR"
        self.error = f"{type(error).__name__}: {str(error)}"
        self.set_attribute("error.type", type(error).__name__)
        self.set_attribute("error.message", str(error))

    def finish(self) -> None:
        """Close span, calculate duration, and record completion."""
        if self.end_time is None:
            self.end_time = time.perf_counter()
            self.duration_ms = round((self.end_time - self.start_time) * 1000.0, 2)
            self.set_attribute("duration_ms", self.duration_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize span to dictionary for reporting or log export."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "error": self.error,
        }


class SpanContextManager:
    """Supports both synchronous and asynchronous context management for a Span."""

    def __init__(self, tracer: Tracer, span: Span):
        self.tracer = tracer
        self.span = span
        self._prev_span_id_token = None

    def __enter__(self) -> Span:
        self._prev_span_id_token = set_span_id(self.span.span_id)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_val:
            self.span.record_error(exc_val)
        self.span.finish()
        self.tracer.record_completed_span(self.span)
        if self._prev_span_id_token:
            set_span_id(self.span.parent_span_id or "")
        return False  # Do not suppress exceptions

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return self.__exit__(exc_type, exc_val, exc_tb)


class Tracer:
    """In-process distributed tracer managing span lifecycles and W3C context headers."""

    def __init__(self):
        self._completed_spans: List[Span] = []

    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
    ) -> SpanContextManager:
        """Create and start a new child span."""
        # Ensure active trace_id exists
        current_trace_id = get_trace_id()
        if not current_trace_id:
            current_trace_id = uuid.uuid4().hex
            set_trace_id(current_trace_id)

        active_parent = parent_span_id or get_span_id() or None
        new_span_id = uuid.uuid4().hex[:16]

        attr = attributes.copy() if attributes else {}
        ten = get_tenant_id()
        if ten and "tenant_id" not in attr:
            attr["tenant_id"] = ten

        span = Span(
            name=name,
            trace_id=current_trace_id,
            span_id=new_span_id,
            parent_span_id=active_parent,
            attributes=attr,
        )

        return SpanContextManager(self, span)

    def record_completed_span(self, span: Span) -> None:
        """Store span for local trace tree inspection."""
        self._completed_spans.append(span)

    def get_completed_spans(self) -> List[Span]:
        """Return list of all spans finished by this tracer."""
        return list(self._completed_spans)

    def clear(self) -> None:
        """Reset completed spans."""
        self._completed_spans.clear()

    @staticmethod
    def create_traceparent(trace_id: str, span_id: str) -> str:
        """Generate a standard W3C traceparent header: '00-{32hex}-{16hex}-01'."""
        tid = trace_id.replace("-", "")[:32].zfill(32)
        sid = span_id.replace("-", "")[:16].zfill(16)
        return f"00-{tid}-{sid}-01"

    @staticmethod
    def parse_traceparent(header_value: str) -> Optional[Tuple[str, str]]:
        """Parse W3C traceparent header into (trace_id, parent_span_id)."""
        if not header_value:
            return None
        parts = header_value.strip().split("-")
        if len(parts) >= 3 and len(parts[1]) == 32 and len(parts[2]) == 16:
            return parts[1], parts[2]
        return None


_GLOBAL_TRACER: Tracer = Tracer()


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    return _GLOBAL_TRACER
