"""Structured JSON logging with contextual trace metadata and automatic secret/PII redaction."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from rag.observability.context import get_request_id, get_tenant_id, get_trace_id

# Regex patterns matching secrets and sensitive patterns
SECRET_PATTERNS = [
    re.compile(r"(?i)(bearer\s+)[a-zA-Z0-9_\-\.]{15,}", re.IGNORECASE),
    re.compile(r"(?i)(api[_-]?key[\"'\s:=]+)[a-zA-Z0-9_\-\.]{15,}", re.IGNORECASE),
    re.compile(r"(?i)(sk-[a-zA-Z0-9]{20,})", re.IGNORECASE),
    re.compile(r"(?i)(password[\"'\s:=]+)[^\s,\"']+", re.IGNORECASE),
    re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),  # Credit card
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
]


def mask_sensitive_data(val: Any, max_text_len: int = 500) -> Any:
    """Recursively mask secrets, auth credentials, and truncate raw unrestricted text dumps."""
    if isinstance(val, str):
        masked = val
        for pat in SECRET_PATTERNS:
            masked = pat.sub(r"\1******", masked) if r"\1" in pat.pattern else pat.sub("******", masked)
        if len(masked) > max_text_len:
            masked = masked[:max_text_len] + f"... [truncated {len(masked) - max_text_len} chars]"
        return masked

    if isinstance(val, dict):
        sanitized = {}
        for k, v in val.items():
            if any(s in k.lower() for s in ("token", "secret", "password", "key", "auth", "credential")):
                sanitized[k] = "******"
            elif k.lower() in ("raw_content", "full_document", "document_bytes"):
                sanitized[k] = "[redacted raw content]"
            else:
                sanitized[k] = mask_sensitive_data(v, max_text_len)
        return sanitized

    if isinstance(val, (list, tuple)):
        return [mask_sensitive_data(x, max_text_len) for x in val]

    return val


class StructuredJsonFormatter(logging.Formatter):
    """Custom logging formatter rendering log records as secure, standardized JSON."""

    RAG_FIELDS = (
        "retrieval_method",
        "candidate_counts",
        "reranking_status",
        "context_token_count",
        "model",
        "provider",
        "latency_ms",
        "error_category",
        "fallback_usage",
        "query_length",
        "top_k",
        "cache_hit",
    )

    def format(self, record: logging.LogRecord) -> str:
        # Base metadata
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": mask_sensitive_data(record.getMessage()),
        }

        # Contextual identifiers
        req_id = get_request_id()
        if req_id:
            log_obj["request_id"] = req_id

        tr_id = get_trace_id()
        if tr_id:
            log_obj["trace_id"] = tr_id

        ten_id = get_tenant_id()
        if ten_id:
            log_obj["tenant_id"] = ten_id

        # Ingest recognized RAG operational fields from extra attributes
        for field in self.RAG_FIELDS:
            if hasattr(record, field):
                log_obj[field] = mask_sensitive_data(getattr(record, field))

        # Capture exception information if present
        if record.exc_info:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "UnknownException",
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_obj)


def configure_structured_logging(level: int = logging.INFO) -> None:
    """Configure root handler with StructuredJsonFormatter."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    handler = logging.StreamHandler()
    handler.setFormatter(StructuredJsonFormatter())
    root_logger.addHandler(handler)
