"""Cache key models and cache domain definitions with strict multi-tenant isolation."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class CacheDomain(str, Enum):
    """Isolated cache partitions for different RAG pipeline artifacts."""

    QUERY_EMBEDDINGS = "query_embeddings"
    RETRIEVAL_RESULTS = "retrieval_results"
    RERANKING_RESULTS = "reranking_results"
    LLM_RESPONSES = "llm_responses"


class CacheKey(BaseModel):
    """Deterministic, tenant-isolated cache key incorporating all relevant artifact versions."""

    tenant_id: str = Field(..., description="Tenant organization key (strict partition boundary)")
    domain: CacheDomain = Field(..., description="Target cache domain")
    query_text: str = Field(default="", description="Original query text")
    query_hash: str = Field(default="", description="SHA-256 hash of the normalized query text")
    embedding_model: str = Field(default="", description="Embedding model name and version")
    retrieval_config_hash: str = Field(default="", description="Hash of retrieval filters, top_k, and fusion parameters")
    document_index_version: str = Field(default="v1", description="Active document index / corpus version")
    reranker_model: str = Field(default="", description="Reranker model name and version")
    prompt_version: str = Field(default="", description="Prompt template version string")

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.query_hash and self.query_text:
            normalized = " ".join(self.query_text.strip().lower().split())
            self.query_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]

    def to_key_string(self) -> str:
        """Render deterministic cache key prefixed strictly with tenant_id."""
        parts = [
            f"t={self.tenant_id}",
            f"d={self.domain.value}",
            f"q={self.query_hash}",
        ]
        if self.embedding_model:
            parts.append(f"em={self.embedding_model}")
        if self.retrieval_config_hash:
            parts.append(f"rc={self.retrieval_config_hash}")
        if self.document_index_version:
            parts.append(f"div={self.document_index_version}")
        if self.reranker_model:
            parts.append(f"rm={self.reranker_model}")
        if self.prompt_version:
            parts.append(f"pv={self.prompt_version}")

        return ":".join(parts)

    @classmethod
    def compute_config_hash(cls, config_dict: Dict[str, Any]) -> str:
        """Generate short hash of retrieval configuration dict."""
        serialized = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
