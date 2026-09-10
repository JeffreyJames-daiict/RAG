"""Pydantic request and response models for Vector Retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from rag.retrieval.exceptions import (
    EmptyQueryError,
    QueryTooLongError,
    TenantContextMissingError,
)
from rag.retrieval.filters import RetrievalFilters


class RetrievalRequest(BaseModel):
    """User request for vector semantic retrieval."""

    query: str = Field(..., min_length=1, max_length=32000, description="Natural language search query")
    tenant_id: str = Field(default="default", min_length=1, max_length=128, description="Tenant organization key for multi-tenant isolation")
    top_k: int = Field(default=5, ge=1, le=1000, description="Number of top relevant chunks to retrieve")
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Minimum cosine similarity score required for candidate chunks",
    )
    filters: Optional[RetrievalFilters] = Field(default=None, description="Typed metadata filters")
    include_inactive: bool = Field(default=False, description="Whether to include soft-deleted or non-indexed documents")
    query_embedding_override: Optional[List[float]] = Field(
        default=None,
        description="Optional vector override for testing and diagnostic purposes only",
    )
    max_query_length: int = Field(default=4000, ge=1, le=32000, description="Maximum allowed query length in characters")

    @field_validator("query")
    @classmethod
    def _validate_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise EmptyQueryError("Retrieval query must not be empty or whitespace-only.")
        return stripped

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise TenantContextMissingError("tenant_id must be provided for tenant-isolated retrieval.")
        return stripped

    @model_validator(mode="after")
    def _validate_length(self) -> "RetrievalRequest":
        if len(self.query) > self.max_query_length:
            raise QueryTooLongError(
                f"Retrieval query length ({len(self.query)}) exceeds maximum permitted ({self.max_query_length})."
            )
        return self


class RetrievedChunkResult(BaseModel):
    """Richly-hydrated retrieval candidate chunk with complete provenance and score."""

    chunk_id: str
    document_id: str
    document_version_id: str = ""
    tenant_id: str = "default"
    text: str
    title: str = ""
    section_path: List[str] = Field(default_factory=list, description="Hierarchical breadcrumb of section headers")
    page_number: Optional[int] = None
    source_uri: str = ""
    parent_chunk_id: Optional[str] = None
    asset_id: Optional[str] = None
    raw_score: float = 0.0
    normalized_score: float = 0.0
    similarity_score: float = Field(default=0.0, description="Cosine similarity score between query and chunk vector")
    rank: int = Field(default=1, ge=1, description="1-based rank in retrieval results")
    retrieval_method: str = "vector"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Backwards-compatible fields
    document_version: int = 1
    heading: Optional[str] = None
    page: Optional[int] = None
    filename: str = ""
    mime_type: str = ""
    chunk_index: int = 0
    chunk_type: str = "text"
    chunking_version: str = "v1.0"
    content_hash: str = ""
    embedding_model: str = ""
    embedding_provider: str = ""
    embedding_dimension: int = 0
    table_id: Optional[str] = None
    rrf_score: Optional[float] = Field(default=None, description="Reciprocal Rank Fusion score if hybrid retrieval used")
    bm25_score: Optional[float] = Field(default=None, description="BM25 keyword search score if hybrid/BM25 retrieval used")
    dense_raw_score: Optional[float] = Field(default=None, description="Raw similarity score from dense vector search")
    sparse_raw_score: Optional[float] = Field(default=None, description="Raw score from sparse BM25 search")
    dense_rank: Optional[int] = Field(default=None, description="1-based rank in dense vector candidate list")
    sparse_rank: Optional[int] = Field(default=None, description="1-based rank in sparse BM25 candidate list")
    retrieval_methods: List[str] = Field(default_factory=list, description="List of retrieval methods that matched this chunk")
    fusion_strategy: Optional[str] = Field(default=None, description="Fusion algorithm used (e.g. 'rrf', 'combsum')")
    reranker_raw_score: Optional[float] = Field(default=None, description="Raw score assigned by reranker model")
    reranker_normalized_score: Optional[float] = Field(default=None, description="Normalized score assigned by reranker (0 to 1)")
    reranker_model: Optional[str] = Field(default=None, description="Model identifier used for reranking")
    reranking_failed: bool = Field(default=False, description="Whether reranking failed for this result or overall")

    @model_validator(mode="after")
    def _sync_fields(self) -> "RetrievedChunkResult":
        # Keep aliases synchronized
        if not self.document_version_id and self.document_version:
            self.document_version_id = f"v{self.document_version}"
        if self.page is not None and self.page_number is None:
            self.page_number = self.page
        elif self.page_number is not None and self.page is None:
            self.page = self.page_number
        if not self.title and self.filename:
            self.title = self.filename
        if self.raw_score == 0.0 and self.similarity_score != 0.0:
            self.raw_score = self.similarity_score
        if self.normalized_score == 0.0 and self.similarity_score != 0.0:
            # Cosine similarity in [-1, 1] mapped to [0, 1]
            self.normalized_score = max(0.0, min(1.0, (self.similarity_score + 1.0) / 2.0))
        if self.similarity_score == 0.0 and self.raw_score != 0.0:
            self.similarity_score = self.raw_score
        if self.dense_raw_score is not None and self.similarity_score == 0.0:
            self.similarity_score = self.dense_raw_score
        elif self.similarity_score != 0.0 and self.dense_raw_score is None:
            self.dense_raw_score = self.similarity_score
        if self.sparse_raw_score is not None and self.bm25_score is None:
            self.bm25_score = self.sparse_raw_score
        elif self.bm25_score is not None and self.sparse_raw_score is None:
            self.sparse_raw_score = self.bm25_score
        if not self.retrieval_methods and self.retrieval_method:
            self.retrieval_methods = [self.retrieval_method]
        return self


class RetrievalResponse(BaseModel):
    """Top-level response returned by the Retrieval Services."""

    query: str
    tenant_id: str
    results: List[RetrievedChunkResult] = Field(default_factory=list)
    total_candidates: int = 0
    returned_results: int = 0
    retrieval_method: str = "vector"
    retrieval_backend: str = "python"
    embedding_model: str = ""
    embedding_dimensions: int = 0
    embedding_dimension: int = 0
    index_version: str = "v1.0"
    filters_applied: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    duration_ms: float = 0.0
    query_embedding_latency_ms: float = 0.0
    search_latency_ms: float = 0.0
    bm25_latency_ms: Optional[float] = None
    fusion_latency_ms: Optional[float] = None
    fusion_method: Optional[str] = None
    reranking_latency_ms: Optional[float] = Field(default=None, description="Latency of reranking step in milliseconds")
    reranking_method: Optional[str] = Field(default=None, description="Reranking model/provider used")
    diagnostics: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sync_response(self) -> "RetrievalResponse":
        if self.duration_ms and not self.latency_ms:
            self.latency_ms = self.duration_ms
        elif self.latency_ms and not self.duration_ms:
            self.duration_ms = self.latency_ms
        if self.embedding_dimension and not self.embedding_dimensions:
            self.embedding_dimensions = self.embedding_dimension
        elif self.embedding_dimensions and not self.embedding_dimension:
            self.embedding_dimension = self.embedding_dimensions
        if not self.returned_results:
            self.returned_results = len(self.results)
        return self
