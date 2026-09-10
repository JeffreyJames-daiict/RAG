"""Interfaces and data contracts for the Reranking layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, runtime_checkable

from rag.retrieval.models import RetrievedChunkResult


@dataclass
class RerankResult:
    """Individual reranked candidate chunk result with raw and normalized scores."""

    chunk_id: str
    reranker_raw_score: float
    reranker_normalized_score: float  # Normalized to [0.0, 1.0]
    final_rank: int  # 1-based rank position after reranking
    original_result: RetrievedChunkResult  # Preserves all vector, BM25, RRF fields


@dataclass
class RerankResponse:
    """Aggregated response produced by a Reranker provider or RerankingService."""

    results: List[RerankResult] = field(default_factory=list)
    model: str = "unknown"
    latency_ms: float = 0.0
    input_count: int = 0
    output_count: int = 0
    input_tokens: int = 0  # Total characters/tokens evaluated
    reranking_failed: bool = False
    fallback_used: bool = False
    error_message: Optional[str] = None


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranker providers (e.g. Mock, Cross-Encoder, Cohere, OpenRouter)."""

    @property
    def model_name(self) -> str:
        """Name or checkpoint of the reranking model."""
        ...

    @property
    def model_version(self) -> str:
        """Version string of the model or provider."""
        ...

    async def rerank(
        self,
        query: str,
        documents: Sequence[RetrievedChunkResult],
        *,
        top_k: int,
    ) -> RerankResponse:
        """Asynchronously compute relevance scores and return reranked candidates."""
        ...

    def rerank_sync(
        self,
        query: str,
        documents: Sequence[RetrievedChunkResult],
        *,
        top_k: int,
    ) -> RerankResponse:
        """Synchronously compute relevance scores and return reranked candidates."""
        ...
