"""Deterministic mock reranker implementation for unit tests and local development."""

from __future__ import annotations

import asyncio
import time
from typing import List, Optional, Sequence

from rag.retrieval.exceptions import RerankingProviderError, RerankingTimeoutError
from rag.retrieval.models import RetrievedChunkResult
from rag.retrieval.reranking.interfaces import RerankResponse, RerankResult, Reranker


class MockReranker:
    """Mock reranker that produces deterministic, predictable relevance scores.

    Scoring logic:
    - Base score = 1.0 / (1.0 + len(doc.text))
    - Plus small deterministic bonus for chunk_id hash to break ties
    - Allows configurable failure or timeout simulation for resilience testing.
    """

    def __init__(
        self,
        model_name: str = "mock-reranker",
        model_version: str = "v1.0",
        *,
        should_fail: bool = False,
        should_timeout: bool = False,
        simulated_latency_sec: float = 0.0,
        fail_message: str = "Simulated mock reranker failure",
    ):
        self._model_name = model_name
        self._model_version = model_version
        self.should_fail = should_fail
        self.should_timeout = should_timeout
        self.simulated_latency_sec = simulated_latency_sec
        self.fail_message = fail_message
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    def _score_documents(
        self,
        query: str,
        documents: Sequence[RetrievedChunkResult],
        top_k: int,
    ) -> RerankResponse:
        start_time = time.perf_counter()
        self.call_count += 1

        if self.should_timeout:
            raise RerankingTimeoutError("Mock reranker timed out")

        if self.should_fail:
            raise RerankingProviderError(self.fail_message)

        if not documents:
            return RerankResponse(
                results=[],
                model=self._model_name,
                latency_ms=0.0,
                input_count=0,
                output_count=0,
                input_tokens=0,
                reranking_failed=False,
                fallback_used=False,
            )

        # Raw scores: inverse text length + deterministic hash offset
        raw_scores: List[float] = []
        total_tokens = len(query)

        for doc in documents:
            doc_len = len(doc.text) if doc.text else 0
            total_tokens += doc_len
            # Deterministic hash in range [0, 0.01] for unique ordering on equal lengths
            hash_val = (abs(hash(doc.chunk_id)) % 1000) / 100000.0
            raw = (1000.0 / (10.0 + doc_len)) + hash_val
            raw_scores.append(raw)

        # Min-max normalization
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        score_range = max_score - min_score

        normalized_scores: List[float] = []
        for s in raw_scores:
            if score_range > 1e-9:
                normalized_scores.append((s - min_score) / score_range)
            else:
                normalized_scores.append(1.0)

        # Pair up and rank descending by score, tie-breaking on chunk_id ascending
        ranked_pairs = sorted(
            zip(documents, raw_scores, normalized_scores),
            key=lambda item: (-item[1], item[0].chunk_id),
        )

        selected = ranked_pairs[: max(1, top_k)]
        results: List[RerankResult] = []
        for rank_idx, (doc, raw_s, norm_s) in enumerate(selected, start=1):
            results.append(
                RerankResult(
                    chunk_id=doc.chunk_id,
                    reranker_raw_score=round(raw_s, 6),
                    reranker_normalized_score=round(norm_s, 6),
                    final_rank=rank_idx,
                    original_result=doc,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return RerankResponse(
            results=results,
            model=self._model_name,
            latency_ms=round(latency_ms, 3),
            input_count=len(documents),
            output_count=len(results),
            input_tokens=total_tokens,
            reranking_failed=False,
            fallback_used=False,
        )

    async def rerank(
        self,
        query: str,
        documents: Sequence[RetrievedChunkResult],
        *,
        top_k: int,
    ) -> RerankResponse:
        if self.simulated_latency_sec > 0:
            await asyncio.sleep(self.simulated_latency_sec)
        return self._score_documents(query, documents, top_k)

    def rerank_sync(
        self,
        query: str,
        documents: Sequence[RetrievedChunkResult],
        *,
        top_k: int,
    ) -> RerankResponse:
        if self.simulated_latency_sec > 0:
            time.sleep(self.simulated_latency_sec)
        return self._score_documents(query, documents, top_k)
