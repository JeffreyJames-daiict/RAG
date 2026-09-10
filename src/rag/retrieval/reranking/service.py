"""Production orchestration service for candidate reranking."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import copy
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

from rag.core.config import RerankingConfig
from rag.retrieval.exceptions import (
    RerankingError,
    RerankingProviderError,
    RerankingTimeoutError,
)
from rag.retrieval.models import RetrievedChunkResult
from rag.retrieval.reranking.interfaces import RerankResponse, RerankResult, Reranker
from rag.retrieval.reranking.mock_provider import MockReranker

logger = logging.getLogger(__name__)


def get_reranker(config: RerankingConfig) -> Reranker:
    """Factory to instantiate the appropriate reranker provider from config."""
    provider_type = (config.provider or "mock").lower()
    if provider_type == "mock":
        return MockReranker(
            model_name=config.model,
            model_version=config.model_version,
        )
    # Extensible for cross_encoder, cohere, openrouter
    raise ValueError(f"Unsupported reranker provider: '{config.provider}'")


class RerankingService:
    """Orchestrates candidate preparation, batching, provider calls, caching, and fallback."""

    def __init__(
        self,
        config: Optional[RerankingConfig] = None,
        provider: Optional[Reranker] = None,
    ):
        self.config = config or RerankingConfig()
        self.provider: Reranker = provider or get_reranker(self.config)
        self._cache: OrderedDict[Tuple, RerankResponse] = OrderedDict()

    def _get_cache_key(
        self,
        tenant_id: str,
        query: str,
        documents: Sequence[RetrievedChunkResult],
        top_k: int,
    ) -> Tuple:
        doc_ids = tuple(sorted(d.chunk_id for d in documents))
        return (
            tenant_id,
            query,
            doc_ids,
            top_k,
            self.provider.model_name,
            self.provider.model_version,
        )

    def _check_cache(self, key: Tuple) -> Optional[RerankResponse]:
        if not self.config.cache_enabled:
            return None
        if key in self._cache:
            self._cache.move_to_end(key)
            # Return deepcopy to ensure caller mutations don't corrupt cached objects
            return copy.deepcopy(self._cache[key])
        return None

    def _store_cache(self, key: Tuple, response: RerankResponse) -> None:
        if not self.config.cache_enabled:
            return
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = copy.deepcopy(response)
        while len(self._cache) > self.config.cache_max_size:
            self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        """Clear all cached responses."""
        self._cache.clear()

    def _prepare_documents(
        self, documents: Sequence[RetrievedChunkResult]
    ) -> List[RetrievedChunkResult]:
        """Prepare documents by applying text length limits without altering source objects."""
        max_doc_len = self.config.max_document_length
        prepared: List[RetrievedChunkResult] = []
        for doc in documents:
            if doc.text and len(doc.text) > max_doc_len:
                doc_copy = doc.model_copy(update={"text": doc.text[:max_doc_len]})
                prepared.append(doc_copy)
            else:
                prepared.append(doc)
        return prepared

    def _map_fallback(
        self,
        documents: Sequence[RetrievedChunkResult],
        top_k: int,
        error_msg: str,
        start_time: float,
    ) -> RerankResponse:
        """Construct a graceful fallback response preserving RRF ranking."""
        selected = list(documents[: max(1, top_k)])
        results: List[RerankResult] = []
        for rank_idx, doc in enumerate(selected, start=1):
            doc.reranking_failed = True
            results.append(
                RerankResult(
                    chunk_id=doc.chunk_id,
                    reranker_raw_score=0.0,
                    reranker_normalized_score=0.0,
                    final_rank=rank_idx,
                    original_result=doc,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return RerankResponse(
            results=results,
            model=getattr(self.provider, "model_name", "fallback"),
            latency_ms=round(latency_ms, 3),
            input_count=len(documents),
            output_count=len(results),
            input_tokens=0,
            reranking_failed=True,
            fallback_used=True,
            error_message=error_msg,
        )

    def _hydrate_results(
        self, rerank_response: RerankResponse
    ) -> List[RetrievedChunkResult]:
        """Update retrieved chunk models with reranker scores and new final ranks."""
        hydrated: List[RetrievedChunkResult] = []
        for r in rerank_response.results:
            chunk = r.original_result
            chunk.rank = r.final_rank
            chunk.reranker_raw_score = r.reranker_raw_score
            chunk.reranker_normalized_score = r.reranker_normalized_score
            chunk.reranker_model = rerank_response.model
            chunk.reranking_failed = rerank_response.reranking_failed
            hydrated.append(chunk)
        return hydrated

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunkResult],
        *,
        top_k: Optional[int] = None,
        tenant_id: str = "default",
    ) -> Tuple[List[RetrievedChunkResult], RerankResponse]:
        """Asynchronously rerank candidates, returning both hydrated chunks and telemetry."""
        start_time = time.perf_counter()
        target_top_k = top_k or self.config.top_k

        if not candidates:
            empty_resp = RerankResponse(
                results=[],
                model=getattr(self.provider, "model_name", "unknown"),
                latency_ms=0.0,
                input_count=0,
                output_count=0,
                input_tokens=0,
            )
            return [], empty_resp

        # Truncate query if needed
        cleaned_query = query[: self.config.max_query_length] if query else ""

        # Check Cache
        cache_key = self._get_cache_key(tenant_id, cleaned_query, candidates, target_top_k)
        cached = self._check_cache(cache_key)
        if cached is not None:
            return self._hydrate_results(cached), cached

        prepared_docs = self._prepare_documents(candidates)

        try:
            # Enforce timeout budget
            resp = await asyncio.wait_for(
                self.provider.rerank(cleaned_query, prepared_docs, top_k=target_top_k),
                timeout=self.config.timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            logger.warning("Reranker provider timed out after %.2fs", self.config.timeout_sec)
            if not self.config.fallback_enabled:
                raise RerankingTimeoutError(
                    f"Reranking exceeded timeout of {self.config.timeout_sec}s"
                ) from exc
            resp = self._map_fallback(candidates, target_top_k, "Timeout", start_time)
        except Exception as exc:
            logger.warning("Reranker provider failed: %s", exc)
            if not self.config.fallback_enabled:
                if isinstance(exc, RerankingError):
                    raise
                raise RerankingProviderError(f"Reranking provider failed: {exc}") from exc
            resp = self._map_fallback(candidates, target_top_k, str(exc), start_time)

        if not resp.reranking_failed:
            self._store_cache(cache_key, resp)

        hydrated = self._hydrate_results(resp)
        return hydrated, resp

    def rerank_sync(
        self,
        query: str,
        candidates: Sequence[RetrievedChunkResult],
        *,
        top_k: Optional[int] = None,
        tenant_id: str = "default",
    ) -> Tuple[List[RetrievedChunkResult], RerankResponse]:
        """Synchronously rerank candidates, returning both hydrated chunks and telemetry."""
        start_time = time.perf_counter()
        target_top_k = top_k or self.config.top_k

        if not candidates:
            empty_resp = RerankResponse(
                results=[],
                model=getattr(self.provider, "model_name", "unknown"),
                latency_ms=0.0,
                input_count=0,
                output_count=0,
                input_tokens=0,
            )
            return [], empty_resp

        cleaned_query = query[: self.config.max_query_length] if query else ""

        cache_key = self._get_cache_key(tenant_id, cleaned_query, candidates, target_top_k)
        cached = self._check_cache(cache_key)
        if cached is not None:
            return self._hydrate_results(cached), cached

        prepared_docs = self._prepare_documents(candidates)

        try:
            resp = self.provider.rerank_sync(cleaned_query, prepared_docs, top_k=target_top_k)
        except RerankingTimeoutError as exc:
            logger.warning("Reranker provider timed out: %s", exc)
            if not self.config.fallback_enabled:
                raise
            resp = self._map_fallback(candidates, target_top_k, str(exc), start_time)
        except Exception as exc:
            logger.warning("Reranker provider failed: %s", exc)
            if not self.config.fallback_enabled:
                if isinstance(exc, RerankingError):
                    raise
                raise RerankingProviderError(f"Reranking provider failed: {exc}") from exc
            resp = self._map_fallback(candidates, target_top_k, str(exc), start_time)

        if not resp.reranking_failed:
            self._store_cache(cache_key, resp)

        hydrated = self._hydrate_results(resp)
        return hydrated, resp
