"""BM25 Retrieval Service — high-level orchestrator for sparse keyword search.

Wires together: query validation → BM25 inverted index search → candidate ranking → response hydration.

Implements both synchronous and asynchronous retrieval conforming to the
standard retrieval service protocol, with comprehensive diagnostics and observability.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

from rag.core.config import BM25Config, get_settings
from rag.core.database import DatabaseManager
from rag.retrieval.bm25 import BM25Retriever, TenantBM25IndexManager
from rag.retrieval.exceptions import RetrievalError
from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.interfaces import RetrievedCandidate
from rag.retrieval.models import RetrievalRequest, RetrievalResponse, RetrievedChunkResult
from rag.retrieval.validators import QueryValidator, ResultValidator

logger = logging.getLogger(__name__)


def _query_hash(query: str) -> str:
    """Compute a safe, non-reversible hash ID for log correlation without leaking query text."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]


class BM25RetrievalService:
    """Production service for BM25 sparse keyword retrieval.

    Accepts a user query, validates request parameters, performs weighted
    lexical search across tenant-isolated inverted indices, validates and hydrates
    results with full provenance metadata.

    Usage::

        from rag.core.database import DatabaseManager
        from rag.retrieval.bm25_service import BM25RetrievalService
        from rag.retrieval.models import RetrievalRequest

        db_mgr = DatabaseManager(database_url)
        service = BM25RetrievalService(db_mgr)

        request = RetrievalRequest(query="US 11,234,567 B2", tenant_id="acme")
        response = service.retrieve(request)
        for r in response.results:
            print(f"[{r.rank}] BM25={r.raw_score:.4f} — {r.title}: {r.text[:60]}")
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        retriever: Optional[BM25Retriever] = None,
        config: Optional[BM25Config] = None,
    ):
        self.db_manager = db_manager
        self.config = config or get_settings().bm25
        self.retriever = retriever or BM25Retriever(
            db_manager=db_manager,
            k1=self.config.k1,
            b=self.config.b,
            config=self.config,
        )

    @property
    def index_manager(self) -> TenantBM25IndexManager:
        return self.retriever.index_manager

    # ── Primary Request-Based API ─────────────────────────────────────────

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Execute full BM25 sparse retrieval pipeline.

        Steps:
            1. Validate the retrieval request (query, tenant, top_k).
            2. Search the tenant-isolated BM25 index.
            3. Validate each candidate result.
            4. Hydrate and return the response with full provenance.

        Args:
            request: Validated retrieval request.

        Returns:
            RetrievalResponse with ranked, hydrated results.

        Raises:
            RetrievalError: On invalid request or search failure.
        """
        total_start = time.time()
        qhash = _query_hash(request.query)

        # 1. Validate request
        QueryValidator.validate_request(request, max_top_k=self.config.top_k * 20 if self.config.top_k else 100)

        # 2. Search via BM25 retriever
        search_start = time.time()
        try:
            candidates, total_candidates = self.retriever.search(request)
        except Exception as e:
            logger.error("[%s] BM25 search failed: %s", qhash, e, exc_info=True)
            raise RetrievalError(f"BM25 search failed: {e}") from e

        search_ms = (time.time() - search_start) * 1000.0

        # 3. Validate candidate results
        for cand in candidates:
            ResultValidator.validate_candidate(cand, expected_tenant_id=request.tenant_id)

        # 4. Hydrate into RetrievedChunkResult list
        hydrate_start = time.time()
        results: List[RetrievedChunkResult] = []
        for rank_idx, cand in enumerate(candidates, start=1):
            sec_path = self._extract_section_path(cand)
            result = RetrievedChunkResult(
                chunk_id=cand.chunk_id,
                document_id=cand.document_id,
                document_version_id=f"v{cand.document_version}",
                document_version=cand.document_version,
                tenant_id=request.tenant_id,
                text=cand.text,
                title=cand.title or cand.filename,
                section_path=sec_path,
                heading=cand.section,
                page_number=cand.page,
                page=cand.page,
                source_uri=cand.source_uri,
                filename=cand.filename,
                mime_type=cand.mime_type,
                raw_score=cand.raw_score,
                normalized_score=cand.normalized_score,
                similarity_score=cand.raw_score,
                bm25_score=cand.raw_score,
                rank=rank_idx,
                retrieval_method="bm25",
                chunk_index=cand.chunk_index,
                chunk_type=cand.chunk_metadata.get("chunk_type", "text"),
                chunking_version=cand.chunking_version,
                content_hash=cand.content_hash,
                metadata=cand.chunk_metadata,
            )
            results.append(result)

        hydrate_ms = (time.time() - hydrate_start) * 1000.0
        total_ms = (time.time() - total_start) * 1000.0

        status = self.index_manager.get_status(request.tenant_id)

        diagnostics: Dict[str, Any] = {
            "query_hash": qhash,
            "tenant_id": request.tenant_id,
            "total_candidates": total_candidates,
            "returned_results": len(results),
            "search_latency_ms": round(search_ms, 2),
            "hydrate_latency_ms": round(hydrate_ms, 2),
            "total_latency_ms": round(total_ms, 2),
            "bm25_k1": self.config.k1,
            "bm25_b": self.config.b,
            "index_freshness_sec": status.get("index_freshness_sec", 0.0),
            "index_version": status.get("index_version", "v1.0"),
            "corpus_size": status.get("corpus_size", total_candidates),
        }

        logger.info(
            "[%s] BM25 retrieved %d/%d candidates in %.2f ms (search=%.2f ms, hydrate=%.2f ms)",
            qhash, len(results), total_candidates, total_ms, search_ms, hydrate_ms,
        )

        return RetrievalResponse(
            query=request.query,
            tenant_id=request.tenant_id,
            results=results,
            total_candidates=total_candidates,
            returned_results=len(results),
            retrieval_method="bm25",
            retrieval_backend="bm25_inverted_index",
            index_version="v1.0",
            filters_applied=request.filters.model_dump(exclude_none=True) if request.filters else None,
            latency_ms=round(total_ms, 2),
            duration_ms=round(total_ms, 2),
            search_latency_ms=round(search_ms, 2),
            bm25_latency_ms=round(search_ms, 2),
            diagnostics=diagnostics,
        )

    # ── Convenient Parameter-Based APIs ───────────────────────────────────

    def retrieve_sync(
        self,
        query: str,
        *,
        tenant_id: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilters] = None,
        similarity_threshold: Optional[float] = None,
        include_inactive: bool = False,
    ) -> RetrievalResponse:
        """Synchronously execute BM25 search with convenient parameter signature."""
        request = RetrievalRequest(
            query=query,
            tenant_id=tenant_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filters=filters,
            include_inactive=include_inactive,
        )
        return self.retrieve(request)

    async def retrieve_async(
        self,
        query: str,
        *,
        tenant_id: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilters] = None,
        similarity_threshold: Optional[float] = None,
        include_inactive: bool = False,
    ) -> RetrievalResponse:
        """Asynchronously execute BM25 search offloaded to worker thread."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.retrieve_sync(
                query=query,
                tenant_id=tenant_id,
                top_k=top_k,
                filters=filters,
                similarity_threshold=similarity_threshold,
                include_inactive=include_inactive,
            ),
        )

    # ── Provenance Utilities ──────────────────────────────────────────────

    @staticmethod
    def _extract_section_path(cand: RetrievedCandidate) -> List[str]:
        """Extract or construct hierarchical section breadcrumbs from candidate metadata."""
        meta = cand.chunk_metadata or {}
        if "section_path" in meta and isinstance(meta["section_path"], (list, tuple)):
            return [str(p) for p in meta["section_path"]]
        if "sections" in meta and isinstance(meta["sections"], (list, tuple)):
            return [str(p) for p in meta["sections"]]
        if cand.section:
            return [cand.section]
        return []
