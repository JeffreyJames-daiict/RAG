"""Hybrid Retrieval Service — orchestrates parallel dense vector and sparse BM25 retrieval with RRF fusion.

Features:
- Parallel execution of dense and sparse branches with separate timeout budgets
- Four-tier candidate budgeting (dense_top_k, sparse_top_k, hybrid_candidate_k, final_top_k)
- Lossless deduplication preserving dense and sparse raw scores and 1-based ranks
- Resilient partial failure handling (graceful fallback when one branch fails)
- Shared security filters (tenant isolation, active version, non-deleted status, metadata filters)
- Rich diagnostics telemetry
- Both synchronous and asynchronous APIs
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from rag.core.config import RetrievalConfig, get_settings
from rag.core.database import DatabaseManager
from rag.embeddings.service import EmbeddingService
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.exceptions import RetrievalError, RetrievalTimeoutError
from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.fusion import FusedCandidateInfo, FusionDiagnostics, RRFFusion
from rag.retrieval.interfaces import RetrievedCandidate
from rag.retrieval.models import RetrievalRequest, RetrievalResponse, RetrievedChunkResult
from rag.retrieval.repositories import BaseVectorRepository, get_vector_repository
from rag.retrieval.reranking import RerankingService
from rag.retrieval.validators import QueryValidator, ResultValidator
from rag.retrieval.vector_service import VectorRetrievalService

logger = logging.getLogger(__name__)


def _query_hash(query: str) -> str:
    """Compute a safe hash ID for log correlation without leaking query text."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]


class HybridRetrievalService:
    """Production service for Hybrid Retrieval combining Dense Vector + Sparse BM25 via RRF.

    Usage::

        from rag.core.database import DatabaseManager
        from rag.embeddings.service import EmbeddingService
        from rag.retrieval.hybrid_service import HybridRetrievalService
        from rag.retrieval.models import RetrievalRequest

        db_mgr = DatabaseManager(database_url)
        emb_svc = EmbeddingService(db_mgr)
        hybrid_retriever = HybridRetrievalService(db_mgr, emb_svc)

        request = RetrievalRequest(query="quarterly revenue growth", tenant_id="acme", top_k=5)
        response = hybrid_retriever.retrieve(request)
        for r in response.results:
            print(f"[{r.rank}] RRF={r.rrf_score:.4f} Dense={r.dense_raw_score} BM25={r.sparse_raw_score}")
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_service: EmbeddingService,
        vector_repository: Optional[BaseVectorRepository] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        config: Optional[RetrievalConfig] = None,
        reranking_service: Optional[RerankingService] = None,
    ):
        self.db_manager = db_manager
        self.embedding_service = embedding_service
        self.config = config or get_settings().retrieval
        self.vector_repository = vector_repository or get_vector_repository(db_manager, self.config)
        self.vector_service = VectorRetrievalService(
            db_manager=self.db_manager,
            embedding_service=self.embedding_service,
            repository=self.vector_repository,
            config=self.config,
        )
        self.bm25_retriever = bm25_retriever or BM25Retriever(
            db_manager=self.db_manager,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
            candidate_limit=self.config.candidate_limit,
            config=self.config.bm25,
        )
        self.fusion = RRFFusion(
            k=self.config.rrf_k,
            weights=self.config.get_rrf_weights(),
        )
        if reranking_service is not None:
            self.reranking_service: Optional[RerankingService] = reranking_service
        elif getattr(self.config, "reranking", None) and self.config.reranking.enabled:
            self.reranking_service = RerankingService(config=self.config.reranking)
        else:
            self.reranking_service = None

    # ── Primary Unified Retrieval API ─────────────────────────────────────

    def retrieve(
        self,
        query_or_request: Union[str, RetrievalRequest],
        *,
        tenant_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[RetrievalFilters] = None,
        include_inactive: bool = False,
    ) -> RetrievalResponse:
        """Synchronously execute hybrid retrieval combining dense vector and sparse BM25.

        Accepts either a RetrievalRequest object or keyword query parameters.
        Executes dense and sparse branches concurrently in a worker pool with separate timeout budgets.
        """
        request = self._coerce_request(
            query_or_request,
            tenant_id=tenant_id,
            top_k=top_k,
            filters=filters,
            include_inactive=include_inactive,
        )
        return self._execute_hybrid(request)

    def retrieve_sync(
        self,
        query: str,
        *,
        tenant_id: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilters] = None,
        include_inactive: bool = False,
    ) -> RetrievalResponse:
        """Convenience synchronous retrieval API."""
        return self.retrieve(
            query,
            tenant_id=tenant_id,
            top_k=top_k,
            filters=filters,
            include_inactive=include_inactive,
        )

    async def retrieve_async(
        self,
        query_or_request: Union[str, RetrievalRequest],
        *,
        tenant_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[RetrievalFilters] = None,
        include_inactive: bool = False,
    ) -> RetrievalResponse:
        """Asynchronously execute hybrid retrieval combining dense and sparse branches."""
        request = self._coerce_request(
            query_or_request,
            tenant_id=tenant_id,
            top_k=top_k,
            filters=filters,
            include_inactive=include_inactive,
        )
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._execute_hybrid(request))

    # ── Internal Orchestration Pipeline ───────────────────────────────────

    def _coerce_request(
        self,
        query_or_request: Union[str, RetrievalRequest],
        *,
        tenant_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[RetrievalFilters] = None,
        include_inactive: bool = False,
    ) -> RetrievalRequest:
        """Normalizes input into a validated RetrievalRequest instance."""
        if isinstance(query_or_request, RetrievalRequest):
            return query_or_request

        return RetrievalRequest(
            query=query_or_request,
            tenant_id=tenant_id or "default",
            top_k=top_k or self.config.final_top_k,
            filters=filters,
            include_inactive=include_inactive,
        )

    def _run_dense_branch(
        self,
        request: RetrievalRequest,
        dense_k: int,
    ) -> Tuple[List[RetrievedCandidate], int, float, float]:
        """Executes the dense vector retrieval branch.
        
        Returns:
            Tuple of (candidates, total_cands, embed_latency_ms, search_latency_ms)
        """
        dense_req = request.model_copy(update={"top_k": dense_k})
        embed_start = time.time()

        if request.query_embedding_override is not None:
            query_vector = request.query_embedding_override
        else:
            query_vector = self.embedding_service.embed_query(request.query)
        embed_ms = (time.time() - embed_start) * 1000.0

        search_start = time.time()
        candidates, total_cands = self.vector_repository.search_vectors(
            query_vector=query_vector,
            request=dense_req,
            model_name=self.embedding_service.model_name,
            model_version=self.embedding_service.model_version,
        )
        search_ms = (time.time() - search_start) * 1000.0

        return candidates, total_cands, embed_ms, search_ms

    def _run_sparse_branch(
        self,
        request: RetrievalRequest,
        sparse_k: int,
    ) -> Tuple[List[RetrievedCandidate], int, float]:
        """Executes the sparse BM25 retrieval branch.
        
        Returns:
            Tuple of (candidates, total_cands, search_latency_ms)
        """
        sparse_req = request.model_copy(update={"top_k": sparse_k})
        search_start = time.time()
        candidates, total_cands = self.bm25_retriever.search(sparse_req)
        search_ms = (time.time() - search_start) * 1000.0
        return candidates, total_cands, search_ms

    def _execute_hybrid(self, request: RetrievalRequest) -> RetrievalResponse:
        """Executes the complete parallel hybrid retrieval pipeline."""
        total_start = time.time()
        qhash = _query_hash(request.query)

        # 1. Validate request parameters
        QueryValidator.validate_request(request, max_top_k=self.config.max_top_k)

        # 2. Setup candidate limits
        dense_k = getattr(self.config, "dense_top_k", 50)
        sparse_k = getattr(self.config, "sparse_top_k", 50)
        hybrid_candidate_k = getattr(self.config, "hybrid_candidate_k", 50)
        final_top_k = request.top_k or getattr(self.config, "final_top_k", 5)

        dense_timeout = getattr(self.config, "hybrid_dense_timeout_sec", 10.0)
        sparse_timeout = getattr(self.config, "hybrid_sparse_timeout_sec", 10.0)
        allow_partial_failure = getattr(self.config, "allow_partial_failure", True)

        # 3. Parallel execution of Dense and Sparse retrievers
        vector_candidates: List[RetrievedCandidate] = []
        bm25_candidates: List[RetrievedCandidate] = []
        total_vector_cands = 0
        total_bm25_cands = 0
        query_embed_ms = 0.0
        vector_search_ms = 0.0
        bm25_ms = 0.0

        dense_error: Optional[Exception] = None
        sparse_error: Optional[Exception] = None
        failed_methods: List[str] = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(self._run_dense_branch, request, dense_k)
            sparse_future = executor.submit(self._run_sparse_branch, request, sparse_k)

            # Wait for dense branch
            try:
                vc, tvc, ems, sms = dense_future.result(timeout=dense_timeout)
                vector_candidates, total_vector_cands = vc, tvc
                query_embed_ms, vector_search_ms = ems, sms
            except FuturesTimeoutError as e:
                dense_error = RetrievalTimeoutError(f"Dense vector retrieval timed out after {dense_timeout}s")
                logger.warning("[%s] Dense retrieval timed out: %s", qhash, dense_error)
            except Exception as e:
                dense_error = e
                logger.warning("[%s] Dense vector retrieval failed: %s", qhash, e)

            # Wait for sparse branch
            try:
                bc, tbc, bms = sparse_future.result(timeout=sparse_timeout)
                bm25_candidates, total_bm25_cands = bc, tbc
                bm25_ms = bms
            except FuturesTimeoutError as e:
                sparse_error = RetrievalTimeoutError(f"Sparse BM25 retrieval timed out after {sparse_timeout}s")
                logger.warning("[%s] Sparse retrieval timed out: %s", qhash, sparse_error)
            except Exception as e:
                sparse_error = e
                logger.warning("[%s] Sparse BM25 retrieval failed: %s", qhash, e)

        # 4. Partial failure evaluation
        if dense_error is not None:
            failed_methods.append("vector")
        if sparse_error is not None:
            failed_methods.append("bm25")

        # If BOTH failed, raise typed RetrievalError
        if dense_error is not None and sparse_error is not None:
            raise RetrievalError(
                f"Both dense and sparse retrieval failed: vector error: {dense_error}; bm25 error: {sparse_error}"
            )

        # If one failed and partial failure is not allowed, raise
        if failed_methods and not allow_partial_failure:
            err = dense_error or sparse_error
            raise RetrievalError(f"Retrieval branch failed: {err}") from err

        partial_failure = len(failed_methods) > 0

        # 5. RRF Multi-Source Fusion & Deduplication
        fusion_start = time.time()
        named_lists = {}
        if "vector" not in failed_methods and vector_candidates:
            named_lists["vector"] = vector_candidates
        if "bm25" not in failed_methods and bm25_candidates:
            named_lists["bm25"] = bm25_candidates

        # If both candidate lists are empty but didn't error
        if not named_lists:
            total_ms = (time.time() - total_start) * 1000.0
            return self._empty_response(
                request=request,
                duration_ms=total_ms,
                partial_failure=partial_failure,
                failed_methods=failed_methods,
                dense_ms=vector_search_ms,
                sparse_ms=bm25_ms,
                query_embed_ms=query_embed_ms,
            )

        fused_infos: List[FusedCandidateInfo] = self.fusion.fuse_sources(
            named_lists=named_lists,
            top_k=getattr(self.config, 'rrf_top_k', hybrid_candidate_k),
        )
        fusion_diag: FusionDiagnostics = self.fusion.diagnostics()
        fusion_ms = (time.time() - fusion_start) * 1000.0

        # Determine candidate hydration limit:
        # If reranking is active, hydrate candidate_k results to feed reranker
        reranker_active = bool(
            getattr(self.config, "reranking", None)
            and self.config.reranking.enabled
            and self.reranking_service is not None
        )
        reranker_candidate_k = (
            getattr(self.config.reranking, "candidate_k", 50)
            if reranker_active
            else final_top_k
        )
        hydration_limit = (
            max(final_top_k, reranker_candidate_k) if reranker_active else final_top_k
        )

        # 6. Hydrate results up to hydration_limit
        results: List[RetrievedChunkResult] = []
        for rank_idx, fused in enumerate(fused_infos[:hydration_limit], start=1):
            cand = fused.candidate

            try:
                ResultValidator.validate_candidate(cand, request.tenant_id)
            except RetrievalError as e:
                logger.warning("Skipping invalid hybrid candidate %s: %s", cand.chunk_id, e)
                continue

            section_path = VectorRetrievalService._extract_section_path(cand)
            dense_score = fused.source_scores.get("vector")
            sparse_score = fused.source_scores.get("bm25")
            dense_rk = fused.source_ranks.get("vector")
            sparse_rk = fused.source_ranks.get("bm25")

            sim_score = dense_score if dense_score is not None else (sparse_score or 0.0)

            result = RetrievedChunkResult(
                chunk_id=cand.chunk_id,
                document_id=cand.document_id,
                document_version=cand.document_version,
                document_version_id=f"v{cand.document_version}",
                tenant_id=request.tenant_id,
                text=cand.text,
                similarity_score=float(sim_score),
                raw_score=float(sim_score),
                rank=rank_idx,
                section_path=section_path,
                heading=cand.section,
                page=cand.page,
                page_number=cand.page,
                source_uri=cand.source_uri,
                filename=cand.filename,
                mime_type=cand.mime_type,
                chunk_type=cand.chunk_metadata.get("chunk_type", "text"),
                chunking_version=cand.chunking_version,
                content_hash=cand.content_hash,
                embedding_model=self.embedding_service.model_name,
                embedding_provider=self.embedding_service.provider_name,
                embedding_dimension=self.embedding_service.dimension,
                retrieval_method="hybrid",
                asset_id=cand.chunk_metadata.get("asset_id"),
                table_id=cand.chunk_metadata.get("table_id"),
                rrf_score=round(fused.rrf_score, 6),
                bm25_score=round(sparse_score, 4) if sparse_score is not None else None,
                dense_raw_score=dense_score,
                sparse_raw_score=sparse_score,
                dense_rank=dense_rk,
                sparse_rank=sparse_rk,
                retrieval_methods=fused.retrieval_methods,
                fusion_strategy="rrf",
                metadata=cand.chunk_metadata,
            )
            results.append(result)

        # 7. Reranking (optional post-fusion stage)
        rerank_ms: Optional[float] = None
        rerank_method: Optional[str] = None
        rerank_failed = False
        rerank_fallback_used = False
        rerank_error_msg: Optional[str] = None
        rerank_input_count = len(results)

        if reranker_active and results:
            rerank_start = time.perf_counter()
            results, rerank_resp = self.reranking_service.rerank_sync(
                query=request.query,
                candidates=results,
                top_k=final_top_k,
                tenant_id=request.tenant_id,
            )
            rerank_ms = (time.perf_counter() - rerank_start) * 1000.0
            rerank_method = rerank_resp.model
            rerank_failed = rerank_resp.reranking_failed
            rerank_fallback_used = rerank_resp.fallback_used
            rerank_error_msg = rerank_resp.error_message
            rerank_input_count = rerank_resp.input_count
        else:
            results = results[:final_top_k]

        total_ms = (time.time() - total_start) * 1000.0

        diagnostics: Dict[str, Any] = {
            "query_hash": qhash,
            "tenant_id": request.tenant_id,
            "dense_result_count": len(vector_candidates),
            "sparse_result_count": len(bm25_candidates),
            "unique_candidate_count": len(fused_infos),
            "returned_results": len(results),
            "dense_latency_ms": round(vector_search_ms, 2),
            "sparse_latency_ms": round(bm25_ms, 2),
            "query_embedding_latency_ms": round(query_embed_ms, 2),
            "fusion_latency_ms": round(fusion_ms, 2),
            "reranking_latency_ms": round(rerank_ms, 2) if rerank_ms is not None else None,
            "reranking_enabled": reranker_active,
            "reranking_model": rerank_method,
            "reranking_input_count": rerank_input_count,
            "reranking_output_count": len(results),
            "reranking_failed": rerank_failed,
            "reranking_fallback_used": rerank_fallback_used,
            "reranking_error": rerank_error_msg,
            "total_latency_ms": round(total_ms, 2),
            "partial_failure": partial_failure,
            "failed_methods": failed_methods,
            "rrf_k": self.fusion.k,
            "dense_top_k": dense_k,
            "sparse_top_k": sparse_k,
            "hybrid_candidate_k": hybrid_candidate_k,
            "filters_applied": request.filters.model_dump(exclude_none=True) if request.filters else None,
            # RRF fusion diagnostics
            "fusion_strategy": fusion_diag.fusion_strategy,
            "source_weights": fusion_diag.source_weights,
            "input_result_counts": fusion_diag.input_result_counts,
            "fusion_unique_count": fusion_diag.unique_result_count,
            "fusion_final_count": fusion_diag.final_result_count,
        }

        logger.info(
            "[%s] HybridRetrieval: '%s' (tenant=%s) → %d results (dense=%d, sparse=%d, rerank=%s, partial=%s, total=%.1fms)",
            qhash, request.query[:60], request.tenant_id, len(results),
            len(vector_candidates), len(bm25_candidates), rerank_method, partial_failure, total_ms,
        )

        return RetrievalResponse(
            query=request.query,
            tenant_id=request.tenant_id,
            results=results,
            total_candidates=max(total_vector_cands, total_bm25_cands),
            returned_results=len(results),
            retrieval_method="hybrid",
            retrieval_backend=self.vector_repository.backend_name,
            embedding_model=self.embedding_service.model_name,
            embedding_dimension=self.embedding_service.dimension,
            duration_ms=round(total_ms, 2),
            latency_ms=round(total_ms, 2),
            query_embedding_latency_ms=round(query_embed_ms, 2),
            search_latency_ms=round(vector_search_ms, 2),
            bm25_latency_ms=round(bm25_ms, 2),
            fusion_latency_ms=round(fusion_ms, 2),
            fusion_method="rrf",
            reranking_latency_ms=round(rerank_ms, 2) if rerank_ms is not None else None,
            reranking_method=rerank_method,
            filters_applied=diagnostics["filters_applied"],
            diagnostics=diagnostics,
        )

    def _empty_response(
        self,
        request: RetrievalRequest,
        duration_ms: float,
        partial_failure: bool,
        failed_methods: List[str],
        dense_ms: float,
        sparse_ms: float,
        query_embed_ms: float,
    ) -> RetrievalResponse:
        """Helper to construct a valid empty response when no candidates match."""
        return RetrievalResponse(
            query=request.query,
            tenant_id=request.tenant_id,
            results=[],
            total_candidates=0,
            returned_results=0,
            retrieval_method="hybrid",
            retrieval_backend=self.vector_repository.backend_name,
            embedding_model=self.embedding_service.model_name,
            embedding_dimension=self.embedding_service.dimension,
            duration_ms=round(duration_ms, 2),
            latency_ms=round(duration_ms, 2),
            query_embedding_latency_ms=round(query_embed_ms, 2),
            search_latency_ms=round(dense_ms, 2),
            bm25_latency_ms=round(sparse_ms, 2),
            fusion_latency_ms=0.0,
            fusion_method="rrf",
            filters_applied=request.filters.model_dump(exclude_none=True) if request.filters else None,
            diagnostics={
                "dense_result_count": 0,
                "sparse_result_count": 0,
                "unique_candidate_count": 0,
                "returned_results": 0,
                "partial_failure": partial_failure,
                "failed_methods": failed_methods,
                "total_latency_ms": round(duration_ms, 2),
            },
        )
