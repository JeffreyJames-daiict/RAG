"""Vector Retrieval Service — high-level orchestrator for semantic search.

Wires together: query embedding → vector similarity search → validation → response hydration.

Implements both synchronous and asynchronous retrieval conforming to the
``VectorRetriever`` protocol, with comprehensive diagnostics and observability.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

from rag.core.config import RetrievalConfig, get_settings
from rag.core.database import DatabaseManager
from rag.embeddings.service import EmbeddingService
from rag.retrieval.exceptions import (
    IncompatibleEmbeddingError,
    MissingEmbeddingProviderError,
    RetrievalError,
    RetrievalTimeoutError,
)
from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.interfaces import RetrievedCandidate
from rag.retrieval.models import RetrievalRequest, RetrievalResponse, RetrievedChunkResult
from rag.retrieval.repositories import BaseVectorRepository, get_vector_repository
from rag.retrieval.validators import (
    EmbeddingCompatibilityValidator,
    QueryValidator,
    ResultValidator,
)

logger = logging.getLogger(__name__)


def _query_hash(query: str) -> str:
    """Compute a safe, non-reversible hash ID for log correlation without leaking query text."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]


class VectorRetrievalService:
    """Production service for vector-based semantic retrieval.

    Accepts a user query, generates the query embedding via the configured
    embedding provider, searches persisted chunk embeddings using vector
    similarity, validates and hydrates results with full provenance metadata.

    Implements the ``VectorRetriever`` protocol with both async and sync APIs.

    Usage::

        from rag.core.database import DatabaseManager
        from rag.embeddings.service import EmbeddingService
        from rag.retrieval.vector_service import VectorRetrievalService
        from rag.retrieval.models import RetrievalRequest

        db_mgr = DatabaseManager(database_url)
        emb_svc = EmbeddingService(db_mgr)
        retriever = VectorRetrievalService(db_mgr, emb_svc)

        request = RetrievalRequest(query="What is the patent status?", tenant_id="acme")
        response = retriever.retrieve(request)
        for r in response.results:
            print(f"[{r.rank}] {r.similarity_score:.4f} — {r.text[:80]}")
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_service: EmbeddingService,
        repository: Optional[BaseVectorRepository] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self.db_manager = db_manager
        self.embedding_service = embedding_service
        self.config = config or get_settings().retrieval
        self.repository = repository or get_vector_repository(db_manager, self.config)

    @property
    def model_name(self) -> str:
        return self.embedding_service.model_name

    @property
    def model_version(self) -> str:
        return self.embedding_service.model_version

    @property
    def dimension(self) -> int:
        return self.embedding_service.dimension

    # ── Primary Request-Based API ─────────────────────────────────────────

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Execute full vector semantic retrieval pipeline.

        Steps:
            1. Validate the retrieval request (query, tenant, top_k).
            2. Generate query embedding via the embedding provider.
            3. Validate query vector (dimension, finiteness, non-zero).
            4. Search persisted embeddings via the vector repository.
            5. Validate each candidate result.
            6. Hydrate and return the response with full provenance.

        Args:
            request: Validated retrieval request.

        Returns:
            RetrievalResponse with ranked, hydrated results.

        Raises:
            RetrievalError: On invalid request or retrieval failure.
            IncompatibleEmbeddingError: On dimension/model mismatch.
        """
        total_start = time.time()
        qhash = _query_hash(request.query)

        # 1. Validate request
        QueryValidator.validate_request(request, max_top_k=self.config.max_top_k)

        # 2. Generate query embedding
        embed_start = time.time()
        if request.query_embedding_override is not None:
            query_vector = request.query_embedding_override
            logger.info("[%s] Using provided query embedding override (len=%d).", qhash, len(query_vector))
        else:
            try:
                query_vector = self.embedding_service.embed_query(request.query)
            except Exception as e:
                if "provider" in str(e).lower() or "not configured" in str(e).lower():
                    raise MissingEmbeddingProviderError(
                        f"Embedding provider unavailable: {e}"
                    ) from e
                raise RetrievalError(f"Failed to generate query embedding: {e}") from e

        query_embed_ms = (time.time() - embed_start) * 1000.0

        # 3. Validate query vector
        if self.config.strict_embedding_validation:
            EmbeddingCompatibilityValidator.validate_query_vector(
                vector=query_vector,
                expected_dimension=self.dimension,
                expected_model=self.model_name,
            )

        # 4. Search via repository
        search_start = time.time()
        try:
            candidates, total_candidates = self.repository.search_vectors(
                query_vector=query_vector,
                request=request,
                model_name=self.model_name,
                model_version=self.model_version,
            )
        except Exception as e:
            raise RetrievalError(f"Vector search failed: {e}") from e

        search_ms = (time.time() - search_start) * 1000.0

        # 5. Validate and hydrate results
        hydrate_start = time.time()
        results: List[RetrievedChunkResult] = []
        for rank_idx, candidate in enumerate(candidates, start=1):
            try:
                ResultValidator.validate_candidate(candidate, request.tenant_id)
            except RetrievalError as e:
                logger.warning("[%s] Skipping invalid candidate %s: %s", qhash, candidate.chunk_id, e)
                continue

            section_path = self._extract_section_path(candidate)

            results.append(
                RetrievedChunkResult(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    document_version=candidate.document_version,
                    chunk_index=candidate.chunk_index,
                    text=candidate.text,
                    similarity_score=candidate.similarity_score,
                    raw_score=candidate.raw_score,
                    normalized_score=candidate.normalized_score,
                    rank=rank_idx,
                    tenant_id=candidate.tenant_id,
                    title=candidate.title or candidate.filename,
                    section_path=section_path,
                    heading=candidate.section,
                    page=candidate.page,
                    page_number=candidate.page,
                    parent_chunk_id=candidate.parent_chunk_id,
                    source_uri=candidate.source_uri,
                    filename=candidate.filename,
                    mime_type=candidate.mime_type,
                    chunk_type=candidate.chunk_metadata.get("chunk_type", "text"),
                    chunking_version=candidate.chunking_version,
                    content_hash=candidate.content_hash,
                    embedding_model=candidate.embedding_model,
                    embedding_provider=candidate.embedding_provider,
                    embedding_dimension=candidate.embedding_dimension,
                    retrieval_method="vector",
                    asset_id=candidate.chunk_metadata.get("asset_id"),
                    table_id=candidate.chunk_metadata.get("table_id"),
                    metadata=candidate.chunk_metadata,
                )
            )

        hydrate_ms = (time.time() - hydrate_start) * 1000.0
        total_ms = (time.time() - total_start) * 1000.0

        # Build diagnostics dict
        diagnostics = {
            "query_hash": qhash,
            "backend": self.repository.backend_name,
            "distance_metric": self.config.distance_metric,
            "strict_validation": self.config.strict_embedding_validation,
            "query_embed_latency_ms": round(query_embed_ms, 2),
            "search_latency_ms": round(search_ms, 2),
            "hydration_latency_ms": round(hydrate_ms, 2),
            "total_latency_ms": round(total_ms, 2),
            "total_candidates_evaluated": total_candidates,
            "results_returned": len(results),
            "top_k_requested": request.top_k,
            "similarity_threshold": request.similarity_threshold,
            "embedding_model": self.model_name,
            "embedding_dimension": self.dimension,
            "index_type": self.config.index_type,
        }

        # Build filter tracking
        filters_applied = None
        if request.filters:
            filters_applied = request.filters.model_dump() if hasattr(request.filters, "model_dump") else {}

        response = RetrievalResponse(
            query=request.query,
            tenant_id=request.tenant_id,
            results=results,
            total_candidates=total_candidates,
            returned_results=len(results),
            retrieval_backend=self.repository.backend_name,
            embedding_model=self.model_name,
            embedding_dimension=self.dimension,
            embedding_dimensions=self.dimension,
            index_version=self.config.index_type,
            duration_ms=round(total_ms, 2),
            latency_ms=round(total_ms, 2),
            query_embedding_latency_ms=round(query_embed_ms, 2),
            search_latency_ms=round(search_ms, 2),
            filters_applied=filters_applied,
            diagnostics=diagnostics,
        )

        logger.info(
            "[%s] VectorRetrieval: query='%s' tenant='%s' → %d results from %d candidates "
            "(embed=%.1fms, search=%.1fms, hydrate=%.1fms, total=%.1fms) backend=%s",
            qhash, request.query[:60], request.tenant_id, len(results),
            total_candidates, query_embed_ms, search_ms, hydrate_ms, total_ms,
            self.repository.backend_name,
        )

        return response

    # ── VectorRetriever Protocol: Sync API ────────────────────────────────

    def retrieve_sync(
        self,
        query: str,
        *,
        tenant_id: str = "default",
        top_k: int = 5,
        filters: Optional[RetrievalFilters] = None,
    ) -> RetrievalResponse:
        """Synchronously execute vector similarity search.

        Convenience wrapper that constructs a RetrievalRequest and delegates to
        the primary ``retrieve()`` method.
        """
        request = RetrievalRequest(
            query=query,
            tenant_id=tenant_id,
            top_k=top_k,
            filters=filters,
            max_query_length=self.config.max_query_length,
        )
        return self.retrieve(request)

    # ── VectorRetriever Protocol: Async API ───────────────────────────────

    async def aretrieve(
        self,
        query: str,
        *,
        tenant_id: str = "default",
        top_k: int = 5,
        filters: Optional[RetrievalFilters] = None,
    ) -> RetrievalResponse:
        """Asynchronously execute vector similarity search.

        Runs the synchronous retrieval pipeline in a thread pool executor
        to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.retrieve_sync(
                query, tenant_id=tenant_id, top_k=top_k, filters=filters
            ),
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_section_path(candidate: RetrievedCandidate) -> List[str]:
        """Extract hierarchical section breadcrumb from candidate metadata."""
        # Check for explicit section_path in metadata first
        meta = candidate.chunk_metadata or {}
        if "section_path" in meta and isinstance(meta["section_path"], list):
            return meta["section_path"]

        # Fall back to splitting the section string on common delimiters
        if candidate.section:
            for delim in [" > ", " → ", " / ", " | "]:
                if delim in candidate.section:
                    return [s.strip() for s in candidate.section.split(delim) if s.strip()]
            return [candidate.section]

        return []
