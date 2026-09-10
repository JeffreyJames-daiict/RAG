"""Python in-memory vector similarity search repository.

Implements cosine similarity search over persisted embeddings without
external vector database dependencies (no pgvector required). Suitable
for development, testing, and small-to-medium scale deployments.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, select

from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, Embedding
from rag.core.models import DocumentStatus
from rag.retrieval.interfaces import RetrievedCandidate
from rag.retrieval.models import RetrievalRequest
from rag.retrieval.repositories.base import BaseVectorRepository

logger = logging.getLogger(__name__)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a float in [-1.0, 1.0]. Returns 0.0 if either vector is zero-length.
    """
    if len(vec_a) != len(vec_b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0.0:
        return 0.0

    sim = dot / denom
    # Clamp to [-1, 1] to handle floating point drift
    return max(-1.0, min(1.0, sim))


def _l2_distance(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute Euclidean (L2) distance between two vectors."""
    if len(vec_a) != len(vec_b):
        return float("inf")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))


def _inner_product(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute inner product (dot product) between two vectors."""
    if len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


class PythonVectorRepository(BaseVectorRepository):
    """In-memory cosine/L2/inner-product similarity search over persisted embeddings.

    Loads candidate embeddings from PostgreSQL/SQLite, computes vector
    similarity in pure Python, and returns ranked candidates.

    Usage::

        repo = PythonVectorRepository(db_manager)
        candidates, total = repo.search_vectors(
            query_vector=[0.1, 0.2, ...],
            request=retrieval_request,
            model_name="text-embedding-3-small",
            model_version="v1.0",
        )
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        candidate_limit: int = 1000,
        distance_metric: str = "cosine",
    ):
        super().__init__(db_manager)
        self._candidate_limit = candidate_limit
        self._distance_metric = distance_metric.lower()

    @property
    def backend_name(self) -> str:
        return "python"

    def is_available(self) -> bool:
        """Python backend is always available."""
        return True

    def search_vectors(
        self,
        query_vector: List[float],
        request: RetrievalRequest,
        model_name: str,
        model_version: str,
    ) -> Tuple[List[RetrievedCandidate], int]:
        """Execute in-memory vector similarity search.

        Args:
            query_vector:  Dense embedding vector for the user query.
            request:       Retrieval request with top_k, filters, tenant_id, similarity_threshold.
            model_name:    Embedding model name to match stored embeddings.
            model_version: Embedding model version to match stored embeddings.

        Returns:
            Tuple of (sorted candidates descending by similarity, total candidates evaluated).
        """
        start_time = time.time()

        stmt = self.build_candidate_select_stmt(request, model_name, model_version)
        stmt = stmt.limit(self._candidate_limit)

        with self.db_manager.session() as session:
            rows = session.execute(stmt).all()

        total_candidates = len(rows)
        if total_candidates == 0:
            logger.info(
                "PythonVectorRepository: 0 candidates found for tenant='%s', model='%s/%s'.",
                request.tenant_id, model_name, model_version,
            )
            return [], 0

        # Score and rank candidates
        scored_candidates: List[Tuple[float, float, Any]] = []

        for row in rows:
            stored_vector = row.vector
            if not stored_vector or not isinstance(stored_vector, (list, tuple)):
                logger.debug("Skipping chunk %s: invalid stored vector.", row.chunk_id)
                continue

            # Dimension check
            if len(stored_vector) != len(query_vector):
                logger.debug(
                    "Skipping chunk %s: dimension mismatch (query=%d, stored=%d).",
                    row.chunk_id, len(query_vector), len(stored_vector),
                )
                continue

            # Check for degenerate vectors (all zeros)
            if all(abs(v) < 1e-12 for v in stored_vector):
                logger.debug("Skipping chunk %s: zero vector.", row.chunk_id)
                continue

            # Compute similarity based on configured metric
            if self._distance_metric == "cosine":
                similarity = _cosine_similarity(query_vector, stored_vector)
                distance = 1.0 - similarity
            elif self._distance_metric == "l2":
                distance = _l2_distance(query_vector, stored_vector)
                # Convert L2 distance to a similarity-like score (higher = better)
                similarity = 1.0 / (1.0 + distance)
            elif self._distance_metric == "inner_product":
                similarity = _inner_product(query_vector, stored_vector)
                distance = -similarity  # negative because higher IP = lower distance
            else:
                similarity = _cosine_similarity(query_vector, stored_vector)
                distance = 1.0 - similarity

            # Apply similarity threshold filter
            threshold = request.similarity_threshold
            if threshold is not None and similarity < threshold:
                continue

            # Apply in-memory metadata filters (chunk_types, table_id, language, json match)
            if request.filters:
                chunk_meta = row.chunk_metadata or {}
                if not request.filters.matches_chunk_metadata(chunk_meta):
                    continue

            scored_candidates.append((similarity, distance, row))

        # Deterministic tie-breaking: similarity DESC, chunk_index ASC, chunk_id ASC
        scored_candidates.sort(
            key=lambda x: (-x[0], x[2].chunk_index, x[2].chunk_id)
        )

        # Truncate to top_k
        top_k = request.top_k
        top_candidates = scored_candidates[:top_k]

        # Build RetrievedCandidate objects
        results: List[RetrievedCandidate] = []
        for similarity, distance, row in top_candidates:
            chunk_meta = row.chunk_metadata or {}

            # Normalize score to [0, 1]
            normalized = max(0.0, min(1.0, (similarity + 1.0) / 2.0))

            results.append(
                RetrievedCandidate(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    document_version=row.version,
                    chunk_index=row.chunk_index,
                    text=row.text,
                    similarity_score=round(similarity, 6),
                    distance=round(distance, 6),
                    raw_score=round(similarity, 6),
                    normalized_score=round(normalized, 6),
                    section=row.section,
                    page=row.page,
                    chunk_metadata=chunk_meta,
                    chunking_version=row.chunking_version or "v1.0",
                    embedding_model=row.embedding_model or "",
                    embedding_provider=row.provider or "",
                    embedding_dimension=row.dimension or 0,
                    source_uri=row.source_uri or "",
                    filename=row.filename or "",
                    mime_type=row.mime_type or "",
                    content_hash=row.content_hash or "",
                )
            )

        elapsed_ms = (time.time() - start_time) * 1000.0
        logger.info(
            "PythonVectorRepository: %d candidates evaluated, %d returned (top_k=%d) in %.2f ms. "
            "Metric=%s, tenant='%s', model='%s/%s'.",
            total_candidates, len(results), top_k, elapsed_ms,
            self._distance_metric, request.tenant_id, model_name, model_version,
        )

        return results, total_candidates

