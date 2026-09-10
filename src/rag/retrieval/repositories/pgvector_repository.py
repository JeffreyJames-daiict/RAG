"""PostgreSQL pgvector native similarity search repository.

Implements high-performance vector similarity search using pgvector's HNSW
index and native distance operators (<=>, <->, <#>).

Requires:
  - PostgreSQL with pgvector extension enabled.
  - The ``embedding_vector`` column (type ``vector(1536)``) on the ``embeddings``
    table, populated by the migration backfill.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, literal_column, select, text
from sqlalchemy.exc import OperationalError, ProgrammingError

from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, Embedding
from rag.core.models import DocumentStatus
from rag.retrieval.exceptions import (
    BackendUnavailableError,
    DatabaseRetrievalError,
    UnsupportedMetricError,
)
from rag.retrieval.interfaces import RetrievedCandidate
from rag.retrieval.models import RetrievalRequest
from rag.retrieval.repositories.base import BaseVectorRepository

logger = logging.getLogger(__name__)

# pgvector distance operators mapped by metric name
_PG_DISTANCE_OPERATORS = {
    "cosine": "<=>",       # cosine distance (1 - cosine_similarity)
    "l2": "<->",           # Euclidean (L2) distance
    "inner_product": "<#>",  # negative inner product
}


class PgVectorRepository(BaseVectorRepository):
    """Native pgvector similarity search over PostgreSQL with HNSW indexing.

    Uses the ``embedding_vector`` column (``vector(1536)``) and pgvector distance
    operators for in-database similarity ranking with HNSW acceleration.

    Falls back gracefully if the ``embedding_vector`` column is not populated or
    the pgvector extension is unavailable.

    Usage::

        repo = PgVectorRepository(db_manager, distance_metric="cosine")
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
        distance_metric: str = "cosine",
        candidate_limit: int = 1000,
    ):
        super().__init__(db_manager)
        self._distance_metric = distance_metric.lower()
        self._candidate_limit = candidate_limit

        if self._distance_metric not in _PG_DISTANCE_OPERATORS:
            raise UnsupportedMetricError(
                f"Unsupported distance metric '{distance_metric}'. "
                f"Supported: {list(_PG_DISTANCE_OPERATORS.keys())}"
            )

    @property
    def backend_name(self) -> str:
        return "pgvector"

    def is_available(self) -> bool:
        """Check whether pgvector extension and embedding_vector column exist."""
        try:
            with self.db_manager.session() as session:
                # Check extension
                result = session.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                )
                if not result.fetchone():
                    return False

                # Check column exists
                result = session.execute(
                    text(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_schema = 'public' AND table_name = 'embeddings' "
                        "AND column_name = 'embedding_vector'"
                    )
                )
                if not result.fetchone():
                    return False

                # Check at least one non-null embedding_vector row
                result = session.execute(
                    text(
                        "SELECT 1 FROM embeddings "
                        "WHERE embedding_vector IS NOT NULL LIMIT 1"
                    )
                )
                return result.fetchone() is not None

        except Exception as e:
            logger.warning("pgvector availability check failed: %s", e)
            return False

    def search_vectors(
        self,
        query_vector: List[float],
        request: RetrievalRequest,
        model_name: str,
        model_version: str,
    ) -> Tuple[List[RetrievedCandidate], int]:
        """Execute native pgvector similarity search with HNSW acceleration.

        Args:
            query_vector:  Dense embedding vector for the user query.
            request:       Retrieval request with top_k, filters, tenant_id.
            model_name:    Embedding model name to match stored embeddings.
            model_version: Embedding model version to match stored embeddings.

        Returns:
            Tuple of (sorted candidates descending by similarity, total evaluated).

        Raises:
            DatabaseRetrievalError: On SQL execution failure.
        """
        start_time = time.time()
        operator = _PG_DISTANCE_OPERATORS[self._distance_metric]

        # Format query vector as pgvector literal: '[0.1,0.2,...]'
        vec_literal = "[" + ",".join(str(v) for v in query_vector) + "]"

        try:
            with self.db_manager.session() as session:
                # Set HNSW search ef_search for quality control
                session.execute(text("SET LOCAL hnsw.ef_search = 200"))

                # Build the native pgvector query
                distance_expr = text(
                    f"embeddings.embedding_vector {operator} :query_vec"
                ).bindparams(query_vec=vec_literal)

                # Core query: join Chunk, Document, Embedding with tenant isolation
                stmt = (
                    select(
                        Chunk.chunk_id,
                        Chunk.document_id,
                        Chunk.version,
                        Chunk.chunk_index,
                        Chunk.text,
                        Chunk.section,
                        Chunk.page,
                        Chunk.chunk_metadata,
                        Chunk.chunking_version,
                        Embedding.embedding_model,
                        Embedding.embedding_version,
                        Embedding.dimension,
                        Embedding.provider,
                        Document.source_uri,
                        Document.filename,
                        Document.mime_type,
                        Document.content_hash,
                        Document.tenant_id,
                        distance_expr.label("distance"),
                    )
                    .select_from(Chunk)
                    .join(Document, Chunk.document_id == Document.document_id)
                    .join(
                        Embedding,
                        and_(
                            Chunk.chunk_id == Embedding.chunk_id,
                            Embedding.embedding_model == model_name,
                            Embedding.embedding_version == model_version,
                        ),
                    )
                    .where(
                        Document.tenant_id == request.tenant_id,
                        Embedding.embedding_vector.isnot(None),
                    )
                )

                # Active document / version filtering
                if not request.include_inactive:
                    stmt = stmt.where(
                        Document.deleted_at.is_(None),
                        Document.status == DocumentStatus.INDEXED.value,
                        Chunk.version == Document.version,
                    )

                # SQL-level metadata filters
                if request.filters:
                    clauses = request.filters.build_sql_clauses()
                    if clauses:
                        stmt = stmt.where(and_(*clauses))

                # Similarity threshold cutoff (convert to distance threshold)
                if request.similarity_threshold is not None:
                    if self._distance_metric == "cosine":
                        # cosine_distance = 1 - cosine_similarity
                        max_distance = 1.0 - request.similarity_threshold
                        stmt = stmt.where(
                            text(f"embeddings.embedding_vector {operator} :q_vec").bindparams(
                                q_vec=vec_literal
                            )
                            <= max_distance
                        )

                # Deterministic ordering: distance ASC, chunk_index ASC, chunk_id ASC
                stmt = stmt.order_by(
                    text("distance ASC"),
                    Chunk.chunk_index.asc(),
                    Chunk.chunk_id.asc(),
                )

                # Limit to top_k
                stmt = stmt.limit(request.top_k)

                rows = session.execute(stmt).all()

        except (OperationalError, ProgrammingError) as e:
            raise DatabaseRetrievalError(
                f"pgvector search query failed: {e}"
            ) from e

        total_candidates = len(rows)
        if total_candidates == 0:
            logger.info(
                "PgVectorRepository: 0 candidates found for tenant='%s', model='%s/%s'.",
                request.tenant_id, model_name, model_version,
            )
            return [], 0

        # Build RetrievedCandidate objects
        results: List[RetrievedCandidate] = []
        for row in rows:
            distance = float(row.distance)

            # Convert distance to similarity based on metric
            if self._distance_metric == "cosine":
                similarity = 1.0 - distance
            elif self._distance_metric == "l2":
                similarity = 1.0 / (1.0 + distance)
            elif self._distance_metric == "inner_product":
                similarity = -distance  # pgvector <#> returns negative IP
            else:
                similarity = 1.0 - distance

            # Normalize score to [0, 1]
            normalized = max(0.0, min(1.0, (similarity + 1.0) / 2.0))

            chunk_meta = row.chunk_metadata or {}

            results.append(
                RetrievedCandidate(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    document_version=row.version,
                    chunk_index=row.chunk_index,
                    text=row.text,
                    similarity_score=round(similarity, 6),
                    distance=round(distance, 6),
                    tenant_id=row.tenant_id,
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
            "PgVectorRepository: %d candidates returned (top_k=%d) in %.2f ms. "
            "Metric=%s, operator='%s', tenant='%s', model='%s/%s'.",
            len(results), request.top_k, elapsed_ms,
            self._distance_metric, operator,
            request.tenant_id, model_name, model_version,
        )

        return results, total_candidates
