"""Abstract base repository for vector retrieval operations."""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import Select, and_, select

from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, Embedding
from rag.core.models import DocumentStatus
from rag.retrieval.interfaces import RetrievedCandidate
from rag.retrieval.models import RetrievalRequest


class BaseVectorRepository(abc.ABC):
    """Base class for vector repositories with shared query building and candidate mapping."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        """Name of the repository backend."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if backend is supported and available in the current database."""
        raise NotImplementedError

    def build_candidate_select_stmt(
        self,
        request: RetrievalRequest,
        model_name: str,
        model_version: str,
    ) -> Select:
        """Construct base SELECT statement joining Chunk, Document, and Embedding with tenant filtering."""
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
                Embedding.vector,
                Document.source_uri,
                Document.filename,
                Document.mime_type,
                Document.content_hash,
            )
            .join(Document, Chunk.document_id == Document.document_id)
            .join(
                Embedding,
                and_(
                    Chunk.chunk_id == Embedding.chunk_id,
                    Embedding.embedding_model == model_name,
                    Embedding.embedding_version == model_version,
                ),
            )
            .where(Document.tenant_id == request.tenant_id)
        )

        # Inactive / deleted document filtering
        if not request.include_inactive:
            stmt = stmt.where(
                Document.deleted_at.is_(None),
                Document.status == DocumentStatus.INDEXED.value,
                Chunk.version == Document.version,
            )

        # Apply metadata filters
        if request.filters:
            clauses = request.filters.build_sql_clauses()
            if clauses:
                stmt = stmt.where(and_(*clauses))

        return stmt

    @abc.abstractmethod
    def search_vectors(
        self,
        query_vector: List[float],
        request: RetrievalRequest,
        model_name: str,
        model_version: str,
    ) -> Tuple[List[RetrievedCandidate], int]:
        """Execute similarity search and return sorted candidates."""
        raise NotImplementedError
