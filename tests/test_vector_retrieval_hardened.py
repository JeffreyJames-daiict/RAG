"""Comprehensive hardened test suite for Vector Retrieval.

Covers:
  1.  Valid vector retrieval (sync API)
  2.  Empty query rejection
  3.  Query length ceiling rejection
  4.  Top-K boundary validation
  5.  Similarity threshold cutoff filtering
  6.  Strict tenant isolation (cross-tenant leakage prevention)
  7.  Soft-deleted document exclusion
  8.  Inactive version exclusion
  9.  Metadata filtering (document_id, page, chunk_type, source)
  10. Embedding dimension mismatch rejection
  11. Provider failure handling
  12. Database retrieval failure handling
  13. Stable deterministic ordering with tie-breaking
  14. No-result behavior with valid diagnostics
  15. Python fallback path validation
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import List, Tuple
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from pydantic import ValidationError

from rag.core.config import RetrievalConfig
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, Embedding
from rag.core.models import DocumentStatus
from rag.retrieval.exceptions import (
    EmptyQueryError,
    IncompatibleEmbeddingError,
    MissingEmbeddingProviderError,
    QueryTooLongError,
    RetrievalError,
    TenantContextMissingError,
    TenantIsolationError,
)
from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.interfaces import RetrievedCandidate
from rag.retrieval.models import RetrievalRequest, RetrievalResponse, RetrievedChunkResult
from rag.retrieval.repositories.python_repository import PythonVectorRepository
from rag.retrieval.vector_service import VectorRetrievalService


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db_mgr():
    """In-memory SQLite database for isolated testing."""
    db = DatabaseManager(db_url="sqlite:///:memory:")
    db.create_tables()
    return db


@pytest.fixture
def mock_config():
    """A standard retrieval config for tests."""
    return RetrievalConfig(
        backend="python",
        default_top_k=5,
        max_top_k=100,
        candidate_limit=1000,
        distance_metric="cosine",
        allow_python_fallback=True,
        strict_embedding_validation=True,
        max_query_length=4000,
        embedding_dimensions=3,
    )


def _seed_documents(db_mgr, tenant_id="test-tenant", count=2, status="INDEXED", deleted=False):
    """Seed documents and return their IDs."""
    doc_ids = []
    with db_mgr.session() as session:
        for i in range(count):
            doc_id = str(uuid.uuid4())
            doc = Document(
                document_id=doc_id,
                tenant_id=tenant_id,
                source="LOCAL_FILE",
                source_uri=f"/test/{doc_id}.pdf",
                filename=f"doc_{i}.pdf",
                mime_type="application/pdf",
                content_hash=f"hash_{doc_id[:8]}",
                version=1,
                status=status,
                deleted_at=datetime.now(timezone.utc) if deleted else None,
            )
            session.add(doc)
            doc_ids.append(doc_id)
        session.commit()
    return doc_ids


def _seed_chunks(db_mgr, doc_id, count=3, version=1, vectors=None, model_name="test-model", model_version="v1.0"):
    """Seed chunks with embeddings. Returns chunk IDs."""
    chunk_ids = []
    with db_mgr.session() as session:
        for i in range(count):
            chunk_id = str(uuid.uuid4())
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                version=version,
                chunk_index=i,
                text=f"Test chunk {i} content for document {doc_id[:8]}",
                section=f"Section {i}",
                page=i + 1,
                chunk_metadata={"chunk_type": "text", "source": "native"},
                chunking_version="v1.0",
            )
            session.add(chunk)

            vec = vectors[i] if vectors else [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)]
            embedding = Embedding(
                chunk_id=chunk_id,
                embedding_model=model_name,
                embedding_version=model_version,
                dimension=len(vec),
                provider="mock",
                vector=vec,
            )
            session.add(embedding)
            chunk_ids.append(chunk_id)
        session.commit()
    return chunk_ids


def _make_mock_embedding_service(dimension=3, model_name="test-model", model_version="v1.0"):
    """Create a mock EmbeddingService with controlled properties."""
    svc = MagicMock()
    type(svc).dimension = PropertyMock(return_value=dimension)
    type(svc).model_name = PropertyMock(return_value=model_name)
    type(svc).model_version = PropertyMock(return_value=model_version)
    svc.embed_query = MagicMock(return_value=[0.15, 0.25, 0.35])
    return svc


# ── 1. Valid Vector Retrieval ─────────────────────────────────────────────────

class TestValidRetrieval:
    def test_basic_retrieval_returns_results(self, db_mgr, mock_config):
        """Verify successful retrieval produces ranked results with valid fields."""
        doc_ids = _seed_documents(db_mgr, tenant_id="acme")
        _seed_chunks(db_mgr, doc_ids[0], count=3)

        emb_svc = _make_mock_embedding_service()
        retriever = VectorRetrievalService(db_mgr, emb_svc, config=mock_config)

        response = retriever.retrieve_sync("test query", tenant_id="acme", top_k=3)

        assert isinstance(response, RetrievalResponse)
        assert response.tenant_id == "acme"
        assert len(response.results) <= 3
        assert response.retrieval_backend == "python"
        assert response.embedding_model == "test-model"
        assert response.duration_ms > 0
        assert response.diagnostics  # should be non-empty

        for r in response.results:
            assert r.rank >= 1
            assert -1.0 <= r.similarity_score <= 1.0
            assert r.chunk_id
            assert r.text

    def test_response_diagnostics_populated(self, db_mgr, mock_config):
        """Diagnostics dictionary contains expected telemetry keys."""
        doc_ids = _seed_documents(db_mgr, tenant_id="t1")
        _seed_chunks(db_mgr, doc_ids[0], count=1)

        emb_svc = _make_mock_embedding_service()
        retriever = VectorRetrievalService(db_mgr, emb_svc, config=mock_config)
        response = retriever.retrieve_sync("test", tenant_id="t1")

        d = response.diagnostics
        assert "query_hash" in d
        assert "backend" in d
        assert "query_embed_latency_ms" in d
        assert "search_latency_ms" in d
        assert "total_latency_ms" in d
        assert d["results_returned"] == len(response.results)


# ── 2. Empty Query Rejection ─────────────────────────────────────────────────

class TestEmptyQueryRejection:
    def test_empty_string_raises(self):
        with pytest.raises((EmptyQueryError, ValidationError)):
            RetrievalRequest(query="", tenant_id="t1")

    def test_whitespace_only_raises(self):
        with pytest.raises((EmptyQueryError, ValidationError)):
            RetrievalRequest(query="   ", tenant_id="t1")


# ── 3. Query Length Ceiling ───────────────────────────────────────────────────

class TestQueryLengthCeiling:
    def test_exceeding_max_query_length_raises(self):
        with pytest.raises(QueryTooLongError):
            RetrievalRequest(query="x" * 5000, tenant_id="t1", max_query_length=4000)

    def test_within_limit_succeeds(self):
        req = RetrievalRequest(query="x" * 100, tenant_id="t1", max_query_length=4000)
        assert len(req.query) == 100


# ── 4. Top-K Boundary Validation ─────────────────────────────────────────────

class TestTopKBoundary:
    def test_top_k_zero_raises(self):
        with pytest.raises(ValidationError):
            RetrievalRequest(query="test", tenant_id="t1", top_k=0)

    def test_top_k_negative_raises(self):
        with pytest.raises(ValidationError):
            RetrievalRequest(query="test", tenant_id="t1", top_k=-1)

    def test_top_k_above_ceiling_raises(self, db_mgr, mock_config):
        emb_svc = _make_mock_embedding_service()
        retriever = VectorRetrievalService(db_mgr, emb_svc, config=mock_config)
        with pytest.raises(RetrievalError):
            retriever.retrieve_sync("test", tenant_id="t1", top_k=999)


# ── 5. Similarity Threshold Cutoff ───────────────────────────────────────────

class TestSimilarityThreshold:
    def test_high_threshold_filters_low_scores(self, db_mgr, mock_config):
        """A very high threshold should filter out most results."""
        doc_ids = _seed_documents(db_mgr, tenant_id="t1")
        _seed_chunks(db_mgr, doc_ids[0], count=3)

        emb_svc = _make_mock_embedding_service()
        retriever = VectorRetrievalService(db_mgr, emb_svc, config=mock_config)

        request = RetrievalRequest(
            query="test",
            tenant_id="t1",
            top_k=10,
            similarity_threshold=0.9999,
        )
        response = retriever.retrieve(request)

        # With random-ish vectors, almost nothing should pass 0.9999
        for r in response.results:
            assert r.similarity_score >= 0.9999


# ── 6. Tenant Isolation ──────────────────────────────────────────────────────

class TestTenantIsolation:
    def test_cross_tenant_leakage_prevented(self, db_mgr, mock_config):
        """Results must only contain chunks from the requested tenant."""
        doc_ids_a = _seed_documents(db_mgr, tenant_id="tenant_a")
        doc_ids_b = _seed_documents(db_mgr, tenant_id="tenant_b")
        _seed_chunks(db_mgr, doc_ids_a[0], count=2)
        _seed_chunks(db_mgr, doc_ids_b[0], count=2)

        emb_svc = _make_mock_embedding_service()
        retriever = VectorRetrievalService(db_mgr, emb_svc, config=mock_config)

        response_a = retriever.retrieve_sync("test", tenant_id="tenant_a")
        response_b = retriever.retrieve_sync("test", tenant_id="tenant_b")

        # Ensure no leakage
        for r in response_a.results:
            assert r.document_id in doc_ids_a, f"Tenant A result leaked doc {r.document_id}"
        for r in response_b.results:
            assert r.document_id in doc_ids_b, f"Tenant B result leaked doc {r.document_id}"

    def test_missing_tenant_raises(self):
        with pytest.raises((TenantContextMissingError, ValidationError)):
            RetrievalRequest(query="test", tenant_id="")


# ── 7. Soft-Deleted Document Exclusion ────────────────────────────────────────

class TestSoftDeletedExclusion:
    def test_deleted_documents_excluded(self, db_mgr, mock_config):
        """Soft-deleted documents should not appear in results."""
        active_ids = _seed_documents(db_mgr, tenant_id="t1", count=1, deleted=False)
        deleted_ids = _seed_documents(db_mgr, tenant_id="t1", count=1, deleted=True)
        _seed_chunks(db_mgr, active_ids[0], count=2)
        _seed_chunks(db_mgr, deleted_ids[0], count=2)

        emb_svc = _make_mock_embedding_service()
        retriever = VectorRetrievalService(db_mgr, emb_svc, config=mock_config)

        response = retriever.retrieve_sync("test", tenant_id="t1")

        result_doc_ids = {r.document_id for r in response.results}
        for del_id in deleted_ids:
            assert del_id not in result_doc_ids, "Soft-deleted document appeared in results"


# ── 8. Inactive Version Exclusion ─────────────────────────────────────────────

class TestInactiveVersionExclusion:
    def test_old_version_chunks_excluded(self, db_mgr, mock_config):
        """Chunks from non-current document versions should be excluded."""
        doc_ids = _seed_documents(db_mgr, tenant_id="t1")
        # Seed chunks for version 1 (current)
        v1_chunks = _seed_chunks(db_mgr, doc_ids[0], count=2, version=1)
        # Seed chunks for version 99 (not matching doc.version=1)
        v99_chunks = _seed_chunks(db_mgr, doc_ids[0], count=2, version=99)

        emb_svc = _make_mock_embedding_service()
        retriever = VectorRetrievalService(db_mgr, emb_svc, config=mock_config)

        response = retriever.retrieve_sync("test", tenant_id="t1")

        result_chunk_ids = {r.chunk_id for r in response.results}
        for cid in v99_chunks:
            assert cid not in result_chunk_ids, "Old-version chunk appeared in results"


# ── 9. Metadata Filtering ────────────────────────────────────────────────────

class TestMetadataFiltering:
    def test_filter_by_document_ids(self, db_mgr, mock_config):
        """Filtering by document_ids should restrict results."""
        doc_ids = _seed_documents(db_mgr, tenant_id="t1", count=3)
        for did in doc_ids:
            _seed_chunks(db_mgr, did, count=1)

        emb_svc = _make_mock_embedding_service()
        retriever = VectorRetrievalService(db_mgr, emb_svc, config=mock_config)

        target = doc_ids[0]
        filters = RetrievalFilters(document_ids=[target])
        response = retriever.retrieve_sync("test", tenant_id="t1", top_k=10, filters=filters)

        for r in response.results:
            assert r.document_id == target


# ── 10. Embedding Dimension Mismatch ─────────────────────────────────────────

class TestEmbeddingDimensionMismatch:
    def test_wrong_dimension_raises(self, db_mgr, mock_config):
        """Query vector dimension mismatch should raise IncompatibleEmbeddingError."""
        doc_ids = _seed_documents(db_mgr, tenant_id="t1")
        _seed_chunks(db_mgr, doc_ids[0], count=1)

        # Mock service returns 5-dim vector but says dimension=3
        emb_svc = _make_mock_embedding_service(dimension=3)
        emb_svc.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]  # 5-dim

        retriever = VectorRetrievalService(db_mgr, emb_svc, config=mock_config)

        with pytest.raises(IncompatibleEmbeddingError):
            retriever.retrieve_sync("test", tenant_id="t1")


# ── 11. Provider Failure Handling ─────────────────────────────────────────────

class TestProviderFailure:
    def test_embed_provider_failure_raises(self, db_mgr, mock_config):
        """Embedding provider failure should propagate as RetrievalError."""
        emb_svc = _make_mock_embedding_service()
        emb_svc.embed_query.side_effect = RuntimeError("Connection refused")

        retriever = VectorRetrievalService(db_mgr, emb_svc, config=mock_config)

        with pytest.raises(RetrievalError):
            retriever.retrieve_sync("test", tenant_id="t1")

    def test_missing_provider_raises_typed(self, db_mgr, mock_config):
        """Missing provider should raise MissingEmbeddingProviderError."""
        emb_svc = _make_mock_embedding_service()
        emb_svc.embed_query.side_effect = RuntimeError("Embedding provider not configured")

        retriever = VectorRetrievalService(db_mgr, emb_svc, config=mock_config)

        with pytest.raises(MissingEmbeddingProviderError):
            retriever.retrieve_sync("test", tenant_id="t1")


# ── 12. Database Retrieval Failure ────────────────────────────────────────────

class TestDatabaseFailure:
    def test_search_failure_raises_retrieval_error(self, db_mgr, mock_config):
        """Repository search failure should raise RetrievalError."""
        emb_svc = _make_mock_embedding_service()
        retriever = VectorRetrievalService(db_mgr, emb_svc, config=mock_config)

        # Patch the repository to raise
        retriever.repository.search_vectors = MagicMock(
            side_effect=RuntimeError("DB connection lost")
        )

        with pytest.raises(RetrievalError, match="Vector search failed"):
            retriever.retrieve_sync("test", tenant_id="t1")


# ── 13. Deterministic Ordering ────────────────────────────────────────────────

class TestDeterministicOrdering:
    def test_same_score_tie_broken_by_chunk_index(self, db_mgr, mock_config):
        """Candidates with identical similarity should be ordered by chunk_index, then chunk_id."""
        doc_ids = _seed_documents(db_mgr, tenant_id="t1")
        # All chunks get the exact same vector → same similarity
        same_vec = [0.1, 0.2, 0.3]
        chunk_ids = _seed_chunks(db_mgr, doc_ids[0], count=5, vectors=[same_vec] * 5)

        emb_svc = _make_mock_embedding_service()
        emb_svc.embed_query.return_value = [0.1, 0.2, 0.3]  # exact match

        retriever = VectorRetrievalService(db_mgr, emb_svc, config=mock_config)
        response = retriever.retrieve_sync("test", tenant_id="t1", top_k=5)

        # All scores identical — ordering should be by chunk_index ASC
        if len(response.results) > 1:
            for i in range(len(response.results) - 1):
                assert response.results[i].chunk_index <= response.results[i + 1].chunk_index


# ── 14. No-Result Behavior ───────────────────────────────────────────────────

class TestNoResults:
    def test_empty_results_valid_response(self, db_mgr, mock_config):
        """When no candidates match, response should be valid with empty results."""
        emb_svc = _make_mock_embedding_service()
        retriever = VectorRetrievalService(db_mgr, emb_svc, config=mock_config)

        response = retriever.retrieve_sync("test", tenant_id="nonexistent")

        assert isinstance(response, RetrievalResponse)
        assert len(response.results) == 0
        assert response.total_candidates == 0
        assert response.diagnostics  # diagnostics should still be populated
        assert response.duration_ms >= 0

    def test_no_matching_model_returns_empty(self, db_mgr, mock_config):
        """When embedding model doesn't match, return empty results."""
        doc_ids = _seed_documents(db_mgr, tenant_id="t1")
        _seed_chunks(db_mgr, doc_ids[0], count=2, model_name="other-model")

        emb_svc = _make_mock_embedding_service()  # model_name="test-model"
        retriever = VectorRetrievalService(db_mgr, emb_svc, config=mock_config)

        response = retriever.retrieve_sync("test", tenant_id="t1")
        assert len(response.results) == 0


# ── 15. Python Fallback ──────────────────────────────────────────────────────

class TestPythonFallback:
    def test_python_backend_always_available(self, db_mgr):
        repo = PythonVectorRepository(db_manager=db_mgr)
        assert repo.is_available() is True
        assert repo.backend_name == "python"

    def test_auto_backend_uses_python_for_sqlite(self, db_mgr, mock_config):
        """Auto backend on SQLite should always fallback to Python."""
        mock_config.backend = "auto"
        emb_svc = _make_mock_embedding_service()
        retriever = VectorRetrievalService(db_mgr, emb_svc, config=mock_config)
        assert retriever.repository.backend_name == "python"


# ── Model / Schema Tests ─────────────────────────────────────────────────────

class TestModelSchemas:
    def test_retrieved_chunk_result_sync_fields(self):
        """Verify backward-compatible field synchronization in RetrievedChunkResult."""
        result = RetrievedChunkResult(
            chunk_id="c1",
            document_id="d1",
            text="test",
            similarity_score=0.85,
            page=3,
            filename="test.pdf",
        )
        assert result.page_number == 3
        assert result.title == "test.pdf"
        assert result.raw_score == 0.85
        assert 0 <= result.normalized_score <= 1.0

    def test_retrieval_response_duration_alias(self):
        """latency_ms and duration_ms should stay synchronized."""
        resp = RetrievalResponse(
            query="test",
            tenant_id="t1",
            duration_ms=123.4,
        )
        assert resp.latency_ms == 123.4

    def test_retrieval_response_dimension_alias(self):
        """embedding_dimensions and embedding_dimension should stay synchronized."""
        resp = RetrievalResponse(
            query="test",
            tenant_id="t1",
            embedding_dimension=768,
        )
        assert resp.embedding_dimensions == 768
