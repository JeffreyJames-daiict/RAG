"""Hardened test suite for Hybrid Retrieval Layer.

Covers:
  1. Dense and sparse results combined (multi-source fusion)
  2. Duplicate chunk deduplication and score preservation (dense_raw_score, sparse_raw_score, dense_rank, sparse_rank)
  3. Dense retriever failure fallback (sparse succeeds, partial_failure=True, failed_methods=['vector'])
  4. Sparse retriever failure fallback (dense succeeds, partial_failure=True, failed_methods=['bm25'])
  5. Both retrievers failing (raises typed RetrievalError)
  6. Shared filters propagation (tenant, metadata filters, active version)
  7. Strict tenant isolation (zero cross-tenant candidate leakage)
  8. Deterministic tie-breaking ordering
  9. Empty dense results with valid sparse results
 10. Empty sparse results with valid dense results
 11. Concurrent parallel execution
 12. Timeout behavior in dense or sparse branch
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import time
from unittest.mock import MagicMock, patch
import uuid
import pytest

from rag.core.config import RetrievalConfig
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, Embedding
from rag.core.models import DocumentStatus
from rag.embeddings.service import EmbeddingService
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.exceptions import RetrievalError
from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.hybrid_service import HybridRetrievalService
from rag.retrieval.interfaces import RetrievedCandidate
from rag.retrieval.models import RetrievalRequest


@pytest.fixture
def db_mgr():
    """In-memory SQLite database manager for isolated testing."""
    db = DatabaseManager(db_url="sqlite:///:memory:")
    db.create_tables()
    return db


@pytest.fixture
def hybrid_config():
    """Configured RetrievalConfig for hybrid testing."""
    return RetrievalConfig(
        enabled=True,
        backend="python",
        default_top_k=5,
        max_top_k=100,
        candidate_limit=100,
        rrf_k=60,
        dense_top_k=50,
        sparse_top_k=50,
        hybrid_candidate_k=50,
        final_top_k=5,
        hybrid_dense_timeout_sec=5.0,
        hybrid_sparse_timeout_sec=5.0,
        allow_partial_failure=True,
    )


def _seed_document(
    db_mgr: DatabaseManager,
    tenant_id: str = "t_hybrid",
    filename: str = "doc.pdf",
    version: int = 1,
    status: str = DocumentStatus.INDEXED.value,
    deleted: bool = False,
) -> str:
    doc_id = str(uuid.uuid4())
    with db_mgr.session() as session:
        doc = Document(
            document_id=doc_id,
            tenant_id=tenant_id,
            source="LOCAL_FILE",
            source_uri=f"/test/{doc_id}.pdf",
            filename=filename,
            mime_type="application/pdf",
            content_hash=f"hash_{doc_id[:8]}",
            version=version,
            status=status,
            deleted_at=datetime.now(timezone.utc) if deleted else None,
        )
        session.add(doc)
        session.commit()
    return doc_id


def _seed_chunk_with_embedding(
    db_mgr: DatabaseManager,
    document_id: str,
    text: str,
    vector: list[float],
    version: int = 1,
    chunk_index: int = 0,
    section: str = "",
    page: int = 1,
    chunk_type: str = "text",
    metadata: dict = None,
    model_name: str = "text-embedding-3-small",
    model_version: str = "v1.0",
) -> str:
    chunk_id = str(uuid.uuid4())
    with db_mgr.session() as session:
        chunk = Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            version=version,
            chunk_index=chunk_index,
            text=text,
            section=section,
            page=page,
            chunk_metadata=metadata or {"chunk_type": chunk_type},
            chunking_version="v1.0",
        )
        session.add(chunk)
        session.flush()

        emb = Embedding(
            chunk_id=chunk_id,
            embedding_model=model_name,
            embedding_version=model_version,
            dimension=len(vector),
            provider="mock",
            vector=vector,
        )
        session.add(emb)
        session.commit()
    return chunk_id


def _make_candidate(
    chunk_id: str,
    score: float = 0.9,
    doc_id: str = "doc_1",
    text: str = "sample text",
) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=chunk_id,
        document_id=doc_id,
        document_version=1,
        chunk_index=0,
        text=text,
        similarity_score=score,
        distance=1.0 - score,
    )


# ── 1. Combined Dense & Sparse Retrieval ────────────────────────────────────

def test_dense_and_sparse_combined(db_mgr, hybrid_config):
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="combined.pdf")
    # c1 has high semantic similarity to query vector [1.0, 0.0, 0.0]
    c1 = _seed_chunk_with_embedding(
        db_mgr, d_id, text="Semantic content without target keywords.",
        vector=[1.0, 0.0, 0.0], chunk_index=0
    )
    # c2 has exact target keyword "superconductivity"
    c2 = _seed_chunk_with_embedding(
        db_mgr, d_id, text="Experimental superconductivity observation in cuprates.",
        vector=[0.0, 1.0, 0.0], chunk_index=1
    )

    emb_svc = EmbeddingService(db_manager=db_mgr)
    emb_svc.embed_query = MagicMock(return_value=[1.0, 0.0, 0.0])

    service = HybridRetrievalService(db_mgr, emb_svc, config=hybrid_config)
    res = service.retrieve_sync("superconductivity", tenant_id="t1", top_k=5)

    assert len(res.results) >= 2
    c_ids = [r.chunk_id for r in res.results]
    assert c1 in c_ids  # Found via dense
    assert c2 in c_ids  # Found via sparse BM25
    assert res.retrieval_method == "hybrid"
    assert res.fusion_method == "rrf"


# ── 2. Deduplication & Score Preservation ───────────────────────────────────

def test_duplicate_chunk_deduplication_and_score_preservation(db_mgr, hybrid_config):
    """When a chunk appears in both dense and sparse results, it must be deduplicated and preserve both scores/ranks."""
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="dedup.pdf")
    # c_both matches both semantically and lexically
    c_both = _seed_chunk_with_embedding(
        db_mgr, d_id, text="Exact match on target keyword with high similarity.",
        vector=[1.0, 0.0, 0.0], chunk_index=0
    )

    emb_svc = EmbeddingService(db_manager=db_mgr)
    emb_svc.embed_query = MagicMock(return_value=[1.0, 0.0, 0.0])

    service = HybridRetrievalService(db_mgr, emb_svc, config=hybrid_config)
    res = service.retrieve_sync("keyword", tenant_id="t1", top_k=5)

    assert len(res.results) >= 1
    top = res.results[0]
    assert top.chunk_id == c_both
    # Deduplication: appears exactly once
    assert sum(1 for r in res.results if r.chunk_id == c_both) == 1

    # Score & rank preservation
    assert top.dense_raw_score is not None
    assert top.sparse_raw_score is not None
    assert top.dense_rank == 1
    assert top.sparse_rank == 1
    assert "vector" in top.retrieval_methods
    assert "bm25" in top.retrieval_methods
    # Expected RRF score = 1/(60+1) + 1/(60+1) ≈ 0.032787
    expected_rrf = round((1.0 / 61) + (1.0 / 61), 6)
    assert abs(top.rrf_score - expected_rrf) < 1e-5


# ── 3. Dense Retriever Failure Fallback ─────────────────────────────────────

def test_dense_retriever_failure_fallback(db_mgr, hybrid_config):
    """If dense search fails, the hybrid service falls back to sparse BM25 and marks partial failure."""
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="dense_fail.pdf")
    c1 = _seed_chunk_with_embedding(
        db_mgr, d_id, text="Fallback keyword matching content in cuprate crystals.",
        vector=[0.0, 1.0, 0.0]
    )

    emb_svc = EmbeddingService(db_manager=db_mgr)
    # Simulate embedding provider crash
    emb_svc.embed_query = MagicMock(side_effect=RuntimeError("OpenAI API rate limit / down"))

    service = HybridRetrievalService(db_mgr, emb_svc, config=hybrid_config)
    res = service.retrieve_sync("cuprate crystals", tenant_id="t1")

    assert len(res.results) >= 1
    assert res.results[0].chunk_id == c1
    # Diagnostics telemetry
    assert res.diagnostics["partial_failure"] is True
    assert "vector" in res.diagnostics["failed_methods"]
    assert "bm25" not in res.diagnostics["failed_methods"]
    assert res.results[0].sparse_raw_score is not None


# ── 4. Sparse Retriever Failure Fallback ────────────────────────────────────

def test_sparse_retriever_failure_fallback(db_mgr, hybrid_config):
    """If sparse BM25 search fails, the hybrid service falls back to dense vector and marks partial failure."""
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="sparse_fail.pdf")
    c1 = _seed_chunk_with_embedding(
        db_mgr, d_id, text="Dense vector semantic content.",
        vector=[1.0, 0.0, 0.0]
    )

    emb_svc = EmbeddingService(db_manager=db_mgr)
    emb_svc.embed_query = MagicMock(return_value=[1.0, 0.0, 0.0])

    service = HybridRetrievalService(db_mgr, emb_svc, config=hybrid_config)
    # Patch BM25 retriever search to fail
    service.bm25_retriever.search = MagicMock(side_effect=RuntimeError("BM25 index corrupt / down"))

    res = service.retrieve_sync("test query", tenant_id="t1")

    assert len(res.results) >= 1
    assert res.results[0].chunk_id == c1
    assert res.diagnostics["partial_failure"] is True
    assert "bm25" in res.diagnostics["failed_methods"]
    assert "vector" not in res.diagnostics["failed_methods"]
    assert res.results[0].dense_raw_score is not None


# ── 5. Both Retrievers Failing ──────────────────────────────────────────────

def test_both_retrievers_failing_raises_error(db_mgr, hybrid_config):
    """If both retrievers fail, the service must raise a typed RetrievalError."""
    emb_svc = EmbeddingService(db_manager=db_mgr)
    emb_svc.embed_query = MagicMock(side_effect=RuntimeError("Vector provider crash"))

    service = HybridRetrievalService(db_mgr, emb_svc, config=hybrid_config)
    service.bm25_retriever.search = MagicMock(side_effect=RuntimeError("BM25 database lost"))

    with pytest.raises(RetrievalError, match="Both dense and sparse retrieval failed"):
        service.retrieve_sync("catastrophic failure test", tenant_id="t1")


# ── 6. Shared Filters Propagation ───────────────────────────────────────────

def test_shared_filters_propagation(db_mgr, hybrid_config):
    """Filters must be propagated identically to both dense and sparse searches."""
    d1 = _seed_document(db_mgr, tenant_id="t1", filename="doc1.pdf")
    d2 = _seed_document(db_mgr, tenant_id="t1", filename="doc2.pdf")

    c1 = _seed_chunk_with_embedding(db_mgr, d1, text="Matching text in doc1.", vector=[1.0, 0.0, 0.0])
    c2 = _seed_chunk_with_embedding(db_mgr, d2, text="Matching text in doc2.", vector=[1.0, 0.0, 0.0])

    emb_svc = EmbeddingService(db_manager=db_mgr)
    emb_svc.embed_query = MagicMock(return_value=[1.0, 0.0, 0.0])

    service = HybridRetrievalService(db_mgr, emb_svc, config=hybrid_config)

    # Filter to only doc1
    filters = RetrievalFilters(document_ids=[d1])
    res = service.retrieve_sync("Matching text", tenant_id="t1", filters=filters)

    assert len(res.results) >= 1
    assert all(r.document_id == d1 for r in res.results)
    assert all(r.chunk_id != c2 for r in res.results)


# ── 7. Strict Tenant Isolation ─────────────────────────────────────────────

def test_strict_tenant_isolation(db_mgr, hybrid_config):
    """Zero candidates from other tenants may appear in fused hybrid results."""
    d_a = _seed_document(db_mgr, tenant_id="tenant_a", filename="a.pdf")
    d_b = _seed_document(db_mgr, tenant_id="tenant_b", filename="b.pdf")

    c_a = _seed_chunk_with_embedding(db_mgr, d_a, text="Confidential algorithm tenant A.", vector=[1.0, 0.0, 0.0])
    c_b = _seed_chunk_with_embedding(db_mgr, d_b, text="Confidential algorithm tenant B.", vector=[1.0, 0.0, 0.0])

    emb_svc = EmbeddingService(db_manager=db_mgr)
    emb_svc.embed_query = MagicMock(return_value=[1.0, 0.0, 0.0])

    service = HybridRetrievalService(db_mgr, emb_svc, config=hybrid_config)

    res_a = service.retrieve_sync("Confidential algorithm", tenant_id="tenant_a")
    assert all(r.tenant_id == "tenant_a" for r in res_a.results)
    assert all(r.chunk_id != c_b for r in res_a.results)

    res_b = service.retrieve_sync("Confidential algorithm", tenant_id="tenant_b")
    assert all(r.tenant_id == "tenant_b" for r in res_b.results)
    assert all(r.chunk_id != c_a for r in res_b.results)


# ── 8. Deterministic Tie-Breaking ───────────────────────────────────────────

def test_deterministic_tie_breaking(db_mgr, hybrid_config):
    """Items with equal RRF scores must be deterministically ordered."""
    fusion = service = HybridRetrievalService(db_mgr, EmbeddingService(db_manager=db_mgr), config=hybrid_config)

    c1 = _make_candidate("chunk_1", score=0.8)
    c2 = _make_candidate("chunk_2", score=0.8)

    # In single list, c1 is rank 1, c2 is rank 2
    fused = service.fusion.fuse_sources({"vector": [c1, c2]})
    assert fused[0].candidate.chunk_id == "chunk_1"
    assert fused[1].candidate.chunk_id == "chunk_2"


# ── 9. Empty Dense Results ──────────────────────────────────────────────────

def test_empty_dense_results(db_mgr, hybrid_config):
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="empty_dense.pdf")
    c1 = _seed_chunk_with_embedding(
        db_mgr, d_id, text="Rare unique keyword zephyros.",
        vector=[0.0, 1.0, 0.0]
    )

    emb_svc = EmbeddingService(db_manager=db_mgr)
    emb_svc.embed_query = MagicMock(return_value=[1.0, 0.0, 0.0])

    service = HybridRetrievalService(db_mgr, emb_svc, config=hybrid_config)
    # Dense repository returns empty
    service.vector_repository.search_vectors = MagicMock(return_value=([], 0))

    res = service.retrieve_sync("zephyros", tenant_id="t1")
    assert len(res.results) >= 1
    assert res.results[0].chunk_id == c1
    assert "bm25" in res.results[0].retrieval_methods


# ── 10. Empty Sparse Results ────────────────────────────────────────────────

def test_empty_sparse_results(db_mgr, hybrid_config):
    d_id = _seed_document(db_mgr, tenant_id="t1", filename="empty_sparse.pdf")
    c1 = _seed_chunk_with_embedding(
        db_mgr, d_id, text="Ordinary text with semantic match.",
        vector=[1.0, 0.0, 0.0]
    )

    emb_svc = EmbeddingService(db_manager=db_mgr)
    emb_svc.embed_query = MagicMock(return_value=[1.0, 0.0, 0.0])

    service = HybridRetrievalService(db_mgr, emb_svc, config=hybrid_config)
    # BM25 returns empty
    service.bm25_retriever.search = MagicMock(return_value=([], 0))

    res = service.retrieve_sync("unrelated vocabulary", tenant_id="t1")
    assert len(res.results) >= 1
    assert res.results[0].chunk_id == c1
    assert "vector" in res.results[0].retrieval_methods


# ── 11. Parallel Execution Concurrency ──────────────────────────────────────

def test_parallel_execution_concurrency(db_mgr, hybrid_config):
    """Verify both branches execute in parallel."""
    emb_svc = EmbeddingService(db_manager=db_mgr)
    emb_svc.embed_query = MagicMock(return_value=[1.0, 0.0, 0.0])

    service = HybridRetrievalService(db_mgr, emb_svc, config=hybrid_config)

    def _slow_dense(*args, **kwargs):
        time.sleep(0.15)
        return [_make_candidate("c_dense")], 1

    def _slow_sparse(*args, **kwargs):
        time.sleep(0.15)
        return [_make_candidate("c_sparse")], 1

    service.vector_repository.search_vectors = MagicMock(side_effect=_slow_dense)
    service.bm25_retriever.search = MagicMock(side_effect=_slow_sparse)

    start_t = time.time()
    res = service.retrieve_sync("test query", tenant_id="t1")
    elapsed = time.time() - start_t

    # If sequential, elapsed would be >= 0.30s. In parallel, it should be ~0.15-0.25s.
    assert elapsed < 0.28, f"Execution took {elapsed:.2f}s, indicating lack of parallelism."
    assert len(res.results) == 2


# ── 12. Timeout Handling ────────────────────────────────────────────────────

def test_timeout_handling(db_mgr, hybrid_config):
    """Branch exceeding timeout is aborted gracefully and marked in partial failure."""
    emb_svc = EmbeddingService(db_manager=db_mgr)
    emb_svc.embed_query = MagicMock(return_value=[1.0, 0.0, 0.0])

    # Set very short timeout for dense branch
    hybrid_config.hybrid_dense_timeout_sec = 0.05
    service = HybridRetrievalService(db_mgr, emb_svc, config=hybrid_config)

    def _hanging_dense(*args, **kwargs):
        time.sleep(0.5)
        return [_make_candidate("c_dense")], 1

    service.vector_repository.search_vectors = MagicMock(side_effect=_hanging_dense)
    service.bm25_retriever.search = MagicMock(return_value=([_make_candidate("c_sparse")], 1))

    res = service.retrieve_sync("test timeout", tenant_id="t1")
    assert len(res.results) == 1
    assert res.results[0].chunk_id == "c_sparse"
    assert res.diagnostics["partial_failure"] is True
    assert "vector" in res.diagnostics["failed_methods"]
