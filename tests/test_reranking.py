"""Hardened test suite for the Reranking subsystem and its integration into Hybrid Retrieval.

Covers:
- MockReranker protocol compliance and deterministic scoring
- Min-max score normalization [0, 1]
- Top-k candidate truncation
- RerankingService orchestration (sync & async)
- Query and document text truncation
- Complete score and metadata preservation (vector, BM25, RRF)
- Resilience & graceful degradation on provider error and timeout
- Strict error propagation when fallback is disabled
- In-memory LRU caching and tenant isolation
- End-to-end integration with HybridRetrievalService
"""

import asyncio
import pytest

from rag.core.config import RerankingConfig, RetrievalConfig
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, Embedding
from rag.core.models import DocumentStatus
from rag.embeddings.service import EmbeddingService
from rag.retrieval.exceptions import (
    RerankingError,
    RerankingProviderError,
    RerankingTimeoutError,
)
from rag.retrieval.hybrid_service import HybridRetrievalService
from rag.retrieval.models import RetrievalRequest, RetrievedChunkResult
from rag.retrieval.reranking import (
    MockReranker,
    Reranker,
    RerankingService,
    get_reranker,
)


# ── Test helpers ───────────────────────────────────────────────────────────────

def _make_chunk_result(
    chunk_id: str,
    text: str,
    rank: int = 1,
    sim_score: float = 0.85,
    rrf_score: float = 0.03,
    bm25_score: float = 4.2,
    dense_rank: int = 1,
    sparse_rank: int = 2,
    tenant_id: str = "tenant_test",
) -> RetrievedChunkResult:
    """Construct a full RetrievedChunkResult simulating post-fusion state."""
    return RetrievedChunkResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        document_version=1,
        document_version_id="v1",
        tenant_id=tenant_id,
        text=text,
        similarity_score=sim_score,
        raw_score=sim_score,
        rank=rank,
        rrf_score=rrf_score,
        bm25_score=bm25_score,
        dense_raw_score=sim_score,
        sparse_raw_score=bm25_score,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        retrieval_methods=["vector", "bm25"],
        fusion_strategy="rrf",
        metadata={"category": "engineering", "asset_id": "asset_100"},
    )


# ── 1. MockReranker Tests ──────────────────────────────────────────────────────

class TestMockReranker:
    """Verify MockReranker contract, protocol compliance, and scoring determinism."""

    def test_protocol_compliance(self):
        """MockReranker must satisfy the Reranker runtime protocol."""
        mock = MockReranker()
        assert isinstance(mock, Reranker)
        assert mock.model_name == "mock-reranker"
        assert mock.model_version == "v1.0"

    def test_deterministic_scoring_order(self):
        """Shorter text chunks should produce higher scores deterministically."""
        mock = MockReranker()
        docs = [
            _make_chunk_result("c_long", "A very very long descriptive document text for testing chunk ranking.", rank=1),
            _make_chunk_result("c_short", "Short text.", rank=2),
            _make_chunk_result("c_med", "Medium length document chunk text.", rank=3),
        ]
        resp = mock.rerank_sync("test query", docs, top_k=3)

        assert resp.output_count == 3
        # c_short should be rank 1 because it's the shortest
        assert resp.results[0].chunk_id == "c_short"
        assert resp.results[0].final_rank == 1
        assert resp.results[0].reranker_normalized_score >= resp.results[1].reranker_normalized_score

    def test_normalization_bounds(self):
        """Normalized scores must fall strictly in [0.0, 1.0]."""
        mock = MockReranker()
        docs = [
            _make_chunk_result(f"c_{i}", "x" * (i * 20), rank=i)
            for i in range(1, 6)
        ]
        resp = mock.rerank_sync("query", docs, top_k=5)

        for r in resp.results:
            assert 0.0 <= r.reranker_normalized_score <= 1.0
            assert r.reranker_raw_score > 0.0

        # Highest score should normalize to 1.0, lowest to 0.0
        assert resp.results[0].reranker_normalized_score == pytest.approx(1.0, abs=1e-4)
        assert resp.results[-1].reranker_normalized_score == pytest.approx(0.0, abs=1e-4)

    def test_top_k_truncation(self):
        """Only top_k candidates should be returned in output."""
        mock = MockReranker()
        docs = [_make_chunk_result(f"c_{i}", f"Text {i}", rank=i) for i in range(1, 11)]
        resp = mock.rerank_sync("query", docs, top_k=3)

        assert resp.output_count == 3
        assert len(resp.results) == 3
        assert [r.final_rank for r in resp.results] == [1, 2, 3]

    def test_empty_candidates(self):
        """Empty candidate sequence returns empty response cleanly."""
        mock = MockReranker()
        resp = mock.rerank_sync("query", [], top_k=5)
        assert resp.output_count == 0
        assert resp.results == []

    @pytest.mark.asyncio
    async def test_async_matches_sync(self):
        """Async and sync invocations produce identical results."""
        mock = MockReranker()
        docs = [
            _make_chunk_result("c_1", "Short text"),
            _make_chunk_result("c_2", "Longer text here for matching test"),
        ]
        sync_resp = mock.rerank_sync("query", docs, top_k=2)
        async_resp = await mock.rerank("query", docs, top_k=2)

        assert [r.chunk_id for r in sync_resp.results] == [r.chunk_id for r in async_resp.results]
        assert [r.reranker_raw_score for r in sync_resp.results] == [r.reranker_raw_score for r in async_resp.results]


# ── 2. RerankingService Orchestration ──────────────────────────────────────────

class TestRerankingService:
    """Verify RerankingService orchestration, document truncation, and score assignment."""

    def test_hydration_updates_ranks_and_fields(self):
        """Service sets reranker scores, model, and final ranks on chunk results."""
        cfg = RerankingConfig(top_k=2)
        svc = RerankingService(config=cfg)

        docs = [
            _make_chunk_result("c_long", "A very long text document chunk that will score lower.", rank=1),
            _make_chunk_result("c_short", "Tiny.", rank=2),
        ]
        hydrated, resp = svc.rerank_sync("query", docs, top_k=2)

        assert len(hydrated) == 2
        # c_short should now be ranked #1
        assert hydrated[0].chunk_id == "c_short"
        assert hydrated[0].rank == 1
        assert hydrated[0].reranker_raw_score is not None
        assert hydrated[0].reranker_normalized_score is not None
        assert hydrated[0].reranker_model == "mock-reranker"
        assert hydrated[0].reranking_failed is False

        assert hydrated[1].chunk_id == "c_long"
        assert hydrated[1].rank == 2

    def test_document_text_truncation(self):
        """Long text is truncated when prepared for provider without corrupting source object."""
        cfg = RerankingConfig(max_document_length=64)
        svc = RerankingService(config=cfg)

        original_text = "This is a very long text that definitely exceeds sixty-four characters in total length for sure."
        doc = _make_chunk_result("c_1", original_text)

        prepared = svc._prepare_documents([doc])
        assert len(prepared[0].text) == 64
        # Original doc must remain untouched
        assert len(doc.text) == len(original_text)

    def test_query_truncation(self):
        """Queries exceeding max_query_length are safely truncated."""
        cfg = RerankingConfig(max_query_length=10)
        svc = RerankingService(config=cfg)

        docs = [_make_chunk_result("c_1", "Sample text")]
        hydrated, resp = svc.rerank_sync("A query that is way too long for config", docs, top_k=1)
        assert len(hydrated) == 1
        assert resp.output_count == 1

    def test_empty_candidates_handling(self):
        """Empty input candidates returns empty list without error."""
        svc = RerankingService()
        hydrated, resp = svc.rerank_sync("query", [], top_k=5)
        assert hydrated == []
        assert resp.output_count == 0
        assert resp.input_count == 0

    @pytest.mark.asyncio
    async def test_async_rerank_flow(self):
        """Async service method operates smoothly."""
        svc = RerankingService()
        docs = [_make_chunk_result(f"c_{i}", f"Text {i}") for i in range(3)]
        hydrated, resp = await svc.rerank("query", docs, top_k=2)
        assert len(hydrated) == 2
        assert resp.output_count == 2


# ── 3. Score & Contract Preservation ──────────────────────────────────────────

class TestScorePreservation:
    """Verify that reranking does not wipe or corrupt any pre-existing retrieval scores."""

    def test_all_prior_scores_intact(self):
        """Vector, BM25, and RRF scores and metadata must remain intact after reranking."""
        svc = RerankingService()
        original_doc = _make_chunk_result(
            "c_test",
            "Document content",
            rank=1,
            sim_score=0.92,
            rrf_score=0.0321,
            bm25_score=5.67,
            dense_rank=2,
            sparse_rank=1,
        )

        hydrated, resp = svc.rerank_sync("query", [original_doc], top_k=1)
        res = hydrated[0]

        # Prior scores must remain unaltered
        assert res.similarity_score == pytest.approx(0.92)
        assert res.raw_score == pytest.approx(0.92)
        assert res.dense_raw_score == pytest.approx(0.92)
        assert res.bm25_score == pytest.approx(5.67)
        assert res.sparse_raw_score == pytest.approx(5.67)
        assert res.rrf_score == pytest.approx(0.0321)
        assert res.dense_rank == 2
        assert res.sparse_rank == 1
        assert res.fusion_strategy == "rrf"
        assert res.retrieval_methods == ["vector", "bm25"]
        assert res.metadata == {"category": "engineering", "asset_id": "asset_100"}

        # New reranker scores must be populated
        assert res.reranker_raw_score is not None
        assert res.reranker_normalized_score is not None
        assert res.reranker_model == "mock-reranker"
        assert res.rank == 1


# ── 4. Failure & Fallback Resilience ──────────────────────────────────────────

class TestFailureAndFallback:
    """Verify resilient degradation on provider errors and timeouts."""

    def test_fallback_on_provider_error(self):
        """When provider fails and fallback_enabled=True, original order is preserved with failure flags."""
        failing_provider = MockReranker(should_fail=True)
        cfg = RerankingConfig(fallback_enabled=True)
        svc = RerankingService(config=cfg, provider=failing_provider)

        docs = [
            _make_chunk_result("c_1", "First document", rank=1),
            _make_chunk_result("c_2", "Second document", rank=2),
        ]
        hydrated, resp = svc.rerank_sync("query", docs, top_k=2)

        assert resp.reranking_failed is True
        assert resp.fallback_used is True
        assert resp.error_message is not None
        assert len(hydrated) == 2
        # Order preserved
        assert hydrated[0].chunk_id == "c_1"
        assert hydrated[1].chunk_id == "c_2"
        assert hydrated[0].reranking_failed is True
        assert hydrated[1].reranking_failed is True

    def test_fallback_disabled_raises_error(self):
        """When fallback_enabled=False, provider error raises RerankingProviderError."""
        failing_provider = MockReranker(should_fail=True)
        cfg = RerankingConfig(fallback_enabled=False)
        svc = RerankingService(config=cfg, provider=failing_provider)

        docs = [_make_chunk_result("c_1", "Doc")]
        with pytest.raises(RerankingProviderError):
            svc.rerank_sync("query", docs, top_k=1)

    def test_fallback_on_timeout(self):
        """When provider times out and fallback_enabled=True, gracefully fall back."""
        timeout_provider = MockReranker(should_timeout=True)
        cfg = RerankingConfig(fallback_enabled=True)
        svc = RerankingService(config=cfg, provider=timeout_provider)

        docs = [_make_chunk_result("c_1", "Doc", rank=1)]
        hydrated, resp = svc.rerank_sync("query", docs, top_k=1)

        assert resp.reranking_failed is True
        assert resp.fallback_used is True
        assert hydrated[0].chunk_id == "c_1"

    def test_timeout_without_fallback_raises_error(self):
        """When fallback_enabled=False, timeout raises RerankingTimeoutError."""
        timeout_provider = MockReranker(should_timeout=True)
        cfg = RerankingConfig(fallback_enabled=False)
        svc = RerankingService(config=cfg, provider=timeout_provider)

        docs = [_make_chunk_result("c_1", "Doc")]
        with pytest.raises(RerankingTimeoutError):
            svc.rerank_sync("query", docs, top_k=1)


# ── 5. Caching & Tenant Isolation ──────────────────────────────────────────────

class TestCachingAndTenantIsolation:
    """Verify in-memory LRU cache and strict tenant boundary isolation."""

    def test_cache_hit_avoids_provider_call(self):
        """Subsequent identical requests hit the cache."""
        provider = MockReranker()
        cfg = RerankingConfig(cache_enabled=True)
        svc = RerankingService(config=cfg, provider=provider)

        docs = [_make_chunk_result("c_1", "Content")]
        # First call: hits provider
        svc.rerank_sync("query", docs, top_k=1, tenant_id="t1")
        assert provider.call_count == 1

        # Second call: hits cache
        svc.rerank_sync("query", docs, top_k=1, tenant_id="t1")
        assert provider.call_count == 1

    def test_tenant_isolation_in_cache(self):
        """Different tenants querying the same documents do not share cache keys."""
        provider = MockReranker()
        cfg = RerankingConfig(cache_enabled=True)
        svc = RerankingService(config=cfg, provider=provider)

        docs = [_make_chunk_result("c_1", "Content")]
        # Tenant 1 call
        svc.rerank_sync("query", docs, top_k=1, tenant_id="tenant_alpha")
        assert provider.call_count == 1

        # Tenant 2 call (same query and docs, different tenant)
        svc.rerank_sync("query", docs, top_k=1, tenant_id="tenant_beta")
        assert provider.call_count == 2

    def test_cache_clear(self):
        """Clearing the cache forces subsequent calls to invoke the provider."""
        provider = MockReranker()
        cfg = RerankingConfig(cache_enabled=True)
        svc = RerankingService(config=cfg, provider=provider)

        docs = [_make_chunk_result("c_1", "Content")]
        svc.rerank_sync("query", docs, top_k=1)
        assert provider.call_count == 1

        svc.clear_cache()
        svc.rerank_sync("query", docs, top_k=1)
        assert provider.call_count == 2


# ── 6. End-to-End Hybrid Retrieval Integration ─────────────────────────────────

class TestHybridWithReranking:
    """Verify end-to-end integration of reranking into HybridRetrievalService."""

    def test_hybrid_pipeline_with_reranker(self):
        """Hybrid pipeline executes reranker, applies scores, and populates diagnostics."""
        db = DatabaseManager(db_url="sqlite:///:memory:")
        db.create_tables()
        emb_svc = EmbeddingService(db_manager=db)
        dim = emb_svc.dimension

        with db.session() as session:
            doc = Document(
                document_id="doc_rk",
                tenant_id="tenant_rk",
                source="local",
                filename="sensor_manual.pdf",
                source_uri="s3://sensor_manual.pdf",
                mime_type="application/pdf",
                content_hash="hash_rk",
                status=DocumentStatus.INDEXED.value,
                version=1,
            )
            c1 = Chunk(
                chunk_id="c_rk_1",
                document_id="doc_rk",
                version=1,
                chunk_index=0,
                text="Longer text about optical sensors and camera modules.",
                section="Section 1",
                page=1,
            )
            c2 = Chunk(
                chunk_id="c_rk_2",
                document_id="doc_rk",
                version=1,
                chunk_index=1,
                text="Short camera spec.",
                section="Section 2",
                page=2,
            )
            e1 = Embedding(
                chunk_id="c_rk_1",
                embedding_model=emb_svc.model_name,
                embedding_version=emb_svc.model_version,
                dimension=dim,
                provider=emb_svc.provider_name,
                vector=[0.1] * dim,
            )
            e2 = Embedding(
                chunk_id="c_rk_2",
                embedding_model=emb_svc.model_name,
                embedding_version=emb_svc.model_version,
                dimension=dim,
                provider=emb_svc.provider_name,
                vector=[0.08] * dim,
            )
            session.add_all([doc, c1, c2, e1, e2])

        config = RetrievalConfig()
        config.reranking.enabled = True
        config.reranking.top_k = 2

        hybrid_svc = HybridRetrievalService(
            db_manager=db,
            embedding_service=emb_svc,
            config=config,
        )

        req = RetrievalRequest(query="camera sensors", tenant_id="tenant_rk", top_k=2)
        resp = hybrid_svc.retrieve(req)

        assert resp.retrieval_method == "hybrid"
        assert resp.reranking_method == "mock-reranker"
        assert resp.reranking_latency_ms is not None
        assert resp.reranking_latency_ms >= 0
        assert len(resp.results) >= 1

        # Each result has reranking fields populated
        for r in resp.results:
            assert r.reranker_model == "mock-reranker"
            assert r.reranker_raw_score is not None
            assert r.reranker_normalized_score is not None
            assert r.reranking_failed is False
            assert r.rrf_score is not None  # RRF score preserved!

        # Diagnostics populated
        diag = resp.diagnostics
        assert diag.get("reranking_enabled") is True
        assert diag.get("reranking_model") == "mock-reranker"
        assert "reranking_latency_ms" in diag
        assert diag.get("reranking_failed") is False

    def test_hybrid_pipeline_with_reranker_fallback(self):
        """When reranker in hybrid pipeline fails, pipeline gracefully falls back to RRF results."""
        db = DatabaseManager(db_url="sqlite:///:memory:")
        db.create_tables()
        emb_svc = EmbeddingService(db_manager=db)
        dim = emb_svc.dimension

        with db.session() as session:
            doc = Document(
                document_id="doc_rk_fail",
                tenant_id="tenant_rk_fail",
                source="local",
                filename="manual.pdf",
                source_uri="s3://manual.pdf",
                mime_type="application/pdf",
                content_hash="hash_rk_fail",
                status=DocumentStatus.INDEXED.value,
                version=1,
            )
            c1 = Chunk(
                chunk_id="c_rk_f1",
                document_id="doc_rk_fail",
                version=1,
                chunk_index=0,
                text="LiDAR optics and autonomous driving sensors.",
                section="Section 1",
                page=1,
            )
            e1 = Embedding(
                chunk_id="c_rk_f1",
                embedding_model=emb_svc.model_name,
                embedding_version=emb_svc.model_version,
                dimension=dim,
                provider=emb_svc.provider_name,
                vector=[0.1] * dim,
            )
            session.add_all([doc, c1, e1])

        config = RetrievalConfig()
        config.reranking.enabled = True
        config.reranking.fallback_enabled = True

        failing_provider = MockReranker(should_fail=True)
        failing_rerank_svc = RerankingService(config=config.reranking, provider=failing_provider)

        hybrid_svc = HybridRetrievalService(
            db_manager=db,
            embedding_service=emb_svc,
            config=config,
            reranking_service=failing_rerank_svc,
        )

        req = RetrievalRequest(query="LiDAR optics", tenant_id="tenant_rk_fail", top_k=2)
        resp = hybrid_svc.retrieve(req)

        assert resp.retrieval_method == "hybrid"
        assert len(resp.results) >= 1
        assert resp.diagnostics.get("reranking_failed") is True
        assert resp.diagnostics.get("reranking_fallback_used") is True
        for r in resp.results:
            assert r.reranking_failed is True

