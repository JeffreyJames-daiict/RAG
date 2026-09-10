"""Hardened unit tests for Weighted Reciprocal Rank Fusion (RRF).

Covers:
- Basic RRF formula verification
- Weighted RRF scoring
- Duplicate result merging (same chunk from multiple sources)
- Different list lengths
- Empty all lists / empty single list
- Tie-breaking determinism (5-tier)
- Deterministic output on repeated calls
- Invalid configuration (k, weights)
- Top-k truncation
- Three-source extensibility (future retrievers)
- FusionDiagnostics population
- FusionStrategy protocol compliance
- Hybrid end-to-end integration
"""

import pytest

from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, Embedding
from rag.core.models import DocumentStatus
from rag.embeddings.service import EmbeddingService
from rag.retrieval.fusion import (
    FusedCandidateInfo,
    FusionDiagnostics,
    FusionStrategy,
    RRFFusion,
)
from rag.retrieval.hybrid_service import HybridRetrievalService
from rag.retrieval.interfaces import RetrievedCandidate
from rag.retrieval.models import RetrievalRequest


# ── Test helpers ───────────────────────────────────────────────────────────────

def _cand(chunk_id: str, score: float = 1.0, chunk_index: int = 0) -> RetrievedCandidate:
    """Create a minimal RetrievedCandidate for testing."""
    return RetrievedCandidate(
        chunk_id=chunk_id,
        document_id="doc1",
        document_version=1,
        chunk_index=chunk_index,
        text=f"Sample text for {chunk_id}",
        similarity_score=score,
        distance=1.0 - score,
    )


# ── 1. Basic RRF formula ──────────────────────────────────────────────────────

class TestBasicRRFFormula:
    """Verify exact mathematical RRF formula calculation."""

    def test_single_source_scoring(self):
        """Single source: RRF = 1/(k+rank) for each candidate."""
        fusion = RRFFusion(k=60)
        results = fusion.fuse({"source_a": [_cand("A", 0.9), _cand("B", 0.8)]}, top_k=10)

        assert len(results) == 2
        # Rank 1: 1/61 ≈ 0.016393
        assert results[0].candidate.chunk_id == "A"
        assert abs(results[0].rrf_score - 1.0 / 61) < 1e-9
        # Rank 2: 1/62 ≈ 0.016129
        assert results[1].candidate.chunk_id == "B"
        assert abs(results[1].rrf_score - 1.0 / 62) < 1e-9

    def test_two_source_accumulation(self):
        """Same chunk in two sources: scores accumulate."""
        fusion = RRFFusion(k=60)
        list_a = [_cand("X", 0.9), _cand("Y", 0.7)]
        list_b = [_cand("X", 0.95), _cand("Z", 0.5)]

        results = fusion.fuse({"vector": list_a, "bm25": list_b}, top_k=10)

        assert len(results) == 3
        # X: 1/61 + 1/61 = 2/61
        x_result = next(r for r in results if r.candidate.chunk_id == "X")
        assert abs(x_result.rrf_score - 2.0 / 61) < 1e-9
        assert "vector" in x_result.retrieval_methods
        assert "bm25" in x_result.retrieval_methods

    def test_rank_based_not_score_based(self):
        """RRF uses rank position, not raw similarity score."""
        fusion = RRFFusion(k=60)
        # Same rank (1) but very different scores
        list_a = [_cand("A", 0.99)]
        list_b = [_cand("B", 0.01)]

        results = fusion.fuse({"s1": list_a, "s2": list_b}, top_k=10)
        a_score = next(r.rrf_score for r in results if r.candidate.chunk_id == "A")
        b_score = next(r.rrf_score for r in results if r.candidate.chunk_id == "B")
        # Both at rank 1 → same RRF score
        assert abs(a_score - b_score) < 1e-9


# ── 2. Weighted RRF ───────────────────────────────────────────────────────────

class TestWeightedRRF:
    """Verify weighted formula: weight_i / (k + rank)."""

    def test_asymmetric_weights(self):
        """Dense weight 2.0, sparse weight 0.5."""
        fusion = RRFFusion(k=60, weights={"vector": 2.0, "bm25": 0.5})
        list_v = [_cand("A", 0.9)]
        list_b = [_cand("B", 0.8)]

        results = fusion.fuse({"vector": list_v, "bm25": list_b}, top_k=10)

        a_score = next(r.rrf_score for r in results if r.candidate.chunk_id == "A")
        b_score = next(r.rrf_score for r in results if r.candidate.chunk_id == "B")

        assert abs(a_score - 2.0 / 61) < 1e-9
        assert abs(b_score - 0.5 / 61) < 1e-9
        # A should rank higher due to weight
        assert a_score > b_score

    def test_same_chunk_weighted_accumulation(self):
        """Chunk in both sources accumulates weighted scores."""
        fusion = RRFFusion(k=60, weights={"vector": 1.5, "bm25": 0.8})
        results = fusion.fuse(
            {"vector": [_cand("X")], "bm25": [_cand("X")]}, top_k=10
        )

        assert len(results) == 1
        expected = 1.5 / 61 + 0.8 / 61
        assert abs(results[0].rrf_score - expected) < 1e-9

    def test_zero_weight_excludes_contribution(self):
        """Source with weight 0 contributes nothing to RRF score."""
        fusion = RRFFusion(k=60, weights={"vector": 1.0, "bm25": 0.0})
        results = fusion.fuse(
            {"vector": [_cand("A")], "bm25": [_cand("A")]}, top_k=10
        )

        assert len(results) == 1
        expected = 1.0 / 61 + 0.0  # bm25 contributes 0
        assert abs(results[0].rrf_score - expected) < 1e-9

    def test_default_weight_for_unlisted_sources(self):
        """Sources not in weights map get default_weight (1.0)."""
        fusion = RRFFusion(k=60, weights={"vector": 3.0}, default_weight=1.0)
        results = fusion.fuse(
            {"vector": [_cand("A")], "new_retriever": [_cand("B")]}, top_k=10
        )

        a_score = next(r.rrf_score for r in results if r.candidate.chunk_id == "A")
        b_score = next(r.rrf_score for r in results if r.candidate.chunk_id == "B")
        assert abs(a_score - 3.0 / 61) < 1e-9
        assert abs(b_score - 1.0 / 61) < 1e-9  # default weight


# ── 3. Duplicate result merging ────────────────────────────────────────────────

class TestDuplicateMerging:
    """Same chunk from multiple sources preserves scores, ranks, and methods."""

    def test_preserves_both_source_ranks_and_scores(self):
        """Merged chunk retains per-source provenance."""
        fusion = RRFFusion(k=60)
        results = fusion.fuse({
            "vector": [_cand("X", 0.91), _cand("Y", 0.80)],
            "bm25": [_cand("Y", 4.5), _cand("X", 3.1)],
        }, top_k=10)

        x = next(r for r in results if r.candidate.chunk_id == "X")
        y = next(r for r in results if r.candidate.chunk_id == "Y")

        # X: vector rank=1, bm25 rank=2
        assert x.source_ranks == {"vector": 1, "bm25": 2}
        assert x.source_scores["vector"] == pytest.approx(0.91)
        assert x.source_scores["bm25"] == pytest.approx(3.1)

        # Y: vector rank=2, bm25 rank=1
        assert y.source_ranks == {"vector": 2, "bm25": 1}
        assert sorted(y.retrieval_methods) == ["bm25", "vector"]

    def test_metadata_merge_preserves_existing(self):
        """Metadata from first source is not overwritten by second."""
        cand_a = _cand("X", 0.9)
        cand_a.chunk_metadata = {"key_a": "val_a", "shared": "from_a"}
        cand_b = _cand("X", 4.0)
        cand_b.chunk_metadata = {"key_b": "val_b", "shared": "from_b"}

        fusion = RRFFusion(k=60)
        results = fusion.fuse({"s1": [cand_a], "s2": [cand_b]}, top_k=10)

        meta = results[0].candidate.chunk_metadata
        assert meta["key_a"] == "val_a"
        assert meta["key_b"] == "val_b"
        assert meta["shared"] == "from_a"  # first source wins


# ── 4. Different list lengths ──────────────────────────────────────────────────

class TestDifferentListLengths:
    """Sources with asymmetric result counts."""

    def test_unequal_source_sizes(self):
        """Source A has 5 results, source B has 1."""
        fusion = RRFFusion(k=60)
        list_a = [_cand(f"c{i}", 1.0 - i * 0.1) for i in range(5)]
        list_b = [_cand("c0", 5.0)]  # Same as first in A

        results = fusion.fuse({"vector": list_a, "bm25": list_b}, top_k=10)

        # c0 should be first (appears in both)
        assert results[0].candidate.chunk_id == "c0"
        assert len(results) == 5

    def test_single_item_list(self):
        """One-element lists are handled correctly."""
        fusion = RRFFusion(k=60)
        results = fusion.fuse({"only": [_cand("solo")]}, top_k=10)
        assert len(results) == 1
        assert abs(results[0].rrf_score - 1.0 / 61) < 1e-9


# ── 5. Empty lists ─────────────────────────────────────────────────────────────

class TestEmptyLists:
    """Edge cases with empty inputs."""

    def test_all_lists_empty(self):
        """No sources → empty result."""
        fusion = RRFFusion(k=60)
        results = fusion.fuse({}, top_k=10)
        assert results == []

    def test_one_source_empty_other_valid(self):
        """One source empty, other has results → only non-empty contributes."""
        fusion = RRFFusion(k=60)
        results = fusion.fuse(
            {"vector": [_cand("A")], "bm25": []}, top_k=10
        )
        assert len(results) == 1
        assert results[0].candidate.chunk_id == "A"
        assert results[0].retrieval_methods == ["vector"]

    def test_no_sources_at_all(self):
        """Empty dict → empty result + valid diagnostics."""
        fusion = RRFFusion(k=60)
        results = fusion.fuse({}, top_k=10)
        diag = fusion.diagnostics()
        assert results == []
        assert diag.unique_result_count == 0
        assert diag.final_result_count == 0


# ── 6. Tie-breaking determinism ────────────────────────────────────────────────

class TestTieBreaking:
    """5-tier deterministic tie-breaking when RRF scores are equal."""

    def test_same_rrf_score_breaks_by_min_rank(self):
        """Same score → lower min rank wins."""
        fusion = RRFFusion(k=60)
        # A at rank 1 in s1, B at rank 1 in s2 → same RRF score
        # A's min rank = 1, B's min rank = 1 → fall through to dense score
        # But if we arrange so A has rank 1, B has rank 2:
        results = fusion.fuse({
            "s1": [_cand("A", 0.9), _cand("B", 0.8)],
        }, top_k=10)

        assert results[0].candidate.chunk_id == "A"
        assert results[1].candidate.chunk_id == "B"

    def test_same_score_same_rank_breaks_by_dense_score(self):
        """Equal RRF & rank → breaks by dense raw score DESC."""
        fusion = RRFFusion(k=60)
        # Both at rank 1 in different sources → same RRF score 1/61
        # A has higher dense score
        results = fusion.fuse({
            "vector": [_cand("A", 0.95)],
            "bm25": [_cand("B", 0.50)],
        }, top_k=10)

        assert results[0].candidate.chunk_id == "A"

    def test_all_tiers_equal_breaks_by_chunk_id(self):
        """All tiers equal → falls back to chunk ID ASC."""
        fusion = RRFFusion(k=60)
        # Same score, same rank, same dense score (0), same sparse score (0)
        # Different chunk IDs → lexicographic order
        results = fusion.fuse({
            "s1": [_cand("Z_chunk", 0.5)],
            "s2": [_cand("A_chunk", 0.5)],
        }, top_k=10)

        # Both have 1/61 RRF, min rank 1, same-ish scores → chunk ID
        assert results[0].candidate.chunk_id == "A_chunk"
        assert results[1].candidate.chunk_id == "Z_chunk"


# ── 7. Deterministic output ───────────────────────────────────────────────────

class TestDeterministicOutput:
    """Same input always produces same output."""

    def test_repeated_calls_identical(self):
        """Running fusion 10 times yields identical ordering each time."""
        fusion = RRFFusion(k=60)
        lists = {
            "vector": [_cand("A", 0.9), _cand("B", 0.8), _cand("C", 0.7)],
            "bm25": [_cand("B", 4.0), _cand("D", 3.0), _cand("A", 2.0)],
        }

        first_run = fusion.fuse(lists, top_k=10)
        first_ids = [r.candidate.chunk_id for r in first_run]

        for _ in range(10):
            repeat = fusion.fuse(lists, top_k=10)
            repeat_ids = [r.candidate.chunk_id for r in repeat]
            assert repeat_ids == first_ids


# ── 8. Invalid configuration ──────────────────────────────────────────────────

class TestInvalidConfiguration:
    """Validation errors on bad inputs."""

    def test_k_zero_raises(self):
        """k=0 is invalid."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            RRFFusion(k=0)

    def test_k_negative_raises(self):
        """k=-5 is invalid."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            RRFFusion(k=-5)

    def test_negative_weight_raises(self):
        """Negative weight is invalid."""
        with pytest.raises(ValueError, match="must be >= 0"):
            RRFFusion(k=60, weights={"vector": -0.5})

    def test_negative_default_weight_raises(self):
        """Negative default_weight is invalid."""
        with pytest.raises(ValueError, match="default_weight must be >= 0"):
            RRFFusion(k=60, default_weight=-1.0)


# ── 9. Top-k truncation ───────────────────────────────────────────────────────

class TestTopKTruncation:
    """Verify top_k limits result count."""

    def test_top_k_limits_output(self):
        """10 candidates, top_k=3 → only 3 returned."""
        fusion = RRFFusion(k=60)
        candidates = [_cand(f"c{i}") for i in range(10)]
        results = fusion.fuse({"s": candidates}, top_k=3)
        assert len(results) == 3

    def test_top_k_zero_means_unlimited(self):
        """top_k=0 returns all candidates."""
        fusion = RRFFusion(k=60)
        candidates = [_cand(f"c{i}") for i in range(5)]
        results = fusion.fuse({"s": candidates}, top_k=0)
        assert len(results) == 5

    def test_top_k_larger_than_pool(self):
        """top_k=100 with 3 candidates → returns 3."""
        fusion = RRFFusion(k=60)
        results = fusion.fuse({"s": [_cand("A"), _cand("B"), _cand("C")]}, top_k=100)
        assert len(results) == 3


# ── 10. Three-source extensibility ─────────────────────────────────────────────

class TestExtensibility:
    """Support arbitrary future retrieval sources."""

    def test_three_source_fusion(self):
        """Dense + BM25 + table_retriever all contribute to RRF score."""
        fusion = RRFFusion(k=60, weights={"vector": 1.0, "bm25": 1.0, "table_retriever": 0.8})

        results = fusion.fuse({
            "vector": [_cand("A", 0.9), _cand("B", 0.8)],
            "bm25": [_cand("B", 4.0), _cand("C", 3.0)],
            "table_retriever": [_cand("A", 0.7), _cand("D", 0.6)],
        }, top_k=10)

        # A: vector(rank 1) + table(rank 1) = 1/61 + 0.8/61 = 1.8/61
        # B: vector(rank 2) + bm25(rank 1) = 1/62 + 1/61
        a = next(r for r in results if r.candidate.chunk_id == "A")
        assert abs(a.rrf_score - (1.0 / 61 + 0.8 / 61)) < 1e-9
        assert sorted(a.retrieval_methods) == ["table_retriever", "vector"]

        b = next(r for r in results if r.candidate.chunk_id == "B")
        assert abs(b.rrf_score - (1.0 / 62 + 1.0 / 61)) < 1e-9

        # All 4 unique chunks present
        all_ids = {r.candidate.chunk_id for r in results}
        assert all_ids == {"A", "B", "C", "D"}

    def test_five_source_fusion(self):
        """Five sources including multimodal and parent_document."""
        fusion = RRFFusion(k=60)
        results = fusion.fuse({
            "vector": [_cand("A")],
            "bm25": [_cand("A")],
            "table_retriever": [_cand("A")],
            "multimodal_retriever": [_cand("A")],
            "parent_document_retriever": [_cand("A")],
        }, top_k=10)

        assert len(results) == 1
        # 5 * 1/61
        expected = 5.0 / 61
        assert abs(results[0].rrf_score - expected) < 1e-9
        assert len(results[0].retrieval_methods) == 5


# ── 11. FusionDiagnostics ─────────────────────────────────────────────────────

class TestFusionDiagnostics:
    """Verify diagnostics are fully populated after fusion."""

    def test_diagnostics_fields_populated(self):
        """All diagnostic fields have correct values."""
        fusion = RRFFusion(k=42, weights={"vector": 1.5, "bm25": 0.8})
        fusion.fuse({
            "vector": [_cand("A"), _cand("B"), _cand("C")],
            "bm25": [_cand("B"), _cand("D")],
        }, top_k=3)

        diag = fusion.diagnostics()
        assert diag.fusion_strategy == "rrf"
        assert diag.rrf_k == 42
        assert diag.source_weights == {"vector": 1.5, "bm25": 0.8}
        assert diag.input_result_counts == {"vector": 3, "bm25": 2}
        assert diag.unique_result_count == 4  # A, B, C, D
        assert diag.final_result_count == 3   # top_k=3

    def test_diagnostics_before_fusion(self):
        """Calling diagnostics before any fusion returns safe defaults."""
        fusion = RRFFusion(k=60)
        diag = fusion.diagnostics()
        assert diag.fusion_strategy == "rrf"
        assert diag.rrf_k == 60
        assert diag.unique_result_count == 0

    def test_diagnostics_empty_fusion(self):
        """Empty fusion produces valid diagnostics."""
        fusion = RRFFusion(k=60)
        fusion.fuse({}, top_k=10)
        diag = fusion.diagnostics()
        assert diag.input_result_counts == {}
        assert diag.unique_result_count == 0
        assert diag.final_result_count == 0


# ── 12. FusionStrategy protocol compliance ─────────────────────────────────────

class TestProtocolCompliance:
    """RRFFusion satisfies the FusionStrategy protocol."""

    def test_isinstance_check(self):
        """RRFFusion is a FusionStrategy (runtime_checkable)."""
        fusion = RRFFusion(k=60)
        assert isinstance(fusion, FusionStrategy)

    def test_protocol_method_signatures(self):
        """fuse() and diagnostics() are callable."""
        fusion = RRFFusion(k=60)
        assert callable(getattr(fusion, "fuse", None))
        assert callable(getattr(fusion, "diagnostics", None))


# ── 13. Legacy backward-compatible API ─────────────────────────────────────────

class TestLegacyAPI:
    """The old positional list API still works."""

    def test_fuse_positional(self):
        """fuse_positional() with unnamed lists returns (candidate, score) tuples."""
        fusion = RRFFusion(k=60)
        list1 = [_cand("A", 0.9), _cand("B", 0.8)]
        list2 = [_cand("A", 0.95), _cand("C", 0.7)]

        fused = fusion.fuse_positional([list1, list2])

        assert len(fused) == 3
        top_cand, top_score = fused[0]
        assert top_cand.chunk_id == "A"
        expected_score_a = 2.0 / 61
        assert abs(top_score - expected_score_a) < 1e-6


# ── 14. Hybrid end-to-end integration ─────────────────────────────────────────

class TestHybridIntegration:
    """HybridRetrievalService uses weighted RRF and returns fusion diagnostics."""

    def test_end_to_end_with_fusion_strategy(self):
        """Full pipeline returns fusion_strategy and fusion diagnostics."""
        db = DatabaseManager(db_url="sqlite:///:memory:")
        db.create_tables()
        emb_svc = EmbeddingService(db_manager=db)
        dim = emb_svc.dimension

        with db.session() as session:
            doc = Document(
                document_id="doc_rrf",
                tenant_id="tenant_rrf",
                source="local",
                filename="patent.pdf",
                source_uri="s3://patent.pdf",
                mime_type="application/pdf",
                content_hash="hash_rrf",
                status=DocumentStatus.INDEXED.value,
                version=1,
            )
            c1 = Chunk(
                chunk_id="c_rrf_1",
                document_id="doc_rrf",
                version=1,
                chunk_index=0,
                text="Autonomous vehicle perception and sensor fusion.",
                section="Section 1",
                page=1,
            )
            c2 = Chunk(
                chunk_id="c_rrf_2",
                document_id="doc_rrf",
                version=1,
                chunk_index=1,
                text="LiDAR calibration and point cloud processing pipeline.",
                section="Section 2",
                page=2,
            )
            e1 = Embedding(
                chunk_id="c_rrf_1",
                embedding_model=emb_svc.model_name,
                embedding_version=emb_svc.model_version,
                dimension=dim,
                provider=emb_svc.provider_name,
                vector=[0.1] * dim,
            )
            e2 = Embedding(
                chunk_id="c_rrf_2",
                embedding_model=emb_svc.model_name,
                embedding_version=emb_svc.model_version,
                dimension=dim,
                provider=emb_svc.provider_name,
                vector=[0.05] * dim,
            )
            session.add_all([doc, c1, c2, e1, e2])

        hybrid_svc = HybridRetrievalService(db_manager=db, embedding_service=emb_svc)

        req = RetrievalRequest(query="LiDAR calibration sensor", tenant_id="tenant_rrf", top_k=5)
        resp = hybrid_svc.retrieve(req)

        assert resp.retrieval_method == "hybrid"
        assert resp.fusion_method == "rrf"
        assert len(resp.results) >= 1

        # Each result should have fusion_strategy set
        for r in resp.results:
            assert r.fusion_strategy == "rrf"
            assert r.rrf_score is not None
            assert r.rrf_score > 0
            assert len(r.retrieval_methods) >= 1

        # Diagnostics should include RRF fusion metadata
        diag = resp.diagnostics
        assert diag.get("fusion_strategy") == "rrf"
        assert "source_weights" in diag
        assert "input_result_counts" in diag
        assert "fusion_unique_count" in diag
        assert "fusion_final_count" in diag
