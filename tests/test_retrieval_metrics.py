"""Unit tests for Information Retrieval Quality Metrics (Precision, Recall, MRR, NDCG)."""

import pytest

from rag.retrieval.metrics import RetrievalEvaluator, ScoreDistribution


def test_precision_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc9"}

    # Precision@3: doc1 and doc3 are relevant -> 2 / 3 ≈ 0.6667
    assert abs(RetrievalEvaluator.precision_at_k(retrieved, relevant, k=3) - 0.6667) < 1e-3
    # Precision@5: doc1 and doc3 -> 2 / 5 = 0.4
    assert abs(RetrievalEvaluator.precision_at_k(retrieved, relevant, k=5) - 0.4000) < 1e-3
    # No matches
    assert RetrievalEvaluator.precision_at_k(retrieved, {"other"}, k=5) == 0.0


def test_recall_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc9", "doc10"}  # 4 relevant in total

    # Recall@3: retrieved has doc1 and doc3 (2 out of 4) -> 0.5
    assert abs(RetrievalEvaluator.recall_at_k(retrieved, relevant, k=3) - 0.5000) < 1e-3
    # Recall@5: still 2 out of 4 -> 0.5
    assert abs(RetrievalEvaluator.recall_at_k(retrieved, relevant, k=5) - 0.5000) < 1e-3
    # Empty relevant
    assert RetrievalEvaluator.recall_at_k(retrieved, set(), k=5) == 0.0


def test_mrr():
    # First relevant at rank 1 -> MRR = 1.0
    assert RetrievalEvaluator.mrr(["d1", "d2"], {"d1"}) == 1.0

    # First relevant at rank 2 -> MRR = 0.5
    assert RetrievalEvaluator.mrr(["d1", "d2", "d3"], {"d2"}) == 0.5

    # First relevant at rank 4 -> MRR = 0.25
    assert RetrievalEvaluator.mrr(["d1", "d2", "d3", "d4"], {"d4"}) == 0.25

    # None relevant -> MRR = 0.0
    assert RetrievalEvaluator.mrr(["d1", "d2"], {"d9"}) == 0.0


def test_ndcg_at_k():
    # Perfect ranking: all relevant items in top positions
    retrieved_perfect = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc1", "doc2"}

    # Perfect ranking should have NDCG = 1.0
    assert RetrievalEvaluator.ndcg_at_k(retrieved_perfect, relevant, k=2) == 1.0

    # Degraded ranking: relevant items at ranks 3 and 4
    retrieved_imperfect = ["doc3", "doc4", "doc1", "doc2"]
    ndcg_imperfect = RetrievalEvaluator.ndcg_at_k(retrieved_imperfect, relevant, k=4)
    assert 0.0 < ndcg_imperfect < 1.0


def test_score_distribution():
    scores = [0.1, 0.5, 0.9]
    dist = ScoreDistribution.compute(scores)

    assert dist.count == 3
    assert abs(dist.mean - 0.5) < 1e-4
    assert abs(dist.min - 0.1) < 1e-4
    assert abs(dist.max - 0.9) < 1e-4
    assert dist.stddev > 0.0

    # Empty
    empty_dist = ScoreDistribution.compute([])
    assert empty_dist.count == 0


def test_evaluate_batch():
    results = {
        "q1": ["d1", "d2", "d3"],
        "q2": ["d4", "d5", "d6"],
    }
    ground_truth = {
        "q1": {"d1"},
        "q2": {"d5"},
    }

    metrics = RetrievalEvaluator.evaluate_batch(results, ground_truth, k=3)
    assert "precision@3" in metrics
    assert "recall@3" in metrics
    assert "mrr" in metrics
    # q1 MRR=1.0, q2 MRR=0.5 -> average MRR = 0.75
    assert abs(metrics["mrr"] - 0.75) < 1e-3
