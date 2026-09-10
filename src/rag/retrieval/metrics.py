"""Retrieval Quality Metrics and Evaluation Suite.

Provides standard IR (Information Retrieval) metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Mean Average Precision (MAP@K)
- Score distribution analytics (mean, min, max, stddev)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Union


@dataclass
class ScoreDistribution:
    """Summary statistics for score distributions in retrieval results."""

    count: int = 0
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0
    stddev: float = 0.0

    @classmethod
    def compute(cls, scores: Sequence[float]) -> ScoreDistribution:
        """Compute distribution summary for a list of scores."""
        if not scores:
            return cls()

        n = len(scores)
        mean_val = sum(scores) / n
        min_val = min(scores)
        max_val = max(scores)

        if n > 1:
            variance = sum((x - mean_val) ** 2 for x in scores) / (n - 1)
            std_val = math.sqrt(max(0.0, variance))
        else:
            std_val = 0.0

        return cls(
            count=n,
            mean=round(mean_val, 4),
            min=round(min_val, 4),
            max=round(max_val, 4),
            stddev=round(std_val, 4),
        )


class RetrievalEvaluator:
    """Calculates Information Retrieval quality metrics given retrieved item IDs and ground truth."""

    @staticmethod
    def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
        """Precision@K: fraction of top-k retrieved items that are relevant."""
        if k <= 0:
            return 0.0
        top_k = retrieved[:k]
        if not top_k:
            return 0.0
        hits = sum(1 for item in top_k if item in relevant)
        return round(hits / k, 4)

    @staticmethod
    def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
        """Recall@K: fraction of all relevant items that are retrieved in top-k."""
        if not relevant or k <= 0:
            return 0.0
        top_k = retrieved[:k]
        hits = sum(1 for item in top_k if item in relevant)
        return round(hits / len(relevant), 4)

    @staticmethod
    def mrr(retrieved: Sequence[str], relevant: Set[str]) -> float:
        """Mean Reciprocal Rank (MRR): 1 / rank of the first relevant retrieved item."""
        for rank_0, item in enumerate(retrieved):
            if item in relevant:
                return round(1.0 / (rank_0 + 1), 4)
        return 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved: Sequence[str],
        relevant_relevance: Union[Set[str], Dict[str, float]],
        k: int,
    ) -> float:
        """Normalized Discounted Cumulative Gain at K (NDCG@K).

        Args:
            retrieved: Ordered list of retrieved item IDs.
            relevant_relevance: Set of relevant IDs (binary relevance 1.0) or
                                dict mapping item ID to graded relevance score.
            k: Evaluation depth.
        """
        if k <= 0:
            return 0.0

        top_k = retrieved[:k]
        if not top_k:
            return 0.0

        # Convert set to dict if binary relevance
        if isinstance(relevant_relevance, (set, list, tuple)):
            relevance_map = {item_id: 1.0 for item_id in relevant_relevance}
        else:
            relevance_map = relevant_relevance

        # Compute DCG@K
        dcg = 0.0
        for i, item in enumerate(top_k):
            rel = relevance_map.get(item, 0.0)
            dcg += rel / math.log2(i + 2)  # rank is i + 1, log2(rank + 1) = log2(i + 2)

        # Compute IDCG@K (Ideal DCG)
        ideal_relevances = sorted(relevance_map.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_relevances):
            idcg += rel / math.log2(i + 2)

        if idcg <= 0.0:
            return 0.0

        return round(dcg / idcg, 4)

    @staticmethod
    def average_precision(retrieved: Sequence[str], relevant: Set[str], k: Optional[int] = None) -> float:
        """Average Precision (AP) for a single query."""
        if not relevant:
            return 0.0

        items = retrieved[:k] if k is not None else retrieved
        hits = 0
        sum_precisions = 0.0

        for i, item in enumerate(items):
            if item in relevant:
                hits += 1
                sum_precisions += hits / (i + 1)

        if hits == 0:
            return 0.0
        return round(sum_precisions / min(len(relevant), len(items)), 4)

    @classmethod
    def evaluate_query(
        cls,
        retrieved: Sequence[str],
        relevant: Set[str],
        k: int = 5,
    ) -> Dict[str, float]:
        """Compute a full suite of quality metrics for a single query."""
        return {
            f"precision@{k}": cls.precision_at_k(retrieved, relevant, k),
            f"recall@{k}": cls.recall_at_k(retrieved, relevant, k),
            "mrr": cls.mrr(retrieved, relevant),
            f"ndcg@{k}": cls.ndcg_at_k(retrieved, relevant, k),
            "map": cls.average_precision(retrieved, relevant, k),
        }

    @classmethod
    def evaluate_batch(
        cls,
        results_by_query: Dict[str, Sequence[str]],
        ground_truth: Dict[str, Set[str]],
        k: int = 5,
    ) -> Dict[str, float]:
        """Compute aggregate quality metrics across a batch of queries."""
        metrics_accum: Dict[str, float] = {
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            "mrr": 0.0,
            f"ndcg@{k}": 0.0,
            "map": 0.0,
        }

        query_count = 0
        for q_id, retrieved in results_by_query.items():
            if q_id not in ground_truth:
                continue
            relevant = ground_truth[q_id]
            q_metrics = cls.evaluate_query(retrieved, relevant, k)
            for m_key, val in q_metrics.items():
                metrics_accum[m_key] += val
            query_count += 1

        if query_count == 0:
            return {k: 0.0 for k in metrics_accum}

        return {k: round(v / query_count, 4) for k, v in metrics_accum.items()}
