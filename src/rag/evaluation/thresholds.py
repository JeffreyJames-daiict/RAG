"""Regression thresholds evaluator and quality gatekeeper for CI/CD pipelines."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from rag.evaluation.models import (
    AggregateEvaluationMetrics,
    EvaluationReport,
    EvaluationThresholds,
    ThresholdCheckResult,
)

logger = logging.getLogger(__name__)


class ThresholdEvaluator:
    """Evaluates evaluation aggregates against quality thresholds and baseline runs."""

    def __init__(self, default_thresholds: Optional[EvaluationThresholds] = None):
        self.thresholds = default_thresholds or EvaluationThresholds()

    def evaluate(
        self,
        metrics: AggregateEvaluationMetrics,
        thresholds: Optional[EvaluationThresholds] = None,
    ) -> ThresholdCheckResult:
        """Evaluate a single configuration's aggregate metrics against configured thresholds."""
        t = thresholds or self.thresholds
        failed: List[str] = []
        details: Dict[str, Dict[str, float]] = {}

        # 1. Recall@5
        details["recall_at_5"] = {"actual": metrics.mean_recall_at_5, "threshold": t.min_recall_at_5}
        if metrics.mean_recall_at_5 < t.min_recall_at_5:
            failed.append(
                f"Recall@5 ({metrics.mean_recall_at_5:.4f}) is below minimum threshold ({t.min_recall_at_5:.4f})"
            )

        # 2. MRR
        details["mrr"] = {"actual": metrics.mean_mrr, "threshold": t.min_mrr}
        if metrics.mean_mrr < t.min_mrr:
            failed.append(
                f"MRR ({metrics.mean_mrr:.4f}) is below minimum threshold ({t.min_mrr:.4f})"
            )

        # 3. NDCG@5
        details["ndcg_at_5"] = {"actual": metrics.mean_ndcg_at_5, "threshold": t.min_ndcg_at_5}
        if metrics.mean_ndcg_at_5 < t.min_ndcg_at_5:
            failed.append(
                f"NDCG@5 ({metrics.mean_ndcg_at_5:.4f}) is below minimum threshold ({t.min_ndcg_at_5:.4f})"
            )

        # 4. Citation validity
        details["citation_validity"] = {"actual": metrics.mean_citation_validity, "threshold": t.min_citation_validity}
        if metrics.mean_citation_validity < t.min_citation_validity:
            failed.append(
                f"Citation validity ({metrics.mean_citation_validity:.4f}) is below minimum threshold ({t.min_citation_validity:.4f})"
            )

        # 5. Unsupported citation rate
        details["unsupported_citation_rate"] = {
            "actual": metrics.mean_unsupported_citation_rate,
            "threshold": t.max_unsupported_citation_rate,
        }
        if metrics.mean_unsupported_citation_rate > t.max_unsupported_citation_rate:
            failed.append(
                f"Unsupported citation rate ({metrics.mean_unsupported_citation_rate:.4f}) exceeds maximum threshold ({t.max_unsupported_citation_rate:.4f})"
            )

        # 6. Mean Latency
        details["latency_ms"] = {"actual": metrics.mean_latency_ms, "threshold": t.max_mean_latency_ms}
        if metrics.mean_latency_ms > t.max_mean_latency_ms:
            failed.append(
                f"Mean latency ({metrics.mean_latency_ms:.1f}ms) exceeds maximum threshold ({t.max_mean_latency_ms:.1f}ms)"
            )

        # 7. Error rate
        details["error_rate"] = {"actual": metrics.error_rate, "threshold": t.max_error_rate}
        if metrics.error_rate > t.max_error_rate:
            failed.append(
                f"Case error rate ({metrics.error_rate:.4f}) exceeds maximum threshold ({t.max_error_rate:.4f})"
            )

        # 8. Abstention accuracy
        details["abstention_accuracy"] = {"actual": metrics.abstention_accuracy, "threshold": t.min_abstention_accuracy}
        if metrics.abstention_accuracy < t.min_abstention_accuracy:
            failed.append(
                f"Abstention accuracy ({metrics.abstention_accuracy:.4f}) is below minimum threshold ({t.min_abstention_accuracy:.4f})"
            )

        return ThresholdCheckResult(
            passed=(len(failed) == 0),
            failed_criteria=failed,
            details=details,
        )

    def evaluate_report(
        self,
        report: EvaluationReport,
        target_config: Optional[str] = None,
        thresholds: Optional[EvaluationThresholds] = None,
    ) -> EvaluationReport:
        """Apply thresholds to all or specified configurations in an evaluation report."""
        t = thresholds or self.thresholds
        overall_passed = True
        threshold_results: Dict[str, ThresholdCheckResult] = {}

        configs_to_check = [target_config] if target_config and target_config in report.configurations else list(report.configurations.keys())

        for cfg_name in configs_to_check:
            agg = report.configurations[cfg_name]
            check_res = self.evaluate(agg, thresholds=t)
            threshold_results[cfg_name] = check_res
            if not check_res.passed:
                overall_passed = False

        report.threshold_results = threshold_results
        report.overall_passed = overall_passed
        return report
