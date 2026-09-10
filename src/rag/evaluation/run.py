"""CLI entrypoint for running RAG evaluation benchmarks: python -m rag.evaluation.run."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rag.evaluation.models import (
    EvaluationReport,
    EvaluationThresholds,
    GoldenDataset,
)
from rag.evaluation.reporters import CsvReporter, HtmlReporter, JsonReporter
from rag.evaluation.runner import EvaluationConfigType, EvaluationRunner
from rag.evaluation.thresholds import ThresholdEvaluator

logger = logging.getLogger("rag.evaluation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG Pipeline Evaluation and Regression Testing Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Path to golden evaluation dataset (JSON format)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="all",
        choices=["dense_only", "bm25_only", "hybrid", "hybrid_rrf", "hybrid_reranked", "full_rag", "all"],
        help="Pipeline configuration to evaluate, or 'all' to benchmark and compare all 6 configurations",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="reports/evaluation_report.json",
        help="Output report file path (.json, .csv, or .html)",
    )
    parser.add_argument(
        "--html-output",
        type=str,
        default=None,
        help="Optional additional HTML dashboard output path",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        help="Number of candidate chunks to retrieve per query",
    )
    parser.add_argument(
        "--thresholds",
        "-t",
        type=str,
        default=None,
        help="Path to JSON file containing custom regression thresholds",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        default=False,
        help="Exit with non-zero exit code if evaluation thresholds fail",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed logging output",
    )
    return parser


def main(args: Optional[list[str]] = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)

    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 1. Load dataset
    dataset_path = Path(parsed.dataset)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset file not found: {dataset_path}", file=sys.stderr)
        return 2

    try:
        dataset = GoldenDataset.from_file(dataset_path)
        print(f"📊 Loaded {len(dataset.cases)} evaluation cases from '{dataset.dataset_id}'")
    except Exception as exc:
        print(f"❌ Error loading dataset: {exc}", file=sys.stderr)
        return 2

    # 2. Parse thresholds
    thresholds = EvaluationThresholds()
    if parsed.thresholds:
        t_path = Path(parsed.thresholds)
        if t_path.exists():
            with open(t_path, "r", encoding="utf-8") as f:
                t_dict = json.load(f)
            thresholds = EvaluationThresholds(**t_dict)
            print(f"⚙️ Loaded custom regression thresholds from {t_path}")

    # 3. Setup runner
    runner = EvaluationRunner()

    target_cfg = EvaluationConfigType(parsed.config)
    print(f"🚀 Running benchmark with configuration: '{target_cfg.value}' (top_k={parsed.top_k})...")

    if target_cfg == EvaluationConfigType.ALL:
        report: EvaluationReport = runner.compare_configurations(dataset=dataset, top_k=parsed.top_k)
    else:
        agg, results = runner.evaluate_dataset(dataset, config_type=target_cfg, top_k=parsed.top_k)
        report = EvaluationReport(
            run_id=f"eval_{dataset.dataset_id}",
            timestamp="",
            dataset_id=dataset.dataset_id,
            total_cases=len(dataset.cases),
            configurations={target_cfg.value: agg},
            case_results=results,
        )

    # 4. Check thresholds
    th_evaluator = ThresholdEvaluator(default_thresholds=thresholds)
    report = th_evaluator.evaluate_report(report)

    # 5. Save outputs
    out_path = Path(parsed.output)
    ext = out_path.suffix.lower()

    if ext == ".html":
        HtmlReporter.save(report, out_path)
    elif ext == ".csv":
        CsvReporter.save(report, out_path)
    else:
        JsonReporter.save(report, out_path)
    print(f"💾 Report saved to: {out_path}")

    if parsed.html_output:
        HtmlReporter.save(report, Path(parsed.html_output))
        print(f"🌐 HTML Dashboard saved to: {parsed.html_output}")

    # 6. Print summary
    print("\n" + "=" * 60)
    print(f"BENCHMARK SUMMARY (Run ID: {report.run_id})")
    print("=" * 60)
    for name, agg in report.configurations.items():
        pass_str = "✅ PASS" if report.threshold_results.get(name, {}).passed else "❌ FAIL"
        print(
            f"[{pass_str}] {name:<18} | Recall@5: {agg.mean_recall_at_5:.3f} | MRR: {agg.mean_mrr:.3f} "
            f"| NDCG@5: {agg.mean_ndcg_at_5:.3f} | Latency: {agg.mean_latency_ms:.1f}ms"
        )

    if not report.overall_passed:
        print("\n⚠️ Regression threshold failures:")
        for cfg_name, chk in report.threshold_results.items():
            if not chk.passed:
                for crit in chk.failed_criteria:
                    print(f"  - [{cfg_name}] {crit}")

    print("=" * 60)

    if parsed.fail_on_regression and not report.overall_passed:
        print("❌ Regression thresholds breached! Exiting with non-zero status.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
