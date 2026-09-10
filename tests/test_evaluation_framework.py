"""Unit and integration tests for the RAG Evaluation Framework."""

import json
from pathlib import Path
import pytest

from rag.evaluation import (
    AggregateEvaluationMetrics,
    EvaluationCase,
    EvaluationConfigType,
    EvaluationReport,
    EvaluationRunner,
    EvaluationThresholds,
    GenerationMetricsCalculator,
    GoldenDataset,
    HtmlReporter,
    JsonReporter,
    CsvReporter,
    RetrievalMetricsCalculator,
    ThresholdEvaluator,
)
from rag.evaluation.run import main
from rag.retrieval.models import RetrievedChunkResult


@pytest.fixture
def sample_case() -> EvaluationCase:
    return EvaluationCase(
        case_id="test_case_1",
        tenant_id="tenant_1",
        query="What is the operating voltage?",
        query_type="factoid",
        expected_document_ids=["doc_spec"],
        expected_chunk_ids=["chunk_spec_voltage"],
        expected_answer="The operating voltage is 48V nominal.",
        expected_citations=["[C1]"],
        required_phrases=["48V", "operating voltage"],
    )


@pytest.fixture
def unanswerable_case() -> EvaluationCase:
    return EvaluationCase(
        case_id="test_case_unans",
        tenant_id="tenant_1",
        query="What is the warp drive speed?",
        query_type="unanswerable",
        expected_document_ids=[],
        expected_chunk_ids=[],
        is_unanswerable=True,
    )


# ---------------------------------------------------------------------------
# Test 1: Metric Calculations (Recall@k, MRR, NDCG, Precision, Hit Rate)
# ---------------------------------------------------------------------------
def test_retrieval_metric_calculations(sample_case: EvaluationCase):
    # Simulated retrieved results where target chunk is at rank 2
    chunks = [
        RetrievedChunkResult(chunk_id="chunk_unrelated", document_id="doc_other", text="Other info", rank=1),
        RetrievedChunkResult(chunk_id="chunk_spec_voltage", document_id="doc_spec", text="48V nominal voltage", rank=2),
        RetrievedChunkResult(chunk_id="chunk_misc", document_id="doc_spec", text="Misc notes", rank=3),
    ]

    metrics = RetrievalMetricsCalculator.calculate(
        case=sample_case,
        retrieved_results=chunks,
        latency_ms=45.2,
    )

    assert metrics.recall_at_k[1] == 0.0
    assert metrics.recall_at_k[3] == 1.0
    assert metrics.recall_at_k[5] == 1.0
    assert metrics.precision_at_k[3] == pytest.approx(1 / 3, rel=1e-2)
    assert metrics.hit_rate_at_k[1] == 0.0
    assert metrics.hit_rate_at_k[3] == 1.0
    assert metrics.mrr == 0.50  # 1 / 2
    assert metrics.ndcg_at_k[3] > 0.0
    assert metrics.exact_id_accuracy == 1.0
    assert metrics.parent_doc_accuracy == 1.0
    assert metrics.latency_ms == 45.2


# ---------------------------------------------------------------------------
# Test 2: Empty Result Sets
# ---------------------------------------------------------------------------
def test_empty_retrieval_results(sample_case: EvaluationCase):
    metrics = RetrievalMetricsCalculator.calculate(
        case=sample_case,
        retrieved_results=[],
        latency_ms=10.0,
    )

    assert metrics.recall_at_k[5] == 0.0
    assert metrics.mrr == 0.0
    assert metrics.ndcg_at_k[5] == 0.0
    assert metrics.exact_id_accuracy == 0.0
    assert metrics.parent_doc_accuracy == 0.0
    assert metrics.retrieved_count == 0


# ---------------------------------------------------------------------------
# Test 3: Unanswerable queries & Abstention correctness
# ---------------------------------------------------------------------------
def test_unanswerable_query_abstention(unanswerable_case: EvaluationCase):
    # Case A: Correctly abstained
    answer_abstained = "Based on the provided context, there is not enough information to answer."
    gen_metrics_abstained = GenerationMetricsCalculator.calculate(
        case=unanswerable_case,
        answer=answer_abstained,
        citations_used=[],
        valid_evidence_citations=set(),
    )
    assert gen_metrics_abstained.abstention_correctness is True

    # Case B: Hallucinated / failed to abstain
    answer_hallucinated = "Warp drive operates at Warp 9.975 in deep space [C1]."
    gen_metrics_hallucinated = GenerationMetricsCalculator.calculate(
        case=unanswerable_case,
        answer=answer_hallucinated,
        citations_used=["[C1]"],
        valid_evidence_citations=set(),
    )
    assert gen_metrics_hallucinated.abstention_correctness is False
    assert gen_metrics_hallucinated.unsupported_citation_rate == 1.0


# ---------------------------------------------------------------------------
# Test 4: Citation validation and phrase coverage
# ---------------------------------------------------------------------------
def test_generation_metrics_citations_and_phrases(sample_case: EvaluationCase):
    answer = "The operating voltage is measured at 48V nominal [C1]."
    gen_metrics = GenerationMetricsCalculator.calculate(
        case=sample_case,
        answer=answer,
        citations_used=["[C1]"],
        valid_evidence_citations={"[C1]"},
        input_tokens=150,
        output_tokens=25,
        estimated_cost_usd=0.0001,
    )

    assert gen_metrics.answer_presence is True
    assert gen_metrics.citation_validity == 1.0
    assert gen_metrics.unsupported_citation_rate == 0.0
    assert gen_metrics.citation_completeness == 1.0
    assert gen_metrics.phrase_coverage == 1.0
    assert gen_metrics.answer_coverage_f1 > 0.60


# ---------------------------------------------------------------------------
# Test 5: Table and Figure Accuracy
# ---------------------------------------------------------------------------
def test_table_and_figure_accuracy():
    table_case = EvaluationCase(
        case_id="c_tbl",
        query="Financial metrics in table 2",
        query_type="table",
        expected_chunk_ids=["tbl_2"],
    )
    chunk_table = RetrievedChunkResult(
        chunk_id="tbl_2",
        document_id="doc_fin",
        text="| Metric | Value |",
        table_id="tbl_2",
        chunk_type="table",
    )
    metrics_tbl = RetrievalMetricsCalculator.calculate(table_case, [chunk_table])
    assert metrics_tbl.table_accuracy == 1.0

    figure_case = EvaluationCase(
        case_id="c_fig",
        query="Hardware architecture diagram",
        query_type="figure",
        expected_chunk_ids=["fig_1"],
    )
    chunk_fig = RetrievedChunkResult(
        chunk_id="fig_1",
        document_id="doc_arch",
        text="Figure 1: SoC layout",
        asset_id="fig_1",
        chunk_type="asset",
    )
    metrics_fig = RetrievalMetricsCalculator.calculate(figure_case, [chunk_fig])
    assert metrics_fig.figure_accuracy == 1.0


# ---------------------------------------------------------------------------
# Test 6: Golden dataset loading from file
# ---------------------------------------------------------------------------
def test_golden_dataset_loading(tmp_path: Path):
    ds_file = tmp_path / "test_ds.json"
    ds = GoldenDataset(
        dataset_id="test_suite",
        cases=[
            EvaluationCase(case_id="1", query="Q1", expected_chunk_ids=["c1"]),
            EvaluationCase(case_id="2", query="Q2", expected_chunk_ids=["c2"]),
        ],
    )
    ds.to_file(ds_file)

    loaded = GoldenDataset.from_file(ds_file)
    assert loaded.dataset_id == "test_suite"
    assert len(loaded.cases) == 2
    assert loaded.cases[0].case_id == "1"


# ---------------------------------------------------------------------------
# Test 7: Regression thresholds evaluation (Pass vs Fail)
# ---------------------------------------------------------------------------
def test_threshold_evaluation_pass_and_fail():
    evaluator = ThresholdEvaluator(
        default_thresholds=EvaluationThresholds(
            min_recall_at_5=0.70,
            min_mrr=0.50,
            min_citation_validity=0.90,
        )
    )

    # Passing aggregates
    passing_agg = AggregateEvaluationMetrics(
        config_name="hybrid_reranked",
        mean_recall_at_5=0.85,
        mean_mrr=0.75,
        mean_ndcg_at_5=0.80,
        mean_citation_validity=0.95,
        mean_unsupported_citation_rate=0.05,
        abstention_accuracy=1.0,
        error_rate=0.0,
    )
    res_pass = evaluator.evaluate(passing_agg)
    assert res_pass.passed is True
    assert len(res_pass.failed_criteria) == 0

    # Failing aggregates (Recall@5 = 0.50 < 0.70)
    failing_agg = AggregateEvaluationMetrics(
        config_name="dense_only",
        mean_recall_at_5=0.50,
        mean_mrr=0.40,
        mean_ndcg_at_5=0.45,
        mean_citation_validity=0.80,
        mean_unsupported_citation_rate=0.20,
        abstention_accuracy=0.60,
        error_rate=0.10,
    )
    res_fail = evaluator.evaluate(failing_agg)
    assert res_fail.passed is False
    assert len(res_fail.failed_criteria) >= 3


# ---------------------------------------------------------------------------
# Test 8: Multi-Configuration Comparison with Simulated Pipeline
# ---------------------------------------------------------------------------
def test_runner_compare_configurations():
    def simulated_retriever(query: str, cfg: EvaluationConfigType, case: EvaluationCase) -> list[RetrievedChunkResult]:
        # Simulate incremental improvements across configurations
        target = case.expected_chunk_ids[0] if case.expected_chunk_ids else "c_fallback"
        if cfg == EvaluationConfigType.DENSE_ONLY:
            # Rank 3
            return [
                RetrievedChunkResult(chunk_id="other_1", document_id="d1", text="text1", rank=1),
                RetrievedChunkResult(chunk_id="other_2", document_id="d2", text="text2", rank=2),
                RetrievedChunkResult(chunk_id=target, document_id="d_target", text="target text", rank=3),
            ]
        elif cfg == EvaluationConfigType.BM25_ONLY:
            # Rank 2
            return [
                RetrievedChunkResult(chunk_id="other_1", document_id="d1", text="text1", rank=1),
                RetrievedChunkResult(chunk_id=target, document_id="d_target", text="target text", rank=2),
            ]
        else:
            # Hybrid / RRF / Reranked places target at rank 1!
            return [
                RetrievedChunkResult(chunk_id=target, document_id="d_target", text="target text", rank=1),
                RetrievedChunkResult(chunk_id="other_1", document_id="d1", text="text1", rank=2),
            ]

    runner = EvaluationRunner(simulated_retriever=simulated_retriever)
    ds = GoldenDataset(
        dataset_id="test_compare",
        cases=[
            EvaluationCase(case_id="c1", query="q1", expected_chunk_ids=["target_1"]),
            EvaluationCase(case_id="c2", query="q2", expected_chunk_ids=["target_2"]),
        ],
    )

    report = runner.compare_configurations(
        dataset=ds,
        configurations=[
            EvaluationConfigType.DENSE_ONLY,
            EvaluationConfigType.BM25_ONLY,
            EvaluationConfigType.HYBRID_RRF,
        ],
    )

    assert len(report.configurations) == 3
    assert report.configurations["dense_only"].mean_mrr == pytest.approx(1 / 3, rel=1e-2)
    assert report.configurations["bm25_only"].mean_mrr == 0.50
    assert report.configurations["hybrid_rrf"].mean_mrr == 1.0  # Rank 1!


# ---------------------------------------------------------------------------
# Test 9: Reporters (JSON, CSV, HTML)
# ---------------------------------------------------------------------------
def test_reporters(tmp_path: Path):
    report = EvaluationReport(
        run_id="run_123",
        timestamp="2026-09-10T12:00:00Z",
        dataset_id="ds_test",
        total_cases=1,
        configurations={
            "hybrid_rrf": AggregateEvaluationMetrics(
                config_name="hybrid_rrf",
                total_cases=1,
                successful_cases=1,
                mean_recall_at_5=1.0,
                mean_mrr=1.0,
            )
        },
        case_results=[],
    )

    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    html_path = tmp_path / "report.html"

    JsonReporter.save(report, json_path)
    CsvReporter.save(report, csv_path)
    HtmlReporter.save(report, html_path)

    assert json_path.exists()
    assert csv_path.exists()
    assert html_path.exists()

    with open(json_path) as f:
        data = json.load(f)
        assert data["run_id"] == "run_123"

    with open(csv_path) as f:
        content = f.read()
        assert "hybrid_rrf" in content

    with open(html_path) as f:
        html_content = f.read()
        assert "RAG Evaluation Benchmark Report" in html_content
        assert "hybrid_rrf" in html_content


# ---------------------------------------------------------------------------
# Test 10: CLI execution end-to-end
# ---------------------------------------------------------------------------
def test_cli_execution_with_fixture(tmp_path: Path):
    golden_fixture = Path("tests/evaluation/golden_dataset.json")
    assert golden_fixture.exists()

    out_file = tmp_path / "cli_report.json"
    html_out = tmp_path / "cli_report.html"

    # Run CLI
    exit_code = main([
        "--dataset", str(golden_fixture),
        "--config", "hybrid_reranked",
        "--output", str(out_file),
        "--html-output", str(html_out),
    ])

    assert exit_code == 0
    assert out_file.exists()
    assert html_out.exists()

    with open(out_file) as f:
        rep_data = json.load(f)
        assert "hybrid_reranked" in rep_data["configurations"]
        assert rep_data["total_cases"] == 5


# ---------------------------------------------------------------------------
# Test 11: CLI Regression Failure Non-Zero Exit Code
# ---------------------------------------------------------------------------
def test_cli_regression_failure_exit_code(tmp_path: Path):
    golden_fixture = Path("tests/evaluation/golden_dataset.json")
    thresholds_file = tmp_path / "strict_thresholds.json"
    with open(thresholds_file, "w") as f:
        json.dump({"min_recall_at_5": 0.999, "min_mrr": 0.999}, f)

    out_file = tmp_path / "fail_report.json"

    exit_code = main([
        "--dataset", str(golden_fixture),
        "--config", "hybrid",
        "--output", str(out_file),
        "--thresholds", str(thresholds_file),
        "--fail-on-regression",
    ])

    assert exit_code == 1

