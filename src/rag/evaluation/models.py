"""Data models and schemas for the RAG Evaluation Framework."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Categorization of evaluation query types."""

    FACTOID = "factoid"
    TABLE = "table"
    FIGURE = "figure"
    PARENT_DOC = "parent_doc"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"


class EvaluationCase(BaseModel):
    """Single test case in a golden evaluation benchmark dataset."""

    case_id: str = Field(..., description="Unique case identifier")
    tenant_id: str = Field(default="default", description="Tenant organization key")
    query: str = Field(..., description="Evaluation query text")
    query_type: str = Field(default="factoid", description="Query type (factoid, table, figure, parent_doc, unanswerable)")
    expected_document_ids: List[str] = Field(default_factory=list, description="Target document IDs containing the answer")
    expected_chunk_ids: List[str] = Field(default_factory=list, description="Target chunk IDs containing the evidence")
    expected_answer: Optional[str] = Field(default=None, description="Ground truth answer text or key assertion")
    expected_citations: List[str] = Field(default_factory=list, description="Expected citation identifiers e.g. ['[C1]']")
    difficulty: str = Field(default="easy", description="Difficulty level: easy, medium, hard")
    language: str = Field(default="en", description="Query language code")
    is_unanswerable: bool = Field(default=False, description="Whether query intentionally lacks evidence in the corpus")
    required_phrases: List[str] = Field(default_factory=list, description="Key phrases or tokens that must appear in answer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom domain-specific metadata")


class GoldenDataset(BaseModel):
    """Container for a collection of evaluation cases."""

    dataset_id: str = Field(default="golden_dataset", description="Identifier for this benchmark dataset")
    version: str = Field(default="1.0", description="Dataset version string")
    description: str = Field(default="", description="Benchmark description")
    cases: List[EvaluationCase] = Field(default_factory=list, description="List of evaluation cases")

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> GoldenDataset:
        """Load golden dataset from a JSON file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Golden dataset file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return cls(cases=[EvaluationCase(**item) for item in data])
        return cls(**data)

    def to_file(self, file_path: Union[str, Path]) -> None:
        """Save golden dataset to a JSON file."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))


class RetrievalCaseMetrics(BaseModel):
    """Detailed Information Retrieval metrics for a single query."""

    recall_at_k: Dict[int, float] = Field(default_factory=dict, description="Recall at k (k=1, 3, 5, 10)")
    precision_at_k: Dict[int, float] = Field(default_factory=dict, description="Precision at k")
    hit_rate_at_k: Dict[int, float] = Field(default_factory=dict, description="Hit rate at k (1.0 if any hit, 0.0 else)")
    mrr: float = Field(default=0.0, description="Mean Reciprocal Rank")
    ndcg_at_k: Dict[int, float] = Field(default_factory=dict, description="NDCG at k")
    exact_id_accuracy: float = Field(default=0.0, description="1.0 if expected chunk_ids retrieved in top-k, else 0.0")
    table_accuracy: Optional[float] = Field(default=None, description="1.0 if expected table retrieved for table query")
    figure_accuracy: Optional[float] = Field(default=None, description="1.0 if expected figure/asset retrieved")
    parent_doc_accuracy: float = Field(default=0.0, description="1.0 if any chunk from expected_document_ids is retrieved")
    retrieved_count: int = Field(default=0, description="Total candidates returned")
    latency_ms: float = Field(default=0.0, description="Retrieval latency in milliseconds")


class GenerationCaseMetrics(BaseModel):
    """Deterministic, rule-based generation metrics for a single answer."""

    answer_presence: bool = Field(default=False, description="True if answer is non-empty and substantive")
    citation_validity: float = Field(default=1.0, description="Fraction of citations that belong to retrieved evidence")
    citation_completeness: float = Field(default=1.0, description="Fraction of expected citations present")
    unsupported_citation_rate: float = Field(default=0.0, description="Fraction of citations that are hallucinated/mismatched")
    abstention_correctness: bool = Field(default=True, description="True if abstained correctly on unanswerable query")
    phrase_coverage: float = Field(default=1.0, description="Fraction of required_phrases found in answer")
    answer_coverage_f1: float = Field(default=1.0, description="Token-level F1 overlap with expected_answer")
    input_tokens: int = Field(default=0, description="Prompt tokens used")
    output_tokens: int = Field(default=0, description="Completion tokens used")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated USD cost")
    latency_ms: float = Field(default=0.0, description="Generation latency in milliseconds")


class EvaluationCaseResult(BaseModel):
    """Evaluation result for an individual case under a specific pipeline configuration."""

    case_id: str
    query: str
    config_name: str
    retrieval_metrics: RetrievalCaseMetrics
    generation_metrics: Optional[GenerationCaseMetrics] = None
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    retrieved_document_ids: List[str] = Field(default_factory=list)
    generated_answer: Optional[str] = None
    citations_used: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class AggregateEvaluationMetrics(BaseModel):
    """Aggregated benchmark statistics across an entire dataset for a given configuration."""

    config_name: str
    total_cases: int = 0
    successful_cases: int = 0
    error_rate: float = 0.0

    # Retrieval aggregates
    mean_recall_at_1: float = 0.0
    mean_recall_at_3: float = 0.0
    mean_recall_at_5: float = 0.0
    mean_recall_at_10: float = 0.0
    mean_precision_at_5: float = 0.0
    mean_hit_rate_at_5: float = 0.0
    mean_mrr: float = 0.0
    mean_ndcg_at_5: float = 0.0
    mean_exact_id_accuracy: float = 0.0
    mean_table_accuracy: Optional[float] = None
    mean_figure_accuracy: Optional[float] = None
    mean_parent_doc_accuracy: float = 0.0

    # Generation aggregates
    mean_citation_validity: float = 1.0
    mean_citation_completeness: float = 1.0
    mean_unsupported_citation_rate: float = 0.0
    abstention_accuracy: float = 1.0
    mean_phrase_coverage: float = 1.0
    mean_answer_coverage_f1: float = 1.0

    # Operational telemetry
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0


class EvaluationThresholds(BaseModel):
    """Configurable quality gates for CI/CD regression detection."""

    min_recall_at_5: float = Field(default=0.70, description="Minimum acceptable Recall@5")
    min_mrr: float = Field(default=0.50, description="Minimum acceptable Mean Reciprocal Rank")
    min_ndcg_at_5: float = Field(default=0.50, description="Minimum acceptable NDCG@5")
    min_citation_validity: float = Field(default=0.85, description="Minimum acceptable citation validity ratio")
    max_unsupported_citation_rate: float = Field(default=0.15, description="Maximum allowable unsupported/hallucinated citation rate")
    max_mean_latency_ms: float = Field(default=5000.0, description="Maximum acceptable mean latency in ms")
    max_error_rate: float = Field(default=0.05, description="Maximum acceptable case failure rate")
    min_abstention_accuracy: float = Field(default=0.80, description="Minimum acceptable unanswerable abstention accuracy")


class ThresholdCheckResult(BaseModel):
    """Result of validating aggregated metrics against regression thresholds."""

    passed: bool = Field(default=True, description="True if all thresholds are satisfied")
    failed_criteria: List[str] = Field(default_factory=list, description="List of thresholds that breached")
    details: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Comparison dict with actual vs threshold values")


class EvaluationReport(BaseModel):
    """Top-level evaluation report containing benchmark runs, comparison tables, and CI/CD results."""

    run_id: str
    timestamp: str
    dataset_id: str
    total_cases: int
    configurations: Dict[str, AggregateEvaluationMetrics] = Field(default_factory=dict)
    case_results: List[EvaluationCaseResult] = Field(default_factory=list)
    threshold_results: Dict[str, ThresholdCheckResult] = Field(default_factory=dict)
    overall_passed: bool = True
