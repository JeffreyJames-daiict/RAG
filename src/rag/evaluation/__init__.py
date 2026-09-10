"""Production-Grade RAG Evaluation Framework."""

from rag.evaluation.metrics import (
    GenerationMetricsCalculator,
    LLMJudgeProvider,
    MockJudgeProvider,
    RetrievalMetricsCalculator,
)
from rag.evaluation.models import (
    AggregateEvaluationMetrics,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationReport,
    EvaluationThresholds,
    GenerationCaseMetrics,
    GoldenDataset,
    QueryType,
    RetrievalCaseMetrics,
    ThresholdCheckResult,
)
from rag.evaluation.reporters import (
    CsvReporter,
    HtmlReporter,
    JsonReporter,
)
from rag.evaluation.runner import (
    EvaluationConfigType,
    EvaluationRunner,
)
from rag.evaluation.thresholds import (
    ThresholdEvaluator,
)

__all__ = [
    "EvaluationCase",
    "GoldenDataset",
    "QueryType",
    "RetrievalCaseMetrics",
    "GenerationCaseMetrics",
    "EvaluationCaseResult",
    "AggregateEvaluationMetrics",
    "EvaluationThresholds",
    "ThresholdCheckResult",
    "EvaluationReport",
    "RetrievalMetricsCalculator",
    "GenerationMetricsCalculator",
    "LLMJudgeProvider",
    "MockJudgeProvider",
    "EvaluationConfigType",
    "EvaluationRunner",
    "JsonReporter",
    "CsvReporter",
    "HtmlReporter",
    "ThresholdEvaluator",
]
