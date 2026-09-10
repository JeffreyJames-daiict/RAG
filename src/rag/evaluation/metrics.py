"""Deterministic metrics calculators for retrieval quality, citation verification, and answer evaluation."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple, runtime_checkable

from rag.evaluation.models import (
    EvaluationCase,
    GenerationCaseMetrics,
    QueryType,
    RetrievalCaseMetrics,
)
from rag.retrieval.models import RetrievedChunkResult


@runtime_checkable
class LLMJudgeProvider(Protocol):
    """Optional protocol for future LLM-based qualitative evaluation."""

    async def evaluate_answer(
        self,
        query: str,
        answer: str,
        context: str,
        expected_answer: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate generation quality on dimensions like faithfulness, correctness, and completeness."""
        ...


class MockJudgeProvider:
    """Mock judge provider for offline or unit testing."""

    async def evaluate_answer(
        self,
        query: str,
        answer: str,
        context: str,
        expected_answer: Optional[str] = None,
    ) -> Dict[str, float]:
        return {"faithfulness": 0.95, "correctness": 0.90, "relevance": 0.92}


class RetrievalMetricsCalculator:
    """Calculates deterministic Information Retrieval metrics against ground truth expectations."""

    DEFAULT_K_VALUES = (1, 3, 5, 10)

    @classmethod
    def calculate(
        cls,
        case: EvaluationCase,
        retrieved_results: Sequence[RetrievedChunkResult],
        latency_ms: float = 0.0,
        k_values: Sequence[int] = DEFAULT_K_VALUES,
    ) -> RetrievalCaseMetrics:
        """Compute all retrieval metrics for an individual evaluation case."""
        retrieved_chunk_ids = [r.chunk_id for r in retrieved_results]
        retrieved_doc_ids = [r.document_id for r in retrieved_results]

        expected_chunks = set(case.expected_chunk_ids)
        expected_docs = set(case.expected_document_ids)

        recall_map: Dict[int, float] = {}
        precision_map: Dict[int, float] = {}
        hit_map: Dict[int, float] = {}
        ndcg_map: Dict[int, float] = {}

        # If unanswerable case and no expected chunks, handling:
        for k in k_values:
            top_k_chunks = retrieved_chunk_ids[:k]

            if not expected_chunks:
                # If no chunks expected, recall is 1.0 if empty, else 0.0
                recall_map[k] = 1.0 if not top_k_chunks else 0.0
                precision_map[k] = 1.0 if not top_k_chunks else 0.0
                hit_map[k] = 0.0
                ndcg_map[k] = 1.0 if not top_k_chunks else 0.0
                continue

            hits = sum(1 for cid in top_k_chunks if cid in expected_chunks)
            recall_map[k] = round(hits / len(expected_chunks), 4)
            precision_map[k] = round(hits / max(1, len(top_k_chunks)), 4)
            hit_map[k] = 1.0 if hits > 0 else 0.0

            # Compute NDCG@k
            dcg = 0.0
            for rank_0, cid in enumerate(top_k_chunks):
                if cid in expected_chunks:
                    dcg += 1.0 / math.log2(rank_0 + 2)

            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(expected_chunks))))
            ndcg_map[k] = round(dcg / idcg, 4) if idcg > 0 else 0.0

        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        if expected_chunks:
            for rank_0, cid in enumerate(retrieved_chunk_ids):
                if cid in expected_chunks:
                    mrr = round(1.0 / (rank_0 + 1), 4)
                    break

        # Exact identifier retrieval accuracy (in top-5 or top-k)
        exact_id_acc = 0.0
        if expected_chunks:
            top_5 = set(retrieved_chunk_ids[:5])
            exact_id_acc = 1.0 if any(cid in top_5 for cid in expected_chunks) else 0.0

        # Parent-document retrieval accuracy
        parent_doc_acc = 0.0
        if expected_docs:
            top_5_docs = set(retrieved_doc_ids[:5])
            parent_doc_acc = 1.0 if any(did in top_5_docs for did in expected_docs) else 0.0
        elif expected_chunks:
            # If doc IDs were not explicitly specified, evaluate via retrieved results
            parent_doc_acc = exact_id_acc

        # Table retrieval accuracy
        table_acc: Optional[float] = None
        if case.query_type == QueryType.TABLE.value or "table" in case.query.lower():
            table_hits = any(
                r.table_id is not None
                or r.chunk_type == "table"
                or (expected_chunks and r.chunk_id in expected_chunks)
                for r in retrieved_results[:5]
            )
            table_acc = 1.0 if table_hits else 0.0

        # Figure retrieval accuracy
        figure_acc: Optional[float] = None
        if case.query_type == QueryType.FIGURE.value or any(w in case.query.lower() for w in ("diagram", "figure", "chart")):
            figure_hits = any(
                r.asset_id is not None
                or r.chunk_type == "asset"
                or (expected_chunks and r.chunk_id in expected_chunks)
                for r in retrieved_results[:5]
            )
            figure_acc = 1.0 if figure_hits else 0.0

        return RetrievalCaseMetrics(
            recall_at_k=recall_map,
            precision_at_k=precision_map,
            hit_rate_at_k=hit_map,
            mrr=mrr,
            ndcg_at_k=ndcg_map,
            exact_id_accuracy=exact_id_acc,
            table_accuracy=table_acc,
            figure_accuracy=figure_acc,
            parent_doc_accuracy=parent_doc_acc,
            retrieved_count=len(retrieved_results),
            latency_ms=round(latency_ms, 2),
        )


class GenerationMetricsCalculator:
    """Deterministic, rule-based calculator for generation quality, citation integrity, and grounding."""

    ABSTENTION_PATTERNS = [
        re.compile(r"(?i)\b(?:not\s+enough\s+information|insufficient\s+evidence|cannot\s+be\s+determined)\b"),
        re.compile(r"(?i)\b(?:not\s+provided\s+in\s+the\s+context|context\s+does\s+not\s+contain|do\s+not\s+have\s+enough)\b"),
        re.compile(r"(?i)\b(?:unable\s+to\s+answer|i\s+cannot\s+answer|no\s+mention\s+of)\b"),
    ]

    CIT_PATTERN = re.compile(r"\[C\d+\]", re.IGNORECASE)

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    @classmethod
    def calculate_token_f1(cls, prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 overlap between prediction and ground truth answer."""
        pred_tokens = cls._tokenize(prediction)
        gt_tokens = cls._tokenize(ground_truth)

        if not pred_tokens or not gt_tokens:
            return 1.0 if pred_tokens == gt_tokens else 0.0

        common = sum((min(pred_tokens.count(w), gt_tokens.count(w)) for w in set(pred_tokens)))
        if common == 0:
            return 0.0

        precision = common / len(pred_tokens)
        recall = common / len(gt_tokens)
        return round(2 * (precision * recall) / (precision + recall), 4)

    @classmethod
    def is_abstained(cls, answer: str) -> bool:
        """Check whether the answer indicates abstention due to lack of evidence."""
        if not answer or not answer.strip():
            return True
        for pat in cls.ABSTENTION_PATTERNS:
            if pat.search(answer):
                return True
        return False

    @classmethod
    def calculate(
        cls,
        case: EvaluationCase,
        answer: Optional[str],
        citations_used: Sequence[str],
        valid_evidence_citations: Set[str],
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
        latency_ms: float = 0.0,
    ) -> GenerationCaseMetrics:
        """Compute deterministic generation metrics for a single response."""
        ans = (answer or "").strip()
        answer_presence = len(ans) > 5

        # Citation validation
        norm_used = [c.upper() for c in citations_used]
        norm_valid = {c.upper() for c in valid_evidence_citations}

        valid_count = sum(1 for c in norm_used if c in norm_valid)
        citation_validity = (valid_count / len(norm_used)) if norm_used else 1.0

        unsupported_count = len(norm_used) - valid_count
        unsupported_rate = (unsupported_count / len(norm_used)) if norm_used else 0.0

        # Citation completeness against expected citations
        if case.expected_citations:
            norm_expected = {c.upper() for c in case.expected_citations}
            expected_hits = sum(1 for c in norm_expected if c in norm_used)
            citation_completeness = expected_hits / len(norm_expected)
        else:
            # If no specific expected citations, completeness is 1.0 if at least one citation used when answerable
            if not case.is_unanswerable and answer_presence:
                citation_completeness = 1.0 if len(norm_used) > 0 else 0.0
            else:
                citation_completeness = 1.0

        # Abstention correctness
        abstained = cls.is_abstained(ans)
        if case.is_unanswerable:
            abstention_correctness = abstained
        else:
            abstention_correctness = not abstained if answer_presence else False

        # Required phrase coverage
        if case.required_phrases:
            ans_lower = ans.lower()
            matched_phrases = sum(1 for phrase in case.required_phrases if phrase.lower() in ans_lower)
            phrase_coverage = round(matched_phrases / len(case.required_phrases), 4)
        else:
            phrase_coverage = 1.0

        # Answer coverage F1 against expected_answer
        if case.expected_answer:
            answer_coverage_f1 = cls.calculate_token_f1(ans, case.expected_answer)
        else:
            answer_coverage_f1 = 1.0 if answer_presence else 0.0

        return GenerationCaseMetrics(
            answer_presence=answer_presence,
            citation_validity=round(citation_validity, 4),
            citation_completeness=round(citation_completeness, 4),
            unsupported_citation_rate=round(unsupported_rate, 4),
            abstention_correctness=abstention_correctness,
            phrase_coverage=phrase_coverage,
            answer_coverage_f1=answer_coverage_f1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=round(estimated_cost_usd, 6),
            latency_ms=round(latency_ms, 2),
        )
