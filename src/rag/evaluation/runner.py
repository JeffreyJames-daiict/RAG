"""Evaluation Runner — executes single-case and dataset benchmarks across multiple RAG configurations."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from rag.context import ContextBuilder
from rag.evaluation.metrics import (
    GenerationMetricsCalculator,
    RetrievalMetricsCalculator,
)
from rag.evaluation.models import (
    AggregateEvaluationMetrics,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationReport,
    GoldenDataset,
)
from rag.generation.models import AnswerResponse
from rag.generation.service import LLMGenerationService
from rag.provenance.service import ProvenanceService
from rag.retrieval.hybrid_service import HybridRetrievalService
from rag.retrieval.models import RetrievalRequest, RetrievalResponse, RetrievedChunkResult

logger = logging.getLogger(__name__)


class EvaluationConfigType(str, Enum):
    """Supported pipeline configuration modes."""

    DENSE_ONLY = "dense_only"
    BM25_ONLY = "bm25_only"
    HYBRID = "hybrid"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RERANKED = "hybrid_reranked"
    FULL_RAG = "full_rag"
    ALL = "all"


class EvaluationRunner:
    """Benchmark orchestrator evaluating retrieval and end-to-end RAG across multiple configurations."""

    def __init__(
        self,
        hybrid_service: Optional[HybridRetrievalService] = None,
        context_builder: Optional[ContextBuilder] = None,
        generation_service: Optional[LLMGenerationService] = None,
        provenance_service: Optional[ProvenanceService] = None,
        simulated_retriever: Optional[Callable[[str, EvaluationConfigType, EvaluationCase], List[RetrievedChunkResult]]] = None,
    ):
        self.hybrid_service = hybrid_service
        self.context_builder = context_builder or ContextBuilder()
        self.generation_service = generation_service
        self.provenance_service = provenance_service or ProvenanceService()
        self.simulated_retriever = simulated_retriever

    def _execute_retrieval_stage(
        self,
        query: str,
        config_type: EvaluationConfigType,
        case: EvaluationCase,
        top_k: int = 10,
    ) -> Tuple[List[RetrievedChunkResult], float]:
        """Execute retrieval under the specified configuration, measuring latency."""
        start_time = time.perf_counter()

        # If a custom or simulated retriever is registered, use it
        if self.simulated_retriever is not None:
            results = self.simulated_retriever(query, config_type, case)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return results, latency_ms

        if not self.hybrid_service:
            # Fallback mock empty or metadata-derived chunks if no live service provided
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return [], latency_ms

        req = RetrievalRequest(
            query=query,
            tenant_id=case.tenant_id,
            top_k=top_k,
        )

        if config_type == EvaluationConfigType.DENSE_ONLY:
            resp = self.hybrid_service.vector_service.retrieve(req)
            results = resp.results

        elif config_type == EvaluationConfigType.BM25_ONLY:
            raw_bm25 = self.hybrid_service.bm25_retriever.retrieve(
                query=query,
                tenant_id=case.tenant_id,
                limit=top_k,
            )
            # Hydrate into RetrievedChunkResult objects
            results = [
                RetrievedChunkResult(
                    chunk_id=item["chunk_id"],
                    document_id=item.get("document_id", "doc_unknown"),
                    text=item.get("text", ""),
                    rank=idx,
                    similarity_score=float(item.get("score", 0.0)),
                    retrieval_method="bm25",
                )
                for idx, item in enumerate(raw_bm25, start=1)
            ]

        elif config_type == EvaluationConfigType.HYBRID:
            resp = self.hybrid_service.retrieve(req, fusion_method="linear")
            results = resp.results

        elif config_type == EvaluationConfigType.HYBRID_RRF:
            resp = self.hybrid_service.retrieve(req, fusion_method="rrf")
            results = resp.results

        elif config_type in (EvaluationConfigType.HYBRID_RERANKED, EvaluationConfigType.FULL_RAG):
            resp = self.hybrid_service.retrieve(req, fusion_method="rrf")
            results = resp.results

        else:
            resp = self.hybrid_service.retrieve(req)
            results = resp.results

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return results, latency_ms

    def evaluate_case(
        self,
        case: EvaluationCase,
        config_type: EvaluationConfigType,
        top_k: int = 10,
    ) -> EvaluationCaseResult:
        """Run evaluation on a single test case under the chosen configuration."""
        warnings: List[str] = []
        error: Optional[str] = None

        try:
            # 1. Execute retrieval
            chunks, ret_latency_ms = self._execute_retrieval_stage(
                query=case.query,
                config_type=config_type,
                case=case,
                top_k=top_k,
            )

            # 2. Compute retrieval metrics
            ret_metrics = RetrievalMetricsCalculator.calculate(
                case=case,
                retrieved_results=chunks,
                latency_ms=ret_latency_ms,
            )

            gen_metrics: Optional[Any] = None
            answer_text: Optional[str] = None
            citations_used: List[str] = []

            # 3. If FULL_RAG, execute context assembly, generation, and citation validation
            if config_type == EvaluationConfigType.FULL_RAG and self.generation_service:
                gen_start = time.perf_counter()

                # Build context package
                ctx_pkg = self.context_builder.build(case.query, chunks)

                # Generate answer
                gen_resp: AnswerResponse = self.generation_service.generate_answer(
                    query=case.query,
                    retrieval_source=ctx_pkg,
                )
                gen_latency_ms = (time.perf_counter() - gen_start) * 1000.0

                answer_text = gen_resp.answer
                citations_used = gen_resp.cited_markers

                # Validate citations against context package
                valid_cits = {c.citation_id for c in ctx_pkg.citations}
                gen_metrics = GenerationMetricsCalculator.calculate(
                    case=case,
                    answer=answer_text,
                    citations_used=citations_used,
                    valid_evidence_citations=valid_cits,
                    input_tokens=gen_resp.input_tokens,
                    output_tokens=gen_resp.output_tokens,
                    estimated_cost_usd=gen_resp.estimated_cost_usd,
                    latency_ms=gen_latency_ms,
                )
                warnings.extend(gen_resp.warnings)

            return EvaluationCaseResult(
                case_id=case.case_id,
                query=case.query,
                config_name=config_type.value,
                retrieval_metrics=ret_metrics,
                generation_metrics=gen_metrics,
                retrieved_chunk_ids=[c.chunk_id for c in chunks],
                retrieved_document_ids=[c.document_id for c in chunks],
                generated_answer=answer_text,
                citations_used=citations_used,
                warnings=warnings,
                error=error,
            )

        except Exception as exc:
            logger.exception("Evaluation failed for case %s under config %s: %s", case.case_id, config_type, exc)
            return EvaluationCaseResult(
                case_id=case.case_id,
                query=case.query,
                config_name=config_type.value,
                retrieval_metrics=RetrievalMetricsCalculator.calculate(case, []),
                warnings=[str(exc)],
                error=str(exc),
            )

    def evaluate_dataset(
        self,
        dataset: GoldenDataset,
        config_type: EvaluationConfigType = EvaluationConfigType.HYBRID_RERANKED,
        top_k: int = 10,
    ) -> Tuple[AggregateEvaluationMetrics, List[EvaluationCaseResult]]:
        """Evaluate an entire golden dataset under a single configuration."""
        case_results: List[EvaluationCaseResult] = []
        for case in dataset.cases:
            res = self.evaluate_case(case, config_type=config_type, top_k=top_k)
            case_results.append(res)

        aggregate = self.aggregate_metrics(config_type.value, case_results)
        return aggregate, case_results

    def compare_configurations(
        self,
        dataset: GoldenDataset,
        configurations: Optional[Sequence[EvaluationConfigType]] = None,
        top_k: int = 10,
    ) -> EvaluationReport:
        """Benchmark and compare all configurations side-by-side."""
        run_id = f"eval_run_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        configs_to_run = (
            [
                EvaluationConfigType.DENSE_ONLY,
                EvaluationConfigType.BM25_ONLY,
                EvaluationConfigType.HYBRID,
                EvaluationConfigType.HYBRID_RRF,
                EvaluationConfigType.HYBRID_RERANKED,
                EvaluationConfigType.FULL_RAG,
            ]
            if not configurations or EvaluationConfigType.ALL in configurations
            else configurations
        )

        all_aggregates: Dict[str, AggregateEvaluationMetrics] = {}
        all_case_results: List[EvaluationCaseResult] = []

        for cfg in configs_to_run:
            agg, results = self.evaluate_dataset(dataset, config_type=cfg, top_k=top_k)
            all_aggregates[cfg.value] = agg
            all_case_results.extend(results)

        return EvaluationReport(
            run_id=run_id,
            timestamp=timestamp,
            dataset_id=dataset.dataset_id,
            total_cases=len(dataset.cases),
            configurations=all_aggregates,
            case_results=all_case_results,
        )

    @staticmethod
    def aggregate_metrics(config_name: str, case_results: Sequence[EvaluationCaseResult]) -> AggregateEvaluationMetrics:
        """Compute mean and percentile statistics across all case results."""
        total = len(case_results)
        if total == 0:
            return AggregateEvaluationMetrics(config_name=config_name)

        successful = sum(1 for r in case_results if r.error is None)
        error_rate = (total - successful) / total

        # Retrieval aggregates
        r1_list = [r.retrieval_metrics.recall_at_k.get(1, 0.0) for r in case_results]
        r3_list = [r.retrieval_metrics.recall_at_k.get(3, 0.0) for r in case_results]
        r5_list = [r.retrieval_metrics.recall_at_k.get(5, 0.0) for r in case_results]
        r10_list = [r.retrieval_metrics.recall_at_k.get(10, 0.0) for r in case_results]

        p5_list = [r.retrieval_metrics.precision_at_k.get(5, 0.0) for r in case_results]
        hit5_list = [r.retrieval_metrics.hit_rate_at_k.get(5, 0.0) for r in case_results]
        mrr_list = [r.retrieval_metrics.mrr for r in case_results]
        ndcg5_list = [r.retrieval_metrics.ndcg_at_k.get(5, 0.0) for r in case_results]

        exact_id_list = [r.retrieval_metrics.exact_id_accuracy for r in case_results]
        parent_doc_list = [r.retrieval_metrics.parent_doc_accuracy for r in case_results]

        table_scores = [r.retrieval_metrics.table_accuracy for r in case_results if r.retrieval_metrics.table_accuracy is not None]
        fig_scores = [r.retrieval_metrics.figure_accuracy for r in case_results if r.retrieval_metrics.figure_accuracy is not None]

        # Latencies
        latencies = sorted([r.retrieval_metrics.latency_ms for r in case_results])
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
        p50_lat = latencies[int(len(latencies) * 0.50)] if latencies else 0.0
        p95_lat = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

        # Generation aggregates
        gen_cases = [r.generation_metrics for r in case_results if r.generation_metrics is not None]
        if gen_cases:
            mean_cit_valid = sum(g.citation_validity for g in gen_cases) / len(gen_cases)
            mean_cit_comp = sum(g.citation_completeness for g in gen_cases) / len(gen_cases)
            mean_unsupp = sum(g.unsupported_citation_rate for g in gen_cases) / len(gen_cases)
            abst_acc = sum(1.0 for g in gen_cases if g.abstention_correctness) / len(gen_cases)
            mean_phrase = sum(g.phrase_coverage for g in gen_cases) / len(gen_cases)
            mean_f1 = sum(g.answer_coverage_f1 for g in gen_cases) / len(gen_cases)
            tot_tok = sum(g.input_tokens + g.output_tokens for g in gen_cases)
            tot_cost = sum(g.estimated_cost_usd for g in gen_cases)
        else:
            mean_cit_valid = 1.0
            mean_cit_comp = 1.0
            mean_unsupp = 0.0
            abst_acc = 1.0
            mean_phrase = 1.0
            mean_f1 = 1.0
            tot_tok = 0
            tot_cost = 0.0

        return AggregateEvaluationMetrics(
            config_name=config_name,
            total_cases=total,
            successful_cases=successful,
            error_rate=round(error_rate, 4),
            mean_recall_at_1=round(sum(r1_list) / total, 4),
            mean_recall_at_3=round(sum(r3_list) / total, 4),
            mean_recall_at_5=round(sum(r5_list) / total, 4),
            mean_recall_at_10=round(sum(r10_list) / total, 4),
            mean_precision_at_5=round(sum(p5_list) / total, 4),
            mean_hit_rate_at_5=round(sum(hit5_list) / total, 4),
            mean_mrr=round(sum(mrr_list) / total, 4),
            mean_ndcg_at_5=round(sum(ndcg5_list) / total, 4),
            mean_exact_id_accuracy=round(sum(exact_id_list) / total, 4),
            mean_table_accuracy=round(sum(table_scores) / len(table_scores), 4) if table_scores else None,
            mean_figure_accuracy=round(sum(fig_scores) / len(fig_scores), 4) if fig_scores else None,
            mean_parent_doc_accuracy=round(sum(parent_doc_list) / total, 4),
            mean_citation_validity=round(mean_cit_valid, 4),
            mean_citation_completeness=round(mean_cit_comp, 4),
            mean_unsupported_citation_rate=round(mean_unsupp, 4),
            abstention_accuracy=round(abst_acc, 4),
            mean_phrase_coverage=round(mean_phrase, 4),
            mean_answer_coverage_f1=round(mean_f1, 4),
            mean_latency_ms=round(mean_lat, 2),
            p50_latency_ms=round(p50_lat, 2),
            p95_latency_ms=round(p95_lat, 2),
            total_tokens=tot_tok,
            total_cost_usd=round(tot_cost, 6),
        )
