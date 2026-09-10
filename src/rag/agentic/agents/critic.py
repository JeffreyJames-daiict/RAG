"""Critic and evidence evaluation agent."""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field

from rag.agentic.models import AgenticState


class EvaluationResult(BaseModel):
    """Result of evidence sufficiency evaluation by CriticAgent."""

    is_sufficient: bool = Field(..., description="Whether accumulated evidence is sufficient to answer the query")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in sufficiency assessment")
    missing_information: List[str] = Field(default_factory=list, description="Specific gaps in knowledge")
    suggested_query: Optional[str] = Field(default=None, description="Refined query to gather missing information")
    can_short_circuit: bool = Field(default=False, description="True if evidence is fully sufficient to answer early")


class CriticAgent:
    """Evaluates accumulated evidence against the user query to determine if more steps are needed."""

    def __init__(self, llm_caller: Optional[Any] = None):
        self._llm = llm_caller

    def evaluate(self, query: str, state: AgenticState) -> EvaluationResult:
        """Evaluate evidence sufficiency using heuristic rules and evidence analysis."""
        evidence_count = len(state.retrieved_evidence)
        if evidence_count == 0:
            return EvaluationResult(
                is_sufficient=False,
                confidence=0.1,
                missing_information=["No evidence retrieved yet"],
                suggested_query=query,
                can_short_circuit=False,
            )

        # Extract keywords from query
        query_words = [w.lower() for w in query.split() if len(w) > 3]
        all_snippets_text = " ".join(e.snippet.lower() for e in state.retrieved_evidence)

        # Check coverage of query words
        covered_words = [w for w in query_words if w in all_snippets_text]
        coverage_ratio = len(covered_words) / max(len(query_words), 1)

        # If high coverage (>= 0.7) and at least 1-2 chunks, evidence is sufficient
        if coverage_ratio >= 0.7 or evidence_count >= 3:
            return EvaluationResult(
                is_sufficient=True,
                confidence=round(min(1.0, 0.6 + coverage_ratio * 0.4), 2),
                missing_information=[],
                suggested_query=None,
                can_short_circuit=True,
            )

        # Some gaps remain
        missing_terms = [w for w in query_words if w not in all_snippets_text]
        suggested = f"{query} {' '.join(missing_terms[:3])}" if missing_terms else query

        return EvaluationResult(
            is_sufficient=False,
            confidence=round(coverage_ratio, 2),
            missing_information=[f"Missing information on: {', '.join(missing_terms)}"] if missing_terms else [],
            suggested_query=suggested,
            can_short_circuit=False,
        )
