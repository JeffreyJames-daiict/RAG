"""Retriever tool for agentic workflows."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from rag.agentic.models import RetrievedEvidence
from rag.agentic.tools.base import BaseTool


class RetrieverInput(BaseModel):
    """Input arguments for retriever tool."""

    query: str = Field(..., description="Target search query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of results to retrieve")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata filters")


class RetrieverOutput(BaseModel):
    """Output results from retriever tool."""

    evidence: List[RetrievedEvidence] = Field(default_factory=list, description="Retrieved evidence snippets")
    count: int = Field(default=0, description="Number of items retrieved")


class RetrieverTool(BaseTool):
    """Tool wrapping semantic and lexical retrieval."""

    name = "retriever"
    description = "Retrieves relevant passages, tables, and documents for a search query."
    input_schema = RetrieverInput
    output_schema = RetrieverOutput

    def __init__(
        self,
        retriever_fn: Optional[Callable[[str, int, Dict[str, Any]], List[RetrievedEvidence]]] = None,
        corpus: Optional[Dict[str, str]] = None,
    ):
        self._retriever_fn = retriever_fn
        self._corpus = corpus or {}

    def execute(self, inputs: RetrieverInput) -> RetrieverOutput:
        """Execute retrieval query."""
        if self._retriever_fn is not None:
            results = self._retriever_fn(inputs.query, inputs.top_k, inputs.filters)
            return RetrieverOutput(evidence=results, count=len(results))

        # Default keyword-matching fallback over in-memory corpus
        evidence: List[RetrievedEvidence] = []
        terms = [w.lower() for w in inputs.query.split() if len(w) > 2]

        for doc_id, text in self._corpus.items():
            matched_terms = sum(1 for term in terms if term in text.lower())
            if matched_terms > 0 or not terms:
                score = matched_terms / max(len(terms), 1)
                evidence.append(
                    RetrievedEvidence(
                        query=inputs.query,
                        snippet=text,
                        document_id=doc_id,
                        score=round(score, 3),
                    )
                )

        evidence.sort(key=lambda x: x.score, reverse=True)
        selected = evidence[: inputs.top_k]
        return RetrieverOutput(evidence=selected, count=len(selected))
