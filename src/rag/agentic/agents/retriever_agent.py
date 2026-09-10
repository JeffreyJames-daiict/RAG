"""Retriever agent for managing evidence acquisition."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from rag.agentic.models import AgenticState, RetrievedEvidence
from rag.agentic.tools.retriever_tool import RetrieverInput, RetrieverTool

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """Agent responsible for formulating search calls and ingesting retrieved evidence."""

    def __init__(self, retriever_tool: RetrieverTool):
        self.tool = retriever_tool

    def retrieve(
        self,
        query: str,
        state: AgenticState,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedEvidence]:
        """Execute a retrieval query and accumulate unique evidence in the agentic state."""
        tool_input = RetrieverInput(query=query, top_k=top_k, filters=filters or {})
        try:
            output = self.tool.execute(tool_input)
        except Exception as exc:
            logger.warning("Retrieval failed for query '%s': %s", query, exc)
            state.errors.append(f"Retrieval error: {exc}")
            return []

        # Ingest and deduplicate evidence against state.retrieved_evidence
        existing_snippets = {e.snippet.strip().lower() for e in state.retrieved_evidence}
        new_evidence: List[RetrievedEvidence] = []

        for item in output.evidence:
            if item.snippet.strip().lower() not in existing_snippets:
                state.retrieved_evidence.append(item)
                new_evidence.append(item)
                existing_snippets.add(item.snippet.strip().lower())

        return new_evidence
