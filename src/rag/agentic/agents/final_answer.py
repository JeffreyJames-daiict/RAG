"""Final answer agent for synthesizing evidence into grounded answers."""

from __future__ import annotations

from typing import Any, List, Optional
from rag.agentic.models import AgenticState, ExecutionStatus


class FinalAnswerAgent:
    """Synthesizes verified evidence into a grounded final answer or generates explicit abstention."""

    def __init__(self, llm_caller: Optional[Any] = None):
        self._llm = llm_caller

    def generate_answer(self, query: str, state: AgenticState) -> str:
        """Generate a grounded response based on accumulated evidence and observations."""
        evidence = state.retrieved_evidence
        observations = state.observations

        # 1. Abstention if no evidence gathered
        if not evidence and not observations:
            return "I cannot answer this question based on the retrieved evidence."

        # 2. Check for calculation / tool results
        calc_results = [
            tr.outputs.get("formatted") or tr.outputs.get("result")
            for tr in state.tool_results
            if tr.success and tr.tool_name == "calculator" and isinstance(tr.outputs, dict)
        ]

        # 3. Assemble text evidence
        snippets: List[str] = []
        for idx, ev in enumerate(evidence, 1):
            source = f"[{ev.document_id}]" if ev.document_id else f"[C{idx}]"
            snippets.append(f"{source}: {ev.snippet}")

        joined_evidence = "\n\n".join(snippets)

        # 4. Generate grounded synthesis
        answer_parts: List[str] = []

        # If partial or budget exhausted, add notice
        if state.status in (ExecutionStatus.BUDGET_EXHAUSTED, ExecutionStatus.PARTIAL):
            answer_parts.append("*(Partial answer based on available evidence before resource budget was exhausted)*\n")

        # Synthesize main content
        if calc_results:
            answer_parts.append(f"Based on the analysis, the calculated result is **{calc_results[-1]}**.")

        if snippets:
            summary = "\n".join(f"- {s}" for s in snippets[:4])
            answer_parts.append(f"Key evidence from documents:\n{summary}")

        final_text = "\n\n".join(answer_parts)
        if not final_text.strip():
            return "I cannot answer this question based on the retrieved evidence."

        return final_text
