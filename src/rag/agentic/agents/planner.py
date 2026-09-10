"""Planner agent for decomposing queries into executable plans."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from rag.agentic.models import ExecutionPlan, PlanStep, PlanStepStatus


class PlannerAgent:
    """Generates an initial execution plan or detects single-pass simple queries."""

    def __init__(self, llm_caller: Optional[Any] = None):
        self._llm = llm_caller

    def plan(self, query: str) -> ExecutionPlan:
        """Analyze the query and create a structured execution plan."""
        stripped = query.strip()

        # 1. Detect if query is simple (single retrieval needed)
        # Non-complex queries without comparison, multi-part questions, or math
        is_complex = (
            " and " in stripped.lower()
            or " versus " in stripped.lower()
            or " vs " in stripped.lower()
            or "compare" in stripped.lower()
            or "which then" in stripped.lower()
            or "calculate" in stripped.lower()
            or "difference between" in stripped.lower()
            or stripped.count("?") > 1
        )

        if not is_complex:
            # Simple query -> Single-step plan
            return ExecutionPlan(
                goal=stripped,
                is_simple=True,
                steps=[
                    PlanStep(
                        step_id=1,
                        description=f"Retrieve relevant evidence for: {stripped}",
                        tool_name="retriever",
                        tool_args={"query": stripped, "top_k": 3},
                        status=PlanStepStatus.PENDING,
                    )
                ],
            )

        # 2. Decompose multi-step query into atomic sub-steps
        steps: List[PlanStep] = []

        # Check for calculation / math need
        has_math = any(w in stripped.lower() for w in ("calculate", "sum", "total", "percentage", "difference between"))

        # Check for comparison
        if "compare" in stripped.lower() or " vs " in stripped.lower() or " versus " in stripped.lower():
            # Example: "Compare X vs Y"
            parts = re.split(r"\s+(?:vs\.?|versus|and)\s+", stripped, flags=re.IGNORECASE)
            sub1 = parts[0].replace("compare", "").strip() if len(parts) > 0 else stripped
            sub2 = parts[1].strip() if len(parts) > 1 else stripped

            steps.append(
                PlanStep(
                    step_id=1,
                    description=f"Retrieve information about {sub1}",
                    tool_name="retriever",
                    tool_args={"query": sub1, "top_k": 3},
                )
            )
            steps.append(
                PlanStep(
                    step_id=2,
                    description=f"Retrieve information about {sub2}",
                    tool_name="retriever",
                    tool_args={"query": sub2, "top_k": 3},
                )
            )
        elif " which then " in stripped.lower() or " then " in stripped.lower():
            # Chained causal reasoning: "How does A affect B which then affects C?"
            parts = re.split(r"\s+(?:which then affects?|and then|then)\s+", stripped, flags=re.IGNORECASE)
            for idx, part in enumerate(parts, 1):
                clean_part = part.strip()
                steps.append(
                    PlanStep(
                        step_id=idx,
                        description=f"Investigate: {clean_part}",
                        tool_name="retriever",
                        tool_args={"query": clean_part, "top_k": 3},
                    )
                )
        else:
            # Standard multi-hop decomposition
            steps.append(
                PlanStep(
                    step_id=1,
                    description=f"Retrieve background and initial facts for: {stripped}",
                    tool_name="retriever",
                    tool_args={"query": stripped, "top_k": 3},
                )
            )
            if has_math:
                steps.append(
                    PlanStep(
                        step_id=2,
                        description="Compute quantitative metric based on retrieved data",
                        tool_name="calculator",
                        tool_args={"expression": "0"},  # to be updated dynamically by orchestrator
                    )
                )

        return ExecutionPlan(
            goal=stripped,
            is_simple=False,
            steps=steps,
        )
