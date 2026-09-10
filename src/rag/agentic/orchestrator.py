"""Agentic RAG orchestrator coordinating multi-agent planning, tool execution, and synthesis."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from rag.agentic.agents.critic import CriticAgent
from rag.agentic.agents.executor import ToolExecutorAgent
from rag.agentic.agents.final_answer import FinalAnswerAgent
from rag.agentic.agents.memory import MemoryAgent
from rag.agentic.agents.planner import PlannerAgent
from rag.agentic.agents.retriever_agent import RetrieverAgent
from rag.agentic.models import (
    AgenticState,
    ExecutionBudget,
    ExecutionStatus,
    PlanStepStatus,
)
from rag.agentic.safety import CancellationToken, SafetyController
from rag.agentic.tools.base import BaseTool
from rag.agentic.tools.calculator_tool import CalculatorTool
from rag.agentic.tools.metadata_tool import MetadataLookupTool
from rag.agentic.tools.retriever_tool import RetrieverTool
from rag.agentic.tools.table_tool import TableLookupTool

logger = logging.getLogger(__name__)


class AgenticRAGOrchestrator:
    """Production orchestrator for multi-step agentic RAG.

    Coordinates:
    - Planner Agent: creates decomposition plan
    - Retriever Agent: gathers relevant evidence
    - Tool Executor Agent: executes typed auxiliary tools with permission and duplicate controls
    - Critic Agent: evaluates evidence sufficiency and triggers query refinement
    - Final Answer Agent: synthesizes grounded answer with citations or abstains
    - Memory Agent: maintains scratchpad and historical observation context
    - Safety Controller: enforces hard budget limits, loop detection, and cancellation
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        budget: Optional[ExecutionBudget] = None,
        allowed_tools: Optional[List[str]] = None,
        retriever_tool: Optional[RetrieverTool] = None,
        tools: Optional[Dict[str, BaseTool]] = None,
        planner: Optional[PlannerAgent] = None,
        critic: Optional[CriticAgent] = None,
        final_agent: Optional[FinalAnswerAgent] = None,
        memory: Optional[MemoryAgent] = None,
    ):
        self._enabled = enabled
        self._budget = budget or ExecutionBudget()

        # Tools setup
        self._retriever_tool = retriever_tool or RetrieverTool()
        self._tools: Dict[str, BaseTool] = tools or {}
        if "retriever" not in self._tools:
            self._tools["retriever"] = self._retriever_tool
        if "calculator" not in self._tools:
            self._tools["calculator"] = CalculatorTool()
        if "table_lookup" not in self._tools:
            self._tools["table_lookup"] = TableLookupTool()
        if "metadata_lookup" not in self._tools:
            self._tools["metadata_lookup"] = MetadataLookupTool()

        self._safety = SafetyController(
            budget=self._budget,
            allowed_tools=allowed_tools or list(self._tools.keys()),
        )

        # Agents setup
        self._planner = planner or PlannerAgent()
        self._retriever_agent = RetrieverAgent(self._retriever_tool)
        self._executor = ToolExecutorAgent(self._tools, self._safety)
        self._critic = critic or CriticAgent()
        self._final_agent = final_agent or FinalAnswerAgent()
        self._memory = memory or MemoryAgent()

    def run(
        self,
        query: str,
        *,
        tenant_id: Optional[str] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AgenticState:
        """Execute the agentic RAG workflow on the user query."""
        start_time = time.time()
        state = AgenticState(
            run_id=str(uuid.uuid4()),
            query=query.strip(),
            remaining_budget=self._budget,
            status=ExecutionStatus.PLANNING,
        )

        # ── 1. Check if disabled ──────────────────────────────────────────────
        if not self._enabled:
            state.status = ExecutionStatus.COMPLETED
            state.final_answer = "Agentic RAG is currently disabled in configuration."
            state.record_trace("disabled_bypass", {"query": state.query})
            return state

        # ── 2. Planning Phase ─────────────────────────────────────────────────
        state.record_trace("planning_started", {"query": state.query})
        try:
            plan = self._planner.plan(state.query)
            state.plan = plan
            state.record_trace("plan_created", plan.model_dump())
        except Exception as exc:
            logger.error("Planner failed: %s", exc)
            state.errors.append(f"Planning failed: {exc}")
            state.status = ExecutionStatus.FAILED
            state.final_answer = "I was unable to construct an execution plan for this query."
            return state

        # ── 3. Simple Query Fast Path ─────────────────────────────────────────
        if plan.is_simple:
            state.status = ExecutionStatus.EXECUTING
            step = plan.steps[0]
            step.status = PlanStepStatus.RUNNING

            self._safety.record_step()
            self._safety.record_retrieval()
            state.current_step = 1

            self._retriever_agent.retrieve(state.query, state, top_k=3)
            step.status = PlanStepStatus.COMPLETED

            state.status = ExecutionStatus.COMPLETED
            state.final_answer = self._final_agent.generate_answer(state.query, state)
            state.record_trace("simple_execution_completed", {"evidence_count": len(state.retrieved_evidence)})
            return state

        # ── 4. Multi-Step Execution Loop ──────────────────────────────────────
        state.status = ExecutionStatus.EXECUTING

        for step in plan.steps:
            # Check safety bounds before step
            can_proceed, reason = self._safety.check_can_execute_step(cancellation_token)
            if not can_proceed:
                logger.warning("Agentic run halted: %s", reason)
                state.errors.append(reason)
                if cancellation_token and cancellation_token.is_cancelled:
                    state.status = ExecutionStatus.CANCELLED
                else:
                    state.status = ExecutionStatus.BUDGET_EXHAUSTED
                step.status = PlanStepStatus.SKIPPED
                break

            self._safety.record_step()
            state.current_step += 1
            step.status = PlanStepStatus.RUNNING
            state.record_trace("step_started", {"step_id": step.step_id, "tool": step.tool_name})

            # Execute step action
            if step.tool_name == "retriever":
                self._safety.record_retrieval()
                sub_q = step.tool_args.get("query", state.query)
                new_ev = self._retriever_agent.retrieve(sub_q, state)
                step.status = PlanStepStatus.COMPLETED
                step.result = f"Retrieved {len(new_ev)} chunks"
            else:
                call_rec = self._executor.execute(step.tool_name, step.tool_args, state)
                if call_rec.success:
                    step.status = PlanStepStatus.COMPLETED
                    step.result = call_rec.outputs
                else:
                    step.status = PlanStepStatus.FAILED
                    step.error = call_rec.error

            # ── Critic Evaluation ─────────────────────────────────────────────
            state.status = ExecutionStatus.EVALUATING
            eval_res = self._critic.evaluate(state.query, state)
            state.record_trace("critic_evaluation", eval_res.model_dump())

            if eval_res.can_short_circuit:
                logger.info("Critic confirmed evidence is sufficient at step %d; short-circuiting.", state.current_step)
                break

            # Handle knowledge gap / query refinement if suggested
            if not eval_res.is_sufficient and eval_res.suggested_query and self._safety.budget.can_rewrite():
                if self._safety.budget.can_retrieve():
                    self._safety.record_rewrite()
                    self._safety.record_retrieval()
                    logger.info("Critic suggested refinement: '%s'", eval_res.suggested_query)
                    state.record_trace("refinement_query", {"query": eval_res.suggested_query})
                    self._retriever_agent.retrieve(eval_res.suggested_query, state, top_k=2)

        # ── 5. Final Answer Synthesis ─────────────────────────────────────────
        if state.status not in (ExecutionStatus.CANCELLED, ExecutionStatus.BUDGET_EXHAUSTED):
            state.status = ExecutionStatus.COMPLETED

        final_ans = self._final_agent.generate_answer(state.query, state)
        state.final_answer = final_ans
        state.remaining_budget.elapsed_time_sec = round(time.time() - start_time, 2)
        state.record_trace("workflow_finished", {"status": state.status.value})

        return state
