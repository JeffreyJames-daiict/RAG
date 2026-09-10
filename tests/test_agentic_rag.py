"""Comprehensive test suite for the Agentic RAG Orchestration Layer.

Covers all required areas:
1. Simple query requiring no multi-step agentic behavior
2. Multi-step query with decomposition & execution
3. Retrieval failure handling & resilience
4. Tool failure handling without pipeline crash
5. Loop detection
6. Step limit enforcement & partial-result synthesis
7. Cost limit enforcement
8. Timeout enforcement
9. Invalid tool arguments schema validation
10. Unsupported / unanswerable questions
11. Final answer without sufficient evidence (grounded abstention)
12. Duplicate tool-call prevention
13. Tool permission control
14. Cancellation token support
15. Configuration integration
"""

import time
import pytest
from typing import Any, Dict, List

from rag.agentic import (
    AgenticRAGOrchestrator,
    AgenticState,
    BaseTool,
    CalculatorTool,
    CancellationToken,
    ExecutionBudget,
    ExecutionStatus,
    MetadataLookupTool,
    PlanStepStatus,
    RetrieverInput,
    RetrieverOutput,
    RetrieverTool,
    SafetyController,
    TableLookupTool,
)
from rag.agentic.models import RetrievedEvidence
from rag.core.config import AgenticRAGConfig, Settings


class FailingTool(BaseTool):
    """Tool that deliberately raises an unhandled exception."""

    name = "failing_tool"
    description = "A tool that always fails."
    input_schema = RetrieverInput
    output_schema = RetrieverOutput

    def execute(self, inputs: RetrieverInput) -> RetrieverOutput:
        raise RuntimeError("Fatal hardware failure during tool execution")


def mock_retriever_failure(query: str, top_k: int, filters: Dict[str, Any]) -> List[RetrievedEvidence]:
    """Mock retrieval function that throws an error."""
    raise ConnectionError("Retriever database connection timed out")


# ═══════════════════════════════════════════════════════════════════════════
# 1. SIMPLE QUERY FAST-PATH
# ═══════════════════════════════════════════════════════════════════════════

class TestSimpleQuery:
    def test_simple_query_requires_single_step(self):
        corpus = {"doc1": "Paris is the capital and largest city of France."}
        retriever = RetrieverTool(corpus=corpus)
        orchestrator = AgenticRAGOrchestrator(retriever_tool=retriever)

        state = orchestrator.run("What is the capital of France?")

        assert state.status == ExecutionStatus.COMPLETED
        assert state.plan is not None
        assert state.plan.is_simple is True
        assert state.current_step == 1
        assert len(state.retrieved_evidence) >= 1
        assert "Paris" in state.final_answer


# ═══════════════════════════════════════════════════════════════════════════
# 2. MULTI-STEP QUERY
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiStepQuery:
    def test_multi_step_comparison_query(self):
        corpus = {
            "doc_py": "Python is dynamically typed and uses GIL for thread management.",
            "doc_rs": "Rust provides memory safety without garbage collection through ownership.",
        }
        retriever = RetrieverTool(corpus=corpus)
        orchestrator = AgenticRAGOrchestrator(retriever_tool=retriever)

        query = "Compare Python vs Rust threading models"
        state = orchestrator.run(query)

        assert state.status == ExecutionStatus.COMPLETED
        assert state.plan is not None
        assert state.plan.is_simple is False
        assert len(state.plan.steps) >= 2
        assert state.current_step >= 2
        assert len(state.retrieved_evidence) >= 1
        assert "Python" in state.final_answer or "Rust" in state.final_answer


# ═══════════════════════════════════════════════════════════════════════════
# 3. RETRIEVAL FAILURE
# ═══════════════════════════════════════════════════════════════════════════

class TestRetrievalFailure:
    def test_retrieval_failure_recovers_without_crash(self):
        failing_retriever = RetrieverTool(retriever_fn=mock_retriever_failure)
        orchestrator = AgenticRAGOrchestrator(retriever_tool=failing_retriever)

        state = orchestrator.run("What is the quarterly revenue?")

        # Orchestrator does not crash, terminates cleanly and abstains
        assert state.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED)
        assert len(state.retrieved_evidence) == 0
        assert "cannot answer" in state.final_answer.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 4. TOOL FAILURE
# ═══════════════════════════════════════════════════════════════════════════

class TestToolFailure:
    def test_tool_failure_recorded_and_continues(self):
        failing_tool = FailingTool()
        orchestrator = AgenticRAGOrchestrator(
            tools={"failing_tool": failing_tool},
            allowed_tools=["retriever", "failing_tool"],
        )

        state = AgenticState(query="Test failure")
        rec = orchestrator._executor.execute("failing_tool", {"query": "test"}, state)

        assert rec.success is False
        assert "Fatal hardware failure" in rec.error
        assert len(state.errors) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 5. LOOP DETECTION
# ═══════════════════════════════════════════════════════════════════════════

class TestLoopDetection:
    def test_immediate_repetition_loop_detection(self):
        budget = ExecutionBudget()
        safety = SafetyController(budget=budget)

        assert safety.check_and_record_action_loop("retriever:query1") is False
        # Second identical action in immediate sequence triggers loop detection
        assert safety.check_and_record_action_loop("retriever:query1") is True

    def test_oscillating_action_loop_detection(self):
        budget = ExecutionBudget()
        safety = SafetyController(budget=budget)

        safety.check_and_record_action_loop("action_A")
        safety.check_and_record_action_loop("action_B")
        safety.check_and_record_action_loop("action_A")
        # 4th element matches A-B-A-B oscillation
        assert safety.check_and_record_action_loop("action_B") is True


# ═══════════════════════════════════════════════════════════════════════════
# 6. STEP LIMIT ENFORCEMENT & PARTIAL RESULTS
# ═══════════════════════════════════════════════════════════════════════════

class TestStepLimit:
    def test_step_limit_enforced_with_partial_result(self):
        corpus = {
            "d1": "Step 1 evidence regarding microchip supply chains.",
            "d2": "Step 2 evidence regarding automotive manufacturing.",
        }
        retriever = RetrieverTool(corpus=corpus)
        budget = ExecutionBudget(max_steps=1)  # Only 1 step permitted
        orchestrator = AgenticRAGOrchestrator(retriever_tool=retriever, budget=budget)

        query = "How does supply shortage affect manufacturing which then affects sales?"
        state = orchestrator.run(query)

        assert state.status == ExecutionStatus.BUDGET_EXHAUSTED
        assert state.current_step == 1
        assert any("Maximum steps limit reached" in err for err in state.errors)
        # Produces partial answer with available evidence
        assert state.final_answer is not None


# ═══════════════════════════════════════════════════════════════════════════
# 7. COST LIMIT ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════

class TestCostLimit:
    def test_cost_limit_enforced(self):
        budget = ExecutionBudget(max_cost=0.001)
        budget.used_cost = 0.002  # Already exceeded

        safety = SafetyController(budget=budget)
        can_proceed, reason = safety.check_can_execute_step()

        assert can_proceed is False
        assert "Cost budget limit reached" in reason


# ═══════════════════════════════════════════════════════════════════════════
# 8. TIMEOUT ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════

class TestTimeoutEnforcement:
    def test_timeout_stops_execution(self):
        budget = ExecutionBudget(max_time_sec=0.05)
        orchestrator = AgenticRAGOrchestrator(budget=budget)

        # Simulate sleep
        time.sleep(0.06)

        can_proceed, reason = orchestrator._safety.check_can_execute_step()
        assert can_proceed is False
        assert "Execution time limit exceeded" in reason


# ═══════════════════════════════════════════════════════════════════════════
# 9. INVALID TOOL ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════

class TestInvalidToolArguments:
    def test_calculator_invalid_expression(self):
        calc = CalculatorTool()
        with pytest.raises(ValueError, match="failed to evaluate"):
            calc.execute(calc.input_schema(expression="INVALID_SYNTAX ++ ** 99"))

    def test_invalid_tool_arguments_handled_in_executor(self):
        orchestrator = AgenticRAGOrchestrator()
        state = AgenticState(query="Calculate")

        # Missing required 'expression' field for calculator tool
        rec = orchestrator._executor.execute("calculator", {"wrong_field": 123}, state)

        assert rec.success is False
        assert "Invalid tool arguments" in rec.error


# ═══════════════════════════════════════════════════════════════════════════
# 10. UNSUPPORTED / UNANSWERABLE QUESTION
# ═══════════════════════════════════════════════════════════════════════════

class TestUnsupportedQuestion:
    def test_unsupported_question_abstains(self):
        # Empty corpus
        retriever = RetrieverTool(corpus={})
        orchestrator = AgenticRAGOrchestrator(retriever_tool=retriever)

        state = orchestrator.run("What will the stock price of Apple be on July 14, 2075?")

        assert "cannot answer" in state.final_answer.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 11. INSUFFICIENT EVIDENCE ABSTENTION
# ═══════════════════════════════════════════════════════════════════════════

class TestInsufficientEvidence:
    def test_insufficient_evidence_produces_abstention(self):
        orchestrator = AgenticRAGOrchestrator()
        state = AgenticState(query="What is quantum entanglement?")
        # No evidence in state
        answer = orchestrator._final_agent.generate_answer(state.query, state)

        assert "cannot answer this question based on the retrieved evidence" in answer


# ═══════════════════════════════════════════════════════════════════════════
# 12. DUPLICATE TOOL-CALL PREVENTION
# ═══════════════════════════════════════════════════════════════════════════

class TestDuplicateToolCalls:
    def test_duplicate_tool_call_is_memoized(self):
        calc = CalculatorTool()
        budget = ExecutionBudget()
        safety = SafetyController(budget=budget)
        orchestrator = AgenticRAGOrchestrator(tools={"calculator": calc})
        orchestrator._safety = safety
        orchestrator._executor = orchestrator._executor.__class__(orchestrator._tools, safety)

        state = AgenticState(query="Math")

        rec1 = orchestrator._executor.execute("calculator", {"expression": "10 * 5"}, state)
        assert rec1.success is True
        assert safety.budget.used_tools == 1

        # Identical second call
        rec2 = orchestrator._executor.execute("calculator", {"expression": "10 * 5"}, state)
        assert rec2.success is True
        # Tool call count should not increment because output was reused from cache
        assert safety.budget.used_tools == 1
        assert any("Reused previous result" in o.content for o in state.observations)


# ═══════════════════════════════════════════════════════════════════════════
# 13. TOOL PERMISSION CONTROL
# ═══════════════════════════════════════════════════════════════════════════

class TestToolPermissions:
    def test_unpermitted_tool_raises_permission_error(self):
        calc = CalculatorTool()
        budget = ExecutionBudget()
        # Only permit 'retriever', NOT 'calculator'
        safety = SafetyController(budget=budget, allowed_tools=["retriever"])
        orchestrator = AgenticRAGOrchestrator(tools={"calculator": calc})
        orchestrator._safety = safety
        orchestrator._executor = orchestrator._executor.__class__(orchestrator._tools, safety)

        state = AgenticState(query="Math")
        rec = orchestrator._executor.execute("calculator", {"expression": "5 + 5"}, state)

        assert rec.success is False
        assert "not permitted" in rec.error


# ═══════════════════════════════════════════════════════════════════════════
# 14. CANCELLATION TOKEN
# ═══════════════════════════════════════════════════════════════════════════

class TestCancellation:
    def test_cancellation_token_stops_workflow(self):
        token = CancellationToken()
        token.cancel()  # Cancel immediately

        orchestrator = AgenticRAGOrchestrator()
        query = "Compare X vs Y and then calculate metrics"
        state = orchestrator.run(query, cancellation_token=token)

        assert state.status == ExecutionStatus.CANCELLED
        assert any("cancelled" in err.lower() for err in state.errors)


# ═══════════════════════════════════════════════════════════════════════════
# 15. CONFIG INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigIntegration:
    def test_settings_has_agentic_config(self):
        settings = Settings()
        assert hasattr(settings, "agentic")
        assert isinstance(settings.agentic, AgenticRAGConfig)
        assert settings.agentic.enabled is False  # disabled by default
        assert settings.agentic.max_steps == 5
        assert "retriever" in settings.agentic.allowed_tools
        assert "calculator" in settings.agentic.allowed_tools
