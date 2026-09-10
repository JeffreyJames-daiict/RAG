"""Safety controller and guardrails for agentic RAG orchestration."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set

from rag.agentic.models import ExecutionBudget

logger = logging.getLogger(__name__)


class CancellationToken:
    """Cooperative cancellation token for interrupting long-running agentic runs."""

    def __init__(self):
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled


class SafetyController:
    """Enforces execution budgets, loop detection, duplicate prevention, and tool permissions."""

    def __init__(
        self,
        budget: ExecutionBudget,
        allowed_tools: Optional[List[str]] = None,
        loop_window_size: int = 6,
        max_repeated_actions: int = 2,
    ):
        self.budget = budget
        self.allowed_tools: Set[str] = set(
            allowed_tools or ["retriever", "calculator", "table_lookup", "metadata_lookup"]
        )
        self.loop_window_size = loop_window_size
        self.max_repeated_actions = max_repeated_actions

        self._action_history: deque[str] = deque(maxlen=loop_window_size)
        self._tool_cache: Dict[str, Any] = {}
        self._start_time: float = time.time()

    def update_elapsed_time(self) -> None:
        """Update elapsed execution time."""
        self.budget.elapsed_time_sec = time.time() - self._start_time

    def check_can_execute_step(self, cancellation_token: Optional[CancellationToken] = None) -> Tuple[bool, Optional[str]]:
        """Validate whether the orchestrator can proceed with the next execution step."""
        self.update_elapsed_time()

        if cancellation_token and cancellation_token.is_cancelled:
            return False, "Execution was cancelled by client"

        if self.budget.is_timed_out():
            return False, f"Execution time limit exceeded ({self.budget.elapsed_time_sec:.2f}s >= {self.budget.max_time_sec}s)"

        if not self.budget.can_step():
            return False, f"Maximum steps limit reached ({self.budget.used_steps} >= {self.budget.max_steps})"

        if not self.budget.can_consume_cost():
            return False, f"Cost budget limit reached (${self.budget.used_cost:.4f} >= ${self.budget.max_cost:.4f})"

        return True, None

    def check_tool_permission(self, tool_name: str) -> None:
        """Verify that the tool is in the allowed permission set."""
        if tool_name not in self.allowed_tools:
            raise PermissionError(
                f"Tool '{tool_name}' is not permitted. Allowed tools: {sorted(self.allowed_tools)}"
            )

    def check_and_record_action_loop(self, action_signature: str) -> bool:
        """Detect action loops using cycle detection in recent action signatures.

        Returns True if an execution loop was detected.
        """
        self._action_history.append(action_signature)

        # Check for immediate repetition (A, A, A)
        if len(self._action_history) >= self.max_repeated_actions:
            last_actions = list(self._action_history)[-self.max_repeated_actions:]
            if len(set(last_actions)) == 1:
                logger.warning("Immediate action repetition detected: %s", action_signature)
                return True

        # Check for 2-step oscillation (A, B, A, B)
        if len(self._action_history) >= 4:
            hist = list(self._action_history)
            if hist[-4] == hist[-2] and hist[-3] == hist[-1] and hist[-4] != hist[-3]:
                logger.warning("2-step oscillatory action loop detected: %s <-> %s", hist[-2], hist[-1])
                return True

        return False

    def get_cached_tool_output(self, call_hash: str) -> Optional[Any]:
        """Retrieve output of an identical tool call if already executed in this run."""
        return self._tool_cache.get(call_hash)

    def cache_tool_output(self, call_hash: str, output: Any) -> None:
        """Cache tool output by unique call hash to prevent redundant execution."""
        self._tool_cache[call_hash] = output

    def consume_tokens_and_cost(self, tokens: int = 0, cost_usd: float = 0.0) -> None:
        """Record consumed tokens and USD cost."""
        self.budget.used_tokens += tokens
        self.budget.used_cost += cost_usd

    def record_retrieval(self) -> None:
        """Record a retrieval call count."""
        self.budget.used_retrievals += 1

    def record_tool_call(self) -> None:
        """Record a general tool call count."""
        self.budget.used_tools += 1

    def record_step(self) -> None:
        """Record an execution step count."""
        self.budget.used_steps += 1

    def record_rewrite(self) -> None:
        """Record a query rewrite / refinement count."""
        self.budget.used_rewrites += 1
