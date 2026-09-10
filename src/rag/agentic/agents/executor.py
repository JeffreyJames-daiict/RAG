"""Tool executor agent for executing tools safely with input validation and caching."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional
from pydantic import ValidationError

from rag.agentic.models import AgenticState, Observation, ToolCallRecord
from rag.agentic.safety import SafetyController
from rag.agentic.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolExecutorAgent:
    """Executes tools with permission validation, input schema enforcement, and deduplication."""

    def __init__(self, tools: Dict[str, BaseTool], safety: SafetyController):
        self.tools = tools
        self.safety = safety

    def execute(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        state: AgenticState,
    ) -> ToolCallRecord:
        """Execute a tool safely and update agentic state with observation."""
        start_time = time.perf_counter()

        # 1. Permission check
        try:
            self.safety.check_tool_permission(tool_name)
        except PermissionError as exc:
            rec = ToolCallRecord(
                tool_name=tool_name,
                inputs=tool_args,
                success=False,
                error=str(exc),
                latency_ms=0.0,
            )
            state.tool_results.append(rec)
            state.errors.append(str(exc))
            return rec

        tool = self.tools.get(tool_name)
        if not tool:
            err = f"Tool '{tool_name}' not found in registered tools"
            rec = ToolCallRecord(
                tool_name=tool_name,
                inputs=tool_args,
                success=False,
                error=err,
                latency_ms=0.0,
            )
            state.tool_results.append(rec)
            state.errors.append(err)
            return rec

        # 2. Input validation
        try:
            typed_input = tool.input_schema(**tool_args)
        except ValidationError as exc:
            latency = (time.perf_counter() - start_time) * 1000
            err_msg = f"Invalid tool arguments for '{tool_name}': {exc}"
            rec = ToolCallRecord(
                tool_name=tool_name,
                inputs=tool_args,
                success=False,
                error=err_msg,
                latency_ms=round(latency, 2),
            )
            state.tool_results.append(rec)
            state.errors.append(err_msg)
            return rec

        # 3. Duplicate prevention check
        call_hash = tool.call_hash(typed_input)
        cached = self.safety.get_cached_tool_output(call_hash)
        if cached is not None:
            latency = (time.perf_counter() - start_time) * 1000
            rec = ToolCallRecord(
                tool_name=tool_name,
                inputs=tool_args,
                outputs=cached.model_dump() if hasattr(cached, "model_dump") else cached,
                success=True,
                latency_ms=round(latency, 2),
            )
            state.tool_results.append(rec)
            state.observations.append(
                Observation(
                    step=state.current_step,
                    action=f"{tool_name}(cached)",
                    content=f"Reused previous result for {tool_name}",
                )
            )
            return rec

        # 4. Check action loop
        if self.safety.check_and_record_action_loop(f"{tool_name}:{call_hash}"):
            err_msg = f"Action loop detected on tool call '{tool_name}'. Terminating repetitive branch."
            rec = ToolCallRecord(
                tool_name=tool_name,
                inputs=tool_args,
                success=False,
                error=err_msg,
                latency_ms=0.0,
            )
            state.tool_results.append(rec)
            state.errors.append(err_msg)
            return rec

        # 5. Execute tool
        try:
            output = tool.execute(typed_input)
            latency = (time.perf_counter() - start_time) * 1000
            self.safety.cache_tool_output(call_hash, output)
            self.safety.record_tool_call()

            rec = ToolCallRecord(
                tool_name=tool_name,
                inputs=tool_args,
                outputs=output.model_dump() if hasattr(output, "model_dump") else output,
                success=True,
                latency_ms=round(latency, 2),
            )
            state.tool_results.append(rec)
            state.observations.append(
                Observation(
                    step=state.current_step,
                    action=tool_name,
                    content=str(output),
                )
            )
            return rec

        except Exception as exc:
            latency = (time.perf_counter() - start_time) * 1000
            err_msg = f"Tool '{tool_name}' failed during execution: {exc}"
            logger.warning(err_msg)
            rec = ToolCallRecord(
                tool_name=tool_name,
                inputs=tool_args,
                success=False,
                error=err_msg,
                latency_ms=round(latency, 2),
            )
            state.tool_results.append(rec)
            state.errors.append(err_msg)
            return rec
