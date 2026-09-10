"""Memory agent for maintaining scratchpad notes and execution context."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from rag.agentic.models import AgenticState


class MemoryAgent:
    """Maintains scratchpad notes, deduplicates evidence, and formats context for reasoning."""

    def __init__(self):
        self._scratchpad: List[str] = []

    def add_note(self, note: str) -> None:
        """Append a note to the working scratchpad."""
        if note not in self._scratchpad:
            self._scratchpad.append(note)

    def get_context_summary(self, state: AgenticState) -> str:
        """Format a condensed summary of evidence and observations for LLM context."""
        parts: List[str] = []

        if self._scratchpad:
            parts.append("Scratchpad Notes:\n" + "\n".join(f"- {n}" for n in self._scratchpad))

        if state.observations:
            obs_text = "\n".join(f"Step {o.step}: [{o.action}] {o.content[:150]}" for o in state.observations[-3:])
            parts.append("Recent Observations:\n" + obs_text)

        return "\n\n".join(parts)
