"""Base abstractions for typed tools in agentic RAG."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from pydantic import BaseModel


class BaseTool(ABC):
    """Abstract base class for all agentic tools with typed I/O and idempotency hashing."""

    name: str
    description: str
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]

    def call_hash(self, inputs: BaseModel) -> str:
        """Compute a deterministic hash of tool name and inputs for duplicate detection."""
        dumped = json.dumps(inputs.model_dump(), sort_keys=True, default=str)
        payload = f"{self.name}:{dumped}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @abstractmethod
    def execute(self, inputs: BaseModel) -> BaseModel:
        """Execute the tool with validated inputs and return typed output."""
        ...
