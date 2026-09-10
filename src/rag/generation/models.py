"""Pydantic models and data structures for LLM Answer Generation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GenerationContextChunk(BaseModel):
    """Normalized context passage supplied to the LLM generator."""

    rank: int = Field(..., description="1-based rank of the retrieved chunk")
    text: str = Field(..., description="Cleaned content text of the chunk")
    chunk_id: str = Field(default="", description="Unique chunk ID")
    filename: Optional[str] = Field(default="", description="Source document filename")
    page: Optional[int] = Field(default=None, description="Page number where chunk originated")
    section: Optional[str] = Field(default=None, description="Section heading breadcrumbs")
    chunk_type: str = Field(default="text", description="Type of chunk (text, table, figure)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw metadata")


class GenerationRequest(BaseModel):
    """User request for LLM Answer Synthesis."""

    query: str = Field(..., min_length=1, description="Natural language user query")
    context_chunks: List[GenerationContextChunk] = Field(
        default_factory=list,
        description="Ranked context passages for grounding the answer",
    )
    primary_model: Optional[str] = Field(
        default=None,
        description="Preferred OpenRouter model identifier. If omitted, uses default roster.",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for deterministic generation",
    )
    max_tokens: int = Field(
        default=1500,
        ge=64,
        le=8192,
        description="Max token budget for response",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional override for the grounding system prompt",
    )


class ModelAttemptRecord(BaseModel):
    """Telemetry record for each candidate model evaluated."""

    model: str
    success: bool
    error: Optional[str] = None
    retries_attempted: int = 0
    duration_ms: float = 0.0


class GenerationResponse(BaseModel):
    """Structured result returned by the LLM generation service."""

    query: str
    answer: str
    model_used: str
    fallback_occurred: bool = False
    attempted_models: List[ModelAttemptRecord] = Field(default_factory=list)
    cited_ranks: List[int] = Field(
        default_factory=list,
        description="List of context chunk ranks explicitly referenced with [N] citations",
    )
    latency_ms: float = 0.0
    total_tokens: Optional[int] = None
    raw_response: Optional[Dict[str, Any]] = None
