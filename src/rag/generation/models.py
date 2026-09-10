"""Pydantic models and data structures for LLM Answer Generation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Standard message payload sent to an LLM provider."""

    role: str = Field(..., description="Message role: 'system', 'user', 'assistant'")
    content: str = Field(..., description="Message text content")


class LLMResponse(BaseModel):
    """Standardized response from an LLM provider."""

    text: str = Field(..., description="Generated text response")
    model: str = Field(..., description="Model name that generated the response")
    provider: str = Field(default="openrouter", description="Provider identifier")
    input_tokens: int = Field(default=0, description="Prompt token count")
    output_tokens: int = Field(default=0, description="Completion token count")
    total_tokens: int = Field(default=0, description="Sum of input and output tokens")
    cost_usd: float = Field(default=0.0, description="Estimated cost in USD")
    latency_ms: float = Field(default=0.0, description="Provider roundtrip latency in ms")
    finish_reason: Optional[str] = Field(default=None, description="Completion termination reason")
    raw: Optional[Dict[str, Any]] = Field(default=None, description="Raw provider response payload")


class LLMStreamChunk(BaseModel):
    """Streaming chunk event emitted during token-by-token generation."""

    delta: str = Field(default="", description="Token or text fragment delta")
    is_finished: bool = Field(default=False, description="True if stream has completed")
    model: str = Field(default="", description="Active model generating the stream")
    finish_reason: Optional[str] = Field(default=None, description="Stream termination reason")
    token_usage: Optional[Dict[str, int]] = Field(default=None, description="Final token usage metadata")
    error: Optional[str] = Field(default=None, description="Error message if stream failed")


class ValidationResult(BaseModel):
    """Result of answer validation and citation verification."""

    is_valid: bool = Field(default=True, description="True if response passes validation checks")
    clean_text: str = Field(default="", description="Sanitized and validated answer text")
    valid_citations: List[str] = Field(default_factory=list, description="Citations verified against ContextPackage")
    hallucinated_citations: List[str] = Field(
        default_factory=list,
        description="Citation markers found in answer that were not present in ContextPackage",
    )
    warnings: List[str] = Field(default_factory=list, description="Validation warnings or anomaly notices")


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
        default=2000,
        ge=64,
        le=16384,
        description="Max token budget for response",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional override for the grounding system prompt",
    )
    context_package: Optional[Any] = Field(
        default=None,
        description="Structured ContextPackage if built via ContextBuilder",
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
    provider_used: str = "openrouter"
    model_version: str = "v1.0"
    fallback_occurred: bool = False
    attempted_models: List[ModelAttemptRecord] = Field(default_factory=list)
    cited_ranks: List[int] = Field(
        default_factory=list,
        description="List of context chunk ranks explicitly referenced with [N] citations",
    )
    cited_markers: List[str] = Field(
        default_factory=list,
        description="List of citation tags found in text, e.g. ['[C1]', '[C2]']",
    )
    hallucinated_citations: List[str] = Field(
        default_factory=list,
        description="Citation markers referenced by LLM that do not exist in ContextPackage",
    )
    citations: List[Any] = Field(
        default_factory=list,
        description="Registry of citation objects associated with the context package",
    )
    context_package: Optional[Any] = Field(
        default=None,
        description="ContextPackage used for generation",
    )
    latency_ms: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost_usd: float = 0.0
    retry_count: int = 0
    warnings: List[str] = Field(default_factory=list)
    raw_response: Optional[Dict[str, Any]] = None


# Alias for explicit answer generation response terminology
AnswerResponse = GenerationResponse
