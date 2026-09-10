"""Data models for the query rewriting and expansion layer."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Classified type of user query for strategy routing."""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    NAVIGATIONAL = "navigational"
    COMPARISON = "comparison"
    PROCEDURAL = "procedural"
    IDENTIFIER_LOOKUP = "identifier_lookup"
    CONVERSATIONAL = "conversational"
    UNANSWERABLE = "unanswerable"


class RewriteStrategyType(str, Enum):
    """Available query rewrite strategies."""

    ORIGINAL_ONLY = "original_only"
    EXPANSION = "expansion"
    MULTI_QUERY = "multi_query"
    KEYWORD_EXTRACTION = "keyword_extraction"
    IDENTIFIER_EXTRACTION = "identifier_extraction"
    HYDE = "hyde"
    DECOMPOSITION = "decomposition"
    CONVERSATION_AWARE = "conversation_aware"


class ConversationTurn(BaseModel):
    """Single turn in a conversation history for context-aware rewriting."""

    role: str = Field(..., description="Speaker role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class QueryAnalysis(BaseModel):
    """Result of deterministic query analysis before rewrite strategy selection."""

    original_query: str = Field(..., description="The unmodified original query")
    query_type: QueryType = Field(default=QueryType.FACTUAL, description="Classified query type")
    detected_identifiers: List[str] = Field(
        default_factory=list,
        description="Exact identifiers extracted (UUIDs, version strings, model names, file paths)",
    )
    detected_keywords: List[str] = Field(
        default_factory=list,
        description="Top-k domain keywords extracted from query",
    )
    language: str = Field(default="en", description="Detected language code")
    is_conversational: bool = Field(default=False, description="True if query appears to reference conversation history")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in type classification")
    suggested_strategies: List[RewriteStrategyType] = Field(
        default_factory=list,
        description="Ordered list of recommended rewrite strategies",
    )
    is_complex: bool = Field(default=False, description="True if query contains multiple sub-questions or conjunctions")
    word_count: int = Field(default=0, description="Word count of original query")


class RewrittenQuery(BaseModel):
    """A single rewritten or expanded query variant."""

    text: str = Field(..., description="Rewritten query text")
    strategy: RewriteStrategyType = Field(..., description="Strategy that produced this variant")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in the quality of this rewrite")
    is_original: bool = Field(default=False, description="True if this is the preserved original query")
    identifiers_preserved: bool = Field(default=True, description="True if all extracted identifiers are present")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Strategy-specific metadata")


class QueryRewriteResult(BaseModel):
    """Complete result of the query rewriting pipeline."""

    original_query: str = Field(..., description="The unmodified original query")
    rewritten_queries: List[RewrittenQuery] = Field(
        default_factory=list,
        description="All query variants including the original",
    )
    selected_strategy: RewriteStrategyType = Field(
        default=RewriteStrategyType.ORIGINAL_ONLY,
        description="Primary strategy that was executed",
    )
    analysis: Optional[QueryAnalysis] = Field(default=None, description="Query analysis result")
    fallback_used: bool = Field(default=False, description="True if rewriting failed and original was used as fallback")
    latency_ms: float = Field(default=0.0, description="Total rewriting latency in milliseconds")
    token_usage: int = Field(default=0, description="Total tokens consumed by LLM rewrite calls")
    cost_usd: float = Field(default=0.0, description="Estimated USD cost of LLM rewrite calls")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings encountered during rewriting")

    @property
    def query_count(self) -> int:
        """Number of query variants (including original)."""
        return len(self.rewritten_queries)

    @property
    def query_texts(self) -> List[str]:
        """All query texts for retrieval."""
        return [q.text for q in self.rewritten_queries]
