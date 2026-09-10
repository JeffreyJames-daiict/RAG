"""Data models and schemas for query routing layer."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Classified semantic and structural query category."""

    GENERAL_SEMANTIC = "general_semantic"
    EXACT_IDENTIFIER = "exact_identifier"
    METADATA = "metadata"
    TABLE = "table"
    FIGURE = "figure"
    SUMMARIZATION = "summarization"
    DOCUMENT_SPECIFIC = "document_specific"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"


class RouteTarget(str, Enum):
    """Execution route strategy for retrieval and answering."""

    BM25_FIRST = "bm25_first"
    HYBRID_RRF = "hybrid_rrf"
    TABLE_AWARE = "table_aware"
    MULTIMODAL = "multimodal"
    METADATA_LOOKUP = "metadata_lookup"
    AGENTIC_WORKFLOW = "agentic_workflow"
    SUMMARIZATION = "summarization"
    DOCUMENT_FILTERED = "document_filtered"
    DIRECT_ABSTAIN = "direct_abstain"


class RouterType(str, Enum):
    """Active router mechanism."""

    RULE_BASED = "rule_based"
    LLM = "llm"
    HYBRID = "hybrid"


class ExtractedFeatures(BaseModel):
    """Features extracted from the query during analysis."""

    detected_identifiers: List[str] = Field(default_factory=list, description="Extracted UUIDs, patents, versions, codes")
    detected_keywords: List[str] = Field(default_factory=list, description="Extracted domain keywords")
    target_documents: List[str] = Field(default_factory=list, description="Explicit document names, titles, or filenames detected")
    has_table_signals: bool = Field(default=False, description="Presence of table, row, column, financial metrics signals")
    has_figure_signals: bool = Field(default=False, description="Presence of figure, diagram, chart, image signals")
    has_metadata_signals: bool = Field(default=False, description="Presence of author, date, page count, file size signals")
    has_summarization_signals: bool = Field(default=False, description="Presence of summarize, overview, tldr signals")
    has_multi_hop_signals: bool = Field(default=False, description="Presence of multi-step, causal, cross-document signals")
    is_empty_or_gibberish: bool = Field(default=False, description="Query is empty, whitespace, or consonant keyboard mash")
    word_count: int = Field(default=0, description="Query word count")


class RoutingDecision(BaseModel):
    """Result of query routing evaluation."""

    query: str = Field(..., description="Original input query")
    route: RouteTarget = Field(..., description="Target retrieval strategy")
    query_type: QueryType = Field(..., description="Classified query type")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Routing confidence score (0.0 to 1.0)")
    fallback_used: bool = Field(default=False, description="True if fallback route was triggered")
    router_type: RouterType = Field(default=RouterType.RULE_BASED, description="Router implementation used")
    features: ExtractedFeatures = Field(default_factory=ExtractedFeatures, description="Extracted query features")
    target_document_ids: List[str] = Field(default_factory=list, description="Target document IDs if document-scoped")
    latency_ms: float = Field(default=0.0, description="Routing computation time in ms")
    cost_usd: float = Field(default=0.0, description="Cost in USD incurred by router")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Tenant-safe routing metadata and reasoning")
