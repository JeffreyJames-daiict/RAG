"""Data models for structured LLM context assembly, citation mapping, and token telemetry."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Stable citation mapping linking an in-text marker like [C1] to source provenance."""

    citation_id: str = Field(..., description="Unique citation marker, e.g. '[C1]'")
    chunk_id: str = Field(..., description="Referenced chunk identifier")
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(default="", description="Original file name")
    page: Optional[int] = Field(default=None, description="Page number of evidence")
    section: Optional[str] = Field(default=None, description="Section heading breadcrumb")
    asset_id: Optional[str] = Field(default=None, description="Referenced visual asset ID if applicable")
    table_id: Optional[str] = Field(default=None, description="Referenced structured table ID if applicable")
    snippet: str = Field(default="", description="Short verbatim textual evidence snippet")
    score: Optional[float] = Field(default=None, description="Retrieval or reranking score")
    source_uri: Optional[str] = Field(default=None, description="Storage or source URI")


class TableContext(BaseModel):
    """Structured representation of a preserved or reconstructed tabular passage."""

    table_id: str = Field(..., description="Unique table identifier")
    caption: Optional[str] = Field(default=None, description="Table caption or heading")
    page: Optional[int] = Field(default=None, description="Source page number")
    formatted_table: str = Field(..., description="Markdown or structured serialized table text")
    is_reconstructed: bool = Field(default=False, description="True if table rows or split chunks were stitched")
    columns: List[str] = Field(default_factory=list, description="Extracted column header names")
    total_rows: int = Field(default=0, description="Number of data rows")


class AssetContext(BaseModel):
    """Visual asset associated with retrieved evidence (diagram, figure, chart)."""

    asset_id: str = Field(..., description="Unique asset identifier")
    asset_type: str = Field(default="FIGURE", description="Asset category: FIGURE, DIAGRAM, CHART")
    caption: Optional[str] = Field(default=None, description="Figure title or caption")
    ocr_text: Optional[str] = Field(default=None, description="OCR text detected inside asset")
    semantic_description: Optional[str] = Field(default=None, description="AI vision description of asset content")
    page: Optional[int] = Field(default=None, description="Page number where asset appears")
    storage_uri: Optional[str] = Field(default=None, description="Storage URI for image/diagram")
    entities: List[str] = Field(default_factory=list, description="Extracted diagram entities/nodes")


class ContextChunk(BaseModel):
    """Single context passage admitted into the token budget with citation assignment."""

    chunk_id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Document identifier")
    text: str = Field(..., description="Admitted chunk content")
    rank: int = Field(default=1, description="Original retrieval/reranking rank (1-based)")
    page: Optional[int] = Field(default=None, description="Page number")
    section: Optional[str] = Field(default=None, description="Section heading")
    chunk_type: str = Field(default="text", description="Type: text, table, asset")
    is_expanded: bool = Field(default=False, description="True if chunk was added via neighbor or parent expansion")
    expansion_type: Optional[str] = Field(
        default=None,
        description="Type of expansion: 'seed', 'neighbor_prev', 'neighbor_next', 'parent', 'same_section'",
    )
    citation_id: str = Field(default="", description="Assigned citation ID, e.g. '[C1]'")
    token_count: int = Field(default=0, description="Estimated or exact token length")
    score: Optional[float] = Field(default=None, description="Relevance score (reranked or similarity)")
    table_id: Optional[str] = Field(default=None, description="Associated table ID if applicable")
    asset_id: Optional[str] = Field(default=None, description="Associated visual asset ID if applicable")
    filename: str = Field(default="", description="Source document filename")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw metadata dictionary")


class GroupedDocumentContext(BaseModel):
    """Hierarchical grouping of admitted context chunks by document and section."""

    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(default="", description="Document filename")
    version: int = Field(default=1, description="Document version number")
    sections: Dict[str, List[ContextChunk]] = Field(
        default_factory=dict,
        description="Map of section headings to ordered context chunks",
    )
    tables: List[TableContext] = Field(default_factory=list, description="Preserved tables from this document")
    assets: List[AssetContext] = Field(default_factory=list, description="Associated visual assets from this document")
    total_tokens: int = Field(default=0, description="Aggregated token count for this document")


class ContextPackage(BaseModel):
    """Complete, token-budgeted, citation-ready context package delivered to LLM generation."""

    query: str = Field(..., description="Original user query")
    selected_chunks: List[ContextChunk] = Field(
        default_factory=list,
        description="Flat list of selected chunks in presentation order",
    )
    grouped_documents: List[GroupedDocumentContext] = Field(
        default_factory=list,
        description="Hierarchically structured documents and sections",
    )
    formatted_context: str = Field(
        default="",
        description="Sanitized, XML-fenced, citation-annotated context string for LLM prompt",
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Full registry of assigned citations mapping markers to evidence",
    )
    token_count: int = Field(default=0, description="Total tokens consumed by formatted_context")
    token_budget: int = Field(default=4000, description="Configured token budget limit")
    truncated: bool = Field(default=False, description="True if any candidate was trimmed or dropped")
    truncation_reason: Optional[str] = Field(default=None, description="Explanation if truncation occurred")
    context_strategy: str = Field(
        default="hierarchical_xml",
        description="Context formatting strategy: hierarchical_xml | flat_blocks",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal notices, token overages, or injection neutralizations",
    )
