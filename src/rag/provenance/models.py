"""Structured data models and types for citation, provenance tracking, and answer validation."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator


class ProvenanceType(str, Enum):
    """Classification of evidence origin and extraction modality."""

    DIRECT_CHUNK = "direct_chunk"
    PARENT_CHUNK = "parent_chunk"
    NEIGHBOR_CHUNK = "neighbor_chunk"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    OCR_TEXT = "ocr_text"
    METADATA = "metadata"


class Citation(BaseModel):
    """Stable, traceable citation linking an in-text reference marker to verified evidence."""

    citation_id: str = Field(..., description="Unique citation marker, e.g. '[C1]'")
    document_id: str = Field(..., description="Document identifier")
    document_version_id: str = Field(default="v1", description="Exact document version hash or tag")
    chunk_id: str = Field(default="", description="Referenced chunk identifier")
    asset_id: Optional[str] = Field(default=None, description="Referenced visual asset identifier if applicable")
    source_uri: Optional[str] = Field(default=None, description="Storage or source URI")
    document_title: str = Field(default="", description="Document title or original filename")
    page_number: Optional[int] = Field(default=None, description="Source page number of evidence")
    section_path: List[str] = Field(default_factory=list, description="Hierarchical breadcrumb of section headers")
    quoted_text: str = Field(default="", description="Verbatim textual evidence snippet or table excerpt")
    retrieval_score: Optional[float] = Field(default=None, description="Retrieval, fusion, or reranking score")
    retrieval_method: str = Field(default="retrieval", description="Method that surfaced evidence (vector, bm25, hybrid, reranker)")
    provenance_type: ProvenanceType = Field(default=ProvenanceType.DIRECT_CHUNK, description="Provenance category")
    tenant_id: str = Field(default="default", description="Tenant organization key for security isolation")

    # Backwards compatibility alias fields
    filename: Optional[str] = Field(default=None, description="Alias for document_title")
    page: Optional[int] = Field(default=None, description="Alias for page_number")
    section: Optional[str] = Field(default=None, description="Alias for single section heading or path")
    snippet: Optional[str] = Field(default=None, description="Alias for quoted_text")
    score: Optional[float] = Field(default=None, description="Alias for retrieval_score")
    table_id: Optional[str] = Field(default=None, description="Table identifier if applicable")

    @model_validator(mode="after")
    def _sync_aliases(self) -> "Citation":
        # Synchronize document_title <-> filename
        if not self.document_title and self.filename:
            self.document_title = self.filename
        elif self.document_title and not self.filename:
            self.filename = self.document_title

        # Synchronize page_number <-> page
        if self.page_number is None and self.page is not None:
            self.page_number = self.page
        elif self.page_number is not None and self.page is None:
            self.page = self.page_number

        # Synchronize quoted_text <-> snippet
        if not self.quoted_text and self.snippet:
            self.quoted_text = self.snippet
        elif self.quoted_text and not self.snippet:
            self.snippet = self.quoted_text

        # Synchronize retrieval_score <-> score
        if self.retrieval_score is None and self.score is not None:
            self.retrieval_score = self.score
        elif self.retrieval_score is not None and self.score is None:
            self.score = self.retrieval_score

        # Synchronize section_path <-> section
        if not self.section_path and self.section:
            self.section_path = [self.section]
        elif self.section_path and not self.section:
            self.section = " › ".join(self.section_path)

        return self


class RetrievedEvidenceItem(BaseModel):
    """Standardized representation of retrieved evidence before or after registration."""

    item_id: str = Field(..., description="Unique chunk or asset identifier")
    document_id: str = Field(..., description="Parent document identifier")
    document_version_id: str = Field(default="v1", description="Document version identifier")
    tenant_id: str = Field(default="default", description="Tenant organization key")
    text: str = Field(..., description="Full or excerpted evidence text")
    title: str = Field(default="", description="Document title or filename")
    page_number: Optional[int] = Field(default=None, description="Page number")
    section_path: List[str] = Field(default_factory=list, description="Section hierarchy breadcrumbs")
    source_uri: Optional[str] = Field(default=None, description="Source URI or URL")
    asset_id: Optional[str] = Field(default=None, description="Visual asset ID if applicable")
    table_id: Optional[str] = Field(default=None, description="Structured table ID if applicable")
    retrieval_score: Optional[float] = Field(default=None, description="Relevance score")
    retrieval_method: str = Field(default="retrieval", description="Retrieval method")
    provenance_type: ProvenanceType = Field(default=ProvenanceType.DIRECT_CHUNK, description="Provenance category")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom document or chunk metadata")


class ClaimValidation(BaseModel):
    """Validation report for an individual claim or sentence extracted from the answer."""

    sentence: str = Field(..., description="Claim sentence evaluated")
    cited_markers: List[str] = Field(default_factory=list, description="Citation tags present in the sentence")
    is_supported: bool = Field(default=False, description="True if claim is substantiated by cited evidence")
    evidence_overlap_score: float = Field(default=0.0, description="Lexical and keyword overlap score (0.0 to 1.0)")
    mismatched_markers: List[str] = Field(default_factory=list, description="Citation markers whose content does not substantiate this claim")
    reason: Optional[str] = Field(default=None, description="Detailed explanation if claim is unsupported or mismatched")


class CitationValidationResult(BaseModel):
    """Aggregated validation diagnostics for citation integrity and grounding."""

    is_valid: bool = Field(default=True, description="True if all used citations are strictly valid and ground their claims")
    valid_citations: List[str] = Field(default_factory=list, description="Valid citation IDs used in answer")
    invalid_citations: List[str] = Field(default_factory=list, description="Citation IDs used in answer that do not exist in registry")
    missing_citations: List[str] = Field(default_factory=list, description="Sentences/claims identified as needing citations but having none")
    unused_citations: List[str] = Field(default_factory=list, description="Citation IDs provided in evidence but unreferenced in answer")
    duplicate_citations: List[str] = Field(default_factory=list, description="Redundant repeated citation markers in the same sentence")
    cross_tenant_citations: List[str] = Field(default_factory=list, description="Citations referencing another tenant's evidence")
    version_mismatch_citations: List[str] = Field(default_factory=list, description="Citations referencing an obsolete or mismatched document version")
    mismatched_citations: List[str] = Field(default_factory=list, description="Citations whose quoted evidence fails to substantiate the citing claim")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings and audit notes")


class GroundedAnswerResponse(BaseModel):
    """Complete traceable answer contract with validated provenance and confidence."""

    answer: str = Field(..., description="Generated answer text")
    citations: List[Citation] = Field(default_factory=list, description="All registered citations available for this answer")
    retrieved_evidence: List[RetrievedEvidenceItem] = Field(default_factory=list, description="Raw retrieved evidence passages registered")
    citation_validation: CitationValidationResult = Field(default_factory=CitationValidationResult, description="Detailed citation validation diagnostics")
    unsupported_claims: List[str] = Field(default_factory=list, description="Claims made in the answer lacking verified evidence")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Calibrated confidence score (0.0 to 1.0)")
    warnings: List[str] = Field(default_factory=list, description="Operational and integrity warnings")
