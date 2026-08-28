"""Canonical data models for the Content Normalization Layer.

Provides structured, traceable, chunk-ready content representation bridging
extraction outputs (native text, OCR, tables, mixed) with downstream chunkers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from rag.parsing.models import TableElement


class NormalizedContentType(str, Enum):
    """Classification of normalized content blocks."""
    TEXT = "text"
    HEADING = "heading"
    LIST_ITEM = "list"
    TABLE = "table"
    IMAGE_OCR = "image_ocr"
    MIXED = "mixed"


class ExtractionMethod(str, Enum):
    """Source extraction mechanism that produced the content."""
    NATIVE_PDF = "native_pdf"
    OCR = "ocr"
    TABLE_EXTRACTION = "table_extraction"
    MIXED = "mixed"


@dataclass
class NormalizedContentBlock:
    """Canonical, chunk-ready content block with full provenance and auditability.

    Attributes:
        block_id: Unique identifier for this normalized content block.
        document_id: UUID string of the parent document.
        page_number: 1-indexed page number where this block originated.
        content_type: Semantic classification (text, heading, list, table, image_ocr, mixed).
        source_type: Original source designation ("native", "scanned", "mixed").
        normalized_text: Cleaned, validated, normalized textual representation.
        original_text: Unmodified raw text as extracted prior to normalization.
        extraction_method: Mechanism used to extract this block.
        language: ISO-639-3 or standard language tag if detected (e.g., 'eng').
        language_confidence: Language classifier certainty (0.0 to 1.0).
        confidence_score: Composite extraction/OCR confidence score (0.0 to 1.0).
        quality_flags: List of anomalies or notices (e.g. 'low_confidence', 'ocr_noise').
        normalization_actions: History of transformations applied (e.g. 'whitespace_normalization').
        metadata: Domain and structural metadata (bbox, section, table headers/metrics).
        provenance: Extraction provenance (provider name, profile, duplicate resolution info).
        is_suppressed: True if this block was suppressed during duplicate resolution.
        suppression_reason: Explanation if is_suppressed is True.
        table_element: Associated structured TableElement if content_type is TABLE.
    """
    block_id: str = field(default_factory=lambda: str(uuid4()))
    document_id: str = ""
    page_number: int = 1
    content_type: NormalizedContentType = NormalizedContentType.TEXT
    source_type: str = "native"
    normalized_text: str = ""
    original_text: str = ""
    extraction_method: ExtractionMethod = ExtractionMethod.NATIVE_PDF
    language: str = "eng"
    language_confidence: float = 1.0
    confidence_score: float = 1.0
    quality_flags: List[str] = field(default_factory=list)
    normalization_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    is_suppressed: bool = False
    suppression_reason: Optional[str] = None
    table_element: Optional[TableElement] = None

    def add_action(self, action_name: str) -> None:
        """Record a normalization transformation if not already present."""
        if action_name not in self.normalization_actions:
            self.normalization_actions.append(action_name)

    def add_flag(self, flag_name: str) -> None:
        """Add a quality flag or anomaly notice if not already present."""
        if flag_name not in self.quality_flags:
            self.quality_flags.append(flag_name)


@dataclass
class NormalizedPage:
    """Container for all normalized content blocks belonging to a single page."""
    page_number: int
    blocks: List[NormalizedContentBlock] = field(default_factory=list)
    raw_text: str = ""
    normalized_text: str = ""
    primary_language: str = "eng"
    language_confidence: float = 1.0
    content_source: str = "native"
    ocr_applied: bool = False
    ocr_confidence: Optional[float] = None
    quality_score: Optional[float] = None
    tables: List[TableElement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def active_blocks(self) -> List[NormalizedContentBlock]:
        """Return non-suppressed content blocks."""
        return [b for b in self.blocks if not b.is_suppressed]


@dataclass
class NormalizedDocument:
    """Canonical representation of an entire normalized document ready for chunking."""
    document_id: str
    filename: str
    pages: List[NormalizedPage] = field(default_factory=list)
    tables: List[TableElement] = field(default_factory=list)
    primary_language: str = "eng"
    languages_detected: List[str] = field(default_factory=list)
    normalization_diagnostics: Dict[str, Any] = field(
        default_factory=lambda: {
            "total_pages": 0,
            "blocks_processed": 0,
            "blocks_active": 0,
            "blocks_dropped": 0,
            "duplicates_detected": 0,
            "headers_removed": 0,
            "footers_removed": 0,
            "standalone_page_numbers_removed": 0,
            "low_confidence_blocks": 0,
            "quality_flags_count": 0,
        }
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def all_active_blocks(self) -> List[NormalizedContentBlock]:
        """Aggregate all non-suppressed blocks across all pages."""
        res: List[NormalizedContentBlock] = []
        for page in self.pages:
            res.extend(page.active_blocks)
        return res
