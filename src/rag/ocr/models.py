"""Data models for OCR processing, page classification, and quality metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ContentSource(str, Enum):
    """Source classification of page or block content."""
    NATIVE = "native"
    SCANNED = "scanned"
    MIXED = "mixed"
    OCR = "ocr"


class OCRStatus(str, Enum):
    """Operational status of OCR processing."""
    PASS = "PASS"
    LOW_QUALITY = "LOW_QUALITY"
    RETRY_REQUIRED = "RETRY_REQUIRED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class PageClassificationResult:
    """Evaluation of native page text and layout quality."""
    page_number: int
    source_type: ContentSource
    native_text_quality: float
    ocr_required: bool
    reason: str
    image_count: int = 0
    text_length: int = 0
    printable_ratio: float = 1.0
    meaningful_word_ratio: float = 1.0
    image_coverage: float = 0.0
    detected_language: str = "eng"
    language_confidence: float = 1.0
    unexpected_symbol_count: int = 0
    unexpected_symbol_ratio: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRBlock:
    """A recognized text block from an OCR provider with spatial bounds."""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    confidence: float = 1.0
    block_type: int = 0  # 0: Text, 1: Image/other
    block_no: int = 0
    lines: List[str] = field(default_factory=list)


@dataclass
class OCRPageResult:
    """Structured output from an OCR execution on a single rendered page."""
    page_number: int
    text: str
    blocks: List[OCRBlock] = field(default_factory=list)
    confidence: float = 1.0
    quality_score: float = 1.0
    status: OCRStatus = OCRStatus.PASS
    attempt: int = 1
    preprocessing_profile: str = "standard"
    error_message: Optional[str] = None
    raw_response: Optional[Any] = None
    provider_name: str = "tesseract"
    processing_time_ms: float = 0.0


@dataclass
class OCRProvenance:
    """Metadata tracking document and page-level OCR history."""
    content_source: str = "native"
    ocr_applied: bool = False
    ocr_provider: Optional[str] = None
    ocr_confidence: Optional[float] = None
    quality_score: Optional[float] = None
    processing_attempt: int = 1
    preprocessing_profile: Optional[str] = None
    native_text_available: bool = True
    error_message: Optional[str] = None
