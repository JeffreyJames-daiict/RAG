"""Content Normalization Layer for Enterprise RAG.

Bridges extraction outputs (native text, OCR, tables, mixed content) with downstream
chunkers by converting raw content into canonical, validated, normalized blocks.
"""

from rag.normalization.duplicate_resolver import DuplicateResolutionDecision, DuplicateResolver
from rag.normalization.header_footer_detector import HeaderFooterDetectionResult, HeaderFooterDetector
from rag.normalization.models import (
    ExtractionMethod,
    NormalizedContentBlock,
    NormalizedContentType,
    NormalizedDocument,
    NormalizedPage,
)
from rag.normalization.normalizer import ContentNormalizer
from rag.normalization.ocr_normalizer import OCRNormalizer
from rag.normalization.text_normalizer import TextNormalizer

__all__ = [
    "ContentNormalizer",
    "DuplicateResolutionDecision",
    "DuplicateResolver",
    "ExtractionMethod",
    "HeaderFooterDetectionResult",
    "HeaderFooterDetector",
    "NormalizedContentBlock",
    "NormalizedContentType",
    "NormalizedDocument",
    "NormalizedPage",
    "OCRNormalizer",
    "TextNormalizer",
]
