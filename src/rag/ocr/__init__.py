"""Modular Optical Character Recognition (OCR) subsystem for Enterprise RAG.

Exports:
    BaseOCRProvider
    OCRProviderRegistry
    PageQualityAnalyzer
    ImagePreprocessor
    OCRQualityEvaluator
    MixedContentMerger
    OCRPageResult
    OCRBlock
    PageClassificationResult
    ContentSource
    OCRStatus
"""

from rag.ocr.analyzer import PageQualityAnalyzer
from rag.ocr.base import BaseOCRProvider, OCRProviderRegistry
from rag.ocr.evaluator import OCRQualityEvaluator
from rag.ocr.language_detector import LanguageDetectionResult, LanguageDetector
from rag.ocr.merger import MixedContentMerger
from rag.ocr.models import (
    ContentSource,
    OCRBlock,
    OCRPageResult,
    OCRProvenance,
    OCRStatus,
    PageClassificationResult,
)
from rag.ocr.preprocessor import ImagePreprocessor
# Import providers to register them
import rag.ocr.providers  # noqa: F401

__all__ = [
    "BaseOCRProvider",
    "OCRProviderRegistry",
    "PageQualityAnalyzer",
    "ImagePreprocessor",
    "OCRQualityEvaluator",
    "LanguageDetector",
    "LanguageDetectionResult",
    "MixedContentMerger",
    "OCRPageResult",
    "OCRBlock",
    "OCRProvenance",
    "PageClassificationResult",
    "ContentSource",
    "OCRStatus",
]
