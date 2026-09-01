"""Visual asset and figure processing subsystem for enterprise documents."""

from __future__ import annotations

from rag.assets.analyzer import FigureAnalyzer
from rag.assets.caption_detector import CaptionDetector
from rag.assets.classifier import AssetClassifier
from rag.assets.chunker import AssetChunker
from rag.assets.exceptions import (
    AssetError,
    AssetExtractionError,
    AssetClassificationError,
    AssetOCRError,
    VisionAnalysisError,
    AssetStorageError,
)
from rag.assets.extractor import AssetExtractor
from rag.assets.models import (
    AssetClassificationResult,
    AssetLifecycleStatus,
    ExtractedVisualCandidate,
    FigureAnalysisResult,
    FigureEntityRelationship,
    FigureOCRResult,
    ProcessedAsset,
)
from rag.assets.ocr import FigureOCRRunner
from rag.assets.processor import AssetProcessor
from rag.assets.providers import (
    BaseVisionProvider,
    MockVisionProvider,
    OpenAIVisionProvider,
    get_vision_provider,
)
from rag.assets.storage import AssetStorage

__all__ = [
    "AssetError",
    "AssetExtractionError",
    "AssetClassificationError",
    "AssetOCRError",
    "VisionAnalysisError",
    "AssetStorageError",
    "AssetLifecycleStatus",
    "ExtractedVisualCandidate",
    "AssetClassificationResult",
    "FigureOCRResult",
    "FigureEntityRelationship",
    "FigureAnalysisResult",
    "ProcessedAsset",
    "AssetStorage",
    "AssetClassifier",
    "CaptionDetector",
    "FigureOCRRunner",
    "BaseVisionProvider",
    "MockVisionProvider",
    "OpenAIVisionProvider",
    "get_vision_provider",
    "FigureAnalyzer",
    "AssetExtractor",
    "AssetChunker",
    "AssetProcessor",
]
