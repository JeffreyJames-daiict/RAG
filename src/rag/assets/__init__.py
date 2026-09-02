"""Visual asset and figure processing subsystem for enterprise documents."""

from __future__ import annotations

from rag.assets.analyzer import FigureAnalyzer
from rag.assets.caption_detector import CaptionDetector
from rag.assets.classifier import AssetClassifier
from rag.assets.chunker import AssetChunker
from rag.assets.deduplicator import (
    AssetDeduplicator,
    compute_phash,
    compute_sha256,
    hamming_distance,
)
from rag.assets.exceptions import (
    AssetClassificationError,
    AssetError,
    AssetExtractionError,
    AssetOCRError,
    AssetStorageError,
    AssetValidationError,
    VisionAnalysisError,
)
from rag.assets.extractor import AssetExtractor
from rag.assets.models import (
    AssetClassificationResult,
    AssetContextInfo,
    AssetLifecycleStatus,
    AssetProcessingMetrics,
    AssetRelevanceResult,
    ExtractedVisualCandidate,
    FigureAnalysisResult,
    FigureEntityRelationship,
    FigureOCRResult,
    ProcessedAsset,
    StructuredVisionOutput,
)
from rag.assets.ocr import FigureOCRRunner
from rag.assets.processor import AssetProcessor
from rag.assets.providers import (
    BaseVisionProvider,
    MockVisionProvider,
    OpenAIVisionProvider,
    OpenRouterVisionProvider,
    get_vision_provider,
)
from rag.assets.relevance import AssetRelevanceScorer
from rag.assets.storage import AssetStorage

__all__ = [
    "AssetError",
    "AssetExtractionError",
    "AssetClassificationError",
    "AssetOCRError",
    "VisionAnalysisError",
    "AssetStorageError",
    "AssetValidationError",
    "AssetLifecycleStatus",
    "ExtractedVisualCandidate",
    "AssetContextInfo",
    "AssetClassificationResult",
    "AssetRelevanceResult",
    "AssetProcessingMetrics",
    "StructuredVisionOutput",
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
    "OpenRouterVisionProvider",
    "get_vision_provider",
    "FigureAnalyzer",
    "AssetExtractor",
    "AssetChunker",
    "AssetProcessor",
    "AssetDeduplicator",
    "AssetRelevanceScorer",
    "compute_sha256",
    "compute_phash",
    "hamming_distance",
]
