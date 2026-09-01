"""Orchestrator for visual asset extraction, classification, OCR, vision, and storage."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from rag.assets.analyzer import FigureAnalyzer
from rag.assets.caption_detector import CaptionDetector
from rag.assets.classifier import AssetClassifier
from rag.assets.chunker import AssetChunker
from rag.assets.models import (
    AssetClassificationResult,
    ExtractedVisualCandidate,
    FigureOCRResult,
    ProcessedAsset,
)
from rag.assets.ocr import FigureOCRRunner
from rag.assets.storage import AssetStorage
from rag.core.config import AssetConfig
from rag.core.models import AssetType, ChunkSchema, DocumentAssetSchema
from rag.storage.base import RawStorageBackend

logger = logging.getLogger(__name__)


class AssetProcessor:
    """End-to-end coordinator for visual figures, diagrams, and architecture illustrations."""

    def __init__(
        self,
        config: Optional[AssetConfig] = None,
        storage_backend: Optional[RawStorageBackend] = None,
        figure_analyzer: Optional[FigureAnalyzer] = None,
        ocr_runner: Optional[FigureOCRRunner] = None,
    ):
        self.config = config or AssetConfig()
        self.storage = AssetStorage(storage_backend=storage_backend)
        self.classifier = AssetClassifier(config=self.config)
        self.caption_detector = CaptionDetector(vertical_tolerance_pts=self.config.caption_vertical_distance_pts)
        self.ocr_runner = ocr_runner or FigureOCRRunner(config=self.config)
        self.analyzer = figure_analyzer or FigureAnalyzer(config=self.config)
        self.chunker = AssetChunker()

    def process_candidate(
        self,
        candidate: ExtractedVisualCandidate,
        tenant_id: str,
        captions_on_page: Optional[List[Tuple[Tuple[float, float, float, float], str]]] = None,
    ) -> Optional[ProcessedAsset]:
        """Process a single visual candidate through classification, OCR, vision, and storage."""
        # 1. Caption Association
        caption = None
        if self.config.caption_detection_enabled and captions_on_page:
            caption = self.caption_detector.associate_caption_to_figure(
                figure_bbox=candidate.bbox,
                captions=captions_on_page,
            )

        # 2. Classification
        classification = self.classifier.classify(candidate=candidate, caption=caption)
        if classification.is_decorative and self.config.filter_decorative:
            logger.debug(
                "Skipping decorative asset %s on page %d: %s",
                candidate.asset_id, candidate.page_number, classification.reason,
            )
            return None

        # 3. Figure-Specific OCR
        ocr_res = None
        if self.config.figure_ocr_enabled:
            ocr_res = self.ocr_runner.process_figure_image(
                image_bytes=candidate.image_bytes,
                page_number=candidate.page_number,
            )

        ocr_text = ocr_res.ocr_text if ocr_res else None
        ocr_conf = ocr_res.confidence if ocr_res else None

        # 4. Multimodal Vision Semantic Analysis
        analysis_res = self.analyzer.analyze(
            image_bytes=candidate.image_bytes,
            asset_type=classification.asset_type,
            caption=caption,
            ocr_text=ocr_text,
            mime_type=candidate.mime_type,
        )

        # 5. Asset Storage Persistence
        content_hash = self.storage.compute_hash(candidate.image_bytes)
        storage_uri = self.storage.store_asset(
            tenant_id=tenant_id,
            document_id=candidate.document_id,
            version=candidate.version,
            asset_type=classification.asset_type,
            asset_id=candidate.asset_id,
            image_bytes=candidate.image_bytes,
            mime_type=candidate.mime_type,
        )

        # 6. Build DocumentAssetSchema
        bbox_dict = {
            "x0": round(candidate.bbox[0], 2),
            "y0": round(candidate.bbox[1], 2),
            "x1": round(candidate.bbox[2], 2),
            "y1": round(candidate.bbox[3], 2),
        }
        entities = analysis_res.entities if analysis_res else []
        relationships = [r.model_dump() for r in analysis_res.relationships] if analysis_res else []
        status = "READY"
        if analysis_res and analysis_res.status == "FAILED":
            status = "PARTIAL_SUCCESS"  # OCR + Image intact, vision failed gracefully

        asset_schema = DocumentAssetSchema(
            asset_id=UUID(candidate.asset_id),
            document_id=UUID(candidate.document_id),
            version=candidate.version,
            page_number=candidate.page_number,
            asset_type=classification.asset_type,
            asset_subtype=classification.asset_subtype,
            storage_uri=storage_uri,
            content_hash=content_hash,
            bbox=bbox_dict,
            width=candidate.width,
            height=candidate.height,
            mime_type=candidate.mime_type,
            caption=caption,
            ocr_text=ocr_text,
            ocr_confidence=ocr_conf,
            semantic_description=analysis_res.description if analysis_res else None,
            entities=entities,
            relationships=relationships,
            processing_status=status,
            processor_version="asset-v1.0",
            asset_metadata={
                "source_type": candidate.source_type,
                "classification_reason": classification.reason,
                "confidence": classification.confidence,
                "vision_provider": analysis_res.provider_name if analysis_res else "none",
            },
        )

        processed = ProcessedAsset(
            schema=asset_schema,
            image_bytes=candidate.image_bytes,
            classification=classification,
            ocr_result=ocr_res,
            analysis_result=analysis_res,
        )
        return processed

    def create_chunks_for_assets(
        self,
        processed_assets: List[ProcessedAsset],
        start_chunk_index: int = 0,
    ) -> List[ChunkSchema]:
        """Convert a collection of processed assets into searchable ChunkSchema instances."""
        chunks = []
        curr_idx = start_chunk_index
        for pa in processed_assets:
            chk = self.chunker.create_chunk(processed_asset=pa, chunk_index=curr_idx)
            chunks.append(chk)
            curr_idx += 1
        return chunks
