"""Orchestrator for visual asset extraction, deduplication, classification, relevance, OCR, vision, and storage."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from rag.assets.analyzer import FigureAnalyzer
from rag.assets.caption_detector import CaptionDetector
from rag.assets.classifier import AssetClassifier
from rag.assets.chunker import AssetChunker
from rag.assets.deduplicator import AssetDeduplicator
from rag.assets.models import (
    AssetClassificationResult,
    AssetContextInfo,
    AssetLifecycleStatus,
    AssetProcessingMetrics,
    AssetRelevanceResult,
    ExtractedVisualCandidate,
    FigureOCRResult,
    ProcessedAsset,
)
from rag.assets.ocr import FigureOCRRunner
from rag.assets.relevance import AssetRelevanceScorer
from rag.assets.storage import AssetStorage
from rag.core.config import AssetConfig
from rag.core.models import AssetType, ChunkSchema, DocumentAssetSchema, IndexingPolicy
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
        deduplicator: Optional[AssetDeduplicator] = None,
        relevance_scorer: Optional[AssetRelevanceScorer] = None,
    ):
        self.config = config or AssetConfig()
        self.storage = AssetStorage(storage_backend=storage_backend)
        self.classifier = AssetClassifier(config=self.config)
        self.caption_detector = CaptionDetector(vertical_tolerance_pts=self.config.caption_vertical_distance_pts)
        self.ocr_runner = ocr_runner or FigureOCRRunner(config=self.config)
        self.analyzer = figure_analyzer or FigureAnalyzer(config=self.config)
        self.chunker = AssetChunker()
        self.deduplicator = deduplicator or AssetDeduplicator(
            dedup_enabled=self.config.dedup_enabled,
            phash_enabled=self.config.perceptual_hash_enabled,
            phash_distance_threshold=self.config.phash_hamming_distance_threshold,
        )
        self.relevance_scorer = relevance_scorer or AssetRelevanceScorer(config=self.config)
        self.metrics = AssetProcessingMetrics()
        self._vision_calls_made: int = 0

    def reset_session(self) -> None:
        """Reset session metrics and deduplication state for a new document."""
        self.metrics = AssetProcessingMetrics()
        self.deduplicator.reset()
        self._vision_calls_made = 0

    def process_candidate(
        self,
        candidate: ExtractedVisualCandidate,
        tenant_id: str,
        captions_on_page: Optional[List[Tuple[Tuple[float, float, float, float], str]]] = None,
        context: Optional[AssetContextInfo] = None,
    ) -> Optional[ProcessedAsset]:
        """Process a single visual candidate through hashing, deduplication, classification,

        selective OCR, relevance scoring, selective vision, and MinIO storage.
        """
        self.metrics.total_extracted += 1

        # ── Step 1: Storage in MinIO / Object Storage (Non-destructive preservation)
        # Construct preliminary storage URI; actual storage is performed idempotently
        content_hash = self.storage.compute_hash(candidate.image_bytes)
        storage_uri = self.storage.store_asset(
            tenant_id=tenant_id,
            document_id=candidate.document_id,
            version=candidate.version,
            asset_type=AssetType.FIGURE,
            asset_id=candidate.asset_id,
            image_bytes=candidate.image_bytes,
            mime_type=candidate.mime_type,
        )
        self.metrics.stored_minio += 1

        # ── Step 2: Hashing & Deduplication (SHA-256 + 64-bit Perceptual Hash) ───
        (
            sha,
            phash,
            is_duplicate,
            canonical_asset_id,
            frequency,
        ) = self.deduplicator.evaluate_asset(
            image_bytes=candidate.image_bytes,
            asset_id=candidate.asset_id,
            page_number=candidate.page_number,
        )
        if is_duplicate:
            self.metrics.duplicates_detected += 1

        # ── Step 3: Caption & Context Association ─────────────────────────────
        caption = None
        if self.config.caption_detection_enabled and captions_on_page:
            caption = self.caption_detector.associate_caption_to_figure(
                figure_bbox=candidate.bbox,
                captions=captions_on_page,
            )

        if context is None:
            context = AssetContextInfo(
                caption=caption,
                nearby_text=candidate.nearby_text,
                section_heading=candidate.section_heading,
            )
        elif caption and not context.caption:
            context.caption = caption

        # ── Step 4: Cheap Candidate Classification (Pre-Vision) ───────────────
        classification = self.classifier.classify(
            candidate=candidate,
            caption=caption,
            context=context,
            frequency=frequency,
        )

        # ── Step 5: Selective OCR ─────────────────────────────────────────────
        ocr_res: Optional[FigureOCRResult] = None
        cached_analysis, cached_ocr = (None, None)
        if is_duplicate:
            cached_item = self.deduplicator.get_cached_analysis(sha, phash)
            if cached_item:
                cached_analysis, cached_ocr = cached_item
                ocr_res = cached_ocr

        if ocr_res is None and self.config.figure_ocr_enabled and not classification.is_decorative:
            ocr_res = self.ocr_runner.process_figure_image(
                image_bytes=candidate.image_bytes,
                page_number=candidate.page_number,
            )
            if ocr_res and ocr_res.ocr_applied:
                self.metrics.ocr_processed += 1

        if ocr_res and ocr_res.ocr_text:
            context.ocr_text = ocr_res.ocr_text

        # ── Step 6: Relevance Scoring & Policy Selection ──────────────────────
        relevance = self.relevance_scorer.evaluate(
            candidate=candidate,
            classification=classification,
            caption=caption,
            ocr_result=ocr_res,
            context=context,
            frequency=frequency,
        )
        policy = relevance.indexing_policy

        if policy == IndexingPolicy.STORE_ONLY:
            self.metrics.store_only += 1
        elif policy == IndexingPolicy.IGNORE:
            self.metrics.ignored += 1

        # ── Step 7: Selective Vision Analysis (Only for VISION_ANALYSIS policy) 
        analysis_res = cached_analysis
        processing_error = None
        status = AssetLifecycleStatus.READY.value

        if policy in (IndexingPolicy.VISION_ANALYSIS, IndexingPolicy.FULL_MULTIMODAL):
            if analysis_res is None:
                # Check vision budget per document
                if self._vision_calls_made < self.config.max_vision_calls_per_document:
                    self._vision_calls_made += 1
                    self.metrics.vision_analyzed += 1

                    analysis_res = self.analyzer.analyze(
                        image_bytes=candidate.image_bytes,
                        asset_type=classification.asset_type,
                        caption=caption,
                        ocr_text=ocr_res.ocr_text if ocr_res else None,
                        mime_type=candidate.mime_type,
                        context=context,
                        indexing_policy=policy,
                    )
                    if analysis_res.status == "FAILED":
                        self.metrics.vision_failures += 1
                        processing_error = analysis_res.error_message or "Vision provider failed"
                        status = "PARTIAL_SUCCESS"

                    # Cache analysis for duplicates
                    self.deduplicator.cache_analysis(
                        content_hash=sha,
                        perceptual_hash=phash,
                        analysis=analysis_res,
                        ocr=ocr_res,
                    )
                else:
                    logger.info(
                        "Vision call budget reached (%d/%d) for document %s; falling back to OCR/caption representation.",
                        self._vision_calls_made, self.config.max_vision_calls_per_document, candidate.document_id,
                    )
                    policy = IndexingPolicy.OCR_ONLY
                    relevance.indexing_policy = policy
                    relevance.relevance_reason += " (Vision call budget reached; downgraded to OCR_ONLY)"
        elif policy == IndexingPolicy.STORE_ONLY:
            status = AssetLifecycleStatus.STORE_ONLY.value

        # ── Step 8: Build DocumentAssetSchema & ProcessedAsset ─────────────────
        bbox_dict = {
            "x0": round(candidate.bbox[0], 2),
            "y0": round(candidate.bbox[1], 2),
            "x1": round(candidate.bbox[2], 2),
            "y1": round(candidate.bbox[3], 2),
        }
        entities = analysis_res.entities if analysis_res else []
        relationships = [r.model_dump() for r in analysis_res.relationships] if analysis_res else []

        asset_schema = DocumentAssetSchema(
            asset_id=UUID(candidate.asset_id),
            document_id=UUID(candidate.document_id),
            version=candidate.version,
            page_number=candidate.page_number,
            asset_type=classification.asset_type,
            asset_subtype=classification.asset_subtype,
            storage_uri=storage_uri,
            content_hash=content_hash,
            perceptual_hash=phash,
            bbox=bbox_dict,
            width=candidate.width,
            height=candidate.height,
            mime_type=candidate.mime_type,
            caption=caption,
            ocr_text=ocr_res.ocr_text if ocr_res else None,
            ocr_confidence=ocr_res.confidence if ocr_res else None,
            semantic_description=analysis_res.description if analysis_res else None,
            entities=entities,
            relationships=relationships,
            relevance_score=relevance.relevance_score,
            relevance_reason=relevance.relevance_reason,
            indexing_policy=policy,
            processing_status=status,
            processing_error=processing_error,
            processor_version="asset-v2.0",
            asset_metadata={
                "source_type": candidate.source_type,
                "classification_reason": classification.reason,
                "confidence": classification.confidence,
                "frequency": frequency,
                "is_duplicate": is_duplicate,
                "canonical_asset_id": canonical_asset_id,
                "vision_provider": analysis_res.provider_name if analysis_res else "none",
            },
        )

        return ProcessedAsset(
            schema=asset_schema,
            image_bytes=candidate.image_bytes,
            classification=classification,
            relevance=relevance,
            ocr_result=ocr_res,
            analysis_result=analysis_res,
        )

    def create_chunks_for_assets(
        self,
        processed_assets: List[ProcessedAsset],
        start_chunk_index: int = 0,
    ) -> List[ChunkSchema]:
        """Convert only indexable visual assets into searchable ChunkSchema instances."""
        chunks = []
        curr_idx = start_chunk_index
        for pa in processed_assets:
            if self.chunker.is_indexable(pa):
                chk = self.chunker.create_chunk(processed_asset=pa, chunk_index=curr_idx)
                chunks.append(chk)
                curr_idx += 1
                self.metrics.embedded_chunks += 1
            else:
                logger.debug(
                    "Skipping chunk creation for asset %s (policy=%s, relevance=%.2f)",
                    pa.schema.asset_id, pa.schema.indexing_policy.value, pa.schema.relevance_score or 0.0,
                )
        return chunks

