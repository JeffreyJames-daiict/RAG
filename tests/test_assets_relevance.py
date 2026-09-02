"""Unit tests for heuristic figure relevance scoring and indexing policy selection."""

from __future__ import annotations

import pytest

from rag.assets.models import (
    AssetClassificationResult,
    AssetContextInfo,
    ExtractedVisualCandidate,
    FigureOCRResult,
)
from rag.assets.relevance import AssetRelevanceScorer
from rag.core.config import AssetConfig, RelevanceWeightsConfig
from rag.core.models import AssetType, IndexingPolicy


def _make_candidate(
    width: int = 400,
    height: int = 300,
    nearby_text: str = "",
    section_heading: str = "",
    page_coverage: float = 0.20,
) -> ExtractedVisualCandidate:
    return ExtractedVisualCandidate(
        asset_id="cand_1",
        document_id="doc_1",
        version=1,
        page_number=1,
        bbox=(50.0, 100.0, 50.0 + width, 100.0 + height),
        image_bytes=b"fake_image_bytes",
        width=width,
        height=height,
        nearby_text=nearby_text,
        section_heading=section_heading,
        page_coverage=page_coverage,
    )


def test_high_relevance_architecture_diagram_routes_to_vision():
    scorer = AssetRelevanceScorer()
    cand = _make_candidate(
        nearby_text="The microservice architecture routes incoming queries to the vector database.",
        section_heading="System Architecture",
    )
    classification = AssetClassificationResult(
        asset_type=AssetType.ARCHITECTURE_DIAGRAM,
        confidence=0.95,
        is_decorative=False,
    )
    ocr_result = FigureOCRResult(
        ocr_text="API Gateway -> Embedding Service -> Vector Database",
        confidence=0.90,
        ocr_applied=True,
    )
    caption = "Figure 2: End-to-End Ingestion and Retrieval Pipeline"

    res = scorer.evaluate(
        candidate=cand,
        classification=classification,
        caption=caption,
        ocr_result=ocr_result,
    )

    assert res.relevance_score >= 0.65
    assert res.indexing_policy == IndexingPolicy.VISION_ANALYSIS
    assert "Figure 2" in res.relevance_reason or "caption" in res.relevance_reason.lower()


def test_decorative_logo_routes_to_store_only():
    scorer = AssetRelevanceScorer()
    cand = _make_candidate(width=120, height=40, page_coverage=0.02)
    classification = AssetClassificationResult(
        asset_type=AssetType.LOGO,
        asset_subtype="repeating_header_logo",
        confidence=0.90,
        is_decorative=True,
    )
    # Repeating across 4 pages
    res = scorer.evaluate(
        candidate=cand,
        classification=classification,
        caption=None,
        frequency=4,
    )

    assert res.relevance_score < 0.35
    assert res.indexing_policy == IndexingPolicy.STORE_ONLY
    assert "logo" in res.relevance_reason.lower() or "repeated" in res.relevance_reason.lower()


def test_moderate_figure_with_ocr_routes_to_ocr_only():
    scorer = AssetRelevanceScorer(
        config=AssetConfig(
            analyze_threshold=0.85,
            store_only_threshold=0.30,
        )
    )
    cand = _make_candidate(width=300, height=200, page_coverage=0.08)
    classification = AssetClassificationResult(
        asset_type=AssetType.FIGURE,
        confidence=0.75,
        is_decorative=False,
    )
    ocr_result = FigureOCRResult(
        ocr_text="Step 1: Ingest Step 2: Validate Step 3: Embed",
        confidence=0.80,
        ocr_applied=True,
    )

    res = scorer.evaluate(
        candidate=cand,
        classification=classification,
        caption=None,
        ocr_result=ocr_result,
        frequency=1,
    )

    # Moderate score with substantial OCR text should route to OCR_ONLY
    assert 0.30 <= res.relevance_score < 0.85
    assert res.indexing_policy == IndexingPolicy.OCR_ONLY


def test_technical_context_boosts_score():
    scorer = AssetRelevanceScorer()
    cand = _make_candidate(width=350, height=250)
    classification = AssetClassificationResult(
        asset_type=AssetType.DIAGRAM,
        confidence=0.85,
    )
    context = AssetContextInfo(
        document_title="Distributed Transformer Scaling",
        section_heading="GPU Cluster Topology and Network Latency",
        nearby_text="Each worker node communicates via high throughput interface.",
    )

    res = scorer.evaluate(
        candidate=cand,
        classification=classification,
        context=context,
    )

    assert res.features.get("context_tech_matches", 0) >= 2
    assert "technical section context" in res.relevance_reason
