"""Integration and unit tests for Selective Vision Analysis and Chunk Filtering."""

from __future__ import annotations

import io
from PIL import Image, ImageDraw
import pytest

from rag.assets.analyzer import FigureAnalyzer
from rag.assets.models import (
    AssetContextInfo,
    ExtractedVisualCandidate,
    FigureAnalysisResult,
)
from rag.assets.processor import AssetProcessor
from rag.assets.providers.mock_provider import MockVisionProvider
from rag.core.config import AssetConfig
from rag.core.models import AssetType, IndexingPolicy
from rag.storage.local_storage import LocalRawStorageBackend


def _create_sample_png(width: int = 300, height: int = 200, color: str = "blue") -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TrackingVisionProvider(MockVisionProvider):
    """Mock vision provider that counts the number of times it was invoked."""
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def analyze_figure(self, *args, **kwargs):
        self.call_count += 1
        return super().analyze_figure(*args, **kwargs)


def test_selective_vision_skips_low_relevance_assets(tmp_path):
    storage_backend = LocalRawStorageBackend(base_dir=str(tmp_path / "raw_storage"))
    vision_provider = TrackingVisionProvider()
    analyzer = FigureAnalyzer(vision_provider=vision_provider)
    processor = AssetProcessor(
        config=AssetConfig(analyze_threshold=0.65, store_only_threshold=0.35),
        storage_backend=storage_backend,
        figure_analyzer=analyzer,
    )

    # 1. High value diagram with caption
    cand_diagram = ExtractedVisualCandidate(
        asset_id="11111111-1111-1111-1111-111111111111",
        document_id="00000000-0000-0000-0000-000000000001",
        version=1,
        page_number=1,
        bbox=(50, 50, 450, 350),
        image_bytes=_create_sample_png(400, 300, "teal"),
        width=400,
        height=300,
        nearby_text="System Architecture and workflow pipeline",
    )
    processed_diag = processor.process_candidate(
        candidate=cand_diagram,
        tenant_id="default",
        captions_on_page=[((50, 360, 450, 380), "Figure 1: Microservice Architecture Workflow")],
    )

    assert processed_diag is not None
    assert processed_diag.schema.indexing_policy == IndexingPolicy.VISION_ANALYSIS
    assert processed_diag.schema.relevance_score >= 0.65
    assert vision_provider.call_count == 1

    # 2. Low value tiny decorative image / logo
    cand_logo = ExtractedVisualCandidate(
        asset_id="22222222-2222-2222-2222-222222222222",
        document_id="00000000-0000-0000-0000-000000000001",
        version=1,
        page_number=2,
        bbox=(20, 20, 100, 50),
        image_bytes=_create_sample_png(80, 30, "gray"),
        width=80,
        height=30,
    )
    processed_logo = processor.process_candidate(
        candidate=cand_logo,
        tenant_id="default",
    )

    assert processed_logo is not None
    assert processed_logo.schema.indexing_policy == IndexingPolicy.STORE_ONLY
    assert processed_logo.schema.relevance_score < 0.35
    # Vision should NOT have been called again
    assert vision_provider.call_count == 1


def test_selective_chunk_creation(tmp_path):
    storage_backend = LocalRawStorageBackend(base_dir=str(tmp_path / "raw_storage"))
    processor = AssetProcessor(
        config=AssetConfig(analyze_threshold=0.65, store_only_threshold=0.35),
        storage_backend=storage_backend,
    )

    # Informative figure
    cand1 = ExtractedVisualCandidate(
        asset_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        document_id="00000000-0000-0000-0000-000000000001",
        version=1,
        page_number=1,
        bbox=(50, 50, 450, 350),
        image_bytes=_create_sample_png(400, 300),
        width=400,
        height=300,
    )
    processed1 = processor.process_candidate(
        candidate=cand1,
        tenant_id="default",
        captions_on_page=[((50, 360, 450, 380), "Figure 1: Ingestion Pipeline Topology")],
    )

    # Decorative header element
    cand2 = ExtractedVisualCandidate(
        asset_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        document_id="00000000-0000-0000-0000-000000000001",
        version=1,
        page_number=1,
        bbox=(10, 10, 80, 30),
        image_bytes=_create_sample_png(70, 20),
        width=70,
        height=20,
    )
    processed2 = processor.process_candidate(
        candidate=cand2,
        tenant_id="default",
    )

    chunks = processor.create_chunks_for_assets([processed1, processed2])
    # Only the informative asset generates a search chunk; the decorative asset is preserved in storage only
    assert len(chunks) == 1
    assert chunks[0].metadata["asset_id"] == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert "Figure 1: Ingestion Pipeline Topology" in chunks[0].text


def test_vision_budget_safety_limit(tmp_path):
    storage_backend = LocalRawStorageBackend(base_dir=str(tmp_path / "raw_storage"))
    vision_provider = TrackingVisionProvider()
    analyzer = FigureAnalyzer(vision_provider=vision_provider)
    # Configure budget limit of 2 calls
    processor = AssetProcessor(
        config=AssetConfig(max_vision_calls_per_document=2, analyze_threshold=0.50),
        storage_backend=storage_backend,
        figure_analyzer=analyzer,
    )

    # Process 3 high value visually distinct figures
    for i in range(1, 4):
        img = Image.new("RGB", (400, 300), color="white")
        draw = ImageDraw.Draw(img)
        if i == 1:
            draw.rectangle([20, 20, 380, 100], fill="blue")
        elif i == 2:
            draw.ellipse([50, 50, 350, 250], fill="red")
        else:
            draw.line([(0, 0), (400, 300)], fill="green", width=20)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        cand = ExtractedVisualCandidate(
            asset_id=f"00000000-0000-0000-0000-00000000000{i}",
            document_id="00000000-0000-0000-0000-000000000001",
            version=1,
            page_number=i,
            bbox=(50, 50, 450, 350),
            image_bytes=buf.getvalue(),
            width=400,
            height=300,
        )
        processor.process_candidate(
            candidate=cand,
            tenant_id="default",
            captions_on_page=[((50, 360, 450, 380), f"Figure {i}: Architecture Diagram {i}")],
        )

    # Vision provider must have been invoked at most 2 times
    assert vision_provider.call_count == 2
