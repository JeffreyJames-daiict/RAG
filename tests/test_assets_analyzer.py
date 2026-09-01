"""Unit tests for figure vision analysis and failure isolation."""

import io
from PIL import Image
import pytest

from rag.assets.analyzer import FigureAnalyzer
from rag.assets.exceptions import VisionAnalysisError
from rag.assets.providers.base import BaseVisionProvider
from rag.assets.providers.mock_provider import MockVisionProvider
from rag.core.config import AssetConfig, VisionConfig
from rag.core.models import AssetType


def _make_dummy_image_bytes() -> bytes:
    img = Image.new("RGB", (200, 200), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_mock_vision_provider_analysis():
    provider = MockVisionProvider()
    img_bytes = _make_dummy_image_bytes()

    res = provider.analyze_figure(
        image_bytes=img_bytes,
        caption="Figure 1: RAG System Architecture",
        ocr_text="Ingestion Storage Chunking Embedding VectorDB",
        asset_type=AssetType.ARCHITECTURE_DIAGRAM,
    )

    assert res.status == "SUCCESS"
    assert len(res.entities) >= 3
    assert len(res.relationships) >= 2
    assert "architecture" in res.description.lower()
    assert "Vector" in " ".join(res.entities) or "Ingestion" in " ".join(res.entities)


class FailingVisionProvider(BaseVisionProvider):
    def analyze_figure(self, *args, **kwargs):
        raise VisionAnalysisError("Simulated remote vision provider 500 error")


def test_figure_analyzer_failure_isolation():
    # Verify that vision failure does not crash the pipeline and returns a structured fallback
    cfg = AssetConfig(vision=VisionConfig(enabled=True, max_retries=1))
    analyzer = FigureAnalyzer(config=cfg, vision_provider=FailingVisionProvider())
    img_bytes = _make_dummy_image_bytes()

    res = analyzer.analyze(
        image_bytes=img_bytes,
        asset_type=AssetType.ARCHITECTURE_DIAGRAM,
        caption="Figure 2: Pipeline Design",
        ocr_text="Extractor Normalizer Embedder",
    )

    assert res.status == "FAILED"
    assert "Simulated remote vision provider" in res.error_message
    assert "Figure 2: Pipeline Design" in res.description
