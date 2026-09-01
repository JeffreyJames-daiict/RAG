"""Unit tests for asset classification and decorative filtering."""

from PIL import Image
import io
import pytest

from rag.assets.classifier import AssetClassifier
from rag.assets.models import ExtractedVisualCandidate
from rag.core.config import AssetConfig
from rag.core.models import AssetType


def _make_candidate(w: int, h: int, nearby_text: str = "") -> ExtractedVisualCandidate:
    img = Image.new("RGB", (w, h), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return ExtractedVisualCandidate(
        asset_id="test_cand_1",
        document_id="doc_1",
        version=1,
        page_number=1,
        bbox=(50.0, 50.0, 50.0 + w, 50.0 + h),
        image_bytes=buf.getvalue(),
        mime_type="image/png",
        width=w,
        height=h,
        nearby_text=nearby_text,
    )


def test_classify_filters_tiny_icon_as_decorative():
    classifier = AssetClassifier(config=AssetConfig(filter_decorative=True, min_image_width=64, min_image_height=64))
    cand = _make_candidate(w=32, h=32)
    res = classifier.classify(cand)
    assert res.is_decorative is True
    assert res.asset_type == AssetType.DECORATIVE


def test_classify_filters_banner_as_decorative():
    classifier = AssetClassifier(config=AssetConfig(filter_decorative=True, max_aspect_ratio=8.0))
    cand = _make_candidate(w=800, h=40)  # 20:1 aspect ratio
    res = classifier.classify(cand)
    assert res.is_decorative is True
    assert res.asset_type == AssetType.DECORATIVE


def test_classify_architecture_diagram():
    classifier = AssetClassifier(config=AssetConfig())
    cand = _make_candidate(w=400, h=300)
    res = classifier.classify(cand, caption="Figure 3: System Architecture Overview")
    assert res.is_decorative is False
    assert res.asset_type == AssetType.ARCHITECTURE_DIAGRAM


def test_classify_flowchart():
    classifier = AssetClassifier(config=AssetConfig())
    cand = _make_candidate(w=400, h=300)
    res = classifier.classify(cand, caption="Flowchart of document ingestion workflow")
    assert res.is_decorative is False
    assert res.asset_type == AssetType.FLOWCHART


def test_classify_chart():
    classifier = AssetClassifier(config=AssetConfig())
    cand = _make_candidate(w=400, h=300)
    res = classifier.classify(cand, caption="Figure 4: Bar chart of latency comparison")
    assert res.is_decorative is False
    assert res.asset_type == AssetType.CHART


def test_classify_fallback_standard_figure():
    classifier = AssetClassifier(config=AssetConfig())
    cand = _make_candidate(w=400, h=300)
    res = classifier.classify(cand, caption="Figure 5: Experimental setup")
    assert res.is_decorative is False
    assert res.asset_type == AssetType.FIGURE
