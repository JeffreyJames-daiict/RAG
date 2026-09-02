"""Unit tests for SHA-256 and Perceptual Hash deduplication of visual document assets."""

from __future__ import annotations

import io
from PIL import Image, ImageDraw

from rag.assets.deduplicator import (
    AssetDeduplicator,
    compute_phash,
    compute_sha256,
    hamming_distance,
)
from rag.assets.models import FigureAnalysisResult, FigureOCRResult


def _create_test_image(color: str = "red", shape: str = "rectangle", size: tuple = (200, 200)) -> bytes:
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    if shape == "rectangle":
        draw.rectangle([20, 20, 180, 180], fill=color)
    elif shape == "circle":
        draw.ellipse([20, 20, 180, 180], fill=color)
    elif shape == "text":
        draw.rectangle([10, 10, 190, 50], fill=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_exact_sha256_hashing():
    img1 = _create_test_image(color="blue", shape="rectangle")
    img2 = _create_test_image(color="blue", shape="rectangle")
    img3 = _create_test_image(color="green", shape="rectangle")

    hash1 = compute_sha256(img1)
    hash2 = compute_sha256(img2)
    hash3 = compute_sha256(img3)

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64


def test_perceptual_phash_dhash():
    # 1. Base image: circle
    img1 = _create_test_image(color="blue", shape="circle", size=(200, 200))
    # 2. Rescaled version of the exact same visual content
    img1_pil = Image.open(io.BytesIO(img1))
    buf2 = io.BytesIO()
    img1_pil.resize((260, 260), resample=Image.Resampling.LANCZOS).save(buf2, format="PNG")
    img2 = buf2.getvalue()

    # 3. Completely different pattern: asymmetric stripes
    img_diff = Image.new("RGB", (200, 200), color="white")
    d = ImageDraw.Draw(img_diff)
    d.line([(0, 0), (200, 200)], fill="black", width=25)
    d.rectangle([100, 0, 200, 100], fill="black")
    buf3 = io.BytesIO()
    img_diff.save(buf3, format="PNG")
    img3 = buf3.getvalue()

    phash1 = compute_phash(img1)
    phash2 = compute_phash(img2)
    phash3 = compute_phash(img3)

    assert len(phash1) == 16
    assert len(phash2) == 16

    # Rescaled version should have 0 or very small hamming distance
    dist_same = hamming_distance(phash1, phash2)
    assert dist_same <= 4

    # Completely different visual pattern should have distinctly larger distance
    dist_diff = hamming_distance(phash1, phash3)
    assert dist_diff > dist_same


def test_deduplicator_exact_match_and_frequency():
    dedup = AssetDeduplicator(dedup_enabled=True)
    img_logo = _create_test_image(color="red", shape="rectangle")

    # Page 1 occurrence
    sha1, phash1, is_dup1, orig1, freq1 = dedup.evaluate_asset(img_logo, asset_id="logo_p1", page_number=1)
    assert is_dup1 is False
    assert orig1 is None
    assert freq1 == 1

    # Page 2 occurrence (exact same image bytes)
    sha2, phash2, is_dup2, orig2, freq2 = dedup.evaluate_asset(img_logo, asset_id="logo_p2", page_number=2)
    assert is_dup2 is True
    assert orig2 == "logo_p1"
    assert freq2 == 2
    assert sha1 == sha2

    # Page 3 occurrence
    _, _, is_dup3, orig3, freq3 = dedup.evaluate_asset(img_logo, asset_id="logo_p3", page_number=3)
    assert is_dup3 is True
    assert orig3 == "logo_p1"
    assert freq3 == 3
    assert dedup.get_frequency(sha1) == 3


def test_deduplicator_analysis_caching():
    dedup = AssetDeduplicator()
    img_bytes = _create_test_image(color="purple", shape="circle")
    sha, phash, _, _, _ = dedup.evaluate_asset(img_bytes, asset_id="fig_1", page_number=1)

    mock_analysis = FigureAnalysisResult(
        description="High level diagram of caching workflow",
        entities=["Cache", "Storage"],
        status="SUCCESS",
    )
    mock_ocr = FigureOCRResult(ocr_text="Cache -> Storage", confidence=0.95, ocr_applied=True)

    dedup.cache_analysis(content_hash=sha, perceptual_hash=phash, analysis=mock_analysis, ocr=mock_ocr)

    # Retrieve cached analysis by SHA-256
    cached_res = dedup.get_cached_analysis(sha, phash)
    assert cached_res is not None
    cached_analysis, cached_ocr = cached_res
    assert cached_analysis.description == "High level diagram of caching workflow"
    assert cached_ocr.ocr_text == "Cache -> Storage"
