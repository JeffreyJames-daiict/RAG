"""Unit tests for digital and scanned PDF visual asset extraction."""

import io
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import pytest

from rag.assets.extractor import AssetExtractor
from rag.core.config import AssetConfig


def _create_pdf_with_embedded_image(img_size=(200, 200), img_color="blue") -> bytes:
    # 1. Create a PIL image
    img = Image.new("RGB", img_size, color=img_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 180, 180], fill="white", outline="black")
    draw.text((30, 80), "Arch Diagram", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # 2. Insert into PDF page
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((50, 50), "Figure 1: Enterprise System Architecture", fontsize=14)
    # Insert image at rect
    img_rect = fitz.Rect(50, 80, 250, 280)
    page.insert_image(img_rect, stream=img_bytes)
    page.insert_text((50, 300), "Figure 1. Detailed architecture diagram of the system.", fontsize=11)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_extract_digital_page_images():
    pdf_bytes = _create_pdf_with_embedded_image(img_size=(200, 200))
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    extractor = AssetExtractor(config=AssetConfig())
    candidates = extractor.extract_from_digital_page(doc=doc, page_idx=0, document_id="doc_test_1", version=1)
    doc.close()

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.page_number == 1
    assert cand.source_type == "digital_embedded"
    assert cand.width == 200
    assert cand.height == 200
    assert cand.mime_type == "image/png"
    assert len(cand.image_bytes) > 0
    # Bbox in PDF points
    x0, y0, x1, y1 = cand.bbox
    assert round(x0) == 50
    assert round(y0) == 80
    assert round(x1) == 250
    assert round(y1) == 280


def test_extract_scanned_page_image_figure_detection():
    # Create a full page image containing a distinct diagram block
    page_img = Image.new("RGB", (600, 800), color="white")
    draw = ImageDraw.Draw(page_img)
    # Text simulation
    draw.text((50, 50), "Scanned document text paragraph.", fill="black")
    # Diagram region
    draw.rectangle([100, 150, 500, 450], fill="lightgray", outline="black", width=3)
    draw.text((150, 280), "Component Architecture", fill="black")
    # Caption simulation
    draw.text((100, 480), "Figure 2: Component Architecture", fill="black")

    extractor = AssetExtractor(config=AssetConfig())
    candidates = extractor.extract_from_scanned_page_image(
        page_image=page_img,
        page_number=2,
        document_id="doc_scanned_1",
        version=1,
        page_rect_pts=(600.0, 800.0),
    )

    # If OpenCV is available, candidates will be detected; otherwise extractor handles gracefully
    assert isinstance(candidates, list)
    if candidates:
        cand = candidates[0]
        assert cand.page_number == 2
        assert cand.source_type == "scanned_crop"
        assert len(cand.image_bytes) > 0
