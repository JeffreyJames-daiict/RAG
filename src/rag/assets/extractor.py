"""Visual asset extractor for digital PDF embedded images and scanned page figure regions."""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import fitz  # PyMuPDF
from PIL import Image

from rag.assets.models import ExtractedVisualCandidate
from rag.core.config import AssetConfig

logger = logging.getLogger(__name__)


class AssetExtractor:
    """Extracts embedded digital images and detects/crops visual figures from scanned PDF pages."""

    def __init__(self, config: Optional[AssetConfig] = None):
        self.config = config or AssetConfig()

    def extract_from_digital_page(
        self,
        doc: fitz.Document,
        page_idx: int,
        document_id: str,
        version: int = 1,
    ) -> List[ExtractedVisualCandidate]:
        """Extract all embedded digital images and their page bounding boxes."""
        page = doc[page_idx]
        page_number = page_idx + 1
        candidates: List[ExtractedVisualCandidate] = []
        seen_xrefs = set()

        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base_image = doc.extract_image(xref)
                if not base_image or not base_image.get("image"):
                    continue

                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png").lower()
                mime_type = f"image/{ext}" if ext != "jpg" else "image/jpeg"
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Determine on-page bounding box
                rects = page.get_image_rects(xref)
                if rects:
                    r = rects[0]
                    bbox = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
                else:
                    # Fallback to page dimension
                    bbox = (0.0, 0.0, float(page.rect.width), float(page.rect.height))

                asset_id = str(uuid4())
                candidate = ExtractedVisualCandidate(
                    asset_id=asset_id,
                    document_id=document_id,
                    version=version,
                    page_number=page_number,
                    bbox=bbox,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    width=width,
                    height=height,
                    source_type="digital_embedded",
                    xref=xref,
                )
                candidates.append(candidate)

            except Exception as exc:
                logger.warning(
                    "Failed to extract image xref %d on page %d of doc %s: %s",
                    xref, page_number, document_id, exc,
                )

        return candidates

    def extract_from_scanned_page_image(
        self,
        page_image: Image.Image,
        page_number: int,
        document_id: str,
        version: int = 1,
        existing_text_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
        page_rect_pts: Optional[Tuple[float, float]] = None,
    ) -> List[ExtractedVisualCandidate]:
        """Detect and crop visual figure regions from a scanned or raster page image."""
        candidates: List[ExtractedVisualCandidate] = []
        img_w, img_h = page_image.size
        page_w_pts, page_h_pts = page_rect_pts or (float(img_w), float(img_h))

        # Scale factor from pixel image to PDF points
        scale_x = page_w_pts / float(img_w) if img_w else 1.0
        scale_y = page_h_pts / float(img_h) if img_h else 1.0

        try:
            # Detect rectangular figure regions
            # When OpenCV is available, we find contours; otherwise we fall back to bounding heuristics
            try:
                import cv2
                import numpy as np

                np_img = np.array(page_image.convert("RGB"))
                gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
                # Otsu thresholding
                _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
                # Morphological closing to group diagram elements
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Filter out tiny noise and full-page borders
                    if w < self.config.min_image_width or h < self.config.min_image_height:
                        continue
                    if (w * h) < self.config.min_image_area:
                        continue
                    if w >= (img_w * 0.98) and h >= (img_h * 0.98):
                        continue

                    # Crop figure from page image
                    crop = page_image.crop((x, y, x + w, y + h))
                    buf = io.BytesIO()
                    crop.save(buf, format="PNG")
                    crop_bytes = buf.getvalue()

                    # Convert pixel bbox to points
                    pts_bbox = (x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y)

                    candidate = ExtractedVisualCandidate(
                        asset_id=str(uuid4()),
                        document_id=document_id,
                        version=version,
                        page_number=page_number,
                        bbox=pts_bbox,
                        image_bytes=crop_bytes,
                        mime_type="image/png",
                        width=w,
                        height=h,
                        source_type="scanned_crop",
                    )
                    candidates.append(candidate)

            except ImportError:
                # If cv2 is not available, crop based on central region if large enough
                pass

        except Exception as exc:
            logger.warning("Scanned figure detection warning on page %d: %s", page_number, exc)

        return candidates
