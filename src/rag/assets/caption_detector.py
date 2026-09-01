"""Spatial and pattern-based caption detection and association engine."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Regular expressions matching standard figure / diagram caption patterns
_CAPTION_PATTERNS = [
    re.compile(r"^(?:Figure|Fig\.?|Diagram|Chart|Illustration|Exhibit|Image)\s*(\d+[A-Za-z0-9\.\-\_]*)\s*[:.\-—]\s*(.+)", re.IGNORECASE),
    re.compile(r"^(?:Figure|Fig\.?|Diagram|Chart|Illustration|Exhibit|Image)\s*(\d+[A-Za-z0-9\.\-\_]*)\s*$", re.IGNORECASE),
    re.compile(r"^(?:Architecture|System\s+Overview|Data\s+Flow)\s*[:.\-—]\s*(.+)", re.IGNORECASE),
]


class CaptionDetector:
    """Detects figure captions on a page and spatially associates them with visual assets."""

    def __init__(self, vertical_tolerance_pts: float = 80.0):
        self.vertical_tolerance_pts = vertical_tolerance_pts

    def is_caption_text(self, text: str) -> bool:
        """Check if a single text line matches a caption regex pattern."""
        clean = text.strip()
        if not clean:
            return False
        for pat in _CAPTION_PATTERNS:
            if pat.match(clean):
                return True
        return False

    def find_captions_on_page(
        self,
        text_blocks: List[Tuple[float, float, float, float, str]],
    ) -> List[Tuple[Tuple[float, float, float, float], str]]:
        """Identify all caption candidates and their bounding boxes from page text blocks.

        Args:
            text_blocks: List of (x0, y0, x1, y1, text) tuples.

        Returns:
            List of ((x0, y0, x1, y1), caption_text) tuples.
        """
        captions = []
        for x0, y0, x1, y1, text in text_blocks:
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            for line in lines:
                if self.is_caption_text(line):
                    captions.append(((x0, y0, x1, y1), line))
                    break
        return captions

    def associate_caption_to_figure(
        self,
        figure_bbox: Tuple[float, float, float, float],
        captions: List[Tuple[Tuple[float, float, float, float], str]],
    ) -> Optional[str]:
        """Find the most spatially appropriate caption for a figure bounding box."""
        if not captions:
            return None

        fx0, fy0, fx1, fy1 = figure_bbox
        best_caption = None
        min_distance = float("inf")

        for (cx0, cy0, cx1, cy1), cap_text in captions:
            # 1. Check if caption is below the figure (most common case in academic/enterprise docs)
            if cy0 >= fy1 - 10:
                v_dist = cy0 - fy1
                # Prefer caption within configured vertical tolerance
                if 0 <= v_dist <= self.vertical_tolerance_pts:
                    # Check horizontal overlap or alignment
                    h_overlap = max(0.0, min(fx1, cx1) - max(fx0, cx0))
                    total_dist = v_dist - (h_overlap * 0.1)  # Reward horizontal overlap
                    if total_dist < min_distance:
                        min_distance = total_dist
                        best_caption = cap_text

            # 2. Check if caption is above the figure
            elif cy1 <= fy0 + 10:
                v_dist = fy0 - cy1
                if 0 <= v_dist <= self.vertical_tolerance_pts:
                    h_overlap = max(0.0, min(fx1, cx1) - max(fx0, cx0))
                    total_dist = (v_dist * 1.2) - (h_overlap * 0.1)  # Slight penalty vs below
                    if total_dist < min_distance:
                        min_distance = total_dist
                        best_caption = cap_text

        return best_caption
