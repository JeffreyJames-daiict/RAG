"""Document-level repetitive header and footer detector and remover.

Scans across pages in a document to detect identical or near-identical header
and footer artifacts (e.g. 'CONFIDENTIAL', document titles, running headers)
appearing consistently at page boundaries, and strips them with audit provenance.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HeaderFooterDetectionResult:
    """Detected header and footer patterns across a document."""
    detected_headers: Set[str] = field(default_factory=set)
    detected_footers: Set[str] = field(default_factory=set)
    header_page_counts: Dict[str, int] = field(default_factory=dict)
    footer_page_counts: Dict[str, int] = field(default_factory=dict)


class HeaderFooterDetector:
    """Detects and removes repeating header and footer patterns across multiple pages."""

    def __init__(
        self,
        header_repeat_threshold: int = 3,
        footer_repeat_threshold: int = 3,
        max_boundary_lines: int = 2,
        min_line_length: int = 4,
        max_line_length: int = 120,
    ):
        self.header_repeat_threshold = header_repeat_threshold
        self.footer_repeat_threshold = footer_repeat_threshold
        self.max_boundary_lines = max_boundary_lines
        self.min_line_length = min_line_length
        self.max_line_length = max_line_length

    def analyze_document(self, page_texts: List[str]) -> HeaderFooterDetectionResult:
        """Scan all page texts in a document to find recurring header/footer candidates."""
        if len(page_texts) < min(self.header_repeat_threshold, self.footer_repeat_threshold):
            return HeaderFooterDetectionResult()

        header_candidates: List[str] = []
        footer_candidates: List[str] = []

        for page_text in page_texts:
            lines = [l.strip() for l in page_text.split("\n") if l.strip()]
            if not lines:
                continue

            # Header candidates from top lines
            top_lines = lines[:self.max_boundary_lines]
            for line in top_lines:
                norm = self._canonicalize_line(line)
                if self._is_valid_candidate(norm):
                    header_candidates.append(norm)

            # Footer candidates from bottom lines
            bottom_lines = lines[-self.max_boundary_lines:]
            for line in bottom_lines:
                norm = self._canonicalize_line(line)
                if self._is_valid_candidate(norm):
                    footer_candidates.append(norm)

        # Count frequencies across pages
        header_counts = Counter(header_candidates)
        footer_counts = Counter(footer_candidates)

        detected_headers = {
            pat for pat, cnt in header_counts.items()
            if cnt >= self.header_repeat_threshold
        }
        detected_footers = {
            pat for pat, cnt in footer_counts.items()
            if cnt >= self.footer_repeat_threshold
        }

        logger.debug(
            "Detected %d repetitive headers and %d repetitive footers across %d pages",
            len(detected_headers), len(detected_footers), len(page_texts),
        )

        return HeaderFooterDetectionResult(
            detected_headers=detected_headers,
            detected_footers=detected_footers,
            header_page_counts=dict(header_counts),
            footer_page_counts=dict(footer_counts),
        )

    def strip_headers_and_footers(
        self,
        page_text: str,
        detection: HeaderFooterDetectionResult,
    ) -> Tuple[str, List[str], List[str]]:
        """Remove detected headers and footers from a single page's text.

        Returns:
            Tuple of (cleaned_page_text, list_of_removed_headers, list_of_removed_footers).
        """
        if not page_text:
            return "", [], []

        if not detection.detected_headers and not detection.detected_footers:
            return page_text, [], []

        lines = page_text.split("\n")
        removed_headers: List[str] = []
        removed_footers: List[str] = []

        # Check top lines for headers
        start_idx = 0
        while start_idx < min(len(lines), self.max_boundary_lines * 2):
            raw_line = lines[start_idx].strip()
            norm_line = self._canonicalize_line(raw_line)
            if norm_line and norm_line in detection.detected_headers:
                removed_headers.append(raw_line)
                start_idx += 1
            elif not raw_line:
                start_idx += 1
            else:
                break

        # Check bottom lines for footers
        end_idx = len(lines)
        while end_idx > max(start_idx, len(lines) - self.max_boundary_lines * 2):
            raw_line = lines[end_idx - 1].strip()
            norm_line = self._canonicalize_line(raw_line)
            if norm_line and norm_line in detection.detected_footers:
                removed_footers.append(raw_line)
                end_idx -= 1
            elif not raw_line:
                end_idx -= 1
            else:
                break

        remaining_lines = lines[start_idx:end_idx]
        cleaned_text = "\n".join(remaining_lines).strip()
        return cleaned_text, removed_headers, removed_footers

    def _canonicalize_line(self, line: str) -> str:
        """Strip surrounding punctuation, multiple spaces, and numbers for matching."""
        # Replace standalone page numbers or date stamps within header with placeholder
        cleaned = re.sub(r"\bpage\s+\d+(?:\s+(?:of|\/)\s+\d+)?\b", "[PAGE]", line.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned.lower()

    def _is_valid_candidate(self, line: str) -> bool:
        """Check if line is a reasonable header/footer candidate."""
        if not (self.min_line_length <= len(line) <= self.max_line_length):
            return False
        # Discard if it contains excessive alphanumeric prose (> 20 words)
        words = line.split()
        if len(words) > 15:
            return False
        return True
