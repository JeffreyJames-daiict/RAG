"""OCR-specific normalization, noise detection, and quality flagging.

Evaluates OCR output text without speculative hallucination or silent alteration,
recording quality flags, detecting OCR-specific artifacts (repetition, broken characters,
extreme symbol density), and applying safe spacing fixes.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from rag.normalization.models import NormalizedContentBlock

logger = logging.getLogger(__name__)

# Pattern for runs of 4+ identical punctuation/symbols (e.g. "------", ".......", "=====")
_REPEATED_SYMBOLS_PATTERN = re.compile(r"([^a-zA-Z0-9\s])\1{3,}")

# Pattern for runs of 5+ identical alphabetic characters (e.g. "aaaaaa", "xxxxxx")
_REPEATED_CHARS_PATTERN = re.compile(r"([a-zA-Z])\1{4,}")

# Pattern for broken spaced words (e.g. "w o r d s   l i k e   t h i s")
_BROKEN_SPACING_PATTERN = re.compile(r"\b([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\b")


# Pattern for isolated graphical divider / arrow noise lines (e.g. "————-_-—»}", "====>", "-------___---")
_GRAPHICAL_DIVIDER_PATTERN = re.compile(
    r"^[\s—–\-_=>»\>\|\~\+\*\#\{\}\[\]\(\)\.\,\/\\:;^`']+$"
)

# Common deterministic OCR corruption of standard URL protocols and prefixes
_URL_PROTOCOL_REPAIRS = [
    (re.compile(r"\b(?:hitps|htps|hftps|hwtps):\/\/", re.IGNORECASE), "https://"),
    (re.compile(r"\b(?:hitp|htp|hftp|hwtp):\/\/", re.IGNORECASE), "http://"),
    (re.compile(r"\b(?:wwrw|wvwv|vvww)\.", re.IGNORECASE), "www."),
]

# Pattern for disparate OCR bullets at the start of a line
_OCR_BULLET_LINE_PATTERN = re.compile(r"^[•°◦o\*]\s+", re.MULTILINE)


class OCRNormalizer:
    """Evaluates and safely normalizes OCR-extracted content blocks."""

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        max_symbol_ratio: float = 0.30,
        excessive_repetition_threshold: float = 0.70,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_symbol_ratio = max_symbol_ratio
        self.excessive_repetition_threshold = excessive_repetition_threshold

    def process_block(self, block: NormalizedContentBlock) -> NormalizedContentBlock:
        """Evaluate and flag OCR content block, recording quality flags and safe fixes."""
        text = block.normalized_text
        if not text:
            return block

        # 1. Strip standalone graphical noise / divider lines
        cleaned_div, div_stripped = self._strip_graphical_divider_lines(text)
        if div_stripped:
            text = cleaned_div
            block.normalized_text = text
            block.add_action("graphical_dividers_stripped")

        # 2. Repair deterministic URL protocol and prefix corruptions
        cleaned_url, url_repaired = self._repair_ocr_url_protocols(text)
        if url_repaired:
            text = cleaned_url
            block.normalized_text = text
            block.add_action("url_protocols_repaired")

        # 3. Standardize disparate OCR bullets to standard Markdown list items
        cleaned_bullets, bullets_standardized = self._standardize_bullet_markers(text)
        if bullets_standardized:
            text = cleaned_bullets
            block.normalized_text = text
            block.add_action("bullet_markers_standardized")

        # 4. Check OCR confidence
        if block.confidence_score < self.confidence_threshold:
            block.add_flag("low_confidence")

        # 5. Check symbol density
        total_len = len(text)
        if total_len > 0:
            symbol_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
            symbol_ratio = symbol_count / total_len
            if symbol_ratio > self.max_symbol_ratio and total_len >= 15:
                block.add_flag("high_symbol_density")

        # 6. Check for excessive repeated characters / noise artifacts
        if _REPEATED_CHARS_PATTERN.search(text):
            block.add_flag("excessive_character_repetition")
            block.add_flag("ocr_noise_detected")

        if _REPEATED_SYMBOLS_PATTERN.search(text):
            block.add_flag("repeated_symbol_artifacts")

        # 7. Check for broken letter spacing (e.g. "e x a m p l e") and safely fix if isolated
        fixed_text, spacing_fixed = self._repair_broken_letter_spacing(text)
        if spacing_fixed:
            block.normalized_text = fixed_text
            block.add_action("ocr_spacing_repaired")

        # 8. Check for suspected corruption / garbled content
        if "high_symbol_density" in block.quality_flags and "low_confidence" in block.quality_flags:
            block.add_flag("suspected_corruption")

        return block

    def _strip_graphical_divider_lines(self, text: str) -> Tuple[str, bool]:
        """Strip standalone lines consisting solely of graphic/diagram symbols."""
        if not text:
            return "", False

        lines = text.split("\n")
        filtered: List[str] = []
        applied = False

        for line in lines:
            stripped = line.strip()
            # If line is entirely punctuation/symbols and at least 3 chars long
            if stripped and len(stripped) >= 3 and _GRAPHICAL_DIVIDER_PATTERN.match(stripped):
                applied = True
                continue
            filtered.append(line)

        return "\n".join(filtered).strip(), applied

    def _repair_ocr_url_protocols(self, text: str) -> Tuple[str, bool]:
        """Repair common deterministic OCR errors in URLs (e.g. hitps:// -> https://)."""
        if not text:
            return "", False

        current = text
        applied = False
        for pattern, replacement in _URL_PROTOCOL_REPAIRS:
            if pattern.search(current):
                current = pattern.sub(replacement, current)
                applied = True

        return current, applied

    def _standardize_bullet_markers(self, text: str) -> Tuple[str, bool]:
        """Convert varied OCR bullet markers (•, °, ◦, *, o) to standard markdown '- '."""
        if not text:
            return "", False

        if _OCR_BULLET_LINE_PATTERN.search(text):
            repaired = _OCR_BULLET_LINE_PATTERN.sub("- ", text)
            return repaired, repaired != text

        return text, False

    def _repair_broken_letter_spacing(self, text: str) -> Tuple[str, bool]:
        """Safely repair sequences of single spaced letters forming words."""
        if not text or len(text) < 7:
            return text, False

        def _merge_match(match: re.Match) -> str:
            return f"{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}"

        if _BROKEN_SPACING_PATTERN.search(text):
            repaired = _BROKEN_SPACING_PATTERN.sub(_merge_match, text)
            return repaired, repaired != text

        return text, False
