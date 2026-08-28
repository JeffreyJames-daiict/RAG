"""Safe, rule-based text normalization engine for RAG ingestion.

Handles whitespace normalization, unicode sanitation, control character removal,
line break soft-wrap repair, dehyphenation with compound preservation, and
artifact filtering without destroying paragraph boundaries or multilingual content.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)

# Zero-width and invisible characters to strip
_ZERO_WIDTH_CHARS = {
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # zero-width no-break space / byte order mark
    "\u2060",  # word joiner
    "\u00ad",  # soft hyphen
}

# Non-breaking space variants to convert to regular ASCII space
_NBSP_CHARS = {
    "\u00a0",  # no-break space
    "\u202f",  # narrow no-break space
    "\u2007",  # figure space
}

# Known legitimate prefixes that should retain hyphens when split across lines
_LEGITIMATE_HYPHEN_PREFIXES: Set[str] = {
    "self", "non", "pre", "post", "anti", "co", "sub", "super", "hyper",
    "ultra", "multi", "cross", "inter", "intra", "semi", "pseudo", "quasi",
    "all", "well", "ill", "high", "low", "long", "short", "open", "closed",
    "end", "real", "state", "user", "server", "client", "cost", "risk",
}

# Common hyphenated terms that must retain their hyphen
_LEGITIMATE_COMPOUNDS: Set[str] = {
    "state-of-the-art",
    "end-to-end",
    "real-time",
    "model-based",
    "fine-tuning",
    "fine-tuned",
    "zero-shot",
    "few-shot",
    "out-of-the-box",
    "plug-and-play",
    "open-source",
    "built-in",
    "cross-platform",
    "multi-agent",
    "multi-table",
    "token-bounded",
    "decision-making",
    "trade-off",
    "trade-offs",
    "point-to-point",
    "high-level",
    "low-level",
    "large-scale",
    "small-scale",
}

# Pattern detecting standalone page number lines (e.g., "1", "- 1 -", "Page 1 of 10")
_PAGE_NUMBER_PATTERN = re.compile(
    r"^(?:page\s+\d+(?:\s+(?:of|\/)\s+\d+)?|[-—–]\s*\d+\s*[-—–]|\d+)\s*$",
    re.IGNORECASE,
)

# Pattern for list items or bullet points
_LIST_BULLET_PATTERN = re.compile(r"^\s*(?:[•\-\*\+✓✗▪▫►◦]|\d+[\.\)]|[a-zA-Z][\.\)])\s+")

# Pattern for heading-like short lines
_HEADING_CANDIDATE = re.compile(r"^(?:[A-Z0-9\s\-_:,\.]{3,60}|#{1,6}\s+.*)$")


class TextNormalizer:
    """Deterministic, auditable text normalizer for extracted document content."""

    def __init__(
        self,
        unicode_form: str = "NFC",
        repair_soft_hyphens: bool = True,
        remove_standalone_page_numbers: bool = True,
    ):
        self.unicode_form = unicode_form.upper() if unicode_form else "NFC"
        self.repair_soft_hyphens = repair_soft_hyphens
        self.remove_standalone_page_numbers = remove_standalone_page_numbers

    def normalize(self, text: str) -> Tuple[str, List[str]]:
        """Run all normalization stages on text, returning (normalized_text, actions_taken).

        Args:
            text: Raw input text.

        Returns:
            Tuple of (cleaned_text, list_of_action_tags).
        """
        if not text or not text.strip():
            return "", []

        actions: List[str] = []
        current = text

        # 1. Unicode Normalization
        cleaned_unicode, uni_applied = self.normalize_unicode(current)
        if uni_applied:
            actions.append("unicode_normalization")
            current = cleaned_unicode

        # 2. Control Characters & Invisible Character Stripping
        cleaned_ctrl, ctrl_applied = self.remove_control_characters(current)
        if ctrl_applied:
            actions.append("control_characters_removed")
            current = cleaned_ctrl

        # 3. Line Endings & Soft Hyphen Dehyphenation
        if self.repair_soft_hyphens:
            cleaned_hyphens, hyphens_applied = self.repair_hyphenation(current)
            if hyphens_applied:
                actions.append("dehyphenation")
                current = cleaned_hyphens

        # 4. Standalone Page Number Removal
        if self.remove_standalone_page_numbers:
            cleaned_pg, pg_applied = self.filter_standalone_page_numbers(current)
            if pg_applied:
                actions.append("page_numbers_removed")
                current = cleaned_pg

        # 5. Paragraph and Whitespace Formatting
        cleaned_ws, ws_applied = self.normalize_whitespace(current)
        if ws_applied:
            actions.append("whitespace_normalization")
            current = cleaned_ws

        return current, actions

    def normalize_unicode(self, text: str) -> Tuple[str, bool]:
        """Apply Unicode normalization (NFC/NFKC) and replace non-breaking spaces."""
        if not text:
            return "", False

        original = text
        # Convert NBSP to normal space
        for ch in _NBSP_CHARS:
            if ch in text:
                text = text.replace(ch, " ")

        # Strip zero-width non-printable characters
        for ch in _ZERO_WIDTH_CHARS:
            if ch in text:
                text = text.replace(ch, "")

        # Standardize curly quotes and apostrophes to clean equivalents where standard
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("“", '"').replace("”", '"')

        if self.unicode_form in ("NFC", "NFKC", "NFD", "NFKD"):
            text = unicodedata.normalize(self.unicode_form, text)

        applied = text != original
        return text, applied

    def remove_control_characters(self, text: str) -> Tuple[str, bool]:
        """Strip raw ASCII/Unicode control characters while preserving valid formatting."""
        if not text:
            return "", False

        # Keep \n, \t, \r, and \f (which we turn to \n)
        cleaned_chars = []
        applied = False

        for ch in text:
            code = ord(ch)
            if code == 12:  # \f form feed
                cleaned_chars.append("\n")
                applied = True
            elif code < 32 and ch not in "\n\t\r":
                applied = True
                continue
            elif code == 127 or (0x80 <= code <= 0x9F):
                applied = True
                continue
            elif ch == "\ufffd":  # Unicode replacement character
                applied = True
                continue
            else:
                cleaned_chars.append(ch)

        return "".join(cleaned_chars), applied

    def repair_hyphenation(self, text: str) -> Tuple[str, bool]:
        """Merge words broken across line endings with hyphens, preserving legitimate compounds."""
        if not text:
            return "", False

        lines = text.split("\n")
        if len(lines) < 2:
            return text, False

        applied = False
        changed = True

        # Iteratively join lines ending with hyphens
        while changed:
            changed = False
            new_lines: List[str] = []
            i = 0
            while i < len(lines):
                line = lines[i]
                # Match trailing hyphen preceded by a word (or hyphenated compound like state-of-the-)
                match = re.search(r"(\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)-$", line.rstrip())

                if match and i + 1 < len(lines):
                    next_line = lines[i + 1].lstrip()
                    next_match = re.match(r"^([A-Za-z0-9]+)\b(.*)$", next_line)

                    if next_match:
                        prefix_compound = match.group(1)
                        # The last token before the hyphen (e.g., 'real' in 'real-', or 'the' in 'state-of-the-')
                        last_token = prefix_compound.split("-")[-1]
                        suffix_word = next_match.group(1)
                        rest_of_next_line = next_match.group(2)

                        full_compound = f"{prefix_compound.lower()}-{suffix_word.lower()}"
                        is_legit = (
                            last_token.lower() in _LEGITIMATE_HYPHEN_PREFIXES
                            or prefix_compound.lower() in _LEGITIMATE_HYPHEN_PREFIXES
                            or full_compound in _LEGITIMATE_COMPOUNDS
                            or (last_token.isupper() and suffix_word.isupper() and len(last_token) <= 3)
                        )

                        line_prefix = line.rstrip()[:-1]  # remove trailing hyphen

                        if is_legit:
                            # Keep hyphen, join to next line
                            repaired_line = f"{line_prefix}-{suffix_word}{rest_of_next_line}"
                        else:
                            # Dehyphenate (merge words directly)
                            repaired_line = f"{line_prefix}{suffix_word}{rest_of_next_line}"

                        new_lines.append(repaired_line)
                        applied = True
                        changed = True
                        # Advance past the merged next_line
                        i += 2
                        continue

                new_lines.append(line)
                i += 1

            lines = new_lines

        return "\n".join(lines), applied

    def filter_standalone_page_numbers(self, text: str) -> Tuple[str, bool]:
        """Remove standalone page number lines while keeping meaningful text."""
        if not text:
            return "", False

        lines = text.split("\n")
        filtered: List[str] = []
        applied = False

        for line in lines:
            stripped = line.strip()
            if _PAGE_NUMBER_PATTERN.match(stripped) and len(stripped) < 20:
                applied = True
                continue
            filtered.append(line)

        return "\n".join(filtered), applied

    def normalize_whitespace(self, text: str) -> Tuple[str, bool]:
        """Normalize line breaks, spaces, tabs, and paragraph spacing."""
        if not text:
            return "", False

        original = text

        # 1. Standardize line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Split into paragraph blocks (separated by 2+ newlines)
        raw_paragraphs = re.split(r"\n\s*\n+", text)
        cleaned_paragraphs: List[str] = []

        for p in raw_paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue

            lines = p_clean.split("\n")
            cleaned_lines: List[str] = []

            for line in lines:
                # Replace multiple spaces / tabs with single space
                line_normalized = re.sub(r"[ \t]+", " ", line).strip()
                if line_normalized:
                    cleaned_lines.append(line_normalized)

            if not cleaned_lines:
                continue

            # Join lines within a single paragraph with space unless they look like lists/headings
            joined_paragraph_lines: List[str] = []
            curr_acc: List[str] = []

            for line in cleaned_lines:
                if _LIST_BULLET_PATTERN.match(line):
                    if curr_acc:
                        joined_paragraph_lines.append(" ".join(curr_acc))
                        curr_acc = []
                    joined_paragraph_lines.append(line)
                else:
                    curr_acc.append(line)

            if curr_acc:
                joined_paragraph_lines.append(" ".join(curr_acc))

            cleaned_paragraphs.append("\n".join(joined_paragraph_lines))

        result = "\n\n".join(cleaned_paragraphs).strip()
        applied = result != original
        return result, applied
