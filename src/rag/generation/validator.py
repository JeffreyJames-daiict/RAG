"""Answer response validation and citation integrity verification."""

from __future__ import annotations

import re
from typing import List, Optional, Set

from rag.context.models import ContextPackage
from rag.generation.models import ValidationResult

REFUSAL_PATTERNS = [
    re.compile(r"(?i)\bi\s+(?:am\s+sorry|apologize|cannot\s+fulfill|cannot\s+assist|can\'t\s+help\s+with\s+that)\b"),
    re.compile(r"(?i)\bas\s+an\s+ai(?:\s+language\s+model)?,\s+i\s+cannot\b"),
]


class ResponseValidator:
    """Validates generated LLM responses, detects refusals, and cross-checks citations against evidence."""

    def __init__(self, validate_citations: bool = True):
        self.validate_citations = validate_citations

    def extract_citations(self, text: str) -> List[str]:
        """Extract all citation markers like [C1], [C2] or [1], [2] from text."""
        markers: Set[str] = set()
        # [C1], [C2], [c1]
        c_matches = re.findall(r"\[C\d+\]", text, re.IGNORECASE)
        for m in c_matches:
            markers.add(m.upper())

        # Also capture bracketed numbers like [1], [2]
        num_matches = re.findall(r"\[(\d+)\]", text)
        for m in num_matches:
            markers.add(f"[{m}]")

        return sorted(markers)

    def validate(
        self,
        raw_text: Optional[str],
        context_package: Optional[ContextPackage] = None,
        max_output_length: int = 50000,
    ) -> ValidationResult:
        """Execute full validation checks on raw LLM completion text."""
        warnings: List[str] = []

        if not raw_text or not raw_text.strip():
            return ValidationResult(
                is_valid=False,
                clean_text="",
                warnings=["Response is completely empty"],
            )

        text = raw_text.strip()

        # 1. Check for provider refusal
        for pattern in REFUSAL_PATTERNS:
            if pattern.search(text[:300]):
                warnings.append("Provider refusal or safety declination detected in response")
                return ValidationResult(
                    is_valid=False,
                    clean_text=text,
                    warnings=warnings,
                )

        # 2. Check excessive length
        if len(text) > max_output_length:
            warnings.append(f"Response exceeded expected maximum character length ({len(text)} > {max_output_length})")
            text = text[:max_output_length]

        # 3. Validate citations against ContextPackage
        cited_markers = self.extract_citations(text)
        valid_citations: List[str] = []
        hallucinated_citations: List[str] = []

        if self.validate_citations and context_package:
            registered_citation_ids = {c.citation_id.upper() for c in context_package.citations}
            # Also support rank index as valid numeric reference
            registered_ranks = {f"[{c.page}]" for c in context_package.citations if c.page}
            registered_indices = {f"[{idx}]" for idx in range(1, len(context_package.selected_chunks) + 1)}

            for marker in cited_markers:
                if marker in registered_citation_ids or marker in registered_ranks or marker in registered_indices:
                    valid_citations.append(marker)
                else:
                    hallucinated_citations.append(marker)
                    warnings.append(f"Hallucinated citation detected: {marker} is not present in ContextPackage")
        else:
            valid_citations = cited_markers

        return ValidationResult(
            is_valid=True,
            clean_text=text,
            valid_citations=valid_citations,
            hallucinated_citations=hallucinated_citations,
            warnings=warnings,
        )
