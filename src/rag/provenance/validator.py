"""Citation & Provenance Validator — validates citation integrity, tenant isolation, document versioning, claim grounding, and calibrated confidence."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from rag.provenance.models import (
    Citation,
    CitationValidationResult,
    ClaimValidation,
    GroundedAnswerResponse,
    RetrievedEvidenceItem,
)
from rag.provenance.registry import CitationRegistry

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "its", "itself", "let's", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
}

NON_CLAIM_PREFIXES = (
    "based on", "according to", "in summary", "to summarize", "in conclusion",
    "here is", "the provided", "the following", "as stated", "note:", "disclaimer:",
)


class CitationProvenanceValidator:
    """Production validator for answer claims, citation grounding, and security constraints."""

    def __init__(
        self,
        min_overlap_threshold: float = 0.12,
        strict_tenant_check: bool = True,
        strict_version_check: bool = True,
    ):
        self.min_overlap_threshold = min_overlap_threshold
        self.strict_tenant_check = strict_tenant_check
        self.strict_version_check = strict_version_check
        self._cit_pattern = re.compile(r"\[C\d+\]", re.IGNORECASE)

    def extract_citation_markers(self, text: str) -> List[str]:
        """Extract all citation markers like [C1], [C2] in order of appearance."""
        return [m.upper() for m in self._cit_pattern.findall(text)]

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving bracketed citation tags."""
        if not text or not text.strip():
            return []

        # Split on sentence terminals followed by whitespace, avoiding decimals/abbreviations
        raw_chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", text.strip())
        sentences: List[str] = []
        for c in raw_chunks:
            s = c.strip()
            # Also break on newlines if multiple distinct lines exist
            for sub_s in s.split("\n"):
                clean = sub_s.strip()
                if clean and len(clean) > 3:
                    sentences.append(clean)
        return sentences

    def _tokenize(self, text: str) -> Set[str]:
        """Normalize and tokenize text into meaningful alphanumeric content words."""
        # Strip citation markers first
        clean = self._cit_pattern.sub(" ", text)
        words = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", clean.lower())
        return {w for w in words if w not in STOPWORDS}

    def _is_substantive_claim(self, sentence: str) -> bool:
        """Heuristic determining if a sentence is an informational claim requiring evidence."""
        tokens = self._tokenize(sentence)
        if len(tokens) < 3:
            return False
        lower = sentence.lower()
        if any(lower.startswith(p) for p in ("hello", "hi ", "thank you", "sure,", "certainly")):
            return False
        return True

    def calculate_evidence_overlap(self, sentence: str, evidence_text: str) -> float:
        """Calculate token overlap ratio between claim sentence and evidence passage."""
        claim_tokens = self._tokenize(sentence)
        if not claim_tokens:
            return 0.0

        evidence_tokens = self._tokenize(evidence_text)
        if not evidence_tokens:
            return 0.0

        intersection = claim_tokens.intersection(evidence_tokens)
        return len(intersection) / len(claim_tokens)

    def validate(
        self,
        answer: str,
        registry: CitationRegistry,
        expected_tenant_id: Optional[str] = None,
        expected_document_versions: Optional[Dict[str, str]] = None,
    ) -> Tuple[CitationValidationResult, List[ClaimValidation], float]:
        """Validate answer against citation registry.
        
        Returns:
            (validation_result, claim_validations, calibrated_confidence)
        """
        target_tenant = expected_tenant_id or registry.tenant_id
        target_versions = expected_document_versions or registry.expected_document_versions

        valid_citations: List[str] = []
        invalid_citations: List[str] = []
        missing_citations: List[str] = []
        duplicate_citations: List[str] = []
        cross_tenant_citations: List[str] = []
        version_mismatch_citations: List[str] = []
        mismatched_citations: List[str] = []
        unsupported_claims: List[str] = []
        warnings: List[str] = []

        all_used_markers: Set[str] = set()
        sentences = self.split_into_sentences(answer)
        claim_validations: List[ClaimValidation] = []

        supported_claims_count = 0
        total_claims_evaluated = 0

        for sentence in sentences:
            markers = self.extract_citation_markers(sentence)
            all_used_markers.update(markers)

            # Check for intra-sentence duplicate citations (e.g. "... [C1] and [C1].")
            seen_in_sent: Set[str] = set()
            for m in markers:
                if m in seen_in_sent:
                    duplicate_citations.append(m)
                    warnings.append(f"Duplicate citation '{m}' repeated in sentence: '{sentence[:60]}...'")
                seen_in_sent.add(m)

            is_substantive = self._is_substantive_claim(sentence)

            if not markers:
                if is_substantive:
                    missing_citations.append(sentence)
                    unsupported_claims.append(sentence)
                    claim_validations.append(
                        ClaimValidation(
                            sentence=sentence,
                            cited_markers=[],
                            is_supported=False,
                            evidence_overlap_score=0.0,
                            mismatched_markers=[],
                            reason="Missing citation: Claim has no supporting evidence marker.",
                        )
                    )
                    total_claims_evaluated += 1
                continue

            total_claims_evaluated += 1
            sent_mismatched: List[str] = []
            best_overlap = 0.0
            sentence_supported = False

            for m in markers:
                cit = registry.get(m)
                if not cit:
                    invalid_citations.append(m)
                    warnings.append(f"Invalid citation: Marker '{m}' does not exist in retrieval registry.")
                    continue

                valid_citations.append(m)

                # Tenant Isolation Validation
                if self.strict_tenant_check and cit.tenant_id != target_tenant:
                    cross_tenant_citations.append(m)
                    warnings.append(
                        f"Cross-tenant violation: Citation '{m}' belongs to tenant '{cit.tenant_id}', "
                        f"violating expected tenant '{target_tenant}'."
                    )

                # Document Version Validation
                doc_expected_ver = target_versions.get(cit.document_id)
                if self.strict_version_check and doc_expected_ver and cit.document_version_id != doc_expected_ver:
                    version_mismatch_citations.append(m)
                    warnings.append(
                        f"Version mismatch: Citation '{m}' references version '{cit.document_version_id}', "
                        f"expected '{doc_expected_ver}' for doc '{cit.document_id}'."
                    )

                # Grounding & Mismatch Validation
                # Check overlap against both quoted_text and raw evidence text if available
                raw_ev = registry.get_evidence(m)
                full_ev_text = (raw_ev.text if raw_ev else "") or cit.quoted_text
                overlap = self.calculate_evidence_overlap(sentence, full_ev_text)
                if overlap > best_overlap:
                    best_overlap = overlap

                if overlap < self.min_overlap_threshold:
                    sent_mismatched.append(m)
                    mismatched_citations.append(m)
                    warnings.append(
                        f"Citation mismatch: Evidence for '{m}' has low lexical overlap ({overlap:.2f}) "
                        f"with claim '{sentence[:60]}...'."
                    )
                else:
                    sentence_supported = True

            if sentence_supported and not sent_mismatched:
                supported_claims_count += 1
                claim_validations.append(
                    ClaimValidation(
                        sentence=sentence,
                        cited_markers=markers,
                        is_supported=True,
                        evidence_overlap_score=round(best_overlap, 3),
                        mismatched_markers=[],
                    )
                )
            else:
                unsupported_claims.append(sentence)
                claim_validations.append(
                    ClaimValidation(
                        sentence=sentence,
                        cited_markers=markers,
                        is_supported=False,
                        evidence_overlap_score=round(best_overlap, 3),
                        mismatched_markers=sent_mismatched,
                        reason="Citation mismatch: Evidence failed to substantiate the claim.",
                    )
                )

        # Citations provided in evidence but not referenced
        unused_citations = [
            c.citation_id for c in registry.all_citations()
            if c.citation_id not in all_used_markers
        ]

        # Determine overall validity
        has_violations = (
            bool(invalid_citations)
            or bool(cross_tenant_citations)
            or bool(version_mismatch_citations)
        )
        is_valid = (not has_violations) and (len(unsupported_claims) == 0)

        # Calculate calibrated confidence score
        confidence = self._compute_confidence(
            total_claims=total_claims_evaluated,
            supported_claims=supported_claims_count,
            registry=registry,
            used_markers=all_used_markers,
            invalid_citations=invalid_citations,
            cross_tenant_citations=cross_tenant_citations,
            version_mismatches=version_mismatch_citations,
            mismatched_citations=mismatched_citations,
            missing_citations=missing_citations,
        )

        val_result = CitationValidationResult(
            is_valid=is_valid,
            valid_citations=sorted(list(set(valid_citations))),
            invalid_citations=sorted(list(set(invalid_citations))),
            missing_citations=missing_citations,
            unused_citations=sorted(unused_citations),
            duplicate_citations=sorted(list(set(duplicate_citations))),
            cross_tenant_citations=sorted(list(set(cross_tenant_citations))),
            version_mismatch_citations=sorted(list(set(version_mismatch_citations))),
            mismatched_citations=sorted(list(set(mismatched_citations))),
            warnings=warnings,
        )

        return val_result, claim_validations, confidence

    def _compute_confidence(
        self,
        total_claims: int,
        supported_claims: int,
        registry: CitationRegistry,
        used_markers: Set[str],
        invalid_citations: List[str],
        cross_tenant_citations: List[str],
        version_mismatches: List[str],
        mismatched_citations: List[str],
        missing_citations: List[str],
    ) -> float:
        """Compute calibrated confidence in [0.0, 1.0]."""
        # Critical security penalty: cross-tenant leak drops confidence immediately
        if cross_tenant_citations:
            return 0.0

        if total_claims == 0:
            # If no substantive claims, but citations exist without violations
            return 0.5 if not invalid_citations else 0.0

        grounded_ratio = supported_claims / total_claims

        # Average retrieval score of cited evidence
        retrieval_scores: List[float] = []
        for m in used_markers:
            c = registry.get(m)
            if c and c.retrieval_score is not None:
                retrieval_scores.append(max(0.0, min(1.0, float(c.retrieval_score))))

        avg_retrieval = (sum(retrieval_scores) / len(retrieval_scores)) if retrieval_scores else 0.7

        # Composite score
        score = (grounded_ratio * 0.7) + (avg_retrieval * 0.3)

        # Deductions
        if invalid_citations:
            score -= (0.25 * len(set(invalid_citations)))
        if version_mismatches:
            score -= (0.20 * len(set(version_mismatches)))
        if mismatched_citations:
            score -= (0.15 * len(set(mismatched_citations)))
        if missing_citations:
            score -= (0.10 * len(missing_citations))

        return max(0.0, min(1.0, round(score, 3)))
