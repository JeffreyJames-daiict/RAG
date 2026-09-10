"""Provenance Service — coordinates evidence registration, answer validation, and grounded response synthesis."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from rag.provenance.models import (
    Citation,
    CitationValidationResult,
    ClaimValidation,
    GroundedAnswerResponse,
    ProvenanceType,
    RetrievedEvidenceItem,
)
from rag.provenance.registry import CitationRegistry
from rag.provenance.validator import CitationProvenanceValidator
from rag.retrieval.models import RetrievalResponse, RetrievedChunkResult

logger = logging.getLogger(__name__)


class ProvenanceService:
    """High-level orchestration service for the citation and provenance subsystem."""

    def __init__(
        self,
        validator: Optional[CitationProvenanceValidator] = None,
        citation_prefix: str = "C",
        min_overlap_threshold: float = 0.12,
        strict_tenant_check: bool = True,
        strict_version_check: bool = True,
    ):
        self.validator = validator or CitationProvenanceValidator(
            min_overlap_threshold=min_overlap_threshold,
            strict_tenant_check=strict_tenant_check,
            strict_version_check=strict_version_check,
        )
        self.citation_prefix = citation_prefix

    def build_registry(
        self,
        retrieval_source: Any,
        tenant_id: str = "default",
        expected_document_versions: Optional[Dict[str, str]] = None,
    ) -> CitationRegistry:
        """Construct a sealed CitationRegistry solely from retrieved evidence."""
        from rag.context.models import ContextPackage

        registry = CitationRegistry(
            tenant_id=tenant_id,
            citation_prefix=self.citation_prefix,
            expected_document_versions=expected_document_versions,
        )

        if isinstance(retrieval_source, ContextPackage):
            # Register context chunks
            for ch in retrieval_source.selected_chunks:
                registry.register_chunk(ch)

            # Register visual assets
            for doc in retrieval_source.grouped_documents:
                for asset in doc.visual_assets:
                    registry.register_asset(
                        asset_id=asset.asset_id,
                        document_id=doc.document_id,
                        description=asset.semantic_description or asset.caption or "Diagram asset",
                        asset_type=asset.asset_type,
                        caption=asset.caption,
                        ocr_text=asset.ocr_text,
                        page_number=asset.page,
                        tenant_id=tenant_id,
                    )
                for tbl in doc.tables:
                    registry.register_table(
                        table_id=tbl.table_id,
                        document_id=doc.document_id,
                        formatted_table=tbl.formatted_table,
                        title=tbl.caption or "Table",
                        page_number=tbl.page,
                        tenant_id=tenant_id,
                    )

        elif isinstance(retrieval_source, RetrievalResponse):
            for res in retrieval_source.results:
                registry.register_chunk(res)

        elif isinstance(retrieval_source, (list, tuple)):
            for item in retrieval_source:
                registry.register_chunk(item)

        # Freeze registry so no unverified evidence can be injected during generation
        registry.freeze()
        return registry

    def validate_answer(
        self,
        answer: str,
        registry: CitationRegistry,
        expected_tenant_id: Optional[str] = None,
        expected_document_versions: Optional[Dict[str, str]] = None,
    ) -> GroundedAnswerResponse:
        """Validate answer against citation registry and return full GroundedAnswerResponse contract."""
        val_result, claim_vals, confidence = self.validator.validate(
            answer=answer,
            registry=registry,
            expected_tenant_id=expected_tenant_id,
            expected_document_versions=expected_document_versions,
        )

        unsupported = [cv.sentence for cv in claim_vals if not cv.is_supported]

        all_cits = registry.all_citations()
        all_ev = registry.all_evidence()

        return GroundedAnswerResponse(
            answer=answer,
            citations=all_cits,
            retrieved_evidence=all_ev,
            citation_validation=val_result,
            unsupported_claims=unsupported,
            confidence=confidence,
            warnings=val_result.warnings,
        )

    def sanitize_answer(self, answer: str, registry: CitationRegistry) -> str:
        """Remove invalid or hallucinated citation markers from answer while preserving valid ones."""
        markers = self.validator.extract_citation_markers(answer)
        clean = answer
        for m in markers:
            if not registry.is_registered(m):
                # Remove hallucinated citation marker e.g. " [C99]" or "[C99]"
                clean = clean.replace(f" {m}", "").replace(m, "")
        return clean.strip()
