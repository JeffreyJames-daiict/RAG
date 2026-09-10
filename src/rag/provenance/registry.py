"""Citation Registry — registers and tracks verified retrieved evidence passages with immutable [C#] keys."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from rag.provenance.models import Citation, ProvenanceType, RetrievedEvidenceItem
from rag.retrieval.models import RetrievedChunkResult

logger = logging.getLogger(__name__)


class CitationRegistryError(Exception):
    """Raised when an illegal operation is attempted on CitationRegistry."""
    pass


class CitationRegistry:
    """Deterministic, tenant-safe registry mapping in-text citation markers ([C1], [C2]) to retrieved evidence."""

    def __init__(
        self,
        tenant_id: str = "default",
        citation_prefix: str = "C",
        expected_document_versions: Optional[Dict[str, str]] = None,
    ):
        self.tenant_id = tenant_id
        self.citation_prefix = citation_prefix
        # Mapping from doc_id -> expected version_id (e.g. {"doc-1": "v2"})
        self.expected_document_versions: Dict[str, str] = expected_document_versions or {}

        self._citations: Dict[str, Citation] = {}
        self._evidence_items: Dict[str, RetrievedEvidenceItem] = {}
        self._chunk_to_citation_id: Dict[str, str] = {}
        self._asset_to_citation_id: Dict[str, str] = {}
        self._counter: int = 1
        self._is_frozen: bool = False

    @property
    def is_frozen(self) -> bool:
        """True if the registry has been frozen to prevent mutation during or after answer generation."""
        return self._is_frozen

    def freeze(self) -> None:
        """Lock the registry from admitting further evidence."""
        self._is_frozen = True

    def _next_citation_id(self) -> str:
        cid = f"[{self.citation_prefix}{self._counter}]"
        self._counter += 1
        return cid

    def register_evidence(
        self,
        evidence: RetrievedEvidenceItem,
        citation_id: Optional[str] = None,
    ) -> Citation:
        """Register a verified retrieved evidence item.
        
        Only retrieved evidence may be registered.
        """
        if self._is_frozen:
            raise CitationRegistryError("Cannot register evidence: CitationRegistry is frozen.")

        # Check if already registered by chunk_id or asset_id
        if evidence.item_id in self._chunk_to_citation_id:
            existing_id = self._chunk_to_citation_id[evidence.item_id]
            return self._citations[existing_id]

        if evidence.asset_id and evidence.asset_id in self._asset_to_citation_id:
            existing_id = self._asset_to_citation_id[evidence.asset_id]
            return self._citations[existing_id]

        cid = citation_id or self._next_citation_id()

        # Build clean quoted text snippet
        text_snippet = evidence.text.strip()
        if len(text_snippet) > 160:
            text_snippet = text_snippet[:157] + "..."

        citation = Citation(
            citation_id=cid,
            document_id=evidence.document_id,
            document_version_id=evidence.document_version_id,
            chunk_id=evidence.item_id,
            asset_id=evidence.asset_id,
            source_uri=evidence.source_uri,
            document_title=evidence.title,
            page_number=evidence.page_number,
            section_path=evidence.section_path,
            quoted_text=text_snippet,
            retrieval_score=evidence.retrieval_score,
            retrieval_method=evidence.retrieval_method,
            provenance_type=evidence.provenance_type,
            tenant_id=evidence.tenant_id,
            table_id=evidence.table_id,
        )

        self._citations[cid] = citation
        self._evidence_items[cid] = evidence

        if evidence.item_id:
            self._chunk_to_citation_id[evidence.item_id] = cid
        if evidence.asset_id:
            self._asset_to_citation_id[evidence.asset_id] = cid

        return citation

    def register_chunk(
        self,
        chunk: Any,
        provenance_type: Optional[ProvenanceType] = None,
    ) -> Citation:
        """Register a retrieved chunk candidate or context chunk."""
        from rag.context.models import ContextChunk
        if isinstance(chunk, ContextChunk):
            ptype = provenance_type or (
                ProvenanceType.PARENT_CHUNK
                if chunk.expansion_type == "parent"
                else ProvenanceType.NEIGHBOR_CHUNK
                if chunk.is_expanded
                else ProvenanceType.TABLE
                if chunk.chunk_type == "table"
                else ProvenanceType.FIGURE
                if chunk.chunk_type == "asset"
                else ProvenanceType.DIRECT_CHUNK
            )
            evidence = RetrievedEvidenceItem(
                item_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_version_id=chunk.metadata.get("document_version_id", "v1"),
                tenant_id=chunk.metadata.get("tenant_id", self.tenant_id),
                text=chunk.text,
                title=chunk.filename or chunk.metadata.get("title", ""),
                page_number=chunk.page,
                section_path=[chunk.section] if chunk.section else [],
                source_uri=chunk.metadata.get("source_uri"),
                asset_id=chunk.asset_id,
                table_id=chunk.table_id,
                retrieval_score=chunk.score,
                retrieval_method=chunk.metadata.get("retrieval_method", "retrieval"),
                provenance_type=ptype,
                metadata=chunk.metadata,
            )
            return self.register_evidence(evidence, citation_id=chunk.citation_id if chunk.citation_id else None)

        if isinstance(chunk, RetrievedChunkResult):
            ptype = provenance_type or (
                ProvenanceType.TABLE
                if chunk.table_id or chunk.chunk_type == "table"
                else ProvenanceType.FIGURE
                if chunk.asset_id or chunk.chunk_type == "asset"
                else ProvenanceType.DIRECT_CHUNK
            )
            evidence = RetrievedEvidenceItem(
                item_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_version_id=chunk.document_version_id or f"v{chunk.document_version}",
                tenant_id=chunk.tenant_id or self.tenant_id,
                text=chunk.text,
                title=chunk.title or chunk.filename,
                page_number=chunk.page_number or chunk.page,
                section_path=chunk.section_path or ([chunk.heading] if chunk.heading else []),
                source_uri=chunk.source_uri,
                asset_id=chunk.asset_id,
                table_id=chunk.table_id,
                retrieval_score=chunk.reranker_normalized_score or chunk.similarity_score or chunk.normalized_score,
                retrieval_method=chunk.retrieval_method,
                provenance_type=ptype,
                metadata=chunk.metadata,
            )
            return self.register_evidence(evidence)

        if isinstance(chunk, dict):
            ptype = provenance_type or ProvenanceType.DIRECT_CHUNK
            evidence = RetrievedEvidenceItem(
                item_id=str(chunk.get("chunk_id", f"chk_{self._counter}")),
                document_id=str(chunk.get("document_id", "doc_unknown")),
                document_version_id=str(chunk.get("document_version_id", "v1")),
                tenant_id=str(chunk.get("tenant_id", self.tenant_id)),
                text=str(chunk.get("text", "")),
                title=str(chunk.get("title", chunk.get("filename", ""))),
                page_number=chunk.get("page_number", chunk.get("page")),
                section_path=chunk.get("section_path", [chunk["section"]] if "section" in chunk else []),
                source_uri=chunk.get("source_uri"),
                asset_id=chunk.get("asset_id"),
                table_id=chunk.get("table_id"),
                retrieval_score=chunk.get("score"),
                retrieval_method=str(chunk.get("retrieval_method", "retrieval")),
                provenance_type=ptype,
                metadata=chunk.get("metadata", {}),
            )
            return self.register_evidence(evidence, citation_id=chunk.get("citation_id"))

        raise ValueError(f"Unsupported chunk type for citation registration: {type(chunk)}")

    def register_table(
        self,
        table_id: str,
        document_id: str,
        formatted_table: str,
        title: str = "",
        page_number: Optional[int] = None,
        section_path: Optional[List[str]] = None,
        document_version_id: str = "v1",
        tenant_id: Optional[str] = None,
        score: Optional[float] = None,
    ) -> Citation:
        """Register a table structure with table provenance."""
        evidence = RetrievedEvidenceItem(
            item_id=table_id,
            document_id=document_id,
            document_version_id=document_version_id,
            tenant_id=tenant_id or self.tenant_id,
            text=formatted_table,
            title=title,
            page_number=page_number,
            section_path=section_path or [],
            table_id=table_id,
            retrieval_score=score,
            retrieval_method="table_retrieval",
            provenance_type=ProvenanceType.TABLE,
        )
        return self.register_evidence(evidence)

    def register_asset(
        self,
        asset_id: str,
        document_id: str,
        description: str,
        asset_type: str = "figure",
        caption: Optional[str] = None,
        ocr_text: Optional[str] = None,
        page_number: Optional[int] = None,
        document_version_id: str = "v1",
        tenant_id: Optional[str] = None,
        score: Optional[float] = None,
    ) -> Citation:
        """Register a visual asset (diagram, figure, chart, OCR)."""
        ptype = (
            ProvenanceType.OCR_TEXT
            if asset_type.lower() == "ocr" or (ocr_text and not description)
            else ProvenanceType.CAPTION
            if caption and not description
            else ProvenanceType.FIGURE
        )
        full_text = f"Figure: {caption or ''}\nDescription: {description}" if caption else description
        if ocr_text:
            full_text += f"\nOCR: {ocr_text}"

        evidence = RetrievedEvidenceItem(
            item_id=asset_id,
            document_id=document_id,
            document_version_id=document_version_id,
            tenant_id=tenant_id or self.tenant_id,
            text=full_text,
            title=caption or f"Asset {asset_id}",
            page_number=page_number,
            asset_id=asset_id,
            retrieval_score=score,
            retrieval_method="asset_retrieval",
            provenance_type=ptype,
        )
        return self.register_evidence(evidence)

    def get(self, citation_id: str) -> Optional[Citation]:
        """Retrieve Citation by its marker, e.g. '[C1]'."""
        return self._citations.get(citation_id)

    def get_evidence(self, citation_id: str) -> Optional[RetrievedEvidenceItem]:
        """Retrieve raw evidence item for a citation marker."""
        return self._evidence_items.get(citation_id)

    def lookup_by_chunk_id(self, chunk_id: str) -> Optional[Citation]:
        """Find citation assigned to a given chunk ID."""
        cid = self._chunk_to_citation_id.get(chunk_id)
        return self._citations.get(cid) if cid else None

    def lookup_by_asset_id(self, asset_id: str) -> Optional[Citation]:
        """Find citation assigned to a given asset ID."""
        cid = self._asset_to_citation_id.get(asset_id)
        return self._citations.get(cid) if cid else None

    def is_registered(self, citation_id: str) -> bool:
        """True if the citation marker was produced from verified retrieved evidence in this registry."""
        return citation_id in self._citations

    def all_citations(self) -> List[Citation]:
        """Return list of all registered citations."""
        return list(self._citations.values())

    def all_evidence(self) -> List[RetrievedEvidenceItem]:
        """Return list of all registered raw evidence items."""
        return list(self._evidence_items.values())

    def validate_membership(
        self,
        citation_id: str,
        expected_tenant_id: str,
        expected_version_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Validate whether a citation belongs to this registry, tenant, and document version.
        
        Returns:
            (is_valid, failure_reason)
        """
        cit = self._citations.get(citation_id)
        if not cit:
            return False, f"Citation '{citation_id}' was not found in the retrieved evidence registry."

        if cit.tenant_id != expected_tenant_id:
            return False, (
                f"Cross-tenant violation: Citation '{citation_id}' belongs to tenant "
                f"'{cit.tenant_id}', but expected tenant is '{expected_tenant_id}'."
            )

        # Check explicit expected version or registered expected document versions
        target_version = expected_version_id or self.expected_document_versions.get(cit.document_id)
        if target_version and cit.document_version_id != target_version:
            return False, (
                f"Document version mismatch: Citation '{citation_id}' refers to version "
                f"'{cit.document_version_id}', but active document version is '{target_version}'."
            )

        return True, None
