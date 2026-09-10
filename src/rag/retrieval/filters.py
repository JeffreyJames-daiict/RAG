"""Typed metadata filters and SQL clause builders for Vector Retrieval."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import ColumnElement, and_, or_

from rag.core.db_models import Chunk, Document
from rag.retrieval.exceptions import FilterValidationError


class RetrievalFilters(BaseModel):
    """Strongly-typed metadata and positional filters for vector retrieval queries."""

    document_ids: Optional[List[str]] = Field(default=None, description="Restrict retrieval to specific document IDs")
    document_versions: Optional[List[int]] = Field(default=None, description="Specific document versions")
    source_types: Optional[List[str]] = Field(default=None, description="Source connector types (e.g. LOCAL_FILE, S3)")
    filenames: Optional[List[str]] = Field(default=None, description="Filter by source filenames")
    languages: Optional[List[str]] = Field(default=None, description="Filter by language codes (e.g. eng)")
    chunk_types: Optional[List[str]] = Field(default=None, description="Chunk types: text, table, figure")
    document_statuses: Optional[List[str]] = Field(default=None, description="Allowed document statuses")
    page_numbers: Optional[List[int]] = Field(default=None, description="Restrict to specific page numbers")
    created_after: Optional[datetime] = Field(default=None, description="Document creation time floor")
    created_before: Optional[datetime] = Field(default=None, description="Document creation time ceiling")
    table_id: Optional[str] = Field(default=None, description="Restrict to specific table ID")
    metadata_json_match: Optional[Dict[str, Any]] = Field(default=None, description="Exact key-value match on chunk metadata JSON")

    @field_validator("document_ids", "filenames", "chunk_types", "source_types")
    @classmethod
    def _validate_lists(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            if len(v) > 50:
                raise FilterValidationError("Filter list exceeds maximum allowable size (50).")
            for item in v:
                if not isinstance(item, str) or not item.strip():
                    raise FilterValidationError("Filter list elements must be non-empty strings.")
        return v

    def build_sql_clauses(self) -> List[ColumnElement[bool]]:
        """Construct safe SQLAlchemy binary expression clauses from filter criteria."""
        clauses: List[ColumnElement[bool]] = []

        if self.document_ids:
            clauses.append(Chunk.document_id.in_(self.document_ids))

        if self.document_versions:
            clauses.append(Chunk.version.in_(self.document_versions))

        if self.filenames:
            clauses.append(Document.filename.in_(self.filenames))

        if self.source_types:
            clauses.append(Document.source.in_(self.source_types))

        if self.document_statuses:
            clauses.append(Document.status.in_(self.document_statuses))

        if self.page_numbers:
            clauses.append(Chunk.page.in_(self.page_numbers))

        if self.created_after:
            clauses.append(Document.created_at >= self.created_after)

        if self.created_before:
            clauses.append(Document.created_at <= self.created_before)

        # Note: chunk_types, table_id, and metadata_json_match are handled in candidate evaluation
        # across JSON chunk_metadata field in a DB-independent manner
        return clauses

    def matches_chunk_metadata(self, meta: Dict[str, Any]) -> bool:
        """Evaluate in-memory JSON metadata predicates against chunk metadata."""
        if not meta:
            meta = {}

        if self.chunk_types:
            ch_type = meta.get("chunk_type", "text")
            if ch_type not in self.chunk_types:
                return False

        if self.table_id:
            if meta.get("table_id") != self.table_id:
                return False

        if self.languages:
            lang = meta.get("language")
            if lang not in self.languages:
                return False

        if self.metadata_json_match:
            for k, expected_v in self.metadata_json_match.items():
                if meta.get(k) != expected_v:
                    return False

        return True
