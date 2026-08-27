"""Pydantic schemas and DTOs for the RAG pipeline."""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict


class SourceType(str, Enum):
    """Supported source connector types."""
    LOCAL_FILE = "LOCAL_FILE"
    S3 = "S3"
    WEB = "WEB"
    DATABASE = "DATABASE"


class DocumentStatus(str, Enum):
    """Lifecycle status of a document in the RAG pipeline."""
    DISCOVERED = "DISCOVERED"
    FETCHED = "FETCHED"
    PARSED = "PARSED"
    CHUNKED = "CHUNKED"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


class DocumentDescriptor(BaseModel):
    """Lightweight document manifest returned by connector discovery."""
    source: SourceType
    source_uri: str
    filename: str
    mime_type: str
    tenant_id: str = "default"
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)
    last_modified_at: Optional[datetime] = None
    expected_hash: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class RawDocumentPayload(BaseModel):
    """Container for raw content bytes or stream and its descriptor."""
    descriptor: DocumentDescriptor
    content: bytes
    content_hash: str
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ----------------------------------------------------------------------
# 47. Production Data Model Schemas
# ----------------------------------------------------------------------

class DocumentSchema(BaseModel):
    """Schema representing the `documents` table record."""
    document_id: UUID = Field(default_factory=uuid4)
    tenant_id: str = "default"
    source: SourceType
    source_uri: str
    filename: str
    mime_type: str
    content_hash: str
    version: int = 1
    status: DocumentStatus = DocumentStatus.DISCOVERED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)


class DocumentVersionSchema(BaseModel):
    """Schema representing the `document_versions` table record."""
    document_id: UUID
    version: int
    content_hash: str
    raw_location: Optional[str] = None
    parser_version: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)


class ChunkSchema(BaseModel):
    """Schema representing the `chunks` table record."""
    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    version: int
    chunk_index: int
    text: str
    section: Optional[str] = None
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunking_version: str = "v1.0"

    model_config = ConfigDict(from_attributes=True)


class EmbeddingSchema(BaseModel):
    """Schema representing the `embeddings` table record."""
    chunk_id: UUID
    embedding_model: str
    embedding_version: str = "v1.0"
    vector: List[float]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)


class SyncSummary(BaseModel):
    """Summary metrics of a connector ingestion run."""
    source_type: SourceType
    discovered_count: int = 0
    fetched_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    errors: List[str] = Field(default_factory=list)
