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
    DELETED = "DELETED"
    FAILED = "FAILED"


class DocumentChangeType(str, Enum):
    """Classification of document change status during discovery."""
    NEW = "NEW"
    CHANGED = "CHANGED"
    UNCHANGED = "UNCHANGED"
    DELETED = "DELETED"


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


class DiscoveredItem(BaseModel):
    """Represents a discovered document and its evaluated change state."""
    descriptor: DocumentDescriptor
    change_type: DocumentChangeType
    reason: str
    existing_document_id: Optional[str] = None
    existing_version: Optional[int] = None
    existing_hash: Optional[str] = None


class DeletionCandidate(BaseModel):
    """Represents a tracked document missing from discovery to be marked deleted."""
    document_id: str
    source_uri: str
    filename: str
    tenant_id: str
    source: SourceType
    last_known_version: int


class DiscoveryReport(BaseModel):
    """Comprehensive discovery differential classifying all documents."""
    tenant_id: str
    source_type: SourceType
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_discovered: int = 0
    new_items: List[DiscoveredItem] = Field(default_factory=list)
    changed_items: List[DiscoveredItem] = Field(default_factory=list)
    unchanged_items: List[DiscoveredItem] = Field(default_factory=list)
    deleted_items: List[DeletionCandidate] = Field(default_factory=list)

    @property
    def total_existing(self) -> int:
        """Total documents that already existed in DB prior to sync."""
        return len(self.unchanged_items) + len(self.changed_items) + len(self.deleted_items)

    @property
    def summary_counts(self) -> Dict[str, int]:
        return {
            "total_discovered": self.total_discovered,
            "new": len(self.new_items),
            "changed": len(self.changed_items),
            "unchanged": len(self.unchanged_items),
            "deleted": len(self.deleted_items),
        }


# ----------------------------------------------------------------------
# Production Data Model Schemas
# ----------------------------------------------------------------------

class PipelineStage(str, Enum):
    """Stages of the RAG pipeline."""
    DISCOVERY = "DISCOVERY"
    FETCH = "FETCH"
    PARSE = "PARSE"
    ASSET = "ASSET"
    CHUNK = "CHUNK"
    EMBED = "EMBED"
    PIPELINE = "PIPELINE"


class AssetType(str, Enum):
    """Classification of visual document assets."""
    FIGURE = "FIGURE"
    DIAGRAM = "DIAGRAM"
    ARCHITECTURE_DIAGRAM = "ARCHITECTURE_DIAGRAM"
    FLOWCHART = "FLOWCHART"
    TECHNICAL_DRAWING = "TECHNICAL_DRAWING"
    CHART = "CHART"
    PHOTO = "PHOTO"
    ILLUSTRATION = "ILLUSTRATION"
    TABLE_IMAGE = "TABLE_IMAGE"
    EQUATION_IMAGE = "EQUATION_IMAGE"
    DECORATIVE = "DECORATIVE"
    UNKNOWN = "UNKNOWN"


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
    deleted_at: Optional[datetime] = None
    error: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class PipelineErrorSchema(BaseModel):
    """Schema representing the `errors` table record."""
    error_id: UUID = Field(default_factory=uuid4)
    document_id: Optional[UUID] = None
    tenant_id: str = "default"
    source_uri: Optional[str] = None
    stage: PipelineStage = PipelineStage.PIPELINE
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

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


class DocumentAssetSchema(BaseModel):
    """Schema representing the `document_assets` table record."""
    asset_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    version: int = 1
    page_number: int = 1
    asset_type: AssetType = AssetType.FIGURE
    asset_subtype: Optional[str] = None
    storage_uri: str
    content_hash: str
    bbox: Optional[Dict[str, Any]] = None
    width: int = 0
    height: int = 0
    mime_type: str = "image/png"
    caption: Optional[str] = None
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    semantic_description: Optional[str] = None
    entities: List[str] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    processing_status: str = "READY"
    processor_version: str = "v1.0"
    asset_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

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
    dimension: Optional[int] = None
    provider: Optional[str] = None
    vector: List[float]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)


class SyncSummary(BaseModel):
    """Summary metrics of a connector ingestion run."""
    source_type: SourceType
    discovered_count: int = 0
    new_count: int = 0
    changed_count: int = 0
    unchanged_count: int = 0
    deleted_count: int = 0
    fetched_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    errors: List[str] = Field(default_factory=list)
    discovery_report: Optional[DiscoveryReport] = None

