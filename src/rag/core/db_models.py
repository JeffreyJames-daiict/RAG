"""SQLAlchemy 2.0 ORM database models for the Enterprise RAG pipeline."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from rag.core.models import DocumentStatus, SourceType


class Base(DeclarativeBase):
    """Base declarative class for all ORM models."""
    pass


class Document(Base):
    """Table: documents

    Stores the primary document entity, its current active version, source URI,
    content hash, and lifecycle status.
    """
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        doc="Unique identifier for the document",
    )
    tenant_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="default",
        index=True,
        doc="Tenant or organization identifier for multi-tenant isolation",
    )
    source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        doc="Source connector type (e.g., LOCAL_FILE, S3, WEB, DATABASE)",
    )
    source_uri: Mapped[str] = mapped_column(
        String(2048),
        nullable=False,
        index=True,
        doc="Original URI, path, or identifier in the source system",
    )
    filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        doc="Name of the file or object",
    )
    mime_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        doc="MIME / Content-Type of the document",
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="SHA-256 hash of the current document payload for change detection",
    )
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        doc="Current active version number of the document",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=DocumentStatus.DISCOVERED.value,
        index=True,
        doc="Current pipeline processing status",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        doc="Timestamp when document was first discovered",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        doc="Timestamp of last update or status change",
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp when document was soft deleted from source",
    )
    error: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Last failure reason or error message encountered during pipeline execution",
    )

    # Relationships
    versions: Mapped[List["DocumentVersion"]] = relationship(
        "DocumentVersion",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentVersion.version.desc()",
    )
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    errors: Mapped[List["PipelineError"]] = relationship(
        "PipelineError",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="PipelineError.created_at.desc()",
    )

    __table_args__ = (
        Index("ix_documents_tenant_source_uri", "tenant_id", "source_uri", unique=True),
    )


class DocumentVersion(Base):
    """Table: document_versions

    Maintains immutable historical snapshots of document revisions, their raw payload locations,
    and associated parser versions.
    """
    __tablename__ = "document_versions"

    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        primary_key=True,
        doc="Foreign key to documents table",
    )
    version: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        doc="Version sequence number (1, 2, 3...)",
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="SHA-256 hash of this specific version",
    )
    raw_location: Mapped[Optional[str]] = mapped_column(
        String(2048),
        nullable=True,
        doc="Location (e.g. S3 key or local storage path) where raw binary is preserved",
    )
    parser_version: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        doc="Version of parser/extractor used on this revision",
    )
    metadata_snapshot: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
        doc="Aggregated document version diagnostics, table stats, OCR metrics, and quality scores",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        doc="Timestamp when this version snapshot was created",
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="versions",
    )
    metadata_record: Mapped[Optional["DocumentVersionMetadata"]] = relationship(
        "DocumentVersionMetadata",
        back_populates="version_record",
        uselist=False,
        cascade="all, delete-orphan",
    )


class DocumentVersionMetadata(Base):
    """Table: document_version_metadata

    Relational normalization of document version metadata, diagnostics,
    quality metrics, and table/OCR execution statistics.
    """
    __tablename__ = "document_version_metadata"

    document_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        doc="Foreign key to document_versions table",
    )
    version: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        doc="Document version number",
    )
    primary_language: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="eng",
        doc="Primary detected language (ISO 639-2)",
    )
    languages_detected: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        doc="List of all detected languages across document pages",
    )
    num_pages: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Total page count",
    )
    total_chunks: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Total chunks generated for this version",
    )
    text_chunks: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Count of plain text chunks",
    )
    table_chunks: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Count of structured table chunks",
    )
    avg_quality_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Average composite quality score across all pages (0.0 to 1.0)",
    )
    avg_ocr_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Average OCR confidence score across OCR processed pages",
    )
    tables_detected: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Total raw table elements detected",
    )
    tables_stitched: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Number of multi-page tables stitched",
    )
    tables_processed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Total final table elements processed",
    )
    tables_warning: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Count of tables with validation warnings",
    )
    tables_failed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Count of tables failed during extraction",
    )
    pages_native: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Count of digital native pages",
    )
    pages_scanned: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Count of scanned raster pages",
    )
    pages_mixed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Count of mixed native/scanned pages",
    )
    ocr_pages_processed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Count of pages processed with OCR engine",
    )
    ocr_pages_failed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Count of pages where OCR failed",
    )
    pdf_title: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
        doc="Title from PDF metadata",
    )
    pdf_author: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        doc="Author from PDF metadata",
    )
    page_scores: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        doc="Per-page score breakdown list",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        doc="Timestamp of record creation",
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "version"],
            ["document_versions.document_id", "document_versions.version"],
            ondelete="CASCADE",
        ),
    )

    # Relationship
    version_record: Mapped["DocumentVersion"] = relationship(
        "DocumentVersion",
        back_populates="metadata_record",
    )


class Chunk(Base):
    """Table: chunks

    Stores atomic text chunks extracted from parsed documents with positional metadata
    (page, section, hierarchy).
    """
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        doc="Unique identifier for the chunk",
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Foreign key to parent document",
    )
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Document version number from which this chunk was derived",
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Sequential order of chunk within document version",
    )
    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Text content of the chunk",
    )
    section: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
        doc="Section heading or structural breadcrumb",
    )
    page: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Source document page number (if applicable)",
    )
    chunk_metadata: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
        doc="JSON metadata dictionary (custom attributes, tables, tokens)",
    )
    chunking_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="v1.0",
        doc="Identifier / version of chunking strategy",
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
    )
    embeddings: Mapped[List["Embedding"]] = relationship(
        "Embedding",
        back_populates="chunk",
        cascade="all, delete-orphan",
    )
    metadata_record: Mapped[Optional["ChunkMetadata"]] = relationship(
        "ChunkMetadata",
        back_populates="chunk",
        uselist=False,
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_chunks_doc_ver_idx", "document_id", "version", "chunk_index"),
    )


class ChunkMetadata(Base):
    """Table: chunk_metadata

    Relational normalization of chunk metadata, table boundaries,
    language, OCR provenance, and quality attributes.
    """
    __tablename__ = "chunk_metadata"

    chunk_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
        primary_key=True,
        doc="Foreign key to parent chunk",
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Foreign key to document",
    )
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Document version number",
    )
    chunk_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="text",
        index=True,
        doc="Chunk category: 'text', 'table', 'table_summary'",
    )
    page: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Primary page number",
    )
    page_start: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Start page for multi-page table chunks",
    )
    page_end: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="End page for multi-page table chunks",
    )
    token_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Token length of chunk text",
    )
    language: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="eng",
        doc="Detected language code",
    )
    language_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Language identification confidence (0.0 to 1.0)",
    )
    content_source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="native",
        doc="Source modality: 'native', 'scanned', 'mixed', 'ocr'",
    )
    quality_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Quality score of extracted content (0.0 to 1.0)",
    )
    ocr_applied: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="True if OCR was executed for this chunk",
    )
    ocr_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="OCR confidence score",
    )
    ocr_provider: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        doc="Name of OCR provider used",
    )
    preprocessing_profile: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
        doc="Image preprocessing profile used",
    )
    ocr_attempt: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        doc="OCR retry attempt number",
    )
    table_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        doc="Unique identifier of table element",
    )
    parent_table_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        doc="Root parent table identifier for expansion",
    )
    row_start: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Starting row index (1-indexed)",
    )
    row_end: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Ending row index (1-indexed)",
    )
    total_rows: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Total rows in parent table",
    )
    total_columns: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Total columns in parent table",
    )
    is_continuation: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="True if table continues across pages",
    )
    validation_status: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
        default="valid",
        doc="Table validation status: 'valid', 'warning', 'error'",
    )
    serialization_strategy: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
        default="hybrid",
        doc="Serialization mode used for table chunk",
    )
    extra_attributes: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        doc="Any additional custom metadata attributes",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        doc="Timestamp of creation",
    )

    # Relationship
    chunk: Mapped["Chunk"] = relationship(
        "Chunk",
        back_populates="metadata_record",
    )


class Embedding(Base):
    """Table: embeddings

    Stores dense vector representations for chunks, linked to specific embedding model
    and model version.
    """
    __tablename__ = "embeddings"

    chunk_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
        primary_key=True,
        doc="Foreign key to chunk",
    )
    embedding_model: Mapped[str] = mapped_column(
        String(128),
        primary_key=True,
        doc="Identifier of model used (e.g., text-embedding-3-small, bge-large-en)",
    )
    embedding_version: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default="v1.0",
        doc="Version of model or embedding configuration",
    )
    vector: Mapped[Any] = mapped_column(
        JSON,
        nullable=False,
        doc="Vector representation stored as serialized float array",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        doc="Timestamp when embedding was generated",
    )

    # Relationships
    chunk: Mapped["Chunk"] = relationship(
        "Chunk",
        back_populates="embeddings",
    )


class PipelineError(Base):
    """Table: errors

    Stores detailed error logs and failure reasons when pipeline operations fail.
    """
    __tablename__ = "errors"

    error_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        doc="Unique identifier for the error record",
    )
    document_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        doc="Foreign key to document if error is associated with a specific document",
    )
    tenant_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="default",
        index=True,
        doc="Tenant identifier",
    )
    source_uri: Mapped[Optional[str]] = mapped_column(
        String(2048),
        nullable=True,
        doc="Source URI of document or resource that caused error",
    )
    stage: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="PIPELINE",
        index=True,
        doc="Pipeline stage where error occurred (DISCOVERY, FETCH, PARSE, CHUNK, EMBED)",
    )
    error_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        doc="Exception class name or category",
    )
    error_message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Error message or reason",
    )
    stack_trace: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Detailed stack trace for debugging",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        doc="Timestamp when error occurred",
    )

    # Relationships
    document: Mapped[Optional["Document"]] = relationship(
        "Document",
        back_populates="errors",
    )


class RawDocumentPayload(Base):
    """Table: raw.document_payloads

    Stores raw immutable binary document payloads in a dedicated 'raw' database schema.
    """
    __tablename__ = "document_payloads"
    __table_args__ = (
        UniqueConstraint("document_id", "version", name="uq_raw_doc_version"),
        {"schema": "raw"},
    )

    payload_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        doc="Unique identifier for the raw payload record",
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        doc="Document ID string",
    )
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        doc="Document version number",
    )
    tenant_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="default",
        index=True,
        doc="Tenant key",
    )
    filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        default="",
        doc="Original filename",
    )
    mime_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="application/octet-stream",
        doc="MIME type",
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Document content SHA-256 hash",
    )
    sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Integrity SHA-256 hash of the binary payload",
    )
    size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        doc="Size of the raw binary payload in bytes",
    )
    payload: Mapped[bytes] = mapped_column(
        LargeBinary,
        nullable=False,
        doc="Raw binary payload (PDF / document bytes)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        doc="Creation timestamp",
    )


