"""SQLAlchemy 2.0 ORM database models for the Enterprise RAG pipeline."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Enum as SQLEnum,
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

    __table_args__ = (
        Index("ix_chunks_doc_ver_idx", "document_id", "version", "chunk_index"),
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


