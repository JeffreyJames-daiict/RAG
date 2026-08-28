"""Ingestion Orchestrator: Bridges source connectors with database models, version tracking,

and robust incremental discovery (What exists, What is new, What is changed, What was deleted).
Parse and chunk stages are separate, explicit pipeline steps.
"""

import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from rag.connectors.base import BaseSourceConnector
from rag.core.database import DatabaseManager
from rag.core.db_models import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentVersion,
    DocumentVersionMetadata,
    PipelineError,
)
from rag.core.models import (
    DeletionCandidate,
    DiscoveredItem,
    DiscoveryReport,
    DocumentChangeType,
    DocumentDescriptor,
    DocumentStatus,
    PipelineStage,
    RawDocumentPayload,
    SourceType,
    SyncSummary,
)
from rag.parsing import ChunkingConfig, PDFParser, RecursiveTextChunker, TableChunker
from rag.storage.base import RawStorageBackend
from rag.core.config import get_settings

logger = logging.getLogger(__name__)

# MIME types supported by the PDF parser — read from central config
_settings = get_settings()
_PDF_MIME_TYPES: frozenset = _settings.pdf_supported_mime_types


def _normalize_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize datetime object to UTC timezone-aware datetime."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class IngestionPipeline:
    """Orchestrates source discovery, two-tier change detection (last-modified timestamp

    and SHA-256 cryptographic hash), version bumping, soft-deletion reconciliation,
    database persistence, and optional immutable raw storage.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        raw_storage: Optional[RawStorageBackend] = None,
        raw_storage_dir: Optional[str] = None,  # legacy — ignored if raw_storage provided
    ):
        self.db_manager = db_manager
        self.raw_storage = raw_storage
        self.connectors: Dict[str, BaseSourceConnector] = {}

        # Back-compat: auto-create a LocalRawStorageBackend from raw_storage_dir if given
        if raw_storage is None and raw_storage_dir:
            from rag.storage.local_storage import LocalRawStorageBackend
            self.raw_storage = LocalRawStorageBackend(base_dir=raw_storage_dir)
            logger.info("Auto-created LocalRawStorageBackend from raw_storage_dir: %s", raw_storage_dir)

    def register_connector(self, name: str, connector: BaseSourceConnector) -> None:
        """Register a source connector with a unique identifier."""
        self.connectors[name] = connector

    def get_connector(self, name: str) -> BaseSourceConnector:
        """Retrieve a registered connector by name."""
        if name not in self.connectors:
            raise ValueError(f"Connector '{name}' is not registered.")
        return self.connectors[name]

    def discover_changes(
        self,
        connector_name: str,
        since: Optional[datetime] = None,
        full_scan: bool = True,
        verify_content_hash: bool = False,
    ) -> DiscoveryReport:
        """Perform differential discovery against the database without modifying state.

        Determines:
        1. What exists / is unchanged (identical timestamp/expected hash or content hash)
        2. What is new (unseen source URIs or resurrected from previous deletion)
        3. What is changed (source timestamp newer or content hash mismatch)
        4. What was deleted (active database documents absent from source discovery)
        """
        connector = self.get_connector(connector_name)
        report = DiscoveryReport(
            tenant_id=connector.tenant_id,
            source_type=connector.source_type,
        )

        with self.db_manager.session() as session:
            # Query all existing documents tracked for this tenant and source type
            stmt = select(Document).where(
                Document.tenant_id == connector.tenant_id,
                Document.source == connector.source_type.value,
            )
            existing_docs: Dict[str, Document] = {
                doc.source_uri: doc for doc in session.execute(stmt).scalars().all()
            }

            discovered_uris: Set[str] = set()

            for descriptor in connector.discover(since=since):
                report.total_discovered += 1
                discovered_uris.add(descriptor.source_uri)
                existing_doc = existing_docs.get(descriptor.source_uri)

                if existing_doc is None:
                    # Brand new document
                    report.new_items.append(
                        DiscoveredItem(
                            descriptor=descriptor,
                            change_type=DocumentChangeType.NEW,
                            reason="New document discovered in source.",
                        )
                    )
                elif existing_doc.status == DocumentStatus.DELETED.value:
                    # Document was previously deleted and has reappeared (resurrection)
                    report.new_items.append(
                        DiscoveredItem(
                            descriptor=descriptor,
                            change_type=DocumentChangeType.NEW,
                            reason="Document resurrected after prior deletion.",
                            existing_document_id=existing_doc.document_id,
                            existing_version=existing_doc.version,
                            existing_hash=existing_doc.content_hash,
                        )
                    )
                elif existing_doc.status == DocumentStatus.FAILED.value:
                    # Document failed in previous run -> queue for retry
                    report.changed_items.append(
                        DiscoveredItem(
                            descriptor=descriptor,
                            change_type=DocumentChangeType.CHANGED,
                            reason="Document previously in FAILED state; queued for retry.",
                            existing_document_id=existing_doc.document_id,
                            existing_version=existing_doc.version,
                            existing_hash=existing_doc.content_hash,
                        )
                    )
                else:
                    # Document exists in DB in active status -> evaluate change detection
                    norm_last_mod = _normalize_utc(descriptor.last_modified_at)
                    norm_db_updated = _normalize_utc(existing_doc.updated_at)

                    if descriptor.expected_hash and descriptor.expected_hash == existing_doc.content_hash:
                        report.unchanged_items.append(
                            DiscoveredItem(
                                descriptor=descriptor,
                                change_type=DocumentChangeType.UNCHANGED,
                                reason="Source expected hash/ETag matches stored content hash.",
                                existing_document_id=existing_doc.document_id,
                                existing_version=existing_doc.version,
                                existing_hash=existing_doc.content_hash,
                            )
                        )
                    elif (
                        norm_last_mod
                        and norm_db_updated
                        and norm_last_mod <= norm_db_updated
                        and not verify_content_hash
                    ):
                        report.unchanged_items.append(
                            DiscoveredItem(
                                descriptor=descriptor,
                                change_type=DocumentChangeType.UNCHANGED,
                                reason=f"Last-modified timestamp ({norm_last_mod.isoformat()}) <= DB updated timestamp ({norm_db_updated.isoformat()}).",
                                existing_document_id=existing_doc.document_id,
                                existing_version=existing_doc.version,
                                existing_hash=existing_doc.content_hash,
                            )
                        )
                    elif verify_content_hash:
                        # Full cryptographic verification
                        try:
                            payload = connector.fetch(descriptor)
                            if payload.content_hash == existing_doc.content_hash:
                                report.unchanged_items.append(
                                    DiscoveredItem(
                                        descriptor=descriptor,
                                        change_type=DocumentChangeType.UNCHANGED,
                                        reason="Cryptographic SHA-256 content hash is identical.",
                                        existing_document_id=existing_doc.document_id,
                                        existing_version=existing_doc.version,
                                        existing_hash=existing_doc.content_hash,
                                    )
                                )
                            else:
                                report.changed_items.append(
                                    DiscoveredItem(
                                        descriptor=descriptor,
                                        change_type=DocumentChangeType.CHANGED,
                                        reason=f"Content hash mismatch (DB: {existing_doc.content_hash[:12]}..., Source: {payload.content_hash[:12]}...).",
                                        existing_document_id=existing_doc.document_id,
                                        existing_version=existing_doc.version,
                                        existing_hash=existing_doc.content_hash,
                                    )
                                )
                        except Exception as e:
                            report.changed_items.append(
                                DiscoveredItem(
                                    descriptor=descriptor,
                                    change_type=DocumentChangeType.CHANGED,
                                    reason=f"Failed to verify content hash ({e}); marked for ingestion.",
                                    existing_document_id=existing_doc.document_id,
                                    existing_version=existing_doc.version,
                                    existing_hash=existing_doc.content_hash,
                                )
                            )
                    else:
                        # Timestamp is newer or no expected_hash provided
                        reason_msg = "Source last-modified timestamp is newer than DB record." if descriptor.last_modified_at else "No hash/timestamp available; marked for content inspection."
                        report.changed_items.append(
                            DiscoveredItem(
                                descriptor=descriptor,
                                change_type=DocumentChangeType.CHANGED,
                                reason=reason_msg,
                                existing_document_id=existing_doc.document_id,
                                existing_version=existing_doc.version,
                                existing_hash=existing_doc.content_hash,
                            )
                        )

            # Deletion Detection: active DB documents absent from full discovery crawl
            if full_scan and since is None:
                for uri, doc in existing_docs.items():
                    if doc.status != DocumentStatus.DELETED.value and uri not in discovered_uris:
                        report.deleted_items.append(
                            DeletionCandidate(
                                document_id=doc.document_id,
                                source_uri=doc.source_uri,
                                filename=doc.filename,
                                tenant_id=doc.tenant_id,
                                source=SourceType(doc.source),
                                last_known_version=doc.version,
                            )
                        )

        return report

    def ingest_from_connector(
        self,
        connector_name: str,
        since: Optional[datetime] = None,
        force_fetch: bool = False,
        handle_deletions: bool = True,
        dry_run: bool = False,
    ) -> SyncSummary:
        """Execute differential discovery and incremental ingestion for a specific connector.

        Handles new insertions, version bumps for content changes, idempotent skips
        for unchanged items, and soft-deletions for removed source files.
        """
        connector = self.get_connector(connector_name)
        report = self.discover_changes(
            connector_name=connector_name,
            since=since,
            full_scan=(since is None),
            verify_content_hash=force_fetch,
        )

        summary = SyncSummary(
            source_type=connector.source_type,
            discovered_count=report.total_discovered,
            new_count=len(report.new_items),
            changed_count=len(report.changed_items),
            unchanged_count=len(report.unchanged_items),
            deleted_count=len(report.deleted_items),
            discovery_report=report,
        )

        if dry_run:
            logger.info("Dry-run requested: skipping database mutations.")
            return summary

        with self.db_manager.session() as session:
            # 1. Process NEW documents
            for item in report.new_items:
                try:
                    payload: RawDocumentPayload = connector.fetch(item.descriptor)

                    # Check if document row already exists (resurrection case)
                    existing_doc = None
                    if item.existing_document_id:
                        existing_doc = session.get(Document, item.existing_document_id)

                    if existing_doc:
                        # Restore previously deleted document
                        new_version = existing_doc.version + 1
                        existing_doc.version = new_version
                        existing_doc.content_hash = payload.content_hash
                        existing_doc.filename = item.descriptor.filename
                        existing_doc.mime_type = item.descriptor.mime_type
                        existing_doc.status = DocumentStatus.FETCHED.value
                        existing_doc.deleted_at = None
                        existing_doc.error = None
                        existing_doc.updated_at = datetime.now(timezone.utc)

                        doc_version = DocumentVersion(
                            document_id=existing_doc.document_id,
                            version=new_version,
                            content_hash=payload.content_hash,
                            raw_location=self._save_raw(
                                payload=payload,
                                doc_id=existing_doc.document_id,
                                version=new_version,
                                tenant_id=item.descriptor.tenant_id,
                            ),
                            parser_version=None,
                        )
                        session.add(doc_version)
                    else:
                        # Brand new document creation
                        new_doc = Document(
                            tenant_id=item.descriptor.tenant_id,
                            source=item.descriptor.source.value,
                            source_uri=item.descriptor.source_uri,
                            filename=item.descriptor.filename,
                            mime_type=item.descriptor.mime_type,
                            content_hash=payload.content_hash,
                            version=1,
                            status=DocumentStatus.FETCHED.value,
                            deleted_at=None,
                            error=None,
                        )
                        session.add(new_doc)
                        session.flush()

                        doc_version = DocumentVersion(
                            document_id=new_doc.document_id,
                            version=1,
                            content_hash=payload.content_hash,
                            raw_location=self._save_raw(
                                payload=payload,
                                doc_id=new_doc.document_id,
                                version=1,
                                tenant_id=item.descriptor.tenant_id,
                            ),
                            parser_version=None,
                        )
                        session.add(doc_version)

                    summary.fetched_count += 1

                except Exception as e:
                    summary.failed_count += 1
                    error_msg = f"Failed to ingest new document {item.descriptor.source_uri}: {e}"
                    logger.exception(error_msg)
                    summary.errors.append(error_msg)

                    # Persist failed document state and structured error record
                    try:
                        doc = None
                        if item.existing_document_id:
                            doc = session.get(Document, item.existing_document_id)
                        if not doc:
                            stmt = select(Document).where(
                                Document.tenant_id == item.descriptor.tenant_id,
                                Document.source_uri == item.descriptor.source_uri,
                            )
                            doc = session.execute(stmt).scalar_one_or_none()

                        if doc:
                            doc.status = DocumentStatus.FAILED.value
                            doc.error = str(e)
                            doc.updated_at = datetime.now(timezone.utc)
                            doc_id = doc.document_id
                        else:
                            failed_doc = Document(
                                tenant_id=item.descriptor.tenant_id,
                                source=item.descriptor.source.value,
                                source_uri=item.descriptor.source_uri,
                                filename=item.descriptor.filename,
                                mime_type=item.descriptor.mime_type,
                                content_hash="",
                                version=1,
                                status=DocumentStatus.FAILED.value,
                                error=str(e),
                            )
                            session.add(failed_doc)
                            session.flush()
                            doc_id = failed_doc.document_id

                        # Record in errors table
                        err_entry = PipelineError(
                            document_id=doc_id,
                            tenant_id=item.descriptor.tenant_id,
                            source_uri=item.descriptor.source_uri,
                            stage=PipelineStage.FETCH.value,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            stack_trace=traceback.format_exc(),
                        )
                        session.add(err_entry)
                    except Exception as log_err:
                        logger.error(f"Failed to record pipeline error to database: {log_err}")

            # 2. Process CHANGED documents
            for item in report.changed_items:
                try:
                    payload: RawDocumentPayload = connector.fetch(item.descriptor)

                    stmt = select(Document).where(
                        Document.tenant_id == item.descriptor.tenant_id,
                        Document.source_uri == item.descriptor.source_uri,
                    )
                    existing_doc = session.execute(stmt).scalar_one_or_none()

                    if not existing_doc:
                        summary.failed_count += 1
                        summary.errors.append(f"Document {item.descriptor.source_uri} not found in DB for update.")
                        continue

                    # Check if actual content hash changed or if only timestamp changed (only if not recovering from FAILED)
                    if existing_doc.status != DocumentStatus.FAILED.value and existing_doc.content_hash == payload.content_hash and not force_fetch:
                        # Content is identical, update metadata only without version bump
                        existing_doc.error = None
                        existing_doc.updated_at = datetime.now(timezone.utc)
                        summary.skipped_count += 1
                        continue

                    # If recovering from FAILED with empty hash, keep version=1, else bump version
                    if existing_doc.status == DocumentStatus.FAILED.value and existing_doc.content_hash == "":
                        new_version = existing_doc.version
                    else:
                        new_version = existing_doc.version + 1

                    existing_doc.version = new_version
                    existing_doc.content_hash = payload.content_hash
                    existing_doc.filename = item.descriptor.filename
                    existing_doc.mime_type = item.descriptor.mime_type
                    existing_doc.status = DocumentStatus.FETCHED.value
                    existing_doc.deleted_at = None
                    existing_doc.error = None
                    existing_doc.updated_at = datetime.now(timezone.utc)

                    doc_version = DocumentVersion(
                        document_id=existing_doc.document_id,
                        version=new_version,
                        content_hash=payload.content_hash,
                        raw_location=self._save_raw(
                            payload=payload,
                            doc_id=existing_doc.document_id,
                            version=new_version,
                            tenant_id=item.descriptor.tenant_id,
                        ),
                        parser_version=None,
                    )
                    session.add(doc_version)
                    summary.fetched_count += 1

                except Exception as e:
                    summary.failed_count += 1
                    error_msg = f"Failed to ingest changed document {item.descriptor.source_uri}: {e}"
                    logger.exception(error_msg)
                    summary.errors.append(error_msg)

                    # Persist failed document state and structured error record
                    try:
                        doc = session.get(Document, item.existing_document_id) if item.existing_document_id else None
                        if doc:
                            doc.status = DocumentStatus.FAILED.value
                            doc.error = str(e)
                            doc.updated_at = datetime.now(timezone.utc)

                        err_entry = PipelineError(
                            document_id=item.existing_document_id,
                            tenant_id=item.descriptor.tenant_id,
                            source_uri=item.descriptor.source_uri,
                            stage=PipelineStage.FETCH.value,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            stack_trace=traceback.format_exc(),
                        )
                        session.add(err_entry)
                    except Exception as log_err:
                        logger.error(f"Failed to record pipeline error to database: {log_err}")

            # 3. Process UNCHANGED documents
            summary.skipped_count += len(report.unchanged_items)

            # 4. Process DELETED documents
            if handle_deletions:
                for candidate in report.deleted_items:
                    try:
                        doc_to_delete = session.get(Document, candidate.document_id)
                        if doc_to_delete:
                            doc_to_delete.status = DocumentStatus.DELETED.value
                            doc_to_delete.deleted_at = datetime.now(timezone.utc)
                            doc_to_delete.updated_at = datetime.now(timezone.utc)
                            logger.info(f"Soft-deleted document {candidate.source_uri} (ID: {candidate.document_id})")
                    except Exception as e:
                        error_msg = f"Failed to mark document {candidate.source_uri} as deleted: {e}"
                        logger.exception(error_msg)
                        summary.errors.append(error_msg)

                        try:
                            err_entry = PipelineError(
                                document_id=candidate.document_id,
                                tenant_id=candidate.tenant_id,
                                source_uri=candidate.source_uri,
                                stage=PipelineStage.PIPELINE.value,
                                error_type=type(e).__name__,
                                error_message=str(e),
                                stack_trace=traceback.format_exc(),
                            )
                            session.add(err_entry)
                        except Exception as log_err:
                            logger.error(f"Failed to record deletion error to database: {log_err}")

        return summary

    def ingest_all(
        self,
        since: Optional[datetime] = None,
        force_fetch: bool = False,
        handle_deletions: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, SyncSummary]:
        """Run discovery and ingestion across all registered source connectors."""
        results = {}
        for name in self.connectors:
            results[name] = self.ingest_from_connector(
                name,
                since=since,
                force_fetch=force_fetch,
                handle_deletions=handle_deletions,
                dry_run=dry_run,
            )
        return results

    # ------------------------------------------------------------------
    # Parse & Chunk Stage
    # ------------------------------------------------------------------

    def parse_and_chunk(
        self,
        connector_name: str,
        chunking_config: Optional[ChunkingConfig] = None,
        tenant_id: Optional[str] = None,
        force_reparse: bool = False,
    ) -> Dict[str, int]:
        """Parse and chunk all FETCHED documents for a connector.

        This is an **explicit, separate** pipeline stage. It is intentionally
        decoupled from ingestion so you can re-parse without re-fetching.

        For each document with status FETCHED (or PARSED/CHUNKED if force_reparse):
          1. Reads raw bytes from the immutable raw store (falls back to connector).
          2. Parses text using PDFParser (PyMuPDF).
          3. Chunks text using RecursiveTextChunker (token-aware, recursive).
          4. Deletes existing chunks for this document+version (idempotent).
          5. Bulk-inserts new Chunk rows.
          6. Updates DocumentVersion.parser_version.
          7. Advances Document.status → PARSED → CHUNKED.

        Args:
            connector_name:   Name of the registered connector to pull bytes from.
            chunking_config:  Override ChunkingConfig (defaults to values from Settings).
            tenant_id:        Tenant scope. Defaults to ``settings.default_tenant_id``.
            force_reparse:    If True, re-parse PARSED and CHUNKED docs too.

        Returns:
            Dict mapping document_id → number of chunks created.
        """
        cfg = get_settings()
        tenant_id = tenant_id or cfg.default_tenant_id

        connector = self.get_connector(connector_name)
        parser = PDFParser(extract_tables=cfg.table.enabled, table_config=cfg.table)
        text_chunker = RecursiveTextChunker(chunking_config or ChunkingConfig())
        effective_chunk_size = (chunking_config.chunk_size if chunking_config else cfg.chunking.chunk_size)
        table_chunker = TableChunker(
            table_config=cfg.table,
            encoding_model=cfg.chunking.encoding_model,
            max_tokens_per_chunk=effective_chunk_size,
            chunking_version=cfg.chunking.chunking_version,
        )

        # Determine which statuses to process
        target_statuses = [DocumentStatus.FETCHED.value]
        if force_reparse:
            target_statuses += [DocumentStatus.PARSED.value, DocumentStatus.CHUNKED.value]

        results: Dict[str, int] = {}

        with self.db_manager.session() as session:
            stmt = select(Document).where(
                Document.tenant_id == tenant_id,
                Document.source == connector.source_type.value,
                Document.status.in_(target_statuses),
            )
            docs: List[Document] = session.execute(stmt).scalars().all()

            logger.info(
                "parse_and_chunk: found %d document(s) to process (statuses: %s)",
                len(docs), target_statuses,
            )

            for doc in docs:
                doc_id = doc.document_id
                label = doc.filename

                try:
                    # ── Step 1: Resolve raw bytes ─────────────────────────────
                    raw_bytes = self._read_raw_or_fetch(
                        doc=doc,
                        connector=connector,
                    )

                    # ── Step 2: Parse (Text + Tables) ─────────────────────────
                    if doc.mime_type not in _PDF_MIME_TYPES:
                        logger.warning(
                            "Skipping '%s': MIME type '%s' is not supported by PDFParser.",
                            label, doc.mime_type,
                        )
                        results[doc_id] = 0
                        continue

                    doc.status = DocumentStatus.PARSED.value
                    doc.updated_at = datetime.now(timezone.utc)

                    parsed = parser.parse(raw_bytes, filename=label)

                    # ── Step 3: Chunk Text & Tables ───────────────────────────
                    # 3a. Plain text chunks
                    all_chunks: List[ChunkSchema] = []
                    text_chunks = text_chunker.chunk(
                        parsed_doc=parsed,
                        document_id=doc_id,
                        version=doc.version,
                    )
                    for tc in text_chunks:
                        if "chunk_type" not in tc.metadata:
                            tc.metadata["chunk_type"] = "text"
                    all_chunks.extend(text_chunks)

                    # 3b. Table chunks (if table processing enabled)
                    table_chunk_count = 0
                    if cfg.table.enabled and parsed.tables:
                        for tab in parsed.tables:
                            t_chunks = table_chunker.chunk_table(
                                table=tab,
                                document_id=doc_id,
                                version=doc.version,
                                document_title=label,
                                starting_chunk_index=len(all_chunks),
                            )
                            all_chunks.extend(t_chunks)
                            table_chunk_count += len(t_chunks)

                    # 3c. Normalize sequential chunk_index
                    for idx, c in enumerate(all_chunks):
                        c.chunk_index = idx

                    # ── Step 4: Delete existing chunks (idempotent) ───────────
                    session.execute(
                        delete(Chunk).where(
                            Chunk.document_id == doc_id,
                            Chunk.version == doc.version,
                        )
                    )

                    # ── Step 5: Bulk-insert new chunks & relational ChunkMetadata ─
                    for cs in all_chunks:
                        ch_id_str = str(cs.chunk_id)
                        meta_dict = cs.metadata or {}

                        db_chunk = Chunk(
                            chunk_id=ch_id_str,
                            document_id=doc_id,
                            version=cs.version,
                            chunk_index=cs.chunk_index,
                            text=cs.text,
                            section=cs.section,
                            page=cs.page,
                            chunk_metadata=meta_dict,
                            chunking_version=cs.chunking_version,
                        )
                        session.add(db_chunk)

                        chunk_meta_row = ChunkMetadata(
                            chunk_id=ch_id_str,
                            document_id=doc_id,
                            version=cs.version,
                            chunk_type=meta_dict.get("chunk_type", "text"),
                            page=cs.page,
                            page_start=meta_dict.get("page_start", cs.page),
                            page_end=meta_dict.get("page_end", cs.page),
                            token_count=meta_dict.get("token_count", 0),
                            language=meta_dict.get("language", getattr(parsed, "primary_language", "eng")),
                            language_confidence=meta_dict.get("language_confidence", 1.0),
                            content_source=meta_dict.get("content_source", "native"),
                            quality_score=meta_dict.get("quality_score"),
                            ocr_applied=meta_dict.get("ocr_applied", False),
                            ocr_confidence=meta_dict.get("ocr_confidence"),
                            ocr_provider=meta_dict.get("ocr_provider"),
                            preprocessing_profile=meta_dict.get("preprocessing_profile"),
                            ocr_attempt=meta_dict.get("ocr_attempt", 1),
                            table_id=meta_dict.get("table_id"),
                            parent_table_id=meta_dict.get("parent_id") or meta_dict.get("table_id"),
                            row_start=meta_dict.get("row_start"),
                            row_end=meta_dict.get("row_end"),
                            total_rows=meta_dict.get("total_rows"),
                            total_columns=meta_dict.get("total_columns"),
                            is_continuation=meta_dict.get("is_continuation", False),
                            validation_status=meta_dict.get("validation_status", "valid"),
                            serialization_strategy=meta_dict.get("serialization_strategy", "hybrid"),
                            extra_attributes={k: v for k, v in meta_dict.items() if k not in {
                                "chunk_type", "page", "page_start", "page_end", "token_count",
                                "language", "language_confidence", "content_source", "quality_score",
                                "ocr_applied", "ocr_confidence", "ocr_provider", "preprocessing_profile",
                                "ocr_attempt", "table_id", "parent_id", "row_start", "row_end",
                                "total_rows", "total_columns", "is_continuation", "validation_status",
                                "serialization_strategy",
                            }},
                        )
                        session.add(chunk_meta_row)

                    # ── Step 6: Update DocumentVersion & DocumentVersionMetadata ─
                    ver_stmt = select(DocumentVersion).where(
                        DocumentVersion.document_id == doc_id,
                        DocumentVersion.version == doc.version,
                    )
                    doc_ver = session.execute(ver_stmt).scalar_one_or_none()
                    if doc_ver:
                        doc_ver.parser_version = parsed.parser_version

                        # Aggregate page quality scores & metrics
                        page_scores = []
                        for p in parsed.pages:
                            page_scores.append({
                                "page_number": p.page_number,
                                "content_source": getattr(p, "content_source", "native"),
                                "detected_language": getattr(p, "detected_language", "eng"),
                                "language_confidence": getattr(p, "language_confidence", 1.0),
                                "ocr_applied": getattr(p, "ocr_applied", False),
                                "quality_score": getattr(p, "quality_score", None),
                                "ocr_confidence": getattr(p, "ocr_confidence", None),
                                "ocr_provider": getattr(p, "ocr_provider", None),
                                "preprocessing_profile": getattr(p, "preprocessing_profile", None),
                                "ocr_attempt": getattr(p, "ocr_attempt", 1),
                            })

                        valid_qualities = [p.quality_score for p in parsed.pages if getattr(p, "quality_score", None) is not None]
                        avg_quality = sum(valid_qualities) / len(valid_qualities) if valid_qualities else 1.0

                        valid_confs = [p.ocr_confidence for p in parsed.pages if getattr(p, "ocr_confidence", None) is not None]
                        avg_ocr_conf = sum(valid_confs) / len(valid_confs) if valid_confs else None

                        tab_diag = parsed.table_diagnostics or {}
                        ocr_diag_map = getattr(parsed, "ocr_diagnostics", {})

                        doc_ver.metadata_snapshot = {
                            "num_pages": parsed.num_pages,
                            "primary_language": getattr(parsed, "primary_language", "eng"),
                            "languages_detected": getattr(parsed, "languages_detected", ["eng"]),
                            "total_chunks": len(all_chunks),
                            "text_chunks": len(text_chunks),
                            "table_chunks": table_chunk_count,
                            "avg_quality_score": round(avg_quality, 4),
                            "avg_ocr_confidence": round(avg_ocr_conf, 4) if avg_ocr_conf is not None else None,
                            "table_diagnostics": tab_diag,
                            "ocr_diagnostics": ocr_diag_map,
                            "page_quality_scores": page_scores,
                            "pdf_metadata": parsed.pdf_metadata,
                        }

                        # Upsert relational DocumentVersionMetadata record
                        session.execute(
                            delete(DocumentVersionMetadata).where(
                                DocumentVersionMetadata.document_id == doc_id,
                                DocumentVersionMetadata.version == doc.version,
                            )
                        )
                        doc_ver_meta = DocumentVersionMetadata(
                            document_id=doc_id,
                            version=doc.version,
                            primary_language=getattr(parsed, "primary_language", "eng"),
                            languages_detected=getattr(parsed, "languages_detected", ["eng"]),
                            num_pages=parsed.num_pages,
                            total_chunks=len(all_chunks),
                            text_chunks=len(text_chunks),
                            table_chunks=table_chunk_count,
                            avg_quality_score=round(avg_quality, 4) if avg_quality is not None else None,
                            avg_ocr_confidence=round(avg_ocr_conf, 4) if avg_ocr_conf is not None else None,
                            tables_detected=tab_diag.get("tables_detected", 0),
                            tables_stitched=tab_diag.get("tables_stitched", 0),
                            tables_processed=tab_diag.get("tables_processed", 0),
                            tables_warning=tab_diag.get("tables_warning", 0),
                            tables_failed=tab_diag.get("tables_failed", 0),
                            pages_native=ocr_diag_map.get("pages_native", 0),
                            pages_scanned=ocr_diag_map.get("pages_scanned", 0),
                            pages_mixed=ocr_diag_map.get("pages_mixed", 0),
                            ocr_pages_processed=ocr_diag_map.get("ocr_pages_processed", 0),
                            ocr_pages_failed=ocr_diag_map.get("ocr_pages_failed", 0),
                            pdf_title=parsed.pdf_metadata.get("title"),
                            pdf_author=parsed.pdf_metadata.get("author"),
                            page_scores=page_scores,
                        )
                        session.add(doc_ver_meta)

                    # ── Step 7: Advance status → CHUNKED ─────────────────────
                    doc.status = DocumentStatus.CHUNKED.value
                    doc.error = None
                    doc.updated_at = datetime.now(timezone.utc)

                    results[doc_id] = len(all_chunks)
                    logger.info(
                        "✅ Parsed & chunked '%s' (doc %s): %d total chunks "
                        "(%d text, %d table chunks across %d tables, %d pages)",
                        label, doc_id, len(all_chunks),
                        len(text_chunks), table_chunk_count, len(parsed.tables), parsed.num_pages,
                    )

                except Exception as exc:
                    results[doc_id] = 0
                    error_msg = str(exc)
                    logger.exception("Failed to parse/chunk '%s' (%s): %s", label, doc_id, error_msg)

                    # Persist failure to document + errors table
                    try:
                        doc.status = DocumentStatus.FAILED.value
                        doc.error = error_msg
                        doc.updated_at = datetime.now(timezone.utc)

                        err_entry = PipelineError(
                            document_id=doc_id,
                            tenant_id=tenant_id,
                            source_uri=doc.source_uri,
                            stage=PipelineStage.PARSE.value,
                            error_type=type(exc).__name__,
                            error_message=error_msg,
                            stack_trace=traceback.format_exc(),
                        )
                        session.add(err_entry)
                    except Exception as log_err:
                        logger.error("Failed to persist parse error to DB: %s", log_err)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rebuild_descriptor(doc: Document) -> DocumentDescriptor:
        """Reconstruct a DocumentDescriptor from a Document ORM row for re-fetching."""
        from rag.core.models import DocumentDescriptor, SourceType
        return DocumentDescriptor(
            source=SourceType(doc.source),
            source_uri=doc.source_uri,
            filename=doc.filename,
            mime_type=doc.mime_type,
            tenant_id=doc.tenant_id,
        )

    def _save_raw(
        self,
        payload: RawDocumentPayload,
        doc_id: str,
        version: int,
        tenant_id: str,
    ) -> str:
        """Persist raw bytes to the raw storage backend (if configured).

        Returns the storage key when raw storage is active, or the
        original ``source_uri`` as a fallback when no backend is set.

        The write is **idempotent**: if the key already exists (e.g. a
        re-ingest of the same version) the existing file is kept and the
        key is returned unchanged — no data is overwritten.

        Args:
            payload:   The fetched payload containing raw bytes and descriptor.
            doc_id:    Document UUID string.
            version:   Version number being stored.
            tenant_id: Tenant scope.

        Returns:
            Storage key string to persist in ``DocumentVersion.raw_location``.
        """
        if self.raw_storage is None:
            # No backend configured — fall back to source_uri so the column
            # is still populated with something meaningful.
            return payload.descriptor.source_uri

        try:
            key = self.raw_storage.write(
                tenant_id=tenant_id,
                document_id=doc_id,
                version=version,
                filename=payload.descriptor.filename,
                content=payload.content,
                force=False,  # immutable — never overwrite
            )
            logger.info(
                "Raw storage: saved %d bytes → key '%s'",
                len(payload.content), key,
            )
            return key
        except FileExistsError:
            # Already stored (re-run scenario) — reconstruct and return the key
            key = self.raw_storage.make_key(
                tenant_id, doc_id, version, payload.descriptor.filename
            )
            logger.debug("Raw storage: key already exists (idempotent): %s", key)
            return key
        except Exception as exc:
            # Storage failure is non-fatal: log and fall back to source_uri
            logger.warning(
                "Raw storage write failed for doc %s v%d — falling back to source_uri. Error: %s",
                doc_id, version, exc,
            )
            return payload.descriptor.source_uri

    def _read_raw_or_fetch(
        self,
        doc: Document,
        connector: BaseSourceConnector,
    ) -> bytes:
        """Return raw bytes, preferring the immutable raw store over live connector.

        Resolution order:
        1. Raw storage backend (if configured) + ``DocumentVersion.raw_location``
           is a valid storage key that exists in the backend.
        2. Fallback: re-fetch from the live connector.

        Args:
            doc:       The Document ORM row (used to look up the version row).
            connector: Registered source connector (used as fallback).

        Returns:
            Raw bytes of the document payload.
        """
        # Try raw storage first
        if self.raw_storage is not None:
            with self.db_manager.session() as ver_session:
                ver_stmt = select(DocumentVersion).where(
                    DocumentVersion.document_id == doc.document_id,
                    DocumentVersion.version == doc.version,
                )
                doc_ver = ver_session.execute(ver_stmt).scalar_one_or_none()

            if doc_ver and doc_ver.raw_location:
                storage_key = doc_ver.raw_location
                if self.raw_storage.exists(storage_key):
                    logger.info(
                        "parse_and_chunk: reading '%s' v%d from raw store (key: %s)",
                        doc.filename, doc.version, storage_key,
                    )
                    return self.raw_storage.read(storage_key)
                else:
                    logger.warning(
                        "Raw storage key '%s' not found on disk — falling back to connector.",
                        storage_key,
                    )

        # Fallback: re-fetch from live connector
        logger.info(
            "parse_and_chunk: fetching '%s' from live connector (no raw store key).",
            doc.filename,
        )
        descriptor = self._rebuild_descriptor(doc)
        payload = connector.fetch(descriptor)
        return payload.content

