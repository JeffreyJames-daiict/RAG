"""Ingestion Orchestrator: Bridges source connectors with database models & version tracking."""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from rag.connectors.base import BaseSourceConnector
from rag.core.database import DatabaseManager
from rag.core.db_models import Document, DocumentVersion
from rag.core.models import DocumentStatus, RawDocumentPayload, SyncSummary

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates source discovery, change-detection hashing, version bumping,

    and database persistence of document records and versions.
    """

    def __init__(self, db_manager: DatabaseManager, raw_storage_dir: Optional[str] = None):
        self.db_manager = db_manager
        self.raw_storage_dir = raw_storage_dir
        self.connectors: Dict[str, BaseSourceConnector] = {}

    def register_connector(self, name: str, connector: BaseSourceConnector) -> None:
        """Register a source connector with a unique identifier."""
        self.connectors[name] = connector

    def ingest_from_connector(
        self,
        connector_name: str,
        since: Optional[datetime] = None,
        force_fetch: bool = False,
    ) -> SyncSummary:
        """Execute discovery and ingestion for a specific connector."""
        if connector_name not in self.connectors:
            raise ValueError(f"Connector '{connector_name}' is not registered.")

        connector = self.connectors[connector_name]
        summary = SyncSummary(source_type=connector.source_type)

        with self.db_manager.session() as session:
            for descriptor in connector.discover(since=since):
                summary.discovered_count += 1
                try:
                    # Query if document already exists
                    stmt = select(Document).where(
                        Document.tenant_id == descriptor.tenant_id,
                        Document.source_uri == descriptor.source_uri,
                    )
                    existing_doc = session.execute(stmt).scalar_one_or_none()

                    # Fast-path skip if expected_hash matches existing content_hash
                    if (
                        existing_doc
                        and descriptor.expected_hash
                        and existing_doc.content_hash == descriptor.expected_hash
                        and not force_fetch
                    ):
                        summary.skipped_count += 1
                        continue

                    # Fetch raw document
                    payload: RawDocumentPayload = connector.fetch(descriptor)

                    if existing_doc:
                        if existing_doc.content_hash == payload.content_hash and not force_fetch:
                            # Content hasn't changed
                            summary.skipped_count += 1
                            continue

                        # Document content has changed: bump version & add historical version
                        new_version = existing_doc.version + 1
                        existing_doc.version = new_version
                        existing_doc.content_hash = payload.content_hash
                        existing_doc.filename = descriptor.filename
                        existing_doc.mime_type = descriptor.mime_type
                        existing_doc.status = DocumentStatus.FETCHED.value
                        existing_doc.updated_at = datetime.now(timezone.utc)

                        doc_version = DocumentVersion(
                            document_id=existing_doc.document_id,
                            version=new_version,
                            content_hash=payload.content_hash,
                            raw_location=descriptor.source_uri,
                            parser_version="v1.0",
                        )
                        session.add(doc_version)
                        summary.fetched_count += 1

                    else:
                        # New document discovery
                        new_doc = Document(
                            tenant_id=descriptor.tenant_id,
                            source=descriptor.source.value,
                            source_uri=descriptor.source_uri,
                            filename=descriptor.filename,
                            mime_type=descriptor.mime_type,
                            content_hash=payload.content_hash,
                            version=1,
                            status=DocumentStatus.FETCHED.value,
                        )
                        session.add(new_doc)
                        session.flush()  # populate document_id

                        doc_version = DocumentVersion(
                            document_id=new_doc.document_id,
                            version=1,
                            content_hash=payload.content_hash,
                            raw_location=descriptor.source_uri,
                            parser_version="v1.0",
                        )
                        session.add(doc_version)
                        summary.fetched_count += 1

                except Exception as e:
                    summary.failed_count += 1
                    error_msg = f"Failed to ingest {descriptor.source_uri}: {e}"
                    logger.exception(error_msg)
                    summary.errors.append(error_msg)

        return summary

    def ingest_all(self, since: Optional[datetime] = None) -> Dict[str, SyncSummary]:
        """Run ingestion across all registered source connectors."""
        results = {}
        for name in self.connectors:
            results[name] = self.ingest_from_connector(name, since=since)
        return results
