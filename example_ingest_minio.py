"""MinIO-backed ingestion pipeline example.

Demonstrates:
  1. Differential document discovery from MinIO bucket
  2. Incremental ingestion with hash-based change detection
  3. Parse & Chunk (text + tables + figures)
  4. Auto-embedding via EmbeddingService
  5. Full DB summary of produced chunks, metadata, and assets
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import func, select

from rag.connectors.s3 import S3Connector
from rag.core.config import get_settings
from rag.core.database import DatabaseManager
from rag.core.db_models import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentAsset,
    DocumentVersion,
    DocumentVersionMetadata,
)
from rag.core.migrate import run_migrations
from rag.ingestion.pipeline import IngestionPipeline
from rag.parsing import ChunkingConfig
from rag.storage import get_storage_backend

load_dotenv()


# ─────────────────────────────────────────────────
# MinIO connection details (read from .env)
# ─────────────────────────────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true"
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rag")
MINIO_PREFIX = os.getenv("MINIO_PREFIX", "")  # e.g. "inputs/" to limit to a subfolder

CONNECTOR_KEY = "minio"


def _build_s3_connector() -> S3Connector:
    scheme = "https" if MINIO_SECURE else "http"
    endpoint_url = f"{scheme}://{MINIO_ENDPOINT}"
    print(f"🪣  MinIO endpoint : {endpoint_url}")
    print(f"🪣  Bucket         : {MINIO_BUCKET}")
    print(f"🪣  Prefix filter  : '{MINIO_PREFIX}' (empty = all)")
    return S3Connector(
        bucket_name=MINIO_BUCKET,
        prefix=MINIO_PREFIX,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        endpoint_url=endpoint_url,
        tenant_id="default",
    )


def _separator(label: str) -> None:
    print("\n" + "=" * 60)
    print(f" {label}")
    print("=" * 60)


def main():
    _separator("🪣  ENTERPRISE RAG — MinIO INGESTION")

    settings = get_settings()
    print(settings.summary())

    # ── 1. Build MinIO connector ───────────────────────────────────────────────
    connector = _build_s3_connector()
    connector.validate_connection()
    print("✅ MinIO connection validated.\n")

    # ── 2. Database & raw storage backend ─────────────────────────────────────
    db_manager = DatabaseManager(
        db_url=settings.database.url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        echo=settings.database.echo,
    )
    raw_store = get_storage_backend(settings=settings, db_manager=db_manager)

    if settings.pipeline.auto_migrate:
        run_migrations(db_manager)

    # ── 3. Pipeline setup ─────────────────────────────────────────────────────
    pipeline = IngestionPipeline(db_manager=db_manager, raw_storage=raw_store)
    pipeline.register_connector(CONNECTOR_KEY, connector)

    # ── 4. Pre-ingestion differential discovery ───────────────────────────────
    _separator("🔍 STEP 1: DIFFERENTIAL DISCOVERY")
    diff_report = pipeline.discover_changes(CONNECTOR_KEY)

    print(f"  Total in MinIO bucket : {diff_report.total_discovered}")
    print(f"  Already tracked in DB : {diff_report.total_existing}")
    print(f"  ➕ New files           : {len(diff_report.new_items)}")
    print(f"  ♻️  Changed files       : {len(diff_report.changed_items)}")
    print(f"  ✅ Unchanged (skip)    : {len(diff_report.unchanged_items)}")
    print(f"  🗑️  Deleted from source : {len(diff_report.deleted_items)}")

    for item in diff_report.new_items:
        sz = item.descriptor.custom_metadata.get("size_bytes", 0)
        print(f"     + [NEW]     {item.descriptor.filename}  ({sz:,} bytes)")
    for item in diff_report.changed_items:
        print(f"     Δ [CHANGED] {item.descriptor.filename}  (v{item.existing_version} → v{item.existing_version + 1})")
    for item in diff_report.unchanged_items:
        print(f"     ✓ [SKIP]    {item.descriptor.filename}  (v{item.existing_version}, hash unchanged)")
    for item in diff_report.deleted_items:
        name = getattr(item, 'filename', getattr(item, 'source_uri', str(item)))
        print(f"     ✗ [DELETED] {name}")

    # ── 5. Incremental ingestion ───────────────────────────────────────────────
    _separator("⚡ STEP 2: INCREMENTAL INGESTION")
    summary = pipeline.ingest_from_connector(CONNECTOR_KEY, handle_deletions=True)

    print(f"  • Discovered : {summary.discovered_count}")
    print(f"  • New        : {summary.new_count}")
    print(f"  • Changed    : {summary.changed_count}")
    print(f"  • Unchanged  : {summary.unchanged_count}")
    print(f"  • Deleted    : {summary.deleted_count}")
    print(f"  • Fetched    : {summary.fetched_count}")
    print(f"  • Failed     : {summary.failed_count}")

    # ── 6. DB document records ─────────────────────────────────────────────────
    _separator("📋 STEP 3: DATABASE DOCUMENT STATE")
    with db_manager.session() as session:
        docs = session.execute(select(Document).order_by(Document.created_at)).scalars().all()
        for doc in docs:
            icon = {"FETCHED": "🟢", "DELETED": "🔴", "FAILED": "⛔", "PARSED": "🔵", "CHUNKED": "🟣", "INDEXED": "✅"}.get(doc.status, "🟡")
            print(f"\n{icon} {doc.filename}")
            print(f"   ID:      {doc.document_id}")
            print(f"   Status:  {doc.status}")
            print(f"   Version: v{doc.version}")
            print(f"   Hash:    {doc.content_hash[:16]}..." if doc.content_hash else "   Hash:    (none)")

            vers = session.execute(
                select(DocumentVersion).where(DocumentVersion.document_id == doc.document_id)
            ).scalars().all()
            for ver in vers:
                raw_loc = ver.raw_location or "(not stored)"
                is_stored = raw_store.exists(raw_loc) if ver.raw_location else False
                stored_icon = "💾" if is_stored else "🔗"
                size_str = ""
                if is_stored and hasattr(raw_store, "get_storage_size_bytes"):
                    sz = raw_store.get_storage_size_bytes(raw_loc)
                    size_str = f" ({sz:,} bytes)" if sz else ""
                print(f"   ↳ v{ver.version} {stored_icon} {raw_loc}{size_str}")

    # ── 7. Parse & Chunk (with auto-embed) ────────────────────────────────────
    _separator("📄 STEP 4: PARSE & CHUNK")
    config = ChunkingConfig(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        chunking_version=settings.chunking.chunking_version,
        encoding_model=settings.chunking.encoding_model,
    )
    chunk_results = pipeline.parse_and_chunk(
        CONNECTOR_KEY,
        chunking_config=config,
        force_reparse=False,
        auto_embed=True,
    )
    total_chunks = sum(chunk_results.values())
    print(f"\n✅ Parsing complete: {len(chunk_results)} doc(s) processed, {total_chunks} total chunks.")

    # ── 8. Chunk summary per document ─────────────────────────────────────────
    print("\n📊 Chunk breakdown per document:\n")
    with db_manager.session() as session:
        docs = session.execute(select(Document).order_by(Document.created_at)).scalars().all()
        for doc in docs:
            all_chunks = session.execute(
                select(Chunk)
                .where(Chunk.document_id == doc.document_id, Chunk.version == doc.version)
                .order_by(Chunk.chunk_index)
            ).scalars().all()

            text_n = sum(1 for c in all_chunks if c.chunk_metadata.get("chunk_type") == "text")
            table_n = sum(1 for c in all_chunks if c.chunk_metadata.get("chunk_type") == "table")
            figure_n = sum(1 for c in all_chunks if c.chunk_metadata.get("chunk_type") == "figure")
            icon = {"FETCHED": "🟢", "DELETED": "🔴", "FAILED": "⛔", "PARSED": "🔵", "CHUNKED": "🟣", "INDEXED": "✅"}.get(doc.status, "🟡")

            print(f"  {icon} {doc.filename}  (v{doc.version}, status={doc.status})")
            print(f"     Total: {len(all_chunks)} chunks  •  {text_n} text / {table_n} table / {figure_n} figure")

            for ch in all_chunks[:4]:
                ch_type = ch.chunk_metadata.get("chunk_type", "text")
                src = ch.chunk_metadata.get("content_source", "native")
                q = ch.chunk_metadata.get("quality_score")
                q_str = f" q={q:.2f}" if q is not None else ""
                badge = {"table": "📊", "figure": "🖼️", "text": "📝"}.get(ch_type, "📄")
                tok = ch.chunk_metadata.get("token_count", "?")
                preview = ch.text[:88].replace("\n", " ")
                print(f"     ↳ {badge} #{ch.chunk_index} pg{ch.page} ({tok} tok | {src}{q_str}) | {preview!r}...")
            if len(all_chunks) > 4:
                print(f"     ↳ ... and {len(all_chunks) - 4} more chunks")

    # ── 9. Document assets (visual figures) ───────────────────────────────────
    _separator("🖼️  STEP 5: DOCUMENT ASSETS (VISUAL FIGURES)")
    with db_manager.session() as session:
        assets = session.execute(select(DocumentAsset).order_by(DocumentAsset.created_at)).scalars().all()
        if not assets:
            print("  (No visual assets extracted — documents may be text-only or figures below threshold)")
        else:
            print(f"  Total assets extracted: {len(assets)}\n")
            for a in assets:
                print(f"  🖼️  {a.asset_type}  (Page {a.page_number}, {a.width}×{a.height})")
                print(f"     Caption     : {a.caption or '(none)'}")
                print(f"     Storage URI : {a.storage_uri}")
                print(f"     OCR text    : {(a.ocr_text or '')[:80]}...")
                print(f"     Entities    : {', '.join(a.entities or [])}")
                print()

    # ── 10. Normalized metadata tables ────────────────────────────────────────
    _separator("🗄️  STEP 6: RELATIONAL METADATA TABLES")
    with db_manager.session() as session:
        print("\n📋 document_version_metadata:")
        v_metas = session.execute(select(DocumentVersionMetadata)).scalars().all()
        for vm in v_metas:
            print(
                f"   • Doc {vm.document_id[:8]}... v{vm.version} | "
                f"Lang: {vm.primary_language} | "
                f"Pages: {vm.num_pages} | "
                f"Chunks: {vm.total_chunks} ({vm.text_chunks} txt, {vm.table_chunks} tab) | "
                f"Avg Quality: {vm.avg_quality_score or 0.0:.2f} | "
                f"OCR Conf: {vm.avg_ocr_confidence or 0.0:.2f}"
            )

        print("\n📋 chunk_metadata (first 10):")
        c_metas = session.execute(
            select(ChunkMetadata).order_by(ChunkMetadata.document_id, ChunkMetadata.page)
        ).scalars().all()
        for cm in c_metas[:10]:
            table_info = f" | Tab: {cm.table_id}" if cm.table_id else ""
            asset_info = f" | Asset: {cm.asset_id[:8]}..." if cm.asset_id else ""
            print(
                f"   • Chunk {cm.chunk_id[:8]}... | Pg {cm.page} | "
                f"Type: {cm.chunk_type:<13} | Src: {cm.content_source:<7} | "
                f"Tokens: {cm.token_count:<4} | Q: {cm.quality_score or 0.0:.2f}"
                f"{table_info}{asset_info}"
            )

    _separator("🎉  MINIO INGESTION COMPLETE")
    print("  All documents from MinIO have been ingested, parsed, chunked, and embedded.")
    print("  The pipeline is ready for retrieval-augmented generation.\n")


if __name__ == "__main__":
    main()
