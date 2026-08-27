"""Example script demonstrating the RAG pipeline stages:

1) Differential document discovery (new / changed / deleted / unchanged)
2) Incremental ingestion (fetch + version tracking + raw storage)
3) Parse & Chunk (reads from raw store — no live connector needed)
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import func, select

from rag.connectors.local_file import LocalFileConnector
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, DocumentVersion
from rag.core.migrate import run_migrations
from rag.ingestion.pipeline import IngestionPipeline
from rag.parsing import ChunkingConfig
from rag.storage.local_storage import LocalRawStorageBackend


def print_discovery_diff(report):
    print("\n" + "=" * 60)
    print(" 📊 DOCUMENT DISCOVERY DIFFERENTIAL REPORT")
    print("=" * 60)
    print(f"Total Discovered in Source: {report.total_discovered}")
    print(f"Total Tracked in DB:        {report.total_existing}")
    print("-" * 60)

    # 1. What exists / Unchanged
    print(f"\n1️⃣  EXISTS / UNCHANGED ({len(report.unchanged_items)} files):")
    if not report.unchanged_items:
        print("   (None)")
    for item in report.unchanged_items:
        print(f"   ✓ {item.descriptor.filename} (v{item.existing_version})")
        print(f"     Reason: {item.reason}")

    # 2. What is New
    print(f"\n2️⃣  WHAT IS NEW ({len(report.new_items)} files):")
    if not report.new_items:
        print("   (None)")
    for item in report.new_items:
        print(f"   + {item.descriptor.filename} ({item.descriptor.mime_type})")
        print(f"     Reason: {item.reason}")

    # 3. What is Changed
    print(f"\n3️⃣  WHAT IS CHANGED ({len(report.changed_items)} files):")
    if not report.changed_items:
        print("   (None)")
    for item in report.changed_items:
        print(f"   Δ {item.descriptor.filename} (Current: v{item.existing_version} -> Target: v{item.existing_version + 1})")
        print(f"     Reason: {item.reason}")

    # 4. What was Deleted
    print(f"\n4️⃣  WHAT WAS DELETED ({len(report.deleted_items)} files):")
    if not report.deleted_items:
        print("   (None)")
    for item in report.deleted_items:
        print(f"   ✗ {item.filename} (ID: {item.document_id}, Last Known: v{item.last_known_version})")
        print(f"     URI: {item.source_uri}")

    print("=" * 60)


def main():
    load_dotenv()
    data_dir = Path(__file__).resolve().parent / "data"
    print(f"📁 Target data folder: {data_dir}")

    # 1. Initialize connector, raw storage backend, and database
    connector = LocalFileConnector(base_dir=data_dir, recursive=True)
    connector.validate_connection()

    db_url = os.getenv("DATABASE_URL", "sqlite:///rag.db")
    raw_storage_dir = os.getenv("RAW_STORAGE_DIR", "./raw_store")
    raw_store = LocalRawStorageBackend(base_dir=raw_storage_dir)

    print(f"🗄️  Database Target:    {db_url.split('@')[-1] if '@' in db_url else db_url}")
    print(f"💾  Raw Storage Dir:    {Path(raw_storage_dir).resolve()}")

    db_manager = DatabaseManager(db_url=db_url)
    # Ensure all tables exist (documents, chunks, embeddings, errors ...)
    run_migrations(db_manager)

    pipeline = IngestionPipeline(db_manager=db_manager, raw_storage=raw_store)
    pipeline.register_connector("local", connector)

    # 2. Step 1: Pre-Ingestion Differential Discovery
    print("\n🔍 Running differential discovery before sync...")
    diff_report = pipeline.discover_changes("local")
    print_discovery_diff(diff_report)

    # 3. Step 2: Execute Incremental Ingestion
    print("\n⚡ Executing Incremental Ingestion...")
    summary = pipeline.ingest_from_connector("local", handle_deletions=True)

    print("\n📈 Ingestion Execution Summary:")
    print(f"   • Discovered in Source: {summary.discovered_count}")
    print(f"   • New Ingested:         {summary.new_count}")
    print(f"   • Changed / Re-indexed: {summary.changed_count}")
    print(f"   • Unchanged (Skipped):  {summary.unchanged_count}")
    print(f"   • Soft-Deleted:         {summary.deleted_count}")
    print(f"   • Payload Fetches:      {summary.fetched_count}")
    print(f"   • Failed:               {summary.failed_count}")

    # 4. Step 3: Query Database State
    print("\n📋 Current Database Document Records:")
    with db_manager.session() as session:
        from rag.core.db_models import PipelineError
        docs = session.execute(select(Document).order_by(Document.created_at)).scalars().all()
        for doc in docs:
            status_icon = {"FETCHED": "🟢", "DELETED": "🔴", "FAILED": "⛔", "PARSED": "🔵"}.get(doc.status, "🟡")
            print(f"\n{status_icon} Document: {doc.filename}")
            print(f"   • ID:         {doc.document_id}")
            print(f"   • Status:     {doc.status}")
            print(f"   • Version:    v{doc.version}")
            print(f"   • Hash:       {doc.content_hash[:16]}..." if doc.content_hash else "   • Hash:       (none)")
            if doc.deleted_at:
                print(f"   • Deleted At: {doc.deleted_at}")
            if doc.error:
                print(f"   ⚠️  Error:     {doc.error}")

            versions = session.execute(
                select(DocumentVersion).where(DocumentVersion.document_id == doc.document_id)
            ).scalars().all()
            for ver in versions:
                raw_loc = ver.raw_location or "(not stored)"
                # Show whether it's a raw-store key or original source URI
                is_stored = raw_store.exists(raw_loc) if ver.raw_location else False
                stored_icon = "💾" if is_stored else "🔗"
                size_str = ""
                if is_stored:
                    size_bytes = raw_store.storage_size_bytes(raw_loc)
                    size_str = f" ({size_bytes:,} bytes)" if size_bytes else ""
                print(f"     ↳ Version v{ver.version} (hash: {ver.content_hash[:16]}...)")
                print(f"       {stored_icon} raw_location: {raw_loc}{size_str}")

        # 5. Step 4: Query Errors Table
        all_errors = session.execute(
            select(PipelineError).order_by(PipelineError.created_at.desc())
        ).scalars().all()
        if all_errors:
            print(f"\n🚨 Pipeline Errors Table ({len(all_errors)} record(s)):")
            for err in all_errors:
                print(f"\n   Error ID:   {err.error_id}")
                print(f"   Doc ID:     {err.document_id or '(none)'}")
                print(f"   Source URI: {err.source_uri}")
                print(f"   Stage:      {err.stage}")
                print(f"   Type:       {err.error_type}")
                print(f"   Message:    {err.error_message}")
                print(f"   Occurred:   {err.created_at}")
        else:
            print("\n✅ No pipeline errors recorded.")

    # ── Step 4: Parse & Chunk ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" 📄 STEP 4: PARSE & CHUNK")
    print("=" * 60)
    config = ChunkingConfig(chunk_size=512, chunk_overlap=64)
    chunk_results = pipeline.parse_and_chunk("local", chunking_config=config)

    total_chunks = sum(chunk_results.values())
    print(f"\n✅ Parsing complete: {len(chunk_results)} document(s) processed, {total_chunks} total chunks.")

    # ── Step 5: Chunk Summary per Document ───────────────────────────────────
    print("\n📊 Chunk Summary per Document:")
    with db_manager.session() as session:
        docs = session.execute(select(Document).order_by(Document.created_at)).scalars().all()
        for doc in docs:
            chunk_count = session.execute(
                select(func.count()).select_from(Chunk).where(
                    Chunk.document_id == doc.document_id,
                    Chunk.version == doc.version,
                )
            ).scalar_one()

            status_icon = {
                "FETCHED": "🟢", "PARSED": "🔵", "CHUNKED": "🟣",
                "DELETED": "🔴", "FAILED": "⛔",
            }.get(doc.status, "🟡")

            print(f"\n{status_icon} {doc.filename}")
            print(f"   • Status:  {doc.status}")
            print(f"   • Version: v{doc.version}")
            print(f"   • Chunks:  {chunk_count}")

            if chunk_count > 0:
                # Show first 2 chunks as preview
                preview_chunks = session.execute(
                    select(Chunk)
                    .where(Chunk.document_id == doc.document_id, Chunk.version == doc.version)
                    .order_by(Chunk.chunk_index)
                    .limit(2)
                ).scalars().all()
                for ch in preview_chunks:
                    preview_text = ch.text[:120].replace("\n", " ")
                    tokens = ch.chunk_metadata.get("token_count", "?")
                    print(f"     ↳ [Chunk {ch.chunk_index}] pg{ch.page} | {tokens} tok | {preview_text!r}...")


if __name__ == "__main__":
    main()
