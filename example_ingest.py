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
from rag.core.config import get_settings
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, DocumentVersion
from rag.core.migrate import run_migrations
from rag.ingestion.pipeline import IngestionPipeline
from rag.parsing import ChunkingConfig
from rag.storage import get_storage_backend


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
    # 1. Load centralized configuration from config.json
    settings = get_settings()
    print(settings.summary())

    data_dir = Path(__file__).resolve().parent / "data"
    print(f"📁 Target data folder: {data_dir}")

    # 2. Initialize connector, raw storage backend, and database
    connector = LocalFileConnector(base_dir=data_dir, recursive=True)
    connector.validate_connection()

    # Initialize database manager first (raw storage backend needs it)
    db_manager = DatabaseManager(
        db_url=settings.database.url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        echo=settings.database.echo,
    )

    raw_store = get_storage_backend(settings=settings, db_manager=db_manager)

    # Ensure all tables exist if configured
    if settings.pipeline.auto_migrate:
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
                if is_stored and hasattr(raw_store, "get_storage_size_bytes"):
                    size_bytes = raw_store.get_storage_size_bytes(raw_loc)
                    size_str = f" ({size_bytes:,} bytes)" if size_bytes else ""
                print(f"     ↳ Version v{ver.version} (hash: {ver.content_hash[:16]}...)")
                print(f"       {stored_icon} raw_location: {raw_loc}{size_str}")
                meta_snap = ver.metadata_snapshot or {}
                if meta_snap:
                    avg_q = meta_snap.get("avg_quality_score")
                    ocr_diag = meta_snap.get("ocr_diagnostics", {})
                    tab_diag = meta_snap.get("table_diagnostics", {})
                    print(f"       📊 Stored DB Metrics: Avg Quality={avg_q} | Tables={tab_diag.get('tables_processed', 0)} | OCR Processed={ocr_diag.get('ocr_pages_processed', 0)}")

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
    config = ChunkingConfig(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        chunking_version=settings.chunking.chunking_version,
        encoding_model=settings.chunking.encoding_model,
    )
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
                all_doc_chunks = session.execute(
                    select(Chunk)
                    .where(Chunk.document_id == doc.document_id, Chunk.version == doc.version)
                    .order_by(Chunk.chunk_index)
                ).scalars().all()

                text_count = sum(1 for c in all_doc_chunks if c.chunk_metadata.get("chunk_type") == "text")
                table_count = sum(1 for c in all_doc_chunks if c.chunk_metadata.get("chunk_type") == "table")
                summary_count = sum(1 for c in all_doc_chunks if c.chunk_metadata.get("chunk_type") == "table_summary")

                print(f"   • Breakdown: {text_count} text, {table_count} table, {summary_count} table_summary")

                for ch in all_doc_chunks[:4]:
                    ch_type = ch.chunk_metadata.get("chunk_type", "text")
                    src = ch.chunk_metadata.get("content_source", "native")
                    q_score = ch.chunk_metadata.get("quality_score")
                    q_str = f" [q={q_score:.2f}]" if q_score is not None else ""
                    type_badge = {"table": "📊 [TABLE]", "table_summary": "📋 [TABLE_SUMMARY]", "text": "📝 [TEXT]"}.get(ch_type, "📄 [CHUNK]")
                    preview_text = ch.text[:90].replace("\n", " ")
                    tokens = ch.chunk_metadata.get("token_count", "?")
                    extra = ""
                    if ch_type == "table":
                        r_start = ch.chunk_metadata.get("row_start", "?")
                        r_end = ch.chunk_metadata.get("row_end", "?")
                        extra = f" | Rows {r_start}-{r_end}"
                    print(f"     ↳ {type_badge} #{ch.chunk_index} pg{ch.page} ({tokens} tok{extra} | {src}{q_str}) | {preview_text!r}...")

    # ── Step 5b: Normalized Relational Metadata Tables ────────────────────────
    print("\n" + "=" * 60)
    print(" 🗄️  STEP 5B: RELATIONAL METADATA TABLES (Normalized DB View)")
    print("=" * 60)
    from rag.core.db_models import ChunkMetadata, DocumentVersionMetadata

    with db_manager.session() as session:
        print("\n📋 Table: document_version_metadata")
        v_metas = session.execute(select(DocumentVersionMetadata)).scalars().all()
        for vm in v_metas:
            print(f"   • Doc {vm.document_id[:8]}... v{vm.version} | Lang: {vm.primary_language} | Pages: {vm.num_pages} | Chunks: {vm.total_chunks} ({vm.text_chunks} txt, {vm.table_chunks} tab) | Avg Quality: {vm.avg_quality_score or 0.0:.2f} | OCR Conf: {vm.avg_ocr_confidence or 0.0:.2f} | Title: {vm.pdf_title or '(none)'}")

        print("\n📋 Table: chunk_metadata (Sample preview)")
        c_metas = session.execute(select(ChunkMetadata).order_by(ChunkMetadata.document_id, ChunkMetadata.page, ChunkMetadata.row_start)).scalars().all()
        for cm in c_metas[:8]:
            table_info = f" | Tab: {cm.table_id} (Rows {cm.row_start}-{cm.row_end})" if cm.table_id else ""
            print(f"   • Chunk {cm.chunk_id[:8]}... | Pg {cm.page} | Type: {cm.chunk_type:<13} | Lang: {cm.language} | Src: {cm.content_source:<7} | Tokens: {cm.token_count:<3} | Quality: {cm.quality_score or 0.0:.2f}{table_info}")

    # ── Step 6: Table-Aware Retrieval Demo ────────────────────────────────────
    print("\n" + "=" * 60)
    print(" 🔍 STEP 6: TABLE-AWARE RETRIEVAL DEMO")
    print("=" * 60)
    from rag.retrieval import RetrievalFilter, TableAwareRetriever
    retriever = TableAwareRetriever(db_manager)

    test_queries = [
        ("Calendar types and formats", RetrievalFilter(chunk_type="table")),
        ("Photo Mug options", RetrievalFilter()),
    ]

    for q_text, q_filter in test_queries:
        filter_label = f" (filter: chunk_type={q_filter.chunk_type})" if q_filter.chunk_type else " (all chunks)"
        print(f"\n🔎 Query: '{q_text}'{filter_label}")
        results = retriever.search(q_text, retrieval_filter=q_filter, top_k=2)
        if not results:
            print("   (No matching chunks found)")
        for res in results:
            type_tag = f"[{res.chunk_type.upper()}]"
            print(f"   ✓ Score {res.score:.2f} | {type_tag} Doc: {res.document_id[:8]}... Chunk #{res.chunk_index} (Page {res.page})")
            print(f"     Text snippet: {res.text[:130].replace(chr(10), ' ')}...")
            if res.table_id:
                siblings = retriever.expand_parent_table(res.table_id, res.document_id)
                print(f"     ↳ Parent Table Expansion: Table '{res.table_id}' has {len(siblings)} related chunk(s)")


if __name__ == "__main__":
    main()

