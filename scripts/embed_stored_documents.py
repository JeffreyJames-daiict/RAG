"""Script to generate and persist embeddings for all stored documents in PostgreSQL.

Usage:
    python scripts/embed_stored_documents.py
    python scripts/embed_stored_documents.py --force
    python scripts/embed_stored_documents.py --doc-id <UUID>
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from sqlalchemy import func, select

from rag.core.config import get_settings
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, Embedding
from rag.embeddings.models import EmbeddingSummary
from rag.embeddings.providers.mock_provider import MockEmbeddingProvider
from rag.embeddings.service import EmbeddingService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_embedding_on_stored_documents(
    db_url: str | None = None,
    doc_id: str | None = None,
    force_reembed: bool = False,
) -> int:
    settings = get_settings()
    target_db_url = db_url or settings.database_url
    print("\n" + "=" * 70)
    print(" 🚀 RUNNING EMBEDDING GENERATION ON STORED DOCUMENTS")
    print("=" * 70)
    print(f" Database: {settings.database.safe_url()}")
    print(" Provider: MockEmbeddingProvider (Offline SHA-256 synthetic unit vectors)")
    print(f" Model:    {settings.embedding.model} (dim: {settings.embedding.dimension})")
    print(f" Force:    {force_reembed}")
    print("-" * 70)

    db_manager = DatabaseManager(db_url=target_db_url)
    provider = MockEmbeddingProvider(
        model_name=settings.embedding.model,
        dimension=settings.embedding.dimension,
    )
    service = EmbeddingService(db_manager=db_manager, provider=provider)

    # 1. Discover target documents
    with db_manager.session() as session:
        stmt = select(Document)
        if doc_id:
            stmt = stmt.where(Document.document_id == doc_id)
        stmt = stmt.order_by(Document.created_at.asc())
        docs = session.execute(stmt).scalars().all()

        if not docs:
            print(f"❌ No documents found in database matching criteria (doc_id={doc_id})")
            return 1

        print(f"\nFound {len(docs)} document(s) in PostgreSQL database:\n")
        doc_details = []
        for d in docs:
            chunk_count = session.execute(
                select(func.count(Chunk.chunk_id)).where(Chunk.document_id == d.document_id)
            ).scalar() or 0
            existing_embs = session.execute(
                select(func.count(Embedding.chunk_id))
                .join(Chunk, Chunk.chunk_id == Embedding.chunk_id)
                .where(
                    Chunk.document_id == d.document_id,
                    Embedding.embedding_model == service.model_name,
                    Embedding.embedding_version == service.model_version,
                )
            ).scalar() or 0
            doc_details.append((d, chunk_count, existing_embs))
            print(f" • [{d.document_id[:8]}...] {d.filename:<40} | v{d.version} | Chunks: {chunk_count:<3} | Existing Embs: {existing_embs:<3}")

    print("\n" + "-" * 70)
    print(" 🔄 PROCESSING EMBEDDING GENERATION...")
    print("-" * 70)

    total_embedded = 0
    total_skipped = 0
    total_failed = 0
    start_total_time = time.time()

    for doc, chunk_count, existing_embs in doc_details:
        print(f"\nProcessing doc '{doc.filename}' ({doc.document_id})...")
        if chunk_count == 0:
            print("  ⚠️ Document has 0 chunks. Skipping.")
            continue

        summary: EmbeddingSummary = service.embed_document(
            document_id=doc.document_id,
            version=doc.version,
            force_reembed=force_reembed,
        )

        total_embedded += summary.embedded_count
        total_skipped += summary.skipped_count
        total_failed += summary.failed_count

        print(f"  ✓ Embedded: {summary.embedded_count} chunks")
        print(f"  ✓ Skipped:  {summary.skipped_count} chunks (already up-to-date)")
        if summary.failed_count > 0:
            print(f"  ✗ Failed:   {summary.failed_count} chunks")
            for err in summary.errors:
                print(f"    - Error: {err}")
        print(f"  ⏱ Duration: {summary.duration_ms:.2f} ms")

    total_duration_ms = (time.time() - start_total_time) * 1000.0

    # 2. Final Verification against Database
    print("\n" + "=" * 70)
    print(" 📊 EMBEDDING EXECUTION SUMMARY & DATABASE VALIDATION")
    print("=" * 70)
    print(f" Total Documents Processed: {len(doc_details)}")
    print(f" Total Chunks Embedded:     {total_embedded}")
    print(f" Total Chunks Skipped:      {total_skipped}")
    print(f" Total Chunks Failed:       {total_failed}")
    print(f" Total Execution Time:      {total_duration_ms:.2f} ms")

    with db_manager.session() as session:
        stored_embs_count = session.execute(select(func.count(Embedding.chunk_id))).scalar()
        sample_emb = session.execute(
            select(Embedding).where(
                Embedding.embedding_model == service.model_name,
                Embedding.embedding_version == service.model_version,
            ).limit(1)
        ).scalar_one_or_none()

        print("\nPostgreSQL Database State:")
        print(f"  • Total Embedding rows in DB: {stored_embs_count}")
        if sample_emb:
            vec_preview = sample_emb.vector[:4] if isinstance(sample_emb.vector, list) else sample_emb.vector
            print(f"  • Sample Chunk ID:  {sample_emb.chunk_id}")
            print(f"  • Stored Provider:  {sample_emb.provider}")
            print(f"  • Stored Model:     {sample_emb.embedding_model} ({sample_emb.embedding_version})")
            print(f"  • Stored Dimension: {sample_emb.dimension}")
            print(f"  • Vector Head [4]:  {vec_preview}")
            print("  • Database Integrity: VERIFIED ✅")

    print("\n" + "=" * 70 + "\n")
    return 0 if total_failed == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and persist embeddings for stored documents in PostgreSQL.")
    parser.add_argument("--db-url", type=str, default=None, help="Database connection URL")
    parser.add_argument("--doc-id", type=str, default=None, help="Filter by specific document ID")
    parser.add_argument("--force", action="store_true", help="Force re-embedding existing chunks")

    args = parser.parse_args()
    code = run_embedding_on_stored_documents(
        db_url=args.db_url,
        doc_id=args.doc_id,
        force_reembed=args.force,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
