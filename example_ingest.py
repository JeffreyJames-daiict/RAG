"""Example script demonstrating discovery, fetching, and ingestion of local PDF files."""

from pathlib import Path
from sqlalchemy import select

from rag.connectors.local_file import LocalFileConnector
from rag.core.database import DatabaseManager
from rag.core.db_models import Document, DocumentVersion
from rag.ingestion.pipeline import IngestionPipeline


def main():
    data_dir = Path(__file__).resolve().parent / "data"
    print(f"📁 Target data folder: {data_dir}")

    # 1. Initialize LocalFileConnector
    print("\n--- [Step 1] Initializing LocalFileConnector ---")
    connector = LocalFileConnector(base_dir=data_dir, recursive=True)
    connector.validate_connection()
    print("Connector connection validated successfully.")

    # 2. Discover local files (filtered to allowed extensions such as .pdf)
    print("\n--- [Step 2] Discovering Files ---")
    descriptors = list(connector.discover())
    for idx, doc in enumerate(descriptors, 1):
        print(f"[{idx}] {doc.filename}")
        print(f"    - MIME Type: {doc.mime_type}")
        print(f"    - Full Path: {doc.source_uri}")
        print(f"    - Size: {doc.custom_metadata.get('file_size_bytes'):,} bytes")
        print(f"    - Modified: {doc.last_modified_at}")

    # 3. Fetch raw content & compute SHA-256 hash
    print("\n--- [Step 3] Fetching Document Payloads ---")
    for doc in descriptors:
        payload = connector.fetch(doc)
        print(f"✓ Fetched '{doc.filename}': SHA-256 = {payload.content_hash}")

    # 4. Orchestrated Ingestion & DB Persistence
    print("\n--- [Step 4] Running Ingestion Pipeline ---")
    db_manager = DatabaseManager(db_url="sqlite:///rag.db")
    db_manager.create_tables()

    pipeline = IngestionPipeline(db_manager=db_manager)
    pipeline.register_connector("data_folder", connector)

    summary = pipeline.ingest_from_connector("data_folder")
    print(f"Ingestion Summary:")
    print(f"  • Discovered: {summary.discovered_count}")
    print(f"  • Fetched:    {summary.fetched_count}")
    print(f"  • Skipped:    {summary.skipped_count}")
    print(f"  • Failed:     {summary.failed_count}")

    # 5. Query stored documents from the SQLite database
    print("\n--- [Step 5] Querying Stored Documents in Database ---")
    with db_manager.session() as session:
        docs = session.execute(select(Document)).scalars().all()
        for doc in docs:
            print(f"\n📄 Document (ID: {doc.document_id})")
            print(f"   • Filename: {doc.filename}")
            print(f"   • Status:   {doc.status}")
            print(f"   • Version:  v{doc.version}")
            print(f"   • Hash:     {doc.content_hash}")

            versions = session.execute(
                select(DocumentVersion).where(DocumentVersion.document_id == doc.document_id)
            ).scalars().all()
            for ver in versions:
                print(f"     ↳ Version Record v{ver.version} (Location: {ver.raw_location})")


if __name__ == "__main__":
    main()
