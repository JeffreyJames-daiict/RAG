"""Database migration helper — ensures all ORM tables, schemas, and columns exist on PostgreSQL / SQLite."""

from __future__ import annotations
import logging
from sqlalchemy import event, text

from rag.core.database import DatabaseManager
from rag.core.db_models import Base

logger = logging.getLogger(__name__)


def _prepare_schemas(engine) -> None:
    """Create the 'raw' schema/database and ensure column migrations depending on the dialect.

    - **PostgreSQL**: ``CREATE SCHEMA IF NOT EXISTS raw;`` + column alterations.
    - **SQLite**: ``ATTACH DATABASE ':memory:' AS raw;``
    """
    dialect = engine.dialect.name

    if dialect == "postgresql":
        with engine.connect() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw;"))

            # Ensure pgvector extension is active
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

            # Column migrations for existing tables
            conn.execute(text("""
                DO $$
                BEGIN
                    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'document_versions') THEN
                        ALTER TABLE public.document_versions ADD COLUMN IF NOT EXISTS metadata JSON NOT NULL DEFAULT '{}'::json;
                    END IF;
                    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'chunk_metadata') THEN
                        ALTER TABLE public.chunk_metadata ADD COLUMN IF NOT EXISTS asset_id VARCHAR(128);
                    END IF;
                    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'embeddings') THEN
                        ALTER TABLE public.embeddings ADD COLUMN IF NOT EXISTS dimension INTEGER;
                        ALTER TABLE public.embeddings ADD COLUMN IF NOT EXISTS provider VARCHAR(64);
                    END IF;
                    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'document_assets') THEN
                        ALTER TABLE public.document_assets ADD COLUMN IF NOT EXISTS perceptual_hash VARCHAR(64);
                        ALTER TABLE public.document_assets ADD COLUMN IF NOT EXISTS relevance_score DOUBLE PRECISION;
                        ALTER TABLE public.document_assets ADD COLUMN IF NOT EXISTS relevance_reason TEXT;
                        ALTER TABLE public.document_assets ADD COLUMN IF NOT EXISTS indexing_policy VARCHAR(32) NOT NULL DEFAULT 'STORE_ONLY';
                        ALTER TABLE public.document_assets ADD COLUMN IF NOT EXISTS processing_error TEXT;
                        ALTER TABLE public.document_assets ADD COLUMN IF NOT EXISTS bbox_x0 DOUBLE PRECISION;
                        ALTER TABLE public.document_assets ADD COLUMN IF NOT EXISTS bbox_y0 DOUBLE PRECISION;
                        ALTER TABLE public.document_assets ADD COLUMN IF NOT EXISTS bbox_x1 DOUBLE PRECISION;
                        ALTER TABLE public.document_assets ADD COLUMN IF NOT EXISTS bbox_y1 DOUBLE PRECISION;
                    END IF;
                END $$;
            """))

            # pgvector: Add native vector column and HNSW index
            conn.execute(text("""
                DO $$
                BEGIN
                    -- Add embedding_vector column (native pgvector type)
                    IF EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = 'embeddings'
                    ) AND NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = 'embeddings'
                          AND column_name = 'embedding_vector'
                    ) THEN
                        ALTER TABLE public.embeddings ADD COLUMN embedding_vector vector(1536);
                    END IF;
                END $$;
            """))

            # Backfill: populate embedding_vector from JSON vector column for existing rows
            conn.execute(text("""
                UPDATE public.embeddings
                SET embedding_vector = (vector::text)::vector
                WHERE embedding_vector IS NULL
                  AND vector IS NOT NULL
                  AND json_array_length(vector) > 0;
            """))

            # Create HNSW cosine index (idempotent via IF NOT EXISTS)
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_embeddings_hnsw_cosine
                ON public.embeddings
                USING hnsw (embedding_vector vector_cosine_ops)
                WITH (m = 16, ef_construction = 200);
            """))

            conn.commit()
        logger.info("Ensured PostgreSQL schema 'raw', column migrations, pgvector backfill, and HNSW index exist.")

    elif dialect == "sqlite":
        # Ensure raw database is attached on existing raw connection
        raw_conn = engine.raw_connection()
        try:
            raw_conn.execute("ATTACH DATABASE ':memory:' AS raw")
        except Exception:
            pass  # Already attached
        finally:
            raw_conn.close()

        # Check SQLite table columns
        with engine.connect() as conn:
            try:
                res = conn.execute(text("PRAGMA table_info(chunk_metadata)")).fetchall()
                existing_cols = {row[1] for row in res}
                if existing_cols and "asset_id" not in existing_cols:
                    conn.execute(text("ALTER TABLE chunk_metadata ADD COLUMN asset_id VARCHAR(128)"))
                    conn.commit()
            except Exception:
                pass

            try:
                res = conn.execute(text("PRAGMA table_info(document_assets)")).fetchall()
                existing_cols = {row[1] for row in res}
                if existing_cols:
                    if "perceptual_hash" not in existing_cols:
                        conn.execute(text("ALTER TABLE document_assets ADD COLUMN perceptual_hash VARCHAR(64)"))
                    if "relevance_score" not in existing_cols:
                        conn.execute(text("ALTER TABLE document_assets ADD COLUMN relevance_score FLOAT"))
                    if "relevance_reason" not in existing_cols:
                        conn.execute(text("ALTER TABLE document_assets ADD COLUMN relevance_reason TEXT"))
                    if "indexing_policy" not in existing_cols:
                        conn.execute(text("ALTER TABLE document_assets ADD COLUMN indexing_policy VARCHAR(32) DEFAULT 'STORE_ONLY'"))
                    if "processing_error" not in existing_cols:
                        conn.execute(text("ALTER TABLE document_assets ADD COLUMN processing_error TEXT"))
                    if "bbox_x0" not in existing_cols:
                        conn.execute(text("ALTER TABLE document_assets ADD COLUMN bbox_x0 FLOAT"))
                    if "bbox_y0" not in existing_cols:
                        conn.execute(text("ALTER TABLE document_assets ADD COLUMN bbox_y0 FLOAT"))
                    if "bbox_x1" not in existing_cols:
                        conn.execute(text("ALTER TABLE document_assets ADD COLUMN bbox_x1 FLOAT"))
                    if "bbox_y1" not in existing_cols:
                        conn.execute(text("ALTER TABLE document_assets ADD COLUMN bbox_y1 FLOAT"))
                    conn.commit()
            except Exception:
                pass


def _create_tabular_views(engine) -> None:
    """Create flattened relational SQL views for clean tabular database inspection."""
    dialect = engine.dialect.name

    with engine.connect() as conn:
        if dialect == "postgresql":
            conn.execute(text("""
                CREATE OR REPLACE VIEW v_chunks_tabular AS
                SELECT 
                    c.chunk_id,
                    c.document_id,
                    c.version,
                    c.chunk_index,
                    cm.chunk_type,
                    c.text,
                    c.section,
                    c.page,
                    cm.page_start,
                    cm.page_end,
                    cm.token_count,
                    cm.language,
                    cm.language_confidence,
                    cm.content_source,
                    cm.quality_score,
                    cm.ocr_applied,
                    cm.ocr_confidence,
                    cm.ocr_provider,
                    cm.table_id,
                    cm.parent_table_id,
                    cm.asset_id,
                    cm.row_start,
                    cm.row_end,
                    cm.total_rows,
                    cm.total_columns,
                    cm.is_continuation,
                    cm.validation_status,
                    c.chunking_version,
                    c.metadata AS raw_json_metadata
                FROM chunks c
                LEFT JOIN chunk_metadata cm ON c.chunk_id = cm.chunk_id;
            """))

            conn.execute(text("""
                CREATE OR REPLACE VIEW v_documents_tabular AS
                SELECT 
                    d.document_id,
                    d.tenant_id,
                    d.filename,
                    d.source,
                    d.source_uri,
                    d.status,
                    d.version,
                    dvm.num_pages,
                    dvm.total_chunks,
                    dvm.text_chunks,
                    dvm.table_chunks,
                    dvm.avg_quality_score,
                    dvm.avg_ocr_confidence,
                    dvm.tables_detected,
                    dvm.tables_processed,
                    dvm.pages_native,
                    dvm.pages_scanned,
                    dvm.pages_mixed,
                    dvm.ocr_pages_processed,
                    dvm.primary_language,
                    d.content_hash,
                    dv.raw_location,
                    d.created_at,
                    d.updated_at,
                    d.error
                FROM documents d
                LEFT JOIN document_versions dv ON d.document_id = dv.document_id AND d.version = dv.version
                LEFT JOIN document_version_metadata dvm ON d.document_id = dvm.document_id AND d.version = dvm.version;
            """))

            conn.execute(text("""
                CREATE OR REPLACE VIEW v_assets_tabular AS
                SELECT 
                    da.asset_id,
                    da.document_id,
                    da.version,
                    da.page_number,
                    da.asset_type,
                    da.asset_subtype,
                    da.storage_uri,
                    da.content_hash,
                    da.perceptual_hash,
                    COALESCE(da.bbox_x0, (da.bbox->>'x0')::float) AS bbox_x0,
                    COALESCE(da.bbox_y0, (da.bbox->>'y0')::float) AS bbox_y0,
                    COALESCE(da.bbox_x1, (da.bbox->>'x1')::float) AS bbox_x1,
                    COALESCE(da.bbox_y1, (da.bbox->>'y1')::float) AS bbox_y1,
                    da.width,
                    da.height,
                    da.mime_type,
                    da.caption,
                    da.ocr_text,
                    da.ocr_confidence,
                    da.semantic_description,
                    da.relevance_score,
                    da.relevance_reason,
                    da.indexing_policy,
                    da.processing_status,
                    da.processing_error,
                    da.created_at
                FROM document_assets da;
            """))

            conn.execute(text("""
                CREATE OR REPLACE VIEW v_page_scores_tabular AS
                SELECT 
                    dps.document_id,
                    d.filename,
                    dps.version,
                    dps.page_number,
                    dps.content_source,
                    dps.detected_language,
                    dps.language_confidence,
                    dps.quality_score,
                    dps.ocr_applied,
                    dps.ocr_confidence,
                    dps.ocr_provider,
                    dps.preprocessing_profile,
                    dps.ocr_attempt,
                    dps.created_at
                FROM document_page_scores dps
                JOIN documents d ON dps.document_id = d.document_id;
            """))

            conn.execute(text("""
                CREATE OR REPLACE VIEW v_asset_entities_tabular AS
                SELECT 
                    ae.asset_id,
                    da.document_id,
                    da.page_number,
                    da.asset_type,
                    ae.entity_index,
                    ae.entity_name,
                    ae.created_at
                FROM asset_entities ae
                JOIN document_assets da ON ae.asset_id = da.asset_id;
            """))

            conn.execute(text("""
                CREATE OR REPLACE VIEW v_asset_relationships_tabular AS
                SELECT 
                    ar.relationship_id,
                    ar.asset_id,
                    da.document_id,
                    da.page_number,
                    da.asset_type,
                    ar.relationship_index,
                    ar.source_entity,
                    ar.relation_type,
                    ar.target_entity,
                    ar.created_at
                FROM asset_relationships ar
                JOIN document_assets da ON ar.asset_id = da.asset_id;
            """))
            conn.commit()

        elif dialect == "sqlite":
            views = [
                ("v_chunks_tabular", """
                    CREATE VIEW IF NOT EXISTS v_chunks_tabular AS
                    SELECT 
                        c.chunk_id, c.document_id, c.version, c.chunk_index,
                        cm.chunk_type, c.text, c.section, c.page, cm.page_start, cm.page_end,
                        cm.token_count, cm.language, cm.language_confidence, cm.content_source,
                        cm.quality_score, cm.ocr_applied, cm.ocr_confidence, cm.ocr_provider,
                        cm.table_id, cm.parent_table_id, cm.asset_id, cm.row_start, cm.row_end,
                        cm.total_rows, cm.total_columns, cm.is_continuation, cm.validation_status,
                        c.chunking_version
                    FROM chunks c
                    LEFT JOIN chunk_metadata cm ON c.chunk_id = cm.chunk_id;
                """),
                ("v_documents_tabular", """
                    CREATE VIEW IF NOT EXISTS v_documents_tabular AS
                    SELECT 
                        d.document_id, d.tenant_id, d.filename, d.source, d.source_uri,
                        d.status, d.version, dvm.num_pages, dvm.total_chunks, dvm.text_chunks,
                        dvm.table_chunks, dvm.avg_quality_score, dvm.avg_ocr_confidence,
                        dvm.tables_detected, dvm.tables_processed, dvm.pages_native,
                        dvm.pages_scanned, dvm.pages_mixed, dvm.ocr_pages_processed,
                        dvm.primary_language, d.content_hash, dv.raw_location, d.created_at, d.updated_at
                    FROM documents d
                    LEFT JOIN document_versions dv ON d.document_id = dv.document_id AND d.version = dv.version
                    LEFT JOIN document_version_metadata dvm ON d.document_id = dvm.document_id AND d.version = dvm.version;
                """),
                ("v_assets_tabular", """
                    CREATE VIEW IF NOT EXISTS v_assets_tabular AS
                    SELECT 
                        da.asset_id, da.document_id, da.version, da.page_number, da.asset_type,
                        da.asset_subtype, da.storage_uri, da.content_hash, da.perceptual_hash,
                        da.bbox_x0, da.bbox_y0, da.bbox_x1, da.bbox_y1, da.width, da.height,
                        da.mime_type, da.caption, da.ocr_text, da.ocr_confidence, da.semantic_description,
                        da.relevance_score, da.relevance_reason, da.indexing_policy, da.processing_status,
                        da.processing_error, da.created_at
                    FROM document_assets da;
                """),
                ("v_page_scores_tabular", """
                    CREATE VIEW IF NOT EXISTS v_page_scores_tabular AS
                    SELECT 
                        dps.document_id, d.filename, dps.version, dps.page_number,
                        dps.content_source, dps.detected_language, dps.language_confidence,
                        dps.quality_score, dps.ocr_applied, dps.ocr_confidence, dps.ocr_provider,
                        dps.preprocessing_profile, dps.ocr_attempt, dps.created_at
                    FROM document_page_scores dps
                    JOIN documents d ON dps.document_id = d.document_id;
                """),
                ("v_asset_entities_tabular", """
                    CREATE VIEW IF NOT EXISTS v_asset_entities_tabular AS
                    SELECT 
                        ae.asset_id, da.document_id, da.page_number, da.asset_type,
                        ae.entity_index, ae.entity_name, ae.created_at
                    FROM asset_entities ae
                    JOIN document_assets da ON ae.asset_id = da.asset_id;
                """),
                ("v_asset_relationships_tabular", """
                    CREATE VIEW IF NOT EXISTS v_asset_relationships_tabular AS
                    SELECT 
                        ar.relationship_id, ar.asset_id, da.document_id, da.page_number,
                        da.asset_type, ar.relationship_index, ar.source_entity, ar.relation_type,
                        ar.target_entity, ar.created_at
                    FROM asset_relationships ar
                    JOIN document_assets da ON ar.asset_id = da.asset_id;
                """),
            ]
            for view_name, sql in views:
                try:
                    conn.execute(text(sql))
                except Exception:
                    pass
            conn.commit()


def backfill_tabular_tables(engine) -> None:
    """Backfill relational tables and bbox columns from existing JSON metadata."""
    from sqlalchemy.orm import Session
    from rag.core.db_models import (
        AssetEntity,
        AssetRelationship,
        DocumentAsset,
        DocumentPageScore,
        DocumentVersionMetadata,
    )

    with Session(engine) as session:
        # 1. Backfill bbox_x0, bbox_y0, bbox_x1, bbox_y1 and asset_entities / asset_relationships
        assets = session.query(DocumentAsset).all()
        for a in assets:
            if a.bbox and isinstance(a.bbox, dict):
                a.bbox_x0 = float(a.bbox.get("x0", 0.0)) if a.bbox.get("x0") is not None else None
                a.bbox_y0 = float(a.bbox.get("y0", 0.0)) if a.bbox.get("y0") is not None else None
                a.bbox_x1 = float(a.bbox.get("x1", 0.0)) if a.bbox.get("x1") is not None else None
                a.bbox_y1 = float(a.bbox.get("y1", 0.0)) if a.bbox.get("y1") is not None else None

            # Entities
            if a.entities and isinstance(a.entities, list):
                for idx, ent in enumerate(a.entities):
                    if isinstance(ent, str) and ent.strip():
                        exists = session.query(AssetEntity).filter_by(
                            asset_id=a.asset_id,
                            entity_index=idx,
                        ).first()
                        if not exists:
                            session.add(AssetEntity(
                                asset_id=a.asset_id,
                                entity_index=idx,
                                entity_name=ent.strip(),
                            ))

            # Relationships
            if a.relationships and isinstance(a.relationships, list):
                for idx, rel in enumerate(a.relationships):
                    if isinstance(rel, dict):
                        src = rel.get("source", "")
                        r_type = rel.get("relationship", "")
                        tgt = rel.get("target", "")
                        if src and tgt:
                            exists = session.query(AssetRelationship).filter_by(
                                asset_id=a.asset_id,
                                relationship_index=idx,
                            ).first()
                            if not exists:
                                session.add(AssetRelationship(
                                    asset_id=a.asset_id,
                                    relationship_index=idx,
                                    source_entity=src,
                                    relation_type=r_type,
                                    target_entity=tgt,
                                    created_at=a.created_at,
                                ))

        # 2. Backfill document_page_scores from document_version_metadata.page_scores
        ver_metas = session.query(DocumentVersionMetadata).all()
        for vm in ver_metas:
            if vm.page_scores and isinstance(vm.page_scores, list):
                for ps in vm.page_scores:
                    if isinstance(ps, dict):
                        p_num = ps.get("page_number", 1)
                        exists = session.query(DocumentPageScore).filter_by(
                            document_id=vm.document_id,
                            version=vm.version,
                            page_number=p_num,
                        ).first()
                        if not exists:
                            session.add(DocumentPageScore(
                                document_id=vm.document_id,
                                version=vm.version,
                                page_number=p_num,
                                content_source=ps.get("content_source", "native"),
                                detected_language=ps.get("detected_language", vm.primary_language or "eng"),
                                language_confidence=ps.get("language_confidence", 1.0),
                                quality_score=ps.get("quality_score"),
                                ocr_applied=ps.get("ocr_applied", False),
                                ocr_confidence=ps.get("ocr_confidence"),
                                ocr_provider=ps.get("ocr_provider"),
                                preprocessing_profile=ps.get("preprocessing_profile"),
                                ocr_attempt=ps.get("ocr_attempt", 1),
                            ))

        session.commit()
    logger.info("Relational metadata backfill completed successfully.")


def ensure_raw_schema(engine) -> None:
    """Public helper to prepare schemas independently."""
    _prepare_schemas(engine)


def run_migrations(db_manager: DatabaseManager) -> None:
    """Create all schemas and tables defined in the ORM Base metadata.

    This is a lightweight, idempotent 'create if not exists' migration.

    Args:
        db_manager: Initialized DatabaseManager with a live engine.
    """
    logger.info("Running schema migrations (create_all)...")
    _prepare_schemas(db_manager.engine)
    Base.metadata.create_all(bind=db_manager.engine)
    _create_tabular_views(db_manager.engine)
    backfill_tabular_tables(db_manager.engine)
    table_names = list(Base.metadata.tables.keys())
    logger.info("Schema up-to-date. Tables: %s", table_names)
    print(f"✅ Schema migration complete. Tables ensured: {table_names}")
