# Enterprise RAG Pipeline — Project Summary

A modular, production-grade **Retrieval-Augmented Generation (RAG)** pipeline in Python featuring multi-source ingestion, intelligent OCR, table extraction, visual asset relevance classification, selective vision analysis, content normalization, hierarchical chunking, dense vector embeddings, and incremental database synchronization.

---

## 1. Architecture Overview

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                        1. SOURCE CONNECTOR LAYER                       │
 │      [Local Files]       [AWS S3 / MinIO]      [Web / HTTP]     [SQL]  │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ (RawDocumentPayload)
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                         2. RAW STORAGE ENGINE                          │
 │         Local Filesystem Storage  /  S3 / MinIO Object Storage         │
 │         (SHA-256 Hash Verification & Content Deduplication)            │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      3. INTELLIGENT OCR SUBSYSTEM                      │
 │   • Page Quality Analyzer (Digital vs Scanned vs Mixed Classification) │
 │   • Preprocessing (Deskewing, Binarization, Contrast, DPI Scaling)     │
 │   • Multi-Provider Engine (Tesseract, EasyOCR, DocTR, Cloud Stubs)     │
 │   • Quality Evaluator & Adaptive Retry Progression                     │
 │   • Language Detection & Multi-Lingual Routing                         │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                  4. DOCUMENT PARSING & TABLE EXTRACTION                │
 │   • Multi-Column Reading Order Reconstruction                          │
 │   • Hierarchical Heading & Section Detection (H1–H6)                   │
 │   • Multi-Page Table Merging & Schema Extraction                       │
 │   • Tabular Representations: Markdown, HTML, JSON & Summaries          │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │      5. VISUAL ASSET INTELLIGENCE & SELECTIVE VISION ANALYSIS          │
 │   • Extracted Figure / Diagram Ingestion & Storage (MinIO)             │
 │   • Exact SHA-256 & 64-bit Perceptual Hash (dHash) Deduplication       │
 │   • Cheap Candidate Classification (Logos, Icons, Watermarks, Diagrams)│
 │   • Spatial Caption Association & Document Context Propagation         │
 │   • Targeted Figure OCR & Technical Terminology Analysis               │
 │   • Heuristic Relevance Scorer & Policy Routing:                       │
 │       - STORE_ONLY / IGNORE: Preserved in MinIO for audit only         │
 │       - OCR_ONLY: Text chunk from OCR & Caption                        │
 │       - VISION_ANALYSIS: Structured Multi-modal OpenRouter Analysis    │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                   6. NORMALIZATION & DEDUPLICATION                     │
 │   • Unicode NFKC Normalization & Ligature Unrolling                    │
 │   • Cross-Page Header / Footer Repetitive Pattern Stripping            │
 │   • OCR Artifact & Typo Correction Heuristics                          │
 │   • Duplicate Resolver (Native Digital Text vs OCR Fallback)           │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                     7. CHUNKING & STRUCTURAL SPLIT                     │
 │   • Recursive Text Chunker (tiktoken token-aware with overlap)         │
 │   • Specialized Table Row & Header Preserving Chunker                  │
 │   • Semantic Figure Chunker (Only for Indexable Visual Policies)       │
 │   • Positional Metadata (Page, Section, Hierarchy, Coordinates, URI)   │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                  8. DATA MODELS & INCREMENTAL SYNC                     │
 │   • Documents & Document Versions (Change-Data-Capture via Hashes)     │
 │   • Chunks, Document Assets & Embeddings Persistence                   │
 │   • SQLAlchemy 2.0 ORM + Pydantic v2 Validation                        │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                       9. DENSE EMBEDDING SERVICE                       │
 │   • BaseEmbeddingProvider (Mock & OpenAI Providers)                    │
 │   • Vector Dimensionality & Validation (NaN / Inf Rejection)           │
 │   • Batch Processing, Exponential Backoff Retries & Idempotency        │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Modules Implemented

| Module | Location | Description |
|---|---|---|
| **Core Models** | `src/rag/core/` | 5 production tables (`documents`, `document_versions`, `chunks`, `document_assets`, `embeddings`), Pydantic schemas (`DocumentAssetSchema` with `perceptual_hash`, `relevance_score`, `indexing_policy`), and unified exceptions. |
| **Connectors** | `src/rag/connectors/` | `BaseSourceConnector`, `LocalFileConnector`, `S3Connector` (MinIO/AWS S3), `WebConnector`, `DatabaseConnector`. |
| **Raw Storage** | `src/rag/storage/` | `LocalRawDocumentStorage`, `S3RawStorageBackend` (MinIO/S3), `DatabaseRawDocumentStorage` with SHA-256 integrity checks and deterministic storage keys. |
| **OCR Engine** | `src/rag/ocr/` | Page quality analysis, image preprocessing profiles (`minimal`, `standard`, `aggressive`), multi-provider OCR, adaptive retries, and language detection. |
| **Parsing** | `src/rag/parsing/` | PDF layout parsing, multi-column reading order, hierarchical structure detection, table extraction & table chunking. |
| **Visual Assets** | `src/rag/assets/` | Extracted figure storage, SHA-256 + 64-bit dHash deduplication, frequency tracking, cheap classification, spatial caption association, figure OCR, OpenRouter / OpenAI vision models, heuristic relevance scoring, policy routing (`STORE_ONLY`, `OCR_ONLY`, `VISION_ANALYSIS`), and selective figure chunking. |
| **Normalization** | `src/rag/normalization/` | Text cleaning, header/footer suppression, OCR error correction, digital-vs-OCR duplicate resolution. |
| **Embeddings** | `src/rag/embeddings/` | Provider abstraction (`BaseEmbeddingProvider`), Mock & OpenAI providers, `EmbeddingValidator`, `EmbeddingService`, batching, retries, idempotency, and transactional persistence. |
| **Ingestion** | `src/rag/ingestion/` | `IngestionPipeline` coordinating end-to-end ingestion, hash-based change detection, table/figure chunking, selective vision analysis, and embedding generation. |

---

## 3. Database Schema (Production Data Model)

1. **`documents`**: Primary document record, tenant isolation, source URI, active version, status (`DISCOVERED`, `FETCHED`, `PARSED`, `CHUNKED`, `INDEXED`, `FAILED`), and latest content hash.
2. **`document_versions`**: Immutable historical version snapshots, content hashes, parser versions, and raw artifact paths.
3. **`chunks`**: Text blocks, section paths, page numbers, positional metadata, and chunking strategy version.
4. **`document_assets`**: Extracted figures, diagrams, architecture illustrations, charts, bounding boxes, storage URIs (MinIO/local), captions, OCR text, semantic descriptions, entities, relationships, `perceptual_hash`, `relevance_score`, `relevance_reason`, `indexing_policy`, `processing_error`, and lifecycle statuses.
5. **`embeddings`**: Dense vector representation (JSON float list), model identifier, model version, dimensionality, and provider tag.

---

## 4. Test Suite Status

All **211 test cases** across 26 test suites are currently passing:

```bash
$ pytest -v
============================= 211 passed in 19.81s =============================
```

- `test_assets_deduplication.py`: SHA-256 and pure-PIL 64-bit perceptual hash (`dHash`), hamming distance, frequency tracking, and analysis cache.
- `test_assets_relevance.py`: Heuristic relevance scoring, weights, technical OCR keywords boost, logo/icon penalties, and `IndexingPolicy` assignment.
- `test_assets_selective_vision.py`: Selective vision routing, chunk generation filtering (`STORE_ONLY` vs `VISION_ANALYSIS`), and safety call budgets.
- `test_assets_openrouter.py`: OpenRouter free vision models candidate failover and contextual structured output extraction.
- `test_assets_extraction.py`: Digital PDF embedded image extraction, scanned page figure detection & cropping, bbox calculations.
- `test_assets_classification.py`: Logo/icon filtering vs diagram/chart/architecture classification.
- `test_assets_caption.py`: Caption detection and spatial figure association.
- `test_assets_ocr.py`: Figure OCR with Tesseract and confidence extraction.
- `test_assets_analyzer.py`: Vision provider abstractions, entity & relationship extraction, failure handling.
- `test_assets_storage.py`: Storage key determinism, image byte persistence, and SHA-256 deduplication.
- `test_assets_pipeline_integration.py`: End-to-end pipeline ingestion with text, tables, figures, diagrams, and embedding persistence.
- `test_embeddings_provider.py`: Mock and OpenAI provider contracts, batching, timeouts, rate limits.
- `test_embeddings_validation.py`: Vector dimensionality, NaN/Inf rejection, and schema validation.
- `test_embeddings_service.py`: Chunk validation, oversized chunk strategies, batch fallback isolation.
- `test_embeddings_idempotency.py`: Skipping existing embeddings, multi-model versioning coexistence.
- `test_embeddings_pipeline_integration.py`: End-to-end pipeline ingestion, auto-embedding, and incremental sync.
- `test_connectors.py`: Local filesystem, S3, Web, and Database connectors.
- `test_models.py`: Pydantic validation and SQLAlchemy ORM lifecycle.
- `test_raw_storage.py`: Local and database raw payload storage, retrieval, and hashing.
- `test_ocr.py`: Page quality scoring, image preprocessing, OCR retries, and mixed-mode pages.
- `test_language_detection.py`: Language identification and multi-lingual OCR routing.
- `test_parsing.py`: Reading order, heading detection, and PDF parsing.
- `test_table_processing.py` & `test_advanced_tables.py`: Table extraction, multi-page merging, formats, and chunking.
- `test_normalization.py`: Text cleaning, header/footer removal, and duplicate resolution.
- `test_incremental_ingestion.py`: Hash-based change detection and versioning.
- `test_config.py`: Configuration loading, validation, and environment overrides.

