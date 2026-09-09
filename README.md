# Enterprise RAG Pipeline

A modular, production-grade **Retrieval-Augmented Generation** pipeline in Python. It ingests documents from multiple sources, intelligently extracts text, tables, and visual figures, normalizes content, chunks it structurally, and produces dense vector embeddings — all tracked through an incremental versioned database.

---

## Architecture

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                    1. SOURCE CONNECTOR LAYER                        │
 │    [Local Files]      [AWS S3 / MinIO]      [Web / HTTP]    [SQL]  │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │ (RawDocumentPayload)
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                      2. RAW STORAGE ENGINE                          │
 │       Local Filesystem  /  S3 / MinIO Object Storage                │
 │       (SHA-256 Content Hashing & Deduplication)                     │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                   3. INTELLIGENT OCR ENGINE                         │
 │   Page Quality → Preprocessing → Multi-Provider OCR → Retry        │
 │   Language Detection → Quality Evaluation → Mixed-Mode Merge       │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │              4. PDF PARSING & TABLE EXTRACTION                      │
 │   Multi-Column Reading Order → Heading Detection → Table Merging   │
 │   pdfplumber + PyMuPDF → Markdown / JSON / HTML Serialization      │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │       5. VISUAL ASSET INTELLIGENCE & SELECTIVE VISION               │
 │   Extract → Deduplicate → Classify → Score → Route Policy          │
 │   OCR_ONLY / VISION_ANALYSIS / STORE_ONLY                          │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                5. NORMALIZATION & CHUNKING                          │
 │   Unicode NFKC → Header/Footer Strip → Token-Aware Chunking       │
 │   Text Chunks + Table Chunks + Figure Chunks                       │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │              6. DATABASE MODELS & EMBEDDING SERVICE                 │
 │   SQLAlchemy 2.0 ORM + Pydantic v2 + Dense Vector Embeddings      │
 │   Incremental Sync (Hash-Based Change Detection & Versioning)      │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Data Sources & Architecture

### Concept
The pipeline supports ingesting documents from multiple heterogeneous sources through a unified connector interface. Each connector knows how to discover, validate, and fetch documents from its source system.

### How the Logic Works
- **`BaseSourceConnector`** defines four abstract operations: `validate_connection()`, `discover()`, `fetch()`, and `get_metadata()`.
- **Discovery** yields lightweight `DocumentDescriptor` objects (URI, filename, content hash, last modified time) without loading file bytes — enabling cheap incremental scans.
- **Fetch** loads the raw bytes on-demand, computes a SHA-256 content hash, and returns a `RawDocumentPayload`.
- Implementations: `LocalFileConnector` (walks directories via `pathlib`), `S3Connector` (MinIO/AWS S3 via `boto3`), `WebConnector` (HTTP fetcher), `DatabaseConnector` (SQL row export).

```
Source System → Connector.discover() → DocumentDescriptor
                Connector.fetch()    → RawDocumentPayload (bytes + SHA-256 hash)
```

---

## 2. Database Models

### Concept
All document lifecycle state, extracted content, and metadata are tracked in a relational database (PostgreSQL in production, SQLite for tests). The schema uses SQLAlchemy 2.0 Mapped Columns with Pydantic v2 validation for all data entering and leaving the ORM layer.

### How the Logic Works

| Table | Purpose |
|---|---|
| `documents` | Primary record: tenant, source, URI, filename, active version, status lifecycle (`DISCOVERED` → `FETCHED` → `PARSED` → `CHUNKED` → `INDEXED`), content hash |
| `document_versions` | Immutable version snapshots — each re-ingestion creates a new version if the content hash changed |
| `document_version_metadata` | Relational normalization of version-level statistics (page counts, language, OCR diagnostics, table diagnostics) |
| `chunks` | Text blocks with section path, page number, chunk index, and raw JSON metadata |
| `chunk_metadata` | Normalized per-chunk attributes: token count, language, quality score, content source, OCR flags, table linkage |
| `document_assets` | Extracted figures/diagrams with bounding box coordinates, storage URIs, captions, OCR text, relevance scores, and indexing policies |
| `document_page_scores` | Per-page quality score, content source classification, language, OCR metrics |
| `asset_entities` | Individual visual entities detected in diagrams (e.g., "Database", "API Gateway") |
| `asset_relationships` | Directed relationships between entities (e.g., "Client" → connects to → "Server") |
| `embeddings` | Dense vector representations with model ID, version, dimensionality, and provider tag |

**Tabular SQL Views** (`v_documents_tabular`, `v_chunks_tabular`, `v_assets_tabular`, `v_page_scores_tabular`, `v_asset_entities_tabular`, `v_asset_relationships_tabular`) flatten all JSON metadata into clean queryable columns.

**Migration** (`migrate.py`) is idempotent — it runs `CREATE TABLE IF NOT EXISTS`, safe `ALTER TABLE ADD COLUMN IF NOT EXISTS`, creates all views, and backfills existing JSON data into the new relational tables.

---

## 3. Source Connectors — LocalFile, S3/MinIO

### Concept
Connectors are the bridge between external data sources and the ingestion pipeline. They implement incremental discovery using content hashes and timestamps so only new or changed documents are re-processed.

### How the Logic Works

**LocalFileConnector:**
1. Walks a directory tree using `pathlib.Path.rglob()` with configurable file extension filters.
2. For each file, yields a `DocumentDescriptor` with the file's absolute path as `source_uri`, the filename, size, and `last_modified` timestamp.
3. `fetch()` reads file bytes, computes SHA-256 hash, returns `RawDocumentPayload`.

**S3Connector (MinIO/AWS S3):**
1. Uses `boto3` to `list_objects_v2()` with configurable prefix and bucket.
2. Supports `since` parameter for incremental discovery — only yields objects modified after the given timestamp.
3. `fetch()` calls `s3.get_object()`, reads the streaming body, and computes SHA-256.
4. MinIO compatibility is achieved by setting `endpoint_url` in the boto3 client configuration.

**Incremental Change Detection:**
```
For each discovered document:
  1. Compute SHA-256 of fetched content
  2. Compare with stored content_hash in documents table
  3. If hash matches → UNCHANGED (skip)
  4. If hash differs → CHANGED (create new version)
  5. If not in DB   → NEW (create document + version 1)
  6. If in DB but not discovered → DELETED (mark for cleanup)
```

---

## 4. OCR Engine — Intelligent Page OCR Pipeline

### Concept
Not all PDF pages contain machine-readable text. Some are scanned images, some are mixed (digital text with embedded scanned sections). The OCR engine classifies each page, applies appropriate preprocessing, runs OCR with adaptive retries, and merges results intelligently.

### How the Logic Works

**Page Quality Analyzer** classifies each page into three categories:
- **Native (digital)**: Machine-readable text with quality score ≥ 0.70, extracted directly via PyMuPDF.
- **Scanned**: Low or zero native text, high image coverage (≥ 35% of page area) → needs full OCR.
- **Mixed**: Some native text exists but image coverage is significant → needs OCR for image regions, then merge.

The quality score is a weighted composite:
```
quality = (printable_ratio × 0.40) + (meaningful_word_ratio × 0.45) + (replacement_penalty × 0.15)
```

**Image Preprocessing** applies profile-based transformations before OCR:
| Profile | Transforms |
|---|---|
| `minimal` | Grayscale conversion only |
| `standard` | Grayscale + contrast enhancement + binarization |
| `aggressive` | Grayscale + deskewing + high-contrast binarization + DPI upscaling |

**Adaptive Retry Progression:**
```
Attempt 1: standard preprocessing → Tesseract OCR
  If quality < threshold:
Attempt 2: aggressive preprocessing → Tesseract OCR
  If quality < threshold:
Attempt 3: aggressive preprocessing → fallback provider (EasyOCR/DocTR)
```

**Language Detection** (`LanguageDetector`) identifies the document language using character frequency analysis, script detection, and `langdetect`. This routes Tesseract to the correct language pack (e.g., `eng`, `fra`, `deu`, `jpn`).

**Mixed Content Merger** combines native digital text with OCR results for mixed pages by subtracting OCR blocks that overlap with already-extracted digital text regions, preventing duplicate content.

---

## 5. PDF Parsing — Layout-Aware PDF Parser

### Concept
PDF documents don't store text in reading order — text blocks are positioned absolutely on a page canvas. The parser reconstructs the logical reading order, detects structural elements (headings, paragraphs, tables, figures), and produces a normalized `ParsedDocument`.

### How the Logic Works

**Multi-Column Reading Order Reconstruction:**
1. Extract all text blocks with bounding boxes from PyMuPDF.
2. Sort blocks by vertical position (top → bottom).
3. Group blocks into horizontal bands (blocks at similar Y-coordinates).
4. Within each band, sort left → right to reconstruct multi-column layouts.
5. This produces the correct reading order for 2-column and 3-column PDF layouts.

**Hierarchical Heading Detection:**
- A regex-based heuristic identifies section headings by pattern:
  - Numbered headings: `1.2.3 Section Title`
  - Capitalized short lines: `INTRODUCTION`, `System Architecture`
  - Font size analysis via PyMuPDF span metadata (larger fonts → higher heading levels H1–H6)
- Heading hierarchy is maintained as a section path stack (e.g., `["1. Introduction", "1.2 Background"]`).

**Scanned vs Native Classification:**
- Each page is classified as `native`, `scanned`, or `mixed` by the `PageQualityAnalyzer`.
- Native pages use direct PyMuPDF text extraction.
- Scanned pages are rendered to high-DPI images and processed through the OCR pipeline.
- Mixed pages merge both sources, with digital text taking priority over OCR in overlapping regions.

**Output:**
```python
ParsedDocument(
    pages=[ParsedPage(elements=[TextElement, HeadingElement, TableElement, FigureElement])],
    num_pages=N,
    pdf_metadata={...},
    table_diagnostics={...},
)
```

---

## 6. Table Extraction — Multi-Table & Multi-Page Processor

### Concept
Tables in PDFs are complex — they span multiple pages, have hierarchical headers (multi-row), contain merged cells, and may be detected by one library but not another. The table processor handles all these cases with a dual-engine approach and produces multiple serialization formats.

### How the Logic Works

**Dual-Engine Extraction:**
1. **pdfplumber** (primary): Uses ruled-line and whitespace-based detection to extract tables with cell coordinates.
2. **PyMuPDF** (fallback): If pdfplumber finds no tables, PyMuPDF's built-in `find_tables()` is tried.
3. Results from both engines can be merged when each detects complementary tables.

**Multi-Page Table Continuation Stitching:**
```
Page N: Table detected with columns ["Name", "Price", "Qty"]
Page N+1: Table detected with identical column structure
→ Stitch into single logical table with flag is_continuation=True
```
The `TableContinuationStitcher` compares column counts and header text similarity to decide if a table on the next page is a continuation of the previous table.

**Hierarchical Header Detection:**
- Multi-row headers are detected when the first 2-3 rows contain merged cells or column-spanning text.
- `TableHeaderAndBoundaryDetector` identifies the header boundary row.

**Table Validation:**
- Checks for minimum cell population, column consistency, and data density.
- Tables with too many empty cells or inconsistent column counts are flagged as low-quality.

**Serialization Formats:**
| Format | Use Case |
|---|---|
| **Markdown** | Default chunking format — preserves structure for LLM consumption |
| **JSON** | Structured key-value pairs per row — ideal for database insertion |
| **HTML** | Formatted `<table>` tags — for web rendering |
| **Summary** | Natural language description of table contents |

**Sparse Cell Inheritance:** When a merged cell spans multiple rows (common in financial tables), the value is inherited downward to all child rows so no information is lost during chunking.

---

## 7. OpenRouter Integration

### Concept
For figures that score high enough to warrant detailed visual analysis, the pipeline calls a multimodal vision model via OpenRouter's API. OpenRouter provides access to free vision-capable models with automatic failover.

### How the Logic Works

**Automatic Free Model Selection & Failover:**
```python
FREE_VISION_MODELS = [
    "openrouter/free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "minimax/minimax-m3:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]
```

The provider tries models in order. If one returns an error (rate limit, timeout, model unavailable), it moves to the next:
```
Try Model 1 → HTTP 429 (rate limited) → Try Model 2 → HTTP 200 → Use result
```

**API Call Structure:**
1. Image bytes are base64-encoded and sent as a `data:image/png;base64,...` URL in the OpenAI-compatible chat format.
2. A structured prompt instructs the model to return JSON with: `description`, `entities[]`, `relationships[{source, relationship, target}]`, and `asset_type_suggestion`.
3. The response JSON is parsed into a `FigureAnalysisResult` dataclass.

**Smart Contextual Fallback (No API Key):**
When no `OPENROUTER_API_KEY` is configured, the provider falls back to a context-aware analysis using:
- OCR text extracted from the figure
- Caption text detected near the figure
- Surrounding document section headings
- Asset type classification from the classifier

This produces a reasonable description without calling any external API.

**Safety Budget:** A configurable `max_vision_calls_per_document` cap prevents runaway API costs on documents with many figures.

---

## 8. Intelligent Figure/Image Relevance Classification & Selective Vision Analysis

### Concept
Not every image in a PDF deserves expensive vision model analysis. Logos, watermarks, and decorative icons should be stored but not analyzed. Architecture diagrams, flowcharts, and data charts deserve full vision analysis. This layer classifies every extracted figure and routes it to the appropriate processing policy.

### How the Logic Works

The pipeline processes figures through a multi-stage classification and scoring pipeline:

```
Extract Image → Deduplicate → Classify Type → Score Relevance → Assign Policy → Process
```

### Stage 1: Extraction
- **Digital PDFs**: Embedded images are extracted directly from the PDF's internal image objects via PyMuPDF.
- **Scanned PDFs**: The page is rendered at high DPI, then figures are identified by detecting image bounding boxes.
- Each extracted candidate includes: image bytes, bounding box coordinates, page number, dimensions, and page coverage ratio.

### Stage 2: Deduplication (SHA-256 + Perceptual Hash)
Two-level deduplication prevents processing the same image multiple times:

1. **Exact Match (SHA-256)**: Byte-identical images (same logo on every page) are detected instantly.
2. **Near-Duplicate (64-bit dHash)**: Perceptually similar images (same logo with slight compression differences) are caught by computing a difference hash:
   ```
   Resize to 9×8 grayscale → Compare adjacent pixel brightness → 64-bit hash
   Hamming distance ≤ 5 bits → Considered duplicate
   ```
3. **Frequency Tracking**: Each unique hash tracks how many pages it appears on. Recurring assets (logos in headers) get penalized in scoring.

### Stage 3: Cheap Classification
Without calling any AI model, the classifier categorizes each figure using heuristics:

| Signal | Classification |
|---|---|
| Width ≤ 64px and height ≤ 64px | `ICON` / `DECORATIVE` |
| Aspect ratio > 5:1 and narrow height | `WATERMARK` / `HEADER_BANNER` |
| Appears on > 50% of pages | `LOGO` / `RECURRING_ELEMENT` |
| Contains text like "Figure 3" in caption | `DIAGRAM` / `CHART` |
| Large area (> 30% page coverage) | `ARCHITECTURE_DIAGRAM` / `FLOWCHART` |

Asset types include: `ARCHITECTURE_DIAGRAM`, `FLOWCHART`, `CHART`, `GRAPH`, `TABLE_IMAGE`, `SCREENSHOT`, `DIAGRAM`, `FIGURE`, `PHOTO`, `LOGO`, `ICON`, `WATERMARK`, `DECORATIVE`, etc.

### Stage 4: Heuristic Relevance Scoring
A weighted composite score [0.0 – 1.0] determines how informative a figure is:

```
Score = Base(type) + Caption(+0.10) + FigureLabel(+0.08) + OCR_Words(+0.08)
      + TechnicalTerms(+0.12) + Context(+0.05) + Geometry(+0.05)
      + PageCoverage(+0.05) - LogoPenalty(-0.15) - WatermarkPenalty(-0.20)
      - RepeatedPenalty(-0.08 × frequency)
```

**Technical Term Boost:** If OCR or caption text contains domain-specific keywords (`architecture`, `pipeline`, `database`, `kubernetes`, `embedding`, `vector`, etc.), the score is boosted significantly.

### Stage 5: Policy Assignment

| Score Range | Policy | Action |
|---|---|---|
| `< 0.25` (or decorative) | `STORE_ONLY` | Save to MinIO for audit, no indexing |
| `0.25 – 0.55` with OCR text | `OCR_ONLY` | Create text chunk from OCR + caption |
| `0.25 – 0.55` without OCR | `STORE_ONLY` | Not enough information to index |
| `≥ 0.55` | `VISION_ANALYSIS` | Send to OpenRouter vision model for structured analysis |

### Stage 6: Selective Processing
Only figures with `VISION_ANALYSIS` policy are sent to the expensive vision model. This typically reduces API calls by 60-80% compared to analyzing every figure.

**For `VISION_ANALYSIS` figures**, the vision model returns:
- Natural language description of the diagram
- List of entities/components detected
- Directed relationships between entities
- Suggested asset type refinement

**For `OCR_ONLY` figures**, a text chunk is generated from:
- OCR text extracted by Tesseract
- Nearby caption text
- Section heading context

**For `STORE_ONLY` figures**, the image is stored in MinIO/local storage with metadata, but no searchable chunk is created.

---

## Module Structure

```
src/rag/
├── assets/               # Visual asset intelligence layer
│   ├── analyzer.py         # Vision provider abstraction
│   ├── caption_detector.py # Spatial caption-to-figure association
│   ├── chunker.py          # Figure-aware chunk generation
│   ├── classifier.py       # Cheap heuristic figure classification
│   ├── deduplicator.py     # SHA-256 + dHash deduplication
│   ├── extractor.py        # PDF figure/image extraction
│   ├── ocr.py              # Figure-specific OCR
│   ├── processor.py        # End-to-end asset processing orchestrator
│   ├── relevance.py        # Heuristic relevance scoring & policy routing
│   ├── storage.py          # MinIO/local figure storage
│   └── providers/          # Vision model providers (OpenRouter, OpenAI)
├── connectors/           # Source data connectors
│   ├── base.py             # Abstract BaseSourceConnector
│   ├── local_file.py       # Local filesystem connector
│   ├── s3.py               # AWS S3 / MinIO connector
│   ├── web.py              # HTTP/web connector
│   └── database.py         # SQL database connector
├── core/                 # Database models, config, exceptions
│   ├── config.py           # Pydantic settings with YAML/env support
│   ├── database.py         # SQLAlchemy engine & session management
│   ├── db_models.py        # All ORM table definitions
│   ├── migrate.py          # Schema migrations, views, backfill
│   ├── models.py           # Pydantic schemas & enums
│   └── exceptions.py       # Domain-specific exception hierarchy
├── embeddings/           # Dense vector embedding service
│   ├── providers/          # Mock & OpenAI embedding providers
│   ├── service.py          # Batch embedding orchestrator
│   └── validator.py        # Vector validation (NaN/Inf rejection)
├── ingestion/            # Pipeline orchestration
│   └── pipeline.py         # End-to-end ingestion coordinator
├── normalization/        # Text cleaning & deduplication
│   └── normalizer.py       # Unicode, header/footer strip, OCR fix
├── ocr/                  # Intelligent OCR subsystem
│   ├── analyzer.py         # Page quality scoring & classification
│   ├── preprocessor.py     # Image preprocessing profiles
│   ├── evaluator.py        # OCR output quality evaluation
│   ├── language_detector.py # Language identification
│   ├── merger.py           # Native + OCR content merger
│   └── providers/          # Tesseract, EasyOCR, DocTR providers
├── parsing/              # PDF parsing & table extraction
│   ├── pdf_parser.py       # Layout-aware PDF parser
│   ├── chunker.py          # Token-aware recursive text chunker
│   └── table_processor.py  # Multi-page table stitching & validation
└── storage/              # Raw document storage backends
    ├── local.py            # Local filesystem storage
    └── s3.py               # S3/MinIO object storage
```

---

## Configuration

All settings are managed through a layered configuration system:
1. **YAML config file** (`configs/default.yaml`)
2. **Environment variables** (override YAML values)
3. **`.env` file** (loaded automatically)

Key configuration blocks: `database`, `storage`, `parser`, `chunking`, `table`, `ocr`, `asset`, `embedding`, `pipeline`, `logging`.

See [`docs/CONFIG.md`](docs/CONFIG.md) for full configuration reference.

---

## Test Suite

All **211 test cases** across 26 test files pass:

```bash
$ pytest
============================= 211 passed in 19.71s =============================
```