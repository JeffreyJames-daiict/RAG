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
 │                6. NORMALIZATION & CHUNKING                          │
 │   Unicode NFKC → Header/Footer Strip → Token-Aware Chunking       │
 │   Text Chunks + Table Chunks + Figure Chunks                       │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │              7. DATABASE MODELS & EMBEDDING SERVICE                 │
 │   SQLAlchemy 2.0 ORM + Pydantic v2 + Dense Vector Embeddings      │
 │   Incremental Sync (Hash-Based Change Detection & Versioning)      │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │              8. HYBRID RETRIEVAL & RANKING ENGINE                   │
 │   Dense Vector (pgvector/Python) + Sparse BM25 + RRF Fusion         │
 │   Table-Aware Retrieval & Parent-Child Segment Expansion            │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │              9. GROUNDED LLM ANSWER GENERATION                      │
 │   OpenRouter Dynamic Failover & Exponential Backoff Retries        │
 │   Numbered Citation Grounding ([1], [2]) + Transparent Streaming    │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │             10. INTERACTIVE STREAMLIT STUDIO UI                     │
 │   Multi-Mode Search, Latency Badges, Table Viewer & Corpus Explorer │
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

## 9. Hybrid Retrieval Subsystem (Dense Vector, BM25, RRF & Table Expansion)

### Concept
The retrieval layer combines the semantic generalization of **dense vector search** with the precision and exact-term recall of **sparse BM25 keyword search** using **Reciprocal Rank Fusion (RRF)**. It also includes table-aware chunk retrieval with parent-child table expansion.

### How the Logic Works

```
User Query ──┬──► Dense Vector Search ──► Vector Candidates (Cosine) ──┐
             │                                                         ├──► RRF Fusion ──► Ranked Hydrated Results
             └──► Sparse BM25 Search   ──► BM25 Candidates (TF-IDF)   ──┘
```

1. **Dense Vector Search (`VectorRetrievalService`)**:
   - Generates the query embedding via `EmbeddingService`.
   - Searches chunk embeddings in PostgreSQL using `pgvector` (or automatic fallback to `PythonVectorRepository` with cosine similarity).
   - Validates vector dimensions, finiteness, and cosine bounds $[-1.0, 1.0]$.
2. **Sparse BM25 Search (`BM25Retriever`)**:
   - Evaluates token frequencies using in-memory BM25 index with configurable $k_1$ (default `1.5`) and $b$ (default `0.75`).
   - Enables exact matching on IDs, article codes, and technical jargon.
3. **Reciprocal Rank Fusion (`RRFFusion`)**:
   - Fuses ranked candidate lists using the formula:
     $$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
     where $k=60$ and $r_m(d)$ is the 1-based rank of candidate $d$ in list $m$.
4. **Table-Aware Retrieval (`TableAwareRetriever`)**:
   - Searches text, tabular, and table summary chunks.
   - `expand_parent_table(table_id, document_id)` identifies all sister chunks belonging to the same parent table (across multi-page continuations or row slices) and stitches them into a unified table context.
5. **Metadata Filtering (`RetrievalFilters`)**:
   - SQL clause builders and in-memory predicates for `document_ids`, `chunk_types`, `page_numbers`, `languages`, `created_after`, and JSON metadata matching.

---

## 10. Grounded LLM Answer Generation (OpenRouter, Retries & Model Failover)

### Concept
The generation engine (`rag.generation`) transforms retrieved context passages and user questions into grounded, natural language answers with inline numbered citations. It is built for production reliability, with automatic exponential backoff retries and dynamic candidate model rotation across OpenRouter's free tier.

### How the Logic Works

```
Query + Retrieved Context Chunks ──► Prompt Construction with Provenance
                                             │
                                             ▼
                                  OpenRouter Generator
                        ┌────────────────────────────────────────┐
                        │ Attempt Primary Model                  │
                        │  ├── Success ──► Stream / Return Answer│
                        │  └── 429 / 404 / 500 / Timeout         │
                        │        ├── Retry with Backoff (2x)     │
                        │        └── Fail? ──► Rotate to Next:   │
                        │              1. openrouter/free        │
                        │              2. google/gemma-4-31b-it  │
                        │              3. google/gemma-4-26b-a4b │
                        │              4. nemotron-3-nano-omni   │
                        │              5. liquid/lfm-2.5-2.6b    │
                        └────────────────────────────────────────┘
```

1. **Structured Prompt Construction (`LLMGenerationService`)**:
   - Injects numbered citation blocks: `[1] • Source: report.pdf • Page 2 • Section: Executive Summary`.
   - Enforces grounding guidelines: synthesizes findings strictly from retrieved passages without hallucination.
2. **Automatic Retry with Exponential Backoff**:
   - Transient errors (HTTP 429 rate limits, 502, 503, 504, or network timeouts) automatically retry up to `max_retries_per_model` times with backoff ($0.6 \times 2^{\text{retry}}$ seconds).
3. **Automatic Candidate Model Rotation**:
   - When a model returns permanent errors (e.g. HTTP 400/404) or exhausts its retry budget, the client logs telemetry into `ModelAttemptRecord` and rotates immediately to the next candidate model in the roster.
   - Default primary model is `openrouter/free`, which dynamically routes to the lowest-latency healthy free model on OpenRouter.
4. **Citation Extraction**:
   - Regex parser automatically extracts all inline bracket citations (e.g. `[1]`, `[2]`) into `cited_ranks` for UI provenance linking.
5. **Streaming & Non-Streaming**:
   - Supports both complete response generation and real-time SSE token streaming (`generate_stream`) with model selection events.

---

## 11. Interactive Streamlit Studio UI

### Concept
The `UI/` folder provides an interactive, dark-mode web application built with Streamlit that connects directly to the RAG backend engine.

### Key UI Features:
- **Search & Multi-Mode Retrieval**:
  - Strategy selector: `Hybrid RRF`, `Dense Vector (Cosine)`, `BM25 Keyword`, `Table-Aware Expansion`.
  - Top-K slider (1–25) and minimum similarity threshold filter.
  - Document filter multi-select (dynamically queried from PostgreSQL database).
  - Chunk type filters (`text`, `table`, `table_summary`, `figure`) and page range filtering.
- **Real-Time Latency & Diagnostics Bar**:
  - Instant breakdown of: Total latency, Query embedding time, Vector search time, BM25 time, and RRF fusion latency.
- **Ranked Result Cards**:
  - Visual score pills: RRF score, Vector similarity score, and BM25 score.
  - Section breadcrumb hierarchy and page provenance.
  - Query term highlighting.
  - Collapsible JSON metadata inspector.
- **Structured Table Viewer & Parent Expansion**:
  - Markdown table rendering.
  - "Expand Parent Table" button to inspect stitched multi-page tables.
- **Knowledge Corpus Explorer**:
  - Ingested documents table (filename, version, chunk count, status, creation timestamp).
  - Visual figure and diagram gallery with relevance scores and OCR snippets.
- **Generative AI Answer Synthesis**:
  - Live streaming answer with inline citations (`[1]`, `[2]`).
  - Real-time model badge showing active model and auto-failover notification if a model was rotated.

**How to Launch the UI:**
```bash
./RAG/venv/bin/python3 -m streamlit run UI/app.py --server.port 8501
```
Navigate to `http://localhost:8501`.

---

## Module Structure

```
├── RAG/
│   ├── configs/              # Centralized JSON configuration & schemas
│   │   ├── config.json         # Master settings (DB, OCR, embeddings, retrieval, vision)
│   │   └── config.schema.json  # Pydantic v2 JSON Schema for IDE validation
│   └── src/rag/
│       ├── assets/           # Visual asset intelligence layer
│       │   ├── analyzer.py     # Vision provider abstraction
│       │   ├── caption_detector.py # Spatial caption-to-figure association
│       │   ├── chunker.py      # Figure-aware chunk generation
│       │   ├── classifier.py   # Heuristic figure classification (logos, banners, diagrams)
│       │   ├── deduplicator.py # SHA-256 + 64-bit dHash deduplication
│       │   ├── extractor.py    # PDF figure and image extraction
│       │   ├── ocr.py          # Figure-specific OCR extraction
│       │   ├── processor.py    # End-to-end asset processing orchestrator
│       │   ├── relevance.py    # Heuristic relevance scoring & policy routing
│       │   ├── storage.py      # MinIO/local figure storage backend
│       │   └── providers/      # OpenRouter and OpenAI vision providers
│       ├── connectors/       # Source data connectors (Local, S3/MinIO, Web, SQL)
│       ├── core/             # Database models, config, migrations, exceptions
│       ├── embeddings/       # Dense vector embedding service (Mock, OpenAI)
│       ├── generation/       # Grounded LLM answer generation subsystem
│       │   ├── models.py       # Pydantic models (GenerationRequest, GenerationResponse)
│       │   ├── openrouter_client.py # OpenRouter client with retries & model rotation
│       │   └── service.py      # LLMGenerationService prompt formatting & citations
│       ├── ingestion/        # Pipeline orchestration & incremental sync
│       ├── normalization/    # Unicode NFKC, header/footer stripping, OCR repair
│       ├── ocr/              # Intelligent OCR subsystem (Tesseract, quality scoring, merge)
│       ├── parsing/          # PDF parsing, recursive token chunking, table processing
│       ├── retrieval/        # Hybrid semantic and keyword retrieval engine
│       │   ├── bm25.py         # BM25 sparse keyword retriever
│       │   ├── filters.py      # Typed metadata filters & SQL clause builders
│       │   ├── fusion.py       # Reciprocal Rank Fusion (RRF) algorithm
│       │   ├── hybrid_service.py # Vector + BM25 + RRF orchestrator
│       │   ├── models.py       # RetrievalRequest, RetrievalResponse, RetrievedChunkResult
│       │   ├── repositories/   # pgvector and Python in-memory vector repositories
│       │   ├── retriever.py    # Table-aware retriever & parent table expansion
│       │   └── vector_service.py # Vector semantic retrieval orchestrator
│       └── storage/          # Raw document payload storage (Local, MinIO/S3, DB)
│
└── UI/                       # Interactive Streamlit Studio
    ├── app.py                # Main Streamlit application entrypoint
    ├── config.py             # UI path resolution & sample query presets
    ├── rag_engine.py         # Adapter bridging Streamlit to RAG retrieval & generation
    ├── requirements.txt      # UI dependencies (streamlit, pandas, etc.)
    ├── styles.css            # Custom dark-mode design system & card styling
    └── components/           # Reusable UI presentation components
        ├── answer_view.py    # Generative AI answer synthesis with citations & model badges
        ├── corpus_view.py    # Ingested documents catalog & visual asset gallery
        ├── header.py         # Branding banner & corpus KPI metrics
        ├── results_view.py   # Latency diagnostics bar & ranked chunk cards
        ├── sidebar.py        # Mode selector, top-k, thresholds, and filters
        └── table_view.py     # Structured table viewer & parent expansion
```

---

## Configuration

All settings are managed through a layered configuration system:
1. **JSON config file** (`configs/config.json`) with environment variable expansion (`${VAR:-default}`)
2. **Environment variables** (override config values)
3. **`.env` file** (loaded automatically)

Key configuration blocks: `app`, `database`, `storage`, `parser`, `chunking`, `table`, `ocr`, `normalization`, `asset`, `embedding`, `retrieval`, `pipeline`, `logging`.

---

## Test Suite

All **211 test cases** across 26 test files pass:

```bash
$ pytest
============================= 211 passed in 19.71s =============================
```