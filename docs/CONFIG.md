# Configuration Guide (`configs/config.json`)

All configuration for this Enterprise RAG repository is centralized in [`configs/config.json`](file:///home/inx/Documents/INX%20Projects/RAG/configs/config.json).


---

## 1. Structure of `config.json`

```json
{
  "$schema": "./config.schema.json",
  "app": {
    "name": "Enterprise RAG Pipeline",
    "version": "0.1.0",
    "env": "development"
  },
  "database": {
    "url": "${DATABASE_URL:-postgresql://rag_user:rag_password_123@127.0.0.1:5432/rag}",
    "pool_size": 5,
    "max_overflow": 10,
    "echo": false
  },
  "storage": {
    "raw_storage_dir": "${RAW_STORAGE_DIR:-./raw_store}",
    "verify_on_read": false,
    "backend_type": "local"
  },
  "parser": {
    "version": "pymupdf-v1.0",
    "extract_images": false,
    "supported_mime_types": [
      "application/pdf",
      "application/x-pdf"
    ]
  },
  "chunking": {
    "chunk_size": 512,
    "chunk_overlap": 64,
    "chunking_version": "v1.0",
    "encoding_model": "cl100k_base"
  },
  "pipeline": {
    "default_tenant_id": "default",
    "handle_deletions": true,
    "force_fetch": false,
    "auto_migrate": true
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

---

## 2. Configuration Sections Explained

### 🚀 `app`
| Field | Type | Description |
|---|---|---|
| `name` | string | Application display name. |
| `version` | string | Application version. |
| `env` | string | Deployment environment (`development`, `staging`, `production`). |

### 🗄️ `database`
| Field | Type | Description |
|---|---|---|
| `url` | string | Full SQLAlchemy database URL (PostgreSQL, SQLite). Supports `${DATABASE_URL}`. |
| `pool_size` | integer | Number of connections to keep open in the pool (default: `5`). |
| `max_overflow` | integer | Number of overflow connections allowed above `pool_size` (default: `10`). |
| `echo` | boolean | If `true`, logs all generated SQL statements (for debugging). |

### 💾 `storage` (Raw Immutable Storage)
| Field | Type | Description |
|---|---|---|
| `raw_storage_dir` | string | Directory path where original document byte payloads and `.sha256` sidecars are preserved. Supports `${RAW_STORAGE_DIR}`. |
| `verify_on_read` | boolean | If `true`, validates content against SHA-256 sidecar on every read to detect bit rot or disk tampering. |
| `backend_type` | string | Storage backend: `local` (filesystem), or future backends (`s3`, `gcs`). |

### 📄 `parser`
| Field | Type | Description |
|---|---|---|
| `version` | string | Parser tag (e.g. `pymupdf-v1.0`) recorded on `document_versions` rows. |
| `extract_images` | boolean | Whether to extract/annotate image blocks during PDF parsing. |
| `supported_mime_types` | array | List of MIME types processed by the PDF parser. |

### 🧩 `chunking`
| Field | Type | Description |
|---|---|---|
| `chunk_size` | integer | Max token budget per chunk (default: `512`). |
| `chunk_overlap` | integer | Overlap tokens between consecutive chunks to preserve context (default: `64`). |
| `chunking_version` | string | Version tag stored on `chunks` rows. Bump when chunking parameters change to enable differential chunk re-indexing. |
| `encoding_model` | string | `tiktoken` tokenizer model used for token counting (default: `cl100k_base`). |

### 🖼️ `asset` (Visual Asset Extraction & Selective Vision)
| Field | Type | Description |
|---|---|---|
| `enabled` | boolean | Master toggle for PDF figure/image extraction and processing (default: `true`). |
| `relevance_enabled` | boolean | Enables intelligent heuristic scoring & indexing policy selection (default: `true`). |
| `analyze_threshold` | float | Minimum score (`[0.0, 1.0]`) to route visual asset to full Vision AI analysis (default: `0.65`). |
| `store_only_threshold` | float | Score below which low-value assets are marked `STORE_ONLY` and skip search indexing (default: `0.35`). |
| `ocr_only_threshold` | float | Minimum score to generate text chunk from OCR and Caption alone (default: `0.50`). |
| `max_vision_calls_per_document` | integer | Per-document safety budget for vision LLM API invocations (default: `20`). |
| `dedup_enabled` | boolean | Enables SHA-256 byte deduplication and analysis result caching (default: `true`). |
| `perceptual_hash_enabled` | boolean | Enables 64-bit gradient difference hash (`dHash`) detection for scaled/recompressed images (default: `true`). |
| `phash_hamming_distance_threshold` | integer | Maximum bit difference to treat two images as perceptual duplicates (default: `5`). |
| `storage_backend_type` | string | Storage backend for visual images: `minio` or `local` (default: `minio`). |
| `figure_ocr_enabled` | boolean | Runs targeted Tesseract OCR on candidate figures for technical keyword detection (default: `true`). |
| `vision.provider` | string | Vision AI provider: `openrouter/free` (auto-routing across free models), `openai`, `mock` (default: `openrouter/free`). |

### 🧠 `embedding`
| Field | Type | Description |
|---|---|---|
| `provider` | string | Embedding provider (`mock`, `openai`). |
| `model` | string | Model name (e.g. `text-embedding-3-small`). |
| `dimension` | integer | Vector dimension (e.g. `1536`). |
| `batch_size` | integer | Chunk batching size for API calls (default: `64`). |
| `auto_embed` | boolean | Automatically embed chunks during `parse_and_chunk` (default: `true`). |

### ⚙️ `pipeline`
| Field | Type | Description |
|---|---|---|
| `default_tenant_id` | string | Multi-tenancy isolation key if none is provided in the descriptor (default: `default`). |
| `handle_deletions` | boolean | If `true`, soft-deletes documents missing from source discovery crawls. |
| `force_fetch` | boolean | If `true`, re-downloads and verifies SHA-256 on all files even if timestamps match. |
| `auto_migrate` | boolean | Ensures database tables exist on startup. |

### 📝 `logging`
| Field | Type | Description |
|---|---|---|
| `level` | string | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `format` | string | Standard Python logging format string. |

---

## 3. How to Handle Environment Variables

The configuration loader automatically interpolates environment variables and values in `.env`:

1. **Syntax with fallback**: `${VARIABLE_NAME:-default_value}`
2. **Syntax without fallback**: `${VARIABLE_NAME}`
3. Secrets (e.g. passwords, DB URLs) can be kept in `.env` and referenced safely in `config.json`.

Example:
```json
"url": "${DATABASE_URL:-postgresql://rag_user:rag_password_123@127.0.0.1:5432/rag}"
```

---

## 4. How to Use in Code

Import `get_settings` anywhere in your code:

```python
from rag.core.config import get_settings

# Load configuration singleton
cfg = get_settings()

print(cfg.app.name)
print(cfg.database.url)
print(cfg.storage.raw_storage_dir)
print(cfg.chunking.chunk_size)

# Print a formatted safe summary (passwords automatically masked)
print(cfg.summary())
```

---

## 5. Custom Config Path

You can point to an alternative config file using the `RAG_CONFIG_PATH` environment variable:

```bash
export RAG_CONFIG_PATH="/path/to/production_config.json"
python example_ingest.py
```

Or programmatically:

```python
from rag.core.config import load_settings

custom_cfg = load_settings("/path/to/custom_config.json")
```

---

## 6. IDE Schema Autocomplete

The `$schema` key at the top of `config.json` points to `./config.schema.json`. This provides:
- Real-time autocomplete in VS Code and JetBrains IDEs.
- Validation warnings if a value violates constraints (e.g. `chunk_overlap >= chunk_size`).

To regenerate the schema after modifying models:
```bash
python -c "from rag.core.config import generate_json_schema; generate_json_schema('config.schema.json')"
```
