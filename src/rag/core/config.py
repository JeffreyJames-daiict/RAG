"""Centralised JSON-based configuration management for Enterprise RAG.

Reads from `config.json` with support for:
1. Environment variable interpolation (e.g. `${DATABASE_URL:-sqlite:///rag.db}`).
2. `.env` file automatic fallback.
3. Strict type validation and sensible defaults via Pydantic v2.
4. Exportable JSON Schema for IDE autocomplete and validation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Pattern for ${VAR_NAME} or ${VAR_NAME:-default_value}
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:-default} in string values."""
    if isinstance(data, str):
        def _replace_match(match: re.Match) -> str:
            var_name = match.group(1)
            default_val = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default_val)

        return _ENV_VAR_PATTERN.sub(_replace_match, data)
    elif isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    return data


# ── Configuration Sub-Models ──────────────────────────────────────────────────

class AppConfig(BaseModel):
    """Application metadata and environment."""
    name: str = Field(default="Enterprise RAG Pipeline", description="Human-readable application name")
    version: str = Field(default="0.1.0", description="Semantic version string")
    env: str = Field(default="development", description="Environment: development | staging | production")

    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"


class DatabaseConfig(BaseModel):
    """Database connection and pooling parameters."""
    url: str = Field(
        default="sqlite:///rag.db",
        description="SQLAlchemy connection URI (e.g. postgresql://user:pass@host:5432/rag)"
    )
    pool_size: int = Field(default=5, ge=1, description="Connection pool size for persistent connections")
    max_overflow: int = Field(default=10, ge=0, description="Max overflow connections allowed")
    echo: bool = Field(default=False, description="Log all generated SQL queries")

    def safe_url(self) -> str:
        """Return connection URL with password redacted."""
        url = self.url
        if "@" in url:
            scheme_creds, host_part = url.rsplit("@", 1)
            if ":" in scheme_creds.split("//", 1)[-1]:
                scheme, rest = scheme_creds.split("//", 1)
                user, _ = rest.rsplit(":", 1)
                return f"{scheme}//{user}:***@{host_part}"
        return url


class StorageConfig(BaseModel):
    """Immutable raw document byte store settings."""
    backend_type: str = Field(
        default="database",
        description="Storage provider (e.g. database | local | s3 | minio)"
    )
    raw_schema: str = Field(
        default="raw",
        description="Database schema name for raw document payload storage"
    )
    raw_storage_dir: str = Field(
        default="./raw_store",
        description="Filesystem root where raw original documents are stored (when backend_type=local)"
    )
    verify_on_read: bool = Field(
        default=True,
        description="Validate content against sha256 checksum on read"
    )
    # S3 / MinIO settings (used when backend_type='s3' or 'minio')
    s3_bucket: Optional[str] = Field(
        default=None,
        description="S3 / MinIO bucket name for raw payload storage"
    )
    s3_endpoint_url: Optional[str] = Field(
        default=None,
        description="S3-compatible endpoint URL (e.g. http://127.0.0.1:9000 for MinIO)"
    )
    s3_access_key: Optional[str] = Field(
        default=None,
        description="S3 / MinIO access key ID"
    )
    s3_secret_key: Optional[str] = Field(
        default=None,
        description="S3 / MinIO secret access key"
    )
    s3_region: Optional[str] = Field(
        default=None,
        description="AWS region name (not required for MinIO)"
    )
    s3_storage_prefix: str = Field(
        default="storage",
        description="Top-level prefix inside the S3 bucket for pipeline artifacts (separate from source documents)"
    )

    @property
    def resolved_path(self) -> Path:
        return Path(self.raw_storage_dir).resolve()


class ParserConfig(BaseModel):
    """Document extraction and parsing settings."""
    version: str = Field(default="pymupdf-v1.0", description="Parser version identifier recorded on snapshots")
    extract_images: bool = Field(default=False, description="Extract and annotate image blocks during PDF parse")
    supported_mime_types: List[str] = Field(
        default_factory=lambda: ["application/pdf", "application/x-pdf"],
        description="List of MIME types handled by the parser"
    )

    @property
    def mime_set(self) -> FrozenSet[str]:
        return frozenset(self.supported_mime_types)


class ChunkingConfigModel(BaseModel):
    """Token-aware recursive text splitting settings."""
    chunk_size: int = Field(default=512, ge=32, le=8192, description="Target token budget per chunk")
    chunk_overlap: int = Field(default=64, ge=0, description="Token overlap between adjacent chunks")
    chunking_version: str = Field(default="v1.0", description="Strategy version tag stored on chunk records")
    encoding_model: str = Field(default="cl100k_base", description="tiktoken encoding model")

    @model_validator(mode="after")
    def _validate_overlap(self) -> "ChunkingConfigModel":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})")
        return self


class TableConfig(BaseModel):
    """Configuration for multi-table detection, serialization, and chunking."""
    enabled: bool = Field(default=True, description="Enable structured table extraction and processing")
    max_rows_per_chunk: int = Field(default=15, ge=1, le=500, description="Max rows grouped into a single table chunk")
    serialization_strategy: str = Field(
        default="hybrid",
        description="Table serialization format: hybrid | markdown | row_based | json"
    )
    summary_enabled: bool = Field(default=True, description="Generate a table-level summary chunk for semantic search")
    parent_child_enabled: bool = Field(default=True, description="Maintain parent_id linking child chunks to parent table")
    wide_table_threshold: int = Field(default=8, ge=2, description="Column count threshold for wide table column grouping")
    validation_enabled: bool = Field(default=True, description="Run structural and consistency checks on extracted tables")
    multipage_continuation_enabled: bool = Field(
        default=True,
        description="Detect and stitch multi-page table continuations into a unified table element"
    )
    continuation_confidence_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required to merge adjacent page tables"
    )
    cell_inheritance_enabled: bool = Field(
        default=True,
        description="Forward-fill empty category cells in hierarchical row structures"
    )
    hierarchical_headers_enabled: bool = Field(
        default=True,
        description="Detect and normalize multi-level/hierarchical header rows into composite column names"
    )
    parent_expansion_neighbor_rows: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Number of neighboring rows included before and after a matched chunk during parent expansion"
    )
    max_expansion_rows: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Hard upper ceiling on total rows returned during parent table expansion"
    )



class OCRConfig(BaseModel):
    """Configuration for optical character recognition on scanned and mixed PDF pages."""
    enabled: bool = Field(default=True, description="Enable OCR processing for scanned/mixed pages")
    provider: str = Field(default="tesseract", description="OCR provider name: tesseract | mock")
    language: str = Field(default="eng", description="OCR language code")
    min_native_text_quality: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Native text quality threshold below which OCR is triggered"
    )
    min_confidence: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum OCR word confidence score required to accept result"
    )
    max_retries: int = Field(default=2, ge=0, le=5, description="Max OCR retries with enhanced preprocessing")
    render_dpi: int = Field(default=300, ge=72, le=600, description="DPI resolution for rendering PDF pages to images")
    default_preprocessing: str = Field(
        default="standard",
        description="Default image preprocessing profile: minimal | standard | aggressive"
    )
    mixed_content_merge: bool = Field(
        default=True,
        description="Safely merge non-overlapping OCR content with native text on mixed pages"
    )
    iou_overlap_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Spatial overlap threshold above which OCR text is suppressed to avoid duplicates"
    )


class PipelineConfig(BaseModel):
    """Ingestion and sync pipeline behavior."""
    default_tenant_id: str = Field(default="default", description="Tenant key when omitted in payloads")
    handle_deletions: bool = Field(default=True, description="Soft-delete DB documents missing from source")
    force_fetch: bool = Field(default=False, description="Force re-fetching payloads regardless of timestamp/hash")
    auto_migrate: bool = Field(default=True, description="Ensure database tables exist on startup")


class NormalizationConfig(BaseModel):
    """Configuration for the content normalization layer between extraction and chunking."""
    enabled: bool = Field(
        default=True,
        description="Master switch for the normalization layer",
    )
    min_text_length: int = Field(
        default=10,
        ge=0,
        description="Drop blocks shorter than this character count",
    )
    min_word_count: int = Field(
        default=3,
        ge=0,
        description="Drop blocks with fewer words than this threshold",
    )
    ocr_confidence_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Flag OCR blocks with confidence below this value as low_confidence",
    )
    duplicate_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Character n-gram overlap ratio above which two blocks are considered duplicates",
    )
    header_repeat_threshold: int = Field(
        default=3,
        ge=2,
        description="Minimum number of pages on which a pattern must appear to be classified as a header",
    )
    footer_repeat_threshold: int = Field(
        default=3,
        ge=2,
        description="Minimum number of pages on which a pattern must appear to be classified as a footer",
    )
    max_symbol_ratio: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Flag blocks where non-alphanumeric symbols exceed this proportion of total characters",
    )
    excessive_repetition_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Flag blocks where a single character or n-gram accounts for more than this ratio",
    )
    unicode_normalization_form: str = Field(
        default="NFC",
        description="Unicode normalization form to apply: NFC | NFKC | none",
    )
    repair_soft_hyphens: bool = Field(
        default=True,
        description="Attempt to rejoin words split by PDF soft hyphens at end-of-line",
    )
    remove_headers: bool = Field(
        default=True,
        description="Remove detected repetitive page headers",
    )
    remove_footers: bool = Field(
        default=True,
        description="Remove detected repetitive page footers",
    )
    remove_standalone_page_numbers: bool = Field(
        default=True,
        description="Remove standalone page number artifacts in extracted text",
    )


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation and vector persistence."""
    enabled: bool = Field(default=True, description="Enable embedding generation during ingestion")
    provider: str = Field(default="mock", description="Embedding provider: mock | openai | fastembed")
    model: str = Field(default="text-embedding-3-small", description="Embedding model identifier")
    model_version: str = Field(default="v1.0", description="Model configuration version tag")
    dimension: int = Field(default=1536, ge=1, description="Expected vector dimensionality")
    batch_size: int = Field(default=64, ge=1, le=2048, description="Batch size for embedding requests")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retries for transient provider failures")
    retry_backoff_sec: float = Field(default=1.0, ge=0.1, le=60.0, description="Initial retry backoff delay in seconds")
    timeout_sec: float = Field(default=30.0, ge=1.0, le=300.0, description="Provider request timeout in seconds")
    max_input_tokens: int = Field(default=8191, ge=32, description="Max token budget accepted by the model")
    oversized_chunk_strategy: str = Field(
        default="truncate",
        description="Handling for oversized chunks: truncate | reject | error"
    )
    api_key: Optional[str] = Field(default=None, description="Provider API key (e.g. OPENAI_API_KEY)")
    endpoint_url: Optional[str] = Field(default=None, description="Custom API endpoint URL for self-hosted/proxy")


class VisionConfig(BaseModel):
    """Configuration for multimodal vision and diagram semantic analysis."""
    enabled: bool = Field(default=True, description="Enable vision-based semantic analysis for diagrams/figures")
    provider: str = Field(default="mock", description="Vision provider: mock | openai")
    model: str = Field(default="gpt-4o", description="Vision model identifier")
    api_key: Optional[str] = Field(default=None, description="Vision provider API key")
    endpoint_url: Optional[str] = Field(default=None, description="Custom vision API endpoint URL")
    timeout_sec: float = Field(default=45.0, ge=1.0, le=300.0, description="Vision request timeout in seconds")
    max_retries: int = Field(default=2, ge=0, le=5, description="Max retries for transient vision errors")
    retry_backoff_sec: float = Field(default=1.0, ge=0.1, le=60.0, description="Initial retry backoff delay")
    max_image_dimension: int = Field(default=2048, ge=256, le=4096, description="Max image width/height sent to vision API")
    extract_relationships: bool = Field(default=True, description="Extract structured entity relationships/connections")


class AssetConfig(BaseModel):
    """Configuration for document asset, figure, diagram, and image processing."""
    enabled: bool = Field(default=True, description="Enable visual asset / figure processing")
    min_image_width: int = Field(default=64, ge=8, description="Minimum image width in pixels to consider as an asset")
    min_image_height: int = Field(default=64, ge=8, description="Minimum image height in pixels to consider as an asset")
    min_image_area: int = Field(default=4096, ge=64, description="Minimum image area in pixels (width * height)")
    max_aspect_ratio: float = Field(default=8.0, ge=1.0, description="Max aspect ratio threshold above which asset is flagged/skipped as a banner/border")
    filter_decorative: bool = Field(default=True, description="Filter out decorative icons, bullets, and backgrounds")
    caption_detection_enabled: bool = Field(default=True, description="Detect and associate figure captions")
    caption_vertical_distance_pts: float = Field(default=60.0, ge=5.0, le=300.0, description="Max vertical distance in points to associate caption with figure")
    figure_ocr_enabled: bool = Field(default=True, description="Run OCR on extracted figures to capture text elements")
    figure_ocr_min_confidence: float = Field(default=0.40, ge=0.0, le=1.0, description="Minimum confidence score for figure OCR words")
    vision: VisionConfig = Field(default_factory=VisionConfig, description="Vision analysis settings")


class LoggingConfig(BaseModel):
    """Logging format and level."""
    level: str = Field(default="INFO", description="Logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Standard logging format string"
    )


# ── Root Settings Container ──────────────────────────────────────────────────

class Settings(BaseModel):
    """Master configuration class representing the full `config.json` schema."""

    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    chunking: ChunkingConfigModel = Field(default_factory=ChunkingConfigModel)
    table: TableConfig = Field(default_factory=TableConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    asset: AssetConfig = Field(default_factory=AssetConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


    # Convenient shortcuts
    @property
    def database_url(self) -> str:
        return self.database.url

    @property
    def raw_storage_dir(self) -> str:
        return self.storage.raw_storage_dir

    @property
    def chunk_size(self) -> int:
        return self.chunking.chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self.chunking.chunk_overlap

    @property
    def chunking_version(self) -> str:
        return self.chunking.chunking_version

    @property
    def default_tenant_id(self) -> str:
        return self.pipeline.default_tenant_id

    @property
    def pdf_supported_mime_types(self) -> FrozenSet[str]:
        return self.parser.mime_set

    def setup_logging(self) -> None:
        """Apply logging configuration to the application."""
        numeric_level = getattr(logging, self.logging.level.upper(), logging.INFO)
        logging.basicConfig(level=numeric_level, format=self.logging.format, force=True)
        logging.getLogger("rag").setLevel(numeric_level)

    def summary(self) -> str:
        """Format a human-readable, safe summary of the loaded settings."""
        return (
            f"\n{'=' * 60}\n"
            f" ⚙️  {self.app.name} v{self.app.version} [{self.app.env.upper()}]\n"
            f"{'=' * 60}\n"
            f"  Database:        {self.database.safe_url()}\n"
            f"  Pool / Overflow: {self.database.pool_size} / {self.database.max_overflow}\n"
            f"  Raw Storage:     {self.storage.resolved_path} ({self.storage.backend_type})\n"
            f"  Verify on Read:  {self.storage.verify_on_read}\n"
            f"  Default Tenant:  {self.pipeline.default_tenant_id}\n"
            f"  Chunk Config:    {self.chunking.chunk_size} tokens (overlap: {self.chunking.chunk_overlap})\n"
            f"  tiktoken Model:  {self.chunking.encoding_model}\n"
            f"  Parser Version:  {self.parser.version}\n"
            f"  MIME Types:      {', '.join(sorted(self.parser.supported_mime_types))}\n"
            f"  Log Level:       {self.logging.level}\n"
            f"{'=' * 60}"
        )


# ── Loader & Factory Functions ───────────────────────────────────────────────

def find_config_file(explicit_path: Optional[str | Path] = None) -> Path:
    """Find the config.json file in standard locations."""
    if explicit_path:
        p = Path(explicit_path).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified configuration file not found: {explicit_path}")

    # Check env var override
    env_path = os.getenv("RAG_CONFIG_PATH")
    if env_path:
        p = Path(env_path).resolve()
        if p.exists():
            return p

    # Standard lookup: configs/ dir, current working directory, or repository root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    candidates = [
        Path.cwd() / "configs" / "config.json",
        Path.cwd() / "config.json",
        repo_root / "configs" / "config.json",
        repo_root / "config.json",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Fallback default path in configs/
    return Path.cwd() / "configs" / "config.json"



def load_settings(config_path: Optional[str | Path] = None) -> Settings:
    """Load configuration from config.json with env interpolation and .env loading."""
    # 1. Load .env file first so variables are available for expansion
    load_dotenv()

    # 2. Locate config file
    path = find_config_file(config_path)

    if not path.exists():
        logger.warning("Configuration file not found at %s. Using default settings.", path)
        settings = Settings()
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)

            # Expand environment variable interpolations (${VAR} or ${VAR:-default})
            expanded_data = _expand_env_vars(raw_json)

            settings = Settings.model_validate(expanded_data)
            logger.info("Loaded configuration from %s", path)
        except Exception as exc:
            logger.error("Failed to parse %s: %s. Falling back to default settings.", path, exc)
            raise

    settings.setup_logging()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return load_settings()


def generate_json_schema(output_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Generate and optionally save the JSON Schema for config.json."""
    schema = Settings.model_json_schema()
    if output_path:
        out = Path(output_path)
        out.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        logger.info("Saved config JSON schema to %s", out)
    return schema
