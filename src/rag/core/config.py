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
    allow_mock: bool = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_ALLOW_MOCK",
            "true" if os.getenv("APP_ENV", "development").lower() != "production" else "false"
        ).lower() in ("1", "true", "yes"),
        description="Explicitly allow mock embedding provider in non-production environments",
    )
    fail_on_provider_error: bool = Field(
        default_factory=lambda: os.getenv("EMBEDDING_FAIL_ON_PROVIDER_ERROR", "true").lower() in ("1", "true", "yes"),
        description="Fail immediately on provider error instead of falling back to dummy embeddings",
    )

    def safe_diagnostic_dict(self, is_production: bool = False) -> Dict[str, Any]:
        """Return safe, non-sensitive diagnostic dictionary of embedding configuration."""
        return {
            "embedding_provider": self.provider,
            "embedding_model": self.model,
            "embedding_model_version": self.model_version,
            "embedding_dimension": self.dimension,
            "embedding_api_base": self.endpoint_url or (
                "https://api.openai.com/v1/embeddings" if self.provider.lower() == "openai" else "none"
            ),
            "embedding_api_key_configured": bool(self.api_key),
            "embedding_allow_mock": self.allow_mock,
            "embedding_fail_on_provider_error": self.fail_on_provider_error,
            "mode": "production" if is_production else "development",
        }


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


class RelevanceWeightsConfig(BaseModel):
    """Configurable scoring weights for heuristic figure relevance classification."""
    caption: float = Field(default=0.20, description="Score boost when a figure caption is detected")
    figure_label: float = Field(default=0.10, description="Score boost for explicit 'Figure X / Fig. X' labels")
    technical_ocr: float = Field(default=0.20, description="Score boost when technical keywords/terms exist in OCR text")
    meaningful_ocr_length: float = Field(default=0.10, description="Score boost for substantial text inside figure (>= 5 words)")
    technical_context: float = Field(default=0.10, description="Score boost when surrounding section/paragraph is technical")
    informative_aspect_size: float = Field(default=0.05, description="Score boost for balanced aspect ratio and substantial area")
    large_page_coverage: float = Field(default=0.05, description="Score boost if figure occupies significant page portion (>= 20%)")
    repeated_asset_penalty: float = Field(default=-0.25, description="Score penalty if asset is repeated across multiple pages")
    logo_penalty: float = Field(default=-0.40, description="Score penalty if asset exhibits logo characteristics")
    watermark_penalty: float = Field(default=-0.50, description="Score penalty for low-contrast/watermark-like assets")
    tiny_icon_penalty: float = Field(default=-0.50, description="Score penalty for tiny icons / bullets")
    empty_ocr_penalty: float = Field(default=-0.05, description="Score penalty if OCR detected zero text")


class AssetConfig(BaseModel):
    """Configuration for document asset, figure, diagram, and image processing."""
    enabled: bool = Field(default=True, description="Enable visual asset / figure processing")
    storage_backend_type: Optional[str] = Field(
        default="minio",
        description="Dedicated storage provider for visual assets/figures: minio | s3 | database | local",
    )
    min_image_width: int = Field(default=64, ge=8, description="Minimum image width in pixels to consider as an asset")
    min_image_height: int = Field(default=64, ge=8, description="Minimum image height in pixels to consider as an asset")
    min_image_area: int = Field(default=4096, ge=64, description="Minimum image area in pixels (width * height)")
    max_aspect_ratio: float = Field(default=8.0, ge=1.0, description="Max aspect ratio threshold above which asset is flagged/skipped as a banner/border")
    filter_decorative: bool = Field(default=True, description="Filter out decorative icons, bullets, and backgrounds")
    caption_detection_enabled: bool = Field(default=True, description="Detect and associate figure captions")
    caption_vertical_distance_pts: float = Field(default=60.0, ge=5.0, le=300.0, description="Max vertical distance in points to associate caption with figure")
    figure_ocr_enabled: bool = Field(default=True, description="Run OCR on extracted figures to capture text elements")
    figure_ocr_min_confidence: float = Field(default=0.40, ge=0.0, le=1.0, description="Minimum confidence score for figure OCR words")
    relevance_enabled: bool = Field(default=True, description="Enable intelligent relevance scoring before vision analysis")
    analyze_threshold: float = Field(default=0.65, ge=0.0, le=1.0, description="Relevance score threshold >= which triggers Vision semantic analysis")
    store_only_threshold: float = Field(default=0.35, ge=0.0, le=1.0, description="Relevance score threshold < which asset is STORE_ONLY (no vision/embedding)")
    ocr_only_threshold: float = Field(default=0.50, ge=0.0, le=1.0, description="Relevance score threshold for OCR_ONLY indexing if vision is skipped")
    max_vision_calls_per_document: int = Field(default=20, ge=1, le=100, description="Safety budget on max vision API calls per document")
    dedup_enabled: bool = Field(default=True, description="Enable exact SHA-256 and perceptual hash deduplication")
    perceptual_hash_enabled: bool = Field(default=True, description="Enable 64-bit perceptual hashing (pHash/dHash) for visual deduplication")
    phash_hamming_distance_threshold: int = Field(default=5, ge=0, le=64, description="Max hamming distance to treat two images as perceptual duplicates")
    weights: RelevanceWeightsConfig = Field(default_factory=RelevanceWeightsConfig, description="Relevance scoring feature weights")
    vision: VisionConfig = Field(default_factory=VisionConfig, description="Vision analysis settings")


class RerankingConfig(BaseModel):
    """Validated configuration for reranking candidates after hybrid retrieval and RRF fusion."""
    enabled: bool = Field(
        default_factory=lambda: os.getenv("RERANKING_ENABLED", "true").lower() in ("1", "true", "yes"),
        description="Enable reranking stage after RRF fusion",
    )
    provider: str = Field(
        default_factory=lambda: os.getenv("RERANKER_PROVIDER", "mock"),
        description="Reranker provider: mock | cross_encoder | openrouter",
    )
    model: str = Field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "mock-reranker"),
        description="Reranker model name or checkpoint",
    )
    model_version: str = Field(
        default_factory=lambda: os.getenv("RERANKER_MODEL_VERSION", "v1.0"),
        description="Model version string for telemetry and caching",
    )
    top_k: int = Field(
        default_factory=lambda: int(os.getenv("RERANKER_TOP_K", "5")),
        ge=1,
        le=1000,
        description="Number of top candidates returned after reranking",
    )
    candidate_k: int = Field(
        default_factory=lambda: int(os.getenv("RERANKER_CANDIDATE_K", "50")),
        ge=1,
        le=1000,
        description="Number of candidate chunks fed to the reranker",
    )
    timeout_sec: float = Field(
        default_factory=lambda: float(os.getenv("RERANKER_TIMEOUT_SECONDS", "10.0")),
        ge=0.1,
        le=120.0,
        description="Timeout budget for reranking call in seconds",
    )
    batch_size: int = Field(
        default_factory=lambda: int(os.getenv("RERANKER_BATCH_SIZE", "25")),
        ge=1,
        le=500,
        description="Batch size for reranker candidate scoring",
    )
    max_document_length: int = Field(
        default_factory=lambda: int(os.getenv("RERANKER_MAX_DOCUMENT_LENGTH", "1024")),
        ge=64,
        le=32000,
        description="Max character length per document before truncation",
    )
    max_query_length: int = Field(
        default_factory=lambda: int(os.getenv("RERANKER_MAX_QUERY_LENGTH", "512")),
        ge=1,
        le=4000,
        description="Max character length for query before truncation/validation",
    )
    fallback_enabled: bool = Field(
        default_factory=lambda: os.getenv("RERANKER_FALLBACK_ENABLED", "true").lower() in ("1", "true", "yes"),
        description="Fallback to RRF results on reranking provider failure",
    )
    cache_enabled: bool = Field(
        default_factory=lambda: os.getenv("RERANKER_CACHE_ENABLED", "false").lower() in ("1", "true", "yes"),
        description="Enable in-memory LRU cache for reranking results",
    )
    cache_max_size: int = Field(
        default_factory=lambda: int(os.getenv("RERANKER_CACHE_MAX_SIZE", "256")),
        ge=1,
        le=10000,
        description="Max size of LRU cache",
    )


class RetrievalConfig(BaseModel):
    """Configuration for vector retrieval, ranking, and metadata filtering."""
    reranking: RerankingConfig = Field(
        default_factory=RerankingConfig,
        description="Reranking configuration for post-fusion scoring",
    )
    enabled: bool = Field(
        default_factory=lambda: os.getenv("VECTOR_RETRIEVAL_ENABLED", "true").lower() in ("1", "true", "yes"),
        description="Enable vector retrieval subsystem",
    )
    backend: str = Field(
        default_factory=lambda: os.getenv("VECTOR_BACKEND", "auto"),
        description="Vector retrieval backend: auto | pgvector | python",
    )
    default_top_k: int = Field(
        default_factory=lambda: int(os.getenv("VECTOR_TOP_K", "5")),
        ge=1,
        le=100,
        description="Default number of chunks returned",
    )
    max_top_k: int = Field(default=100, ge=1, le=1000, description="Hard ceiling for top_k parameter")
    similarity_threshold: Optional[float] = Field(
        default_factory=lambda: float(os.getenv("VECTOR_SIMILARITY_THRESHOLD")) if os.getenv("VECTOR_SIMILARITY_THRESHOLD") else None,
        ge=-1.0,
        le=1.0,
        description="Minimum cosine similarity threshold to qualify result (None disables filter)",
    )
    candidate_limit: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Max candidate embeddings evaluated during search",
    )
    distance_metric: str = Field(
        default_factory=lambda: os.getenv("VECTOR_DISTANCE_METRIC", "cosine"),
        description="Vector distance metric: cosine | l2 | inner_product",
    )
    index_type: str = Field(
        default_factory=lambda: os.getenv("VECTOR_INDEX_TYPE", "hnsw"),
        description="Vector index structure: hnsw | ivfflat | none",
    )
    allow_inactive_documents: bool = Field(
        default=False,
        description="Whether retrieval includes soft-deleted or non-indexed documents",
    )
    allow_python_fallback: bool = Field(
        default=True,
        description="Allow Python in-memory vector similarity when native pgvector is unavailable",
    )
    strict_embedding_validation: bool = Field(
        default=True,
        description="Strictly validate query and stored embedding dimensions and finiteness",
    )
    query_timeout_sec: float = Field(
        default_factory=lambda: float(os.getenv("VECTOR_SEARCH_TIMEOUT_SECONDS", "10.0")),
        ge=0.5,
        le=120.0,
        description="Timeout for query execution in seconds",
    )
    max_query_length: int = Field(
        default_factory=lambda: int(os.getenv("VECTOR_MAX_QUERY_LENGTH", "4000")),
        ge=1,
        le=32000,
        description="Maximum allowed character length for natural language search queries",
    )
    embedding_model: Optional[str] = Field(
        default_factory=lambda: os.getenv("VECTOR_EMBEDDING_MODEL"),
        description="Override target embedding model for retrieval",
    )
    embedding_dimensions: Optional[int] = Field(
        default_factory=lambda: int(os.getenv("VECTOR_EMBEDDING_DIMENSIONS", "1536")) if os.getenv("VECTOR_EMBEDDING_DIMENSIONS") else 1536,
        ge=1,
        description="Expected vector dimensionality",
    )
    max_metadata_filter_size: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Maximum number of criteria allowed in metadata filter",
    )
    hybrid_enabled: bool = Field(default=True, description="Enable hybrid search combining vector + BM25")
    rrf_enabled: bool = Field(
        default_factory=lambda: os.getenv("RRF_ENABLED", "true").lower() in ("1", "true", "yes"),
        description="Master switch for Reciprocal Rank Fusion (disable to bypass fusion and return dense results only)",
    )
    rrf_k: int = Field(default=60, ge=1, le=1000, description="Reciprocal Rank Fusion smoothing constant k")
    rrf_dense_weight: float = Field(
        default_factory=lambda: float(os.getenv("RRF_DENSE_WEIGHT", "1.0")),
        ge=0.0,
        le=100.0,
        description="RRF weight multiplier for dense vector retrieval source",
    )
    rrf_sparse_weight: float = Field(
        default_factory=lambda: float(os.getenv("RRF_SPARSE_WEIGHT", "1.0")),
        ge=0.0,
        le=100.0,
        description="RRF weight multiplier for sparse BM25 retrieval source",
    )
    rrf_top_k: int = Field(
        default_factory=lambda: int(os.getenv("RRF_TOP_K", "50")),
        ge=1,
        le=1000,
        description="Maximum number of fused candidates retained after RRF fusion before final_top_k trim",
    )
    rrf_source_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Extensible per-source weight map for future retriever sources (e.g. table_retriever: 0.8)",
    )
    bm25_k1: float = Field(default=1.5, ge=0.0, le=10.0, description="BM25 term frequency saturation parameter")
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0, description="BM25 document length normalization parameter")
    dense_top_k: int = Field(
        default_factory=lambda: int(os.getenv("DENSE_TOP_K", "50")),
        ge=1,
        le=1000,
        description="Number of candidate chunks retrieved from dense vector search before fusion",
    )
    sparse_top_k: int = Field(
        default_factory=lambda: int(os.getenv("SPARSE_TOP_K", "50")),
        ge=1,
        le=1000,
        description="Number of candidate chunks retrieved from sparse BM25 search before fusion",
    )
    hybrid_candidate_k: int = Field(
        default_factory=lambda: int(os.getenv("HYBRID_CANDIDATE_K", "50")),
        ge=1,
        le=1000,
        description="Ceiling for candidate pool evaluated during hybrid fusion",
    )
    final_top_k: int = Field(
        default_factory=lambda: int(os.getenv("FINAL_TOP_K", "5")),
        ge=1,
        le=1000,
        description="Default final number of chunks returned after fusion",
    )
    hybrid_dense_timeout_sec: float = Field(
        default_factory=lambda: float(os.getenv("HYBRID_DENSE_TIMEOUT_SEC", "10.0")),
        ge=0.1,
        le=120.0,
        description="Timeout budget for dense vector retrieval branch in hybrid search",
    )
    hybrid_sparse_timeout_sec: float = Field(
        default_factory=lambda: float(os.getenv("HYBRID_SPARSE_TIMEOUT_SEC", "10.0")),
        ge=0.1,
        le=120.0,
        description="Timeout budget for sparse BM25 retrieval branch in hybrid search",
    )
    allow_partial_failure: bool = Field(
        default_factory=lambda: os.getenv("HYBRID_ALLOW_PARTIAL_FAILURE", "true").lower() in ("1", "true", "yes"),
        description="Allow returning results from single retriever if the other branch fails",
    )
    bm25: "BM25Config" = Field(default_factory=lambda: BM25Config(), description="BM25 sparse keyword retrieval configuration")

    def get_rrf_weights(self) -> Dict[str, float]:
        """Build the merged per-source weight map for RRF fusion.

        Priority: explicit rrf_source_weights > rrf_dense_weight/rrf_sparse_weight > default 1.0
        """
        weights: Dict[str, float] = {
            "vector": self.rrf_dense_weight,
            "bm25": self.rrf_sparse_weight,
        }
        weights.update(self.rrf_source_weights)
        return weights

    @model_validator(mode="after")
    def _validate_rrf_weights(self) -> "RetrievalConfig":
        all_weights = self.get_rrf_weights()
        for name, w in all_weights.items():
            if w < 0:
                raise ValueError(f"RRF weight for source '{name}' must be >= 0, got {w}")
        if all(w == 0.0 for w in all_weights.values()):
            raise ValueError("At least one RRF source weight must be > 0")
        return self


class BM25Config(BaseModel):
    """Validated configuration for BM25 sparse keyword retrieval."""
    enabled: bool = Field(
        default_factory=lambda: os.getenv("BM25_ENABLED", "true").lower() in ("1", "true", "yes"),
        description="Enable BM25 sparse keyword retrieval subsystem",
    )
    top_k: int = Field(
        default_factory=lambda: int(os.getenv("BM25_TOP_K", "5")),
        ge=1,
        le=1000,
        description="Default number of chunks returned by BM25 retrieval",
    )
    k1: float = Field(
        default_factory=lambda: float(os.getenv("BM25_K1", "1.5")),
        ge=0.0,
        le=10.0,
        description="BM25 term frequency saturation parameter k1",
    )
    b: float = Field(
        default_factory=lambda: float(os.getenv("BM25_B", "0.75")),
        ge=0.0,
        le=1.0,
        description="BM25 document length normalization parameter b",
    )
    min_score: float = Field(
        default_factory=lambda: float(os.getenv("BM25_MIN_SCORE", "0.0")),
        ge=0.0,
        description="Minimum score cutoff for BM25 candidates",
    )
    index_refresh_interval: int = Field(
        default_factory=lambda: int(os.getenv("BM25_INDEX_REFRESH_INTERVAL", "300")),
        ge=0,
        description="Automatic in-memory index refresh interval in seconds (0 to disable)",
    )
    title_weight: float = Field(
        default_factory=lambda: float(os.getenv("BM25_TITLE_WEIGHT", "2.5")),
        ge=0.0,
        description="Field weight multiplier for document/chunk titles",
    )
    heading_weight: float = Field(
        default_factory=lambda: float(os.getenv("BM25_HEADING_WEIGHT", "2.0")),
        ge=0.0,
        description="Field weight multiplier for section headings",
    )
    body_weight: float = Field(
        default_factory=lambda: float(os.getenv("BM25_BODY_WEIGHT", "1.0")),
        ge=0.0,
        description="Field weight multiplier for main chunk body text",
    )
    table_weight: float = Field(
        default_factory=lambda: float(os.getenv("BM25_TABLE_WEIGHT", "1.5")),
        ge=0.0,
        description="Field weight multiplier for structured table chunks",
    )
    caption_weight: float = Field(
        default_factory=lambda: float(os.getenv("BM25_CAPTION_WEIGHT", "1.2")),
        ge=0.0,
        description="Field weight multiplier for figure and diagram captions",
    )



class ContextBuilderConfig(BaseModel):
    """Configuration for LLM context assembly, budgeting, expansion, and isolation."""
    enabled: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_BUILDER_ENABLED", "true").lower() in ("1", "true", "yes"),
        description="Enable structured context builder subsystem",
    )
    token_budget: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_TOKEN_BUDGET", "4000")),
        ge=128,
        le=128000,
        description="Maximum total token ceiling for assembled context",
    )
    max_tokens_per_document: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_MAX_TOKENS_PER_DOC", "2000")),
        ge=64,
        le=64000,
        description="Maximum tokens allowed from a single document",
    )
    max_tokens_per_section: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_MAX_TOKENS_PER_SECTION", "1000")),
        ge=32,
        le=32000,
        description="Maximum tokens allowed from a single section",
    )
    min_content_preservation_tokens: int = Field(
        default=150,
        ge=16,
        description="Minimum token threshold to preserve for top-ranked chunk",
    )
    overlap_threshold: float = Field(
        default=0.85,
        ge=0.1,
        le=1.0,
        description="Jaccard token overlap threshold to treat chunks as duplicates",
    )
    expand_neighbors: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_EXPAND_NEIGHBORS", "true").lower() in ("1", "true", "yes"),
        description="Expand seed chunks with adjacent (previous/next) chunks",
    )
    expand_parents: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_EXPAND_PARENTS", "true").lower() in ("1", "true", "yes"),
        description="Expand seed chunks with parent chunk content if available",
    )
    max_expansion_count: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Maximum additional chunks to expand per retrieved seed chunk",
    )
    max_chunks_per_document: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Ceiling for total chunks included per document",
    )
    tokenizer_encoding: str = Field(
        default_factory=lambda: os.getenv("CONTEXT_TOKENIZER_ENCODING", "cl100k_base"),
        description="Tiktoken encoding name (e.g. cl100k_base, o200k_base)",
    )
    injection_defense_enabled: bool = Field(
        default=True,
        description="Sanitize prompt injection attempts and enforce XML boundary fences",
    )
    citation_prefix: str = Field(
        default="C",
        description="Prefix for citation tags, e.g. 'C' generates [C1], [C2]",
    )


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
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    context: ContextBuilderConfig = Field(default_factory=ContextBuilderConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


    # Convenient shortcuts
    @property
    def database_url(self) -> str:
        return self.database.url

    @property
    def bm25(self) -> BM25Config:
        return self.retrieval.bm25

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
