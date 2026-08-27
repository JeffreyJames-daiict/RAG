"""Canonical structured document representation for multimodal and multi-table parsing.

Defines an intermediate, type-safe object model preserving document structure:
- DocumentElement (HeadingElement, TextElement, TableElement)
- TableElement with raw grid, headers, rows, context hierarchy, and validation issues
- ParsedPage and ParsedDocument containers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ElementType(str, Enum):
    """Classification of structured document elements."""
    HEADING = "heading"
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


@dataclass
class TableValidationIssue:
    """Anomalies or formatting warnings detected during table extraction."""
    issue_type: str        # e.g. "COLUMN_COUNT_MISMATCH", "EMPTY_HEADER", "BROKEN_ROW"
    message: str
    severity: str = "warning"  # "warning" | "error"
    row_idx: Optional[int] = None
    col_idx: Optional[int] = None


@dataclass
class DocumentElement:
    """Base class for all discrete document elements in natural reading sequence."""
    element_id: str = ""
    element_type: ElementType = ElementType.TEXT
    page_number: int = 1  # 1-indexed
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    section: Optional[str] = None                             # Nearest preceding heading


@dataclass
class HeadingElement(DocumentElement):
    """A section heading or title."""
    element_type: ElementType = ElementType.HEADING
    content: str = ""
    level: int = 1


@dataclass
class TextElement(DocumentElement):
    """A paragraph or contiguous block of plain text."""
    element_type: ElementType = ElementType.TEXT
    content: str = ""


@dataclass
class TableElement(DocumentElement):
    """First-class structured table element with headers, grid rows, and context."""
    element_type: ElementType = ElementType.TABLE
    table_id: str = ""                                  # Deterministic: {doc_id}_p{page}_t{idx}
    caption: Optional[str] = None                       # Table title / caption if detected
    headers: List[str] = field(default_factory=list)    # Column names
    rows: List[List[str]] = field(default_factory=list) # Row cell values (list of string lists)
    num_rows: int = 0
    num_cols: int = 0
    context_hierarchy: List[str] = field(default_factory=list) # [Doc Title, Section, Sub-section]
    preceding_text: Optional[str] = None               # Contextual paragraph before table
    raw_markdown: Optional[str] = None                 # Markdown representation
    raw_html: Optional[str] = None                     # HTML representation
    validation_status: str = "valid"                   # "valid" | "warning" | "error"
    validation_issues: List[TableValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_start: int = 1                                # First page containing this table
    page_end: int = 1                                  # Last page (if continued across pages)
    is_continuation: bool = False                      # True if stitched from multi-page fragments
    parent_table_id: Optional[str] = None              # Root parent table ID
    continuation_confidence: float = 1.0               # Confidence score if merged

    def __post_init__(self):
        self.element_type = ElementType.TABLE
        if not self.page_start and self.page_number:
            self.page_start = self.page_number
        if not self.page_end and self.page_number:
            self.page_end = self.page_number
        if not self.parent_table_id and self.table_id:
            self.parent_table_id = self.table_id
        if not self.num_rows and self.rows:
            self.num_rows = len(self.rows)
        if not self.num_cols and self.headers:
            self.num_cols = len(self.headers)
        elif not self.num_cols and self.rows:
            self.num_cols = max(len(r) for r in self.rows)

    @property
    def row_dicts(self) -> List[Dict[str, str]]:
        """Return rows as list of column -> value dictionaries."""
        result = []
        for row in self.rows:
            row_dict = {}
            for i, val in enumerate(row):
                col_name = self.headers[i] if i < len(self.headers) else f"Column_{i+1}"
                row_dict[col_name] = val or ""
            result.append(row_dict)
        return result


# ── Backwards-Compatible Page & Block Containers ─────────────────────────────

@dataclass
class ParsedBlock:
    """A text or image block extracted from a page."""
    text: str
    bbox: tuple
    block_type: int = 0
    block_no: int = 0


@dataclass
class ParsedPage:
    """All content extracted from a single page."""
    page_number: int
    text: str
    blocks: List[ParsedBlock] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)
    tables: List[TableElement] = field(default_factory=list)
    elements: List[DocumentElement] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Canonical representation of an entire parsed document."""
    pages: List[ParsedPage] = field(default_factory=list)
    elements: List[DocumentElement] = field(default_factory=list)
    tables: List[TableElement] = field(default_factory=list)
    pdf_metadata: Dict[str, Any] = field(default_factory=dict)
    parser_version: str = "pymupdf-v2.0"
    table_diagnostics: Dict[str, Any] = field(
        default_factory=lambda: {
            "tables_detected": 0,
            "tables_stitched": 0,
            "tables_processed": 0,
            "tables_warning": 0,
            "tables_failed": 0,
        }
    )

    @property
    def full_text(self) -> str:
        """Concatenated text of all pages."""
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def total_tables(self) -> int:
        return len(self.tables)

