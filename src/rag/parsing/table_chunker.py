"""Advanced table-aware chunking engine with parent-child tracking, wide table slicing, and token-safe serialization.

Features:
- Token-bounded row grouping (never splitting mid-row).
- Wide table column slicing (preserves primary anchor entity column across all slices).
- Multi-page table continuation metadata propagation.
- Context injection (repeats table caption, headers, page range, and row range).
- Parent-child metadata linking.
- Token-safe table summary generation.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import tiktoken

from rag.core.config import TableConfig, get_settings
from rag.core.models import ChunkSchema
from rag.parsing.models import TableElement
from rag.parsing.table_processor import TableSerializer, TableValidator, generate_table_summary

logger = logging.getLogger(__name__)


class TableChunker:
    """Intelligently chunks TableElement instances into self-contained ChunkSchema objects."""

    def __init__(
        self,
        table_config: Optional[TableConfig] = None,
        encoding_model: str = "cl100k_base",
        max_tokens_per_chunk: int = 512,
        chunking_version: str = "v1.0",
    ):
        settings = get_settings()
        self.table_config = table_config or settings.table
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.chunking_version = chunking_version
        self.encoding_model = encoding_model

        try:
            self._enc = tiktoken.get_encoding(self.encoding_model)
        except Exception:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def _token_len(self, text: str) -> int:
        return len(self._enc.encode(text))

    def chunk_table(
        self,
        table: TableElement,
        document_id: str,
        version: int,
        document_title: Optional[str] = None,
        starting_chunk_index: int = 0,
    ) -> List[ChunkSchema]:
        """Convert a TableElement into one or more ChunkSchema objects.

        Handles small tables, large tables (row batching), wide tables (column grouping),
        and multi-page continuation tables.
        """
        # Validate table if validation is enabled
        if self.table_config.validation_enabled:
            table = TableValidator.validate(
                table,
                enable_hierarchical_headers=self.table_config.hierarchical_headers_enabled,
                enable_cell_inheritance=self.table_config.cell_inheritance_enabled,
            )

        chunks: List[ChunkSchema] = []
        current_index = starting_chunk_index
        total_rows = len(table.rows)
        total_cols = table.num_cols

        # ── Empty / Header-Only Table Case ───────────────────────────────────
        if total_rows == 0:
            serialized = (
                f"Table: {table.caption or table.table_id} (Empty/Header-only)\n"
                f"Columns: {', '.join(table.headers)}"
            )
            meta: Dict[str, Any] = {
                "chunk_type": "table",
                "table_id": table.table_id,
                "parent_id": table.parent_table_id or table.table_id,
                "page": table.page_number,
                "page_start": table.page_start,
                "page_end": table.page_end,
                "is_continuation": table.is_continuation,
                "section": table.section,
                "row_start": 0,
                "row_end": 0,
                "total_rows": 0,
                "total_columns": total_cols,
                "columns": table.headers,
                "validation_status": table.validation_status,
                "token_count": self._token_len(serialized),
            }
            chunks.append(
                ChunkSchema(
                    chunk_id=uuid.uuid4(),
                    document_id=uuid.UUID(document_id),
                    version=version,
                    chunk_index=current_index,
                    text=serialized,
                    section=table.section,
                    page=table.page_number,
                    metadata=meta,
                    chunking_version=self.chunking_version,
                )
            )
            return chunks

        # ── Wide Table Slicing Determination ─────────────────────────────────
        # If columns exceed threshold, slice columns while preserving anchor Column 0
        if total_cols > self.table_config.wide_table_threshold:
            column_slices = self._slice_wide_columns(table)
        else:
            column_slices = [(table, None)]

        # ── Generate Table Summary Chunk if multi-chunk & enabled ────────────
        # Summary is generated once for the whole table (before slicing)
        is_multi_chunk = (total_rows > self.table_config.max_rows_per_chunk) or (len(column_slices) > 1)
        if self.table_config.summary_enabled and is_multi_chunk:
            summary_text = generate_table_summary(
                table,
                document_title=document_title,
                max_tokens=self.max_tokens_per_chunk,
                encoding_model=self.encoding_model,
            )
            meta = {
                "chunk_type": "table_summary",
                "table_id": table.table_id,
                "parent_id": table.parent_table_id or table.table_id,
                "page": table.page_number,
                "page_start": table.page_start,
                "page_end": table.page_end,
                "is_continuation": table.is_continuation,
                "section": table.section,
                "total_rows": total_rows,
                "total_columns": total_cols,
                "columns": table.headers,
                "validation_status": table.validation_status,
                "token_count": self._token_len(summary_text),
            }
            chunks.append(
                ChunkSchema(
                    chunk_id=uuid.uuid4(),
                    document_id=uuid.UUID(document_id),
                    version=version,
                    chunk_index=current_index,
                    text=summary_text,
                    section=table.section,
                    page=table.page_number,
                    metadata=meta,
                    chunking_version=self.chunking_version,
                )
            )
            current_index += 1

        # ── Process Each Column Slice (or Whole Table) into Row Batches ──────
        for slice_table, slice_label in column_slices:
            row_batches: List[Tuple[int, List[List[str]]]] = []
            max_rows = self.table_config.max_rows_per_chunk

            current_batch: List[List[str]] = []
            batch_start = 1

            for r_idx, row in enumerate(slice_table.rows, 1):
                test_batch = current_batch + [row]
                test_text = self._serialize_batch(
                    table=slice_table,
                    rows_subset=test_batch,
                    row_start=batch_start,
                    total_rows=total_rows,
                    document_title=document_title,
                    column_slice_label=slice_label,
                )

                if len(current_batch) >= max_rows or (
                    current_batch and self._token_len(test_text) > self.max_tokens_per_chunk
                ):
                    row_batches.append((batch_start, current_batch))
                    current_batch = [row]
                    batch_start = r_idx
                else:
                    current_batch.append(row)

            if current_batch:
                row_batches.append((batch_start, current_batch))

            for b_start, rows_batch in row_batches:
                b_end = b_start + len(rows_batch) - 1
                chunk_text = self._serialize_batch(
                    table=slice_table,
                    rows_subset=rows_batch,
                    row_start=b_start,
                    total_rows=total_rows,
                    document_title=document_title,
                    column_slice_label=slice_label,
                )

                meta = {
                    "chunk_type": "table",
                    "table_id": table.table_id,
                    "parent_id": table.parent_table_id or table.table_id,
                    "page": table.page_number,
                    "page_start": table.page_start,
                    "page_end": table.page_end,
                    "is_continuation": table.is_continuation,
                    "section": table.section,
                    "row_start": b_start,
                    "row_end": b_end,
                    "total_rows": total_rows,
                    "total_columns": total_cols,
                    "columns": slice_table.headers,
                    "column_slice": slice_label,
                    "validation_status": table.validation_status,
                    "serialization_strategy": self.table_config.serialization_strategy,
                    "token_count": self._token_len(chunk_text),
                }

                chunks.append(
                    ChunkSchema(
                        chunk_id=uuid.uuid4(),
                        document_id=uuid.UUID(document_id),
                        version=version,
                        chunk_index=current_index,
                        text=chunk_text,
                        section=table.section,
                        page=table.page_number,
                        metadata=meta,
                        chunking_version=self.chunking_version,
                    )
                )
                current_index += 1

        logger.info(
            "TableChunker: table '%s' (%d rows, %d cols, pages %d..%d) -> %d chunks",
            table.table_id, total_rows, total_cols, table.page_start, table.page_end, len(chunks),
        )
        return chunks

    def _slice_wide_columns(self, table: TableElement) -> List[Tuple[TableElement, str]]:
        """Slice a wide table into column subsets while always repeating anchor Column 0."""
        slice_size = max(4, self.table_config.wide_table_threshold - 1)
        total_cols = table.num_cols

        slices: List[Tuple[TableElement, str]] = []
        anchor_header = table.headers[0] if table.headers else "Entity"

        # Non-anchor column indices
        non_anchor_indices = list(range(1, total_cols))
        col_chunks = [non_anchor_indices[i:i + slice_size] for i in range(0, len(non_anchor_indices), slice_size)]

        for slice_idx, col_idx_group in enumerate(col_chunks, 1):
            slice_headers = [anchor_header] + [table.headers[c] for c in col_idx_group]
            slice_rows = []
            for r in table.rows:
                slice_r = [r[0]] + [r[c] if c < len(r) else "" for c in col_idx_group]
                slice_rows.append(slice_r)

            c_start = col_idx_group[0] + 1
            c_end = col_idx_group[-1] + 1
            slice_label = f"Columns {c_start}–{c_end} of {total_cols}"

            sliced_elem = TableElement(
                element_id=f"{table.element_id}_col{slice_idx}",
                table_id=table.table_id,
                page_number=table.page_number,
                page_start=table.page_start,
                page_end=table.page_end,
                is_continuation=table.is_continuation,
                parent_table_id=table.parent_table_id,
                bbox=table.bbox,
                section=table.section,
                caption=table.caption,
                headers=slice_headers,
                rows=slice_rows,
                num_rows=len(slice_rows),
                num_cols=len(slice_headers),
                context_hierarchy=table.context_hierarchy,
                preceding_text=table.preceding_text,
                validation_status=table.validation_status,
            )
            slices.append((sliced_elem, slice_label))

        return slices

    def _serialize_batch(
        self,
        table: TableElement,
        rows_subset: List[List[str]],
        row_start: int,
        total_rows: int,
        document_title: Optional[str] = None,
        column_slice_label: Optional[str] = None,
    ) -> str:
        """Apply configured serialization strategy to a subset of rows."""
        strategy = self.table_config.serialization_strategy.lower()

        if strategy == "markdown":
            breadcrumb = []
            if document_title:
                breadcrumb.append(f"Document: {document_title}")
            if table.page_start != table.page_end:
                breadcrumb.append(f"Pages {table.page_start}–{table.page_end}")
            else:
                breadcrumb.append(f"Page {table.page_number}")
            if table.section:
                breadcrumb.append(f"Section: {table.section}")
            breadcrumb.append(f"Table: {table.caption or table.table_id}")
            header = f"[{' | '.join(breadcrumb)}]\n"
            range_str = f"Rows {row_start}–{row_start + len(rows_subset) - 1} of {total_rows}"
            if column_slice_label:
                range_str += f" ({column_slice_label})"
            header += f"{range_str}\n"
            return header + TableSerializer.to_markdown(table, rows_subset=rows_subset)

        elif strategy == "row_based":
            return TableSerializer.to_row_grouped(
                table=table,
                rows_subset=rows_subset,
                row_start=row_start,
                total_rows=total_rows,
                document_title=document_title,
            )

        elif strategy == "json":
            return TableSerializer.to_json(table, rows_subset=rows_subset)

        else:
            # Default: hybrid
            return TableSerializer.to_hybrid(
                table=table,
                rows_subset=rows_subset,
                row_start=row_start,
                total_rows=total_rows,
                document_title=document_title,
                column_slice_label=column_slice_label,
            )
