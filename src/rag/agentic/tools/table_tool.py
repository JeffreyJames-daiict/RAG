"""Table and structured data lookup tool for agentic workflows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from rag.agentic.tools.base import BaseTool


class TableLookupInput(BaseModel):
    """Input parameters for table data query."""

    table_name: str = Field(..., description="Name or identifier of the table (e.g., 'balance_sheet')")
    row_key: Optional[str] = Field(default=None, description="Row identifier or metric name to lookup")
    column_name: Optional[str] = Field(default=None, description="Specific column to inspect")


class TableLookupOutput(BaseModel):
    """Result containing table cells or rows."""

    table_name: str = Field(..., description="Target table name")
    data: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted table rows or cells")
    summary: str = Field(..., description="Text summary of the table lookup")


class TableLookupTool(BaseTool):
    """Tool for querying structured table datasets."""

    name = "table_lookup"
    description = "Queries tables, financial statements, and structured metrics by table name, row, or column."
    input_schema = TableLookupInput
    output_schema = TableLookupOutput

    def __init__(self, tables: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        self._tables = tables or {}

    def execute(self, inputs: TableLookupInput) -> TableLookupOutput:
        """Lookup table rows matching criteria."""
        table_data = self._tables.get(inputs.table_name.lower().strip(), [])
        if not table_data:
            return TableLookupOutput(
                table_name=inputs.table_name,
                data=[],
                summary=f"Table '{inputs.table_name}' was not found in registered structured tables.",
            )

        filtered = table_data
        if inputs.row_key:
            filtered = [
                row for row in filtered
                if any(inputs.row_key.lower() in str(val).lower() for val in row.values())
            ]

        if inputs.column_name:
            col = inputs.column_name.lower()
            filtered = [
                {k: v for k, v in row.items() if k.lower() == col}
                for row in filtered
                if any(k.lower() == col for k in row.keys())
            ]

        return TableLookupOutput(
            table_name=inputs.table_name,
            data=filtered,
            summary=f"Found {len(filtered)} matching rows in table '{inputs.table_name}'.",
        )
