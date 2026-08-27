"""Relational Database / SQL source connector."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from rag.connectors.base import BaseSourceConnector
from rag.core.exceptions import (
    ConnectionValidationError,
    DiscoveryError,
    FetchError,
    )
from rag.core.models import DocumentDescriptor, RawDocumentPayload, SourceType


class DatabaseConnector(BaseSourceConnector):
    """Connector for discovering and fetching records from SQL databases.

    Serializes database rows/records into structured JSON documents with metadata.
    """

    def __init__(
        self,
        connection_string: str,
        query: str,
        id_column: str,
        timestamp_column: Optional[str] = None,
        table_name: Optional[str] = None,
        tenant_id: str = "default",
        engine: Optional[Engine] = None,
    ):
        super().__init__(tenant_id=tenant_id)
        self.connection_string = connection_string
        self.query = query.strip().rstrip(";")
        self.id_column = id_column
        self.timestamp_column = timestamp_column
        self.table_name = table_name or "sql_query"
        self.engine = engine or create_engine(self.connection_string)

    @property
    def source_type(self) -> SourceType:
        return SourceType.DATABASE

    def validate_connection(self) -> bool:
        """Verify database connectivity with a test query."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            raise ConnectionValidationError(f"Failed to connect to database: {e}") from e

    def discover(self, since: Optional[datetime] = None) -> Iterator[DocumentDescriptor]:
        """Discover records available in the query result."""
        self.validate_connection()
        try:
            sql = self.query
            params: Dict[str, Any] = {}

            if since and self.timestamp_column:
                if "WHERE" in sql.upper():
                    sql = f"{sql} AND {self.timestamp_column} > :since_param"
                else:
                    sql = f"{sql} WHERE {self.timestamp_column} > :since_param"
                params["since_param"] = since

            with self.engine.connect() as conn:
                result = conn.execute(text(sql), params)
                columns = list(result.keys())

                if self.id_column not in columns:
                    raise DiscoveryError(f"ID column '{self.id_column}' not found in query results: {columns}")

                for row in result:
                    row_dict = dict(zip(columns, row))
                    record_id = str(row_dict[self.id_column])
                    source_uri = f"db://{self.table_name}/{record_id}"

                    modified_at = None
                    if self.timestamp_column and row_dict.get(self.timestamp_column):
                        val = row_dict[self.timestamp_column]
                        if isinstance(val, datetime):
                            modified_at = val if val.tzinfo else val.replace(tzinfo=timezone.utc)

                    yield DocumentDescriptor(
                        source=self.source_type,
                        source_uri=source_uri,
                        filename=f"{self.table_name}_{record_id}.json",
                        mime_type="application/json",
                        tenant_id=self.tenant_id,
                        last_modified_at=modified_at,
                        custom_metadata={
                            "table_name": self.table_name,
                            "record_id": record_id,
                            "columns": columns,
                            "row_data": row_dict,
                        },
                    )
        except Exception as e:
            raise DiscoveryError(f"Failed to discover database records: {e}") from e

    def fetch(self, descriptor: DocumentDescriptor) -> RawDocumentPayload:
        """Fetch/extract row data payload as JSON bytes."""
        try:
            row_data = descriptor.custom_metadata.get("row_data")
            if row_data is None:
                # Re-query single row by id if row_data was not cached in descriptor
                record_id = descriptor.custom_metadata.get("record_id")
                sql = f"SELECT * FROM {self.table_name} WHERE {self.id_column} = :id_val"
                with self.engine.connect() as conn:
                    result = conn.execute(text(sql), {"id_val": record_id})
                    row = result.first()
                    if not row:
                        raise FetchError(f"Record {record_id} not found in {self.table_name}")
                    row_data = dict(zip(result.keys(), row))

            # Helper JSON serializer for datetimes and non-standard types
            def json_serializer(obj: Any) -> Any:
                if isinstance(obj, (datetime,)):
                    return obj.isoformat()
                return str(obj)

            content = json.dumps(row_data, default=json_serializer, indent=2).encode("utf-8")
            content_hash = self.compute_hash(content)

            return RawDocumentPayload(
                descriptor=descriptor,
                content=content,
                content_hash=content_hash,
            )
        except Exception as e:
            raise FetchError(f"Failed to fetch database record {descriptor.source_uri}: {e}") from e
