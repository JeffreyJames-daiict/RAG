"""Amazon S3 / MinIO object storage connector."""

import mimetypes
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional
import boto3
from botocore.exceptions import ClientError
from rag.connectors.base import BaseSourceConnector
from rag.core.exceptions import (
    AuthenticationError,
    ConnectionValidationError,
    DiscoveryError,
    FetchError,
)
from rag.core.models import DocumentDescriptor, RawDocumentPayload, SourceType


class S3Connector(BaseSourceConnector):
    """Connector for discovering and fetching documents from AWS S3 / MinIO buckets."""

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        s3_client: Optional[Any] = None,
        tenant_id: str = "default",
    ):
        super().__init__(tenant_id=tenant_id)
        self.bucket_name = bucket_name
        self.prefix = prefix.lstrip("/")

        if s3_client is not None:
            self.s3_client = s3_client
        else:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
                endpoint_url=endpoint_url,
            )

    @property
    def source_type(self) -> SourceType:
        return SourceType.S3

    def validate_connection(self) -> bool:
        """Verify access to the specified bucket."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("403", "AccessDenied"):
                raise AuthenticationError(f"Access denied to S3 bucket '{self.bucket_name}': {e}") from e
            if error_code in ("404", "NoSuchBucket"):
                raise ConnectionValidationError(f"S3 bucket '{self.bucket_name}' not found: {e}") from e
            raise ConnectionValidationError(f"Failed to connect to S3 bucket '{self.bucket_name}': {e}") from e

    def discover(self, since: Optional[datetime] = None) -> Iterator[DocumentDescriptor]:
        """List objects in bucket matching prefix and newer than `since`."""
        self.validate_connection()
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)

            for page in page_iterator:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    # Skip directory marker keys
                    if key.endswith("/"):
                        continue

                    last_modified = obj.get("LastModified")
                    if last_modified and since and last_modified <= since:
                        continue

                    filename = key.split("/")[-1]
                    mime_type, _ = mimetypes.guess_type(filename)
                    if not mime_type:
                        mime_type = "application/octet-stream"

                    s3_uri = f"s3://{self.bucket_name}/{key}"
                    etag = obj.get("ETag", "").strip('"')

                    yield DocumentDescriptor(
                        source=self.source_type,
                        source_uri=s3_uri,
                        filename=filename,
                        mime_type=mime_type,
                        tenant_id=self.tenant_id,
                        last_modified_at=last_modified,
                        expected_hash=etag,
                        custom_metadata={
                            "bucket": self.bucket_name,
                            "key": key,
                            "etag": etag,
                            "size_bytes": obj.get("Size", 0),
                            "storage_class": obj.get("StorageClass", "STANDARD"),
                        },
                    )
        except Exception as e:
            raise DiscoveryError(f"Failed to discover objects in s3://{self.bucket_name}/{self.prefix}: {e}") from e

    def fetch(self, descriptor: DocumentDescriptor) -> RawDocumentPayload:
        """Download raw object bytes from S3 and compute content SHA-256."""
        bucket = descriptor.custom_metadata.get("bucket", self.bucket_name)
        key = descriptor.custom_metadata.get("key")
        if not key:
            # Parse from s3 URI if needed
            uri = descriptor.source_uri.replace("s3://", "")
            parts = uri.split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""

        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            content = body.read()
            content_hash = self.compute_hash(content)

            # Enrich custom_metadata with S3 headers
            descriptor.custom_metadata["content_type"] = response.get("ContentType", descriptor.mime_type)
            descriptor.custom_metadata["version_id"] = response.get("VersionId")

            return RawDocumentPayload(
                descriptor=descriptor,
                content=content,
                content_hash=content_hash,
            )
        except Exception as e:
            raise FetchError(f"Failed to fetch S3 object {descriptor.source_uri}: {e}") from e
