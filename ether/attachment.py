"""Attachment model for Ether envelopes.

This module defines the Attachment model used in Ether envelopes for handling
binary data, external buffers, and large tensors/tables that should be transported
separately from the main payload to enable zero-copy operations.
"""

import base64
from typing import Any

from pydantic import BaseModel, Field, field_serializer, field_validator


class Attachment(BaseModel):
    """Represents an attachment in an Ether envelope for binary data or external buffers.

    Attachments are used for large tensors, tables, or binary data that should be
    transported separately from the main payload to enable zero-copy operations.
    They can reference external data via URIs or contain inline binary data.

    Examples:
        >>> # Arrow IPC attachment
        >>> att = Attachment(
        ...     id="table-0",
        ...     uri="shm://tables/12345",
        ...     media_type="application/vnd.arrow.ipc",
        ...     codec="ARROW_IPC",
        ...     size_bytes=1024
        ... )
        >>>
        >>> # Inline tensor attachment
        >>> att = Attachment(
        ...     id="tensor-0",
        ...     inline_bytes=b"\\x00\\x00\\x80\\x3f",  # 1.0 in float32
        ...     media_type="application/x-raw-tensor",
        ...     codec="RAW_F32",
        ...     shape=[1],
        ...     dtype="float32"
        ... )
    """

    id: str = Field(description="Unique identifier for the attachment within the envelope")
    uri: str | None = Field(
        default=None, description="Optional URI reference to external data (e.g., shm://, file://, s3://)"
    )
    inline_bytes: bytes | None = Field(
        default=None, description="Optional inline binary data (base64 serialized when transported as JSON)"
    )
    media_type: str = Field(description="MIME type of the attachment (e.g., application/vnd.arrow.ipc)")
    codec: str | None = Field(default=None, description="Codec identifier (e.g., ARROW_IPC, DLPACK, RAW_F32)")
    shape: list[int] | None = Field(default=None, description="Shape dimensions for tensor data")
    dtype: str | None = Field(default=None, description="Data type (e.g., float32, int8, uint8, bfloat16)")
    byte_order: str | None = Field(default="LE", description="Byte order for numeric data (LE or BE)")
    device: str | None = Field(default=None, description="Device identifier (e.g., cpu, cuda:0, mps)")
    size_bytes: int | None = Field(default=None, description="Size of the attachment data in bytes")
    compression: dict[str, Any] | None = Field(
        default=None, description="Compression settings (e.g., {'name': 'zstd', 'level': 3})"
    )
    checksum: dict[str, Any] | None = Field(
        default=None, description="Checksum information (e.g., {'algo': 'crc32c', 'value': '...'})"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Attachment-local metadata")

    @field_serializer("inline_bytes")
    def serialize_inline_bytes(self, value: bytes | None) -> str | None:
        """Serialize inline_bytes as base64 string for JSON serialization."""
        if value is None:
            return None
        return base64.b64encode(value).decode("ascii")

    @field_validator("inline_bytes", mode="before")
    @classmethod
    def validate_inline_bytes(cls, value: Any) -> bytes | None:
        """Validate and convert inline_bytes from base64 string if needed."""
        if value is None:
            return None
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return base64.b64decode(value.encode("ascii"))
        raise ValueError("inline_bytes must be bytes or base64 string")

    def model_post_init(self, __context: Any) -> None:
        """Validate attachment configuration after initialization.

        Ensures that either uri or inline_bytes is provided, but not both.
        """
        if self.uri is not None and self.inline_bytes is not None:
            raise ValueError("Cannot specify both uri and inline_bytes")

        if self.uri is None and self.inline_bytes is None:
            raise ValueError("Must specify either uri or inline_bytes")
