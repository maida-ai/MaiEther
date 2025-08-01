"""Attachment model for Ether envelopes.

This module defines the Attachment model used in Ether envelopes for handling
binary data, external buffers, and large tensors/tables that should be transported
separately from the main payload to enable zero-copy operations.

The Attachment class provides functionality for:
- External data references via URIs
- Inline binary data storage
- Tensor metadata (shape, dtype, device)
- Compression and checksum support
- Base64 serialization for JSON transport
- NumPy array conversion with zero-copy support

Examples:
    >>> from ether import Attachment
    >>>
    >>> # Arrow IPC attachment with external reference
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
    >>>
    >>> # NumPy array attachment
    >>> import numpy as np
    >>> arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    >>> att = Attachment.from_numpy(arr, id="emb-0")
"""

import base64
from typing import Any

from pydantic import BaseModel, Field, field_serializer, field_validator

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class Attachment(BaseModel):
    """Represents an attachment in an Ether envelope for binary data or external buffers.

    Attachments are used for large tensors, tables, or binary data that should be
    transported separately from the main payload to enable zero-copy operations.
    They can reference external data via URIs or contain inline binary data.

    Args:
        id: Unique identifier for the attachment within the envelope
        uri: Optional URI reference to external data (e.g., shm://, file://, s3://)
        inline_bytes: Optional inline binary data (base64 serialized when transported as JSON)
        media_type: MIME type of the attachment (e.g., application/vnd.arrow.ipc)
        codec: Optional codec identifier (e.g., ARROW_IPC, DLPACK, RAW_F32)
        shape: Optional shape dimensions for tensor data
        dtype: Optional data type (e.g., float32, int8, uint8, bfloat16)
        byte_order: Optional byte order for numeric data (LE or BE, default: LE)
        device: Optional device identifier (e.g., cpu, cuda:0, mps)
        size_bytes: Optional size of the attachment data in bytes
        compression: Optional compression settings (e.g., {'name': 'zstd', 'level': 3})
        checksum: Optional checksum information (e.g., {'algo': 'crc32c', 'value': '...'})
        metadata: Optional attachment-local metadata

    Raises:
        ValueError: If both uri and inline_bytes are specified, or if neither is specified

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
        """Serialize inline_bytes as base64 string for JSON serialization.

        Args:
            value: The bytes value to serialize

        Returns:
            Base64 encoded string or None if value is None
        """
        if value is None:
            return None
        return base64.b64encode(value).decode("ascii")

    @field_validator("inline_bytes", mode="before")
    @classmethod
    def validate_inline_bytes(cls, value: Any) -> bytes | None:
        """Validate and convert inline_bytes from base64 string if needed.

        Args:
            value: The value to validate and convert

        Returns:
            Bytes value or None

        Raises:
            ValueError: If value is not bytes or base64 string
        """
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

        Raises:
            ValueError: If both uri and inline_bytes are specified, or if neither is specified
        """
        if self.uri is not None and self.inline_bytes is not None:
            raise ValueError("Cannot specify both uri and inline_bytes")

        if self.uri is None and self.inline_bytes is None:
            raise ValueError("Must specify either uri or inline_bytes")

    @classmethod
    def from_numpy(cls, array: "np.ndarray", *, id: str, uri: str | None = None) -> "Attachment":
        """Create an Attachment from a NumPy array with zero-copy support.

        Creates an attachment that can efficiently transport NumPy arrays without
        unnecessary copying. The array data is stored as inline bytes or referenced
        via URI, with proper metadata including shape, dtype, and size.

        Args:
            array: The NumPy array to convert to an attachment
            id: Unique identifier for the attachment within the envelope
            uri: Optional URI reference to external data (if provided, array data
                 will not be stored inline)

        Returns:
            Attachment instance with array data and metadata

        Raises:
            ImportError: If NumPy is not available
            ValueError: If array is not a valid NumPy array

        Examples:
            >>> import numpy as np
            >>> arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            >>> att = Attachment.from_numpy(arr, id="emb-0")
            >>> att.shape == [3]
            True
            >>> att.dtype == "float32"
            True
            >>> att.size_bytes == 12  # 3 * 4 bytes per float32
            True
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for from_numpy() method")

        if not isinstance(array, np.ndarray):
            raise ValueError("array must be a NumPy ndarray")

        # Determine the codec based on dtype
        dtype_to_codec = {
            np.float32: "RAW_F32",
            np.float64: "RAW_F64",
            np.int32: "RAW_I32",
            np.int64: "RAW_I64",
            np.uint8: "RAW_U8",
            np.uint16: "RAW_U16",
            np.uint32: "RAW_U32",
            np.uint64: "RAW_U64",
            np.int8: "RAW_I8",
            np.int16: "RAW_I16",
        }

        codec = dtype_to_codec.get(array.dtype.type)
        if codec is None:
            # For unsupported dtypes, use a generic codec
            codec = "RAW_BYTES"

        # Convert dtype to string representation
        dtype_str = str(array.dtype)

        # Get array shape and size
        shape = list(array.shape)
        size_bytes = int(array.nbytes)

        # Handle zero-dimensional arrays
        if len(shape) == 0:
            # For zero-dimensional arrays, we need to handle them specially
            shape = []

        # Prepare attachment data
        if uri is not None:
            # Use URI reference (for zero-copy scenarios)
            inline_bytes = None
        else:
            # Store data inline
            inline_bytes = array.tobytes()

        return cls(
            id=id,
            uri=uri,
            inline_bytes=inline_bytes,
            media_type="application/x-raw-tensor",
            codec=codec,
            shape=shape,
            dtype=dtype_str,
            size_bytes=size_bytes,
            byte_order="LE",  # NumPy uses native byte order, typically LE
        )

    def to_numpy(self) -> "np.ndarray":
        """Convert the attachment back to a NumPy array.

        Reconstructs a NumPy array from the attachment's binary data and metadata.
        This method requires the attachment to have inline_bytes or a valid URI
        that can be accessed.

        Returns:
            NumPy array with the original shape, dtype, and data

        Raises:
            ImportError: If NumPy is not available
            ValueError: If attachment cannot be converted to NumPy array
            RuntimeError: If attachment uses URI reference (not yet supported)

        Examples:
            >>> import numpy as np
            >>> arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            >>> att = Attachment.from_numpy(arr, id="emb-0")
            >>> restored = att.to_numpy()
            >>> np.array_equal(arr, restored)
            True
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for to_numpy() method")

        if self.inline_bytes is None:
            if self.uri is not None:
                raise RuntimeError("URI-based attachments not yet supported for to_numpy()")
            else:
                raise ValueError("Attachment has no data to convert")

        if self.shape is None or self.dtype is None:
            raise ValueError("Attachment missing shape or dtype information")

        # Convert string dtype back to NumPy dtype
        try:
            np_dtype = np.dtype(self.dtype)
        except TypeError as e:
            raise ValueError(f"Invalid dtype '{self.dtype}'") from e

        # Reshape the bytes into the original array
        try:
            array = np.frombuffer(self.inline_bytes, dtype=np_dtype)

            # Handle zero-dimensional arrays (empty shape list)
            # We guranteed that the shape is not None earlier
            if len(self.shape) == 0:
                return np.array(array.item(), dtype=np_dtype).reshape(())
            else:
                return array.reshape(self.shape)
        except ValueError as e:
            raise ValueError("Failed to reconstruct array from attachment data") from e
