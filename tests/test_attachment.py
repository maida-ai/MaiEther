"""Tests for the Attachment model."""

import json

import pytest
from pydantic import ValidationError

from ether.attachment import Attachment


class TestAttachmentCreation:
    """Test Attachment model creation and validation."""

    def test_create_attachment_with_uri(self) -> None:
        """Test creating an attachment with URI reference."""
        att = Attachment(
            id="test-0",
            uri="shm://data/12345",
            media_type="application/vnd.arrow.ipc",
            codec="ARROW_IPC",
            size_bytes=1024,
        )

        assert att.id == "test-0"
        assert att.uri == "shm://data/12345"
        assert att.inline_bytes is None
        assert att.media_type == "application/vnd.arrow.ipc"
        assert att.codec == "ARROW_IPC"
        assert att.size_bytes == 1024
        assert att.byte_order == "LE"  # default value
        assert att.metadata == {}  # default empty dict

    def test_create_attachment_with_inline_bytes(self) -> None:
        """Test creating an attachment with inline binary data."""
        data = b"\x00\x00\x80\x3f"  # 1.0 in float32
        att = Attachment(
            id="tensor-0",
            inline_bytes=data,
            media_type="application/x-raw-tensor",
            codec="RAW_F32",
            shape=[1],
            dtype="float32",
        )

        assert att.id == "tensor-0"
        assert att.uri is None
        assert att.inline_bytes == data
        assert att.media_type == "application/x-raw-tensor"
        assert att.codec == "RAW_F32"
        assert att.shape == [1]
        assert att.dtype == "float32"

    def test_create_attachment_with_compression_and_checksum(self) -> None:
        """Test creating an attachment with compression and checksum."""
        att = Attachment(
            id="compressed-0",
            uri="file:///tmp/data.arrow",
            media_type="application/vnd.arrow.ipc",
            codec="ARROW_IPC",
            size_bytes=2048,
            compression={"name": "zstd", "level": 3},
            checksum={"algo": "crc32c", "value": "a1b2c3d4"},
        )

        assert att.compression == {"name": "zstd", "level": 3}
        assert att.checksum == {"algo": "crc32c", "value": "a1b2c3d4"}

    def test_create_attachment_with_metadata(self) -> None:
        """Test creating an attachment with custom metadata."""
        metadata = {"source": "model-v1", "quantized": True}
        att = Attachment(
            id="meta-0",
            uri="s3://bucket/data.arrow",
            media_type="application/vnd.arrow.ipc",
            metadata=metadata,
        )

        assert att.metadata == metadata

    def test_create_attachment_with_custom_byte_order(self) -> None:
        """Test creating an attachment with custom byte order."""
        att = Attachment(
            id="be-0",
            uri="file:///data.bin",
            media_type="application/x-raw-tensor",
            byte_order="BE",
        )

        assert att.byte_order == "BE"


class TestAttachmentValidation:
    """Test Attachment model validation and error cases."""

    def test_validation_error_both_uri_and_inline_bytes(self) -> None:
        """Test that specifying both uri and inline_bytes raises an error."""
        with pytest.raises(ValueError, match="Cannot specify both uri and inline_bytes"):
            Attachment(
                id="invalid-0",
                uri="file:///data.bin",
                inline_bytes=b"data",
                media_type="application/x-raw-tensor",
            )

    def test_validation_error_neither_uri_nor_inline_bytes(self) -> None:
        """Test that specifying neither uri nor inline_bytes raises an error."""
        with pytest.raises(ValueError, match="Must specify either uri or inline_bytes"):
            Attachment(
                id="invalid-0",
                media_type="application/x-raw-tensor",
            )

    def test_validation_error_missing_required_fields(self) -> None:
        """Test that missing required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            Attachment()  # Missing id and media_type

    def test_validation_error_invalid_id_type(self) -> None:
        """Test that invalid id type raises ValidationError."""
        with pytest.raises(ValidationError):
            Attachment(
                id=123,  # Should be str
                uri="file:///data.bin",
                media_type="application/x-raw-tensor",
            )

    def test_validation_error_invalid_media_type_type(self) -> None:
        """Test that invalid media_type type raises ValidationError."""
        with pytest.raises(ValidationError):
            Attachment(
                id="test-0",
                uri="file:///data.bin",
                media_type=123,  # Should be str
            )

    def test_validation_error_invalid_inline_bytes_type(self) -> None:
        """Test that invalid type for inline_bytes raises ValueError with correct message."""
        with pytest.raises(ValueError, match="inline_bytes must be bytes or base64 string"):
            Attachment(
                id="bad-bytes-0",
                inline_bytes=12345,  # Invalid type
                media_type="application/x-raw-tensor",
            )


class TestAttachmentSerialization:
    """Test Attachment model serialization and deserialization."""

    def test_model_dump_round_trip(self) -> None:
        """Test that model_dump() and model_validate() work correctly."""
        original = Attachment(
            id="roundtrip-0",
            uri="shm://data/12345",
            media_type="application/vnd.arrow.ipc",
            codec="ARROW_IPC",
            shape=[10, 20],
            dtype="float32",
            size_bytes=800,
            compression={"name": "zstd", "level": 1},
            checksum={"algo": "sha256", "value": "abc123"},
            metadata={"source": "test"},
        )

        # Serialize to dict
        data = original.model_dump()

        # Deserialize from dict
        restored = Attachment.model_validate(data)

        # Verify all fields match
        assert restored.id == original.id
        assert restored.uri == original.uri
        assert restored.media_type == original.media_type
        assert restored.codec == original.codec
        assert restored.shape == original.shape
        assert restored.dtype == original.dtype
        assert restored.size_bytes == original.size_bytes
        assert restored.compression == original.compression
        assert restored.checksum == original.checksum
        assert restored.metadata == original.metadata

    def test_json_serialization_with_inline_bytes(self) -> None:
        """Test JSON serialization with inline bytes (base64 encoding)."""
        data = b"\x00\x00\x80\x3f\x00\x00\x00\x40"  # [1.0, 2.0] in float32
        att = Attachment(
            id="json-0",
            inline_bytes=data,
            media_type="application/x-raw-tensor",
            codec="RAW_F32",
            shape=[2],
            dtype="float32",
        )

        # Serialize to JSON using Pydantic's JSON serialization
        json_str = att.model_dump_json()
        data_dict = json.loads(json_str)

        # Verify inline_bytes is base64 encoded and round-trips
        assert "inline_bytes" in data_dict
        import base64

        decoded = base64.b64decode(data_dict["inline_bytes"])
        assert decoded == data

        # Deserialize from JSON
        restored = Attachment.model_validate_json(json_str)
        assert restored.inline_bytes == data

    def test_json_serialization_with_uri(self) -> None:
        """Test JSON serialization with URI (no base64 encoding)."""
        att = Attachment(
            id="uri-0",
            uri="file:///tmp/data.arrow",
            media_type="application/vnd.arrow.ipc",
            codec="ARROW_IPC",
        )

        # Serialize to JSON
        json_str = att.model_dump_json()
        data_dict = json.loads(json_str)

        # Verify uri is preserved as string
        assert data_dict["uri"] == "file:///tmp/data.arrow"
        assert data_dict["inline_bytes"] is None

        # Deserialize from JSON
        restored = Attachment.model_validate_json(json_str)
        assert restored.uri == "file:///tmp/data.arrow"
        assert restored.inline_bytes is None


class TestAttachmentEdgeCases:
    """Test Attachment model edge cases and boundary conditions."""

    def test_empty_metadata_default(self) -> None:
        """Test that metadata defaults to empty dict."""
        att = Attachment(
            id="empty-meta-0",
            uri="file:///data.bin",
            media_type="application/x-raw-tensor",
        )
        assert att.metadata == {}

    def test_none_values_for_optional_fields(self) -> None:
        """Test that None values are accepted for optional fields."""
        att = Attachment(
            id="none-0",
            uri="file:///data.bin",
            media_type="application/x-raw-tensor",
            codec=None,
            shape=None,
            dtype=None,
            device=None,
            size_bytes=None,
            compression=None,
            checksum=None,
        )

        assert att.codec is None
        assert att.shape is None
        assert att.dtype is None
        assert att.device is None
        assert att.size_bytes is None
        assert att.compression is None
        assert att.checksum is None

    def test_large_inline_bytes(self) -> None:
        """Test handling of large inline binary data."""
        large_data = b"x" * 10000  # 10KB of data
        att = Attachment(
            id="large-0",
            inline_bytes=large_data,
            media_type="application/x-raw-tensor",
            size_bytes=len(large_data),
        )

        assert att.inline_bytes is not None and len(att.inline_bytes) == 10000
        assert att.size_bytes == 10000

    def test_complex_metadata_structure(self) -> None:
        """Test complex nested metadata structure."""
        complex_metadata = {
            "nested": {
                "array": [1, 2, 3],
                "dict": {"key": "value"},
                "null": None,
            },
            "simple": "string",
        }

        att = Attachment(
            id="complex-0",
            uri="file:///data.bin",
            media_type="application/x-raw-tensor",
            metadata=complex_metadata,
        )

        assert att.metadata == complex_metadata


class TestAttachmentImport:
    """Test that Attachment can be imported from the main package."""

    def test_import_from_package(self) -> None:
        """Test that Attachment can be imported from ether package."""
        from ether import Attachment

        # Verify it's the same class
        from ether.attachment import Attachment as AttachmentDirect

        assert Attachment is AttachmentDirect

    def test_attachment_is_available_in_all(self) -> None:
        """Test that Attachment is included in __all__."""
        from ether import __all__

        assert "Attachment" in __all__
