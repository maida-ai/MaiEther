"""Tests for the core Ether model."""

import json

import pytest
from pydantic import ValidationError

from ether.attachment import Attachment
from ether.core import Ether


class TestEtherCreation:
    """Test Ether model creation and validation."""

    def test_create_ether_with_empty_payload_and_metadata(self) -> None:
        """Test creating an Ether with empty payload and metadata."""
        ether = Ether(
            kind="embedding",
            schema_version=1,
            payload={},
            metadata={},
        )

        assert ether.kind == "embedding"
        assert ether.schema_version == 1
        assert ether.payload == {}
        assert ether.metadata == {}
        assert ether.extra_fields == {}
        assert ether.attachments == []

    def test_create_ether_with_default_schema_version(self) -> None:
        """Test creating an Ether with default schema_version."""
        ether = Ether(
            kind="tokens",
            payload={"ids": [1, 2, 3]},
            metadata={"vocab": "bert-base"},
        )

        assert ether.kind == "tokens"
        assert ether.schema_version == 1  # default value
        assert ether.payload == {"ids": [1, 2, 3]}
        assert ether.metadata == {"vocab": "bert-base"}

    def test_create_ether_with_attachments(self) -> None:
        """Test creating an Ether with attachments."""
        attachment = Attachment(
            id="emb-0",
            uri="shm://embeddings/12345",
            media_type="application/x-raw-tensor",
            codec="RAW_F32",
            shape=[768],
            dtype="float32",
        )

        ether = Ether(
            kind="embedding",
            schema_version=1,
            payload={"dim": 768},
            metadata={"source": "bert-base"},
            attachments=[attachment],
        )

        assert ether.kind == "embedding"
        assert len(ether.attachments) == 1
        assert ether.attachments[0].id == "emb-0"
        assert ether.attachments[0].codec == "RAW_F32"

    def test_create_ether_with_extra_fields(self) -> None:
        """Test creating an Ether with extra fields."""
        ether = Ether(
            kind="text",
            payload={"text": "Hello, world!"},
            metadata={"lang": "en"},
            extra_fields={"note": "test message", "priority": "high"},
        )

        assert ether.extra_fields == {"note": "test message", "priority": "high"}

    def test_create_ether_with_nested_payload_and_metadata(self) -> None:
        """Test creating an Ether with nested payload and metadata."""
        ether = Ether(
            kind="embedding",
            payload={"vec": {"values": [1.0, 2.0, 3.0], "dim": 3}},
            metadata={"model": {"name": "bert-base", "version": "1.0"}},
        )

        assert ether.payload == {"vec": {"values": [1.0, 2.0, 3.0], "dim": 3}}
        assert ether.metadata == {"model": {"name": "bert-base", "version": "1.0"}}


class TestEtherValidation:
    """Test Ether model validation and error cases."""

    def test_validation_error_empty_kind(self) -> None:
        """Test that empty kind raises an error."""
        with pytest.raises(ValueError, match="kind cannot be empty"):
            Ether(
                kind="",
                payload={},
                metadata={},
            )

    def test_validation_error_negative_schema_version(self) -> None:
        """Test that negative schema_version raises an error."""
        with pytest.raises(ValidationError):
            Ether(
                kind="embedding",
                schema_version=0,
                payload={},
                metadata={},
            )

    def test_validation_error_zero_schema_version(self) -> None:
        """Test that zero schema_version raises an error."""
        with pytest.raises(ValidationError):
            Ether(
                kind="embedding",
                schema_version=0,
                payload={},
                metadata={},
            )

    def test_validation_error_missing_required_fields(self) -> None:
        """Test that missing required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            Ether(
                kind="embedding",
                # missing payload and metadata
            )

    def test_validation_error_invalid_kind_type(self) -> None:
        """Test that invalid kind type raises ValidationError."""
        with pytest.raises(ValidationError):
            Ether(
                kind=123,  # should be str
                payload={},
                metadata={},
            )

    def test_validation_error_invalid_schema_version_type(self) -> None:
        """Test that invalid schema_version type raises ValidationError."""
        with pytest.raises(ValidationError):
            Ether(
                kind="embedding",
                schema_version="invalid",  # should be int
                payload={},
                metadata={},
            )


class TestEtherSerialization:
    """Test Ether model serialization and deserialization."""

    def test_model_dump_round_trip(self) -> None:
        """Test that model_dump() and model_validate() work correctly."""
        original = Ether(
            kind="embedding",
            schema_version=2,
            payload={"dim": 768},
            metadata={"source": "bert-base"},
            extra_fields={"note": "test"},
        )

        data = original.model_dump()
        restored = Ether.model_validate(data)

        assert restored.kind == original.kind
        assert restored.schema_version == original.schema_version
        assert restored.payload == original.payload
        assert restored.metadata == original.metadata
        assert restored.extra_fields == original.extra_fields

    def test_json_serialization_round_trip(self) -> None:
        """Test JSON serialization and deserialization."""
        ether = Ether(
            kind="tokens",
            payload={"ids": [1, 2, 3]},
            metadata={"vocab": "bert-base"},
        )

        json_str = ether.model_dump_json()
        data = json.loads(json_str)
        restored = Ether.model_validate(data)

        assert restored.kind == ether.kind
        assert restored.schema_version == ether.schema_version
        assert restored.payload == ether.payload
        assert restored.metadata == ether.metadata

    def test_json_serialization_with_attachments(self) -> None:
        """Test JSON serialization with attachments."""
        attachment = Attachment(
            id="emb-0",
            inline_bytes=b"\x00\x00\x80\x3f",  # 1.0 in float32
            media_type="application/x-raw-tensor",
            codec="RAW_F32",
            shape=[1],
            dtype="float32",
        )

        ether = Ether(
            kind="embedding",
            payload={"dim": 1},
            metadata={"source": "test"},
            attachments=[attachment],
        )

        json_str = ether.model_dump_json()
        data = json.loads(json_str)
        restored = Ether.model_validate(data)

        assert len(restored.attachments) == 1
        assert restored.attachments[0].id == "emb-0"
        assert restored.attachments[0].inline_bytes == b"\x00\x00\x80\x3f"


class TestEtherSummary:
    """Test Ether summary functionality."""

    def test_summary_basic(self) -> None:
        """Test basic summary functionality."""
        ether = Ether(
            kind="embedding",
            schema_version=1,
            payload={"dim": 768},
            metadata={"source": "bert-base"},
        )

        summary = ether.summary()

        assert summary["kind"] == "embedding"
        assert summary["schema_version"] == 1
        assert summary["payload_keys"] == ["dim"]
        assert summary["metadata_keys"] == ["source"]
        assert summary["extra_keys"] == []
        assert summary["attachments"] == []
        assert summary["source_model"] is None

    def test_summary_with_nested_payload_and_metadata(self) -> None:
        """Test summary with nested payload and metadata."""
        ether = Ether(
            kind="embedding",
            payload={"vec": {"values": [1.0, 2.0], "dim": 2}},
            metadata={"model": {"name": "bert", "version": "1.0"}},
        )

        summary = ether.summary()

        assert summary["payload_keys"] == ["vec.dim", "vec.values"]
        assert summary["metadata_keys"] == ["model.name", "model.version"]

    def test_summary_with_attachments(self) -> None:
        """Test summary with attachments."""
        attachment1 = Attachment(
            id="emb-0",
            uri="shm://embeddings/12345",
            media_type="application/x-raw-tensor",
            codec="RAW_F32",
        )
        attachment2 = Attachment(
            id="emb-1",
            uri="shm://embeddings/12346",
            media_type="application/x-raw-tensor",
            codec="RAW_F32",
        )

        ether = Ether(
            kind="embedding",
            payload={},
            metadata={},
            attachments=[attachment1, attachment2],
        )

        summary = ether.summary()

        assert summary["attachments"] == ["emb-0", "emb-1"]

    def test_summary_with_extra_fields(self) -> None:
        """Test summary with extra fields."""
        ether = Ether(
            kind="text",
            payload={"text": "Hello"},
            metadata={},
            extra_fields={"note": "test", "priority": "high"},
        )

        summary = ether.summary()

        assert summary["extra_keys"] == ["note", "priority"]


class TestEtherEdgeCases:
    """Test Ether edge cases and special scenarios."""

    def test_ether_with_large_payload(self) -> None:
        """Test Ether with large payload."""
        large_payload = {"data": list(range(1000))}
        ether = Ether(
            kind="data",
            payload=large_payload,
            metadata={},
        )

        assert len(ether.payload["data"]) == 1000
        assert ether.payload["data"][0] == 0
        assert ether.payload["data"][-1] == 999

    def test_ether_with_complex_metadata(self) -> None:
        """Test Ether with complex metadata structure."""
        complex_metadata = {
            "trace_id": "123e4567-e89b-12d3-a456-426614174000",
            "span_id": "456e7890-e89b-12d3-a456-426614174001",
            "created_at": "2023-01-01T00:00:00Z",
            "producer": "test-node",
            "lineage": [{"node": "input", "version": "1.0", "ts": "2023-01-01T00:00:00Z"}],
        }

        ether = Ether(
            kind="embedding",
            payload={},
            metadata=complex_metadata,
        )

        assert ether.metadata["trace_id"] == "123e4567-e89b-12d3-a456-426614174000"
        assert len(ether.metadata["lineage"]) == 1

    def test_ether_with_empty_attachments_list(self) -> None:
        """Test Ether with empty attachments list."""
        ether = Ether(
            kind="text",
            payload={"text": "Hello"},
            metadata={},
            attachments=[],
        )

        assert ether.attachments == []
        assert len(ether.attachments) == 0

    def test_ether_with_none_values_in_optional_fields(self) -> None:
        """Test Ether with None values in optional fields."""
        ether = Ether(
            kind="embedding",
            payload={"dim": 768},
            metadata={"source": "bert-base"},
            extra_fields={},
            attachments=[],
        )

        assert ether.extra_fields == {}
        assert ether.attachments == []


class TestEtherImport:
    """Test Ether import functionality."""

    def test_import_from_package(self) -> None:
        """Test that Ether can be imported from the package."""
        from ether import Ether

        ether = Ether(
            kind="test",
            payload={},
            metadata={},
        )

        assert ether.kind == "test"

    def test_ether_is_available_in_all(self) -> None:
        """Test that Ether is available in __all__."""
        from ether import __all__

        assert "Ether" in __all__
