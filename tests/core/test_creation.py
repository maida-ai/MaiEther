"""Tests for Ether creation and basic functionality."""

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
        """Test that empty kind is allowed for models without kind specification."""
        # Empty kind is now allowed for models that don't specify a kind
        ether = Ether(
            kind="",
            payload={},
            metadata={},
        )
        assert ether.kind == ""

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
