"""Tests for Ether summary functionality."""

from ether.attachment import Attachment
from ether.core import Ether


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
