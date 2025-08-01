"""Tests for Ether summary functionality."""

from pydantic import BaseModel

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
        # Metadata keys should include both user-provided and auto-populated fields
        assert "source" in summary["metadata_keys"]
        assert "trace_id" in summary["metadata_keys"]
        assert "created_at" in summary["metadata_keys"]
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
        # Metadata keys should include both user-provided and auto-populated fields
        assert "model.name" in summary["metadata_keys"]
        assert "model.version" in summary["metadata_keys"]
        assert "trace_id" in summary["metadata_keys"]
        assert "created_at" in summary["metadata_keys"]

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

    def test_summary_with_source_model(self) -> None:
        """Test summary includes source_model name when created from a registered model."""

        @Ether.register(
            payload=["embedding"],
            metadata=["source"],
            kind="embedding",
        )
        class TestModel(BaseModel):
            embedding: list[float]
            source: str

        model = TestModel(embedding=[1.0, 2.0, 3.0], source="test-model")
        ether = Ether.from_model(model)

        summary = ether.summary()

        assert summary["source_model"] == "TestModel"
        assert summary["kind"] == "embedding"
        assert summary["schema_version"] == 1
        assert summary["payload_keys"] == ["embedding"]
        # Metadata keys should include both user-provided and auto-populated fields
        assert "source" in summary["metadata_keys"]
        assert "trace_id" in summary["metadata_keys"]
        assert "created_at" in summary["metadata_keys"]
        assert summary["extra_keys"] == []
        assert summary["attachments"] == []

    def test_summary_with_complex_nested_structure(self) -> None:
        """Test summary with complex nested structure in payload and metadata."""
        ether = Ether(
            kind="complex",
            payload={
                "data": {"features": {"embeddings": [1.0, 2.0], "metadata": {"dim": 2}}, "config": {"model": "test"}}
            },
            metadata={
                "provenance": {"source": "test", "timestamp": "2023-01-01"},
                "processing": {"steps": ["normalize", "encode"]},
            },
            extra_fields={"debug": True, "version": "1.0"},
        )

        summary = ether.summary()

        assert summary["payload_keys"] == [
            "data.config.model",
            "data.features.embeddings",
            "data.features.metadata.dim",
        ]
        # Metadata keys should include both user-provided and auto-populated fields
        assert "processing.steps" in summary["metadata_keys"]
        assert "provenance.source" in summary["metadata_keys"]
        assert "provenance.timestamp" in summary["metadata_keys"]
        assert "trace_id" in summary["metadata_keys"]
        assert "created_at" in summary["metadata_keys"]
        assert summary["extra_keys"] == ["debug", "version"]
