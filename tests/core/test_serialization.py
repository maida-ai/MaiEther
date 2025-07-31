"""Tests for Ether serialization and deserialization."""

import json

from ether.attachment import Attachment
from ether.core import Ether


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
