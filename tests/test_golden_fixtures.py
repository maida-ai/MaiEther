"""Golden fixtures round-trip test suite.

This module provides parametric tests that load golden fixtures, validate them
against their respective schemas, and test round-trip conversion through Ether.
"""

import json
import pathlib
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from ether import Registry
from ether.core import Ether
from ether.kinds import EmbeddingModel, TextModel, TokenModel
from tests.kinds import SCHEMAS_DIR


class TestGoldenFixtures:
    """Test golden fixtures validation and round-trip conversion."""

    @pytest.fixture(scope="class")
    def golden_dir(self) -> pathlib.Path:
        """Get the golden fixtures directory."""
        return pathlib.Path(__file__).parent / "golden"

    @pytest.fixture(scope="class")
    def schemas_dir(self) -> pathlib.Path:
        """Get the schemas directory."""
        return SCHEMAS_DIR

    def load_golden_fixture(self, golden_dir: pathlib.Path, fixture_name: str) -> dict[str, Any]:
        """Load a golden fixture from JSON file."""
        fixture_path = golden_dir / fixture_name
        with open(fixture_path) as f:
            return json.load(f)

    def load_schema(self, schemas_dir: pathlib.Path, kind: str, version: str) -> dict[str, Any]:
        """Load a schema from JSON file."""
        schema_path = schemas_dir / kind / f"{version}.json"
        with open(schema_path) as f:
            return json.load(f)

    def validate_against_schema(self, data: dict[str, Any], schema: dict[str, Any]) -> None:
        """Validate data against schema."""
        validator = Draft202012Validator(schema)
        errors = list(validator.iter_errors(data))
        assert not errors, f"Schema validation errors: {errors}"

    def test_embedding_v1_golden_fixture(
        self, golden_dir: pathlib.Path, schemas_dir: pathlib.Path, clear_registry
    ) -> None:
        """Test embedding.v1 golden fixture validation and round-trip."""
        # Load golden fixture
        fixture_data = self.load_golden_fixture(golden_dir, "embedding_v1.json")

        # Load schema
        schema = self.load_schema(schemas_dir, "embedding", "v1")

        # Validate against schema
        self.validate_against_schema(fixture_data, schema)

        # Test round-trip conversion

        # Re-register EmbeddingModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("values", "dim"),
            metadata_fields=("source", "norm", "quantized", "dtype"),
            extra_fields="ignore",
            renames={},
            kind="embedding",
        )
        Registry.set_spec(EmbeddingModel, spec)

        # Create Ether from fixture data
        ether = Ether.model_validate(fixture_data)

        # Verify Ether properties
        assert ether.kind == "embedding"
        assert ether.schema_version == 1
        assert ether.payload["values"] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert ether.payload["dim"] == 5
        assert ether.metadata["source"] == "test-embedding-model"
        assert ether.metadata["norm"] == 0.74
        assert ether.metadata["quantized"] is False
        assert ether.metadata["dtype"] == "float32"

        # Convert to EmbeddingModel
        model = ether.as_model(EmbeddingModel)

        # Verify model properties
        assert model.values == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert model.dim == 5
        assert model.source == "test-embedding-model"

        # Convert back to Ether
        round_trip_ether = Ether.from_model(model)

        # Verify round-trip properties
        assert round_trip_ether.kind == "embedding"
        assert round_trip_ether.schema_version == 1
        assert round_trip_ether.payload["values"] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert round_trip_ether.payload["dim"] == 5
        assert round_trip_ether.metadata["source"] == "test-embedding-model"

        # Note: The round-trip may not preserve all metadata fields since they're optional
        # in the EmbeddingModel, but the core payload should be preserved

    def test_text_v1_golden_fixture(self, golden_dir: pathlib.Path, schemas_dir: pathlib.Path, clear_registry) -> None:
        """Test text.v1 golden fixture validation and round-trip."""
        # Load golden fixture
        fixture_data = self.load_golden_fixture(golden_dir, "text_v1.json")

        # Load schema
        schema = self.load_schema(schemas_dir, "text", "v1")

        # Validate against schema
        self.validate_against_schema(fixture_data, schema)

        # Test round-trip conversion
        # Re-register TextModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("text",),
            metadata_fields=("lang", "encoding", "detected_lang_conf"),
            extra_fields="ignore",
            renames={},
            kind="text",
        )
        Registry.set_spec(TextModel, spec)

        # Create Ether from fixture data
        ether = Ether.model_validate(fixture_data)

        # Verify Ether properties
        assert ether.kind == "text"
        assert ether.schema_version == 1
        assert ether.payload["text"] == "Hello, world! This is a test message for the golden fixture."
        assert ether.metadata["lang"] == "en"
        assert ether.metadata["encoding"] == "utf-8"
        assert ether.metadata["detected_lang_conf"] == 0.95

        # Convert to TextModel
        model = ether.as_model(TextModel)

        # Verify model properties
        assert model.text == "Hello, world! This is a test message for the golden fixture."
        assert model.lang == "en"
        assert model.encoding == "utf-8"
        assert model.detected_lang_conf == 0.95

        # Convert back to Ether
        round_trip_ether = Ether.from_model(model)

        # Verify round-trip properties
        assert round_trip_ether.kind == "text"
        assert round_trip_ether.schema_version == 1
        assert round_trip_ether.payload["text"] == "Hello, world! This is a test message for the golden fixture."
        assert round_trip_ether.metadata["lang"] == "en"
        assert round_trip_ether.metadata["encoding"] == "utf-8"
        assert round_trip_ether.metadata["detected_lang_conf"] == 0.95

    def test_tokens_v1_golden_fixture(
        self, golden_dir: pathlib.Path, schemas_dir: pathlib.Path, clear_registry
    ) -> None:
        """Test tokens.v1 golden fixture validation and round-trip."""
        # Load golden fixture
        fixture_data = self.load_golden_fixture(golden_dir, "tokens_v1.json")

        # Load schema
        schema = self.load_schema(schemas_dir, "tokens", "v1")

        # Validate against schema
        self.validate_against_schema(fixture_data, schema)

        # Test round-trip conversion

        # Re-register TokenModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("ids", "mask"),
            metadata_fields=("vocab", "truncation", "offsets"),
            extra_fields="ignore",
            renames={},
            kind="tokens",
        )
        Registry.set_spec(TokenModel, spec)

        # Create Ether from fixture data
        ether = Ether.model_validate(fixture_data)

        # Verify Ether properties
        assert ether.kind == "tokens"
        assert ether.schema_version == 1
        assert ether.payload["ids"] == [101, 2023, 1010, 2088, 1029, 102]
        assert ether.payload["mask"] == [1, 1, 1, 1, 1, 1]
        assert ether.metadata["vocab"] == "bert-base-uncased"
        assert ether.metadata["truncation"] == "longest_first"
        assert ether.metadata["offsets"] is False

        # Convert to TokenModel
        model = ether.as_model(TokenModel)

        # Verify model properties
        assert model.ids == [101, 2023, 1010, 2088, 1029, 102]
        assert model.mask == [1, 1, 1, 1, 1, 1]
        assert model.vocab == "bert-base-uncased"
        assert model.truncation == "longest_first"
        assert model.offsets is False

        # Convert back to Ether
        round_trip_ether = Ether.from_model(model)

        # Verify round-trip properties
        assert round_trip_ether.kind == "tokens"
        assert round_trip_ether.schema_version == 1
        assert round_trip_ether.payload["ids"] == [101, 2023, 1010, 2088, 1029, 102]
        assert round_trip_ether.payload["mask"] == [1, 1, 1, 1, 1, 1]
        assert round_trip_ether.metadata["vocab"] == "bert-base-uncased"
        assert round_trip_ether.metadata["truncation"] == "longest_first"
        assert round_trip_ether.metadata["offsets"] is False

    def test_golden_fixtures_with_attachments(
        self, golden_dir: pathlib.Path, schemas_dir: pathlib.Path, clear_registry
    ) -> None:
        """Test golden fixtures with attachments for large data."""
        # Load embedding fixture and add attachment
        fixture_data = self.load_golden_fixture(golden_dir, "embedding_v1.json")

        # Add attachment to the fixture
        fixture_data["attachments"] = [
            {
                "id": "embedding-attachment-0",
                "uri": "shm://embeddings/12345",
                "media_type": "application/x-raw-tensor",
                "codec": "RAW_F32",
                "shape": [5],
                "dtype": "float32",
                "size_bytes": 20,
                "checksum": {"algo": "crc32c", "value": "a1b2c3d4"},
            }
        ]

        # Set values to None since we're using attachment
        fixture_data["payload"]["values"] = None

        # Validate against schema
        schema = self.load_schema(schemas_dir, "embedding", "v1")
        self.validate_against_schema(fixture_data, schema)

        # Test round-trip conversion with attachment
        # Re-register EmbeddingModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("values", "dim"),
            metadata_fields=("source", "norm", "quantized", "dtype"),
            extra_fields="ignore",
            renames={},
            kind="embedding",
        )
        Registry.set_spec(EmbeddingModel, spec)

        # Create Ether from fixture data
        ether = Ether.model_validate(fixture_data)

        # Verify Ether properties with attachment
        assert ether.kind == "embedding"
        assert ether.schema_version == 1
        assert ether.payload["values"] is None  # Using attachment
        assert ether.payload["dim"] == 5
        assert len(ether.attachments) == 1
        assert ether.attachments[0].id == "embedding-attachment-0"
        assert ether.attachments[0].codec == "RAW_F32"
        assert ether.attachments[0].shape == [5]
        assert ether.attachments[0].dtype == "float32"
        assert ether.attachments[0].size_bytes == 20

        # Convert to EmbeddingModel
        model = ether.as_model(EmbeddingModel)

        # Verify model properties
        assert model.values is None  # Using attachment
        assert model.dim == 5

        # Convert back to Ether
        round_trip_ether = Ether.from_model(model)

        # Verify round-trip properties
        assert round_trip_ether.kind == "embedding"
        assert round_trip_ether.schema_version == 1
        assert round_trip_ether.payload["values"] is None
        assert round_trip_ether.payload["dim"] == 5
        # Note: Attachments are not automatically preserved in round-trip
        # since they're not part of the model fields

    def test_golden_fixtures_edge_cases(self, golden_dir: pathlib.Path, schemas_dir: pathlib.Path) -> None:
        """Test golden fixtures with edge cases and minimal data."""
        # Test embedding with minimal data (no optional fields)
        minimal_embedding = {
            "kind": "embedding",
            "schema_version": 1,
            "payload": {"values": None, "dim": 768},
            "metadata": {},
            "extra_fields": {},
            "attachments": [],
        }

        schema = self.load_schema(schemas_dir, "embedding", "v1")
        self.validate_against_schema(minimal_embedding, schema)

        # Test text with minimal data
        minimal_text = {
            "kind": "text",
            "schema_version": 1,
            "payload": {"text": ""},
            "metadata": {},
            "extra_fields": {},
            "attachments": [],
        }

        schema = self.load_schema(schemas_dir, "text", "v1")
        self.validate_against_schema(minimal_text, schema)

        # Test tokens with minimal data
        minimal_tokens = {
            "kind": "tokens",
            "schema_version": 1,
            "payload": {"ids": []},
            "metadata": {"vocab": "test-vocab"},
            "extra_fields": {},
            "attachments": [],
        }

        schema = self.load_schema(schemas_dir, "tokens", "v1")
        self.validate_against_schema(minimal_tokens, schema)

    def test_golden_fixtures_schema_compliance(self, schemas_dir: pathlib.Path) -> None:
        """Test that all golden fixtures strictly comply with their schemas."""
        # Test embedding schema compliance
        embedding_schema = self.load_schema(schemas_dir, "embedding", "v1")

        # Test required fields
        required_fields = embedding_schema["required"]
        assert "kind" in required_fields
        assert "schema_version" in required_fields
        assert "payload" in required_fields

        # Test payload required fields
        payload_required = embedding_schema["properties"]["payload"]["required"]
        assert "dim" in payload_required

        # Test kind enum
        kind_enum = embedding_schema["properties"]["kind"]["enum"]
        assert kind_enum == ["embedding"]

        # Test schema_version enum
        version_enum = embedding_schema["properties"]["schema_version"]["enum"]
        assert version_enum == [1]

        # Test text schema compliance
        text_schema = self.load_schema(schemas_dir, "text", "v1")
        assert text_schema["properties"]["kind"]["enum"] == ["text"]
        assert text_schema["properties"]["schema_version"]["enum"] == [1]
        assert "text" in text_schema["properties"]["payload"]["required"]

        # Test tokens schema compliance
        tokens_schema = self.load_schema(schemas_dir, "tokens", "v1")
        assert tokens_schema["properties"]["kind"]["enum"] == ["tokens"]
        assert tokens_schema["properties"]["schema_version"]["enum"] == [1]
        assert "ids" in tokens_schema["properties"]["payload"]["required"]
        assert "vocab" in tokens_schema["properties"]["metadata"]["required"]

    def test_embedding_model_validation_coverage(self) -> None:
        """Test EmbeddingModel validation to ensure full coverage of model_post_init."""
        # Test valid case - values length matches dim
        EmbeddingModel(values=[1.0, 2.0, 3.0], dim=3)

        # Test valid case - values is None
        EmbeddingModel(values=None, dim=768)

        # Test invalid case - values length doesn't match dim
        with pytest.raises(ValueError, match="Values length \\(2\\) must match dim \\(3\\)"):
            EmbeddingModel(values=[1.0, 2.0], dim=3)

        # Test invalid case - values length is longer than dim
        with pytest.raises(ValueError, match="Values length \\(4\\) must match dim \\(3\\)"):
            EmbeddingModel(values=[1.0, 2.0, 3.0, 4.0], dim=3)
