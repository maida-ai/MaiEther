"""Tests for EmbeddingModel registration and conversion."""

import json

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

from ether.attachment import Attachment
from ether.core import Ether, _spec_registry
from ether.kinds import EmbeddingModel
from tests.kinds import SCHEMAS_DIR


class TestEmbeddingModelRegistration:
    """Test EmbeddingModel registration with Ether."""

    def test_embedding_model_registration(self) -> None:
        """Test that EmbeddingModel is properly registered with Ether."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register EmbeddingModel since registration happens at class definition time
        from ether.kinds import EmbeddingModel
        from ether.spec import EtherSpec

        # Manually register EmbeddingModel
        spec = EtherSpec(
            payload_fields=("values", "dim"),
            metadata_fields=("source",),
            extra_fields="ignore",
            renames={},
            kind="embedding",
        )
        _spec_registry[EmbeddingModel] = spec

        # Verify registration
        assert EmbeddingModel in _spec_registry
        spec = _spec_registry[EmbeddingModel]
        assert spec.kind == "embedding"
        assert spec.payload_fields == ("values", "dim")
        assert spec.metadata_fields == ("source",)
        assert spec.extra_fields == "ignore"

    def test_embedding_model_round_trip_conversion(self) -> None:
        """Test round-trip conversion: EmbeddingModel -> Ether -> EmbeddingModel."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register EmbeddingModel
        from ether.kinds import EmbeddingModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("values", "dim"),
            metadata_fields=("source",),
            extra_fields="ignore",
            renames={},
            kind="embedding",
        )
        _spec_registry[EmbeddingModel] = spec

        # Create EmbeddingModel instance with inline values
        original_model = EmbeddingModel(values=[1.0, 2.0, 3.0], dim=3, source="test-model")

        # Convert to Ether
        ether = Ether.from_model(original_model)

        # Verify Ether properties
        assert ether.kind == "embedding"
        assert ether.schema_version == 1
        assert ether.payload == {"values": [1.0, 2.0, 3.0], "dim": 3}
        # Metadata should contain both user-provided and auto-populated fields
        assert ether.metadata["source"] == "test-model"
        assert "trace_id" in ether.metadata
        assert "created_at" in ether.metadata
        assert ether.extra_fields == {}
        assert ether._source_model == EmbeddingModel

        # Convert back to EmbeddingModel
        converted_model = ether.as_model(EmbeddingModel)

        # Verify round-trip conversion
        assert converted_model.values == original_model.values
        assert converted_model.dim == original_model.dim
        assert converted_model.source == original_model.source

    def test_embedding_model_with_none_values(self) -> None:
        """Test EmbeddingModel with values=None (for attachment-based transport)."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register EmbeddingModel
        from ether.kinds import EmbeddingModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("values", "dim"),
            metadata_fields=("source",),
            extra_fields="ignore",
            renames={},
            kind="embedding",
        )
        _spec_registry[EmbeddingModel] = spec

        # Create EmbeddingModel with None values
        original_model = EmbeddingModel(values=None, dim=768, source="bert-base-uncased")

        # Convert to Ether
        ether = Ether.from_model(original_model)

        # Verify Ether properties
        assert ether.kind == "embedding"
        assert ether.payload == {"values": None, "dim": 768}
        # Metadata should contain both user-provided and auto-populated fields
        assert ether.metadata["source"] == "bert-base-uncased"
        assert "trace_id" in ether.metadata
        assert "created_at" in ether.metadata

        # Convert back to EmbeddingModel
        converted_model = ether.as_model(EmbeddingModel)

        # Verify conversion
        assert converted_model.values is None
        assert converted_model.dim == 768
        assert converted_model.source == "bert-base-uncased"

    def test_embedding_model_minimal_fields(self) -> None:
        """Test EmbeddingModel with only required fields."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register EmbeddingModel
        from ether.kinds import EmbeddingModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("values", "dim"),
            metadata_fields=("source",),
            extra_fields="ignore",
            renames={},
            kind="embedding",
        )
        _spec_registry[EmbeddingModel] = spec

        # Create EmbeddingModel with only dim field
        original_model = EmbeddingModel(dim=512)

        # Convert to Ether
        ether = Ether.from_model(original_model)

        # Verify Ether properties
        assert ether.kind == "embedding"
        assert ether.payload == {"values": None, "dim": 512}
        # Metadata should contain both user-provided and auto-populated fields
        assert ether.metadata["source"] is None
        assert "trace_id" in ether.metadata
        assert "created_at" in ether.metadata

        # Convert back to EmbeddingModel
        converted_model = ether.as_model(EmbeddingModel)

        # Verify conversion
        assert converted_model.values is None
        assert converted_model.dim == 512
        assert converted_model.source is None

    def test_embedding_model_constructor_with_model(self) -> None:
        """Test Ether constructor with EmbeddingModel."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register EmbeddingModel
        from ether.kinds import EmbeddingModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("values", "dim"),
            metadata_fields=("source",),
            extra_fields="ignore",
            renames={},
            kind="embedding",
        )
        _spec_registry[EmbeddingModel] = spec

        # Create EmbeddingModel and use Ether constructor
        model = EmbeddingModel(values=[1.0, 2.0], dim=2, source="test")
        ether = Ether(model)

        # Verify conversion
        assert ether.kind == "embedding"
        assert ether.payload == {"values": [1.0, 2.0], "dim": 2}
        # Metadata should contain both user-provided and auto-populated fields
        assert ether.metadata["source"] == "test"
        assert "trace_id" in ether.metadata
        assert "created_at" in ether.metadata

    def test_embedding_model_require_kind_validation(self) -> None:
        """Test require_kind validation with EmbeddingModel."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register EmbeddingModel
        from ether.kinds import EmbeddingModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("values", "dim"),
            metadata_fields=("source",),
            extra_fields="ignore",
            renames={},
            kind="embedding",
        )
        _spec_registry[EmbeddingModel] = spec

        # Create EmbeddingModel and convert to Ether
        model = EmbeddingModel(values=[1.0], dim=1)
        ether = Ether.from_model(model)

        # Should succeed with require_kind=True (same kind)
        converted = ether.as_model(EmbeddingModel, require_kind=True)
        assert converted.values == [1.0]
        assert converted.dim == 1

        # Create Ether with different kind
        wrong_kind_ether = Ether(
            kind="text", schema_version=1, payload={"values": [1.0], "dim": 1}, metadata={}  # Different kind
        )

        # Should fail with require_kind=True
        with pytest.raises(Exception) as exc_info:
            wrong_kind_ether.as_model(EmbeddingModel, require_kind=True)
        assert "Kind mismatch" in str(exc_info.value)

    def test_embedding_model_produces_valid_schema_envelope(self) -> None:
        """Test that EmbeddingModel produces Ether envelopes that validate against embedding.v1 schema."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register EmbeddingModel
        from ether.kinds import EmbeddingModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("values", "dim"),
            metadata_fields=("source",),
            extra_fields="ignore",
            renames={},
            kind="embedding",
        )
        _spec_registry[EmbeddingModel] = spec

        # Load the embedding.v1 schema
        schema_path = SCHEMAS_DIR / "embedding" / "v1.json"
        with open(schema_path) as f:
            schema = json.load(f)
        validator = Draft202012Validator(schema)

        # Create EmbeddingModel and convert to Ether
        model = EmbeddingModel(values=[1.0, 2.0, 3.0], dim=3, source="test-model")
        ether = Ether.from_model(model)

        # Convert Ether to dict for schema validation
        ether_dict = ether.model_dump()

        # Validate against embedding.v1 schema
        errors = list(validator.iter_errors(ether_dict))
        assert not errors, f"Schema validation errors: {errors}"

        # Verify key schema requirements
        assert ether_dict["kind"] == "embedding"
        assert ether_dict["schema_version"] == 1
        assert "dim" in ether_dict["payload"]
        assert isinstance(ether_dict["payload"]["dim"], int)

    def test_embedding_model_strict_type_compliance(self) -> None:
        """Test that EmbeddingModel strictly complies with embedding.v1 schema types."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register EmbeddingModel
        from ether.kinds import EmbeddingModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("values", "dim"),
            metadata_fields=("source",),
            extra_fields="ignore",
            renames={},
            kind="embedding",
        )
        _spec_registry[EmbeddingModel] = spec

        # Test 1: Required dim field must be positive integer
        model = EmbeddingModel(dim=768)
        ether = Ether.from_model(model)
        assert isinstance(ether.payload["dim"], int)
        assert ether.payload["dim"] > 0

        # Test 2: Optional values field must be list[float] or None
        model_with_values = EmbeddingModel(values=[1.0, 2.0, 3.0], dim=3)
        ether_with_values = Ether.from_model(model_with_values)
        assert isinstance(ether_with_values.payload["values"], list)
        assert all(isinstance(x, float) for x in ether_with_values.payload["values"])

        model_without_values = EmbeddingModel(values=None, dim=768)
        ether_without_values = Ether.from_model(model_without_values)
        assert ether_without_values.payload["values"] is None

        # Test 3: Optional source field must be string or None
        model_with_source = EmbeddingModel(dim=512, source="bert-base-uncased")
        ether_with_source = Ether.from_model(model_with_source)
        assert isinstance(ether_with_source.metadata["source"], str)

        model_without_source = EmbeddingModel(dim=512, source=None)
        ether_without_source = Ether.from_model(model_without_source)
        assert ether_without_source.metadata["source"] is None

        # Test 4: Kind must be exactly "embedding"
        assert ether.kind == "embedding"
        assert ether_with_values.kind == "embedding"
        assert ether_with_source.kind == "embedding"

        # Test 5: Schema version must be exactly 1
        assert ether.schema_version == 1
        assert ether_with_values.schema_version == 1
        assert ether_with_source.schema_version == 1

        # Test 6: Payload must contain exactly the required fields
        assert "dim" in ether.payload
        assert "values" in ether.payload
        assert len(ether.payload) == 2  # Only dim and values fields

        # Test 7: Metadata must contain exactly the optional fields (even if None)
        assert "source" in ether.metadata
        # Metadata should contain both user-provided and auto-populated fields
        assert "source" in ether.metadata
        assert "trace_id" in ether.metadata
        assert "created_at" in ether.metadata

    def test_embedding_model_binding_mechanism_compliance(self) -> None:
        """Test that EmbeddingModel follows the binding mechanism matrix requirements."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register EmbeddingModel
        from ether.kinds import EmbeddingModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("values", "dim"),
            metadata_fields=("source",),
            extra_fields="ignore",
            renames={},
            kind="embedding",
        )
        _spec_registry[EmbeddingModel] = spec

        # Verify the binding mechanism matrix compliance:
        # | Canonical JSON-Schema file  | Matching edge model    | Binding mechanism                        |
        # | schemas/embedding/v1.json   | EmbeddingModel (Pydantic) | @Ether.register(..., kind="embedding") |

        # Test 1: EmbeddingModel is the matching edge model for schemas/embedding/v1.json
        model = EmbeddingModel(dim=768, source="test-model")
        ether = Ether.from_model(model)

        # Test 2: Binding mechanism uses @Ether.register(..., kind="embedding")
        assert ether.kind == "embedding"  # This comes from the registration

        # Test 3: The model produces envelopes that validate against the canonical schema
        schema_path = SCHEMAS_DIR / "embedding" / "v1.json"
        with open(schema_path) as f:
            schema = json.load(f)
        validator = Draft202012Validator(schema)

        # Create a model with all optional fields set to None
        model_with_nulls = EmbeddingModel(values=None, dim=768, source=None)
        ether_with_nulls = Ether.from_model(model_with_nulls)

        # Convert to dict and remove null values to match schema expectations
        ether_dict = ether_with_nulls.model_dump()

        # Remove null values from metadata to match schema expectations
        # The schema expects optional fields to be omitted, not null
        ether_dict["metadata"] = {k: v for k, v in ether_dict["metadata"].items() if v is not None}

        errors = list(validator.iter_errors(ether_dict))
        assert not errors, f"Binding mechanism validation failed: {errors}"

        # Test 4: Verify the schema file exists and is the canonical one
        assert schema_path.exists(), f"Canonical schema file {schema_path} must exist"

        # Test 5: Verify the schema has the expected structure
        assert schema["properties"]["kind"]["enum"] == ["embedding"]
        assert schema["properties"]["schema_version"]["enum"] == [1]
        assert "dim" in schema["properties"]["payload"]["properties"]

        # Test 6: Verify that a model with actual values also validates
        model_with_values = EmbeddingModel(values=[1.0, 2.0, 3.0], dim=3, source="test-model")
        ether_with_values = Ether.from_model(model_with_values)
        ether_dict_with_values = ether_with_values.model_dump()

        # Remove null values from metadata
        ether_dict_with_values["metadata"] = {
            k: v for k, v in ether_dict_with_values["metadata"].items() if v is not None
        }

        errors_with_values = list(validator.iter_errors(ether_dict_with_values))
        assert not errors_with_values, f"Binding mechanism validation with values failed: {errors_with_values}"


class TestEmbeddingModelMisRegistration:
    """Test mis-registration scenarios for EmbeddingModel."""

    def test_missing_required_field_raises_error(self) -> None:
        """Test that missing required field raises RegistrationError."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Try to register a model with missing required field
        with pytest.raises(Exception) as exc_info:

            @Ether.register(
                payload=["values", "dim", "missing_field"], metadata=[], kind="embedding"
            )  # missing_field doesn't exist
            class InvalidEmbeddingModel(BaseModel):
                values: list[float] | None = None
                dim: int
                # missing_field is not defined

        assert "unknown field" in str(exc_info.value).lower()

    def test_duplicate_field_mapping_raises_error(self) -> None:
        """Test that duplicate field mapping raises error."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Try to register with different field names mapping to the same path
        with pytest.raises(Exception) as exc_info:

            @Ether.register(
                payload=["values", "dim"],
                metadata=[],
                renames={
                    "values": "payload.values",
                    "dim": "payload.values",  # Different fields mapping to same path
                },
                kind="embedding",
            )
            class DuplicatePathModel(BaseModel):
                values: list[float] | None = None
                dim: int

        assert "duplicate mapping" in str(exc_info.value).lower()

    def test_field_in_both_payload_and_metadata_raises_error(self) -> None:
        """Test that field in both payload and metadata raises error."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Try to register with same field in both payload and metadata
        with pytest.raises(Exception) as exc_info:

            @Ether.register(payload=["values"], metadata=["values"], kind="embedding")  # Same field in both
            class DuplicateFieldModel(BaseModel):
                values: list[float] | None = None

        assert "fields in both payload & metadata" in str(exc_info.value).lower()


class TestEmbeddingModelValidation:
    """Test EmbeddingModel field validation."""

    def test_dim_must_be_positive(self) -> None:
        """Test that dim must be a positive integer."""
        # Test valid values
        EmbeddingModel(dim=1)
        EmbeddingModel(dim=768)
        EmbeddingModel(dim=1000000)

        # Test invalid values
        with pytest.raises(ValidationError):  # No zero dimension
            EmbeddingModel(dim=0)

        with pytest.raises(ValidationError):  # No negative dimension
            EmbeddingModel(dim=-1)

        with pytest.raises(ValidationError):  # No non-integer dimension
            EmbeddingModel(dim=1.5)

    def test_values_length_must_match_dim(self) -> None:
        """Test that values length must match dim when values is provided."""
        # Test valid cases
        EmbeddingModel(values=[1.0, 2.0, 3.0], dim=3)
        EmbeddingModel(values=[1.0], dim=1)
        EmbeddingModel(values=None, dim=768)  # None values is always valid

        # Test invalid cases
        with pytest.raises(ValueError):  # Length mismatch
            EmbeddingModel(values=[1.0, 2.0], dim=3)

        with pytest.raises(ValueError):  # Length mismatch
            EmbeddingModel(values=[1.0, 2.0, 3.0, 4.0], dim=3)

    def test_values_must_be_floats(self) -> None:
        """Test that values must be list[float] when provided."""
        # Test valid cases
        EmbeddingModel(values=[1.0, 2.0, 3.0], dim=3)
        EmbeddingModel(values=[0.0, -1.5, 2.7], dim=3)

        # Test invalid cases
        with pytest.raises(ValidationError):  # Mixed types
            EmbeddingModel(values=[1.0, "string", 3.0], dim=3)

        with pytest.raises(ValidationError):  # Non-list
            EmbeddingModel(values="not a list", dim=1)

    def test_embedding_model_extra_fields_ignored(self) -> None:
        """Test that extra fields are ignored in EmbeddingModel registration."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Create a model with extra fields
        @Ether.register(payload=["values", "dim"], metadata=["source"], extra_fields="ignore", kind="embedding")
        class EmbeddingModelWithExtra(BaseModel):
            values: list[float] | None = None
            dim: int
            source: str | None = None
            extra_field: str = "ignored"

        # Convert to Ether
        model = EmbeddingModelWithExtra(values=[1.0], dim=1, extra_field="should_be_ignored")
        ether = Ether.from_model(model)

        # Verify extra field is not in payload or metadata
        assert "extra_field" not in ether.payload
        assert "extra_field" not in ether.metadata
        assert ether.extra_fields == {}  # Should be empty due to "ignore"


class TestEmbeddingModelWithAttachments:
    """Test EmbeddingModel with attachments for large vectors."""

    def test_embedding_model_with_attachment(self) -> None:
        """Test creating model with values=None and an Attachment passes conversion."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register EmbeddingModel
        from ether.kinds import EmbeddingModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("values", "dim"),
            metadata_fields=("source",),
            extra_fields="ignore",
            renames={},
            kind="embedding",
        )
        _spec_registry[EmbeddingModel] = spec

        # Create attachment for large embedding vector
        attachment = Attachment(
            id="embedding-0",
            uri="shm://embeddings/12345",
            media_type="application/x-raw-tensor",
            codec="RAW_F32",
            shape=[768],
            dtype="float32",
            size_bytes=3072,  # 768 * 4 bytes per float32
        )

        # Create EmbeddingModel with None values and attachment
        model = EmbeddingModel(values=None, dim=768, source="bert-base-uncased")

        # Convert to Ether and add attachment
        ether = Ether.from_model(model)
        ether.attachments = [attachment]

        # Verify Ether properties
        assert ether.kind == "embedding"
        assert ether.payload == {"values": None, "dim": 768}
        # Metadata should contain both user-provided and auto-populated fields
        assert ether.metadata["source"] == "bert-base-uncased"
        assert "trace_id" in ether.metadata
        assert "created_at" in ether.metadata
        assert len(ether.attachments) == 1
        assert ether.attachments[0].id == "embedding-0"
        assert ether.attachments[0].codec == "RAW_F32"
        assert ether.attachments[0].shape == [768]

        # Convert back to EmbeddingModel
        converted_model = ether.as_model(EmbeddingModel)

        # Verify conversion
        assert converted_model.values is None
        assert converted_model.dim == 768
        assert converted_model.source == "bert-base-uncased"

    def test_embedding_model_inline_list_round_trip(self) -> None:
        """Test creating model with inline list, round-trips."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register EmbeddingModel
        from ether.kinds import EmbeddingModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("values", "dim"),
            metadata_fields=("source",),
            extra_fields="ignore",
            renames={},
            kind="embedding",
        )
        _spec_registry[EmbeddingModel] = spec

        # Create EmbeddingModel with inline values
        original_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        model = EmbeddingModel(values=original_values, dim=5, source="test-model")

        # Convert to Ether
        ether = Ether.from_model(model)

        # Verify Ether properties
        assert ether.kind == "embedding"
        assert ether.payload == {"values": original_values, "dim": 5}
        # Metadata should contain both user-provided and auto-populated fields
        assert ether.metadata["source"] == "test-model"
        assert "trace_id" in ether.metadata
        assert "created_at" in ether.metadata

        # Convert back to EmbeddingModel
        converted_model = ether.as_model(EmbeddingModel)

        # Verify round-trip conversion
        assert converted_model.values == original_values
        assert converted_model.dim == 5
        assert converted_model.source == "test-model"

        # Verify the values are actually floats
        assert all(isinstance(x, float) for x in converted_model.values)
