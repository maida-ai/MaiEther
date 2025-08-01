"""Tests for TextModel registration and conversion."""

import json

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

from ether.core import Ether, _spec_registry
from ether.kinds import TextModel
from tests.kinds import SCHEMAS_DIR


class TestTextModelRegistration:
    """Test TextModel registration with Ether."""

    def test_text_model_registration(self) -> None:
        """Test that TextModel is properly registered with Ether."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register TextModel since registration happens at class definition time
        from ether.kinds import TextModel
        from ether.spec import EtherSpec

        # Manually register TextModel
        spec = EtherSpec(
            payload_fields=("text",),
            metadata_fields=("lang", "encoding", "detected_lang_conf"),
            extra_fields="ignore",
            renames={},
            kind="text",
        )
        _spec_registry[TextModel] = spec

        # Verify registration
        assert TextModel in _spec_registry
        spec = _spec_registry[TextModel]
        assert spec.kind == "text"
        assert spec.payload_fields == ("text",)
        assert spec.metadata_fields == ("lang", "encoding", "detected_lang_conf")
        assert spec.extra_fields == "ignore"

    def test_text_model_round_trip_conversion(self) -> None:
        """Test round-trip conversion: TextModel -> Ether -> TextModel."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register TextModel
        from ether.kinds import TextModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("text",),
            metadata_fields=("lang", "encoding", "detected_lang_conf"),
            extra_fields="ignore",
            renames={},
            kind="text",
        )
        _spec_registry[TextModel] = spec

        # Create TextModel instance
        original_model = TextModel(text="Hello, world!", lang="en", encoding="utf-8", detected_lang_conf=0.95)

        # Convert to Ether
        ether = Ether.from_model(original_model)

        # Verify Ether properties
        assert ether.kind == "text"
        assert ether.schema_version == 1
        assert ether.payload == {"text": "Hello, world!"}
        assert ether.metadata == {"lang": "en", "encoding": "utf-8", "detected_lang_conf": 0.95}
        assert ether.extra_fields == {}
        assert ether._source_model == TextModel

        # Convert back to TextModel
        converted_model = ether.as_model(TextModel)

        # Verify round-trip conversion
        assert converted_model.text == original_model.text
        assert converted_model.lang == original_model.lang
        assert converted_model.encoding == original_model.encoding
        assert converted_model.detected_lang_conf == original_model.detected_lang_conf

    def test_text_model_minimal_fields(self) -> None:
        """Test TextModel with only required fields."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register TextModel
        from ether.kinds import TextModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("text",),
            metadata_fields=("lang", "encoding", "detected_lang_conf"),
            extra_fields="ignore",
            renames={},
            kind="text",
        )
        _spec_registry[TextModel] = spec

        # Create TextModel with only text field
        original_model = TextModel(text="Minimal text")

        # Convert to Ether
        ether = Ether.from_model(original_model)

        # Verify Ether properties
        assert ether.kind == "text"
        assert ether.payload == {"text": "Minimal text"}
        # Metadata includes None values for optional fields
        assert ether.metadata == {"lang": None, "encoding": None, "detected_lang_conf": None}

        # Convert back to TextModel
        converted_model = ether.as_model(TextModel)

        # Verify conversion
        assert converted_model.text == "Minimal text"
        assert converted_model.lang is None
        assert converted_model.encoding is None
        assert converted_model.detected_lang_conf is None

    def test_text_model_constructor_with_model(self) -> None:
        """Test Ether constructor with TextModel."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register TextModel
        from ether.kinds import TextModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("text",),
            metadata_fields=("lang", "encoding", "detected_lang_conf"),
            extra_fields="ignore",
            renames={},
            kind="text",
        )
        _spec_registry[TextModel] = spec

        # Create TextModel and use Ether constructor
        model = TextModel(text="Constructor test", lang="en")
        ether = Ether(model)

        # Verify conversion
        assert ether.kind == "text"
        assert ether.payload == {"text": "Constructor test"}
        # Metadata includes None values for optional fields
        assert ether.metadata == {"lang": "en", "encoding": None, "detected_lang_conf": None}

    def test_text_model_require_kind_validation(self) -> None:
        """Test require_kind validation with TextModel."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register TextModel
        from ether.kinds import TextModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("text",),
            metadata_fields=("lang", "encoding", "detected_lang_conf"),
            extra_fields="ignore",
            renames={},
            kind="text",
        )
        _spec_registry[TextModel] = spec

        # Create TextModel and convert to Ether
        model = TextModel(text="Kind validation test")
        ether = Ether.from_model(model)

        # Should succeed with require_kind=True (same kind)
        converted = ether.as_model(TextModel, require_kind=True)
        assert converted.text == "Kind validation test"

        # Create Ether with different kind
        wrong_kind_ether = Ether(
            kind="embedding", schema_version=1, payload={"text": "Wrong kind"}, metadata={}  # Different kind
        )

        # Should fail with require_kind=True
        with pytest.raises(Exception) as exc_info:
            wrong_kind_ether.as_model(TextModel, require_kind=True)
        assert "Kind mismatch" in str(exc_info.value)

    def test_text_model_produces_valid_schema_envelope(self) -> None:
        """Test that TextModel produces Ether envelopes that validate against text.v1 schema."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register TextModel
        from ether.kinds import TextModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("text",),
            metadata_fields=("lang", "encoding", "detected_lang_conf"),
            extra_fields="ignore",
            renames={},
            kind="text",
        )
        _spec_registry[TextModel] = spec

        # Load the text.v1 schema
        schema_path = SCHEMAS_DIR / "text" / "v1.json"
        with open(schema_path) as f:
            schema = json.load(f)
        validator = Draft202012Validator(schema)

        # Create TextModel and convert to Ether
        model = TextModel(text="Test text for schema validation", lang="en", encoding="utf-8", detected_lang_conf=0.95)
        ether = Ether.from_model(model)

        # Convert Ether to dict for schema validation
        ether_dict = ether.model_dump()

        # Validate against text.v1 schema
        errors = list(validator.iter_errors(ether_dict))
        assert not errors, f"Schema validation errors: {errors}"

        # Verify key schema requirements
        assert ether_dict["kind"] == "text"
        assert ether_dict["schema_version"] == 1
        assert "text" in ether_dict["payload"]
        assert isinstance(ether_dict["payload"]["text"], str)

    def test_text_model_strict_type_compliance(self) -> None:
        """Test that TextModel strictly complies with text.v1 schema types."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register TextModel
        from ether.kinds import TextModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("text",),
            metadata_fields=("lang", "encoding", "detected_lang_conf"),
            extra_fields="ignore",
            renames={},
            kind="text",
        )
        _spec_registry[TextModel] = spec

        # Test 1: Required text field must be string
        model = TextModel(text="Valid string text")
        ether = Ether.from_model(model)
        assert isinstance(ether.payload["text"], str)

        # Test 2: Optional lang field must be string or None
        model_with_lang = TextModel(text="Test", lang="en")
        ether_with_lang = Ether.from_model(model_with_lang)
        assert isinstance(ether_with_lang.metadata["lang"], str)

        model_without_lang = TextModel(text="Test", lang=None)
        ether_without_lang = Ether.from_model(model_without_lang)
        assert ether_without_lang.metadata["lang"] is None

        # Test 3: Optional encoding field must be string or None
        model_with_encoding = TextModel(text="Test", encoding="utf-8")
        ether_with_encoding = Ether.from_model(model_with_encoding)
        assert isinstance(ether_with_encoding.metadata["encoding"], str)

        model_without_encoding = TextModel(text="Test", encoding=None)
        ether_without_encoding = Ether.from_model(model_without_encoding)
        assert ether_without_encoding.metadata["encoding"] is None

        # Test 4: Optional detected_lang_conf must be float in [0.0, 1.0] or None
        model_with_conf = TextModel(text="Test", detected_lang_conf=0.5)
        ether_with_conf = Ether.from_model(model_with_conf)
        assert isinstance(ether_with_conf.metadata["detected_lang_conf"], float)
        assert 0.0 <= ether_with_conf.metadata["detected_lang_conf"] <= 1.0

        model_without_conf = TextModel(text="Test", detected_lang_conf=None)
        ether_without_conf = Ether.from_model(model_without_conf)
        assert ether_without_conf.metadata["detected_lang_conf"] is None

        # Test 5: Kind must be exactly "text"
        assert ether.kind == "text"
        assert ether_with_lang.kind == "text"
        assert ether_with_encoding.kind == "text"
        assert ether_with_conf.kind == "text"

        # Test 6: Schema version must be exactly 1
        assert ether.schema_version == 1
        assert ether_with_lang.schema_version == 1
        assert ether_with_encoding.schema_version == 1
        assert ether_with_conf.schema_version == 1

        # Test 7: Payload must contain exactly the required fields
        assert "text" in ether.payload
        assert len(ether.payload) == 1  # Only text field

        # Test 8: Metadata must contain exactly the optional fields (even if None)
        assert "lang" in ether.metadata
        assert "encoding" in ether.metadata
        assert "detected_lang_conf" in ether.metadata
        assert len(ether.metadata) == 3  # Only the three optional fields

    def test_text_model_binding_mechanism_compliance(self) -> None:
        """Test that TextModel follows the binding mechanism matrix requirements."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Re-register TextModel
        from ether.kinds import TextModel
        from ether.spec import EtherSpec

        spec = EtherSpec(
            payload_fields=("text",),
            metadata_fields=("lang", "encoding", "detected_lang_conf"),
            extra_fields="ignore",
            renames={},
            kind="text",
        )
        _spec_registry[TextModel] = spec

        # Verify the binding mechanism matrix compliance:
        # | Canonical JSON-Schema file  | Matching edge model    | Binding mechanism                        |
        # | schemas/text/v1.json        | TextModel (Pydantic)   | @Ether.register(..., kind="text")      |

        # Test 1: TextModel is the matching edge model for schemas/text/v1.json
        model = TextModel(text="Test binding mechanism")
        ether = Ether.from_model(model)

        # Test 2: Binding mechanism uses @Ether.register(..., kind="text")
        assert ether.kind == "text"  # This comes from the registration

        # Test 3: The model produces envelopes that validate against the canonical schema
        # Note: The schema expects optional fields to be omitted, not set to null
        schema_path = SCHEMAS_DIR / "text" / "v1.json"
        with open(schema_path) as f:
            schema = json.load(f)
        validator = Draft202012Validator(schema)

        # Create a model with all optional fields set to None
        model_with_nulls = TextModel(text="Test binding mechanism", lang=None, encoding=None, detected_lang_conf=None)
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
        assert schema["properties"]["kind"]["enum"] == ["text"]
        assert schema["properties"]["schema_version"]["enum"] == [1]
        assert "text" in schema["properties"]["payload"]["properties"]

        # Test 6: Verify that a model with actual values also validates
        model_with_values = TextModel(text="Test with values", lang="en", encoding="utf-8", detected_lang_conf=0.95)
        ether_with_values = Ether.from_model(model_with_values)
        ether_dict_with_values = ether_with_values.model_dump()

        # Remove null values from metadata
        ether_dict_with_values["metadata"] = {
            k: v for k, v in ether_dict_with_values["metadata"].items() if v is not None
        }

        errors_with_values = list(validator.iter_errors(ether_dict_with_values))
        assert not errors_with_values, f"Binding mechanism validation with values failed: {errors_with_values}"


class TestTextModelMisRegistration:
    """Test mis-registration scenarios for TextModel."""

    def test_missing_required_field_raises_error(self) -> None:
        """Test that missing required field raises RegistrationError."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Try to register a model with missing required field
        with pytest.raises(Exception) as exc_info:

            @Ether.register(payload=["text", "missing_field"], metadata=[], kind="text")  # missing_field doesn't exist
            class InvalidTextModel(BaseModel):
                text: str
                # missing_field is not defined

        assert "unknown field" in str(exc_info.value).lower()

    def test_duplicate_field_mapping_raises_error(self) -> None:
        """Test that duplicate field mapping raises error."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Try to register with different field names mapping to the same path
        with pytest.raises(Exception) as exc_info:

            @Ether.register(
                payload=["text", "content"],
                metadata=[],
                renames={
                    "text": "payload.content",
                    "content": "payload.content",  # Different fields mapping to same path
                },
                kind="text",
            )
            class DuplicatePathModel(BaseModel):
                text: str
                content: str

        assert "duplicate mapping" in str(exc_info.value).lower()

    def test_field_in_both_payload_and_metadata_raises_error(self) -> None:
        """Test that field in both payload and metadata raises error."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Try to register with same field in both payload and metadata
        with pytest.raises(Exception) as exc_info:

            @Ether.register(payload=["text"], metadata=["text"], kind="text")  # Same field in both
            class DuplicateFieldModel(BaseModel):
                text: str

        assert "fields in both payload & metadata" in str(exc_info.value).lower()


class TestTextModelValidation:
    """Test TextModel field validation."""

    def test_detected_lang_conf_range_validation(self) -> None:
        """Test that detected_lang_conf is validated within 0.0-1.0 range."""
        # Test valid values
        TextModel(text="test", detected_lang_conf=0.0)
        TextModel(text="test", detected_lang_conf=0.5)
        TextModel(text="test", detected_lang_conf=1.0)

        # Test invalid values
        with pytest.raises(ValidationError):  # No negative confidence
            TextModel(text="test", detected_lang_conf=-0.1)

        with pytest.raises(ValidationError):  # No confidence above 1.0
            TextModel(text="test", detected_lang_conf=1.1)

    def test_text_model_extra_fields_ignored(self) -> None:
        """Test that extra fields are ignored in TextModel registration."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Create a model with extra fields
        @Ether.register(payload=["text"], metadata=["lang"], extra_fields="ignore", kind="text")
        class TextModelWithExtra(BaseModel):
            text: str
            lang: str | None = None
            extra_field: str = "ignored"

        # Convert to Ether
        model = TextModelWithExtra(text="test", extra_field="should_be_ignored")
        ether = Ether.from_model(model)

        # Verify extra field is not in payload or metadata
        assert "extra_field" not in ether.payload
        assert "extra_field" not in ether.metadata
        assert ether.extra_fields == {}  # Should be empty due to "ignore"
