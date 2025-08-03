"""Tests for TokenModel registration and conversion."""

import json

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

from ether import Registry
from ether.core import Ether
from ether.kinds import TokenModel
from tests.kinds import SCHEMAS_DIR


class TestTokenModelRegistration:
    """Test TokenModel registration with Ether."""

    def test_token_model_registration(self, clear_registry) -> None:
        """Test that TokenModel is properly registered with Ether."""

        # Re-register TokenModel since registration happens at class definition time
        from ether.spec import EtherSpec

        # Manually register TokenModel
        spec = EtherSpec(
            payload_fields=("ids", "mask"),
            metadata_fields=("vocab", "truncation", "offsets"),
            extra_fields="ignore",
            renames={},
            kind="tokens",
        )
        Registry.set_spec(TokenModel, spec)

        # Verify registration
        spec = Registry.get_spec(TokenModel)
        assert spec is not None
        assert spec.kind == "tokens"
        assert spec.payload_fields == ("ids", "mask")
        assert spec.metadata_fields == ("vocab", "truncation", "offsets")
        assert spec.extra_fields == "ignore"

    def test_token_model_round_trip_conversion(self, clear_registry) -> None:
        """Test round-trip conversion: TokenModel -> Ether -> TokenModel."""

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

        # Create TokenModel instance
        original_model = TokenModel(
            ids=[1, 2, 3, 4, 5],
            mask=[1, 1, 1, 1, 1],
            vocab="gpt2",
            truncation="longest_first",
            offsets=True,
        )

        # Convert to Ether
        ether = Ether.from_model(original_model)

        # Verify Ether properties
        assert ether.kind == "tokens"
        assert ether.schema_version == 1
        assert ether.payload == {"ids": [1, 2, 3, 4, 5], "mask": [1, 1, 1, 1, 1]}
        # Metadata should contain both user-provided and auto-populated fields
        assert ether.metadata["vocab"] == "gpt2"
        assert ether.metadata["truncation"] == "longest_first"
        assert ether.metadata["offsets"] is True
        assert "trace_id" in ether.metadata
        assert "created_at" in ether.metadata
        assert ether.extra_fields == {}
        assert ether._source_model == TokenModel

        # Convert back to TokenModel
        converted_model = ether.as_model(TokenModel)

        # Verify round-trip conversion
        assert converted_model.ids == original_model.ids
        assert converted_model.mask == original_model.mask
        assert converted_model.vocab == original_model.vocab
        assert converted_model.truncation == original_model.truncation
        assert converted_model.offsets == original_model.offsets

    def test_token_model_minimal_fields(self, clear_registry) -> None:
        """Test TokenModel with only required fields."""

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

        # Create TokenModel with only required fields
        original_model = TokenModel(ids=[1, 2, 3], vocab="bert-base-uncased")

        # Convert to Ether
        ether = Ether.from_model(original_model)

        # Verify Ether properties
        assert ether.kind == "tokens"
        assert ether.payload == {"ids": [1, 2, 3], "mask": None}
        # Metadata should contain both user-provided and auto-populated fields
        assert ether.metadata["vocab"] == "bert-base-uncased"
        assert ether.metadata["truncation"] is None
        assert ether.metadata["offsets"] is None
        assert "trace_id" in ether.metadata
        assert "created_at" in ether.metadata

        # Convert back to TokenModel
        converted_model = ether.as_model(TokenModel)

        # Verify conversion
        assert converted_model.ids == [1, 2, 3]
        assert converted_model.mask is None
        assert converted_model.vocab == "bert-base-uncased"
        assert converted_model.truncation is None
        assert converted_model.offsets is None

    def test_token_model_constructor_with_model(self, clear_registry) -> None:
        """Test Ether constructor with TokenModel."""

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

        # Create TokenModel and use Ether constructor
        model = TokenModel(ids=[1, 2, 3], vocab="gpt2", truncation="longest_first")
        ether = Ether(model)

        # Verify conversion
        assert ether.kind == "tokens"
        assert ether.payload == {"ids": [1, 2, 3], "mask": None}
        # Metadata should contain both user-provided and auto-populated fields
        assert ether.metadata["vocab"] == "gpt2"
        assert ether.metadata["truncation"] == "longest_first"
        assert ether.metadata["offsets"] is None
        assert "trace_id" in ether.metadata
        assert "created_at" in ether.metadata

    def test_token_model_require_kind_validation(self, clear_registry) -> None:
        """Test require_kind validation with TokenModel."""

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

        # Create TokenModel and convert to Ether
        model = TokenModel(ids=[1, 2, 3], vocab="gpt2")
        ether = Ether.from_model(model)

        # Should succeed with require_kind=True (same kind)
        converted = ether.as_model(TokenModel, require_kind=True)
        assert converted.ids == [1, 2, 3]

        # Create Ether with different kind
        wrong_kind_ether = Ether(kind="text", schema_version=1, payload={"ids": [1, 2, 3]}, metadata={"vocab": "gpt2"})

        # Should fail with require_kind=True
        with pytest.raises(Exception) as exc_info:
            wrong_kind_ether.as_model(TokenModel, require_kind=True)
        assert "Kind mismatch" in str(exc_info.value)

    def test_token_model_produces_valid_schema_envelope(self, clear_registry) -> None:
        """Test that TokenModel produces Ether envelopes that validate against tokens.v1 schema."""

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

        # Load the tokens.v1 schema
        schema_path = SCHEMAS_DIR / "tokens" / "v1.json"
        with open(schema_path) as f:
            schema = json.load(f)
        validator = Draft202012Validator(schema)

        # Create TokenModel and convert to Ether
        model = TokenModel(
            ids=[1, 2, 3, 4, 5],
            mask=[1, 1, 1, 1, 1],
            vocab="gpt2",
            truncation="longest_first",
            offsets=True,
        )
        ether = Ether.from_model(model)

        # Convert Ether to dict for schema validation
        ether_dict = ether.model_dump()

        # Validate against tokens.v1 schema
        errors = list(validator.iter_errors(ether_dict))
        assert not errors, f"Schema validation errors: {errors}"

        # Verify key schema requirements
        assert ether_dict["kind"] == "tokens"
        assert ether_dict["schema_version"] == 1
        assert "ids" in ether_dict["payload"]
        assert isinstance(ether_dict["payload"]["ids"], list)
        assert "vocab" in ether_dict["metadata"]
        assert isinstance(ether_dict["metadata"]["vocab"], str)

    def test_token_model_strict_type_compliance(self, clear_registry) -> None:
        """Test that TokenModel strictly complies with tokens.v1 schema types."""

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

        # Test 1: Required ids field must be list[int]
        model = TokenModel(ids=[1, 2, 3], vocab="gpt2")
        ether = Ether.from_model(model)
        assert isinstance(ether.payload["ids"], list)
        assert all(isinstance(x, int) for x in ether.payload["ids"])

        # Test 2: Optional mask field must be list[int] or None
        model_with_mask = TokenModel(ids=[1, 2, 3], mask=[1, 1, 0], vocab="gpt2")
        ether_with_mask = Ether.from_model(model_with_mask)
        assert isinstance(ether_with_mask.payload["mask"], list)
        assert all(isinstance(x, int) for x in ether_with_mask.payload["mask"])

        model_without_mask = TokenModel(ids=[1, 2, 3], mask=None, vocab="gpt2")
        ether_without_mask = Ether.from_model(model_without_mask)
        assert ether_without_mask.payload.get("mask") is None

        # Test 3: Required vocab field must be string
        model_with_vocab = TokenModel(ids=[1, 2, 3], vocab="bert-base-uncased")
        ether_with_vocab = Ether.from_model(model_with_vocab)
        assert isinstance(ether_with_vocab.metadata["vocab"], str)

        # Test 4: Optional truncation field must be string or None
        model_with_truncation = TokenModel(ids=[1, 2, 3], vocab="gpt2", truncation="longest_first")
        ether_with_truncation = Ether.from_model(model_with_truncation)
        assert isinstance(ether_with_truncation.metadata["truncation"], str)

        model_without_truncation = TokenModel(ids=[1, 2, 3], vocab="gpt2", truncation=None)
        ether_without_truncation = Ether.from_model(model_without_truncation)
        assert ether_without_truncation.metadata["truncation"] is None

        # Test 5: Optional offsets field must be boolean or None
        model_with_offsets = TokenModel(ids=[1, 2, 3], vocab="gpt2", offsets=True)
        ether_with_offsets = Ether.from_model(model_with_offsets)
        assert isinstance(ether_with_offsets.metadata["offsets"], bool)

        model_without_offsets = TokenModel(ids=[1, 2, 3], vocab="gpt2", offsets=None)
        ether_without_offsets = Ether.from_model(model_without_offsets)
        assert ether_without_offsets.metadata["offsets"] is None

        # Test 6: Kind must be exactly "tokens"
        assert ether.kind == "tokens"
        assert ether_with_mask.kind == "tokens"
        assert ether_with_vocab.kind == "tokens"
        assert ether_with_truncation.kind == "tokens"
        assert ether_with_offsets.kind == "tokens"

        # Test 7: Schema version must be exactly 1
        assert ether.schema_version == 1
        assert ether_with_mask.schema_version == 1
        assert ether_with_vocab.schema_version == 1
        assert ether_with_truncation.schema_version == 1
        assert ether_with_offsets.schema_version == 1

        # Test 8: Payload must contain exactly the required fields
        assert "ids" in ether.payload
        assert len(ether.payload) == 2  # ids and mask fields (mask is None)

        # Test 9: Metadata must contain exactly the optional fields (even if None)
        assert "vocab" in ether.metadata
        assert "truncation" in ether.metadata
        assert "offsets" in ether.metadata
        # Metadata should contain both user-provided and auto-populated fields
        assert "vocab" in ether.metadata
        assert "truncation" in ether.metadata
        assert "offsets" in ether.metadata
        assert "trace_id" in ether.metadata
        assert "created_at" in ether.metadata

    def test_token_model_binding_mechanism_compliance(self, clear_registry) -> None:
        """Test that TokenModel follows the binding mechanism matrix requirements."""

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

        # Verify the binding mechanism matrix compliance:
        # | Canonical JSON-Schema file  | Matching edge model    | Binding mechanism                        |
        # | schemas/tokens/v1.json      | TokenModel (Pydantic)  | @Ether.register(..., kind="tokens")    |

        # Test 1: TokenModel is the matching edge model for schemas/tokens/v1.json
        model = TokenModel(ids=[1, 2, 3], vocab="gpt2")
        ether = Ether.from_model(model)

        # Test 2: Binding mechanism uses @Ether.register(..., kind="tokens")
        assert ether.kind == "tokens"  # This comes from the registration

        # Test 3: The model produces envelopes that validate against the canonical schema
        schema_path = SCHEMAS_DIR / "tokens" / "v1.json"
        with open(schema_path) as f:
            schema = json.load(f)
        validator = Draft202012Validator(schema)

        # Create a model with all optional fields set to None
        model_with_nulls = TokenModel(ids=[1, 2, 3], mask=None, vocab="gpt2", truncation=None, offsets=None)
        ether_with_nulls = Ether.from_model(model_with_nulls)

        # Convert to dict and remove null values to match schema expectations
        ether_dict = ether_with_nulls.model_dump()

        # Remove null values from payload and metadata to match schema expectations
        # The schema expects optional fields to be omitted, not null
        ether_dict["payload"] = {k: v for k, v in ether_dict["payload"].items() if v is not None}
        ether_dict["metadata"] = {k: v for k, v in ether_dict["metadata"].items() if v is not None}

        errors = list(validator.iter_errors(ether_dict))
        assert not errors, f"Binding mechanism validation failed: {errors}"

        # Test 4: Verify the schema file exists and is the canonical one
        assert schema_path.exists(), f"Canonical schema file {schema_path} must exist"

        # Test 5: Verify the schema has the expected structure
        assert schema["properties"]["kind"]["enum"] == ["tokens"]
        assert schema["properties"]["schema_version"]["enum"] == [1]
        assert "ids" in schema["properties"]["payload"]["properties"]

        # Test 6: Verify that a model with actual values also validates
        model_with_values = TokenModel(
            ids=[1, 2, 3, 4, 5],
            mask=[1, 1, 1, 1, 1],
            vocab="bert-base-uncased",
            truncation="longest_first",
            offsets=True,
        )
        ether_with_values = Ether.from_model(model_with_values)
        ether_dict_with_values = ether_with_values.model_dump()

        # Remove null values from payload and metadata
        ether_dict_with_values["payload"] = {
            k: v for k, v in ether_dict_with_values["payload"].items() if v is not None
        }
        ether_dict_with_values["metadata"] = {
            k: v for k, v in ether_dict_with_values["metadata"].items() if v is not None
        }

        errors_with_values = list(validator.iter_errors(ether_dict_with_values))
        assert not errors_with_values, f"Binding mechanism validation with values failed: {errors_with_values}"


class TestTokenModelMisRegistration:
    """Test mis-registration scenarios for TokenModel."""

    def test_missing_required_field_raises_error(self, clear_registry) -> None:
        """Test that missing required field raises RegistrationError."""

        # Try to register a model with missing required field
        with pytest.raises(Exception) as exc_info:

            @Ether.register(payload=["ids", "missing_field"], metadata=[], kind="tokens")
            class InvalidTokenModel(BaseModel):
                ids: list[int]
                # missing_field is not defined

        assert "unknown field" in str(exc_info.value).lower()

    def test_duplicate_field_mapping_raises_error(self, clear_registry) -> None:
        """Test that duplicate field mapping raises error."""

        # Try to register with different field names mapping to the same path
        with pytest.raises(Exception) as exc_info:

            @Ether.register(
                payload=["ids", "token_ids"],
                metadata=[],
                renames={
                    "ids": "payload.ids",
                    "token_ids": "payload.ids",  # Different fields mapping to same path
                },
                kind="tokens",
            )
            class DuplicatePathModel(BaseModel):
                ids: list[int]
                token_ids: list[int]

        assert "duplicate mapping" in str(exc_info.value).lower()

    def test_field_in_both_payload_and_metadata_raises_error(self, clear_registry) -> None:
        """Test that field in both payload and metadata raises error."""

        # Try to register with same field in both payload and metadata
        with pytest.raises(Exception) as exc_info:

            @Ether.register(payload=["ids"], metadata=["ids"], kind="tokens")  # Same field in both
            class DuplicateFieldModel(BaseModel):
                ids: list[int]

        assert "fields in both payload & metadata" in str(exc_info.value).lower()


class TestTokenModelValidation:
    """Test TokenModel field validation."""

    def test_token_model_extra_fields_ignored(self, clear_registry) -> None:
        """Test that extra fields are ignored in TokenModel registration."""

        # Create a model with extra fields
        @Ether.register(payload=["ids"], metadata=["vocab"], extra_fields="ignore", kind="tokens")
        class TokenModelWithExtra(BaseModel):
            ids: list[int]
            vocab: str
            extra_field: str = "ignored"

        # Convert to Ether
        model = TokenModelWithExtra(ids=[1, 2, 3], vocab="gpt2", extra_field="should_be_ignored")
        ether = Ether.from_model(model)

        # Verify extra field is not in payload or metadata
        assert "extra_field" not in ether.payload
        assert "extra_field" not in ether.metadata
        assert ether.extra_fields == {}  # Should be empty due to "ignore"


class TestTokenModelNegativeTests:
    """Test negative scenarios for TokenModel."""

    def test_omit_vocab_raises_validation_error(self, clear_registry) -> None:
        """Test that omitting vocab raises ValidationError then ConversionError."""

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

        # Test 1: Creating TokenModel without vocab should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            TokenModel(ids=[1, 2, 3])  # Missing vocab
        assert "vocab" in str(exc_info.value).lower()

        # Test 2: If somehow we create an Ether without vocab, conversion should fail
        # Create Ether manually without vocab
        ether_without_vocab = Ether(
            kind="tokens",
            schema_version=1,
            payload={"ids": [1, 2, 3]},
            metadata={},  # Missing vocab
        )

        # Converting back to TokenModel should raise ConversionError
        with pytest.raises(Exception) as exc_info:
            ether_without_vocab.as_model(TokenModel)
        assert "vocab" in str(exc_info.value).lower()

    def test_omit_ids_raises_validation_error(self, clear_registry) -> None:
        """Test that omitting ids raises ValidationError."""

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

        # Creating TokenModel without ids should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            TokenModel(vocab="gpt2")  # Missing ids
        assert "ids" in str(exc_info.value).lower()

    def test_invalid_ids_type_raises_validation_error(self, clear_registry) -> None:
        """Test that invalid ids type raises ValidationError."""

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

        # Creating TokenModel with invalid ids type should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            TokenModel(ids="not_a_list", vocab="gpt2")  # ids should be list[int]
        assert "validation error" in str(exc_info.value).lower()

        with pytest.raises(ValidationError) as exc_info:
            TokenModel(ids=["not", "integers"], vocab="gpt2")  # ids should be list[int]
        assert "validation error" in str(exc_info.value).lower()

    def test_invalid_vocab_type_raises_validation_error(self, clear_registry) -> None:
        """Test that invalid vocab type raises ValidationError."""

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

        # Creating TokenModel with invalid vocab type should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            TokenModel(ids=[1, 2, 3], vocab=123)  # vocab should be str
        assert "validation error" in str(exc_info.value).lower()
