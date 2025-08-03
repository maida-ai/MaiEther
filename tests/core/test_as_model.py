"""Tests for Ether.as_model() conversion method."""

import pytest
from pydantic import BaseModel, ValidationError

from ether import Registry
from ether.core import Ether
from ether.errors import ConversionError, RegistrationError


class TestEtherAsModel:
    """Test Ether.as_model() conversion method."""

    def test_as_model_basic_conversion(self) -> None:
        """Test basic conversion from Ether to model."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(
            payload=["embedding"],
            metadata=["source"],
            kind="embedding",
        )
        class TargetModel(BaseModel):
            embedding: list[float]
            source: str

        # Create Ether
        ether = Ether(
            kind="embedding",
            payload={"embedding": [1.0, 2.0, 3.0]},
            metadata={"source": "bert"},
        )

        # Convert to model
        model = ether.as_model(TargetModel)

        # Verify conversion
        assert model.embedding == [1.0, 2.0, 3.0]
        assert model.source == "bert"

    def test_as_model_with_renames(self) -> None:
        """Test as_model with field renames."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(
            payload=["embedding"],
            metadata=["source"],
            renames={
                "embedding": "vec.values",
                "source": "model.source",
            },
            kind="embedding",
        )
        class TargetModel(BaseModel):
            embedding: list[float]
            source: str

        # Create Ether with nested structure
        ether = Ether(
            kind="embedding",
            payload={"vec": {"values": [1.0, 2.0]}},
            metadata={"model": {"source": "bert"}},
        )

        # Convert to model
        model = ether.as_model(TargetModel)

        # Verify conversion
        assert model.embedding == [1.0, 2.0]
        assert model.source == "bert"

    def test_as_model_unregistered_model_raises_error(self) -> None:
        """Test that as_model raises error for unregistered model."""

        class UnregisteredModel(BaseModel):
            field: str

        ether = Ether(kind="test", payload={}, metadata={})

        with pytest.raises(RegistrationError, match="UnregisteredModel not registered"):
            ether.as_model(UnregisteredModel)

    def test_as_model_missing_required_fields_raises_error(self) -> None:
        """Test that as_model raises error for missing required fields."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(payload=["embedding"], metadata=["source"], kind="embedding")
        class TargetModel(BaseModel):
            embedding: list[float]
            source: str
            required_field: str  # This field is required but not provided

        ether = Ether(
            kind="embedding",
            payload={"embedding": [1.0, 2.0]},
            metadata={"source": "bert"},
        )

        with pytest.raises(ConversionError, match="Missing required fields"):
            ether.as_model(TargetModel)

    def test_as_model_require_kind_mismatch_raises_error(self) -> None:
        """Test that as_model with require_kind=True raises error for kind mismatch."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(payload=["data"], metadata=[], kind="type1")
        class TargetModel(BaseModel):
            data: str

        ether = Ether(
            kind="type2",  # Different kind
            payload={"data": "test"},
            metadata={},
        )

        with pytest.raises(ConversionError, match="Kind mismatch"):
            ether.as_model(TargetModel, require_kind=True)

    def test_as_model_require_kind_success(self) -> None:
        """Test that as_model with require_kind=True succeeds for matching kinds."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(payload=["data"], metadata=[], kind="test")
        class TargetModel(BaseModel):
            data: str

        ether = Ether(
            kind="test",  # Matching kind
            payload={"data": "test"},
            metadata={},
        )

        model = ether.as_model(TargetModel, require_kind=True)

        assert model.data == "test"

    def test_as_model_with_adapter(self) -> None:
        """Test as_model using adapter function."""
        # Clear registries for clean test
        Registry.clear_spec()
        Registry.clear_adapter()

        @Ether.register(payload=["embedding"], metadata=["source"], kind="embedding")
        class SourceModel(BaseModel):
            embedding: list[float]
            source: str

        @Ether.register(payload=["count"], metadata=[], kind="count")
        class TargetModel(BaseModel):
            count: int

        @Ether.adapter(SourceModel, TargetModel)
        def source_to_target(eth: Ether) -> dict:
            return {"count": len(eth.payload["embedding"])}

        # Create Ether from source model
        source_model = SourceModel(embedding=[1.0, 2.0, 3.0], source="bert")
        ether = Ether.from_model(source_model)

        # Convert to target model using adapter
        target_model = ether.as_model(TargetModel)

        # Verify conversion
        assert target_model.count == 3

    def test_as_model_from_extra_fields(self) -> None:
        """Test as_model picking fields from extra_fields."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(payload=["data"], metadata=[], kind="test")
        class TargetModel(BaseModel):
            data: str
            extra_field: str

        ether = Ether(
            kind="test",
            payload={"data": "test"},
            metadata={},
            extra_fields={"extra_field": "extra_value"},
        )

        model = ether.as_model(TargetModel)

        assert model.data == "test"
        assert model.extra_field == "extra_value"

    def test_as_model_round_trip(self) -> None:
        """Test round-trip conversion: Model -> Ether -> Model."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(
            payload=["embedding", "dim"],
            metadata=["source"],
            extra_fields="keep",
            renames={"embedding": "vec.values", "dim": "vec.dim"},
            kind="embedding",
        )
        class FooModel(BaseModel):
            embedding: list[float]
            source: str
            dim: int
            note: str = "extra"

        # Original model
        original = FooModel(embedding=[1.0, 2.0, 3.0], source="bert", dim=3, note="test")

        # Convert to Ether
        ether = Ether.from_model(original)

        # Convert back to model
        restored = ether.as_model(FooModel)

        # Verify round-trip
        assert restored.embedding == original.embedding
        assert restored.source == original.source
        assert restored.dim == original.dim
        assert restored.note == original.note

    def test_as_model_pick_from_extras_by_renamed_key(self) -> None:
        """Test as_model picking fields from extras using renamed keys."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(
            payload=["data"],
            metadata=[],
            renames={"data": "payload.data"},
            kind="test",
        )
        class TargetModel(BaseModel):
            data: str

        # Create Ether with renamed key in extras
        ether = Ether(
            kind="test",
            payload={},
            metadata={},
            extra_fields={"payload.data": "test_value"},
        )

        model = ether.as_model(TargetModel)

        assert model.data == "test_value"

    def test_as_model_pick_from_extras_by_model_field_name(self) -> None:
        """Test as_model picking fields from extras using model field name."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(
            payload=["data"],
            metadata=[],
            renames={"data": "payload.data"},
            kind="test",
        )
        class TargetModel(BaseModel):
            data: str

        # Create Ether with model field name in extras
        ether = Ether(
            kind="test",
            payload={},
            metadata={},
            extra_fields={"data": "test_value"},
        )

        model = ether.as_model(TargetModel)

        assert model.data == "test_value"

    def test_as_model_validation_error_without_missing_fields(self) -> None:
        """Test as_model when ValidationError occurs but no fields are missing."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(payload=["data"], metadata=[], kind="test")
        class TargetModel(BaseModel):
            data: str

        # Create Ether with invalid data type
        ether = Ether(
            kind="test",
            payload={"data": 123},  # Should be str, not int
            metadata={},
        )

        # This should raise ValidationError but not ConversionError
        with pytest.raises(ValidationError):
            ether.as_model(TargetModel)

    def test_as_model_pick_returns_false_when_field_not_found(self) -> None:
        """Test as_model when pick function returns False, None for missing fields."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(payload=["data"], metadata=[], kind="test")
        class TargetModel(BaseModel):
            data: str
            missing_field: str  # This field is not in payload, metadata, or extras

        # Create Ether without the required field
        ether = Ether(
            kind="test",
            payload={"data": "test"},
            metadata={},
            extra_fields={},
        )

        # This should raise ConversionError for missing required field
        with pytest.raises(ConversionError, match="Missing required fields"):
            ether.as_model(TargetModel)

    def test_as_model_pick_returns_false_for_optional_field(self) -> None:
        """Test as_model when pick function returns False, None for optional fields."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(payload=["data"], metadata=[], kind="test")
        class TargetModel(BaseModel):
            data: str
            optional_field: str | None = None  # Optional field not in payload/metadata/extras

        # Create Ether without the optional field
        ether = Ether(
            kind="test",
            payload={"data": "test"},
            metadata={},
            extra_fields={},
        )

        # This should succeed and use the default value
        model = ether.as_model(TargetModel)
        assert model.data == "test"
        assert model.optional_field is None

    def test_as_model_pick_returns_false_for_renamed_field_not_in_sources(self) -> None:
        """Test as_model when pick function returns False, None for renamed field not in any source."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(
            payload=["data"], metadata=[], renames={"data": "payload.data", "missing": "payload.missing"}, kind="test"
        )
        class TargetModel(BaseModel):
            data: str
            missing: str | None = None  # Optional field with rename, not in any source

        # Create Ether with data but without the missing field
        ether = Ether(
            kind="test",
            payload={"payload": {"data": "test"}},  # Only data, no missing
            metadata={},
            extra_fields={},
        )

        # This should succeed and use the default value for missing
        model = ether.as_model(TargetModel)
        assert model.data == "test"
        assert model.missing is None

    def test_as_model_pick_returns_false_for_metadata_field_not_found(self) -> None:
        """Test as_model when pick function returns False, None for metadata field not found."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(
            payload=["data"], metadata=["metadata_field"], renames={"metadata_field": "meta.field"}, kind="test"
        )
        class TargetModel(BaseModel):
            data: str
            metadata_field: str | None = None  # Optional metadata field with rename

        # Create Ether without the metadata field
        ether = Ether(
            kind="test",
            payload={"data": "test"},
            metadata={},  # No metadata
            extra_fields={},
        )

        # This should succeed and use the default value
        model = ether.as_model(TargetModel)
        assert model.data == "test"
        assert model.metadata_field is None
