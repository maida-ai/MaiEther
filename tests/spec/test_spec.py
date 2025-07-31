"""Tests for the EtherSpec dataclass."""

import pytest

from ether.spec import EtherSpec


class TestEtherSpec:
    """Test cases for EtherSpec dataclass."""

    def test_valid_spec_construction(self) -> None:
        """Test that a valid EtherSpec can be constructed."""
        spec = EtherSpec(
            payload_fields=("field1", "field2"),
            metadata_fields=("meta1",),
            extra_fields="keep",
            renames={"field1": "payload.renamed"},
            kind="test_kind",
        )

        assert spec.payload_fields == ("field1", "field2")
        assert spec.metadata_fields == ("meta1",)
        assert spec.extra_fields == "keep"
        assert spec.renames == {"field1": "payload.renamed"}
        assert spec.kind == "test_kind"

    def test_spec_with_defaults(self) -> None:
        """Test EtherSpec construction with default values."""
        spec = EtherSpec(payload_fields=("field1",), metadata_fields=("meta1",))

        assert spec.payload_fields == ("field1",)
        assert spec.metadata_fields == ("meta1",)
        assert spec.extra_fields == "ignore"
        assert spec.renames == {}
        assert spec.kind is None

    def test_spec_with_none_renames(self) -> None:
        """Test EtherSpec construction with None renames."""
        spec = EtherSpec(payload_fields=("field1",), metadata_fields=("meta1",), renames=None)

        assert spec.renames == {}

    def test_duplicate_payload_metadata_fields_raises_error(self) -> None:
        """Test that RuntimeError is raised when fields appear in both payload and metadata."""
        with pytest.raises(RuntimeError, match="Fields in both payload & metadata: \\['shared_field'\\]"):
            EtherSpec(payload_fields=("field1", "shared_field"), metadata_fields=("meta1", "shared_field"))

    def test_duplicate_renames_raises_error(self) -> None:
        """Test that RuntimeError is raised when renames create duplicate mappings."""
        with pytest.raises(RuntimeError, match="Duplicate mapping for ether path 'payload.duplicate'"):
            EtherSpec(
                payload_fields=("field1", "field2"),
                metadata_fields=("meta1",),
                renames={"field1": "payload.duplicate", "field2": "payload.duplicate"},
            )

    def test_spec_is_mutable(self) -> None:
        """Test that EtherSpec instances are mutable."""
        spec = EtherSpec(payload_fields=("field1",), metadata_fields=("meta1",))

        # Should be able to modify fields
        spec.payload_fields = ("new_field",)
        assert spec.payload_fields == ("new_field",)

    def test_spec_with_empty_fields(self) -> None:
        """Test EtherSpec with empty field tuples."""
        spec = EtherSpec(payload_fields=(), metadata_fields=())

        assert spec.payload_fields == ()
        assert spec.metadata_fields == ()
        assert spec.renames == {}

    def test_spec_with_complex_renames(self) -> None:
        """Test EtherSpec with complex nested path renames."""
        spec = EtherSpec(
            payload_fields=("embedding", "dim"),
            metadata_fields=("source",),
            renames={"embedding": "vec.values", "dim": "vec.dim", "source": "model.source"},
        )

        assert spec.renames == {"embedding": "vec.values", "dim": "vec.dim", "source": "model.source"}

    def test_spec_equality(self) -> None:
        """Test that identical EtherSpec instances are equal."""
        spec1 = EtherSpec(payload_fields=("field1",), metadata_fields=("meta1",), renames={"field1": "payload.field1"})

        spec2 = EtherSpec(payload_fields=("field1",), metadata_fields=("meta1",), renames={"field1": "payload.field1"})

        assert spec1 == spec2

    def test_spec_inequality(self) -> None:
        """Test that different EtherSpec instances are not equal."""
        spec1 = EtherSpec(payload_fields=("field1",), metadata_fields=("meta1",))

        spec2 = EtherSpec(payload_fields=("field2",), metadata_fields=("meta1",))

        assert spec1 != spec2

    def test_spec_not_hashable(self) -> None:
        """Test that EtherSpec instances are not hashable due to dict field."""
        spec = EtherSpec(payload_fields=("field1",), metadata_fields=("meta1",))

        with pytest.raises(TypeError, match="unhashable type"):
            hash(spec)
