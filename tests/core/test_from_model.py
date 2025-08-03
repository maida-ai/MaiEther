"""Tests for Ether.from_model() factory method."""

import pytest
from pydantic import BaseModel

from ether import Registry
from ether.core import Ether
from ether.errors import ConversionError, RegistrationError


class TestEtherFromModel:
    """Test Ether.from_model() factory method."""

    def test_from_model_basic_conversion(self) -> None:
        """Test basic conversion from model to Ether."""
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

        # Create model instance
        model = FooModel(embedding=[1.0, 2.0, 3.0], source="bert", dim=3, note="test")

        # Convert to Ether
        ether = Ether.from_model(model)

        # Verify conversion
        assert ether.kind == "embedding"
        assert ether.schema_version == 1
        assert ether.payload == {"vec": {"values": [1.0, 2.0, 3.0], "dim": 3}}
        # Metadata should contain both user-provided and auto-populated fields
        assert ether.metadata["source"] == "bert"
        assert "trace_id" in ether.metadata
        assert "created_at" in ether.metadata
        assert ether.extra_fields == {"note": "test"}
        assert ether._source_model == FooModel

    def test_from_model_with_custom_schema_version(self) -> None:
        """Test from_model with custom schema version."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(payload=["data"], metadata=[], kind="test")
        class TestModel(BaseModel):
            data: str

        model = TestModel(data="test")
        ether = Ether.from_model(model, schema_version=2)

        assert ether.schema_version == 2
        assert ether.kind == "test"

    def test_from_model_unregistered_model_raises_error(self) -> None:
        """Test that from_model raises error for unregistered model."""

        class UnregisteredModel(BaseModel):
            field: str

        model = UnregisteredModel(field="test")

        with pytest.raises(RegistrationError, match="UnregisteredModel not registered"):
            Ether.from_model(model)

    def test_from_model_extra_fields_error_policy(self) -> None:
        """Test from_model with extra_fields='error' policy."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(
            payload=["data"],
            metadata=[],
            extra_fields="error",
            kind="test",
        )
        class TestModel(BaseModel):
            data: str
            extra_field: str = "should_fail"

        model = TestModel(data="test", extra_field="fail")

        with pytest.raises(ConversionError, match="Extra fields not allowed"):
            Ether.from_model(model)

    def test_from_model_extra_fields_keep_policy(self) -> None:
        """Test from_model with extra_fields='keep' policy."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(
            payload=["data"],
            metadata=[],
            extra_fields="keep",
            kind="test",
        )
        class TestModel(BaseModel):
            data: str
            extra_field: str = "should_keep"

        model = TestModel(data="test", extra_field="keep_me")

        ether = Ether.from_model(model)

        assert ether.extra_fields == {"extra_field": "keep_me"}

    def test_from_model_extra_fields_ignore_policy(self) -> None:
        """Test from_model with extra_fields='ignore' policy."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(
            payload=["data"],
            metadata=[],
            extra_fields="ignore",
            kind="test",
        )
        class TestModel(BaseModel):
            data: str
            extra_field: str = "should_ignore"

        model = TestModel(data="test", extra_field="ignore_me")

        ether = Ether.from_model(model)

        assert ether.extra_fields == {}

    def test_from_model_with_nested_renames(self) -> None:
        """Test from_model with nested field renames."""
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
        class TestModel(BaseModel):
            embedding: list[float]
            source: str

        model = TestModel(embedding=[1.0, 2.0], source="bert")

        ether = Ether.from_model(model)

        assert ether.payload == {"vec": {"values": [1.0, 2.0]}}
        # Metadata should contain both user-provided and auto-populated fields
        assert ether.metadata["model"] == {"source": "bert"}
        assert "trace_id" in ether.metadata
        assert "created_at" in ether.metadata

    def test_from_model_without_kind(self) -> None:
        """Test from_model with model registered without kind."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(payload=["data"], metadata=[])
        class TestModel(BaseModel):
            data: str

        model = TestModel(data="test")

        ether = Ether.from_model(model)

        assert ether.kind == ""  # Empty string when no kind specified
