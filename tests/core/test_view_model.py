"""Tests for Ether.view_model() method."""

import pytest
from pydantic import BaseModel

from ether.core import Ether
from ether.view import ModelView


class TestEtherViewModel:
    """Test Ether.view_model() method."""

    def test_view_model_basic_access(self, clear_registry) -> None:
        """Test basic view_model functionality."""

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

        # Create view
        view = ether.view_model(TargetModel)

        # Verify view provides access to fields without copying
        assert view.embedding == [1.0, 2.0, 3.0]
        assert view.source == "bert"

        # Verify view type
        assert isinstance(view, ModelView)

    def test_view_model_with_renames(self, clear_registry) -> None:
        """Test view_model with field renames."""

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

        # Create view
        view = ether.view_model(TargetModel)

        # Verify view provides access to renamed fields
        assert view.embedding == [1.0, 2.0]
        assert view.source == "bert"

    def test_view_model_unregistered_model_raises_error(self, clear_registry) -> None:
        """Test that view_model raises error for unregistered model."""

        class UnregisteredModel(BaseModel):
            field: str

        ether = Ether(kind="test", payload={}, metadata={})

        with pytest.raises(ValueError, match="UnregisteredModel is not registered"):
            ether.view_model(UnregisteredModel)

    def test_view_model_as_model_conversion(self, clear_registry) -> None:
        """Test that view can be converted back to model."""

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

        # Create view
        view = ether.view_model(TargetModel)

        # Convert back to model
        model = view.as_model()

        # Verify conversion
        assert isinstance(model, TargetModel)
        assert model.embedding == [1.0, 2.0, 3.0]
        assert model.source == "bert"

    def test_view_model_extra_fields(self, clear_registry) -> None:
        """Test view_model with extra fields."""

        @Ether.register(
            payload=["data"],
            metadata=[],
            extra_fields="keep",
            kind="test",
        )
        class TargetModel(BaseModel):
            data: str
            extra_field: str

        # Create Ether with extra fields
        ether = Ether(
            kind="test",
            payload={"data": "test_data"},
            metadata={},
            extra_fields={"extra_field": "extra_value"},
        )

        # Create view
        view = ether.view_model(TargetModel)

        # Verify view provides access to extra fields
        assert view.data == "test_data"
        assert view.extra_field == "extra_value"

    def test_view_model_dir_includes_model_fields(self, clear_registry) -> None:
        """Test that view.__dir__ includes model fields."""

        @Ether.register(
            payload=["embedding"],
            metadata=["source"],
            kind="embedding",
        )
        class TargetModel(BaseModel):
            embedding: list[float]
            source: str

        ether = Ether(
            kind="embedding",
            payload={"embedding": [1.0, 2.0]},
            metadata={"source": "bert"},
        )

        view = ether.view_model(TargetModel)
        dir_attrs = dir(view)

        # Verify model fields are included in dir
        assert "embedding" in dir_attrs
        assert "source" in dir_attrs

    def test_view_model_repr(self, clear_registry) -> None:
        """Test view representation."""

        @Ether.register(
            payload=["data"],
            metadata=[],
            kind="test",
        )
        class TargetModel(BaseModel):
            data: str

        ether = Ether(
            kind="test",
            payload={"data": "test"},
            metadata={},
        )

        view = ether.view_model(TargetModel)
        repr_str = repr(view)

        # Verify representation includes model name and ether content
        assert "ModelView[TargetModel]" in repr_str
        assert "kind='test'" in repr_str
        assert "payload={'data': 'test'}" in repr_str

    def test_view_model_attribute_error_for_missing_field(self, clear_registry) -> None:
        """Test that view raises AttributeError for missing fields."""

        @Ether.register(
            payload=["data"],
            metadata=[],
            kind="test",
        )
        class TargetModel(BaseModel):
            data: str

        ether = Ether(
            kind="test",
            payload={"data": "test"},
            metadata={},
        )

        view = ether.view_model(TargetModel)

        # Verify accessing non-existent field raises AttributeError
        with pytest.raises(AttributeError, match="Attribute missing_field not found"):
            _ = view.missing_field
