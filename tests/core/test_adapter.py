"""Tests for Ether adapter registration functionality."""

from pydantic import BaseModel

from ether.core import Ether, _adapter_registry


class TestEtherAdapter:
    """Test Ether adapter registration functionality."""

    def test_adapter_registration(self) -> None:
        """Test registering an adapter function."""
        # Clear registry for clean test
        _adapter_registry.clear()

        class SourceModel(BaseModel):
            field1: str

        class DestModel(BaseModel):
            field2: str

        @Ether.adapter(SourceModel, DestModel)
        def source_to_dest(eth: Ether) -> dict:
            return {"field2": eth.payload.get("field1", "default")}

        # Verify adapter was registered
        assert len(_adapter_registry) == 1
        assert (SourceModel, DestModel) in _adapter_registry
        assert _adapter_registry[(SourceModel, DestModel)] == source_to_dest

    def test_adapter_decorator_returns_function(self) -> None:
        """Test that the adapter decorator returns the function."""
        # Clear registry for clean test
        _adapter_registry.clear()

        class SourceModel(BaseModel):
            field1: str

        class DestModel(BaseModel):
            field2: str

        @Ether.adapter(SourceModel, DestModel)
        def test_adapter(eth: Ether) -> dict:
            return {"field2": "test"}

        # Verify the decorator returned the function
        assert test_adapter.__name__ == "test_adapter"
        assert callable(test_adapter)
