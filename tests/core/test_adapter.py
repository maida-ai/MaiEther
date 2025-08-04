"""Tests for Ether adapter registration functionality."""

from pydantic import BaseModel

from ether import Registry
from ether.core import Ether


class TestEtherAdapter:
    """Test Ether adapter registration functionality."""

    def test_adapter_registration(self, clear_registry) -> None:
        """Test registering an adapter function."""

        class SourceModel(BaseModel):
            field1: str

        class DestModel(BaseModel):
            field2: str

        @Ether.adapter(SourceModel, DestModel)
        def source_to_dest(eth: Ether) -> dict:
            return {"field2": eth.payload.get("field1", "default")}

        # Verify adapter was registered
        assert len(Registry.get_adapters()) == 1
        assert (SourceModel, DestModel) in Registry.get_adapters()
        assert Registry.get_adapter((SourceModel, DestModel)) == source_to_dest

    def test_adapter_decorator_returns_function(self, clear_registry) -> None:
        """Test that the adapter decorator returns the function."""

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
