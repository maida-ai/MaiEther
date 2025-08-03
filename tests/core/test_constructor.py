"""Tests for Ether constructor with model argument."""

import pytest
from pydantic import BaseModel

from ether import Registry
from ether.core import Ether


class TestEtherConstructorWithModel:
    """Test Ether constructor with model argument."""

    def test_ether_constructor_with_model(self) -> None:
        """Test Ether constructor with model argument."""
        # Clear registry for clean test
        Registry.clear_spec()

        @Ether.register(payload=["data"], metadata=[], kind="test")
        class TestModel(BaseModel):
            data: str

        model = TestModel(data="test")

        # Use constructor with model
        ether = Ether(model)

        # Verify conversion
        assert ether.kind == "test"
        assert ether.payload == {"data": "test"}
        assert ether._source_model == TestModel

    def test_ether_constructor_with_model_and_kwargs_raises_error(self) -> None:
        """Test that Ether constructor with model and kwargs raises error."""

        class TestModel(BaseModel):
            data: str

        model = TestModel(data="test")

        with pytest.raises(TypeError):
            Ether(model, kind="test")  # Should not accept both model and kwargs
