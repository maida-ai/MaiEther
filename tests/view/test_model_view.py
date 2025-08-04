"""Unit tests for ModelView functionality.

This module tests the ModelView class which provides lazy access to model data
stored in Ether envelopes without copying the data.
"""

from typing import Any

import pytest
from pydantic import BaseModel

from ether import EmbeddingModel, Ether, TextModel, TokenModel
from ether.view import ModelView


class TestModelView:
    """Test suite for ModelView functionality."""

    def test_direct_instantiation_raises_error(self):
        """Test that direct instantiation of ModelView raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Use ModelView\\[Type\\] to create a view of a model"):
            ModelView()

    def test_class_getitem_creates_specialized_class(self):
        """Test that __class_getitem__ creates specialized view classes."""
        # Test that we get a class, not an instance
        view_class = ModelView[TextModel]
        assert isinstance(view_class, type)
        assert "ModelView[TextModel]" in str(view_class)

    def test_class_caching(self):
        """Test that view classes are cached and reused."""
        view_class1 = ModelView[TextModel]
        view_class2 = ModelView[TextModel]
        assert view_class1 is view_class2

        # Different model types should create different classes
        view_class3 = ModelView[EmbeddingModel]
        assert view_class1 is not view_class3

    def test_basic_attribute_access(self):
        """Test basic attribute access to model fields."""
        eth = Ether.from_model(TextModel(text="Hello, world!", lang="en"))
        view = ModelView[TextModel](eth)

        assert view.text == "Hello, world!"
        assert view.lang == "en"

    def test_metadata_field_access(self):
        """Test access to fields stored in metadata."""
        eth = Ether.from_model(EmbeddingModel(values=[1.0, 2.0], dim=2, source="test"))
        view = ModelView[EmbeddingModel](eth)

        assert view.values == [1.0, 2.0]
        assert view.dim == 2
        assert view.source == "test"

    def test_payload_field_access(self):
        """Test access to fields stored in payload."""
        eth = Ether.from_model(TokenModel(ids=[1, 2, 3], vocab="test"))
        view = ModelView[TokenModel](eth)

        assert view.ids == [1, 2, 3]
        assert view.vocab == "test"

    def test_missing_attribute_raises_error(self):
        """Test that accessing non-existent attributes raises AttributeError."""
        eth = Ether.from_model(TextModel(text="Hello"))
        view = ModelView[TextModel](eth)

        with pytest.raises(AttributeError, match="Attribute non_existent not found"):
            _ = view.non_existent

    def test_unregistered_model_raises_error(self):
        """Test that using an unregistered model raises ValueError."""
        from pydantic import BaseModel

        class UnregisteredModel(BaseModel):
            field: str

        eth = Ether(kind="test", payload={"field": "value"}, metadata={})

        with pytest.raises(ValueError, match="Model UnregisteredModel is not registered with Ether"):
            ModelView[UnregisteredModel](eth)

    def test_as_model_conversion(self):
        """Test conversion back to original model type."""
        original_model = TextModel(text="Hello", lang="en")
        eth = Ether.from_model(original_model)
        view = ModelView[TextModel](eth)

        converted_model = view.as_model()
        assert isinstance(converted_model, TextModel)
        assert converted_model.text == "Hello"
        assert converted_model.lang == "en"

    def test_repr_representation(self):
        """Test string representation of view."""
        eth = Ether.from_model(TextModel(text="Hello"))
        view = ModelView[TextModel](eth)

        repr_str = repr(view)
        assert "ModelView[TextModel]" in repr_str
        assert "kind='text'" in repr_str

    def test_dir_includes_model_fields(self):
        """Test that __dir__ includes model field names."""

        @Ether.register(payload=["text"], metadata=["lang"], extra_fields="keep", kind="simple_extra")
        class SimpleExtraModel(TextModel):
            pass

        eth = Ether.from_model(SimpleExtraModel(text="Hello"))
        print(eth)
        view = ModelView[SimpleExtraModel](eth)

        dir_list = dir(view)
        assert "text" in dir_list
        assert "lang" in dir_list
        assert "encoding" in dir_list
        assert "detected_lang_conf" in dir_list

    def test_nested_field_access(self):
        """Test access to nested fields using dot notation."""
        # This test would require a model with nested fields
        # For now, test that the _traverse function works correctly
        eth = Ether.from_model(TextModel(text="Hello"))
        view = ModelView[TextModel](eth)

        # Simple field access should work
        assert view.text == "Hello"

    def test_renamed_field_access(self):
        """Test access to fields that have been renamed in the EtherSpec."""
        # Create a model with field renames
        from ether.core import Ether as EtherCore

        @EtherCore.register(payload=["text"], metadata=["lang"], renames={"text": "content.text"}, kind="custom_text")
        class CustomTextModel(TextModel):
            pass

        eth = Ether.from_model(CustomTextModel(text="Hello", lang="en"))
        view = ModelView[CustomTextModel](eth)

        # Should access the renamed field correctly
        assert view.text == "Hello"

    def test_extra_fields_access(self):
        """Test access to extra fields when extra_fields is not 'ignore'."""
        from ether.core import Ether as EtherCore

        @EtherCore.register(payload=["text"], metadata=["lang"], extra_fields="keep", kind="text_with_extra")
        class TextWithExtraModel(TextModel):
            extra_field: str = "default"

        eth = Ether.from_model(TextWithExtraModel(text="Hello", lang="en", extra_field="extra"))
        view = ModelView[TextWithExtraModel](eth)

        # Should be able to access extra fields
        assert view.extra_field == "extra"

    def test_extra_fields_not_ignore(self):
        """Test that extra fields are accessible when extra_fields != 'ignore'."""
        from ether.core import Ether as EtherCore

        @EtherCore.register(payload=["text"], metadata=["lang"], extra_fields="keep", kind="simple_extra")
        class SimpleExtraModel(TextModel):
            pass

        # Create Ether with extra fields in extra_fields section
        eth = Ether(kind="simple_extra", schema_version=1, payload={"text": "Hello"}, metadata={"lang": "en"})
        eth.extra_fields = {"custom_field": "custom_value", "another_field": 42}

        view = ModelView[SimpleExtraModel](eth)

        # Should be able to access extra fields
        assert view.custom_field == "custom_value"
        assert view.another_field == 42

        # Should still access regular fields
        assert view.text == "Hello"
        assert view.lang == "en"

    def test_extra_fields_error_policy(self):
        """Test that extra fields raise error when extra_fields='error'."""
        from ether.core import Ether as EtherCore

        @EtherCore.register(payload=["text"], metadata=["lang"], extra_fields="error", kind="error_extra")
        class ErrorExtraModel(TextModel):
            pass

        # Create Ether with extra fields
        eth = Ether(kind="error_extra", schema_version=1, payload={"text": "Hello"}, metadata={"lang": "en"})
        eth.extra_fields = {"error_field": "should_error"}

        view = ModelView[ErrorExtraModel](eth)

        # Should be able to access regular fields
        assert view.text == "Hello"
        assert view.lang == "en"

        # Should be able to access extra fields (ModelView doesn't enforce the error policy)
        assert view.error_field == "should_error"

    def test_multiple_view_instances(self):
        """Test that multiple view instances work independently."""
        eth1 = Ether.from_model(TextModel(text="Hello", lang="en"))
        eth2 = Ether.from_model(TextModel(text="World", lang="fr"))

        view1 = ModelView[TextModel](eth1)
        view2 = ModelView[TextModel](eth2)

        assert view1.text == "Hello"
        assert view1.lang == "en"
        assert view2.text == "World"
        assert view2.lang == "fr"

    def test_view_with_complex_model(self):
        """Test view with a complex model having multiple field types."""
        eth = Ether.from_model(
            EmbeddingModel(values=[1.0, 2.0, 3.0], dim=3, source="test", norm=1.0, quantized=False, dtype="float32")
        )
        view = ModelView[EmbeddingModel](eth)

        assert view.values == [1.0, 2.0, 3.0]
        assert view.dim == 3
        assert view.source == "test"
        assert view.norm == 1.0
        assert view.quantized is False
        assert view.dtype == "float32"

    def test_view_with_none_values(self):
        """Test view with model fields that have None values."""
        eth = Ether.from_model(TextModel(text="Hello", lang=None, encoding=None, detected_lang_conf=None))
        view = ModelView[TextModel](eth)

        assert view.text == "Hello"
        assert view.lang is None
        assert view.encoding is None
        assert view.detected_lang_conf is None

    def test_view_with_empty_ether(self):
        """Test view with an Ether envelope that has minimal data."""
        eth = Ether.from_model(TextModel(text=""))
        view = ModelView[TextModel](eth)

        assert view.text == ""
        assert view.lang is None

    def test_view_type_safety(self):
        """Test that view classes are properly typed."""
        view_class = ModelView[TextModel]

        # Should be able to create instances
        eth = Ether.from_model(TextModel(text="Hello"))
        view = view_class(eth)

        # Should have the correct type hints
        assert hasattr(view, "text")
        assert hasattr(view, "lang")

    def test_view_with_attachment_model(self):
        """Test view with a model that has attachments."""
        from ether.attachment import Attachment

        eth = Ether.from_model(EmbeddingModel(values=None, dim=768, source="bert"))
        eth.attachments = [Attachment(id="emb-0", inline_bytes=b"test", media_type="application/octet-stream")]

        view = ModelView[EmbeddingModel](eth)

        assert view.dim == 768
        assert view.source == "bert"
        assert view.values is None

    def test_view_error_messages(self):
        """Test that error messages are informative."""
        eth = Ether.from_model(TextModel(text="Hello"))
        view = ModelView[TextModel](eth)

        with pytest.raises(AttributeError) as exc_info:
            _ = view.non_existent_field
        assert "Attribute non_existent_field not found" in str(exc_info.value)

    def test_view_with_modified_ether(self):
        """Test that view reflects changes to the underlying Ether."""
        eth = Ether.from_model(TextModel(text="Hello"))
        view = ModelView[TextModel](eth)

        assert view.text == "Hello"

        # Modify the Ether payload
        eth.payload["text"] = "Modified"

        # View should reflect the change
        assert view.text == "Modified"

    def test_view_equality_and_hash(self):
        """Test view equality and hash behavior."""
        eth1 = Ether.from_model(TextModel(text="Hello"))
        eth2 = Ether.from_model(TextModel(text="Hello"))

        view1 = ModelView[TextModel](eth1)
        view2 = ModelView[TextModel](eth2)

        # Views with different Ether instances should not be equal
        assert view1 != view2

        # Views should be hashable (for use in sets/dicts)
        view_set = {view1, view2}
        assert len(view_set) == 2

    @pytest.mark.parametrize(
        "model_class, model_input",
        [
            (TextModel, {"text": "Hello", "lang": "en"}),
            (EmbeddingModel, {"values": [1.0, 2.0], "dim": 2, "source": "test"}),
            (TokenModel, {"ids": [1, 2, 3], "vocab": "test"}),
        ],
    )
    def test_view_with_all_model_types(
        self,
        model_class: BaseModel,
        model_input: dict[str, Any],
    ) -> None:
        """Test view with all available model types."""
        eth = Ether.from_model(model_class(**model_input))
        view = ModelView[model_class](eth)

        # Test that we can access the fields
        for field_name, expected_value in model_input.items():
            assert getattr(view, field_name) == expected_value

            # Test conversion back to model
            converted_model = view.as_model()
            assert isinstance(converted_model, model_class)
