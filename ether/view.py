"""ModelView for lazy access to Ether envelope data without copying.

This module provides the ModelView class that enables lazy access to model data
stored in Ether envelopes without creating a copy of the data. ModelView uses
the Ether.as_model functionality internally but provides attribute-based access
to model fields.

The ModelView class is designed to be used with generic type parameters to
provide type-safe access to specific model types. It caches view classes to
avoid repeated class creation.

Examples:
    >>> from ether import Ether, TextModel
    >>> from ether.view import ModelView
    >>>
    >>> # Create an Ether envelope from a model
    >>> eth = Ether.from_model(TextModel(text="Hello, world!", lang="en"))
    >>>
    >>> # Create a view for lazy access
    >>> text_view = ModelView[TextModel](eth)
    >>> text_view.text  # Returns "Hello, world!"
    >>> text_view.lang  # Returns "en"
    >>>
    >>> # Convert back to model when needed
    >>> model = text_view.as_model()
    >>> isinstance(model, TextModel)  # True
"""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from ether import Ether, Registry

_view_cache: dict[type[BaseModel], type["ModelView[Any]"]] = {}
ModelT = TypeVar("ModelT", bound=BaseModel)


class ModelView(Generic[ModelT]):
    """Lazy view for accessing model data from Ether envelopes without copying.

    ModelView provides attribute-based access to model fields stored in Ether
    envelopes. It uses the Ether.as_model functionality internally but provides
    a more convenient interface for accessing individual fields without creating
    a full model instance.

    The class uses Python's `__class_getitem__` to create specialized view
    classes for specific model types. This enables type-safe access and proper
    IDE support while maintaining lazy evaluation.

    ModelView instances are created using the generic syntax:
        ModelView[ModelType](ether_instance)

    The view will provide attribute access to all fields defined in the model's
    EtherSpec, mapping them to the appropriate payload, metadata, or extra_fields
    sections of the Ether envelope.

    Attributes:
        eth: The Ether envelope being viewed
        spec: The EtherSpec for the target model type
        _target: The target model class (for debugging)

    Examples:
        >>> from ether import Ether, TextModel
        >>> from ether.view import ModelView
        >>>
        >>> # Create Ether from model
        >>> eth = Ether.from_model(TextModel(text="Hello", lang="en"))
        >>>
        >>> # Create view for lazy access
        >>> view = ModelView[TextModel](eth)
        >>> view.text  # "Hello"
        >>> view.lang  # "en"
        >>>
        >>> # Convert back to model when needed
        >>> model = view.as_model()
        >>> isinstance(model, TextModel)  # True
    """

    __slots__ = ("eth", "spec", "_model_type")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Prevent direct instantiation of ModelView base class.

        Raises:
            NotImplementedError: Always raised to enforce generic usage
        """
        raise NotImplementedError("Use ModelView[Type] to create a view of a model")

    @classmethod
    def __class_getitem__(cls, model_type: type[BaseModel]) -> Any:
        """Create a specialized ModelView class for the target model type.

        This method implements Python's generic class syntax to create view
        classes that are specific to a particular model type. The created
        class provides type-safe access to the model's fields.

        Args:
            model_type: The model class to create a view for

        Returns:
            A specialized ModelView class for the model_type

        Examples:
            >>> ModelView[TextModel]  # Returns a specialized view class
            >>> ModelView[EmbeddingModel]  # Returns a different specialized class
        """
        if model_type in _view_cache:
            return _view_cache[model_type]

        class _ModelView(ModelView):
            def __init__(self, eth: Ether) -> None:
                """Initialize the model view with an Ether envelope.

                Args:
                    eth: The Ether envelope to create a view for

                Raises:
                    ValueError: If the model_type is not registered with Ether
                """
                self._model_type = model_type
                self.eth = eth
                self.spec = Registry.get_spec(self._model_type)
                if self.spec is None:
                    raise ValueError(f"Model {model_type.__name__} is not registered with Ether")

            def __repr__(self) -> str:
                """Return a string representation of the view."""
                return f"ModelView[{self._model_type.__name__}]({self.eth})"

        view_name = f"ModelView[{model_type.__name__}]"

        _ModelView.__qualname__ = view_name
        _ModelView.__name__ = view_name
        _view_cache[model_type] = _ModelView
        return _ModelView

    def __getattr__(self, name: str) -> Any:
        """Provide attribute access to model fields.

        This method enables attribute-based access to model fields stored in
        the Ether envelope. It maps field names to the appropriate section
        (payload, metadata, or extra_fields) based on the model's EtherSpec.

        The method handles field renames and nested field access using dot
        notation as defined in the EtherSpec.

        Args:
            name: The name of the field to access

        Returns:
            The value of the requested field

        Raises:
            AttributeError: If the field is not found in any section

        Examples:
            >>> view = ModelView[TextModel](eth)
            >>> view.text  # Accesses payload.text
            >>> view.lang  # Accesses metadata.lang
        """

        def _traverse(node: dict, keys: list[str]) -> Any:
            for key in keys:
                node = node[key]
            return node

        key = self.spec.renames.get(name, name)
        keys = key.split(".")
        # See if it's a field in the target
        if name in self.spec.payload_fields:
            return _traverse(self.eth.payload, keys)
        if name in self.spec.metadata_fields:
            return _traverse(self.eth.metadata, keys)
        if self.spec.extra_fields != "ignore":
            return _traverse(self.eth.extra_fields, keys)
        raise AttributeError(f"Attribute {name} not found")

    def __dir__(self) -> list[str]:
        """Return list of available attributes including model fields."""
        base_attrs = set(super().__dir__())
        model_attrs = set(self.spec.payload_fields) | set(self.spec.metadata_fields)
        if self.spec.extra_fields == "keep":
            model_attrs.update(self.eth.extra_fields.keys())
        return sorted(base_attrs | model_attrs)

    def as_model(self) -> Any:
        """Convert the view back to the original model type.

        This method uses the Ether.as_model functionality to create a proper
        model instance from the Ether envelope data. This is useful when you
        need the full model object rather than just individual field access.

        Returns:
            An instance of the target model type with data from the Ether envelope

        Examples:
            >>> view = ModelView[TextModel](eth)
            >>> model = view.as_model()
            >>> isinstance(model, TextModel)  # True
            >>> model.text  # Access as regular model
        """
        return self.eth.as_model(self._model_type)
