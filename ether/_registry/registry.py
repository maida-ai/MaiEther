from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel  # Must be imported at runtime for type checks

if TYPE_CHECKING:
    # Keep this import here to avoid circular imports
    from ether.core import Ether


from ether._errors.errors import RegistrationError
from ether._spec.ether_spec import EtherSpec

from .singleton import Singleton


@runtime_checkable
class AdapterFunc(Protocol):
    def __call__(self, ether: "Ether") -> Mapping[str, Any]: ...


class Registry(metaclass=Singleton):
    """Registry singleton.

    Prohibits:
        - Direct modification of the members

    Allows:
        - Registering new specs
    """

    _SPEC_REGISTRY: dict[type[BaseModel], "EtherSpec[Any]"] = {}
    _ADAPTER_REGISTRY: dict[tuple[type[BaseModel], type[BaseModel]], AdapterFunc] = {}

    # ----- Spec Registry -----

    @classmethod
    def register_spec(cls, model: type[BaseModel], spec: "EtherSpec[BaseModel]") -> None:
        if model in cls._SPEC_REGISTRY:
            raise ValueError(f"Spec already registered for {model.__name__}")
        cls.set_spec(model, spec)

    @classmethod
    def get_spec(cls, model: type[BaseModel]) -> "EtherSpec[BaseModel] | None":
        return cls._SPEC_REGISTRY.get(model, None)

    @classmethod
    def set_spec(cls, model: type[BaseModel], spec: "EtherSpec[BaseModel]") -> None:
        cls._SPEC_REGISTRY[model] = spec

    @classmethod
    def get_specs(cls) -> dict[type[BaseModel], "EtherSpec[BaseModel]"]:
        return cls._SPEC_REGISTRY

    @classmethod
    def clear_spec(cls, force: bool = False, sure: bool = False) -> None:
        if not force and not sure:
            raise ValueError(
                "Are you sure you want to clear the spec registry? This is irreversible. "
                "Use `force=True, sure=True` to bypass this check."
            )
        cls._SPEC_REGISTRY.clear()

    # ----- Adapter Registry -----

    @classmethod
    def register_adapter(cls, key: tuple[type[BaseModel], type[BaseModel]], adapter: AdapterFunc) -> None:
        if key in cls._ADAPTER_REGISTRY:
            raise ValueError(f"Adapter already registered for {key}")
        cls.set_adapter(key, adapter)

    @classmethod
    def get_adapter(cls, key: tuple[type[BaseModel], type[BaseModel]]) -> AdapterFunc | None:
        return cls._ADAPTER_REGISTRY.get(key, None)

    @classmethod
    def set_adapter(cls, key: tuple[type[BaseModel], type[BaseModel]], adapter: AdapterFunc) -> None:
        cls._ADAPTER_REGISTRY[key] = adapter

    @classmethod
    def clear_adapter(cls, force: bool = False, sure: bool = False) -> None:
        if not force and not sure:
            raise ValueError(
                "Are you sure you want to clear the adapter registry? This is irreversible. "
                "Use `force=True, sure=True` to bypass this check."
            )
        cls._ADAPTER_REGISTRY.clear()

    @classmethod
    def get_adapters(cls) -> dict[tuple[type[BaseModel], type[BaseModel]], AdapterFunc]:
        return cls._ADAPTER_REGISTRY


def register_spec(
    *,
    payload: Sequence[str],
    metadata: Sequence[str],
    extra_fields: str = "ignore",
    renames: Mapping[str, str] | None = None,  # model_field -> ether dot path
    kind: str | None = None,
) -> Callable[[type[BaseModel]], type[BaseModel]]:
    """Register a Pydantic model with a spec.

    This decorator registers a Pydantic model, defining how its
    fields map to the Ether envelope's payload and metadata sections.

    Args:
        payload: Sequence of field names to map to Ether.payload
        metadata: Sequence of field names to map to Ether.metadata
        extra_fields: How to handle unmapped fields ("ignore" | "keep" | "error")
        renames: Mapping from model field names to Ether dot paths
        kind: Optional Ether kind identifier

    Returns:
        Decorator function that registers the model

    Raises:
        RegistrationError: If registration fails due to invalid configuration
            or unknown fields

    Examples:
        >>> @register_spec(
        ...     payload=["embedding"],
        ...     metadata=["source", "dim"],
        ...     extra_fields="keep",
        ...     renames={"embedding": "vec.values"},
        ...     kind="embedding",
        ... )
        ... class FooModel(BaseModel):
        ...     embedding: list[float]
        ...     source: str
        ...     dim: int
        ...     note: str = "extra"
    """

    def _decorator(model_cls: type[BaseModel]) -> type[BaseModel]:
        # Check if already registered
        if Registry.get_spec(model_cls) is not None:
            raise RegistrationError(f"{model_cls.__name__} already registered")

        # Validate that all specified fields exist in the model
        known = set(getattr(model_cls, "model_fields", {}).keys())
        for field in list(payload) + list(metadata):
            if known and field not in known:
                raise RegistrationError(f"{model_cls.__name__}: unknown field '{field}'")

        # Create EtherSpec and register it
        spec: EtherSpec[Any] = EtherSpec(
            tuple(payload),
            tuple(metadata),
            extra_fields,
            dict(renames or {}),
            kind,
        )
        Registry.register_spec(model_cls, spec)
        return model_cls

    return _decorator


def register_adapter(*, src: type[BaseModel], dst: type[BaseModel]) -> Callable[[AdapterFunc], AdapterFunc]:
    """Register an adapter function for converting between models.

    This decorator registers an adapter function that can convert from one
    model type to another via an Ether envelope.

    Args:
        src: Source model type
        dst: Destination model type

    Returns:
        Decorator function that registers the adapter

    Examples:
        >>> @register_adapter(FooModel, BarModel)
        ... def foo_to_bar(eth: Ether) -> dict:
        ...     vals = eth.payload["vec"]["values"]
        ...     return {"source": eth.metadata.get("source", "unknown"), "bar_field": len(vals)}
    """

    if Registry.get_adapter((src, dst)) is not None:
        raise RegistrationError(f"Adapter already registered for {src.__name__} -> {dst.__name__}")

    def _decorator(fn: AdapterFunc) -> AdapterFunc:
        from ether.utils import has_valid_arg_types

        if not (is_passing := has_valid_arg_types(fn, ["Ether"])):
            print(is_passing)
            raise RegistrationError(is_passing)
        Registry.register_adapter((src, dst), fn)
        return fn

    return _decorator
