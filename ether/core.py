"""Core Ether model for data transport between nodes/layers.

This module defines the core Ether envelope model that safely transports data
between nodes/layers in composable ML/data systems. The Ether envelope provides
a standardized way to carry structured data with metadata and attachments.

The Ether class provides the main envelope functionality with methods for:
- Registration of Pydantic models for conversion
- Adapter functions for complex model conversions
- Conversion between models and Ether envelopes
- Summary generation for debugging and logging

Examples:
    >>> from ether import Ether, Attachment
    >>> from pydantic import BaseModel
    >>>
    >>> # Register a model for conversion
    >>> @Ether.register(payload=["embedding"], metadata=["source"], kind="embedding")
    ... class EmbeddingModel(BaseModel):
    ...     embedding: list[float]
    ...     source: str
    >>>
    >>> # Convert model to Ether
    >>> model = EmbeddingModel(embedding=[1.0, 2.0], source="bert")
    >>> ether = Ether.from_model(model)
    >>> ether.kind == "embedding"
    True
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, Field, PrivateAttr, ValidationError

from .attachment import Attachment
from .errors import ConversionError, RegistrationError
from .spec import EtherSpec

ModelT = TypeVar("ModelT", bound=BaseModel)

# Registries - defined at module level for easy access
_spec_registry: dict[type[BaseModel], EtherSpec] = {}
_adapter_registry: dict[tuple[type[BaseModel], type[BaseModel]], Callable[["Ether"], dict]] = {}


def _missing_required(model_cls: type[BaseModel], present: Sequence[str]) -> set[str]:
    """Find missing required fields in a model.

    Args:
        model_cls: The Pydantic model class to check
        present: Set of field names that are present

    Returns:
        Set of required field names that are missing
    """
    present_set = set(present)
    missing = set()
    for name, field in model_cls.model_fields.items():
        req = (
            field.is_required()
            if hasattr(field, "is_required")
            else (field.default is None and field.default_factory is None)
        )
        if req and name not in present_set:
            missing.add(name)
    return missing


class Ether(BaseModel):
    """Core Ether envelope for safe data transport between nodes/layers.

    The Ether envelope provides a standardized way to carry structured data
    with metadata and attachments. It serves as the intermediate representation
    (IR) for data transport in composable ML/data systems.

    Args:
        kind: Logical type identifier (e.g., "embedding", "tokens", "image")
        schema_version: Integer version of the schema (>= 1)
        payload: Structured content relevant to the kind
        metadata: Context, provenance, and parameters
        extra_fields: Carry-through for unclassified fields
        attachments: List of binary or external buffers

    Examples:
        >>> # Basic Ether with empty payload and metadata
        >>> ether = Ether(
        ...     kind="embedding",
        ...     schema_version=1,
        ...     payload={},
        ...     metadata={}
        ... )
        >>>
        >>> # Ether with attachments
        >>> ether = Ether(
        ...     kind="embedding",
        ...     schema_version=1,
        ...     payload={"dim": 768},
        ...     metadata={"source": "bert-base"},
        ...     attachments=[Attachment(id="emb-0", ...)]
        ... )
    """

    kind: str = Field(description="Logical type identifier (e.g., 'embedding', 'tokens', 'image')")
    schema_version: int = Field(default=1, ge=1, description="Integer version of the schema (>= 1)")
    payload: dict[str, Any] = Field(description="Structured content relevant to the kind")
    metadata: dict[str, Any] = Field(description="Context, provenance, and parameters")
    extra_fields: dict[str, Any] = Field(default_factory=dict, description="Carry-through for unclassified fields")
    attachments: list[Attachment] = Field(default_factory=list, description="List of binary or external buffers")

    _source_model: type[BaseModel] | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Validate Ether configuration after initialization.

        Performs validation including:
        - Basic configuration validation
        - Large inline data detection (> 64 KB) with suggestions for attachments
        """
        # Check for large inline data across all sections
        self._validate_inline_size_limit()

    def _validate_inline_size_limit(self) -> None:
        """Validate that inline data doesn't exceed size limits.

        Raises:
            ValueError: If large inline data is detected with suggestion to use attachments
        """
        INLINE_SIZE_LIMIT = 64 * 1024  # 64 KB

        def estimate_size(obj: Any) -> int:
            """Estimate the size of an object in bytes."""
            # TODO: Need a better way to estimate size of objects
            if isinstance(obj, list | tuple):
                return sum(estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(estimate_size(v) for v in obj.values())
            elif isinstance(obj, str | bytes):
                return len(obj)
            elif isinstance(obj, int | float | bool):
                return 8  # Approximate size for numeric types
            else:
                # TODO: Unknown type, warn and return 0
                # TODO: Decide if we want to raise an error or not
                return 0

        def check_large_data(obj: Any, path: str = "") -> list[str]:
            """Find paths to large data in the data."""
            large_paths = []

            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    large_paths.extend(check_large_data(value, new_path))
            else:
                size = estimate_size(obj)
                if size > INLINE_SIZE_LIMIT:
                    large_paths.append(f"{path} (estimated {size} bytes)")

            return large_paths

        # Check all sections for large data
        all_large_paths = []

        # Check payload
        payload_paths = check_large_data(self.payload, "payload")
        all_large_paths.extend(payload_paths)

        # Check metadata
        metadata_paths = check_large_data(self.metadata, "metadata")
        all_large_paths.extend(metadata_paths)

        # Check extra_fields
        extra_paths = check_large_data(self.extra_fields, "extra_fields")
        all_large_paths.extend(extra_paths)

        if all_large_paths:
            paths_str = ", ".join(all_large_paths)
            raise ValueError(
                f"Large inline data detected: {paths_str}. "
                f"Consider using attachments for data > {INLINE_SIZE_LIMIT // 1024} KB. "
                "Use Attachment.from_numpy() for NumPy arrays or create attachments manually."
            )

    @classmethod
    def register(
        cls,
        *,
        payload: Sequence[str],
        metadata: Sequence[str],
        extra_fields: str = "ignore",
        renames: Mapping[str, str] | None = None,  # model_field -> ether dot path
        kind: str | None = None,
    ) -> Callable[[type[BaseModel]], type[BaseModel]]:
        """Register a Pydantic model with Ether for conversion.

        This decorator registers a Pydantic model with Ether, defining how its
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
            >>> @Ether.register(
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
            # Validate that all specified fields exist in the model
            known = set(getattr(model_cls, "model_fields", {}).keys())
            for field in list(payload) + list(metadata):
                if known and field not in known:
                    raise RegistrationError(f"{model_cls.__name__}: unknown field '{field}'")

            # Create EtherSpec and register it
            spec = EtherSpec(
                tuple(payload),
                tuple(metadata),
                extra_fields,
                dict(renames or {}),
                kind,
            )
            _spec_registry[model_cls] = spec
            return model_cls

        return _decorator

    @classmethod
    def adapter(
        cls, src: type[BaseModel], dst: type[BaseModel]
    ) -> Callable[[Callable[["Ether"], dict]], Callable[["Ether"], dict]]:
        """Register an adapter function for converting between models.

        This decorator registers an adapter function that can convert from one
        model type to another via an Ether envelope.

        Args:
            src: Source model type
            dst: Destination model type

        Returns:
            Decorator function that registers the adapter

        Examples:
            >>> @Ether.adapter(FooModel, BarModel)
            ... def foo_to_bar(eth: Ether) -> dict:
            ...     vals = eth.payload["vec"]["values"]
            ...     return {"source": eth.metadata.get("source", "unknown"), "bar_field": len(vals)}
        """

        def _decorator(fn: Callable[["Ether"], dict]) -> Callable[["Ether"], dict]:
            _adapter_registry[(src, dst)] = fn
            return fn

        return _decorator

    @classmethod
    def from_model(cls, model_instance: BaseModel, *, schema_version: int = 1) -> "Ether":
        """Create an Ether envelope from a registered Pydantic model.

        Converts a registered Pydantic model instance to an Ether envelope
        according to the model's registration specification.

        Args:
            model_instance: The Pydantic model instance to convert
            schema_version: Optional schema version override (default: 1)

        Returns:
            Ether envelope created from the model

        Raises:
            RegistrationError: If the model type is not registered
            ConversionError: If extra fields are not allowed according to spec

        Examples:
            >>> @Ether.register(payload=["embedding"], metadata=["source"], kind="embedding")
            ... class FooModel(BaseModel):
            ...     embedding: list[float]
            ...     source: str
            >>>
            >>> model = FooModel(embedding=[1.0, 2.0], source="bert")
            >>> ether = Ether.from_model(model)
            >>> ether.kind == "embedding"
            True
        """
        spec = _spec_registry.get(type(model_instance))
        if not spec:
            raise RegistrationError(f"{type(model_instance).__name__} not registered")

        data = model_instance.model_dump()
        payload: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        extras: dict[str, Any] = {}

        def set_by_path(root: dict[str, Any], path: str, value: Any) -> None:
            """Set a value in a nested dictionary using dot notation."""
            parts = path.split(".")
            cur = root
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = value

        listed = set(spec.payload_fields) | set(spec.metadata_fields)
        renames = spec.renames or {}
        for f in spec.payload_fields:
            set_by_path(payload, renames.get(f, f), data.get(f))
        for f in spec.metadata_fields:
            set_by_path(metadata, renames.get(f, f), data.get(f))

        if spec.extra_fields == "error":
            unlisted = [f for f in data if f not in listed]
            if unlisted:
                raise ConversionError(f"Extra fields not allowed: {sorted(unlisted)}")
        elif spec.extra_fields == "keep":
            for f, v in data.items():
                if f not in listed:
                    extras[f] = v

        eth = cls(
            kind=spec.kind or "",
            schema_version=schema_version,
            payload=payload,
            metadata=metadata,
            extra_fields=extras,
            attachments=[],
        )
        eth._source_model = type(model_instance)
        return eth

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize an Ether envelope.

        Supports both direct initialization and conversion from a Pydantic model.
        If a single BaseModel argument is provided, converts it using from_model().

        Args:
            *args: Positional arguments for direct initialization
            **kwargs: Keyword arguments for direct initialization

        Examples:
            >>> # Direct initialization
            >>> ether = Ether(kind="embedding", payload={}, metadata={})
            >>>
            >>> # Conversion from model
            >>> model = SomeModel(field="value")
            >>> ether = Ether(model)  # Equivalent to Ether.from_model(model)
        """
        if len(args) == 1 and isinstance(args[0], BaseModel) and not kwargs:
            eth = self.__class__.from_model(args[0])
            super().__init__(**eth.model_dump())
            self._source_model = eth._source_model
        else:
            super().__init__(*args, **kwargs)

    def as_model(self, target_model: type[ModelT], *, require_kind: bool = False) -> ModelT:
        """Convert the Ether envelope to a registered Pydantic model.

        Converts the Ether envelope to a target Pydantic model according to
        the model's registration specification. Supports adapter functions for
        complex conversions.

        Args:
            target_model: The target Pydantic model class
            require_kind: Whether to require kind matching (default: False)

        Returns:
            Instance of the target model

        Raises:
            RegistrationError: If the target model is not registered
            ConversionError: If kind mismatch or missing required fields

        Examples:
            >>> @Ether.register(payload=["embedding"], metadata=["source"])
            ... class TargetModel(BaseModel):
            ...     embedding: list[float]
            ...     source: str
            >>>
            >>> ether = Ether(kind="embedding", payload={"embedding": [1.0]}, metadata={"source": "bert"})
            >>> model = ether.as_model(TargetModel)
            >>> model.embedding == [1.0]
            True
        """
        spec = _spec_registry.get(target_model)
        if not spec:
            raise RegistrationError(f"{target_model.__name__} not registered")

        if require_kind and spec.kind and self.kind and spec.kind != self.kind:
            raise ConversionError(f"Kind mismatch: Ether={self.kind!r}, Target expects {spec.kind!r}")

        # adapter path
        if self._source_model is not None:
            adapter = _adapter_registry.get((self._source_model, target_model))
            if adapter:
                return target_model.model_validate(adapter(self))  # type: ignore[no-any-return]

        # default field picking
        def flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
            """Flatten nested dictionary into dot-separated keys."""
            out = {}
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, Mapping):
                    out.update(flatten(v, key))
                else:
                    out[key] = v
            return out

        payload_flat = flatten(self.payload)
        metadata_flat = flatten(self.metadata)
        extras = self.extra_fields

        def pick(model_field: str, from_payload: bool) -> tuple[bool, Any]:
            """Pick a field value from flattened payload/metadata or extras."""
            ether_key = spec.renames.get(model_field, model_field) if spec.renames else model_field
            src = payload_flat if from_payload else metadata_flat
            if ether_key in src:
                return True, src[ether_key]
            if model_field in extras:
                return True, extras[model_field]
            if ether_key in extras:
                return True, extras[ether_key]
            # Field not found in any source
            return False, None

        data: dict[str, Any] = {}
        for f in spec.payload_fields:
            ok, val = pick(f, True)
            if ok:
                data[f] = val
        for f in spec.metadata_fields:
            ok, val = pick(f, False)
            if ok:
                data[f] = val

        # Also check for fields in extras that are not in payload/metadata
        for f in target_model.model_fields:
            if f not in data and f in extras:
                data[f] = extras[f]

        try:
            return target_model.model_validate(data)  # type: ignore[no-any-return]
        except ValidationError as ve:
            missing = _missing_required(target_model, list(data.keys()))
            if missing:
                raise ConversionError(
                    f"Missing required fields: {sorted(missing)}; provided={sorted(data.keys())}"
                ) from ve
            raise ve

    def summary(self) -> dict[str, Any]:
        """Generate a summary of the Ether envelope contents.

        Returns:
            Dictionary containing summary information about the Ether envelope
            including kind, schema_version, payload keys, metadata keys,
            extra field keys, attachment IDs, and source model name.

        Examples:
            >>> ether = Ether(kind="embedding", payload={"dim": 768}, metadata={"source": "bert"})
            >>> ether.summary()
            {
                'kind': 'embedding',
                'schema_version': 1,
                'payload_keys': ['dim'],
                'metadata_keys': ['source'],
                'extra_keys': [],
                'attachments': [],
                'source_model': None
            }
        """

        def flatten_keys(d: dict[str, Any]) -> list[str]:
            """Flatten nested dictionary keys into dot-separated paths."""
            out = []

            def rec(prefix: str, obj: dict[str, Any]) -> None:
                for k, v in obj.items():
                    key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        rec(key, v)
                    else:
                        out.append(key)

            rec("", d)
            return sorted(out)

        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "payload_keys": flatten_keys(self.payload),
            "metadata_keys": flatten_keys(self.metadata),
            "extra_keys": sorted(self.extra_fields.keys()),
            "attachments": [att.id for att in self.attachments],
            "source_model": getattr(self._source_model, "__name__", None),
        }
