"""Core Ether model for data transport between nodes/layers.

This module defines the core Ether envelope model that safely transports data
between nodes/layers in composable ML/data systems. The Ether envelope provides
a standardized way to carry structured data with metadata and attachments.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, Field, PrivateAttr

from .attachment import Attachment
from .errors import RegistrationError
from .spec import EtherSpec

ModelT = TypeVar("ModelT", bound=BaseModel)

# Registries - defined at module level for easy access
_spec_registry: dict[type[BaseModel], EtherSpec] = {}
_adapter_registry: dict[tuple[type[BaseModel], type[BaseModel]], Callable[["Ether"], dict]] = {}


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

        Currently performs basic validation. Future implementations may include
        more sophisticated validation based on kind and schema_version.
        """
        # Basic validation - ensure kind is not empty
        if not self.kind:
            raise ValueError("kind cannot be empty")

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
