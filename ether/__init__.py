# ruff: noqa: E402
# mypy: disable-error-code="var-annotated,no-untyped-def"
"""MaiEther - An envelope system for safe data transport between nodes/layers.

This package provides the core Ether envelope system for safely transporting
data between nodes/layers in composable ML/data systems. It defines the
intermediate representation (IR) and Python reference implementation for
the Ether envelope that can be carried over in-memory calls, multiprocess
queues, or binary transports.

The Ether envelope provides a standardized way to carry structured data
with metadata and attachments, serving as the intermediate representation
(IR) for data transport in composable ML/data systems.

Examples:
    >>> from ether import Ether, Attachment
    >>>
    >>> # Create a basic Ether envelope
    >>> ether = Ether(
    ...     kind="embedding",
    ...     schema_version=1,
    ...     payload={"dim": 768},
    ...     metadata={"source": "bert-base"}
    ... )
    >>>
    >>> # Create an attachment for large data
    >>> attachment = Attachment(
    ...     id="emb-0",
    ...     uri="shm://embeddings/12345",
    ...     media_type="application/x-raw-tensor",
    ...     codec="RAW_F32",
    ...     shape=[768],
    ...     dtype="float32"
    ... )
    >>> ether.attachments.append(attachment)
"""

# Version information
__version__ = "0.0.0"

# Public API
__all__ = [
    "Attachment",
    "ConversionError",
    "Ether",
    "EtherSpec",
    "Node",
    "RegistrationError",
    "TextModel",
    "TokenModel",
    "EmbeddingModel",
]


# Define the registry here to avoid circular imports
# We will also ignore the type errors in this file
# As this might require underlying type imports
# and thus circular dependencies


class Singleton(type):
    __INSTANCES = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__INSTANCES:
            cls.__INSTANCES[cls] = super().__call__(*args, **kwargs)
        return cls.__INSTANCES[cls]


class Registry(metaclass=Singleton):
    """Registry singleton.

    Prohibits:
        - Direct modification of the members

    Allows:
        - Registering new specs
    """

    _SPEC_REGISTRY: dict = {}
    _ADAPTER_REGISTRY: dict = {}

    # ----- Spec Registry -----

    @classmethod
    def register_spec(cls, model, spec):
        if model in cls._SPEC_REGISTRY:
            raise ValueError(f"Spec already registered for {model.__name__}")
        cls.set_spec(model, spec)

    @classmethod
    def get_spec(cls, model):
        return cls._SPEC_REGISTRY.get(model)

    @classmethod
    def set_spec(cls, model, spec):
        cls._SPEC_REGISTRY[model] = spec

    @classmethod
    def get_specs(cls) -> dict:
        return cls._SPEC_REGISTRY

    @classmethod
    def clear_spec(cls, force: bool = False, sure: bool = False):
        if not force and not sure:
            raise ValueError(
                "Are you sure you want to clear the spec registry? This is irreversible. "
                "Use `force=True, sure=True` to bypass this check."
            )
        cls._SPEC_REGISTRY.clear()

    # ----- Adapter Registry -----

    @classmethod
    def register_adapter(cls, key, adapter):
        if key in cls._ADAPTER_REGISTRY:
            raise ValueError(f"Adapter already registered for {key}")
        cls.set_adapter(key, adapter)

    @classmethod
    def get_adapter(cls, src, dst):
        return cls._ADAPTER_REGISTRY.get((src, dst))

    @classmethod
    def set_adapter(cls, key, adapter):
        cls._ADAPTER_REGISTRY[key] = adapter

    @classmethod
    def clear_adapter(cls, force: bool = False, sure: bool = False):
        if not force and not sure:
            raise ValueError(
                "Are you sure you want to clear the adapter registry? This is irreversible. "
                "Use `force=True, sure=True` to bypass this check."
            )
        cls._ADAPTER_REGISTRY.clear()

    @classmethod
    def get_adapters(cls) -> dict:
        return cls._ADAPTER_REGISTRY


# --- Imports come last ---

from .attachment import Attachment
from .core import Ether
from .errors import ConversionError, RegistrationError
from .kinds import EmbeddingModel, TextModel, TokenModel
from .node import Node
from .spec import EtherSpec
