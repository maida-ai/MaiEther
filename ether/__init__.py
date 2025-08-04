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
    "ModelView",
]


# --- Imports come last ---
# We will try importing from the "private" subpackages
# directly to avoid circular imports.
#
# For devs: If importing internally, use direct imports

from ._attachment.attachment_model import Attachment
from ._node.node import Node
from ._registry.registry import Registry
from ._spec.ether_spec import EtherSpec
from ._view.model_view import ModelView
from .core import Ether
from .errors import ConversionError, RegistrationError
from .kinds import EmbeddingModel, TextModel, TokenModel
