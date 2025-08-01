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

from .attachment import Attachment
from .core import Ether
from .errors import ConversionError, RegistrationError
from .kinds import TextModel
from .spec import EtherSpec

# Version information
__version__ = "0.0.0"

# Public API
__all__ = [
    "Attachment",
    "ConversionError",
    "Ether",
    "EtherSpec",
    "RegistrationError",
    "TextModel",
]
