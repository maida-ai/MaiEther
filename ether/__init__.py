"""MaiEther - An envelope system for safe data transport between nodes/layers.

This package provides the core Ether envelope system for safely transporting
data between nodes/layers in composable ML/data systems. It defines the
intermediate representation (IR) and Python reference implementation for
the Ether envelope that can be carried over in-memory calls, multiprocess
queues, or binary transports.
"""

from .attachment import Attachment
from .core import Ether
from .errors import ConversionError, RegistrationError
from .spec import EtherSpec

__all__ = ["Attachment", "ConversionError", "Ether", "EtherSpec", "RegistrationError"]
