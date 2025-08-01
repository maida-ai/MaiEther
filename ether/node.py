"""Node base class for Ether processing components.

This module defines the Node base class that provides a skeletal abstraction
for components that process Ether envelopes. Future components will extend
this base class to implement specific processing logic.

The Node class defines the interface for components that can accept and emit
Ether envelopes with specific kind and version requirements.
"""

from .core import Ether


class Node:
    """Base class for Ether processing components.

    This class provides a skeletal abstraction that future components will extend.
    It defines the basic interface for components that can process Ether envelopes.

    Attributes:
        name: The name of the node instance
        version: The version of the node (defaults to "0.1.0")
        accepts: Set of (kind, version) tuples that this node can accept
        emits: Set of (kind, version) tuples that this node can emit
    """

    # Class variables for kind/version capabilities
    accepts: set[tuple[str, int]] = set()
    emits: set[tuple[str, int]] = set()

    def __init__(self, name: str, version: str = "0.1.0") -> None:
        """Initialize a Node instance.

        Args:
            name: The name of the node instance
            version: The version of the node (defaults to "0.1.0")
        """
        self.name = name
        self.version = version

    def process(self, eth: Ether) -> Ether:
        """Process an Ether envelope.

        This method should be overridden by subclasses to implement
        specific processing logic.

        Args:
            eth: The input Ether envelope to process

        Returns:
            The processed Ether envelope

        Raises:
            NotImplementedError: This method must be overridden by subclasses
        """
        raise NotImplementedError(f"Node {self.__class__.__name__} must override process() method")
