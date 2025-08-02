"""Node base class for Ether processing components.

This module defines the Node base class that provides a skeletal abstraction
for components that process Ether envelopes. Future components will extend
this base class to implement specific processing logic.

The Node class defines the interface for components that can accept and emit
Ether envelopes with specific kind and version requirements.
"""

from .core import Ether
from .utils import rfc3339_now


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

    def validate_input(self, eth: Ether) -> None:
        """Validate that the input Ether envelope is acceptable for this node.

        Checks that the envelope's (kind, schema_version) is in the node's accepts set.

        Args:
            eth: The input Ether envelope to validate

        Raises:
            ValueError: If the envelope is not acceptable for this node
        """
        if not self.accepts:
            return  # No restrictions if accepts is empty

        envelope_spec = (eth.kind, eth.schema_version)
        if envelope_spec not in self.accepts:
            accepted_str = ", ".join(f"{kind}.v{version}" for kind, version in sorted(self.accepts))
            raise ValueError(
                f"{self.__class__.__name__} expects one of: {accepted_str}, " f"got {eth.kind}.v{eth.schema_version}"
            )

    def process(self, eth: Ether) -> Ether:
        """Process an Ether envelope.

        This method should NOT be overridden.
        Instead, override the _process method.
        The base implementation validates the input
        and calls the subclass's _process method.

        Args:
            eth: The input Ether envelope to process

        Returns:
            The processed Ether envelope

        Raises:
            ValueError: If the input envelope is not acceptable for this node
            NotImplementedError: If the subclass doesn't override _process
        """
        self.validate_input(eth)
        return self._process(eth)

    def _process(self, eth: Ether) -> Ether:
        """Process an Ether envelope (subclass implementation).

        This method should be overridden by subclasses to implement
        specific processing logic.

        Args:
            eth: The input Ether envelope to process

        Returns:
            The processed Ether envelope

        Raises:
            NotImplementedError: This method must be overridden by subclasses
        """
        raise NotImplementedError(f"Node {self.__class__.__name__} must override _process() method")

    def append_lineage(self, eth: Ether) -> None:
        """Append lineage information to the Ether envelope metadata.

        Adds a lineage entry to the Ether's metadata containing the node's
        name, version, and current timestamp. Creates the lineage list if
        it doesn't exist.

        Args:
            eth: The Ether envelope to append lineage information to
        """
        if "lineage" not in eth.metadata:
            eth.metadata["lineage"] = []

        lineage_entry = {"node": self.name, "version": self.version, "ts": rfc3339_now()}

        eth.metadata["lineage"].append(lineage_entry)
