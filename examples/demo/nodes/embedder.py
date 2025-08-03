"""EmbedderNode for converting tokens to embeddings via fixed-length random vectors.

This module defines the EmbedderNode class that processes Ether envelopes
containing tokenized data and converts them to embedding data using a
deterministic random vector generation strategy.

The EmbedderNode accepts tokens.v1 envelopes and emits embedding.v1 envelopes
with fixed-dimension random vectors generated using numpy.random.RandomState(42).
"""

import numpy as np

from ether.core import Ether
from ether.kinds import EmbeddingModel
from ether.node import Node


class EmbedderNode(Node):
    """Node for converting tokens to embeddings via fixed-length random vectors.

    This node accepts tokens.v1 envelopes and emits embedding.v1 envelopes.
    The embedding process:
    1. Ignores token IDs (as specified for demo)
    2. Generates a dim=128 NumPy vector using np.random.RandomState(42) for determinism
    3. Builds EmbeddingModel with values=list(vec), dim=128, source="demo"
    4. Converts to Ether envelope and appends lineage information

    Attributes:
        name: The name of the node instance (defaults to "embedder")
        version: The version of the node (defaults to "0.1.0")
        accepts: Set of (kind, version) tuples that this node can accept
        emits: Set of (kind, version) tuples that this node can emit
    """

    # Class variables for kind/version capabilities
    accepts: set[tuple[str, int]] = {("tokens", 1)}
    emits: set[tuple[str, int]] = {("embedding", 1)}

    def __init__(self, name: str = "embedder", version: str = "0.1.0") -> None:
        """Initialize an EmbedderNode instance.

        Args:
            name: The name of the node instance (defaults to "embedder")
            version: The version of the node (defaults to "0.1.0")
        """
        super().__init__(name, version)

    def _process(self, eth: Ether) -> Ether:
        """Process an Ether envelope containing tokenized data.

        Converts tokens.v1 envelopes to embedding.v1 envelopes using
        deterministic random vector generation.

        Args:
            eth: The input Ether envelope containing tokenized data

        Returns:
            The processed Ether envelope containing embedding data
        """
        # For demo, ignore token IDs and generate a dim=128 NumPy vector
        # using np.random.RandomState(42) for determinism
        rng = np.random.RandomState(42)
        embedding_vector = rng.random(128).tolist()

        # Build EmbeddingModel with the generated embedding data
        embedding_model = EmbeddingModel(
            values=embedding_vector,
            dim=128,
            source="demo",
        )

        # Convert EmbeddingModel to Ether envelope
        result_eth = Ether.from_model(embedding_model)

        # Append lineage information
        self.append_lineage(result_eth)

        return result_eth
