"""TokenizerNode for converting text to tokens via naïve whitespace split.

This module defines the TokenizerNode class that processes Ether envelopes
containing text data and converts them to tokenized data using a simple
whitespace-based tokenization strategy.

The TokenizerNode accepts text.v1 envelopes and emits tokens.v1 envelopes
with token IDs generated via hash-based mapping.
"""

from ..core import Ether
from ..kinds import TextModel, TokenModel
from ..node import Node


class TokenizerNode(Node):
    """Node for converting text to tokens via naïve whitespace split.

    This node accepts text.v1 envelopes and emits tokens.v1 envelopes.
    The tokenization process:
    1. Lower-cases the input text
    2. Splits on whitespace using str.split()
    3. Maps each token to an int ID via abs(hash(tok)) % 50_000
    4. Builds a TokenModel and converts to Ether envelope
    5. Appends lineage information

    Attributes:
        name: The name of the node instance (defaults to "tokenizer")
        version: The version of the node (defaults to "0.1.0")
        accepts: Set of (kind, version) tuples that this node can accept
        emits: Set of (kind, version) tuples that this node can emit
    """

    # Class variables for kind/version capabilities
    accepts: set[tuple[str, int]] = {("text", 1)}
    emits: set[tuple[str, int]] = {("tokens", 1)}

    def __init__(self, name: str = "tokenizer", version: str = "0.1.0") -> None:
        """Initialize a TokenizerNode instance.

        Args:
            name: The name of the node instance (defaults to "tokenizer")
            version: The version of the node (defaults to "0.1.0")
        """
        super().__init__(name, version)

    def _process(self, eth: Ether) -> Ether:
        """Process an Ether envelope containing text data.

        Converts text.v1 envelopes to tokens.v1 envelopes using naïve
        whitespace-based tokenization.

        Args:
            eth: The input Ether envelope containing text data

        Returns:
            The processed Ether envelope containing tokenized data
        """
        # Convert Ether to TextModel to access the text content
        text_model = eth.as_model(TextModel)

        # Extract text and apply tokenization logic
        text = text_model.text.lower()  # Lower-case the string
        tokens = text.split()  # Split on whitespace

        # Map each token to an int ID via abs(hash(tok)) % 50_000
        token_ids = [abs(hash(tok)) % 50_000 for tok in tokens]

        # Build TokenModel with the tokenized data
        token_model = TokenModel(
            ids=token_ids,
            vocab="naive_whitespace",  # Simple vocabulary identifier
            truncation=None,  # No truncation applied
            offsets=None,  # No character offsets
        )

        # Convert TokenModel to Ether envelope
        result_eth = Ether.from_model(token_model)

        # Append lineage information
        self.append_lineage(result_eth)

        return result_eth
