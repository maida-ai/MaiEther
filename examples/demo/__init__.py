"""Demo nodes for Ether processing components.

This package contains demo node implementations that extend the base Node class
to provide specific processing functionality for Ether envelopes. These are
intended for demonstration and testing purposes only.
"""

from .nodes.embedder import EmbedderNode
from .nodes.tokenizer import TokenizerNode

__all__ = ["TokenizerNode", "EmbedderNode"]
