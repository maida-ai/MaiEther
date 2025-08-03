"""Demo node implementations for Ether processing components.

This package contains demo node implementations that extend the base Node class
to provide specific processing functionality for Ether envelopes. These are
intended for demonstration and testing purposes only.
"""

from .embedder import EmbedderNode
from .tokenizer import TokenizerNode

__all__ = ["TokenizerNode", "EmbedderNode"]
