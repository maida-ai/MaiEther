"""Predefined model kinds for Ether data transport.

This module defines common Pydantic models for different data kinds that can be
transported via Ether envelopes. Each model is registered with the Ether system
for automatic conversion between models and envelopes.

The models follow the schema specifications defined in the schemas/ directory
and provide type-safe interfaces for common data types like text, embeddings,
tokens, etc.

Where applicable, the models follow the binding mechanism matrix:

| Canonical JSON-Schema file  | Matching edge model       | Binding mechanism                      |
| --------------------------- | ------------------------- | -------------------------------------- |
| schemas/text/v1.json        | TextModel (Pydantic)      | @Ether.register(..., kind="text")      |
| schemas/tokens/v1.json      | TokenModel (Pydantic)     | @Ether.register(..., kind="tokens")    |
| schemas/embedding/v1.json   | EmbeddingModel (Pydantic) | @Ether.register(..., kind="embedding") |

"""

from typing import Any

from pydantic import BaseModel, Field

from ether._registry.registry import register_spec


@register_spec(
    payload=["text"],
    metadata=["lang", "encoding", "detected_lang_conf"],
    extra_fields="ignore",
    kind="text",
    renames={"text": "text"},
)
class TextModel(BaseModel):
    """Text data model for transport via Ether envelopes.

    Represents text content with optional language and encoding metadata.
    Follows the text.v1 schema specification (schemas/text/v1.json).

    This model is registered with Ether to enable conversion between TextModel
    instances and Ether envelopes. The registration maps:
    - text field -> Ether.payload.text
    - lang field -> Ether.metadata.lang
    - encoding field -> Ether.metadata.encoding
    - detected_lang_conf field -> Ether.metadata.detected_lang_conf

    The resulting Ether envelope will have:
    - kind="text"
    - schema_version=1
    - payload={"text": "..."}
    - metadata={"lang": "...", "encoding": "...", "detected_lang_conf": ...}

    The model ensures strict type compliance with the schema:
    - Required fields: text (string)
    - Optional fields: lang (string), encoding (string), detected_lang_conf (number [0.0, 1.0])
    - Schema validation: Produces envelopes that validate against text.v1.json

    Args:
        text: The text content to transport (required)
        lang: Optional language identifier (e.g., "en", "es", "fr")
        encoding: Optional text encoding (e.g., "utf-8", "ascii")
        detected_lang_conf: Optional confidence score for detected language (0.0-1.0)

    Examples:
        >>> # Basic text model
        >>> model = TextModel(text="Hello, world!")
        >>> ether = Ether.from_model(model)
        >>> ether.kind == "text"
        True
        >>> ether.schema_version == 1
        True
        >>> ether.payload["text"] == "Hello, world!"
        True
        >>>
        >>> # Text model with metadata
        >>> model = TextModel(
        ...     text="Bonjour le monde!",
        ...     lang="fr",
        ...     encoding="utf-8",
        ...     detected_lang_conf=0.95
        ... )
        >>> ether = Ether.from_model(model)
        >>> ether.metadata["lang"] == "fr"
        True
        >>> ether.metadata["detected_lang_conf"] == 0.95
        True
    """

    text: str = Field(description="The text content to transport")
    lang: str | None = Field(default=None, description="Language identifier (e.g., 'en', 'es', 'fr')")
    encoding: str | None = Field(default=None, description="Text encoding (e.g., 'utf-8', 'ascii')")
    detected_lang_conf: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score for detected language (0.0-1.0)"
    )


@register_spec(
    payload=["ids", "mask"],
    metadata=["vocab", "truncation", "offsets"],
    extra_fields="ignore",
    kind="tokens",
)
class TokenModel(BaseModel):
    """Token data model for transport via Ether envelopes.

    Represents tokenized data with token IDs, optional attention mask, and vocabulary metadata.
    Follows the tokens.v1 schema specification (schemas/tokens/v1.json).

    This model is registered with Ether to enable conversion between TokenModel
    instances and Ether envelopes. The registration maps:
    - ids field -> Ether.payload.ids
    - mask field -> Ether.payload.mask
    - vocab field -> Ether.metadata.vocab
    - truncation field -> Ether.metadata.truncation
    - offsets field -> Ether.metadata.offsets

    The resulting Ether envelope will have:
    - kind="tokens"
    - schema_version=1
    - payload={"ids": [...], "mask": [...]}
    - metadata={"vocab": "...", "truncation": "...", "offsets": ...}

    The model ensures strict type compliance with the schema:
    - Required fields: ids (list[int]), vocab (string)
    - Optional fields: mask (list[int]), truncation (string), offsets (boolean)
    - Schema validation: Produces envelopes that validate against tokens.v1.json

    Args:
        ids: List of token IDs (required)
        mask: Optional attention mask (same length as ids)
        vocab: Vocabulary/model identifier (required)
        truncation: Optional truncation strategy used
        offsets: Optional flag indicating whether character offsets are included

    Examples:
        >>> # Basic token model
        >>> model = TokenModel(ids=[1, 2, 3, 4, 5], vocab="gpt2")
        >>> ether = Ether.from_model(model)
        >>> ether.kind == "tokens"
        True
        >>> ether.schema_version == 1
        True
        >>> ether.payload["ids"] == [1, 2, 3, 4, 5]
        True
        >>> ether.metadata["vocab"] == "gpt2"
        True
        >>>
        >>> # Token model with optional fields
        >>> model = TokenModel(
        ...     ids=[1, 2, 3],
        ...     mask=[1, 1, 0],
        ...     vocab="bert-base-uncased",
        ...     truncation="longest_first",
        ...     offsets=True
        ... )
        >>> ether = Ether.from_model(model)
        >>> ether.payload["mask"] == [1, 1, 0]
        True
        >>> ether.metadata["truncation"] == "longest_first"
        True
        >>> ether.metadata["offsets"] is True
        True
    """

    ids: list[int] = Field(description="List of token IDs")
    mask: list[int] | None = Field(default=None, description="Attention mask (same length as ids)")
    vocab: str = Field(description="Vocabulary/model identifier")
    truncation: str | None = Field(default=None, description="Truncation strategy used")
    offsets: bool | None = Field(default=None, description="Whether character offsets are included")


@register_spec(
    payload=["values", "dim"],
    metadata=["source", "norm", "quantized", "dtype", "codec"],
    extra_fields="ignore",
    kind="embedding",
)
class EmbeddingModel(BaseModel):
    """Embedding data model for transport via Ether envelopes.

    Represents embedding vectors with dimensionality and comprehensive metadata.
    Follows the embedding.v1 schema specification (schemas/embedding/v1.json).

    This model is registered with Ether to enable conversion between EmbeddingModel
    instances and Ether envelopes. The registration maps:
    - values field -> Ether.payload.values
    - dim field -> Ether.payload.dim
    - source field -> Ether.metadata.source
    - norm field -> Ether.metadata.norm
    - quantized field -> Ether.metadata.quantized
    - dtype field -> Ether.metadata.dtype
    - codec field -> Ether.metadata.codec

    The resulting Ether envelope will have:
    - kind="embedding"
    - schema_version=1
    - payload={"values": None, "dim": ...}
    - metadata={"source": "...", "norm": ..., "quantized": ..., "dtype": ..., "codec": ...}

    The model ensures strict type compliance with the schema:
    - Required fields: dim (int)
    - Optional fields: values (list[float] | None), source (string | None),
      norm (float | None), quantized (bool | None), dtype (string | None), codec (string | None)
    - Schema validation: Produces envelopes that validate against embedding.v1.json

    Args:
        values: Optional list of float values for the embedding vector
        dim: Dimensionality of the embedding vector (required)
        source: Optional source identifier for the embedding model
        norm: Optional L2 norm of the embedding vector
        quantized: Optional flag indicating if INT8/other codec is used
        dtype: Optional data type (if values omitted, dtype must be in attachments)
        codec: Optional codec identifier (RAW_F32|RAW_F16|INT8|DLPACK|ARROW_IPC, only if attachments provided)

    Examples:
        >>> # Basic embedding model with inline values
        >>> model = EmbeddingModel(values=[1.0, 2.0, 3.0], dim=3)
        >>> ether = Ether.from_model(model)
        >>> ether.kind == "embedding"
        True
        >>> ether.schema_version == 1
        True
        >>> ether.payload["values"] == [1.0, 2.0, 3.0]
        True
        >>> ether.payload["dim"] == 3
        True
        >>>
        >>> # Embedding model with None values (for attachment-based transport)
        >>> model = EmbeddingModel(
        ...     values=None,
        ...     dim=768,
        ...     source="bert-base-uncased",
        ...     norm=1.0,
        ...     quantized=False,
        ...     dtype="float32",
        ...     codec="RAW_F32"
        ... )
        >>> ether = Ether.from_model(model)
        >>> ether.payload["values"] is None
        True
        >>> ether.payload["dim"] == 768
        True
        >>> ether.metadata["source"] == "bert-base-uncased"
        True
        >>> ether.metadata["norm"] == 1.0
        True
        >>> ether.metadata["quantized"] is False
        True
        >>> ether.metadata["dtype"] == "float32"
        True
        >>> ether.metadata["codec"] == "RAW_F32"
        True
    """

    values: list[float] | None = Field(default=None, description="List of float values for the embedding vector")
    dim: int = Field(gt=0, description="Dimensionality of the embedding vector")
    source: str | None = Field(default=None, description="Source identifier for the embedding model")
    norm: float | None = Field(default=None, ge=0.0, description="L2 norm of the embedding vector")
    quantized: bool | None = Field(default=None, description="Flag indicating if INT8/other codec is used")
    dtype: str | None = Field(default=None, description="Data type (if values omitted, dtype must be in attachments)")
    codec: str | None = Field(
        default=None,
        description="Codec identifier (RAW_F32|RAW_F16|INT8|DLPACK|ARROW_IPC, only if attachments provided)",
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate embedding model after initialization.

        Ensures that if values is provided, its length matches dim.

        Raises:
            ValueError: If values length doesn't match dim when values is provided
        """
        if self.values is not None and len(self.values) != self.dim:
            raise ValueError(f"Values length ({len(self.values)}) must match dim ({self.dim})")
