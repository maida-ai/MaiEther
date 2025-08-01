"""Predefined model kinds for Ether data transport.

This module defines common Pydantic models for different data kinds that can be
transported via Ether envelopes. Each model is registered with the Ether system
for automatic conversion between models and envelopes.

The models follow the schema specifications defined in the schemas/ directory
and provide type-safe interfaces for common data types like text, embeddings,
tokens, etc.
"""

from pydantic import BaseModel, Field

from .core import Ether


@Ether.register(
    payload=["text"],
    metadata=["lang", "encoding", "detected_lang_conf"],
    extra_fields="ignore",
    kind="text",
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

    **Binding Mechanism Compliance:**

    This model follows the binding mechanism matrix:

    | Canonical JSON-Schema file  | Matching edge model    | Binding mechanism                      |
    | --------------------------- | ---------------------- | -------------------------------------- |
    | schemas/text/v1.json        | TextModel (Pydantic)   | @Ether.register(..., kind="text")      |
    | schemas/tokens/v1.json      | TBD (Pydantic)         | @Ether.register(..., kind="tokens")    |
    | TBD                         | TBD (Pydantic)         | @Ether.register(..., kind="embedding") |

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
