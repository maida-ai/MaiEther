"""EtherSpec helper dataclass for payload/metadata field mapping and validation.

This module provides the EtherSpec dataclass that defines how model fields map
to Ether envelope payload and metadata sections, including field renames and
validation rules.

The EtherSpec class provides functionality for:
- Field mapping between models and Ether envelopes
- Field renaming with dot notation support
- Extra field handling policies
- Validation of field mappings

Examples:
    >>> from ether import EtherSpec
    >>> from pydantic import BaseModel
    >>>
    >>> # Create a spec for field mapping
    >>> spec = EtherSpec(
    ...     payload_fields=("embedding",),
    ...     metadata_fields=("source", "dim"),
    ...     extra_fields="keep",
    ...     renames={"embedding": "vec.values"},
    ...     kind="embedding"
    ... )
    >>>
    >>> # Use in Ether registration
    >>> @Ether.register(
    ...     payload=["embedding"],
    ...     metadata=["source", "dim"],
    ...     renames={"embedding": "vec.values"},
    ...     kind="embedding"
    ... )
    ... class EmbeddingModel(BaseModel):
    ...     embedding: list[float]
    ...     source: str
    ...     dim: int
"""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass
class EtherSpec:
    """Defines how model fields map to Ether envelope payload and metadata sections.

    This dataclass specifies which model fields should be mapped to the payload
    and metadata sections of an Ether envelope, along with field renames and
    validation rules for extra fields.

    Args:
        payload_fields: Tuple of field names to map to Ether.payload
        metadata_fields: Tuple of field names to map to Ether.metadata
        extra_fields: How to handle unmapped fields ("ignore" | "keep" | "error")
        renames: Optional mapping from model field names to Ether dot paths
        kind: Optional Ether kind identifier

    Raises:
        RuntimeError: If fields appear in both payload and metadata, or if
                     renames create duplicate mappings

    Examples:
        >>> spec = EtherSpec(
        ...     payload_fields=("embedding",),
        ...     metadata_fields=("source", "dim"),
        ...     extra_fields="keep",
        ...     renames={"embedding": "vec.values"},
        ...     kind="embedding"
        ... )
    """

    payload_fields: tuple[str, ...]
    metadata_fields: tuple[str, ...]
    extra_fields: str = "ignore"  # "ignore" | "keep" | "error"
    renames: Mapping[str, str] | None = None  # model_field -> ether dot path
    kind: str | None = None

    def __post_init__(self) -> None:
        """Validate the EtherSpec configuration after initialization.

        Performs validation checks:
        - Ensures renames is a dict
        - Checks for fields that appear in both payload and metadata
        - Validates that no two fields map to the same Ether path via renames

        Raises:
            RuntimeError: If validation fails

        Examples:
            >>> # This will raise RuntimeError due to duplicate field
            >>> spec = EtherSpec(
            ...     payload_fields=("field1",),
            ...     metadata_fields=("field1",),  # Same field in both
            ... )
        """
        # Convert renames to dict if None
        object.__setattr__(self, "renames", dict(self.renames or {}))

        # Check for fields in both payload and metadata
        dup = set(self.payload_fields) & set(self.metadata_fields)
        if dup:
            raise RuntimeError(f"Fields in both payload & metadata: {sorted(dup)}")

        # Check for duplicate mappings in renames
        used = {}
        if self.renames:
            for model_field, path in self.renames.items():
                if path in used:
                    raise RuntimeError(f"Duplicate mapping for ether path '{path}'")
                used[path] = model_field
