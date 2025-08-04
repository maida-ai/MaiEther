"""Error types for the MaiEther package.

This module defines custom exception types used throughout the MaiEther framework
for handling registration and conversion errors.

The module provides two main exception types:
- RegistrationError: For model registration issues
- ConversionError: For Ether conversion issues

Examples:
    >>> from ether import Ether, RegistrationError, ConversionError
    >>> from pydantic import BaseModel
    >>>
    >>> # RegistrationError example
    >>> try:
    ...     @Ether.register(payload=["unknown_field"])
    ...     class MyModel(BaseModel):
    ...         valid_field: str
    ... except RegistrationError as e:
    ...     print(f"Registration failed: {e}")
    >>>
    >>> # ConversionError example
    >>> ether = Ether(kind="embedding", payload={}, metadata={})
    >>> try:
    ...     ether.as_model(SomeModel, require_kind=True)
    ... except ConversionError as e:
    ...     print(f"Conversion failed: {e}")
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ErrorStatus:
    success: bool = False
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.success

    def __str__(self) -> str:
        return self.message


class RegistrationError(RuntimeError):
    """Raised when there is an error during model registration with Ether.

    This exception is raised when attempting to register a model with Ether
    but the registration fails due to invalid configuration, unknown fields,
    or other registration-related issues.

    Args:
        message: Error message describing the registration failure

    Examples:
        >>> @Ether.register(payload=["unknown_field"])
        ... class MyModel(BaseModel):
        ...     valid_field: str
        ... # Raises RegistrationError: "MyModel: unknown field 'unknown_field'"
    """


class ConversionError(RuntimeError):
    """Raised when there is an error during Ether conversion operations.

    This exception is raised when converting between models and Ether envelopes
    fails due to missing required fields, kind mismatches, or other conversion
    issues.

    Args:
        message: Error message describing the conversion failure

    Examples:
        >>> ether = Ether(kind="embedding", payload={}, metadata={})
        >>> ether.as_model(SomeModel, require_kind=True)
        ... # Raises ConversionError if kind doesn't match
    """
