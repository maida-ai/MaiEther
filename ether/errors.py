"""Error types for the MaiEther package.

This module defines custom exception types used throughout the MaiEther framework
for handling registration and conversion errors.
"""


class RegistrationError(RuntimeError):
    """Raised when there is an error during model registration with Ether.

    This exception is raised when attempting to register a model with Ether
    but the registration fails due to invalid configuration, unknown fields,
    or other registration-related issues.

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

    Examples:
        >>> ether = Ether(kind="embedding", payload={}, metadata={})
        >>> ether.as_model(SomeModel, require_kind=True)
        ... # Raises ConversionError if kind doesn't match
    """
