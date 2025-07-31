"""Tests for the error types module."""

import pytest

from ether.errors import ConversionError, RegistrationError


def test_registration_error_can_be_raised() -> None:
    """Test that RegistrationError can be raised and caught."""
    with pytest.raises(RegistrationError, match="test error"):
        raise RegistrationError("test error")


def test_conversion_error_can_be_raised() -> None:
    """Test that ConversionError can be raised and caught."""
    with pytest.raises(ConversionError, match="test error"):
        raise ConversionError("test error")


def test_errors_are_runtime_error_subclasses() -> None:
    """Test that both error types are subclasses of RuntimeError."""
    assert issubclass(RegistrationError, RuntimeError)
    assert issubclass(ConversionError, RuntimeError)


def test_errors_can_be_imported_from_package() -> None:
    """Test that errors can be imported from the main package."""
    from ether import ConversionError, RegistrationError

    # Verify they are the same classes
    from ether.errors import ConversionError as ConversionErrorDirect
    from ether.errors import RegistrationError as RegistrationErrorDirect

    assert ConversionError is ConversionErrorDirect
    assert RegistrationError is RegistrationErrorDirect


def test_error_messages_are_preserved() -> None:
    """Test that error messages are properly preserved."""
    test_message = "This is a test error message"

    reg_error = RegistrationError(test_message)
    conv_error = ConversionError(test_message)

    assert str(reg_error) == test_message
    assert str(conv_error) == test_message
