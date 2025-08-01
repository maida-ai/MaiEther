"""Utility functions for the Ether package.

This module contains shared utility functions that can be used across
different modules in the Ether package.
"""

from datetime import UTC, datetime


def rfc3339_now() -> str:
    """Generate RFC 3339 timestamp with microsecond precision.

    Uses microsecond precision and 'Z' suffix for better cross-platform
    compatibility and to avoid clock skew issues in distributed systems.

    Returns:
        RFC 3339 formatted timestamp string
    """
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
