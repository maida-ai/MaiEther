"""Tests for utility functions."""

from datetime import datetime, timedelta

from ether.utils import rfc3339_now


class TestRfc3339Now:
    """Test the rfc3339_now utility function."""

    def test_rfc3339_now_format(self) -> None:
        """Test that rfc3339_now() generates correct RFC 3339 format."""
        timestamp = rfc3339_now()

        # Should end with Z
        assert timestamp.endswith("Z")

        # Should be parseable as RFC 3339
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert dt.tzinfo is not None

        # Should have microsecond precision
        assert "." in timestamp
        microseconds = timestamp.split(".")[1].replace("Z", "")
        assert len(microseconds) == 6  # 6 digits for microseconds

    def test_rfc3339_now_timezone(self) -> None:
        """Test that rfc3339_now() uses UTC timezone."""
        timestamp = rfc3339_now()

        # Convert Z to +00:00 for fromisoformat
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        # Should be in UTC (offset should be 0)
        assert dt.tzinfo.utcoffset(dt) == timedelta(0)  # type: ignore[union-attr]

    def test_rfc3339_now_uniqueness(self) -> None:
        """Test that rfc3339_now() generates unique timestamps."""
        timestamps = [rfc3339_now() for _ in range(10)]

        # All timestamps should be unique (microsecond precision should ensure this)
        assert len(set(timestamps)) == len(timestamps)
