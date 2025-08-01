"""Tests for Ether inline size validation."""

import pytest

from ether import Attachment, Ether


class TestEtherInlineSizeValidation:
    """Test Ether constructor validation for large inline data."""

    def test_ether_with_small_payload(self) -> None:
        """Test that Ether accepts small payload data."""
        # Small list should be fine
        small_list = [1.0, 2.0, 3.0] * 1000  # ~24KB
        ether = Ether(
            kind="test",
            payload={"values": small_list},
            metadata={},
        )
        assert ether.payload["values"] == small_list

    def test_ether_rejects_large_payload_list(self) -> None:
        """Test that Ether rejects large lists in payload."""
        # Create a list larger than 64KB
        large_list = [1.0] * 17000  # ~136KB (17000 * 8 bytes per float)

        with pytest.raises(ValueError, match="Large inline data detected"):
            Ether(
                kind="test",
                payload={"values": large_list},
                metadata={},
            )

    def test_ether_rejects_large_metadata_list(self) -> None:
        """Test that Ether rejects large lists in metadata."""
        large_list = [1.0] * 17000  # ~136KB

        with pytest.raises(ValueError, match="Large inline data detected"):
            Ether(
                kind="test",
                payload={},
                metadata={"values": large_list},
            )

    def test_ether_rejects_large_extra_fields_list(self) -> None:
        """Test that Ether rejects large lists in extra_fields."""
        large_list = [1.0] * 17000  # ~136KB

        with pytest.raises(ValueError, match="Large inline data detected"):
            Ether(
                kind="test",
                payload={},
                metadata={},
                extra_fields={"values": large_list},
            )

    def test_ether_rejects_large_nested_data(self) -> None:
        """Test that Ether rejects large data in nested structures."""
        large_list = [1.0] * 17000  # ~136KB

        with pytest.raises(ValueError, match="Large inline data detected"):
            Ether(
                kind="test",
                payload={"nested": {"deep": {"values": large_list}}},
                metadata={},
            )

    def test_ether_rejects_large_string_data(self) -> None:
        """Test that Ether rejects large string data."""
        large_string = "x" * 70000  # ~70KB

        with pytest.raises(ValueError, match="Large inline data detected"):
            Ether(
                kind="test",
                payload={"text": large_string},
                metadata={},
            )

    def test_ether_rejects_large_bytes_data(self) -> None:
        """Test that Ether rejects large bytes data."""
        large_bytes = b"x" * 70000  # ~70KB

        with pytest.raises(ValueError, match="Large inline data detected"):
            Ether(
                kind="test",
                payload={"data": large_bytes},
                metadata={},
            )

    def test_ether_accepts_large_data_in_attachments(self) -> None:
        """Test that Ether accepts large data when properly stored in attachments."""
        # Create a large attachment (this should be fine)
        large_data = b"x" * 100000  # 100KB
        attachment = Attachment(
            id="large-data",
            inline_bytes=large_data,
            media_type="application/x-raw-tensor",
            codec="RAW_BYTES",
            size_bytes=len(large_data),
        )

        # This should work fine
        ether = Ether(
            kind="test",
            payload={},
            metadata={},
            attachments=[attachment],
        )
        assert len(ether.attachments) == 1
        assert ether.attachments[0].id == "large-data"

    def test_ether_error_message_suggests_attachments(self) -> None:
        """Test that error message suggests using attachments."""
        large_list = [1.0] * 17000  # ~136KB

        with pytest.raises(ValueError) as exc_info:
            Ether(
                kind="test",
                payload={"values": large_list},
                metadata={},
            )

        error_msg = str(exc_info.value)
        assert "Consider using attachments" in error_msg
        assert "Use Attachment.from_numpy()" in error_msg
        assert "64 KB" in error_msg

    def test_ether_error_message_includes_size_estimate(self) -> None:
        """Test that error message includes size estimate."""
        large_list = [1.0] * 17000  # ~136KB

        with pytest.raises(ValueError) as exc_info:
            Ether(
                kind="test",
                payload={"values": large_list},
                metadata={},
            )

        error_msg = str(exc_info.value)
        assert "estimated" in error_msg
        assert "bytes" in error_msg

    def test_ether_error_message_includes_path(self) -> None:
        """Test that error message includes the path to large data."""
        large_list = [1.0] * 17000  # ~136KB

        with pytest.raises(ValueError) as exc_info:
            Ether(
                kind="test",
                payload={"nested": {"deep": {"values": large_list}}},
                metadata={},
            )

        error_msg = str(exc_info.value)
        assert "nested.deep.values" in error_msg

    def test_ether_handles_mixed_small_and_large_data(self) -> None:
        """Test that Ether correctly identifies large data in mixed scenarios."""
        small_list = [1.0, 2.0, 3.0]
        large_list = [1.0] * 17000  # ~136KB

        with pytest.raises(ValueError) as exc_info:
            Ether(
                kind="test",
                payload={"small": small_list, "large": large_list},
                metadata={},
            )

        error_msg = str(exc_info.value)
        assert "large" in error_msg  # Should identify the large data
        assert "small" not in error_msg  # Should not mention small data

    def test_ether_handles_multiple_large_data_paths(self) -> None:
        """Test that Ether identifies multiple paths with large data."""
        large_list = [1.0] * 17000  # ~136KB

        with pytest.raises(ValueError) as exc_info:
            Ether(
                kind="test",
                payload={"values1": large_list},
                metadata={"values2": large_list},
            )

        error_msg = str(exc_info.value)
        assert "values1" in error_msg
        assert "values2" in error_msg
