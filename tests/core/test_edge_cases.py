"""Tests for Ether edge cases and special scenarios."""

from ether.core import Ether


class TestEtherEdgeCases:
    """Test Ether edge cases and special scenarios."""

    def test_ether_with_large_payload(self) -> None:
        """Test Ether with large payload."""
        large_payload = {"data": list(range(1000))}
        ether = Ether(
            kind="data",
            payload=large_payload,
            metadata={},
        )

        assert len(ether.payload["data"]) == 1000
        assert ether.payload["data"][0] == 0
        assert ether.payload["data"][-1] == 999

    def test_ether_with_complex_metadata(self) -> None:
        """Test Ether with complex metadata structure."""
        complex_metadata = {
            "trace_id": "123e4567-e89b-12d3-a456-426614174000",
            "span_id": "456e7890-e89b-12d3-a456-426614174001",
            "created_at": "2023-01-01T00:00:00Z",
            "producer": "test-node",
            "lineage": [{"node": "input", "version": "1.0", "ts": "2023-01-01T00:00:00Z"}],
        }

        ether = Ether(
            kind="embedding",
            payload={},
            metadata=complex_metadata,
        )

        assert ether.metadata["trace_id"] == "123e4567-e89b-12d3-a456-426614174000"
        assert len(ether.metadata["lineage"]) == 1

    def test_ether_with_empty_attachments_list(self) -> None:
        """Test Ether with empty attachments list."""
        ether = Ether(
            kind="text",
            payload={"text": "Hello"},
            metadata={},
            attachments=[],
        )

        assert ether.attachments == []
        assert len(ether.attachments) == 0

    def test_ether_with_none_values_in_optional_fields(self) -> None:
        """Test Ether with None values in optional fields."""
        ether = Ether(
            kind="embedding",
            payload={"dim": 768},
            metadata={"source": "bert-base"},
            extra_fields={},
            attachments=[],
        )

        assert ether.extra_fields == {}
        assert ether.attachments == []


class TestEtherImport:
    """Test Ether import functionality."""

    def test_import_from_package(self) -> None:
        """Test that Ether can be imported from the package."""
        from ether import Ether

        ether = Ether(
            kind="test",
            payload={},
            metadata={},
        )

        assert ether.kind == "test"

    def test_ether_is_available_in_all(self) -> None:
        """Test that Ether is available in __all__."""
        from ether import __all__

        assert "Ether" in __all__
