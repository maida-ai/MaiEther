"""Tests for automatic tracing and lineage functionality."""

from datetime import datetime

from pydantic import BaseModel

from ether import Ether, Node


class TestAutomaticTracingMetadata:
    """Test automatic population of tracing metadata in Ether."""

    def test_auto_populate_trace_id_when_missing(self) -> None:
        """Test that trace_id is auto-populated when missing."""
        ether = Ether(kind="test", schema_version=1, payload={}, metadata={})

        assert "trace_id" in ether.metadata
        assert isinstance(ether.metadata["trace_id"], str)
        # Should be a valid UUID
        import uuid

        uuid.UUID(ether.metadata["trace_id"])

    def test_auto_populate_created_at_when_missing(self) -> None:
        """Test that created_at is auto-populated when missing."""
        ether = Ether(kind="test", schema_version=1, payload={}, metadata={})

        assert "created_at" in ether.metadata
        assert isinstance(ether.metadata["created_at"], str)
        # Should be a valid RFC 3339 timestamp
        datetime.fromisoformat(ether.metadata["created_at"])

    def test_preserve_existing_trace_id(self) -> None:
        """Test that existing trace_id is preserved."""
        existing_trace_id = "test-trace-id-123"
        ether = Ether(kind="test", schema_version=1, payload={}, metadata={"trace_id": existing_trace_id})

        assert ether.metadata["trace_id"] == existing_trace_id

    def test_preserve_existing_created_at(self) -> None:
        """Test that existing created_at is preserved."""
        existing_timestamp = "2023-01-01T12:00:00+00:00"
        ether = Ether(kind="test", schema_version=1, payload={}, metadata={"created_at": existing_timestamp})

        assert ether.metadata["created_at"] == existing_timestamp

    def test_auto_populate_both_when_missing(self) -> None:
        """Test that both trace_id and created_at are populated when missing."""
        ether = Ether(kind="test", schema_version=1, payload={}, metadata={})

        assert "trace_id" in ether.metadata
        assert "created_at" in ether.metadata
        assert isinstance(ether.metadata["trace_id"], str)
        assert isinstance(ether.metadata["created_at"], str)

    def test_auto_populate_with_model_conversion(self) -> None:
        """Test that tracing metadata is populated when converting from model."""

        @Ether.register(payload=["data"], metadata=[], kind="test")
        class TestModel(BaseModel):
            data: str

        model = TestModel(data="test")
        ether = Ether(model)

        assert "trace_id" in ether.metadata
        assert "created_at" in ether.metadata


class TestNodeLineage:
    """Test Node.append_lineage() functionality."""

    def test_append_lineage_creates_list_if_missing(self) -> None:
        """Test that append_lineage creates lineage list if missing."""
        node = Node("test-node", "1.0.0")
        ether = Ether(kind="test", schema_version=1, payload={}, metadata={})

        node.append_lineage(ether)

        assert "lineage" in ether.metadata
        assert isinstance(ether.metadata["lineage"], list)
        assert len(ether.metadata["lineage"]) == 1

    def test_append_lineage_adds_correct_entry(self) -> None:
        """Test that append_lineage adds correct lineage entry."""
        node = Node("test-node", "1.0.0")
        ether = Ether(kind="test", schema_version=1, payload={}, metadata={})

        node.append_lineage(ether)

        lineage_entry = ether.metadata["lineage"][0]
        assert lineage_entry["node"] == "test-node"
        assert lineage_entry["version"] == "1.0.0"
        assert "ts" in lineage_entry
        assert isinstance(lineage_entry["ts"], str)
        # Should be a valid RFC 3339 timestamp
        datetime.fromisoformat(lineage_entry["ts"])

    def test_append_lineage_preserves_existing_lineage(self) -> None:
        """Test that append_lineage preserves existing lineage entries."""
        node = Node("test-node", "1.0.0")
        existing_lineage = [{"node": "previous", "version": "0.1.0", "ts": "2023-01-01T12:00:00+00:00"}]
        ether = Ether(kind="test", schema_version=1, payload={}, metadata={"lineage": existing_lineage})

        node.append_lineage(ether)

        assert len(ether.metadata["lineage"]) == 2
        assert ether.metadata["lineage"][0] == existing_lineage[0]
        assert ether.metadata["lineage"][1]["node"] == "test-node"

    def test_append_lineage_multiple_nodes(self) -> None:
        """Test that multiple nodes can append lineage."""
        node1 = Node("alpha", "1.0.0")
        node2 = Node("beta", "2.0.0")
        ether = Ether(kind="test", schema_version=1, payload={}, metadata={})

        node1.append_lineage(ether)
        node2.append_lineage(ether)

        assert len(ether.metadata["lineage"]) == 2
        assert ether.metadata["lineage"][0]["node"] == "alpha"
        assert ether.metadata["lineage"][1]["node"] == "beta"


class TestDummyNodes:
    """Test dummy node subclasses as specified in acceptance criteria."""

    class Alpha(Node):
        """Dummy node Alpha that calls append_lineage and returns envelope."""

        def process(self, eth: Ether) -> Ether:
            """Process by calling append_lineage and returning envelope."""
            self.append_lineage(eth)
            return eth

    class Beta(Node):
        """Dummy node Beta that calls append_lineage and returns envelope."""

        def process(self, eth: Ether) -> Ether:
            """Process by calling append_lineage and returning envelope."""
            self.append_lineage(eth)
            return eth

    def test_dummy_nodes_append_lineage(self) -> None:
        """Test that dummy nodes properly append lineage."""
        alpha = self.Alpha("alpha-node", "1.0.0")
        beta = self.Beta("beta-node", "2.0.0")

        ether = Ether(kind="test", schema_version=1, payload={}, metadata={})

        # Pass through both nodes
        result = beta.process(alpha.process(ether))

        assert len(result.metadata["lineage"]) == 2
        assert result.metadata["lineage"][0]["node"] == "alpha-node"
        assert result.metadata["lineage"][0]["version"] == "1.0.0"
        assert result.metadata["lineage"][1]["node"] == "beta-node"
        assert result.metadata["lineage"][1]["version"] == "2.0.0"

        # Verify all lineage entries have required keys
        for entry in result.metadata["lineage"]:
            assert "node" in entry
            assert "version" in entry
            assert "ts" in entry
            assert entry["node"] != ""
            assert entry["version"] != ""
            assert entry["ts"] != ""

    def test_lineage_timestamps_are_valid_rfc3339(self) -> None:
        """Test that lineage timestamps are valid RFC 3339 format."""
        alpha = self.Alpha("alpha-node", "1.0.0")
        ether = Ether(kind="test", schema_version=1, payload={}, metadata={})

        alpha.process(ether)

        for entry in ether.metadata["lineage"]:
            # Should be parseable as RFC 3339 timestamp with Z suffix
            assert entry["ts"].endswith("Z")
            # Convert Z to +00:00 for fromisoformat
            datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))


class TestTracingAndLineageIntegration:
    """Test integration of tracing and lineage functionality."""

    def test_tracing_and_lineage_work_together(self) -> None:
        """Test that tracing metadata and lineage work together."""
        node = Node("test-node", "1.0.0")
        ether = Ether(kind="test", schema_version=1, payload={}, metadata={})

        # Should have tracing metadata
        assert "trace_id" in ether.metadata
        assert "created_at" in ether.metadata

        # Add lineage
        node.append_lineage(ether)

        # Should have both tracing and lineage
        assert "lineage" in ether.metadata
        assert len(ether.metadata["lineage"]) == 1

        # Verify lineage entry structure
        entry = ether.metadata["lineage"][0]
        assert entry["node"] == "test-node"
        assert entry["version"] == "1.0.0"
        assert "ts" in entry

    def test_trace_id_format_is_uuid_v4(self) -> None:
        """Test that trace_id is a valid UUID v4."""
        ether = Ether(kind="test", schema_version=1, payload={}, metadata={})

        import uuid

        trace_id = ether.metadata["trace_id"]
        uuid_obj = uuid.UUID(trace_id)
        assert uuid_obj.version == 4

    def test_created_at_format_is_rfc3339(self) -> None:
        """Test that created_at is in RFC 3339 format with Z suffix."""
        ether = Ether(kind="test", schema_version=1, payload={}, metadata={})

        created_at = ether.metadata["created_at"]
        # Should be parseable as RFC 3339 timestamp with Z suffix
        assert created_at.endswith("Z")
        # Convert Z to +00:00 for fromisoformat
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        assert dt.tzinfo is not None  # Should have timezone info

    def test_rfc3339_now_function(self) -> None:
        """Test that rfc3339_now() generates correct format."""
        from ether.utils import rfc3339_now

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
