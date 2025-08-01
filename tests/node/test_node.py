"""Tests for the Node base class."""

import pytest

from ether import Ether, Node


class TestNode:
    """Test the Node base class."""

    def test_node_instantiation(self) -> None:
        """Test that Node can be instantiated with required attributes."""
        node = Node("test-node")

        assert node.name == "test-node"
        assert node.version == "0.1.0"

    def test_node_instantiation_with_custom_version(self) -> None:
        """Test that Node can be instantiated with custom version."""
        node = Node("test-node", "1.2.3")

        assert node.name == "test-node"
        assert node.version == "1.2.3"

    def test_node_class_variables_default_to_empty_sets(self) -> None:
        """Test that Node.accepts and Node.emits default to empty sets."""
        assert Node.accepts == set()
        assert Node.emits == set()

    def test_node_process_raises_not_implemented_error(self) -> None:
        """Test that calling process() on Node raises NotImplementedError."""
        node = Node("test-node")

        # Create a minimal Ether for testing
        ether = Ether(kind="test", schema_version=1, payload={}, metadata={})

        with pytest.raises(NotImplementedError) as exc_info:
            node.process(ether)

        assert "Node Node must override process() method" in str(exc_info.value)

    def test_node_subclass_without_override_raises_not_implemented_error(self) -> None:
        """Test that instantiating a subclass without overriding process() raises NotImplementedError."""

        class TestNode(Node):
            """Test subclass that doesn't override process()."""

            pass

        node = TestNode("test-subclass")
        ether = Ether(kind="test", schema_version=1, payload={}, metadata={})

        with pytest.raises(NotImplementedError) as exc_info:
            node.process(ether)

        assert "Node TestNode must override process() method" in str(exc_info.value)

    def test_node_subclass_with_override_works(self) -> None:
        """Test that a subclass that overrides process() works correctly."""

        class WorkingNode(Node):
            """Test subclass that properly overrides process()."""

            def process(self, eth: Ether) -> Ether:
                """Override process method."""
                # Just return the input for this test
                return eth

        node = WorkingNode("working-node")
        ether = Ether(kind="test", schema_version=1, payload={"data": "test"}, metadata={"source": "test"})

        result = node.process(ether)

        assert result is ether  # Should return the same object
        assert result.kind == "test"
        assert result.payload == {"data": "test"}
        assert result.metadata == {"source": "test"}

    def test_node_accepts_and_emits_can_be_modified(self) -> None:
        """Test that accepts and emits can be modified by subclasses."""

        class CustomNode(Node):
            """Test subclass that sets accepts and emits."""

            accepts = {("text", 1), ("tokens", 1)}
            emits = {("embedding", 1)}

        assert CustomNode.accepts == {("text", 1), ("tokens", 1)}
        assert CustomNode.emits == {("embedding", 1)}

        # Base Node class should remain unchanged
        assert Node.accepts == set()
        assert Node.emits == set()
