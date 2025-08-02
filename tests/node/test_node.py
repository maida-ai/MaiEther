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

        assert "Node Node must override _process() method" in str(exc_info.value)

    def test_node_subclass_without_override_raises_not_implemented_error(self) -> None:
        """Test that instantiating a subclass without overriding _process() raises NotImplementedError."""

        class TestNode(Node):
            """Test subclass that doesn't override _process()."""

            pass

        node = TestNode("test-subclass")
        ether = Ether(kind="test", schema_version=1, payload={}, metadata={})

        with pytest.raises(NotImplementedError) as exc_info:
            node.process(ether)

        assert "Node TestNode must override _process() method" in str(exc_info.value)

    def test_node_subclass_with_override_works(self) -> None:
        """Test that a subclass that overrides _process() works correctly."""

        class WorkingNode(Node):
            """Test subclass that properly overrides _process()."""

            def _process(self, eth: Ether) -> Ether:
                """Override _process method."""
                # Just return the input for this test
                return eth

        node = WorkingNode("working-node")
        ether = Ether(kind="test", schema_version=1, payload={"data": "test"}, metadata={"source": "test"})

        result = node.process(ether)

        assert result is ether  # Should return the same object
        assert result.kind == "test"
        assert result.payload == {"data": "test"}
        # Metadata should contain both user-provided and auto-populated fields
        assert result.metadata["source"] == "test"
        assert "trace_id" in result.metadata
        assert "created_at" in result.metadata

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

    def test_node_validation_with_empty_accepts(self) -> None:
        """Test that Node with empty accepts allows any input."""
        node = Node("test-node")
        ether = Ether(kind="any", schema_version=1, payload={}, metadata={})

        # Should not raise any error
        node.validate_input(ether)

    def test_node_validation_with_matching_input(self) -> None:
        """Test that Node validation passes with matching input."""

        class TestNode(Node):
            """Test node with specific accepts."""

            accepts = {("text", 1), ("tokens", 1)}

        node = TestNode("test-node")
        ether = Ether(kind="text", schema_version=1, payload={}, metadata={})

        # Should not raise any error
        node.validate_input(ether)

    def test_node_validation_with_non_matching_input(self) -> None:
        """Test that Node validation fails with non-matching input."""

        class TestNode(Node):
            """Test node with specific accepts."""

            accepts = {("text", 1), ("tokens", 1)}

        node = TestNode("test-node")
        ether = Ether(kind="embedding", schema_version=1, payload={}, metadata={})

        with pytest.raises(ValueError) as exc_info:
            node.validate_input(ether)

        assert "TestNode expects one of: text.v1, tokens.v1, got embedding.v1" in str(exc_info.value)

    def test_node_validation_with_wrong_version(self) -> None:
        """Test that Node validation fails with wrong version."""

        class TestNode(Node):
            """Test node with specific accepts."""

            accepts = {("text", 1)}

        node = TestNode("test-node")
        ether = Ether(kind="text", schema_version=2, payload={}, metadata={})

        with pytest.raises(ValueError) as exc_info:
            node.validate_input(ether)

        assert "TestNode expects one of: text.v1, got text.v2" in str(exc_info.value)
