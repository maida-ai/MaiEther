"""Tests for the EmbedderNode class."""

import pytest

from examples.demo.nodes.embedder import EmbedderNode
from ether import EmbeddingModel, Ether, TokenModel
from ether.core import _spec_registry
from ether.spec import EtherSpec


class TestEmbedderNode:
    """Test the EmbedderNode class."""

    def setup_method(self) -> None:
        """Set up test method by registering required models."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Register TokenModel
        token_spec = EtherSpec(
            payload_fields=("ids", "mask"),
            metadata_fields=("vocab", "truncation", "offsets"),
            extra_fields="ignore",
            renames={},
            kind="tokens",
        )
        _spec_registry[TokenModel] = token_spec

        # Register EmbeddingModel
        embedding_spec = EtherSpec(
            payload_fields=("values", "dim"),
            metadata_fields=("source", "norm", "quantized", "dtype", "codec"),
            extra_fields="ignore",
            renames={},
            kind="embedding",
        )
        _spec_registry[EmbeddingModel] = embedding_spec

    def test_embedder_node_instantiation(self) -> None:
        """Test that EmbedderNode can be instantiated with default attributes."""
        node = EmbedderNode()

        assert node.name == "embedder"
        assert node.version == "0.1.0"

    def test_embedder_node_instantiation_with_custom_attributes(self) -> None:
        """Test that EmbedderNode can be instantiated with custom attributes."""
        node = EmbedderNode("custom-embedder", "1.2.3")

        assert node.name == "custom-embedder"
        assert node.version == "1.2.3"

    def test_embedder_node_accepts_and_emits(self) -> None:
        """Test that EmbedderNode has correct accepts and emits sets."""
        assert EmbedderNode.accepts == {("tokens", 1)}
        assert EmbedderNode.emits == {("embedding", 1)}

    def test_embedder_node_process_basic_tokens(self) -> None:
        """Test that EmbedderNode processes basic tokens correctly."""
        # Create a TokenModel and convert to Ether
        token_model = TokenModel(ids=[1, 2, 3, 4, 5], vocab="test-vocab")
        input_eth = Ether.from_model(token_model)

        # Create EmbedderNode and process
        node = EmbedderNode()
        result_eth = node.process(input_eth)

        # Verify the result is an embedding.v1 envelope
        assert result_eth.kind == "embedding"
        assert result_eth.schema_version == 1

        # Verify the payload contains values and dim
        assert "values" in result_eth.payload
        assert "dim" in result_eth.payload
        assert result_eth.payload["dim"] == 128

        # Verify the values are a list of floats with correct length
        values = result_eth.payload["values"]
        assert isinstance(values, list)
        assert len(values) == 128
        assert all(isinstance(v, float) for v in values)

        # Verify lineage was appended
        assert "lineage" in result_eth.metadata
        assert len(result_eth.metadata["lineage"]) == 1
        lineage_entry = result_eth.metadata["lineage"][0]
        assert lineage_entry["node"] == "embedder"
        assert lineage_entry["version"] == "0.1.0"
        assert "ts" in lineage_entry

    def test_embedder_node_process_empty_tokens(self) -> None:
        """Test that EmbedderNode processes empty tokens correctly."""
        # Create a TokenModel with empty tokens and convert to Ether
        token_model = TokenModel(ids=[], vocab="test-vocab")
        input_eth = Ether.from_model(token_model)

        # Create EmbedderNode and process
        node = EmbedderNode()
        result_eth = node.process(input_eth)

        # Verify the result is an embedding.v1 envelope
        assert result_eth.kind == "embedding"
        assert result_eth.schema_version == 1

        # Convert back to EmbeddingModel to verify content
        embedding_model = result_eth.as_model(EmbeddingModel)
        assert embedding_model.dim == 128
        assert embedding_model.source == "demo"
        assert embedding_model.values is not None
        assert len(embedding_model.values) == 128
        assert all(isinstance(v, float) for v in embedding_model.values)

    def test_embedder_node_process_large_tokens(self) -> None:
        """Test that EmbedderNode processes large token lists correctly."""
        # Create a TokenModel with many tokens and convert to Ether
        token_model = TokenModel(ids=list(range(1000)), vocab="test-vocab")
        input_eth = Ether.from_model(token_model)

        # Create EmbedderNode and process
        node = EmbedderNode()
        result_eth = node.process(input_eth)

        # Verify the result is an embedding.v1 envelope
        assert result_eth.kind == "embedding"
        assert result_eth.schema_version == 1

        # Convert back to EmbeddingModel to verify content
        embedding_model = result_eth.as_model(EmbeddingModel)
        assert embedding_model.dim == 128
        assert embedding_model.source == "demo"
        assert embedding_model.values is not None
        assert len(embedding_model.values) == 128
        assert all(isinstance(v, float) for v in embedding_model.values)

    def test_embedder_node_deterministic_output(self) -> None:
        """Test that EmbedderNode produces deterministic output."""
        # Create a TokenModel and convert to Ether
        token_model = TokenModel(ids=[1, 2, 3], vocab="test-vocab")
        input_eth = Ether.from_model(token_model)

        # Create EmbedderNode and process twice
        node = EmbedderNode()
        result_eth1 = node.process(input_eth)
        result_eth2 = node.process(input_eth)

        # Verify both results are identical
        assert result_eth1.payload["values"] == result_eth2.payload["values"]
        assert result_eth1.payload["dim"] == result_eth2.payload["dim"]
        assert result_eth1.metadata.get("source") == result_eth2.metadata.get("source")

    def test_embedder_node_ignores_token_ids(self) -> None:
        """Test that EmbedderNode ignores token IDs as specified."""
        # Create two TokenModels with different token IDs
        token_model1 = TokenModel(ids=[1, 2, 3], vocab="test-vocab")
        token_model2 = TokenModel(ids=[100, 200, 300], vocab="test-vocab")

        input_eth1 = Ether.from_model(token_model1)
        input_eth2 = Ether.from_model(token_model2)

        # Create EmbedderNode and process both
        node = EmbedderNode()
        result_eth1 = node.process(input_eth1)
        result_eth2 = node.process(input_eth2)

        # Verify both results are identical (ignoring token IDs)
        assert result_eth1.payload["values"] == result_eth2.payload["values"]
        assert result_eth1.payload["dim"] == result_eth2.payload["dim"]
        assert result_eth1.metadata.get("source") == result_eth2.metadata.get("source")

    def test_embedder_node_invalid_input_kind(self) -> None:
        """Test that EmbedderNode raises error for invalid input kind."""
        # Create an Ether with wrong kind
        invalid_eth = Ether(kind="text", schema_version=1, payload={}, metadata={})

        # Create EmbedderNode and attempt to process
        node = EmbedderNode()

        with pytest.raises(ValueError) as exc_info:
            node.process(invalid_eth)

        assert "EmbedderNode expects one of: tokens.v1, got text.v1" in str(exc_info.value)

    def test_embedder_node_invalid_input_version(self) -> None:
        """Test that EmbedderNode raises error for invalid input version."""
        # Create a TokenModel and convert to Ether with wrong version
        token_model = TokenModel(ids=[1, 2, 3], vocab="test-vocab")
        input_eth = Ether.from_model(token_model)
        input_eth.schema_version = 2  # Wrong version

        # Create EmbedderNode and attempt to process
        node = EmbedderNode()

        with pytest.raises(ValueError) as exc_info:
            node.process(input_eth)

        assert "EmbedderNode expects one of: tokens.v1, got tokens.v2" in str(exc_info.value)

    def test_embedder_node_lineage_appended(self) -> None:
        """Test that EmbedderNode appends lineage information correctly."""
        # Create a TokenModel and convert to Ether
        token_model = TokenModel(ids=[1, 2, 3], vocab="test-vocab")
        input_eth = Ether.from_model(token_model)

        # Create EmbedderNode and process
        node = EmbedderNode("test-embedder", "2.0.0")
        result_eth = node.process(input_eth)

        # Verify lineage was appended
        assert "lineage" in result_eth.metadata
        assert len(result_eth.metadata["lineage"]) == 1

        lineage_entry = result_eth.metadata["lineage"][0]
        assert lineage_entry["node"] == "test-embedder"
        assert lineage_entry["version"] == "2.0.0"
        assert "ts" in lineage_entry

    def test_embedder_node_preserves_metadata(self) -> None:
        """Test that EmbedderNode preserves metadata from input."""
        # Create a TokenModel with metadata and convert to Ether
        token_model = TokenModel(
            ids=[1, 2, 3],
            mask=[1, 1, 0],
            vocab="test-vocab",
            truncation="longest_first",
            offsets=True,
        )
        input_eth = Ether.from_model(token_model)

        # Create EmbedderNode and process
        node = EmbedderNode()
        result_eth = node.process(input_eth)

        # Verify the result is an embedding.v1 envelope
        assert result_eth.kind == "embedding"
        assert result_eth.schema_version == 1

        # Convert back to EmbeddingModel to verify content
        embedding_model = result_eth.as_model(EmbeddingModel)
        assert embedding_model.dim == 128
        assert embedding_model.source == "demo"
        assert embedding_model.values is not None
        assert len(embedding_model.values) == 128
        assert all(isinstance(v, float) for v in embedding_model.values)

    def test_embedder_node_acceptance_criteria(self) -> None:
        """Test that EmbedderNode meets all acceptance criteria."""
        # Create a TokenModel and convert to Ether
        token_model = TokenModel(ids=[1, 2, 3, 4, 5], vocab="test-vocab")
        input_eth = Ether.from_model(token_model)

        # Create EmbedderNode and process
        node = EmbedderNode()
        result_eth = node.process(input_eth)

        # Acceptance criteria 1: Unit test passes dummy tokens envelope through EmbedderNode
        # Acceptance criteria 2: Asserts kind=="embedding"
        assert result_eth.kind == "embedding"

        # Acceptance criteria 3: Asserts len(payload["values"])==128
        assert len(result_eth.payload["values"]) == 128

        # Acceptance criteria 4: Lineage length increased by one
        assert "lineage" in result_eth.metadata
        assert len(result_eth.metadata["lineage"]) == 1
