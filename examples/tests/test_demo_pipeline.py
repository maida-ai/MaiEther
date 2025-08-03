"""Integration tests for the demo pipeline.

This module contains integration tests that verify the complete demo pipeline
works correctly, ensuring that the final envelope has the expected properties
and that the pipeline processes data through all nodes as expected.
"""

import time

import pytest

from ether import EmbeddingModel, Ether, Registry, TextModel, TokenModel
from ether.spec import EtherSpec
from examples.demo.nodes.embedder import EmbedderNode
from examples.demo.nodes.tokenizer import TokenizerNode


class TestDemoPipeline:
    """Test the complete demo pipeline integration."""

    def setup_method(self) -> None:
        """Set up test method by registering required models."""
        # Clear registry for clean test
        Registry.clear_spec()

        # Register TextModel
        text_spec = EtherSpec(
            payload_fields=("text",),
            metadata_fields=("lang", "encoding", "detected_lang_conf"),
            extra_fields="ignore",
            renames={},
            kind="text",
        )
        Registry.set_spec(TextModel, text_spec)

        # Register TokenModel
        token_spec = EtherSpec(
            payload_fields=("ids", "mask"),
            metadata_fields=("vocab", "truncation", "offsets"),
            extra_fields="ignore",
            renames={},
            kind="tokens",
        )
        Registry.set_spec(TokenModel, token_spec)

        # Register EmbeddingModel
        embedding_spec = EtherSpec(
            payload_fields=("values", "dim"),
            metadata_fields=("source", "norm", "quantized", "dtype", "codec"),
            extra_fields="ignore",
            renames={},
            kind="embedding",
        )
        Registry.set_spec(EmbeddingModel, embedding_spec)

    def test_demo_pipeline_complete_flow(self) -> None:
        """Test the complete demo pipeline flow from text to embedding."""
        # Step 1: Create TextModel with the specified text
        text_model = TextModel(text="Maida makes tiny AI shine.")
        initial_eth = Ether(text_model)

        # Verify initial envelope
        assert initial_eth.kind == "text"
        assert initial_eth.schema_version == 1
        assert "text" in initial_eth.payload

        # Step 2: Process through TokenizerNode
        tokenizer = TokenizerNode()
        tokenized_eth = tokenizer.process(initial_eth)

        # Verify tokenized envelope
        assert tokenized_eth.kind == "tokens"
        assert tokenized_eth.schema_version == 1
        assert "ids" in tokenized_eth.payload
        assert isinstance(tokenized_eth.payload["ids"], list)
        assert len(tokenized_eth.payload["ids"]) > 0  # Should have tokens

        # Verify lineage was appended
        assert "lineage" in tokenized_eth.metadata
        assert len(tokenized_eth.metadata["lineage"]) == 1
        lineage_entry = tokenized_eth.metadata["lineage"][0]
        assert lineage_entry["node"] == "tokenizer"
        assert lineage_entry["version"] == "0.1.0"

        # Step 3: Process through EmbedderNode
        embedder = EmbedderNode()
        final_eth = embedder.process(tokenized_eth)

        # Verify final envelope properties (acceptance criteria)
        assert final_eth.kind == "embedding", f"Expected kind='embedding', got {final_eth.kind}"
        assert final_eth.schema_version == 1, f"Expected schema_version=1, got {final_eth.schema_version}"

        # Verify embedding-specific properties
        assert "values" in final_eth.payload
        assert "dim" in final_eth.payload
        assert final_eth.payload["dim"] == 128
        assert isinstance(final_eth.payload["values"], list)
        assert len(final_eth.payload["values"]) == 128
        assert all(isinstance(v, float) for v in final_eth.payload["values"])

        # Verify lineage was appended again
        assert "lineage" in final_eth.metadata
        assert len(final_eth.metadata["lineage"]) == 2  # tokenizer + embedder
        embedder_lineage = final_eth.metadata["lineage"][1]
        assert embedder_lineage["node"] == "embedder"
        assert embedder_lineage["version"] == "0.1.0"

    def test_demo_pipeline_performance(self) -> None:
        """Test that the demo pipeline runs in <500ms on a 4-core laptop."""
        # Create the pipeline components
        text_model = TextModel(text="Maida makes tiny AI shine.")
        initial_eth = Ether(text_model)
        tokenizer = TokenizerNode()
        embedder = EmbedderNode()

        # Measure execution time
        start_time = time.time()

        # Run the pipeline
        tokenized_eth = tokenizer.process(initial_eth)
        final_eth = embedder.process(tokenized_eth)

        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000

        # Verify performance requirement
        assert execution_time_ms < 500, f"Pipeline took {execution_time_ms:.2f}ms, expected <500ms"

        # Verify final envelope properties
        assert final_eth.kind == "embedding"
        assert final_eth.schema_version == 1

    def test_demo_pipeline_deterministic_output(self) -> None:
        """Test that the demo pipeline produces deterministic output."""
        text_model = TextModel(text="Maida makes tiny AI shine.")
        initial_eth = Ether(text_model)
        tokenizer = TokenizerNode()
        embedder = EmbedderNode()

        # Run pipeline twice
        tokenized_eth1 = tokenizer.process(initial_eth)
        final_eth1 = embedder.process(tokenized_eth1)

        tokenized_eth2 = tokenizer.process(initial_eth)
        final_eth2 = embedder.process(tokenized_eth2)

        # Verify deterministic tokenization
        assert tokenized_eth1.payload["ids"] == tokenized_eth2.payload["ids"]

        # Verify deterministic embedding (both should use same random seed)
        assert final_eth1.payload["values"] == final_eth2.payload["values"]
        assert final_eth1.payload["dim"] == final_eth2.payload["dim"]

    def test_demo_pipeline_different_inputs(self) -> None:
        """Test that the demo pipeline works with different input texts."""
        test_texts = [
            "Hello world",
            "This is a test sentence.",
            "AI and machine learning are fascinating topics.",
            "",  # Empty text
        ]

        tokenizer = TokenizerNode()
        embedder = EmbedderNode()

        for text in test_texts:
            text_model = TextModel(text=text)
            initial_eth = Ether(text_model)

            # Process through pipeline
            tokenized_eth = tokenizer.process(initial_eth)
            final_eth = embedder.process(tokenized_eth)

            # Verify final envelope properties
            assert final_eth.kind == "embedding"
            assert final_eth.schema_version == 1
            assert final_eth.payload["dim"] == 128
            assert len(final_eth.payload["values"]) == 128

    def test_demo_pipeline_lineage_tracking(self) -> None:
        """Test that the demo pipeline correctly tracks lineage information."""
        text_model = TextModel(text="Maida makes tiny AI shine.")
        initial_eth = Ether(text_model)
        tokenizer = TokenizerNode("test-tokenizer", "1.0.0")
        embedder = EmbedderNode("test-embedder", "2.0.0")

        # Process through pipeline
        tokenized_eth = tokenizer.process(initial_eth)
        final_eth = embedder.process(tokenized_eth)

        # Verify lineage information
        assert "lineage" in final_eth.metadata
        assert len(final_eth.metadata["lineage"]) == 2

        # Check tokenizer lineage
        tokenizer_lineage = final_eth.metadata["lineage"][0]
        assert tokenizer_lineage["node"] == "test-tokenizer"
        assert tokenizer_lineage["version"] == "1.0.0"
        assert "ts" in tokenizer_lineage

        # Check embedder lineage
        embedder_lineage = final_eth.metadata["lineage"][1]
        assert embedder_lineage["node"] == "test-embedder"
        assert embedder_lineage["version"] == "2.0.0"
        assert "ts" in embedder_lineage

    def test_demo_pipeline_error_handling(self) -> None:
        """Test that the demo pipeline handles errors appropriately."""
        tokenizer = TokenizerNode()

        # Test with invalid input kind
        invalid_eth = Ether(kind="embedding", schema_version=1, payload={}, metadata={})

        with pytest.raises(ValueError) as exc_info:
            tokenizer.process(invalid_eth)

        assert "TokenizerNode expects one of: text.v1, got embedding.v1" in str(exc_info.value)

        # Test with valid input but wrong version
        text_model = TextModel(text="test")
        valid_eth = Ether(text_model)
        valid_eth.schema_version = 2  # Wrong version

        with pytest.raises(ValueError) as exc_info:
            tokenizer.process(valid_eth)

        assert "TokenizerNode expects one of: text.v1, got text.v2" in str(exc_info.value)
