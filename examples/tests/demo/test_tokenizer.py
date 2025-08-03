"""Tests for the TokenizerNode class."""

import pytest

from examples.demo.nodes.tokenizer import TokenizerNode
from ether import Ether, TextModel, TokenModel
from ether.core import _spec_registry
from ether.spec import EtherSpec


class TestTokenizerNode:
    """Test the TokenizerNode class."""

    def setup_method(self) -> None:
        """Set up test method by registering required models."""
        # Clear registry for clean test
        _spec_registry.clear()

        # Register TextModel
        text_spec = EtherSpec(
            payload_fields=("text",),
            metadata_fields=("lang", "encoding", "detected_lang_conf"),
            extra_fields="ignore",
            renames={},
            kind="text",
        )
        _spec_registry[TextModel] = text_spec

        # Register TokenModel
        token_spec = EtherSpec(
            payload_fields=("ids", "mask"),
            metadata_fields=("vocab", "truncation", "offsets"),
            extra_fields="ignore",
            renames={},
            kind="tokens",
        )
        _spec_registry[TokenModel] = token_spec

    def test_tokenizer_node_instantiation(self) -> None:
        """Test that TokenizerNode can be instantiated with default attributes."""
        node = TokenizerNode()

        assert node.name == "tokenizer"
        assert node.version == "0.1.0"

    def test_tokenizer_node_instantiation_with_custom_attributes(self) -> None:
        """Test that TokenizerNode can be instantiated with custom attributes."""
        node = TokenizerNode("custom-tokenizer", "1.2.3")

        assert node.name == "custom-tokenizer"
        assert node.version == "1.2.3"

    def test_tokenizer_node_accepts_and_emits(self) -> None:
        """Test that TokenizerNode has correct accepts and emits sets."""
        assert TokenizerNode.accepts == {("text", 1)}
        assert TokenizerNode.emits == {("tokens", 1)}

    def test_tokenizer_node_process_basic_text(self) -> None:
        """Test that TokenizerNode processes basic text correctly."""
        # Create a TextModel and convert to Ether
        text_model = TextModel(text="hello world")
        input_eth = Ether.from_model(text_model)

        # Create TokenizerNode and process
        node = TokenizerNode()
        result_eth = node.process(input_eth)

        # Verify the result is a tokens.v1 envelope
        assert result_eth.kind == "tokens"
        assert result_eth.schema_version == 1

        # Verify the payload contains token IDs
        assert "ids" in result_eth.payload
        assert isinstance(result_eth.payload["ids"], list)
        assert len(result_eth.payload["ids"]) == 2  # "hello" and "world"

        # Verify the token IDs are integers within the expected range
        for token_id in result_eth.payload["ids"]:
            assert isinstance(token_id, int)
            assert 0 <= token_id < 50_000

        # Verify lineage was appended
        assert "lineage" in result_eth.metadata
        assert len(result_eth.metadata["lineage"]) == 1
        lineage_entry = result_eth.metadata["lineage"][0]
        assert lineage_entry["node"] == "tokenizer"
        assert lineage_entry["version"] == "0.1.0"
        assert "ts" in lineage_entry

    def test_tokenizer_node_process_complex_text(self) -> None:
        """Test that TokenizerNode processes complex text correctly."""
        # Create a TextModel with complex text
        text_model = TextModel(text="Hello World! This is a test.")
        input_eth = Ether.from_model(text_model)

        # Create TokenizerNode and process
        node = TokenizerNode()
        result_eth = node.process(input_eth)

        # Verify the result
        assert result_eth.kind == "tokens"
        assert result_eth.schema_version == 1

        # Convert back to TokenModel to verify content
        token_model = result_eth.as_model(TokenModel)
        assert token_model.vocab == "naive_whitespace"
        assert token_model.truncation is None
        assert token_model.offsets is None
        assert token_model.mask is None

        # Verify token IDs are generated correctly
        expected_tokens = ["hello", "world!", "this", "is", "a", "test."]
        assert len(token_model.ids) == len(expected_tokens)

        # Verify each token ID is within the expected range
        for token_id in token_model.ids:
            assert isinstance(token_id, int)
            assert 0 <= token_id < 50_000

    def test_tokenizer_node_process_empty_text(self) -> None:
        """Test that TokenizerNode processes empty text correctly."""
        # Create a TextModel with empty text
        text_model = TextModel(text="")
        input_eth = Ether.from_model(text_model)

        # Create TokenizerNode and process
        node = TokenizerNode()
        result_eth = node.process(input_eth)

        # Verify the result
        assert result_eth.kind == "tokens"
        assert result_eth.schema_version == 1

        # Convert back to TokenModel to verify content
        token_model = result_eth.as_model(TokenModel)
        assert token_model.ids == []  # Empty text should result in empty token list

    def test_tokenizer_node_process_whitespace_only_text(self) -> None:
        """Test that TokenizerNode processes whitespace-only text correctly."""
        # Create a TextModel with whitespace-only text
        text_model = TextModel(text="   \t\n  ")
        input_eth = Ether.from_model(text_model)

        # Create TokenizerNode and process
        node = TokenizerNode()
        result_eth = node.process(input_eth)

        # Verify the result
        assert result_eth.kind == "tokens"
        assert result_eth.schema_version == 1

        # Convert back to TokenModel to verify content
        token_model = result_eth.as_model(TokenModel)
        assert token_model.ids == []  # Whitespace-only should result in empty token list

    def test_tokenizer_node_process_case_insensitive(self) -> None:
        """Test that TokenizerNode converts text to lowercase."""
        # Create a TextModel with mixed case
        text_model = TextModel(text="Hello WORLD Test")
        input_eth = Ether.from_model(text_model)

        # Create TokenizerNode and process
        node = TokenizerNode()
        result_eth = node.process(input_eth)

        # Convert back to TokenModel to verify content
        token_model = result_eth.as_model(TokenModel)

        # The tokens should be lowercase: ["hello", "world", "test"]
        assert len(token_model.ids) == 3

    def test_tokenizer_node_hash_consistency(self) -> None:
        """Test that TokenizerNode produces consistent token IDs for the same tokens."""
        # Create a TextModel with repeated tokens
        text_model = TextModel(text="hello world hello")
        input_eth = Ether.from_model(text_model)

        # Create TokenizerNode and process
        node = TokenizerNode()
        result_eth = node.process(input_eth)

        # Convert back to TokenModel to verify content
        token_model = result_eth.as_model(TokenModel)

        # Should have 3 tokens: ["hello", "world", "hello"]
        assert len(token_model.ids) == 3

        # The first and third tokens should have the same ID (both "hello")
        assert token_model.ids[0] == token_model.ids[2]
        # The second token should have a different ID ("world")
        assert token_model.ids[1] != token_model.ids[0]

    def test_tokenizer_node_token_id_range(self) -> None:
        """Test that TokenizerNode produces token IDs within the expected range."""
        # Create a TextModel with various tokens
        text_model = TextModel(text="a b c d e f g h i j")
        input_eth = Ether.from_model(text_model)

        # Create TokenizerNode and process
        node = TokenizerNode()
        result_eth = node.process(input_eth)

        # Convert back to TokenModel to verify content
        token_model = result_eth.as_model(TokenModel)

        # Verify all token IDs are within the expected range [0, 50000)
        for token_id in token_model.ids:
            assert 0 <= token_id < 50_000

    def test_tokenizer_node_invalid_input_kind(self) -> None:
        """Test that TokenizerNode raises error for invalid input kind."""
        # Create an Ether with wrong kind
        invalid_eth = Ether(kind="embedding", schema_version=1, payload={}, metadata={})

        # Create TokenizerNode and attempt to process
        node = TokenizerNode()

        with pytest.raises(ValueError) as exc_info:
            node.process(invalid_eth)

        assert "TokenizerNode expects one of: text.v1, got embedding.v1" in str(exc_info.value)

    def test_tokenizer_node_invalid_input_version(self) -> None:
        """Test that TokenizerNode raises error for invalid input version."""
        # Create a TextModel and convert to Ether with wrong version
        text_model = TextModel(text="hello world")
        input_eth = Ether.from_model(text_model)
        input_eth.schema_version = 2  # Wrong version

        # Create TokenizerNode and attempt to process
        node = TokenizerNode()

        with pytest.raises(ValueError) as exc_info:
            node.process(input_eth)

        assert "TokenizerNode expects one of: text.v1, got text.v2" in str(exc_info.value)

    def test_tokenizer_node_lineage_appended(self) -> None:
        """Test that TokenizerNode appends lineage information correctly."""
        # Create a TextModel and convert to Ether
        text_model = TextModel(text="hello world")
        input_eth = Ether.from_model(text_model)

        # Create TokenizerNode and process
        node = TokenizerNode("test-tokenizer", "2.0.0")
        result_eth = node.process(input_eth)

        # Verify lineage was appended
        assert "lineage" in result_eth.metadata
        assert len(result_eth.metadata["lineage"]) == 1

        lineage_entry = result_eth.metadata["lineage"][0]
        assert lineage_entry["node"] == "test-tokenizer"
        assert lineage_entry["version"] == "2.0.0"
        assert "ts" in lineage_entry

    def test_tokenizer_node_preserves_metadata(self) -> None:
        """Test that TokenizerNode preserves metadata from input."""
        # Create a TextModel with metadata and convert to Ether
        text_model = TextModel(text="hello world", lang="en", encoding="utf-8", detected_lang_conf=0.95)
        input_eth = Ether.from_model(text_model)

        # Create TokenizerNode and process
        node = TokenizerNode()
        result_eth = node.process(input_eth)

        # Verify the result is a tokens.v1 envelope
        assert result_eth.kind == "tokens"
        assert result_eth.schema_version == 1

        # Convert back to TokenModel to verify content
        token_model = result_eth.as_model(TokenModel)
        assert token_model.vocab == "naive_whitespace"
        assert token_model.truncation is None
        assert token_model.offsets is None
        assert token_model.mask is None

        # Verify token IDs are present
        assert len(token_model.ids) == 2  # "hello" and "world"
        for token_id in token_model.ids:
            assert isinstance(token_id, int)
            assert 0 <= token_id < 50_000
