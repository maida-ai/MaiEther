"""Test tokens schema validation."""

import json

import jsonschema
import pytest

from tests.kinds import SCHEMAS_DIR


def test_tokens_schema_validates_correctly():
    """Test that the tokens schema validates correctly."""
    schema_path = SCHEMAS_DIR / "tokens" / "v1.json"
    assert schema_path.exists(), "Tokens schema file should exist"

    with open(schema_path) as f:
        schema = json.load(f)

    # Test valid document
    valid_doc = {
        "kind": "tokens",
        "schema_version": 1,
        "payload": {"ids": [1, 2, 3, 4, 5]},
        "metadata": {"vocab": "gpt2"},
    }

    jsonschema.validate(valid_doc, schema)

    # Test with optional fields
    valid_doc_with_optional = {
        "kind": "tokens",
        "schema_version": 1,
        "payload": {"ids": [1, 2, 3], "mask": [1, 1, 0]},
        "metadata": {"vocab": "bert-base-uncased", "truncation": "longest_first", "offsets": True},
    }

    jsonschema.validate(valid_doc_with_optional, schema)


def test_tokens_schema_requires_payload_ids():
    """Test that payload.ids is required."""
    schema_path = SCHEMAS_DIR / "tokens" / "v1.json"

    with open(schema_path) as f:
        schema = json.load(f)

    invalid_doc = {"kind": "tokens", "schema_version": 1, "payload": {}, "metadata": {"vocab": "gpt2"}}

    with pytest.raises(jsonschema.exceptions.ValidationError) as exc_info:
        jsonschema.validate(invalid_doc, schema)

    assert "'ids' is a required property" in str(exc_info.value)


def test_tokens_schema_requires_metadata_vocab():
    """Test that metadata.vocab is required."""
    schema_path = SCHEMAS_DIR / "tokens" / "v1.json"

    with open(schema_path) as f:
        schema = json.load(f)

    invalid_doc = {"kind": "tokens", "schema_version": 1, "payload": {"ids": [1, 2, 3]}, "metadata": {}}

    with pytest.raises(jsonschema.exceptions.ValidationError) as exc_info:
        jsonschema.validate(invalid_doc, schema)

    assert "'vocab' is a required property" in str(exc_info.value)


def test_tokens_schema_validates_ids_array():
    """Test that payload.ids must be an array of integers."""
    schema_path = SCHEMAS_DIR / "tokens" / "v1.json"

    with open(schema_path) as f:
        schema = json.load(f)

    # Test with non-array
    invalid_doc = {
        "kind": "tokens",
        "schema_version": 1,
        "payload": {"ids": "not_an_array"},
        "metadata": {"vocab": "gpt2"},
    }

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(invalid_doc, schema)

    # Test with array of non-integers
    invalid_doc = {
        "kind": "tokens",
        "schema_version": 1,
        "payload": {"ids": ["not", "integers"]},
        "metadata": {"vocab": "gpt2"},
    }

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(invalid_doc, schema)


def test_tokens_schema_validates_vocab_string():
    """Test that metadata.vocab must be a string."""
    schema_path = SCHEMAS_DIR / "tokens" / "v1.json"

    with open(schema_path) as f:
        schema = json.load(f)

    invalid_doc = {
        "kind": "tokens",
        "schema_version": 1,
        "payload": {"ids": [1, 2, 3]},
        "metadata": {"vocab": 123},  # Should be string
    }

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(invalid_doc, schema)
