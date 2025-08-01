"""Tests for JSON Schema validation."""

import json

from jsonschema import Draft202012Validator

from tests.kinds import SCHEMAS_DIR


class TestEmbeddingV1Schema:
    """Test the embedding.v1 JSON schema."""

    def setup_method(self) -> None:
        """Load the embedding.v1 schema."""
        schema_path = SCHEMAS_DIR / "embedding" / "v1.json"
        with open(schema_path) as f:
            self.schema = json.load(f)
        self.validator = Draft202012Validator(self.schema)

    def test_schema_validates_against_draft_2020_12(self) -> None:
        """Test that the schema itself validates against JSON-Schema draft 2020-12."""
        # Create a validator for the meta-schema
        meta_validator = Draft202012Validator(Draft202012Validator.META_SCHEMA)

        # Validate our schema against the meta-schema
        errors = list(meta_validator.iter_errors(self.schema))
        assert not errors, f"Schema validation errors: {errors}"

    def test_valid_embedding_v1_document(self) -> None:
        """Test that a valid embedding.v1 document passes validation."""
        valid_doc = {
            "kind": "embedding",
            "schema_version": 1,
            "payload": {"dim": 768},
            "metadata": {"source": "bert-base-uncased"},
            "extra_fields": {},
            "attachments": [],
        }

        errors = list(self.validator.iter_errors(valid_doc))
        assert not errors, f"Validation errors: {errors}"

    def test_required_keys_present(self) -> None:
        """Test that required keys are enforced."""
        # Test missing kind
        doc_without_kind = {"schema_version": 1, "payload": {"dim": 768}}
        errors = list(self.validator.iter_errors(doc_without_kind))
        assert any("kind" in str(error) for error in errors)

        # Test missing schema_version
        doc_without_version = {"kind": "embedding", "payload": {"dim": 768}}
        errors = list(self.validator.iter_errors(doc_without_version))
        assert any("schema_version" in str(error) for error in errors)

        # Test missing payload
        doc_without_payload = {"kind": "embedding", "schema_version": 1}
        errors = list(self.validator.iter_errors(doc_without_payload))
        assert any("payload" in str(error) for error in errors)

        # Test missing payload.dim
        doc_without_dim = {"kind": "embedding", "schema_version": 1, "payload": {}}
        errors = list(self.validator.iter_errors(doc_without_dim))
        assert any("dim" in str(error) for error in errors)

    def test_kind_must_be_embedding(self) -> None:
        """Test that kind must be 'embedding'."""
        doc_with_wrong_kind = {"kind": "text", "schema_version": 1, "payload": {"dim": 768}}
        errors = list(self.validator.iter_errors(doc_with_wrong_kind))
        assert any("kind" in str(error) for error in errors)

    def test_schema_version_must_be_1(self) -> None:
        """Test that schema_version must be 1."""
        doc_with_wrong_version = {"kind": "embedding", "schema_version": 2, "payload": {"dim": 768}}
        errors = list(self.validator.iter_errors(doc_with_wrong_version))
        assert any("schema_version" in str(error) for error in errors)

    def test_payload_dim_must_be_integer(self) -> None:
        """Test that payload.dim must be an integer."""
        doc_with_non_integer_dim = {"kind": "embedding", "schema_version": 1, "payload": {"dim": "768"}}
        errors = list(self.validator.iter_errors(doc_with_non_integer_dim))
        assert any("dim" in str(error) for error in errors)

    def test_payload_dim_must_be_positive(self) -> None:
        """Test that payload.dim must be at least 1."""
        # Test with zero
        doc_with_zero_dim = {"kind": "embedding", "schema_version": 1, "payload": {"dim": 0}}
        errors = list(self.validator.iter_errors(doc_with_zero_dim))
        assert any("dim" in str(error) for error in errors)

        # Test with negative
        doc_with_negative_dim = {"kind": "embedding", "schema_version": 1, "payload": {"dim": -1}}
        errors = list(self.validator.iter_errors(doc_with_negative_dim))
        assert any("dim" in str(error) for error in errors)

    def test_optional_metadata_source(self) -> None:
        """Test that metadata.source is optional and must be string."""
        # Test with valid source
        doc_with_source = {
            "kind": "embedding",
            "schema_version": 1,
            "payload": {"dim": 768},
            "metadata": {"source": "bert-base-uncased"},
        }
        errors = list(self.validator.iter_errors(doc_with_source))
        assert not errors

        # Test with non-string source
        doc_with_invalid_source = {
            "kind": "embedding",
            "schema_version": 1,
            "payload": {"dim": 768},
            "metadata": {"source": 123},
        }
        errors = list(self.validator.iter_errors(doc_with_invalid_source))
        assert any("source" in str(error) for error in errors)

        # Test without source (should be valid)
        doc_without_source = {
            "kind": "embedding",
            "schema_version": 1,
            "payload": {"dim": 768},
            "metadata": {},
        }
        errors = list(self.validator.iter_errors(doc_without_source))
        assert not errors

    def test_attachments_structure(self) -> None:
        """Test that attachments have the correct structure."""
        doc_with_attachments = {
            "kind": "embedding",
            "schema_version": 1,
            "payload": {"dim": 768},
            "attachments": [{"id": "att-1", "media_type": "application/octet-stream", "size_bytes": 1024}],
        }
        errors = list(self.validator.iter_errors(doc_with_attachments))
        assert not errors

        # Test missing required attachment fields
        doc_with_invalid_attachment = {
            "kind": "embedding",
            "schema_version": 1,
            "payload": {"dim": 768},
            "attachments": [
                {
                    "id": "att-1"
                    # Missing media_type
                }
            ],
        }
        errors = list(self.validator.iter_errors(doc_with_invalid_attachment))
        assert any("media_type" in str(error) for error in errors)

    def test_extra_fields_optional(self) -> None:
        """Test that extra_fields is optional."""
        doc_with_extra_fields = {
            "kind": "embedding",
            "schema_version": 1,
            "payload": {"dim": 768},
            "extra_fields": {"custom_field": "value"},
        }
        errors = list(self.validator.iter_errors(doc_with_extra_fields))
        assert not errors

        # Test without extra_fields (should be valid)
        doc_without_extra_fields = {
            "kind": "embedding",
            "schema_version": 1,
            "payload": {"dim": 768},
        }
        errors = list(self.validator.iter_errors(doc_without_extra_fields))
        assert not errors
