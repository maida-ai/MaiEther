"""Tests for JSON Schema validation."""

import json

from jsonschema import Draft202012Validator

from tests.kinds import SCHEMAS_DIR


class TestTextV1Schema:
    """Test the text.v1 JSON schema."""

    def setup_method(self) -> None:
        """Load the text.v1 schema."""
        schema_path = SCHEMAS_DIR / "text" / "v1.json"
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

    def test_valid_text_v1_document(self) -> None:
        """Test that a valid text.v1 document passes validation."""
        valid_doc = {
            "kind": "text",
            "schema_version": 1,
            "payload": {"text": "Hello, world!"},
            "metadata": {"lang": "en"},
            "extra_fields": {},
            "attachments": [],
        }

        errors = list(self.validator.iter_errors(valid_doc))
        assert not errors, f"Validation errors: {errors}"

    def test_required_keys_present(self) -> None:
        """Test that required keys are enforced."""
        # Test missing kind
        doc_without_kind = {"schema_version": 1, "payload": {"text": "Hello"}}
        errors = list(self.validator.iter_errors(doc_without_kind))
        assert any("kind" in str(error) for error in errors)

        # Test missing schema_version
        doc_without_version = {"kind": "text", "payload": {"text": "Hello"}}
        errors = list(self.validator.iter_errors(doc_without_version))
        assert any("schema_version" in str(error) for error in errors)

        # Test missing payload
        doc_without_payload = {"kind": "text", "schema_version": 1}
        errors = list(self.validator.iter_errors(doc_without_payload))
        assert any("payload" in str(error) for error in errors)

        # Test missing payload.text
        doc_without_text = {"kind": "text", "schema_version": 1, "payload": {}}
        errors = list(self.validator.iter_errors(doc_without_text))
        assert any("text" in str(error) for error in errors)

    def test_kind_must_be_text(self) -> None:
        """Test that kind must be 'text'."""
        doc_with_wrong_kind = {"kind": "embedding", "schema_version": 1, "payload": {"text": "Hello"}}
        errors = list(self.validator.iter_errors(doc_with_wrong_kind))
        assert any("kind" in str(error) for error in errors)

    def test_schema_version_must_be_1(self) -> None:
        """Test that schema_version must be 1."""
        doc_with_wrong_version = {"kind": "text", "schema_version": 2, "payload": {"text": "Hello"}}
        errors = list(self.validator.iter_errors(doc_with_wrong_version))
        assert any("schema_version" in str(error) for error in errors)

    def test_payload_text_must_be_string(self) -> None:
        """Test that payload.text must be a string."""
        doc_with_non_string_text = {"kind": "text", "schema_version": 1, "payload": {"text": 123}}
        errors = list(self.validator.iter_errors(doc_with_non_string_text))
        assert any("text" in str(error) for error in errors)

    def test_optional_metadata_lang(self) -> None:
        """Test that metadata.lang is optional and must be string."""
        # Test with valid lang
        doc_with_lang = {"kind": "text", "schema_version": 1, "payload": {"text": "Hello"}, "metadata": {"lang": "en"}}
        errors = list(self.validator.iter_errors(doc_with_lang))
        assert not errors

        # Test with non-string lang
        doc_with_invalid_lang = {
            "kind": "text",
            "schema_version": 1,
            "payload": {"text": "Hello"},
            "metadata": {"lang": 123},
        }
        errors = list(self.validator.iter_errors(doc_with_invalid_lang))
        assert any("lang" in str(error) for error in errors)

    def test_metadata_encoding_optional(self) -> None:
        """Test that metadata.encoding is optional."""
        doc_with_encoding = {
            "kind": "text",
            "schema_version": 1,
            "payload": {"text": "Hello"},
            "metadata": {"encoding": "utf-8"},
        }
        errors = list(self.validator.iter_errors(doc_with_encoding))
        assert not errors

    def test_metadata_detected_lang_conf_range(self) -> None:
        """Test that metadata.detected_lang_conf must be between 0.0 and 1.0."""
        # Test valid confidence
        doc_with_valid_conf = {
            "kind": "text",
            "schema_version": 1,
            "payload": {"text": "Hello"},
            "metadata": {"detected_lang_conf": 0.8},
        }
        errors = list(self.validator.iter_errors(doc_with_valid_conf))
        assert not errors

        # Test invalid confidence (too high)
        doc_with_high_conf = {
            "kind": "text",
            "schema_version": 1,
            "payload": {"text": "Hello"},
            "metadata": {"detected_lang_conf": 1.5},
        }
        errors = list(self.validator.iter_errors(doc_with_high_conf))
        assert any("detected_lang_conf" in str(error) for error in errors)

        # Test invalid confidence (negative)
        doc_with_negative_conf = {
            "kind": "text",
            "schema_version": 1,
            "payload": {"text": "Hello"},
            "metadata": {"detected_lang_conf": -0.1},
        }
        errors = list(self.validator.iter_errors(doc_with_negative_conf))
        assert any("detected_lang_conf" in str(error) for error in errors)

    def test_attachments_structure(self) -> None:
        """Test that attachments have the correct structure."""
        doc_with_attachments = {
            "kind": "text",
            "schema_version": 1,
            "payload": {"text": "Hello"},
            "attachments": [{"id": "att-1", "media_type": "application/octet-stream", "size_bytes": 1024}],
        }
        errors = list(self.validator.iter_errors(doc_with_attachments))
        assert not errors

        # Test missing required attachment fields
        doc_with_invalid_attachment = {
            "kind": "text",
            "schema_version": 1,
            "payload": {"text": "Hello"},
            "attachments": [
                {
                    "id": "att-1"
                    # Missing media_type
                }
            ],
        }
        errors = list(self.validator.iter_errors(doc_with_invalid_attachment))
        assert any("media_type" in str(error) for error in errors)
