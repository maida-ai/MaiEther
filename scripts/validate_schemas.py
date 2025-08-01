#!/usr/bin/env python3
"""
Schema validation script for MaiEther.

This script validates all JSON schema files in the schemas/ directory:
1. Ensures each file is valid JSON
2. Validates each file against the JSON Schema meta-schema
3. Checks for required fields in schema files
4. Ensures schema files follow the expected structure

Usage:
    python scripts/validate_schemas.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def load_json_schema(file_path: Path) -> tuple[bool, dict[str, Any], str]:
    """
    Load and validate a JSON schema file.

    Args:
        file_path: Path to the JSON schema file

    Returns:
        Tuple of (is_valid, schema_data, error_message)
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            schema_data = json.load(f)
        return True, schema_data, ""
    except json.JSONDecodeError as e:
        return False, {}, f"Invalid JSON: {e}"
    except Exception as e:
        return False, {}, f"Error reading file: {e}"


def validate_schema_structure(schema_data: dict[str, Any], file_path: Path) -> tuple[bool, str]:
    """
    Validate that the schema follows the expected MaiEther structure.

    Args:
        schema_data: The parsed schema data
        file_path: Path to the schema file for error reporting

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for required top-level fields
    required_fields = ["$schema", "$id", "title", "description", "type", "properties"]
    missing_fields = [field for field in required_fields if field not in schema_data]

    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"

    # Check that it's an object schema
    if schema_data.get("type") != "object":
        return False, "Schema must be of type 'object'"

    # Check for required properties in Ether envelopes
    required_props = ["kind", "schema_version", "payload"]
    schema_props = schema_data.get("properties", {})
    missing_props = [prop for prop in required_props if prop not in schema_props]

    if missing_props:
        return False, f"Missing required Ether properties: {missing_props}"

    # Check that kind is properly defined
    kind_prop = schema_props.get("kind", {})
    if not isinstance(kind_prop, dict) or "enum" not in kind_prop:
        return False, "kind property must have an enum constraint"

    # Check that schema_version is properly defined
    version_prop = schema_props.get("schema_version", {})
    if not isinstance(version_prop, dict) or "enum" not in version_prop:
        return False, "schema_version property must have an enum constraint"

    return True, ""


def validate_meta_schema(schema_data: dict[str, Any], file_path: Path) -> tuple[bool, str]:
    """
    Validate the schema against the JSON Schema meta-schema.

    Args:
        schema_data: The parsed schema data
        file_path: Path to the schema file for error reporting

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Create a validator for the meta-schema
        validator = Draft202012Validator(Draft202012Validator.META_SCHEMA)

        # Validate the schema
        errors = list(validator.iter_errors(schema_data))

        if errors:
            error_messages = []
            for error in errors:
                error_messages.append(f"  - {error.message} at {'/'.join(str(p) for p in error.path)}")
            return False, "Schema validation failed:\n" + "\n".join(error_messages)

        return True, ""
    except Exception as e:
        return False, f"Error validating against meta-schema: {e}"


def find_schema_files(schemas_dir: Path) -> list[Path]:
    """
    Find all JSON schema files in the schemas directory.

    Args:
        schemas_dir: Path to the schemas directory

    Returns:
        List of paths to JSON schema files
    """
    schema_files: list = []

    if not schemas_dir.exists():
        return schema_files

    for json_file in schemas_dir.rglob("*.json"):
        schema_files.append(json_file)

    return sorted(schema_files)


def main() -> int:
    """
    Main validation function.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    start_time = time.time()

    # Find the schemas directory
    project_root = Path(__file__).parent.parent
    schemas_dir = project_root / "schemas"

    if not schemas_dir.exists():
        print(f"❌ Schemas directory not found: {schemas_dir}")
        return 1

    # Find all schema files
    schema_files = find_schema_files(schemas_dir)

    if not schema_files:
        print(f"❌ No JSON schema files found in {schemas_dir}")
        return 1

    print(f"🔍 Found {len(schema_files)} schema files to validate...")

    # Validate each schema file
    errors = []
    valid_count = 0

    for schema_file in schema_files:
        print(f"  Validating {schema_file.relative_to(project_root)}...")

        # Load and parse JSON
        is_valid_json, schema_data, json_error = load_json_schema(schema_file)
        if not is_valid_json:
            errors.append(f"{schema_file.relative_to(project_root)}: {json_error}")
            continue

        # Validate schema structure
        is_valid_structure, structure_error = validate_schema_structure(schema_data, schema_file)
        if not is_valid_structure:
            errors.append(f"{schema_file.relative_to(project_root)}: {structure_error}")
            continue

        # Validate against meta-schema
        is_valid_meta, meta_error = validate_meta_schema(schema_data, schema_file)
        if not is_valid_meta:
            errors.append(f"{schema_file.relative_to(project_root)}: {meta_error}")
            continue

        valid_count += 1
        print("    ✅ Valid")

    # Report results
    elapsed_time = time.time() - start_time

    if errors:
        print(f"\n❌ Validation failed! Found {len(errors)} error(s):")
        for error in errors:
            print(f"  {error}")
        print(f"\n⏱️  Validation completed in {elapsed_time:.2f}s")
        return 1
    else:
        print(f"\n✅ All {valid_count} schema files are valid!")
        print(f"⏱️  Validation completed in {elapsed_time:.2f}s")
        return 0


if __name__ == "__main__":
    sys.exit(main())
