# MaiEther Schemas

This directory contains JSON Schema definitions for MaiEther data types.

## Available Schemas

### Text
- [`text/v1.json`](text/v1.json) - Text data transport schema (v1)

### Tokens
- [`tokens/v1.json`](tokens/v1.json) - Tokenized data transport schema (v1)

### Embeddings
- [`embedding/v1.json`](embedding/v1.json) - Embedding vector transport schema (v1)

## Schema Validation

All schemas in this directory:
- Use JSON Schema Draft 2020-12
- Include self-validation against the meta-schema
- Follow MaiEther envelope structure with `kind`, `schema_version`, `payload`, `metadata`, `extra_fields`, and `attachments`

## Usage

Schemas can be used for:
- Runtime validation of Ether envelopes
- Documentation and API specification
- Cross-language client generation
- Schema registry integration
