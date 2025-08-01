# Getting Started with MaiEther

MaiEther provides a standardized envelope for safe data transport between nodes/layers in composable ML/data systems. This guide will help you get started with using Ether and creating new bindings.

## Quick Start

### Basic Usage

```python
from ether import Ether, Attachment
from pydantic import BaseModel

# Register a model for conversion
@Ether.register(
    payload=["embedding"],
    metadata=["source", "dim"],
    extra_fields="keep",
    renames={"embedding": "vec.values"},
    kind="embedding",
)
class EmbeddingModel(BaseModel):
    embedding: list[float]
    source: str
    dim: int
    note: str = "extra"  # carried in extra_fields

# Convert model to Ether
model = EmbeddingModel(embedding=[1.0, 2.0, 3.0], source="bert", dim=3)
ether = Ether.from_model(model)

# Convert back to model
converted_model = ether.as_model(EmbeddingModel)
```

### Working with Attachments

For large data like tensors, use attachments for zero-copy transport:

```python
import numpy as np

# Create an embedding as an attachment
arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
att = Attachment(
    id="emb-0",
    uri="shm://embeddings/12345",  # or use inline_bytes=arr.tobytes()
    media_type="application/x-raw-tensor",
    codec="RAW_F32",
    shape=[arr.size],
    dtype="float32",
    size_bytes=int(arr.nbytes),
    checksum={"algo": "crc32c", "value": "..."}
)

eth = Ether(
    kind="embedding",
    schema_version=1,
    payload={"values": None, "dim": int(arr.size)},  # None when using attachment
    metadata={"source": "m1"},
    attachments=[att],
    extra_fields={}
)
```

## Canonical Kinds

MaiEther defines several canonical data types for common ML workloads:

### Text (`text.v1`)

For transporting text content with language metadata:

```python
@Ether.register(
    payload=["text"],
    metadata=["lang", "encoding"],
    kind="text",
)
class TextModel(BaseModel):
    text: str
    lang: str | None = None
    encoding: str | None = None

# Usage
text_model = TextModel(text="Hello, world!", lang="en")
ether = Ether.from_model(text_model)
```

**Schema**: [text/v1.json](../schemas/text/v1.json)

### Tokens (`tokens.v1`)

For transporting tokenized data with vocabulary information:

```python
@Ether.register(
    payload=["ids", "mask"],
    metadata=["vocab"],
    kind="tokens",
)
class TokensModel(BaseModel):
    ids: list[int]
    mask: list[int] | None = None
    vocab: str

# Usage
tokens_model = TokensModel(ids=[101, 2023, 102], vocab="bert-base-uncased")
ether = Ether.from_model(tokens_model)
```

**Schema**: [tokens/v1.json](../schemas/tokens/v1.json)

### Embeddings (`embedding.v1`)

For transporting embedding vectors with metadata:

```python
@Ether.register(
    payload=["values", "dim"],
    metadata=["source", "norm"],
    kind="embedding",
)
class EmbeddingModel(BaseModel):
    values: list[float] | None = None  # None when using attachments
    dim: int
    source: str
    norm: float | None = None

# Usage
embedding_model = EmbeddingModel(
    values=[0.1, 0.2, 0.3],
    dim=3,
    source="bert-base",
    norm=0.374
)
ether = Ether.from_model(embedding_model)
```

**Schema**: [embedding/v1.json](../schemas/embedding/v1.json)

## Creating New Bindings

### Step 1: Define Your Model

Create a Pydantic model that represents your data:

```python
from pydantic import BaseModel

class MyCustomModel(BaseModel):
    data: list[float]
    metadata: str
    timestamp: str
```

### Step 2: Register with Ether

Use the `@Ether.register` decorator to define how your model maps to Ether:

```python
@Ether.register(
    payload=["data"],
    metadata=["metadata", "timestamp"],
    extra_fields="ignore",
    kind="my_custom",
)
class MyCustomModel(BaseModel):
    data: list[float]
    metadata: str
    timestamp: str
```

### Step 3: Create Schema Definition

Create a JSON schema file in the `schemas/` directory:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://maida.ai/schemas/my_custom/v1.json",
  "title": "My Custom v1 Schema",
  "description": "Schema for my custom data transport in MaiEther",
  "type": "object",
  "required": ["kind", "schema_version", "payload"],
  "properties": {
    "kind": {
      "type": "string",
      "enum": ["my_custom"],
      "description": "Logical type identifier for my custom data"
    },
    "schema_version": {
      "type": "integer",
      "enum": [1],
      "description": "Schema version number"
    },
    "payload": {
      "type": "object",
      "required": ["data"],
      "properties": {
        "data": {
          "type": "array",
          "items": {"type": "number"},
          "description": "Custom data array"
        }
      },
      "additionalProperties": false
    },
    "metadata": {
      "type": "object",
      "properties": {
        "metadata": {
          "type": "string",
          "description": "Custom metadata"
        },
        "timestamp": {
          "type": "string",
          "description": "ISO timestamp"
        }
      },
      "additionalProperties": true
    }
  },
  "additionalProperties": false
}
```

### Step 4: Add Tests

Create comprehensive tests for your binding:

```python
import pytest
from ether import Ether
from your_module import MyCustomModel

def test_my_custom_model_registration():
    """Test that MyCustomModel is properly registered with Ether."""
    model = MyCustomModel(
        data=[1.0, 2.0, 3.0],
        metadata="test",
        timestamp="2023-01-01T00:00:00Z"
    )

    # Convert to Ether
    ether = Ether.from_model(model)
    assert ether.kind == "my_custom"
    assert ether.schema_version == 1
    assert ether.payload["data"] == [1.0, 2.0, 3.0]
    assert ether.metadata["metadata"] == "test"
    assert ether.metadata["timestamp"] == "2023-01-01T00:00:00Z"

    # Convert back to model
    converted = ether.as_model(MyCustomModel)
    assert converted.data == [1.0, 2.0, 3.0]
    assert converted.metadata == "test"
    assert converted.timestamp == "2023-01-01T00:00:00Z"
```

## Advanced Features

### Adapters

Create adapters for complex model conversions:

```python
@Ether.adapter(EmbeddingModel, TextModel)
def embedding_to_text(eth: Ether) -> dict:
    """Convert embedding to text representation."""
    values = eth.payload.get("values", [])
    return {
        "text": f"Embedding with {len(values)} dimensions",
        "lang": "en",
        "encoding": "utf-8"
    }

# Usage
embedding_model = EmbeddingModel(values=[1.0, 2.0], dim=2, source="test")
ether = Ether.from_model(embedding_model)
text_model = ether.as_model(TextModel)  # Uses adapter
```

### Field Renaming

Use the `renames` parameter to map model fields to nested Ether paths:

```python
@Ether.register(
    payload=["data"],
    metadata=["info"],
    renames={
        "data": "nested.data",  # Maps to payload.nested.data
        "info": "nested.info",  # Maps to metadata.nested.info
    },
    kind="nested_example",
)
class NestedModel(BaseModel):
    data: list[float]
    info: str
```

### Extra Fields Handling

Control how unmapped fields are handled:

```python
# Ignore extra fields (default)
@Ether.register(payload=["data"], metadata=[], extra_fields="ignore")

# Keep extra fields in extra_fields
@Ether.register(payload=["data"], metadata=[], extra_fields="keep")

# Error on extra fields
@Ether.register(payload=["data"], metadata=[], extra_fields="error")
```

## Best Practices

### Performance

1. **Use attachments for large data**: Avoid putting large arrays in payload
2. **Zero-copy when possible**: Use shared memory URIs for attachments
3. **Validate at boundaries**: Only validate at ingress/egress points

### Error Handling

```python
from ether import ConversionError, RegistrationError

try:
    ether = Ether.from_model(unregistered_model)
except RegistrationError as e:
    print(f"Model not registered: {e}")

try:
    model = ether.as_model(TargetModel)
except ConversionError as e:
    print(f"Conversion failed: {e}")
```

### Tracing and Observability

```python
# Add tracing metadata
ether = Ether(
    kind="embedding",
    schema_version=1,
    payload={"dim": 768},
    metadata={
        "trace_id": "123e4567-e89b-12d3-a456-426614174000",
        "span_id": "span-123",
        "created_at": "2023-01-01T00:00:00Z",
        "producer": "my-node",
        "lineage": [
            {"node": "tokenizer", "version": "1.0", "ts": "2023-01-01T00:00:00Z"}
        ]
    }
)

# Get summary for logging
summary = ether.summary()
print(f"Ether summary: {summary}")
```

**Important Note on Timestamps**: The `created_at` and lineage timestamps are **advisory** and should not be used for causal ordering, especially when envelopes travel across hosts. Clock skew between machines can cause timestamps to appear out of order even when events occurred in the correct sequence. For causal ordering, use the `trace_id` and `span_id` fields for distributed tracing instead.

## Schema Registry

MaiEther maintains a schema registry for validation and documentation. Each canonical kind has:

1. **JSON Schema**: Formal validation rules
2. **Pydantic Model**: Type-safe Python interface
3. **Documentation**: Usage examples and field descriptions

### Available Schemas

- [Text v1](../schemas/text/v1.json) - Text data transport
- [Tokens v1](../schemas/tokens/v1.json) - Tokenized data transport
- [Embedding v1](../schemas/embedding/v1.json) - Embedding vector transport

## Next Steps

1. **Explore the codebase**: Check out the `ether/` module for implementation details
2. **Run tests**: Use `poetry run pytest` to run the test suite
3. **Check coverage**: Use `poetry run pytest --cov=ether` for coverage reports
4. **Read the spec**: See the main README for the complete specification
5. **Contribute**: Add new canonical kinds or improve existing ones

For more information, see the [main README](../README.md) for the complete specification and architecture details.
