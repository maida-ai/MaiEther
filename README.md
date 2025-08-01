# MaiEther

[![codecov](https://codecov.io/github/maida-ai/MaiEther/graph/badge.svg)](https://codecov.io/github/maida-ai/MaiEther)
[![unittests](https://github.com/maida-ai/MaiEther/actions/workflows/tests.yml/badge.svg)](https://github.com/maida-ai/MaiEther/actions/workflows/tests.yml)
[![documentation](https://github.com/maida-ai/MaiEther/actions/workflows/sphinx-docs.yml/badge.svg)](https://github.com/maida-ai/MaiEther/actions/workflows/sphinx-docs.yml)


**Status:** Draft for implementation
**Audience:** Software engineers building a composable ML/data system with swappable nodes (local + cross-process + cross-server).
**Scope:** Defines the *intermediate representation (IR)* and a Python reference implementation (Pydantic v2) for an envelope called **Ether** that safely transports data between nodes/layers. Integrates with adapters and a schema registry; can be carried over in-memory calls, multiprocess queues, or a binary transport (e.g., **XCP**).

## Glossary

| Term | Definition |
| --- | --- |
| Zero-Copy | Data transfer between memory spaces without requiring the CPU to copy the data. |


## 1. Goals & Non-Goals

### Goals
- A **single envelope type** that nodes exchange internally, regardless of the specific model/type used at the edges.
- **Schema-aware**: every envelope declares a `(kind, version)` and validates/adapter-converts at boundaries.
- **Robust conversions** between registered Pydantic models and envelopes (field selection, renaming, nested keys).
- **Cross-process efficiency** with **attachments** for large tensors/tables (zero-copy where possible).
- **Evolvable** via versioning rules and a minimal **schema registry**.
- **Operable**: tracing, provenance, and good error diagnostics.

### Non-Goals
- Not a public API for third parties (though it can evolve into one).
- Not a replacement for a wire protocol. For cross-host transport, pair Ether with a framing/transport (e.g., **XCP**, gRPC, or Arrow Flight).


## 2. Conceptual Model

- An **Ether** is an **envelope** with:
  - **kind** (e.g., `"embedding"`, `"tokens"`, `"image"`)
  - **schema_version** (integer, e.g., `1`)
  - **payload** (structured content relevant to the kind)
  - **metadata** (context/provenance/parameters)
  - **extra_fields** (carry-through for unclassified fields)
  - **attachments** (binary or external buffers: Arrow IPC, DLPack, file/URI, shared memory, etc.)
- Nodes declare what they **accept** and **emit** as `(kind, version range)`.
- At model boundaries, a **registration** declares which model fields map to payload/metadata (with **renames** and **nested** key paths).
- **Adapters** handle non-trivial conversions between models or between Ether kinds/versions.


## 3. IR (Intermediate Representation) Specification

### 3.1 Envelope Schema (language-agnostic)

```json
{
  "kind": "string",            // REQUIRED: logical type (e.g., "embedding", "tokens")
  "schema_version": 1,         // REQUIRED: integer >= 1
  "payload": {                 // REQUIRED: object; semantics depend on (kind,version)
    "...": "..."
  },
  "metadata": {                // REQUIRED: object; free-form, but see reserved keys below
    "trace_id": "uuid-v4?",    // OPTIONAL
    "span_id": "uuid-v4? or short id",
    "created_at": "RFC3339 timestamp",
    "producer": "string",
    "lineage": [ { "node": "string", "version": "string", "ts": "RFC3339" } ],
    "schema_namespace": "string?", // OPTIONAL: for cross-org schemas
    "...": "..."
  },
  "extra_fields": {            // OPTIONAL: fields not explicitly mapped; defaults to {}
    "...": "..."
  },
  "attachments": [             // OPTIONAL: zero or more
    {
      "id": "string",          // REQUIRED: unique per envelope
      "uri": "string?",        // OPTIONAL: e.g., shm://..., file://..., s3://..., flight://...
      "inline_bytes": "base64?", // OPTIONAL: if small enough; mutually exclusive with uri
      "media_type": "string",  // REQUIRED: e.g., application/vnd.arrow.ipc, application/x-dlpack
      "codec": "string?",      // OPTIONAL: e.g., ARROW_IPC, DLPACK, NPZ, RAW_F32
      "shape": [ "int", ... ], // OPTIONAL: for tensors
      "dtype": "string?",      // OPTIONAL: e.g., float32, int8, uint8, bfloat16
      "byte_order": "LE|BE?",  // OPTIONAL (default LE)
      "device": "string?",     // OPTIONAL: cpu, cuda:0, mps, etc.
      "size_bytes": "int?",
      "compression": { "name": "zstd|lz4|none", "level": "int?" },
      "checksum": { "algo": "crc32c|sha256", "value": "hex" },
      "metadata": { "...": "..." }  // OPTIONAL: attachment-local metadata
    }
  ]
}
```

**Reserved metadata keys** (non-breaking to add more later):

* `trace_id`, `span_id`, `created_at`, `producer`, `lineage`
* `schema_namespace` (optional string to disambiguate `kind` across silos)

**Versioning:**

* `schema_version` follows **semantic evolution**:

  * **Minor** changes (additive/optional fields) SHOULD NOT bump major; Ether uses a single `schema_version` integer--treat increments as **compatible** unless flagged otherwise in registry.
  * **Breaking** changes bump the integer and adapters handle upgrades/downgrades.

### 3.2 Canonical "kinds" (v1)

Focus on the common IRs to start:

#### `embedding.v1` (implemented)

<!--
* **payload**:
  * `values`: `list[float]` *or* an **attachment** (preferred for large vectors)
  * `dim`: `int` (MUST = len(values) if values present)
* **metadata** (recommended):
  * `source`: `str` (e.g., model name/version)
  * `norm`: `float?` (L2 norm)
  * `quantized`: `bool?` (if INT8/other codec used)
  * `dtype`: `str?` (if `values` omitted, dtype must be in attachments)
  * `codec`: `RAW_F32|RAW_F16|INT8|DLPACK|ARROW_IPC`? (only if attachements is provided)
* **attachments** (optional, recommended for large):
  * Single attachment with `codec` = `RAW_F32|RAW_F16|INT8|DLPACK|ARROW_IPC` and tensor descriptors.
-->
- **Purpose**: Transport embedding vectors with metadata
- **Payload**: `values` (list[float] or null), `dim` (int)
- **Metadata**: `source`, `norm`, `quantized`, `dtype`, `codec`
- **Attachments**: Single attachment with tensor data (preferred for large vectors)
- **Schema**: [embedding/v1.json](schemas/embedding/v1.json)

#### `tokens.v1` (implemented)

<!--
* **payload**:
  * `ids`: `list[int]`
  * `mask`: `list[int]?` (same length as `ids`)
* **metadata**:
  * `vocab`: `str` (vocabulary/model)
  * `truncation`: `str?`
  * `offsets`: `bool?`
-->
- **Purpose**: Transport tokenized data with vocabulary information
- **Payload**: `ids` (list[int]), `mask` (list[int]?)
- **Metadata**: `vocab` (required), `truncation`, `offsets`
- **Schema**: [tokens/v1.json](schemas/tokens/v1.json)

#### `text.v1` (implemented)

<!--
* **payload**:
  * `text`: `str`
* **metadata**:
  * `lang`: `str?`
  * `encoding`: `str?`
  * `detected_lang_conf`: `float?`
-->
- **Purpose**: Transport text content with language metadata
- **Payload**: `text` (string)
- **Metadata**: `lang`, `encoding`, `detected_lang_conf`
- **Schema**: [text/v1.json](schemas/text/v1.json)

#### `image.v1`

* **attachments**:
  * `data`: `buffer` / URI
* **metadata**:
  * `size`: `[int, int]` (H / W)
  * `mode`: `RGB|L|...`?

#### `logits.v1`

* **payload**:
  * `values`: `list[float]`  (or numpy array)
  * `shape`: `list[int]`
* **metadata**:
  * `task`: str

#### `table.v1`
* **attachments**:
  * `arrow`: `<Arrow buffer>`/URI
* **metadata**: `{...}`


## 4. Python Reference Implementation (Pydantic v2)

### 4.1 Core classes

**TODO**: At some point we should rewrite these in some compiled language

```python
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Callable
from dataclasses import dataclass
from pydantic import BaseModel, Field, PrivateAttr, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)

# ---------- Attachments ----------

class Attachment(BaseModel):
    id: str
    uri: Optional[str] = None
    inline_bytes: Optional[bytes] = None  # base64 serialized when transported as JSON
    media_type: str                       # e.g., application/vnd.arrow.ipc
    codec: Optional[str] = None           # e.g., ARROW_IPC, DLPACK, RAW_F32
    shape: Optional[list[int]] = None
    dtype: Optional[str] = None
    byte_order: Optional[str] = "LE"
    device: Optional[str] = None
    size_bytes: Optional[int] = None
    compression: Optional[dict] = None    # {"name": "zstd", "level": 3}
    checksum: Optional[dict] = None       # {"algo": "crc32c", "value": "..."}
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ---------- Registration/Spec ----------

@dataclass(frozen=True)
class EtherSpec:
    payload_fields: Tuple[str, ...]
    metadata_fields: Tuple[str, ...]
    extra_fields: str = "ignore"         # "ignore" | "keep" | "error"
    renames: Mapping[str, str] = ...     # model_field -> ether dot path
    kind: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "renames", dict(self.renames or {}))
        dup = set(self.payload_fields) & set(self.metadata_fields)
        if dup:
            raise RuntimeError(f"Fields in both payload & metadata: {sorted(dup)}")
        # No two fields map to same ether path
        used = {}
        for mf, path in self.renames.items():
            if path in used:
                raise RuntimeError(f"Duplicate mapping for ether path '{path}'")
            used[path] = mf

class ConversionError(RuntimeError): ...
class RegistrationError(RuntimeError): ...

# ---------- Ether Envelope ----------

class Ether(BaseModel):
    kind: str
    schema_version: int = 1

    payload: Dict[str, Any]
    metadata: Dict[str, Any]
    extra_fields: Dict[str, Any] = Field(default_factory=dict)
    attachments: list[Attachment] = Field(default_factory=list)

    _source_model: Optional[Type[BaseModel]] = PrivateAttr(default=None)

    # Registries
    _spec_registry: Dict[Type[BaseModel], EtherSpec] = {}
    _adapter_registry: Dict[tuple[Type[BaseModel], Type[BaseModel]], Callable[[Ether], dict]] = {}

    # ----- Registration -----

    @classmethod
    def register(
        cls,
        *,
        payload: Sequence[str],
        metadata: Sequence[str],
        extra_fields: str = "ignore",
        renames: Optional[Mapping[str, str]] = None,  # model_field -> ether dot path
        kind: Optional[str] = None,
    ):
        def _decorator(model_cls: Type[BaseModel]) -> Type[BaseModel]:
            known = set(getattr(model_cls, "model_fields", {}).keys())
            for f in list(payload) + list(metadata):
                if known and f not in known:
                    raise RegistrationError(f"{model_cls.__name__}: unknown field '{f}'")
            spec = EtherSpec(tuple(payload), tuple(metadata), extra_fields, dict(renames or {}), kind)
            cls._spec_registry[model_cls] = spec
            return model_cls
        return _decorator

    @classmethod
    def adapter(cls, src: Type[BaseModel], dst: Type[BaseModel]):
        def _decorator(fn: Callable[[Ether], dict]):
            cls._adapter_registry[(src, dst)] = fn
            return fn
        return _decorator

    # ----- Construction -----

    @classmethod
    def from_model(cls, model_instance: BaseModel, *, schema_version: int = 1) -> "Ether":
        spec = cls._spec_registry.get(type(model_instance))
        if not spec:
            raise RegistrationError(f"{type(model_instance).__name__} not registered")

        data = model_instance.model_dump()
        payload: Dict[str, Any] = {}
        metadata: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}

        def set_by_path(root: Dict[str, Any], path: str, value: Any):
            parts = path.split(".")
            cur = root
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = value

        listed = set(spec.payload_fields) | set(spec.metadata_fields)
        for f in spec.payload_fields:
            set_by_path(payload, spec.renames.get(f, f), data.get(f))
        for f in spec.metadata_fields:
            set_by_path(metadata, spec.renames.get(f, f), data.get(f))

        if spec.extra_fields == "error":
            unlisted = [f for f in data if f not in listed]
            if unlisted:
                raise ConversionError(f"Extra fields not allowed: {sorted(unlisted)}")
        elif spec.extra_fields == "keep":
            for f, v in data.items():
                if f not in listed:
                    extras[f] = v

        eth = cls(
            kind=spec.kind or "",
            schema_version=schema_version,
            payload=payload,
            metadata=metadata,
            extra_fields=extras,
            attachments=[],
        )
        eth._source_model = type(model_instance)
        return eth

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], BaseModel) and not kwargs:
            eth = self.__class__.from_model(args[0])
            super().__init__(**eth.model_dump())
            self._source_model = eth._source_model
        else:
            super().__init__(*args, **kwargs)

    # ----- Conversion -----

    def as_model(self, target_model: Type[ModelT], *, require_kind: bool = False) -> ModelT:
        spec = self._spec_registry.get(target_model)
        if not spec:
            raise RegistrationError(f"{target_model.__name__} not registered")

        if require_kind and spec.kind and self.kind and spec.kind != self.kind:
            raise ConversionError(f"Kind mismatch: Ether={self.kind!r}, Target expects {spec.kind!r}")

        # adapter path
        if self._source_model is not None:
            adapter = self._adapter_registry.get((self._source_model, target_model))
            if adapter:
                return target_model.model_validate(adapter(self))

        # default field picking
        def flatten(d: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
            out = {}
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, Mapping):
                    out.update(flatten(v, key))
                else:
                    out[key] = v
            return out

        payload_flat = flatten(self.payload)
        metadata_flat = flatten(self.metadata)
        extras = self.extra_fields

        def pick(model_field: str, from_payload: bool):
            ether_key = spec.renames.get(model_field, model_field)
            src = payload_flat if from_payload else metadata_flat
            if ether_key in src:
                return True, src[ether_key]
            if model_field in extras:
                return True, extras[model_field]
            if ether_key in extras:
                return True, extras[ether_key]
            return False, None

        data: Dict[str, Any] = {}
        for f in spec.payload_fields:
            ok, val = pick(f, True)
            if ok:
                data[f] = val
        for f in spec.metadata_fields:
            ok, val = pick(f, False)
            if ok:
                data[f] = val

        try:
            return target_model.model_validate(data)
        except ValidationError as ve:
            missing = _missing_required(target_model, data.keys())
            if missing:
                raise ConversionError(f"Missing required fields: {sorted(missing)}; provided={sorted(data.keys())}") from ve
            raise

    # ----- Utilities -----

    def summary(self) -> Dict[str, Any]:
        def flatten_keys(d: Mapping[str, Any]) -> list[str]:
            out = []
            def rec(prefix: str, obj: Mapping[str, Any]):
                for k, v in obj.items():
                    key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, Mapping):
                        rec(key, v)
                    else:
                        out.append(key)
            rec("", d)
            return sorted(out)
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "payload_keys": flatten_keys(self.payload),
            "metadata_keys": flatten_keys(self.metadata),
            "extra_keys": sorted(self.extra_fields.keys()),
            "attachments": [att.id for att in self.attachments],
            "source_model": getattr(self._source_model, "__name__", None),
        }

def _missing_required(model_cls: Type[BaseModel], present: Iterable[str]) -> set[str]:
    present = set(present)
    missing = set()
    for name, field in model_cls.model_fields.items():
        req = field.is_required() if hasattr(field, "is_required") else (
            field.default is None and field.default_factory is None
        )
        if req and name not in present:
            missing.add(name)
    return missing
```

### 4.2 Registration examples

```python
from pydantic import BaseModel

@Ether.register(
    payload=["embedding"],
    metadata=["source", "dim"],
    extra_fields="keep",
    renames={
        "embedding": "vec.values",   # nested in Ether.payload
        "dim": "vec.dim",
    },
    kind="embedding",
)
class FooModel(BaseModel):
    embedding: list[float]
    source: str
    dim: int
    note: str = "xtra"  # carried in extra_fields

@Ether.register(
    payload=["source"],
    metadata=[],
    extra_fields="ignore",
    kind="source_only",
)
class BarModel(BaseModel):
    source: str
    bar_field: int = 0

@Ether.adapter(FooModel, BarModel)
def foo_to_bar(eth: Ether) -> dict:
    vals = eth.payload["vec"]["values"]
    return {"source": eth.metadata.get("source", "unknown"), "bar_field": len(vals)}

foo = FooModel(embedding=[1.0,2.0,3.0], source="m1", dim=3)
e = Ether(foo)  # kind="embedding", schema_version=1
bar = e.as_model(BarModel)  # uses adapter
```

### 4.3 Attachments usage (tensors, Arrow)

```python
import numpy as np

# Create an embedding as an attachment rather than inline list
arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
att = Attachment(
    id="emb-0",
    uri="shm://embeddings/12345",   # could also set inline_bytes=arr.tobytes()
    media_type="application/x-raw-tensor",
    codec="RAW_F32",
    shape=[arr.size],
    dtype="float32",
    size_bytes=int(arr.nbytes),
    checksum={"algo":"crc32c", "value":"..."}
)

eth = Ether(
    kind="embedding",
    schema_version=1,
    payload={"values": None, "dim": int(arr.size)},  # explicit None when using attachment
    metadata={"source": "m1"},
    attachments=[att],
    extra_fields={}
)
```

> **Materialization policy:** nodes that understand `"RAW_F32"` (or `DLPACK`, `ARROW_IPC`) can **zero-copy** map buffers (e.g., shared memory, Arrow C data interface). Provide helper utilities in your runtime to fetch `inline_bytes` or `uri` lazily.


## 5. Interop & Transport

* **In-process / intra-host:** pass `Ether` objects directly (queues, asyncio). Avoid copying large `payload.values`; use **attachments**.
* **Cross-process / cross-host:** wrap an `Ether` in a framing protocol (e.g., **XCP**):

  * Control plane (HELLO/CAPS/NEGOTIATE) decides codecs.
  * Carry `payload`/`metadata` as **JSON** or **CBOR** for small messages, or as **Arrow IPC** for structured data.
  * Carry large tensors/tables in **attachments**: `ARROW_IPC`, `DLPACK`, `RAW_F32/F16/INT8`.
  * Compression order must be **compress -> encrypt** (if using AEAD). Define this in XCP.
* **Alternative wire**: provide a `.proto` for `Ether` (minus inline Python-only bits). Example sketch:

```protobuf
message Attachment {
  string id = 1;
  string uri = 2;                 // optional
  bytes inline_bytes = 3;         // optional
  string media_type = 4;
  string codec = 5;
  repeated int64 shape = 6;
  string dtype = 7;
  string byte_order = 8;
  string device = 9;
  int64 size_bytes = 10;
  map<string,string> checksum = 11;
  map<string,string> metadata = 12;
}

message EtherProto {
  string kind = 1;
  uint32 schema_version = 2;
  map<string, google.protobuf.Value> payload = 3;   // or a dedicated oneof per kind
  map<string, google.protobuf.Value> metadata = 4;
  map<string, google.protobuf.Value> extra_fields = 5;
  repeated Attachment attachments = 6;
}
```

> If using **Arrow Flight**, use `attachments` with `media_type=application/vnd.apache.arrow.stream` and place the IPC stream in `inline_bytes` or at a `uri`.


## 6. Validation, Errors, and Diagnostics

* **At model boundaries**:

  * On `from_model`: enforce `extra_fields` (`ignore|keep|error`).
  * On `as_model`: raise `ConversionError` with **missing field list**; prefer clear messages.
  * If `require_kind=True`, enforce `(target.kind == ether.kind)`.

* **In pipelines**:

  * Nodes should assert they produce one of their declared `(kind,version)` tuples.
  * Attach `trace_id`, `span_id`, `lineage` (append `{node, version, ts}`) in `metadata`.

* **Logging/metrics**:

  * Provide `Ether.summary()` for structured logs (keys, sizes, attachment ids).
  * Emit counters for conversions, adapter usage, missing-field errors, and attachment byte totals.


## 7. Performance Guidance

* **Don't copy big arrays.** Prefer **attachments** with `uri` (shared memory, memory-mapped files, Arrow IPC) so downstream nodes can zero-copy read.
* For Python:

  * Use **NumPy memoryviews** and **Arrow C Data Interface** to interop without copies.
  * Only **validate with Pydantic** at the edges (ingress/egress) and adapter outputs.
* **Compression**:

  * For small JSON payloads, skip compression (latency).
  * For large attachments, compress with **zstd** unless already compressed (Arrow IPC often benefits).
* **Quantization**:

  * `embedding.v1` may carry `INT8` attachments with per-row scale/zero-point in attachment `metadata`. Add `quantized=true` in Ether `metadata`.


## 8. Testing Strategy

* **Unit tests**:

  * Registration: unknown fields, rename collisions, nested path round-trips.
  * Conversion: success and missing required fields; `require_kind` enforcement.
  * Attachments: serialize/deserialize; ensure `shape`, `dtype`, `size_bytes` integrity.

* **Property-based** (Hypothesis):

  * Round-trip `Model -> Ether -> Model`.
  * For adapters: `A -> B -> A` invariants when expected lossless.

* **Golden samples**:

  * One JSON file per `(kind,version)` with representative payload/metadata and optional attachments (as fixtures with tiny binary blobs).
  * CI to prevent accidental breaking changes.


## 9. Versioning & Schema Registry

* **Registry entries**: `(schema_namespace?, kind, schema_version) -> { json_schema, pydantic_model, created_at, notes }`.
* **Evolution**:

  * Additive changes: bump `schema_version` (compatible); nodes MAY accept `<= current`.
  * Breaking: bump `schema_version` and ship an adapter (upgrade/downgrade).
* **Negotiation**:

  * In transport (e.g., XCP HELLO), advertise supported `(kind, version range)` and codecs.


## 10. Rollout Plan (v1)

1. **Implement** the Ether class above (core + registration + adapters + attachments).
2. **Define** and document initial kinds: `text.v1`, `tokens.v1`, `embedding.v1`, etc.
3. **Write** Pydantic models for edges (ingress/egress) and register them.
4. **Add** adapter(s) where field names/types diverge.
5. **Introduce** a small **schema registry** directory in the repo (JSON + Pydantic artifacts).
6. **Plumb** tracing (`trace_id`, `span_id`, `lineage`).
7. **Prototype** cross-process sending:

   * JSON for small envelopes.
   * Attachments via Arrow IPC (inline for tests; `file://`/`shm://` for larger).
   * Optionally, a thin XCP binding for HELLO + DATA with `ARROW_IPC` and `DLPACK` codecs.
8. **Benchmarks**:

   * End-to-end latency for `text->tokens->embedding`.
   * Throughput with 10--100k embeddings (inline vs attachments).


## 11. Open Extensions (v1.x)

* **Field-level transforms** in registration (mini-adapters per field) to avoid full pairwise adapters.
* **Computed metadata** (e.g., automatically computing `norm`).
* **Costed adapter graph** and planner integrated at the Ether layer (currently suggested at pipeline/router level).
* **Formal JSON Schema** emission for each `(kind,version)` for non-Python clients.


## 12. Quick Reference (Engineer Cheat Sheet)

* Use `@Ether.register(payload=[...], metadata=[...], renames={...}, kind="embedding")` on every boundary model.
* Convert with `Ether(model)` and `eth.as_model(TargetModel, require_kind=True)`.
* For large arrays/tables, **do not** put them in `payload`; use **attachments** with `codec` (`ARROW_IPC`, `DLPACK`, `RAW_F32`).
* Always set `kind` and `schema_version` on envelopes; nodes **must** output a declared `(kind,version)`.
* Prefer **additive evolution**; ship adapters for breaking changes.

## Additional TODOs:
*Prioritization:* start with **router** (core runtime), then **Arrow helper** (zero-copy perf), finish with **Protobuf** (cross-lang).


### Arrow IPC helper

1. **Define "attachment -> Arrow" mapping**
   * Decide canonical metadata keys: `shape`, `dtype`, `byte_order`, `device`.
   * Pick Arrow tensor extension (`FixedSizeList`, `LargeList`, or `Struct` + buffers).
2. **Write `to_arrow(eth: Ether) -> pa.BufferReader / bytes`**
   * Support `embedding.v1`, `tokens.v1` first.
   * Zero-copy when `attachments[i].uri.startswith("shm://")`.
3. **Write `from_arrow(buf) -> Attachment`**
   * Validate tensor metadata, populate `size_bytes`, `checksum`.
4. **Unit tests**
   * Round-trip NumPy -> Ether.attachments -> Arrow IPC -> Ether.attachments.
5. **Benchmarks** (optional sprint-2)
   * Compare serialization time vs. raw `pickle` for 1 k and 100 k vectors.

### Protobuf schema + serializers

1. **Draft `ether.proto`**
   * Messages: `EtherProto`, `Attachment`.
   * Use `google.protobuf.Struct` for `payload`, `metadata`, `extra_fields`.
   * Add enum for common codecs.
2. **Generate code** (Python, Go) with `protoc` in `scripts/gen_proto.sh`.
3. **Implement `Ether.from_proto(pb: EtherProto)` & `.to_proto()`**
   * Map `bytes` <-> `inline_bytes`; set `schema_version`, `kind`.
4. **Cross-language smoke-test**
   * Serialize in Python, deserialize in Go (or vice-versa).
5. **CI hook**: fail if `ether.proto` changed without running generator.

### Router with `accepts / emits` + auto-adapters

1. **Define `Node` base class**
   ```python
   class Node:
       name: str
       accepts: set[tuple[str,int]]
       emits: set[tuple[str,int]]
       def process(self, eth: Ether) -> Ether: ...
   ```
2. **Adapter registry**
   * `register_adapter(src: KindVer, dst: KindVer, fn, cost=1.0)`
   * Store in dict for Dijkstra.
3. **Planner**
   * Dijkstra to find cheapest path from `(kind,ver)` -> any of `node.accepts`.
   * Cache path per `(src,dst)` pair for speed.
4. **Router pipeline builder**
   ```python
   def route(nodes: list[Node]) -> Callable[[Ether], Ether]:
       ...
   ```
   * Inserts adapters between nodes as needed.
   * Raises `ConversionError` if no path exists.
5. **Runtime checks**
   * After each node, assert output `(kind,ver)` $\in$ `node.emits`.
   * Append lineage entry in `metadata.lineage`.
6. **Tests**
   * Happy-path: tokens -> Embedder -> embedding.
   * Fallback path with adapter insertion.
   * Negative: missing adapter ⇒ raises.
7. **Docs / examples**
   * Minimal `README` with two nodes + one adapter.
