# What the Post-processor Receives from the AI Manager

This document describes everything a post-processor can be handed each frame. A post-processor is a long-running program that stays connected to the AI Manager over a pair of shared-memory IPC channels and, for each frame, reads one MessagePack-encoded message.

For Python processors, unpacking is handled by `nxai_postprocessor_sdk.py` (`ObjectsPostProcessorMessage.from_bytes`, exposed as `processor.parse_objects_message()`) in `nxai-utilities/`. For C++ processors, `nxai-utilities/external_processor_utilities.h` ships the parsers (`parseObjectsPostProcessorMessage`, `parseImagePreprocessorMessage`, `parseTensorMessage`) and the message structs for the fields documented below.

---

## Transport

- **Format:** MessagePack over a pair of shared-memory IPC channels (one input, one output), set up once at process start — not a socket.
- **Launch arguments:** the AI Manager starts the processor as `<input_sync> <output_sync> [--Config-key value ...] --ProcessorName <name>`. The first two positional arguments are serialized IPC sync descriptors, not a socket path. Call `ExternalProcessor.initialize_from_args()` with no argument; it reads `sys.argv[1:]`, parses the descriptors and stores the processor name.
- **Development mode:** if the JSON entry omits `Command`, the AI Manager instead connects as a TCP client to `127.0.0.1:<Port>`. Start the processor yourself with `--port <n>`; `initialize_from_args()` detects the `--port` form and switches to the TCP transport transparently — `wait_for_message()`/`send_response()` behave identically either way.
- **Framing:** Each message is one complete MessagePack document, length-prefixed. `wait_for_message()` / `send_response()` handle the framing (SHM signalling in the IPC case, a 4-byte length header in the `--port` case) so processors don't need to think about it.
- **Connection lifecycle:** one long-lived channel pair for the life of the process — not a socket opened and closed per frame. The processor loops on `wait_for_message()`, which returns `None` on timeout and also on the AI Manager's `EXIT` signal (see "Exit signal" below); it raises `RuntimeError` if the IPC connection is lost.

---

## Messages per frame

The AI Manager sends **one** MessagePack message per frame (v5): the inference-results
message described below, carrying bounding boxes, scores, metadata, and — if
requested — the raw input frame inline as a root `ImageData` map (no separate
image-header message or shared-memory read; see [Raw input frame: `ImageData`](#raw-input-frame-imagedata) below).

> **v4 note (superseded):** the v4 wire sent a *second* message — a small map with a
> shared-memory key pointing at the raw input tensor — gated by `ReceiveInputTensor:
> true`. v5 replaced this with the single-message `ImageData` root key described below,
> gated by `ReceiveInputImage: true`. Processors that use the old two-message/SHM flow
> do not run on a v5 engine.

---

## Exit signal

On v5 shutdown arrives as an IPC signal on the input channel (`IpcSignal.EXIT`), not a MessagePack document. `wait_for_message()` returns `None` for it exactly as it does for a plain timeout — there is no `{"EXIT": ...}` message to parse. To tell the two apart, read `ExternalProcessor.exit_requested` after `None`. The SDK sets it to `True` when it reads the EXIT signal. After a timeout it stays `False`.

> **v4 note (superseded):** the v4 socket transport sent a wire-level termination message, `{"EXIT": <value>}`. v5 does not send it.

Because `wait_for_message()` returns `None` for both cases, a loop that only `continue`s on `None` never stops. The engine then terminates it after its grace period instead of letting it exit cleanly. Check `processor.exit_requested` after each `None` and leave the loop when it is `True`, as `postprocessor-python-image-example` does.

---

## Message: inference results

The full schema, expressed as JSON for readability. `Objects` and `Events` are always
present (possibly empty); the rest are optional — present only when the pipeline hands
off between models (`OriginalObjectID`), the processor's JSON declares `UserSettings`,
or its JSON sets `ReceiveInputImage`/`ReceiveBinaryData`.

```json
{
    "Timestamp": <int64>,
    "DeviceID": "<Device ID>",
    "DeviceName": "<Device Name>",
    "Objects": {
        "<Class Name>": [
            {
                "ID": "<UUID>",
                "x": <float>,
                "y": <float>,
                "width": <float>,
                "height": <float>,
                "confidence": <float>,
                "attributes": { "<key>": "<value>" }
            }
        ]
    },
    "Events": [
        { "ID": "<event id>", "Description": "<text>" }
    ],
    "OriginalObjectID": "<UUID>",
    "UserSettings": {
        "<setting name>": <setting value>
    },
    "ImageData": { "Height": <uint64>, "Width": <uint64>, "Layout": <int>, "Data": <binary blob> },
    "Tensors": [
        { "Name": "<tensor name>", "DataType": <uint8 dtype code>, "Shape": [<uint>, ...], "Data": <binary blob> }
    ]
}
```

There is no `BBoxes_xyxy`/`ObjectsMetaData` root shape and no top-level `Width`/`Height`,
`InputIndex`, `Scores`, `Counts`, `Identity`, or `BinaryOutputs` key on v5 — those were
v4 fields (or, for `BinaryOutputs`, a name later replaced by `Tensors`). See
`nxai-utilities/nxai_postprocessor_sdk.py` for the wire format this section describes.

### Device and frame identity

- **`Timestamp`** (int64) — Timestamp of the frame, in microseconds.
- **`DeviceID`** (string) — Stable identifier for the camera/device the frame came from. Use this to separate state if the same processor is used across multiple cameras.
- **`DeviceName`** (string) — Human-readable name of the device. Suitable for logs; not stable across renames.

### Detections: `Objects`

A map from class name to a list of detected-object structs — the same `Objects` shape the processor sends back (see `output_to_ai_manager.md`). The Python SDK parses each entry into an `ObjectMetadata`:

- **`ID`** — UUID string of the object's existing track. Always present on input (these are already-tracked objects); `NewID` never appears here — it's a response-only field for a processor to mint a new track.
- **`x`, `y`, `width`, `height`** (float) — normalized `[0,1]`, relative to the frame. Not pixels.
- **`confidence`** (float) — detection confidence.
- **`attributes`** (`dict[str, str]`) — key-value pairs attached to the object so far (by the model or an earlier post-processor in the pipeline, if `RunLast` ordering puts this processor after others).

If the model is not a detector, `Objects` is present but empty.

### Pre-existing events: `Events`

A list of `{ID, Description}` events already attached to this frame by an earlier stage of the pipeline (e.g. another post-processor that ran first). Usually empty. Same shape as what the processor appends and sends back — see `output_to_ai_manager.md`.

### Two-stage cross-reference: `OriginalObjectID`

A UUID string that links a downstream inference (e.g. OCR logits over a license-plate crop) back to the detector object that produced the crop. Only present when the pipeline performs a two-stage inference that hands results off between models. See `postprocessor-python-anpr-example/README.md`.

> **v5 — this field is `Schedule: OBJECTS` only.** The engine writes it only into the `Schedule: OBJECTS` message. A TENSOR-schedule message is built by a different function whose root is a *fixed-size* map of six keys — `Timestamp`, `DeviceID`, `DeviceName`, `Tensors`, `TensorShapes`, `TensorDataTypes` — plus an optional `UserSettings`, so it can never carry the field. A processor that needs both a chained stage's tensors and the originating track must therefore register `Schedule: OBJECTS` with `ReceiveBinaryData: true`; `NoResponse: true` is supported alongside them.

### Raw model outputs: `Tensors`

Present only when the processor's JSON entry has `ReceiveBinaryData: true`. An array of output tensors, each of the form:

```json
{
    "Name": "<tensor name>",
    "DataType": <uint8 dtype code>,
    "Shape": [<uint>, ...],
    "Data": <binary blob>
}
```

> **v5:** the key is spelled `Tensors`, never `BinaryOutputs`, and its shape depends on the schedule. On `Schedule: OBJECTS` it is an array of self-describing maps — `{Name, DataType, Shape, Data}`, so the shape *is* transmitted. On `Schedule: TENSOR` it is a map from tensor name to binary blob, with the shapes and dtypes in the parallel root arrays `TensorShapes` and `TensorDataTypes`. `nxai-utilities/nxai_postprocessor_sdk.py` parses each form in the message class matching its schedule.

The dtype codes, defined as `DATA_TYPE_*` constants in `nxai-utilities/nxai_postprocessor_sdk.py`:

| Code | Type   | Code | Type    |
|------|--------|------|---------|
| 1    | FLOAT  | 8    | STRING  |
| 2    | UINT8  | 9    | BOOL    |
| 3    | INT8   | 11   | DOUBLE  |
| 4    | UINT16 | 12   | UINT32  |
| 5    | INT16  | 13   | UINT64  |
| 6    | INT32  |      |         |
| 7    | INT64  |      |         |

Unlike v4's `BinaryOutputs`, the shape is transmitted on the wire as `Shape`, so the processor doesn't need to know it ahead of time. `ObjectsPostProcessorMessage` parses this form into its `tensors` map, keyed by tensor name.

### UI settings from the user: `UserSettings`

A root-level map carrying the current values of the UI controls declared in the processor's JSON `UserSettings` field. Keys match the `"name"` of each control **verbatim** — v5 controls are declared without the old `externalprocessor.` prefix, so a control named `attributeName` arrives under the key `attributeName`. String-typed controls (TextField, PasswordField, ComboBox) arrive as strings; controls with non-string types (SpinBox, DoubleSpinBox, CheckBox, PolygonFigure, etc.) may arrive as their native type or as a JSON-encoded string depending on the runtime — defensive conversion (e.g. `float(value)`) is the safe pattern.

Example, given `TextField` controls with `"name": "attributeName"` and `"name": "attributeValue"`:

```json
"UserSettings": {
    "attributeName": "reviewed",
    "attributeValue": "true"
}
```

The processor reads this every frame and applies whatever the user has configured in the NX plugin settings. The Python SDK parses it into `ObjectsPostProcessorMessage.user_settings`. See `postprocessor-python-settings-example/postprocessor-python-settings-example.py` for the read-and-apply pattern.

**Preprocessors receive the same map.** The same root `UserSettings` key, with the same flat `{name: value}` shape and the same emitted-only-when-non-empty rule, is written on both preprocessor messages — the `TENSOR`-schedule tensors message and the `IMAGE`-schedule image message. The Python SDK parses it into `TensorMessage.user_settings`; `ImagePreprocessorMessage.from_bytes` parses it into `msg.user_settings` in the same way. Settings are **inbound only** on every schedule: a preprocessor echoes the message back, but `to_bytes` never re-emits `UserSettings`. The key is absent while the map is empty, so read it with a hardcoded default as the absent-key fallback (`preprocessor-python-tensor-example` is the reference).

### Raw input frame: `ImageData`

Present only when the processor's JSON entry has `ReceiveInputImage: true`, as a sibling of `Timestamp`/`Objects`/`Events` on the same inference-results message (no second message, no shared-memory read):

```json
{
    "Height": <uint64>,
    "Width":  <uint64>,
    "Layout": <int>,
    "Data":   <binary blob>
}
```

- **`Height`**, **`Width`** — Frame dimensions in pixels.
- **`Layout`** — Channel order: `0` = CHW, `1` = HWC (`ImageOrder` enum in the Python SDK).
- **`Data`** — Raw pixel data, in the order `Layout` specifies. The element type is
  uint8 RGB, 3 bytes per pixel, so the blob is `Height * Width * 3` bytes long. `ImageData`
  has no element-type field.

  The blob is large. A 4K frame is 3840 × 2160 × 3 = 24,883,200 bytes. That is more than
  the initial SHM channel, so a 4K `ReceiveInputImage` processor goes into the SHM
  reallocation path.

The Python SDK parses this automatically: `ObjectsPostProcessorMessage.image_data` is
an `ImageData` instance (`.height`, `.width`, `.layout`, `.bytes`) when the key is
present, else `None`. Canonical read pattern, from
`postprocessor-python-image-example/postprocessor-python-image-example.py`:

```python
objects_message = processor.parse_objects_message(message_bytes)
frame = _decode_frame(objects_message.image_data)
```

`_decode_frame` checks that the buffer holds `height x width x 3` uint8 values. It reshapes the buffer according to `layout` (`CHW` or `HWC`) and builds a Pillow RGB image from it.

An `IMAGE`-schedule preprocessor receives the same frame, with the same keys, the same
layout and the same uint8 RGB element type. `preprocessor-python-image-example` is the
reference.

The preprocessor replies with an `ImageData` map of the same shape. The engine gets the
element type of the reply from the `Data` size and the `Width` and `Height` of the reply
itself:

- `Width * Height * 3 * 4` bytes — float32. The engine uses the buffer as it is.
- `Width * Height * 3` bytes — uint8. The engine widens it to float.
- Any other size — the engine logs this warning and ignores the reply:

  ```
  Warning! Preprocessor <name> returned an image buffer of unexpected size (<N> bytes for <W>x<H>); expected <F> (float32) or <U> (uint8). Ignoring.
  ```

  The frame is not lost. The model runs on the original image, as if the preprocessor
  did not change it. The processor continues to receive frames.

`ImageData` is inbound-only — the SDK never re-emits it on the response message.

---

## Worked example of a parsed message

After `processor.parse_objects_message(message_bytes)`, the resulting `ObjectsPostProcessorMessage` for a single frame from a detection pipeline looks approximately like this (attributes shown as the parsed struct's fields, not the wire dict):

```python
msg.device_id     # 'cam-01'
msg.device_name   # 'Front Gate'
msg.timestamp     # 1729000000000
msg.objects
# {
#     'car': [
#         ObjectMetadata(track_id='a1...', x=0.21, y=0.19, width=0.14, height=0.21, confidence=0.91, attributes={}),
#         ObjectMetadata(track_id='b2...', x=0.57, y=0.28, width=0.13, height=0.19, confidence=0.87, attributes={}),
#     ],
#     'person': [
#         ObjectMetadata(track_id='c3...', x=0.47, y=0.56, width=0.04, height=0.20, confidence=0.76, attributes={}),
#     ],
# }
msg.events         # []
msg.user_settings  # {'attributeName': 'reviewed', 'attributeValue': 'true'}
```

Two cars, one person, per-object track UUIDs and confidences, two user-configured settings. The processor's job from here is to read, optionally mutate `msg.objects`/`msg.events`, and send back — see the companion document on output format.

---

## Evidence in the repo

- Wire format (input side): `nxai-utilities/nxai_postprocessor_sdk.py` and `nxai-utilities/external_processor_utilities.cpp`.
- Python parsing: `nxai-utilities/nxai_postprocessor_sdk.py` (`ObjectsPostProcessorMessage`/`ObjectMetadata`/`EventMetadata`/`ImageData`/`ImageOrder`) and `nxai-utilities/tests/test_message_roundtrip.py` (round-trip tests, including the `ImageData` and `UserSettings` root keys).
- Object/attribute reading: `postprocessor-python-confidences-example/postprocessor-python-confidences-example.py`, `postprocessor-python-settings-example/postprocessor-python-settings-example.py`.
- `OriginalObjectID` handling: `postprocessor-python-anpr-example/README.md`.
- `UserSettings` consumption: `postprocessor-python-settings-example/postprocessor-python-settings-example.py`.
- Dtype codes: `nxai-utilities/nxai_postprocessor_sdk.py` (`DATA_TYPE_*` constants, `DATA_TYPE_NAMES`).
