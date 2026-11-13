# Postprocessor Python ANPR Example

A v5 post-processor for the license-plate OCR model chained after a plate
detector. It decodes each plate crop's OCR logits into readable text and
writes it straight onto the original plate object as attributes, emitting a
deduplicated `plate_recognized` event once the recognized text clears a
confidence threshold.

## What it does

The processor is assigned to the **OCR model only**. The AI Manager sends it
one `OBJECTS`-schedule message per plate crop, carrying:

- `OriginalObjectID` — the track ID of the plate object the crop was taken
  from. It is the only field of that object the
  processor receives.
- A raw `Tensors` block with the crop's NxM float32 character logits.

`handle_message` decodes the logits with `LogitsOcrEngine` and keeps the
best-confidence reading per plate in an `OcrCache` (with a TTL, so stale
entries expire). It then answers with an attribute-only update of the
original plate object (`InferenceMessage.update_original_object`):

- **A reading exists for the plate** (from this crop or an earlier one): sets
  `License Plate Text` and `Confidence` to the best reading so far, and emits
  `anpr.plate_recognized` when the plate's text differs from the last text an
  event was emitted for and its confidence clears `min_confidence`
  (deduplicated via `EventDeduplicationCache`).
- **No reading** (blank decode, unusable tensors, nothing cached): lists both
  attributes under `removeAttributes`, so the plate never keeps text from a
  stale result.

The update carries only the plate's `ID`, its attributes and
`removeAttributes` — no bounding box or confidence — so the AI Manager keeps
the object's detection as-is and only edits its attributes. This needs an
AI Manager plugin that accepts attribute-only object updates and
`removeAttributes`.

## Models

Two ONNX models (a plate detector and a logits-emitting OCR model, chained
in **Feature Extraction** mode) are required to drive this processor — see
[MODELS.md](MODELS.md) for upload and pipeline-chaining steps.

## Behavior change from the earlier v5 version

- **Assigned to the OCR model only.** It used to be assigned to both models,
  caching OCR results from the chained stage and attaching them to the
  detector's message for the same object on a later frame. A chained-stage
  processor can now edit the original object by its ID, so the plate text is
  applied on the same frame.
- **OCR decoding is synchronous**, since the result has to be in the response
  to the same message. The OCR worker pool and its `ocr_worker_count`
  `Config` value are gone.

## Behavior change from v4

- **`min_confidence` is a live `UserSettings` control**,
  read fresh from `message.user_settings` on every message with a hardcoded
  default fallback (`0.95`) — replacing the v4 INI value overridden by an
  `externalprocessor.`-prefixed `ExternalProcessorSettings` key.
- **`ocr_output_name`, `cache_ttl_sec`, `plate_class_name` are launch-time
  `Config` values** (`--key value` in `external_postprocessors.json`),
  replacing the removed `plugin.anpr.ini` file entirely.
- **The per-socket-path singleton lock is gone**: v5 launches exactly one
  process per registration entry over its own dedicated IPC channels, so
  there's no shared socket path to race on.
- **The emitted event drops the v4 `Caption` field** (v5 `EventMetadata`
  carries only `type_id` + `description`) and carries the id
  `anpr.plate_recognized` exactly as registered in
  `external_postprocessors.json`, with no `nx.nxai.` prefix.

## Registration

Add the processor to `external_postprocessors.json` (see the colocated
`external_postprocessors.json` and
`.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "ANPR-Example",
    "Command": "/path/to/build/postprocessor-python-anpr-example/postprocessor-python-anpr-example",
    "Schedule": "OBJECTS",
    "ReceiveBinaryData": true,
    "LogFilepath": "/tmp/postprocess.anpr-example.stdout.log",
    "Config": {
        "LogLevel": "DEBUG",
        "ocr_output_name": "Identity:0",
        "cache_ttl_sec": "300",
        "plate_class_name": "LP"
    },
    "UserSettings": [
        {
            "type": "DoubleSpinBox",
            "name": "min_confidence",
            "caption": "Minimum Confidence",
            "description": "Minimum OCR confidence (0.0-1.0) to emit a plate_recognized event",
            "defaultValue": 0.95
        }
    ],
    "Events": [
        {
            "ID": "anpr.plate_recognized",
            "Name": "License plate recognized"
        }
    ]
}
```

Key points:

- `Schedule: "OBJECTS"` — the processor runs on the OCR model's
  inference-results messages, which carry `OriginalObjectID`.
- `ReceiveBinaryData: true` — **required**. It is what adds the `Tensors`
  block the OCR path reads; without it every plate gets its attributes
  removed and the processor logs a warning per message.
- `Config` carries the OCR output tensor name to extract from the `Tensors`
  block, the cache TTL in seconds, and the class name the original plate
  object is returned under (the detector model's plate class, `LP` for the
  shipped model). The processor falls back to the same hardcoded defaults
  (`Identity:0`, `300.0`, `LP`) if any key is omitted.
- `UserSettings` declares `min_confidence` (`DoubleSpinBox`, default
  `0.95`) as a live, NX-plugin-UI-adjustable control.
- `Events` declares the single bare event id the processor may emit;
  undeclared ids are silently dropped by the AI Manager.
- The processor emits `anpr.plate_recognized` exactly as the JSON declares
  it. The AI Manager copies `Events[].ID` into the DeviceAgent manifest
  verbatim and relays emitted ids verbatim, and the mediaserver drops an
  event whose id is not in the manifest — so a `nx.nxai.`-prefixed emitted
  id never reaches the VMS event pipeline. Object classes are the
  opposite: the engine still prefixes those with `nx.nxai.`.
- Select this processor for the **OCR model only** in the AI Manager
  pipeline editor — see [MODELS.md](MODELS.md) for the chaining steps.

## Build, test, install

Build via the CMake tree:

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-python-anpr-example
cmake --install . --component postprocessor-python-anpr-example
```

Run the colocated unit tests:

```shell
cd postprocessor-python-anpr-example
PYTHONPATH=../nxai-utilities:.. python -m unittest test_postprocessor_python_anpr_example -v
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
