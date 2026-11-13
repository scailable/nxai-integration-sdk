# Postprocessor Python Measure Average Car Speed

A v5 post-processor that decodes OCR logits from a chained license-plate
recognition model (same wire shape as `postprocessor-python-anpr-example`)
and tracks a recognized plate across two configured cameras to compute its
average speed over the distance between them.

## What it does

The processor handles two message shapes on the same `OBJECTS`-schedule
stream:

- **OCR messages** (`OriginalObjectID` + a raw `Tensors` block of NxM
  float32 character logits): decoded via
  `LogitsOcrEngine` on a background `OcrWorkerPool` and stored in an
  `OcrCache` keyed by object ID, with a TTL so stale entries expire.
- **Detector messages** (bounding boxes with object IDs): `handle_message`
  calls `message.handle(ocr_cache)`, which attaches `License Plate Text` /
  `Confidence` attributes for any object with a cached OCR result. Every
  object carrying recognized text is then fed to `SpeedMeasurementCache
  .add_detection()`, keyed by its `DeviceID`. When the same plate has now
  been seen on both configured cameras, the cache returns an average
  speed; `handle_message` attaches a `Speed` attribute (km/h) to that
  object via `SpeedDetectorMessage.add_speed_metadata` and emits a
  `speed.measurement` event. A plate that already has a cached
  speed gets the `Speed` attribute reattached on later frames until
  `second_appearance_timeout_sec` elapses since it was last seen.

## Models

Two ONNX models (a plate detector and a logits-emitting OCR model, chained
in **Feature Extraction** mode) are required to drive this processor, the
same pair used by `postprocessor-python-anpr-example` — see that
processor's `MODELS.md` for upload and pipeline-chaining steps.

## Behavior change from v4

- **`camera_1_id`, `camera_2_id`, `distance_between_cameras_m` and
  `second_appearance_timeout_sec` are live `UserSettings` controls**, read fresh from
  `message.user_settings` on every message with hardcoded defaults (empty camera ids, `100.0` m, `60.0` s)
  — replacing the v4 INI file value overridden by an
  `externalprocessor.`-prefixed `ExternalProcessorSettings` key. Speed
  measurement stays disabled (detector messages are still OCR-annotated)
  until both camera ids and a positive distance are configured via the
  UI.
- **`ocr_worker_count`, `ocr_output_name`, `cache_ttl_sec` are launch-time
  `Config` values** (`--key value` in `external_postprocessors.json`),
  replacing the removed `plugin.measure-average-car-speed.ini` file
  entirely.
- **The per-socket-path singleton lock is gone**: v5 launches exactly one
  process per registration entry over its own dedicated IPC channels, so
  there's no shared socket path to race on.
- **The emitted `speed.measurement` event drops the v4 `Caption` field**
  (v5 `EventMetadata` carries only `type_id` + `description`) and carries
  its id exactly as registered in `external_postprocessors.json`, with no
  `nx.nxai.` prefix.

## Registration

Add the processor to `external_postprocessors.json` (see the colocated
`external_postprocessors.json` and
`.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "Measure-Average-Car-Speed",
    "Command": "/path/to/build/postprocessor-python-measure-average-car-speed/postprocessor-python-measure-average-car-speed",
    "Schedule": "OBJECTS",
    "ReceiveBinaryData": true,
    "LogFilepath": "/tmp/postprocess.measure-average-car-speed.stdout.log",
    "Config": {
        "LogLevel": "DEBUG",
        "ocr_worker_count": "2",
        "ocr_output_name": "Identity:0",
        "cache_ttl_sec": "300"
    },
    "UserSettings": [
        {
            "type": "TextField",
            "name": "camera_1_id",
            "caption": "Camera 1 device ID",
            "description": "DeviceID (UUID) of the first camera on the measured route"
        },
        {
            "type": "TextField",
            "name": "camera_2_id",
            "caption": "Camera 2 device ID",
            "description": "DeviceID (UUID) of the second camera on the measured route"
        },
        {
            "type": "DoubleSpinBox",
            "name": "distance_between_cameras_m",
            "caption": "Distance between cameras (m)",
            "description": "Road distance in meters between camera 1 and camera 2",
            "defaultValue": 100.0,
            "minValue": 0.1
        },
        {
            "type": "DoubleSpinBox",
            "name": "second_appearance_timeout_sec",
            "caption": "Second appearance timeout (s)",
            "description": "How long to wait for the same plate on the other camera before giving up",
            "defaultValue": 60.0,
            "minValue": 0.1
        }
    ],
    "Events": [
        {
            "ID": "speed.measurement",
            "Name": "Average speed measured"
        }
    ]
}
```

Key points:

- `Schedule: "OBJECTS"` — the processor runs on inference-results messages
  (both the OCR model's and the detector's).
- `ReceiveBinaryData: true` — **required**. It is what adds the `Tensors`
  block the OCR path reads; without it the processor still starts and
  logs nothing unusual, but never recognizes a plate.
- `Config` carries the launch-time OCR worker pool size, the OCR output
  tensor name to extract from the `Tensors` block, and the cache TTL in
  seconds; the processor falls back to the same hardcoded defaults
  (`2`, `Identity:0`, `300.0`) if any key is omitted.
- `UserSettings` declares the two camera device ids (`TextField`) and the
  distance/timeout (`DoubleSpinBox`) as live, NX-plugin-UI-adjustable
  controls.
- `Events` declares the single bare event id the processor may emit;
  undeclared ids are silently dropped by the AI Manager.
- The processor emits `speed.measurement` exactly as the JSON declares
  it. The AI Manager copies `Events[].ID` into the DeviceAgent manifest
  verbatim and relays emitted ids verbatim, and the mediaserver drops an
  event whose id is not in the manifest — so a `nx.nxai.`-prefixed emitted
  id never reaches the VMS event pipeline. Object classes are the
  opposite: the engine still prefixes those with `nx.nxai.`.
- Select this processor for **both** models in the AI Manager pipeline
  editor, on both cameras whose feeds should be paired for speed
  measurement.

## Build, test, install

Build via the CMake tree:

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-python-measure-average-car-speed
cmake --install . --component postprocessor-python-measure-average-car-speed
```

Run the colocated unit tests:

```shell
cd postprocessor-python-measure-average-car-speed
PYTHONPATH=../nxai-utilities:.. python -m unittest test_postprocessor_python_measure_average_car_speed test_speed_cache -v
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
