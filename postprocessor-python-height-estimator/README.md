# Postprocessor Python Height Estimator

A v5 post-processor that estimates every detected person's height from their
bounding box, attaches it as a `Height` attribute, and fires the
`person.height.exceeded` event when any person is taller than a
configurable limit.

## What it does

`handle_message` calls `add_height_attributes`, which walks every `person`
object (class name matched case-insensitively) and sets:

```python
height_m = obj.height / frame_fraction_per_meter
obj.attributes["Height"] = f"{height_m:.2f} m"
```

If any person's estimated height exceeds `max_height_cm / 100` metres, it appends
an `EventMetadata` to `msg.events`:

```python
event.type_id = "person.height.exceeded"
event.description = "1 of 1 detected person(s) exceed the height limit of 2.50 m. Tallest: 2.80 m."
```

The emitted id is exactly the `person.height.exceeded` id declared in the
registration JSON — see "Registration" below.

When `msg.objects` has no persons the processor no-ops — no attribute, no event.

## Behavior change from v4

- **Height is computed from the normalized bbox height**: `height_m =
  obj.height / frame_fraction_per_meter`. v5 bounding boxes are frame-relative
  `[0,1]`, so the v4 pixel formula `(y2 - y1) / pixels_per_meter` (which needed
  the raw image) is gone, and the calibration knob is renamed
  `pixels_per_meter` → `frame_fraction_per_meter` (default `0.25`).
- **Calibration is a live `UserSettings` control** (`frame_fraction_per_meter`,
  `max_height_cm`), read fresh from `msg.user_settings` on every frame — the NX
  plugin UI controls calibration at runtime, same as v4. It is not a
  launch-time `Config` arg.
- **`Caption` field dropped**: the v5 `EventMetadata` carries only `type_id` and
  `description`; there is no `Caption` in the v5 wire codec.

## Registration

Add the processor to `external_postprocessors.json` (see the colocated
`external_postprocessors.json` and `.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "Person-Height-Estimator",
    "Command": "/path/to/postprocessor-python-height-estimator",
    "Schedule": "OBJECTS",
    "LogFilepath": "/tmp/postprocess.height-estimator.stdout.log",
    "Config": {
        "LogLevel": "DEBUG"
    },
    "UserSettings": [
        {
            "type": "DoubleSpinBox",
            "name": "frame_fraction_per_meter",
            "caption": "Frame Fraction Per Meter",
            "description": "Fraction of the frame height that spans 1 metre in this camera view",
            "defaultValue": 0.25
        },
        {
            "type": "SpinBox",
            "name": "max_height_cm",
            "caption": "Max Height (cm)",
            "description": "Fire the height-exceeded event when an estimated person height exceeds this value",
            "defaultValue": 250
        }
    ],
    "Events": [
        {
            "ID": "person.height.exceeded",
            "Name": "Person height exceeded"
        }
    ]
}
```

Key points:

- `Schedule: "OBJECTS"` — the processor runs on inference-results messages.
- `Config` delivers only launch-time `LogLevel`; calibration is a live
  `UserSettings` value, not launch-time config.
- `UserSettings` declares `frame_fraction_per_meter` (`DoubleSpinBox`, default
  `0.25`) and `max_height_cm` (`SpinBox`, default `250`); the processor falls
  back to those same hardcoded defaults when the engine forwards no settings
  (e.g. an unpatched engine).
- `Events` declares the single event ID the processor may emit; undeclared IDs
  are silently dropped by the AI Manager.
- The JSON id and the emitted id are the same bare `person.height.exceeded`.
  The AI Manager copies `Events[].ID` into the DeviceAgent manifest verbatim
  and relays emitted event ids verbatim too, and the mediaserver drops events
  whose id is not in the manifest — so both sides must be the bare id
  Event ids differ from object classes here: those the engine
  still prefixes with `nx.nxai.`.
- No `Objects`: the `Height` attribute attaches to existing `person` detections
  and declares no new class.

## Build, test, install

Build via the CMake tree:

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-python-height-estimator
cmake --install . --component postprocessor-python-height-estimator
```

Run the colocated unit tests:

```shell
cd postprocessor-python-height-estimator
PYTHONPATH=../nxai-utilities python -m unittest test_postprocessor_python_height_estimator -v
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
