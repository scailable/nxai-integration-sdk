# Postprocessor Python Object Counter Threshold

A v5 post-processor that counts detected objects per class each frame and fires
the `object.counter.threshold` event when the total count meets or
exceeds a configurable threshold. It is a noise-reduction tool: alerts are only generated
when a meaningful number of objects is present, not on every frame.

## What it does

`handle_message` calls `count_objects` (which counts via `len(msg.objects[cls])`)
and sums the results. If the total is `>= min_object_count`, it appends an
`EventMetadata` to `msg.events`:

```python
event.type_id = "object.counter.threshold"
event.description = "Detected 2 car, 3 person. Total: 5 objects."
```

The emitted id is exactly the `object.counter.threshold` id declared in the
registration JSON — see "Registration" below.

When `msg.objects` is empty the processor no-ops — no event is appended.

## Behavior change from v4

- **Threshold is a live `UserSettings` control** (`min_object_count`), read
  fresh from `msg.user_settings` on every frame — the NX plugin UI controls
  the threshold at runtime, same as v4. It is not a launch-time `Config` arg.
- **`Caption` field dropped**: v4 emitted `{"Caption": "...", "Description": "..."}`
  on the event. The v5 `EventMetadata` carries only `type_id` and `description`;
  there is no `Caption` in the v5 wire codec.

## Registration

Add the processor to `external_postprocessors.json` (see the colocated
`external_postprocessors.json` and `.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "Object-Counter-Threshold",
    "Command": "/path/to/postprocessor-python-object-counter-threshold",
    "Schedule": "OBJECTS",
    "LogFilepath": "/tmp/postprocess.object-counter-threshold.stdout.log",
    "Config": {
        "LogLevel": "DEBUG"
    },
    "UserSettings": [
        {
            "type": "SpinBox",
            "name": "min_object_count",
            "caption": "Minimum Object Count",
            "description": "Fire the threshold event when the total detected object count meets or exceeds this value",
            "defaultValue": 5
        }
    ],
    "Events": [
        {
            "ID": "object.counter.threshold",
            "Name": "Object count threshold exceeded"
        }
    ]
}
```

Key points:

- `Schedule: "OBJECTS"` — the processor runs on inference-results messages.
- `Config` delivers only launch-time `LogLevel`; the threshold is a live
  `UserSettings` value, not launch-time config.
- `UserSettings` declares `min_object_count` as a `SpinBox` with a default of
  `5`; the processor falls back to that same default when the engine forwards
  no settings (e.g. an unpatched engine).
- `Events` declares the single event ID the processor may emit; undeclared IDs
  are silently dropped by the AI Manager.
- The JSON id and the emitted id are the same bare `object.counter.threshold`.
  The AI Manager copies `Events[].ID` into the DeviceAgent manifest verbatim
  and relays emitted event ids verbatim too, and the mediaserver drops events
  whose id is not in the manifest — so both sides must be the bare id
  Event ids differ from object classes here: those the engine
  still prefixes with `nx.nxai.`.
- No `Objects`: the processor counts existing detections and declares no new
  class.

## Build, test, install

Build via the CMake tree:

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-python-object-counter-threshold
cmake --install . --component postprocessor-python-object-counter-threshold
```

Run the colocated unit tests:

```shell
cd postprocessor-python-object-counter-threshold
PYTHONPATH=../nxai-utilities python -m unittest test_postprocessor_python_object_counter_threshold -v
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
