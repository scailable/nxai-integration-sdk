# Postprocessor Python Best-Shots Example

A v5 post-processor that nominates the best shot of an already-tracked `car`
object. The `emit_best_shots` Config flag turns the behavior on or off.

## What it does

`handle_message` returns the message unchanged unless `emit_best_shots` is
true. When the flag is true, it finds the first `car` object that has a
`track_id`. It then appends one `BestShotMetadata` to `msg.best_shots`:

```python
best_shot = nxai_postprocessor_sdk.BestShotMetadata()
best_shot.track_id = target.track_id
best_shot.x, best_shot.y, best_shot.width, best_shot.height = _SENTINEL_BBOX
msg.best_shots.append(best_shot)
```

`_SENTINEL_BBOX` is the fixed rectangle `(0.125, 0.0625, 0.03125, 0.03125)`.
The processor does not use the position of the object. A real detector does
not produce this rectangle, so you can see which best shots come from this
processor.

Only an already-tracked car is eligible. An object that has only
`new_track_id` (first seen in this frame) is skipped. A best shot names an
existing track UUID and never creates one. If the frame has no tracked car,
the processor sends no best shot.

## Best-shot fields

- `track_id` (`TrackId` on the wire) is required. It is the UUID of an
  already-tracked object.
- `x`, `y`, `width` and `height` are optional floats, normalized to `[0,1]`.
  Set all four to crop the shot to that rectangle.
- If `msg.best_shots` is empty, the response has no `BestShots` key.

Every track gets a best shot from the server, even when no post-processor
sends one. `BestShots` only gives you control over which rectangle and frame
the VMS shows. See `.claude/docs/output_to_ai_manager.md` for the full field
reference.

## Registration

Add the processor to `external_postprocessors.json` (see the colocated
`external_postprocessors.json` and `.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "BestShots-Example",
    "Command": "/path/to/build/postprocessor-python-bestshots-example/postprocessor-python-bestshots-example",
    "Schedule": "OBJECTS",
    "LogFilepath": "/tmp/postprocess.bestshots-example.stdout.log",
    "Config": {
        "LogLevel": "DEBUG",
        "emit_best_shots": "true"
    }
}
```

Key points:

- `Schedule: "OBJECTS"`: the processor runs on inference-results messages.
- `Config` delivers launch-time configuration: `LogLevel` and
  `emit_best_shots`. The value `"true"` (any letter case) turns best shots
  on. Any other value, or no key, turns them off.
- No `Objects` and no `Events`: you do not declare best shots in the
  registration entry.

## Build, test, install

Build with the CMake tree:

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-python-bestshots-example
cmake --install . --component postprocessor-python-bestshots-example
```

Run the colocated unit tests from the `postprocessor-python-bestshots-example/`
directory. The `msgpack` module must be installed:

```shell
python -m unittest test_postprocessor_python_bestshots_example -v
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
