# Postprocessor Python Events Example

A v5 post-processor that emits an `ex.example.event` event on every
frame, describing how many objects of each class were detected.

## What it does

`handle_message` builds a description string from `msg.objects` and appends
exactly one `EventMetadata` to `msg.events` on every frame:

```python
event = nxai_postprocessor_sdk.EventMetadata()
event.type_id = "ex.example.event"
event.description = description
msg.events.append(event)
```

The description lists the per-class object count, e.g.
`"\nThere are 2 car's in the frame"`, or `"\nThere are no objects in the
frame."` when nothing was detected.

Preserved from v4 (not a bug fix): building the description is order
dependent on `msg.objects`' class iteration order. A class with zero
detections that appears before the first class with a nonzero count has its
partial "no objects" text fully discarded rather than appended to; classes
encountered after the first nonzero-count class are appended normally
regardless of their own count. See `test_postprocessor_python_events_example.py`
for worked examples of both orderings.

## Behavior change from v4

- v4 set an event `Caption` field; v5 `EventMetadata` has no `Caption` — only
  `type_id` and `description` are serialized.
- v4 derived the per-class count as `len(coordinates) / 4` from flat xyxy
  arrays; v5's `msg.objects[class_name]` is already one entry per detected
  object, so the count is `len(class_objects)` directly. The displayed number
  is unchanged.

## Registration

Add the processor to `external_postprocessors.json` (see the colocated
`external_postprocessors.json` and `.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "Events-Example",
    "Command": "/path/to/build/postprocessor-python-events-example/postprocessor-python-events-example",
    "Schedule": "OBJECTS",
    "LogFilepath": "/tmp/postprocess.events-example.stdout.log",
    "Config": {
        "LogLevel": "DEBUG"
    },
    "Events": [
        {
            "ID": "ex.example.event",
            "Name": "Example event"
        }
    ]
}
```

Key points:

- `Schedule: "OBJECTS"` — the processor runs on inference-results messages.
- `Config` delivers launch-time configuration: only `LogLevel`.
- `Events` declares the `ex.example.event` id emitted every frame. The AI
  Manager passes it into the DeviceAgent manifest verbatim, so the
  emitted id is the same bare id — no `nx.nxai.` prefix, unlike object classes.
- No `Objects`: the processor adds no new classes, it only reads existing
  detections to build the event description.

## Build, test, install

Build via the CMake tree:

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-python-events-example
cmake --install . --component postprocessor-python-events-example
```

Run the colocated unit tests:

```shell
cd postprocessor-python-events-example
PYTHONPATH=../nxai-utilities python -m unittest test_postprocessor_python_events_example -v
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
