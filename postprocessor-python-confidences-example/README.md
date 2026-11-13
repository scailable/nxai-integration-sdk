# Postprocessor Python Confidences Example

A v5 post-processor that adds a `Confidence` attribute to every detected
object, across all classes.

## What it does

`handle_message` iterates `msg.objects` and, for every object in every class,
sets:

```python
obj.attributes["Confidence"] = str(round(obj.confidence, 2))
```

Formatting is preserved from v4 exactly: bare `str(round(x, 2))`, no fixed
decimal width and no unit suffix (e.g. `"0.9"`, `"1.0"`, never `"0.90"`).

## Registration

Add the processor to `external_postprocessors.json` (see the colocated
`external_postprocessors.json` and `.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "Confidences-Example",
    "Command": "/path/to/build/postprocessor-python-confidences-example/postprocessor-python-confidences-example",
    "Schedule": "OBJECTS",
    "LogFilepath": "/tmp/postprocess.confidences-example.stdout.log",
    "Config": {
        "LogLevel": "DEBUG"
    }
}
```

Key points:

- `Schedule: "OBJECTS"` — the processor runs on inference-results messages.
- `Config` delivers launch-time configuration: only `LogLevel`.
- No `Objects` and no `Events`: the processor only annotates existing
  detections with an attribute — it adds no new classes and emits no events.

## Build, test, install

Build via the CMake tree:

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-python-confidences-example
cmake --install . --component postprocessor-python-confidences-example
```

Run the colocated unit tests:

```shell
cd postprocessor-python-confidences-example
PYTHONPATH=../nxai-utilities python -m unittest test_postprocessor_python_confidences_example -v
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
