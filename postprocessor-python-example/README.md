# Postprocessor Python Example

The starting template for a Python post-processor. On every frame it adds
one object of the class `python_example` to the detections.

## What it does

`handle_message` adds an `ObjectMetadata` to `msg.objects["python_example"]`.
The box is at `x=0.1`, `y=0.1`, `width=0.3`, `height=0.3` (normalized to
`[0,1]`), with confidence `0.95` and the attribute `source`.

The object keeps one track for each device. On the first frame from a device,
the example sets `new_track_id` to a new UUID. On later frames from that
device, it sets `track_id` to the same UUID. The example does not share a
track between devices, because the plugin rejects a track ID that another
device created.

`run_message_loop` shows the message loop. `wait_for_message()` returns `None`
for a timeout and for the EXIT signal. The loop checks
`processor.exit_requested` to tell them apart, and returns on EXIT.

## Registration

Add the processor to `external_postprocessors.json` (see the colocated
`external_postprocessors.json` and `.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "postprocessor-python-example",
    "Command": "/path/to/build/postprocessor-python-example/postprocessor-python-example",
    "Schedule": "OBJECTS",
    "LogFilepath": "/tmp/postprocess.python-example.stdout.log",
    "Objects": [
        {
            "ID": "python_example",
            "Name": "Python Example"
        }
    ],
    "Config": {
        "LogLevel": "DEBUG"
    }
}
```

Key points:

- `Schedule: "OBJECTS"` — the processor runs on inference-results messages.
- `Objects` declares the class `python_example`. The `INJECTED_CLASS`
  constant in the code must match this ID. If the ID is not declared, the UI
  drops the box. The ID must also be unique across all registered processors.
- `Config` delivers launch-time configuration: only `LogLevel`.

## Build, test, install

Build via the CMake tree:

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-python-example
cmake --install . --component postprocessor-python-example
```

Run the colocated unit tests:

```shell
cd postprocessor-python-example
PYTHONPATH=../nxai-utilities python -m unittest test_postprocessor_python_example -v
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
