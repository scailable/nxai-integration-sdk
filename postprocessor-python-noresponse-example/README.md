# Postprocessor Python No-Response Example

A v5 post-processor that receives detected objects and events, logs them, and never
sends a response. It demonstrates the fire-and-forget shape: a postprocessor that only
performs a side effect (here, logging) and relies on `"NoResponse": true` in its
registration entry so the AI Manager does not wait for output.

## What it does

`handle_message` logs `msg.device_id`, the number of object classes in `msg.objects`,
and the number of events in `msg.events`, then returns nothing. The main loop calls it
and loops back to `wait_for_message` — it never calls `write_objects_message` or
`send_response`.

## NoResponse

When the registration entry sets `"NoResponse": true`, the AI Manager sends each
message and does not wait for a response. The processor has no return channel to the
AI Manager. Any reply it writes has no effect.

## Registration

```json
[
    {
        "Name": "NoResponse-Example",
        "Command": "/path/to/build/postprocessor-python-noresponse-example/postprocessor-python-noresponse-example",
        "Schedule": "OBJECTS",
        "LogFilepath": "/tmp/postprocess.noresponse-example.stdout.log",
        "NoResponse": true,
        "Config": {
            "LogLevel": "DEBUG"
        }
    }
]
```

Key points:

- `Schedule: "OBJECTS"` — the processor runs on inference-results messages.
- `NoResponse: true` — tells the AI Manager not to wait for a reply after sending the
  message. See "NoResponse" above.
- `Config` delivers launch-time configuration. The script reads only `LogLevel`.
- No `Objects` and no `Events`: the processor doesn't declare or emit either.

## Build, test, install

Build via the CMake tree:

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-python-noresponse-example
cmake --install . --component postprocessor-python-noresponse-example
```

Run the colocated unit tests:

```shell
cd postprocessor-python-noresponse-example
PYTHONPATH=../nxai-utilities python -m unittest test_postprocessor_python_noresponse_example -v
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
