# Postprocessor C++ Objects Example

The starting template for a C++ post-processor that works on detected
objects. On every frame it adds one object of the class `cpp_objects_example`.

## What it does

`handle_message` receives an `ObjectsPostProcessorMessage` and pushes one
`ObjectData` into `msg.objects["cpp_objects_example"]`. The box is at
`x=0.1`, `y=0.1`, `width=0.3`, `height=0.3` (normalized to `[0,1]`), with
confidence `0.95` and the attribute `source`. The empty `new_track_id` tells
the VMS to create a new UUID for the track.

`main` opens the two IPC channels, signals READY on the input channel, and
calls `run_receive_loop`. The loop runs until SIGINT or SIGTERM, or until the
AI Manager sends the EXIT signal. The loop also ends on an unexpected IPC signal
or a failed send. In that case the process exits with code 1.

## Registration

Add the processor to `external_postprocessors.json` (see the colocated
`external_postprocessors.json` and `.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "postprocessor-cpp-objects-example",
    "Command": "/path/to/build/postprocessor-cpp-objects-example/postprocessor-cpp-objects-example",
    "Schedule": "OBJECTS",
    "LogFilepath": "/tmp/postprocess.cpp-objects.stdout.log",
    "Objects": [
        {
            "ID": "cpp_objects_example",
            "Name": "C++ Objects Example"
        }
    ],
    "Config": {
        "LogLevel": "DEBUG"
    }
}
```

Key points:

- `Schedule: "OBJECTS"` — the processor runs on inference-results messages.
- `Objects` declares the class `cpp_objects_example`. The `kInjectedClass`
  constant in `main.cpp` must match this ID. If the ID is not declared, the
  UI drops the box. The ID must also be unique across all registered
  processors.
- `Config` delivers launch-time configuration. The C++ code does not read it.

## Build and install

Build via the CMake tree:

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-cpp-objects-example
cmake --install . --component postprocessor-cpp-objects-example
```

On Windows, run `verify-windows.cmd` in the repository root. It builds the
C++ targets and runs the `ctest` suite. The example has no unit tests of its
own.

## Licence

Copyright 2026, Network Optix, All rights reserved.
