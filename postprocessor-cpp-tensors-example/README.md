# Postprocessor C++ Tensors Example

The starting template for a C++ post-processor that works on raw model
output tensors. It replaces the tensors of every frame with one box tensor
of the class `cpp_tensors_example`.

## What it does

`handle_message` receives a `TensorsMessage`. It prints the name of each
incoming tensor, then clears `msg.tensors`. It then adds one float tensor
named `bboxes-format:xyxysc;0:cpp_tensors_example` with shape `{1, 6}` and
the values `100, 100, 200, 200, 1, 0`. The tensor name carries the box
format (`xyxysc`) and the class name.

The first four values are the box corners `x1, y1, x2, y2` in pixels of the
model's input image. The engine maps them to the frame. The last two values
are the score and the class index.

`main` opens the two IPC channels, signals READY on the input channel, and
calls `run_receive_loop`. The loop runs until SIGINT or SIGTERM, or until the
AI Manager sends the EXIT signal. The loop also ends on an unexpected IPC signal
or a failed send. In that case the process exits with code 1.

## Registration

Add the processor to `external_postprocessors.json` (see the colocated
`external_postprocessors.json` and `.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "postprocessor-cpp-tensors-example",
    "Command": "/path/to/build/postprocessor-cpp-tensors-example/postprocessor-cpp-tensors-example",
    "Schedule": "TENSOR",
    "LogFilepath": "/tmp/postprocess.cpp-tensors.stdout.log",
    "Objects": [
        {
            "ID": "cpp_tensors_example",
            "Name": "C++ Tensors Example"
        }
    ],
    "Config": {
        "LogLevel": "DEBUG"
    }
}
```

Key points:

- `Schedule: "TENSOR"` — the processor runs on the raw model output tensors,
  not on detected objects.
- `Objects` declares the class `cpp_tensors_example`. The class in the tensor
  name must match this ID. If the ID is not declared, the UI drops the box.
  The ID must also be unique across all registered processors.
- `Config` delivers launch-time configuration. The C++ code does not read it.

## Build and install

Build via the CMake tree:

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-cpp-tensors-example
cmake --install . --component postprocessor-cpp-tensors-example
```

On Windows, run `verify-windows.cmd` in the repository root. It builds the
C++ targets and runs the `ctest` suite. The example has no unit tests of its
own.

## Licence

Copyright 2026, Network Optix, All rights reserved.
