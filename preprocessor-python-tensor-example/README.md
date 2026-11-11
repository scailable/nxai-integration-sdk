# Preprocessor Python Tensor Example

A v5 pre-processor that overrides the NMS-sensitivity tensor before the frame
reaches the model.

## What it does

`handle_message` looks up the `"nms_sensitivity-"` tensor on the incoming
`TensorMessage` and, if present, replaces its data with the forwarded
`nmsoverride` UserSettings value, falling back to `DEFAULT_NMS_OVERRIDE` when
the engine forwards no settings (a native-byte-order `float32`, matching both
the tensor's existing `data_type` and the engine's own `memcpy` of tensor
payloads):

```python
tensor = msg.tensors.get(NMS_TENSOR_NAME)
if tensor is not None:
    if "nmsoverride" in msg.user_settings:
        nms_override = float(msg.user_settings["nmsoverride"])
    else:
        nms_override = DEFAULT_NMS_OVERRIDE

    tensor.data = struct.pack("f", nms_override)
```

All other tensors pass through unchanged. When the tensor is absent, the
processor no-ops.

## Behavior change from v4

- **`nmsoverride` is a live UserSettings value again**, read fresh from the
  message every frame. The engine forwards preprocessor settings the same way it
  forwards them to postprocessors: a root `UserSettings` map on the
  TENSOR-schedule message. In v4 only launch-time `Config` carried them.

## Registration

Add the processor to `external_preprocessors.json` (see the colocated
`external_preprocessors.json` and `.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "Tensor-NMS-Override",
    "Command": "/path/to/preprocessor-python-tensor-example",
    "Schedule": "TENSOR",
    "LogFilepath": "/tmp/preprocess.tensor-example.stdout.log",
    "Config": {
        "LogLevel": "DEBUG"
    },
    "UserSettings": [
        {
            "type": "DoubleSpinBox",
            "name": "nmsoverride",
            "caption": "NMS Override",
            "description": "Fixed value the NMS-sensitivity tensor is overridden with",
            "defaultValue": 0.8,
            "minValue": 0.0,
            "maxValue": 1.0
        }
    ]
}
```

Key points:

- `Schedule: "TENSOR"` — the preprocessor runs after the frame has been
  converted into the model's input tensor, and may add/remove/rename tensors.
- `UserSettings` declares `nmsoverride` as a live control; the value is
  delivered every frame in the root `UserSettings` map (bare/unprefixed name).
- `Config` only carries `LogLevel`, the one launch-time key the SDK itself
  consumes.
- No `Objects` or `Events`: a preprocessor mutates model *input*, not the
  objects/events response.

## Build, test, install

Build via the CMake tree:

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target preprocessor-python-tensor-example
cmake --install . --component preprocessor-python-tensor-example
```

Run the colocated unit tests:

```shell
cd preprocessor-python-tensor-example
PYTHONPATH=../nxai-utilities python -m unittest test_preprocessor_python_tensor_example -v
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
