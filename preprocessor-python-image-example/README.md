# Preprocessor Python Image Example

A v5 pre-processor that mirrors the input image horizontally, at full
resolution, before it reaches the rest of the AI Manager's own
preprocessing pipeline.

## What it does

`handle_message` reads `msg.image` (an `ImageData` with `height`, `width`,
`layout`, and raw `bytes`) and, when the forwarded `mirrorimage` UserSettings
value is enabled (falling back to `DEFAULT_MIRROR_IMAGE` when the engine
forwards no settings), replaces it with a true horizontal mirror at
unchanged width and height.

The payload is **uint8 RGB at source resolution**, one byte per channel. A
live v5 engine delivers `CHW` (`channels` planes of `height` rows); Pillow
wants interleaved `HWC`, so the buffer is reshaped and transposed into that
order with numpy, mirrored with `ImageOps.mirror`, and transposed back for
`CHW`:

```python
pixels = np.frombuffer(image.bytes, dtype=np.uint8)
chw = image.layout == nxai_postprocessor_sdk.ImageOrder.CHW
if chw:
    pixels = pixels.reshape(_CHANNELS, image.height, image.width).transpose(1, 2, 0)
else:
    pixels = pixels.reshape(image.height, image.width, _CHANNELS)

mirrored = np.asarray(ImageOps.mirror(Image.fromarray(pixels, mode="RGB")))
if chw:
    mirrored = mirrored.transpose(2, 0, 1)

image.bytes = mirrored.tobytes()
```

The reshape/transpose is a numpy view, not a copy, and Pillow's mirror runs in
C, so this stays well inside the engine's IPC window at full
resolution without any hand-rolled byte slicing.

`Data` length always equals `Height * Width * channels`, matching the v5
wire invariant (`.claude/docs/input_from_ai_manager.md`); `width` and
`height` are unchanged. A live v5 engine delivers `CHW`, so mirroring only
`HWC` made the setting a no-op on real frames; both layouts are
handled now.

When disabled, `msg` passes through unchanged. The engine then resizes,
normalizes, and casts whatever image comes back to the model's input spec.

## Behavior change from v4

- **The mirror is a true horizontal mirror**, not v4's shifted one. v4's
  source-column index was off by one (`width - w_key` instead of
  `width - 1 - w_key`), which wrapped output column 0 across row boundaries.
  v5 fixes this rather than porting the quirk byte-for-byte — a mirror that
  reads across row boundaries is a defect, not a demonstrable feature.
- **`mirrorimage` is a live UserSettings value again**, read fresh from the
  message every frame. The engine forwards preprocessor settings the same way it
  forwards them to postprocessors: a root `UserSettings` map on the
  IMAGE-schedule message. In v4 only launch-time `Config` carried them.

## Registration

Add the processor to `external_preprocessors.json` (see the colocated
`external_preprocessors.json` and `.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "Image-Mirror-Example",
    "Command": "/path/to/preprocessor-python-image-example",
    "Schedule": "IMAGE",
    "LogFilepath": "/tmp/preprocess.image-example.stdout.log",
    "Config": {
        "LogLevel": "DEBUG"
    },
    "UserSettings": [
        {
            "type": "CheckBox",
            "name": "mirrorimage",
            "caption": "Mirror Image",
            "description": "Mirror the input image before the rest of the pipeline sees it",
            "defaultValue": true
        }
    ]
}
```

Key points:

- `Schedule: "IMAGE"` — the preprocessor runs before any of the engine's own
  preprocessing (resize/normalize/cast), and receives the raw input frame.
- `UserSettings` declares `mirrorimage` as a live control; the value is
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
cmake --build . --target preprocessor-python-image-example
cmake --install . --component preprocessor-python-image-example
```

Run the colocated unit tests:

```shell
cd preprocessor-python-image-example
PYTHONPATH=../nxai-utilities python -m unittest test_preprocessor_python_image_example -v
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
