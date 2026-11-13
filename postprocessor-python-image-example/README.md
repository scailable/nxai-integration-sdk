# Postprocessor Python Image Example

A v5 post-processor that reads the input frame, decodes it with Pillow, injects a
"python_image_example" detection recording whether it decoded cleanly, and serves the
decoded frame as a live MJPEG preview you can watch in a browser.

## What it does

`handle_message` decodes the raw frame delivered at `msg.image_data` into a Pillow
image, publishes it to an in-process MJPEG stream, and injects a single detection
recording the decode result:

```python
obj = nxai_postprocessor_sdk.ObjectMetadata()
obj.x, obj.y, obj.width, obj.height = 0.1, 0.1, 0.1, 0.1
obj.confidence = 1.0
obj.attributes["ImageDecodeOk"] = str(decode_error is None)
if decode_error is not None:
    obj.attributes["ImageDecodeError"] = decode_error
msg.objects["python_image_example"] = [obj]
```

If the decode fails, the object also carries the error message in `ImageDecodeError`.
The object keeps one track UUID for each device: the first frame of a device creates
the track with `new_track_id`, and later frames continue it with `track_id`.

`msg.image_data` carries raw pixel data, not an encoded image, so there's nothing to
"parse" that would fail the way a corrupt JPEG does. This example reads it as
**uint8 RGB** (1 byte per channel).

What `_decode_frame` actually catches is a buffer whose size doesn't match
`height x width x 3` elements — a truncated or misaligned frame that a plain byte-sum
check (the v4-ported version this example replaces) accepted silently, since a sum is
identical whether the bytes that are there form a valid frame or not. Corruption that
keeps the right size but scrambles pixels (swapped channels, shifted rows) isn't wrong
by any number; that's what the preview stream is for — watch it and the visual
artifact is immediately obvious.

### Watching the preview stream

On startup the process logs a URL:

```
Image preview stream: http://127.0.0.1:<port>/ (open in a browser to watch frames live)
```

Open it in any browser. The response is a `multipart/x-mixed-replace` MJPEG stream —
every mainstream browser renders it as a live-updating image natively, with no plugin
and no polling. The latest decoded frame is kept in memory and JPEG-encoded on
publish; nothing is ever written to disk, which is what keeps this viable at
frame rate (writing one file per frame would mean thousands of files a minute and no
way to just watch the feed update). The implementation is stdlib `http.server` plus
Pillow, so it behaves identically on Windows and Linux. It binds to `127.0.0.1` only —
loopback, not exposed on the network — and the port is OS-assigned by default; pin it
with the `stream_port` Config key for a stable URL across restarts.

## Behavior change from v4

- v4 received the frame via a second image-header message plus a
  shared-memory read (`SHMKEY`); v5 delivers it inline as the root `ImageData`
  block on the same message, gated by `ReceiveInputImage` — no second message,
  no SHM.
- v4 injected a pixel xyxy box (`BBoxes_xyxy["test"] = [100, 100, 200, 200]`)
  and a parallel `Counts["ImageBytesCumulative"]`; v5 objects are
  frame-relative, so the box is normalized xywh and the decode result is a
  per-object `attributes` entry instead (v5 has no `Counts`).

## Registration

Add the processor to `external_postprocessors.json` (see the colocated
`external_postprocessors.json` and `.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "Image-Example",
    "Command": "/path/to/build/postprocessor-python-image-example/postprocessor-python-image-example",
    "Schedule": "OBJECTS",
    "LogFilepath": "/tmp/postprocess.image-example.stdout.log",
    "ReceiveInputImage": true,
    "Objects": [
        {
            "ID": "python_image_example",
            "Name": "Python Image Example"
        }
    ],
    "Config": {
        "LogLevel": "DEBUG"
    }
}
```

Key points:

- `Schedule: "OBJECTS"` — the processor runs on inference-results messages.
- `ReceiveInputImage: true` — the engine attaches the frame as root
  `ImageData` on every message; without it `msg.image_data` stays `None`.
- `Objects` declares the injected `python_image_example` class — an undeclared
  class is silently dropped by the UI.
- `Config` delivers launch-time configuration: `LogLevel`, plus the optional
  `stream_port` to pin the preview server to a fixed port instead of an
  OS-assigned one.

## Build, test, install

Build via the CMake tree:

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-python-image-example
cmake --install . --component postprocessor-python-image-example
```

Run the colocated unit tests:

```shell
cd postprocessor-python-image-example
PYTHONPATH=../nxai-utilities python -m unittest test_postprocessor_python_image_example -v
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
