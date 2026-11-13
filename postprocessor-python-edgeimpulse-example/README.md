# Postprocessor Edge Impulse Image Uploader Example

A v5 post-processor that collects training data: it uploads frames to an Edge Impulse
project as samples, either on a fixed interval or whenever the frame holds a detection the
model was unsure about. Fire-and-forget — it returns nothing to the AI Manager.

## What it does

`handle_message` decides whether the frame is worth keeping, and queues it as JPEG bytes if
it is:

```python
if not (_sample_is_due() if auto_generator else _has_uncertain_detection(msg)):
    logger.debug("skipping sample")
    return

logger.debug("uploading sample")
samples_buffer.append(_jpeg_from_frame(msg.image_data))
```

On the threshold path, `_has_uncertain_detection` is true for any detection that scores
below `p_value` (`obj.confidence < p_value`). The comparison is `<` on purpose: the samples worth training on are the *uncertain* ones.
With `auto_generator` on, the score is ignored and one frame is queued every
`auto_generator_every_seconds` instead. Queued frames are uploaded in one batch once the
buffer reaches `samples_buffer_flush_size`.

## Behavior change from v4

- v4 ran on `Schedule: TENSOR` (`ReceiveInputTensor` in the old JSON) and hand-unpacked the
  model's output tensor with `struct` to reach every detection's score, reading every sixth
  element starting at index 4. A v5 TENSOR-schedule message carries no `ImageData` and can
  never have one, while this processor needs the score *and* the frame it uploads, so it
  runs on `Schedule: OBJECTS` with `ReceiveInputImage: true` and reads the score off the
  decoded objects' `confidence`. That value is bit-for-bit the one the tensor row carried,
  so `ReceiveBinaryData` would buy nothing and only carry a model-specific tensor layout
  into v5.
- One consequence: the score now arrives via the tracker rather than raw from the model, so
  a detection on a not-yet-warmed-up first frame may be missed. Harmless for an uploader
  whose job is "collect frames like this one".
- v4 received the frame via a second image-header message plus a raw shared-memory read
  (`SHMKey`, `nxai_communication_utils.SharedMemory`); v5 delivers it inline as the root
  `ImageData` block on the same message. That block is uint8 RGB, which the code
  encodes as a JPEG.
- v4 read its ini from `plugin.cloud-edgeimpulse.ini` while its own example file and README
  said `plugin.edgeimpulse.ini`. The code now reads the documented name.
- v4 logged to `plugin.cloud-edgeimpulse.log` next to the ini; the processor now logs to
  stdout, which the AI Manager captures at the entry's `LogFilepath`.
- `auto_generator` is read as a boolean. In v4 it came back as a bare string, so
  `auto_generator = False` was truthy and silently kept the timer running.

## Requirements

Any model will do. You also need an Edge Impulse account at https://edgeimpulse.com/ —
usage charges may apply, though the free tier is enough to try this out.

Create the configuration file as `etc/plugin.edgeimpulse.ini` in the directory one level
above the directory of the processor binary. The processor derives the path from the
location of the binary. For an NX Meta install on Linux, this is
`/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/etc/plugin.edgeimpulse.ini`.
NX Witness uses the root `/opt/networkoptix` instead. On Windows the root is the
mediaserver profile and the path is `nx_ai_manager\etc\plugin.edgeimpulse.ini` below it
(see [plugin.edgeimpulse.ini.example](plugin.edgeimpulse.ini.example)) and put your
project's API key in it — from Edge Impulse Studio, under "Dashboard > Keys", with no
quotes around it. The processor exits at startup while the key is unset.

```ini
[common]
debug_level=INFO
[edgeimpulse]
# Add your own project level Edge Impulse API key
api_key = ei_your_key_here
# Option autogenerate images every x seconds as an alternative to sending based on p_value
auto_generator = True
# If auto_generator True, every how many seconds upload an image?
auto_generator_every_seconds = 1
# Flush the buffer at this length
samples_buffer_flush_size = 20
# Send images below this value to EdgeImpulse. Can be between 0.0 and 1.0
p_value = 0.4
```

The settings stayed in the ini rather than moving to the registration entry's `Config`:
this is a behavior-preserving port. The one exception is *where* the ini lives — the
`Config` key `config_file` overrides the default path above, for a caller that installs
the processor somewhere it cannot predict.

## Registration

Add the processor to `external_postprocessors.json` (see the colocated
`external_postprocessors.json` and `.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "EdgeImpulse",
    "Command": "/path/to/build/postprocessor-python-edgeimpulse-example/postprocessor-python-edgeimpulse-example",
    "Schedule": "OBJECTS",
    "LogFilepath": "/tmp/postprocess.edgeimpulse.stdout.log",
    "ReceiveInputImage": true,
    "NoResponse": true,
    "Config": {
        "LogLevel": "DEBUG"
    }
}
```

Key points:

- `Schedule: "OBJECTS"` — the processor runs on inference-results messages, which is where
  both the detections' `confidence` and (with the flag below) the frame arrive.
- `ReceiveInputImage: true` — the engine attaches the frame as root `ImageData`; without it
  `msg.image_data` stays `None` and there is nothing to upload.
- `NoResponse: true` — the AI Manager does not wait for output. The processor must not
  reply on any path: the `NoResponse` path has no return channel at all, and a processor
  that answers anyway dies on its first frame.
- No `Objects` entry is needed: the processor adds no class of its own.
- `Config.config_file` (optional) — an absolute path to the ini, for an installation that
  cannot use the default location. Without it the processor reads
  `../etc/plugin.edgeimpulse.ini`, relative to the directory of the binary.

## Build, test, install

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-python-edgeimpulse-example
cmake --install . --component postprocessor-python-edgeimpulse-example
```

Run the colocated unit tests (the Edge Impulse boundary is mocked; no live call is made):

```shell
cd postprocessor-python-edgeimpulse-example
PYTHONPATH=../nxai-utilities python -m unittest test_postprocessor_python_edgeimpulse_example -v
```

## Output logging

```shell
tail -f /tmp/postprocess.edgeimpulse.stdout.log
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
