# Postprocessor Cloud Inference Example

A v5 post-processor that offloads classification to a cloud service: it crops each
detected `face` out of the frame, sends the crop to AWS Rekognition, and attaches the
returned description ("A male in his 30s feeling happy.") to the object.

## What it does

`handle_message` scales each `face` object's normalized box to pixels, crops the frame
delivered at `msg.image_data`, writes the crop to the configured `image_path`, and hands
that file to `aws_utils.classify_faces`:

```python
cropped_image = image.crop(_pixel_box(face, image.width, image.height))
cropped_image.save(image_path)
face.attributes["description"] = classify_faces(image_path, logger)
```

At most two faces per frame are classified, to avoid affecting the FPS rate.

## Behavior change from v4

- v4 received the frame via a second image-header message plus a raw shared-memory read
  (`SHMKEY`, `nxai_communication_utils.SharedMemory`); v5 delivers it inline as the root
  `ImageData` block on the same message, gated by `ReceiveInputImage`.
- v4 moved a classified face into a **new class named after the description**
  (`BBoxes_xyxy[description]`). The description is free text, so the class set was
  unbounded, and v5 requires every class to be pre-declared under `Objects` in the
  registration JSON. The description is now a `description` **attribute** on the existing
  `face` object.
- Consequently the classified face is no longer removed from the `face` class — v4 deleted
  it once it had been re-homed under the description class.
- v4 boxes were pixel xyxy; v5 boxes are normalized to `[0,1]` and are scaled by the frame
  dimensions before the crop.
- v4's `plugin.cloud-inference.ini` is gone: `image_path` and the log level now come from
  the registration JSON's `Config`. The ini's `aws_access_key_id` / `aws_secret_access_key`
  / `region_name` keys were already dead in v4 — `aws_utils.create_session()` calls a bare
  `boto3.Session()` and takes credentials from the default AWS chain — so they were not
  ported.

## Requirements

Assign the Face locator model to the camera device, and configure AWS credentials the way
`boto3` expects them, in `~/.aws/credentials`:

```ini
[default]
aws_access_key_id=access_key
aws_secret_access_key=secret_access_key
```

and a region in `~/.aws/config`:

```ini
[default]
region=us-east-1
```

The account needs `AmazonRekognitionReadOnlyAccess`. Amazon usage charges may apply.

## Registration

Add the processor to `external_postprocessors.json` (see the colocated
`external_postprocessors.json` and `.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "Cloud-Inference",
    "Command": "/path/to/build/postprocessor-cloud-inference-example/postprocessor-cloud-inference-example",
    "Schedule": "OBJECTS",
    "LogFilepath": "/tmp/postprocess.cloud-inference.stdout.log",
    "ReceiveInputImage": true,
    "Config": {
        "LogLevel": "DEBUG",
        "image_path": "/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/postprocessors/face.png"
    }
}
```

Key points:

- `Schedule: "OBJECTS"` — the processor runs on inference-results messages.
- `ReceiveInputImage: true` — the engine attaches the frame as root `ImageData`; without it
  `msg.image_data` stays `None` and there is nothing to crop.
- No `Objects` entry is needed: the processor annotates the model's own `face` class and
  introduces no class of its own.
- `Config.image_path` is where the cropped face is written before being uploaded. It is a
  launch-time path, not a per-camera setting, so it belongs in `Config`, not `UserSettings`.

## Build, test, install

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-cloud-inference-example
cmake --install . --component postprocessor-cloud-inference-example
```

Run the colocated unit tests (the AWS boundary is mocked; no live call is made):

```shell
cd postprocessor-cloud-inference-example
PYTHONPATH=../nxai-utilities python -m unittest test_postprocessor_cloud_inference_example -v
```

## Output logging

```shell
tail -f /tmp/postprocess.cloud-inference.stdout.log
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
