# Postprocessor Python Objects Sender

A v5 post-processor that geo-references detections. It maps the center of each
object's bounding box to a latitude and a longitude, attaches them as a
`Coordinates` attribute, and sends the objects as GeoJSON to an HTTP-to-MQTT
bridge.

## What it does

The operator marks up to 10 reference points on the camera view and gives each
point a latitude and a longitude. `handle_message` reads the points from
`msg.user_settings` on every frame:

- With no point, the message goes back unchanged.
- With fewer than 4 points, the processor has no homography. The message goes
  back unchanged, and no request is sent.
- With 4 or more points, the processor fits a homography for the device with
  `cv2.findHomography` (RANSAC). It fits again only when the points of that
  device change. Each `DeviceID` has its own homography.

For each object in `msg.objects[cls]`, the processor maps the bbox center
`(x + width / 2, y + height / 2)` and sets:

```python
obj.attributes["Coordinates"] = f"{lat}, {lon}"
```

Then it POSTs one GeoJSON `Feature` per object to `<server_address>/send`:

```json
{
    "topic": "<topic_path>",
    "QoS": 1,
    "message": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": ["<lat>", "<lon>"]},
            "properties": {
                "object_type": "<class name>",
                "object_id": "<track UUID>",
                "timestamp": "<message Timestamp>"
            }
        }
    ]
}
```

The request has a 10 s timeout. A connection error, a timeout or an HTTP error
is logged, and the message is still returned.

## The UserSettings contract

- `point<N>.figure` (`BoxFigure`, N = 0..9): the reference point is the center
  of the box. The engine sends the value as a JSON string. A point without a
  figure is skipped.
- `point<N>.latitude`, `point<N>.longitude` (`DoubleSpinBox`): degrees times
  1000, because the spin box keeps three decimals. The processor divides by 1000.
- `server_address` (`TextField`, default `http://127.0.0.1:5000`): the base URL
  of the bridge.
- `topic_path` (`TextField`, default `topic1`): the MQTT topic.

The points are declared with a `Repeater` over 10 items. The server expands the
`Repeater` into `point0.*` .. `point9.*`.

## Behavior change from v4

- **Normalized coordinates.** v4 scaled the figure points and the bbox centers
  to pixels with the frame `Width` and `Height`. The v5 message has no frame
  size and its bounding boxes are normalized. The homography absorbs the
  scaling, so `(lat, lon)` does not change.
- **The homography is cached per device.** v4 recomputed it on every frame.
- **A gap in the point numbers is allowed.** v4 stopped with an `IndexError`
  when, for example, point 0 was empty and point 1 was set.

## Registration

Add the processor to `external_postprocessors.json` (see the colocated
`external_postprocessors.json` and `.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "Python-Object-Sender-Postprocessor",
    "Command": "/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/postprocessors/postprocessor-python-objects-sender",
    "Schedule": "OBJECTS",
    "LogFilepath": "/tmp/postprocess.objects-sender.stdout.log",
    "Config": {
        "LogLevel": "DEBUG"
    },
    "UserSettings": [
        {
            "type": "Repeater",
            "count": 10,
            "startIndex": 0,
            "template": {
                "type": "GroupBox",
                "caption": "Point #",
                "filledCheckItems": [
                    "point#.figure"
                ],
                "items": [
                    {
                        "type": "BoxFigure",
                        "name": "point#.figure",
                        "caption": "Point #",
                        "description": "Point # on the camera"
                    },
                    {
                        "type": "DoubleSpinBox",
                        "name": "point#.latitude",
                        "caption": "Latitude",
                        "description": "Latitude of point #",
                        "defaultValue": 33878.754,
                        "minValue": -90000.000,
                        "maxValue": 90000.000
                    },
                    {
                        "type": "DoubleSpinBox",
                        "name": "point#.longitude",
                        "caption": "Longitude",
                        "description": "Longitude of point #",
                        "defaultValue": -84450.569,
                        "minValue": -180000.000,
                        "maxValue": 180000.000
                    }
                ]
            }
        },
        {
            "type": "TextField",
            "name": "server_address",
            "caption": "MQTT plugin address",
            "defaultValue": "http://127.0.0.1:5000"
        },
        {
            "type": "TextField",
            "name": "topic_path",
            "caption": "MQTT topic",
            "defaultValue": "topic1"
        }
    ]
}
```

Key points:

- `Schedule: "OBJECTS"`: the processor runs on inference-results messages.
- `Command` is the NX Meta install path. For NX Witness, use
  `/opt/networkoptix/mediaserver/var/nx_ai_manager/postprocessors/`.
- `UserSettings` declares the controls of "The UserSettings contract" above.
  The processor falls back to the same defaults when a key is absent.
- No `ReceiveInputImage`: the processor needs no frame size.
- No `Objects` or `Events`: `Coordinates` attaches to existing objects.

## Build, test, install

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-python-objects-sender
cmake --install . --component postprocessor-python-objects-sender
```

Run the colocated unit tests. They mock `requests`, so they need no bridge:

```shell
cd postprocessor-python-objects-sender
PYTHONPATH=../nxai-utilities python -m unittest test_postprocessor_python_objects_sender -v
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
