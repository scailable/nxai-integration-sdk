# Postprocessor Python Settings Example

A v5 post-processor that reads **forwarded GUI settings** and applies them as
per-object attributes. It is the reference consumer of the
`UserSettings` forwarding contract: a camera's plugin settings, edited in the NX
client, are delivered to the processor on every frame, and this example turns
two of them into an attribute on every detected object.

## What it does

`handle_message` reads two forwarded settings, `attributeName` and
`attributeValue`, and writes that pair as an attribute onto every object in
every class:

```python
attribute_name = msg.user_settings["attributeName"]
attribute_value = msg.user_settings["attributeValue"]

for objects in msg.objects.values():
    for obj in objects:
        obj.attributes[attribute_name] = attribute_value
```

With the default control values (`attributeName = "reviewed"`,
`attributeValue = "true"`), every object comes back carrying
`attributes["reviewed"] == "true"`.

A single guard — both keys must be present — means the plugin **no-ops cleanly**
when the engine forwards no settings, so it runs safely on an engine that does
not yet implement forwarding.

## The `UserSettings` contract

The AI Manager forwards GUI settings as a single root key `UserSettings` on the
objects message — a sibling of `Timestamp` / `DeviceID` / `DeviceName` /
`Objects` / `Events` — holding a flat `{settingName: stringValue}` map. The
Python SDK parses it into `ObjectsPostProcessorMessage.user_settings`
(parse-only: settings are inbound, never echoed back).

Control names are declared **without** an `externalprocessor.` prefix, so a
control name equals the forwarded `UserSettings` key verbatim. Settings are
inbound only: the processor never echoes them back in its response.

## Registration

Add the processor to `external_postprocessors.json` (see
`external_postprocessors.json` in this directory and
`.claude/docs/external_postprocessors_json.md` for the full field reference).
The key parts:

```json
{
    "Name": "Settings-Example",
    "Command": "/path/to/postprocessor-python-settings-example",
    "Schedule": "OBJECTS",
    "LogFilepath": "/tmp/postprocess.settings.stdout.log",
    "Config": { "LogLevel": "DEBUG" },
    "UserSettings": [
        { "type": "TextField", "name": "attributeName",  "caption": "Attribute Name",  "defaultValue": "reviewed" },
        { "type": "TextField", "name": "attributeValue", "caption": "Attribute Value", "defaultValue": "true" }
    ]
}
```

- `Schedule: "OBJECTS"` — the processor runs on the inference-results (objects)
  message.
- `Config` delivers launch-time configuration (here, the log level); v5 plugins
  are configured through `Config`, not a `plugin.settings.ini` file.
- `UserSettings` is the inline Settings Model whose two `TextField` controls
  appear in the NX plugin UI (v5 key, renamed from the stale `Settings`). The
  full control grammar is documented in `settings_model.md`.
- The plugin declares **no** `Objects` and **no** `Events`: it only appends
  attributes to existing objects, introducing no new classes or event IDs.

## Build, test, install

Build via the CMake tree (modeled on `postprocessor-python-example`):

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-python-settings-example
cmake --install . --component postprocessor-python-settings-example
```

Run the colocated unit test from the `postprocessor-python-settings-example/`
directory. The `msgpack` module must be installed:

```shell
python -m unittest test_postprocessor_python_settings_example
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
