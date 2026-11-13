# `external_postprocessors.json` — Reference

This file is how you tell the NX AI Manager that an external post-processor exists, how to start it, and how to connect to it. Without an entry here, your compiled post-processor is invisible to the AI Manager.

An analogous `external_preprocessors.json` does the same for preprocessors. Most fields are shared; differences are called out at the end of this document.

---

## Where the file lives

Paths below are written relative to `VMS_BASE`, the install root of the VMS product:

| Product | `VMS_BASE` | Service |
|---------|------------|---------|
| NX Meta | `/opt/networkoptix-metavms` | `networkoptix-metavms-mediaserver` |
| NX Witness | `/opt/networkoptix` | `networkoptix-mediaserver` |
| Windows (service profile) | `C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix MetaVMS Media Server` | — |

**Linux**

```
$VMS_BASE/mediaserver/var/nx_ai_manager/postprocessors/external_postprocessors.json
```

**Windows** (this tree has no `mediaserver\var` level)

```
$VMS_BASE\nx_ai_manager\postprocessors\external_postprocessors.json
```

(Preprocessors live in a sibling `preprocessors/` directory with a file named `external_preprocessors.json`.)

The file must be readable by the mediaserver. After creating or editing it you also need to make sure the `postprocessors/` directory and the binaries inside it are executable:

```shell
sudo chmod -R a+x $VMS_BASE/mediaserver/var/nx_ai_manager/postprocessors/
```

Changes take effect only after restarting the mediaserver:

```shell
sudo service networkoptix-metavms-mediaserver restart
```

---

## File shape

On v5 the file is a **bare array** of processor definitions — no wrapper object:

```json
[
    { /* processor 1 */ },
    { /* processor 2 */ }
]
```

Each element in the array describes one external post-processor. Every registration file in this repo uses this shape; the v4 form wrapped the array in a root object keyed `externalPostprocessors`.

---

## Per-processor fields

The following table lists every field that appears in the SDK's example definitions, grouped by purpose. "Required" means the AI Manager needs it to route data; "optional" fields default to `false` for booleans and to absent/empty otherwise.

### Identity and transport

- **`Name`** (string, required) — A human-readable identifier shown in the NX Cloud Pipelines UI. This is how the user selects the processor in a pipeline. It does not need to match the binary filename.
- **`Schedule`** (string, optional for postprocessors) — Controls at which stage the processor is invoked. Postprocessors use `"OBJECTS"` (the default when `Schedule` is absent: runs on the inference-results message) or `"TENSOR"` (see `postprocessor-cpp-tensors-example/`). Every preprocessor example sets `Schedule` explicitly, so set it on a preprocessor too. Preprocessors use `"IMAGE"` (raw frame, before resizing/normalization) or `"TENSOR"` (after the frame has been converted into the model's input tensor) — see "Preprocessor equivalent" below.
- **`Command`** (string) — Absolute path to the post-processor executable. The AI Manager starts this program when the mediaserver boots and terminates it on shutdown. Each entry needs `Command` or `Port`. The AI Manager skips an entry that has neither and logs the warning "Either 'Command' or 'Port' needs to be defined".
- **`Port`** (integer) — The AI Manager launches nothing. It connects to `127.0.0.1:<Port>` and sends the data to the process that listens there. Use it during development (see "The development pattern" below). If an entry has both `Command` and `Port`, `Port` wins and the AI Manager gives no warning.
- **`TimeoutMs`** (unsigned integer, optional) — How long the AI Manager waits for a response from the processor, in milliseconds. It applies to preprocessors and postprocessors.
- **`SocketPath`** — **v4 only, do not use on v5.** The v5 engine has no socket transport and passes no socket path in `argv`: the `argv` is `<input_sync> <output_sync> [--key value ...] --ProcessorName <Name>`. There is no positional name. The `--key value` pairs come from the `Config` entries, in the JSON order. The SDK parses all of that for you (`initialize_from_args()` / `load_config()`).
- **`LogFilepath`** (string, optional) — Absolute path the AI Manager redirects the processor's stdout/stderr to. Without it a `Command`-started processor's output goes nowhere and startup failures are invisible.
- **`Config`** (object of string→string, optional) — Launch-time key/value pairs. The AI Manager skips an entry if its key or its value is not a string. They are delivered to the processor as `--key value` argv pairs and read via the SDK's `load_config()`. Unlike `UserSettings` these are fixed at process start, not per-frame. `LogLevel` is the one key the SDK itself consumes; anything else is the processor's own. Preprocessors have the same two channels as postprocessors — `Config` for launch-time keys, `UserSettings` for live ones (see "Preprocessor equivalent" below).

### What the processor receives

These booleans extend the default inference-results message with additional data. All default to `false`.

- **`ReceiveInputImage`** (bool) — When `true`, the AI Manager attaches the raw input frame as a root `ImageData` block on the same inference-results message; without it, `msg.image_data` stays `None`. This replaces v4's second image-header message plus shared-memory read (`SHMKEY`) — v5 has no second message and no SHM. See `postprocessor-python-image-example/` for the worked example.
- **`ReceiveBinaryData`** (bool) — When `true`, the inference-results message includes a `Tensors` array with the raw model output tensors (name, dtype, binary blob). Used when the processor needs the model's raw logits or embeddings. See `postprocessor-cpp-tensors-example/` and `postprocessor-python-anpr-example/`.
- **`ReceiveConfidenceData`** — **v4 only, do not use on v5.** The v5 engine reads exactly two of these booleans, `ReceiveInputImage` and `ReceiveBinaryData`; a `ReceiveConfidenceData` entry is ignored. There is nothing to opt into: every object arrives with its own `confidence` on `ObjectMetadata`, in place of v4's parallel `ObjectsMetaData.<class>.Confidences` list. See `postprocessor-python-confidences-example/`.

### Response behavior

- **`NoResponse`** (bool, default `false`) — When `true`, the AI Manager does not wait for a response. The processor is expected to consume the message and stay silent. Useful for fire-and-forget side effects like uploading to an external system. See `postprocessor-python-noresponse-example/` and `postprocessor-python-edgeimpulse-example/`.
- **`RunLast`** (bool, default `false`) — When `true`, this processor runs after all other post-processors in the pipeline. Use when the processor depends on outputs produced by the others.

### UI declarations

These fields let the processor declare things that must exist in the UI before the processor tries to use them. They affect the plugin settings dialog and how outputs are rendered, but they don't change the data the processor receives on the IPC channel.

- **`Objects`** (array of `{ID, Name}`, optional) — Declares new class names the processor may add to `msg.objects` in its response (v4's `BBoxes_xyxy`/`ObjectsMetaData` root shape is gone). Without this, any new classes the processor introduces will be dropped by the UI. Example from `postprocessor-python-example/external_postprocessors.json`:

  ```json
  "Objects": [
      { "ID": "python_example", "Name": "Python Example" }
  ]
  ```

  Declaring the class is necessary but not sufficient — it also has to be **enabled in the device's *Class Visualisation* setting**. The plugin renders a class only when its name is present *and* `true` in the pipeline's `active_visualisations` map, and that map is built from the device agent's stored `visualise_classes_<pipeline_id>` value. The gate itself is intended behaviour: it is the `Class Visualisation` group box in the plugin settings dialog, a multi-selection combo box whose selectable range is the model's own output classes plus every `Objects[].ID` of the post-processor assigned to that pipeline (each shown under its declared `Name`). Nothing needs to be done to opt in: a declared class the stored selection has never heard of defaults to visible, so a newly assigned post-processor's boxes appear as soon as the engine picks the processor up. The reverse does not work yet: because a stored selection only ever lists the classes that *are* selected, deselecting a post-processor's own class leaves it absent from the stored value, and the engine then re-defaults it to visible. Filter unwanted classes in the processor instead of in that combo box.

- **`Events`** (array of `{ID, Name}`, optional) — Declares event IDs the processor may emit in the `Events` field of its response. Required for any event to surface in the NX event system. Declare the id exactly as the processor emits it: on v5 (OAAX engine) the AI Manager copies it into the DeviceAgent manifest unprefixed, unlike object classes (see `output_to_ai_manager.md`, "Events"). Example from `postprocessor-python-anpr-example/external_postprocessors.json`:

  ```json
  "Events": [
      { "ID": "anpr.plate_recognized", "Name": "License plate recognized" }
  ]
  ```

- **`UserSettings`** (array, optional) — An inline Settings Model (the v5 key, renamed from the v4 `Settings`). Each entry is a UI control (TextField, SpinBox, CheckBox, ComboBox, PolygonFigure, etc.) that will appear in the NX plugin settings dialog for this processor. The user's selected values are delivered to the processor every frame in the root `UserSettings` map on the inference-results message (see `input_from_ai_manager.md`). Control names are declared **without** the old `externalprocessor.` prefix — a control's `name` is the verbatim key the processor reads from the forwarded `UserSettings` map. Example:

  ```json
  "UserSettings": [
      {
          "type": "TextField",
          "name": "attributeName",
          "caption": "Attribute Name",
          "description": "The name of the example attribute",
          "defaultValue": "Key"
      }
  ]
  ```

  The full list of control types and their JSON shape is documented in `postprocessor-python-settings-example/settings_model.md`. It includes TextField, PasswordField, SpinBox, DoubleSpinBox, ComboBox, RadioButtonGroup, CheckBox, CheckBoxGroup, SwitchButton, Link, Banner, Placeholder, PolygonFigure, BoxFigure, LineFigure, ObjectSizeConstraints, Button, and structural elements (Settings, Section, GroupBox, Separator, Repeater).

---

## The development pattern: use `Port`

For fast iteration, replace `Command` with `Port`:

```json
{
    "Name": "My-Postprocessor",
    "Schedule": "OBJECTS",
    "Port": 5010
}
```

The AI Manager does not start your binary. Instead, it connects to `127.0.0.1:5010` and sends data each frame. You start your processor manually from a terminal with `--port 5010`, iterate as needed, kill it, start it again. Once you are happy, replace `Port` with `Command` and let the AI Manager manage the lifecycle. An entry with neither field is skipped. This pattern is described in the root `README.md` under "Development".

---

## Worked example

The following file registers five post-processors and exercises most of the optional fields. It is illustrative: the `Name`, `Command` and `LogFilepath` values are adapted to one install layout. The `Objects`, `Events` and `Receive*` fields match the `external_postprocessors.json` file of each example.

```json
[
    {
        "Name": "EI-Upload-Postprocessor",
        "Command": "$VMS_BASE/mediaserver/var/nx_ai_manager/postprocessors/postprocessor-python-edgeimpulse-example",
        "Schedule": "OBJECTS",
        "ReceiveInputImage": true,
        "RunLast": false,
        "NoResponse": true
    },
    {
        "Name": "Example-Postprocessor",
        "Command": "$VMS_BASE/mediaserver/var/nx_ai_manager/postprocessors/postprocessor-python-example",
        "Schedule": "OBJECTS",
        "ReceiveInputImage": false,
        "Objects": [
            { "ID": "python_example", "Name": "Python Example" }
        ]
    },
    {
        "Name": "Image-Postprocessor",
        "Command": "$VMS_BASE/mediaserver/var/nx_ai_manager/postprocessors/postprocessor-python-image-example",
        "Schedule": "OBJECTS",
        "LogFilepath": "/tmp/postprocess.image-example.stdout.log",
        "ReceiveInputImage": true,
        "Objects": [
            { "ID": "python_image_example", "Name": "Python Image Example" }
        ]
    },
    {
        "Name": "NoResponse-Postprocessor",
        "Command": "$VMS_BASE/mediaserver/var/nx_ai_manager/postprocessors/postprocessor-python-noresponse-example",
        "Schedule": "OBJECTS",
        "ReceiveInputImage": false,
        "ReceiveBinaryData": false,
        "NoResponse": true
    },
    {
        "Name": "ANPR-Example",
        "Command": "$VMS_BASE/mediaserver/var/nx_ai_manager/postprocessors/postprocessor-python-anpr-example",
        "Schedule": "OBJECTS",
        "ReceiveBinaryData": true,
        "Config": {
            "LogLevel": "DEBUG"
        },
        "Events": [
            { "ID": "anpr.plate_recognized", "Name": "License plate recognized" }
        ]
    }
]
```

Taken together, this file wires up a silent uploader, a simple bounding-box adder, an image-reading processor, a silent inspector, and an event-producing ANPR processor.

---

## Preprocessor equivalent: `external_preprocessors.json`

Preprocessors use a near-identical schema — the same bare array, in a file named `external_preprocessors.json` in the sibling `preprocessors/` directory — with these differences:

- `Schedule` uses `"IMAGE"` or `"TENSOR"` instead of `"OBJECTS"`.
- **`Config` reaches a preprocessor** as `--key value` pairs before `--ProcessorName`, the same as for a postprocessor.
- `ReceiveInputImage` / `ReceiveBinaryData` / `NoResponse` / `RunLast` / `Objects` / `Events` do not apply: a preprocessor mutates model *input*, not the objects/events response.
- **`UserSettings` works exactly as it does for postprocessors.** Declared controls render in the plugin dialog and their values arrive every frame in a root `UserSettings` map on the preprocessor message, flat `{name: value}` with bare names, emitted only when non-empty. Read the value fresh per frame with a hardcoded default as the absent-key fallback, because the map is not sent while it is empty; do not mirror the value into `Config`. See `.claude/docs/input_from_ai_manager.md` for the wire shape.

Minimal example (adapted from `preprocessor-python-tensor-example/external_preprocessors.json`; the `Command` and `LogFilepath` values differ):

```json
[
    {
        "Name": "Tensor-NMS-Override",
        "Command": "$VMS_BASE/mediaserver/var/nx_ai_manager/preprocessors/preprocessor-python-tensor-example",
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
]
```

---

## Checklist before restarting the server

- File parses as valid JSON (a single trailing comma will break the whole file).
- `Schedule` is correct on every entry that sets it. A postprocessor can omit it (default `"OBJECTS"`). Set `"IMAGE"` or `"TENSOR"` on every preprocessor.
- `Command` path exists and is executable (`chmod +x`), or the entry uses `Port` for iteration. An entry with neither `Command` nor `Port` is skipped.
- No `SocketPath` anywhere — it is a v4 field, and the v5 engine passes IPC descriptors in `argv` instead.
- `LogFilepath` is set (and its directory writable), or a failing `Command`-started processor will fail silently.
- Every class name you plan to add to `msg.objects` in your response is listed in `Objects`.
- Every event ID you plan to emit is listed in `Events`, spelled the same way the processor emits it (no `nx.nxai.` prefix on either side).
- If you set `ReceiveInputImage: true`, `msg.image_data` carries the frame on the same message — no second message to read.
- If you set `ReceiveBinaryData: true`, your processor knows how to handle the extra `Tensors` array in the inference-results message.
- Restart: `sudo service networkoptix-metavms-mediaserver restart`. The processor's name should appear in the NX Cloud Pipelines UI.

---

## Evidence in the repo

- Root field list and worked multi-processor file: `README.md` (sections "Defining the pre/postprocessors" and "Development").
- Per-field semantics: `postprocessor-python-example/README.md`, `postprocessor-cpp-tensors-example/README.md`, `postprocessor-python-noresponse-example/README.md`.
- `Objects` and `Events` declarations: `postprocessor-python-example/README.md`, `postprocessor-python-anpr-example/README.md`.
- `UserSettings` and the full Settings Model grammar: `postprocessor-python-settings-example/README.md` and `postprocessor-python-settings-example/settings_model.md`.
- `Schedule`: `postprocessor-python-example/README.md` (`"OBJECTS"`), `preprocessor-python-tensor-example/README.md`, `preprocessor-python-image-example/README.md` (`"IMAGE"`/`"TENSOR"`).
