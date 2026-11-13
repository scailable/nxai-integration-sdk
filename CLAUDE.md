# CLAUDE.md

Orientation for Claude Code working in this repo.

## What this repo is

The **NX AI Integration SDK** — tools and examples for writing external pre- and post-processors that plug into the NX AI Manager (part of Network Optix MetaVMS). A post-processor is a standalone program that receives the AI Manager's inference results as a MessagePack message over a pair of shared-memory IPC channels, optionally mutates them (adds boxes, tags objects, emits events), and returns them. A preprocessor does the analogous job on the input side.

The SDK targets **AI Manager v5**. v4 processors do not run on a v5 engine and there is no compatibility shim: the transport, the launch arguments, the registration-JSON shape and the message keys all changed. A v4 binary is started, fails to signal readiness on the IPC channels, and is dropped with `Error: Failed to start external postprocessor <name>` in the AI Manager log.

The entire SDK is organized around one pattern: per-processor directories at the repo root, built and installed by a CMake tree. There is no runtime, no framework to learn beyond the MessagePack schema.

## Repo shape

- **`postprocessor-*/`** — 18 example post-processors in Python and C++. Copy the closest one as a starting template.
- **`preprocessor-*/`** — 3 example preprocessors.
- **`nxai-utilities/`** — First-party C++ and Python helpers (plus the bundled mpack C library), maintained in-tree (IPC channel setup, MessagePack codecs, shared-memory reader). `nxai_postprocessor_sdk.py` and `external_processor_utilities.{h,cpp}` are the source of record for the v5 wire format.
- **`message_processing_utils/`** — Python helpers used by the ANPR and average-car-speed examples (OCR engine, deduplication caches). Not needed for simple processors.
- **`example_models/`** — ONNX models used by the advanced examples.
- **`CMakeLists.txt`** — Root build file. Add a new processor by appending one line here and creating the processor's own directory.
- **`build_all.sh`** / **`build_docker.sh`** — Convenience wrappers around the CMake invocation.
- **`.claude/docs/`** — Reference documentation for the wire protocol and JSON config, describing v5 and current:
  - `external_postprocessors_json.md` — every field of the JSON registration file.
  - `input_from_ai_manager.md` — every field the processor receives, including the shared-memory IPC transport and the `--port` dev-mode fallback.
  - `output_to_ai_manager.md` — every field the processor can return, including the `Objects`/`Events` response shape and the event-id rule.

  Where they disagree with `nxai-utilities/nxai_postprocessor_sdk.py` or a migrated example's `external_postprocessors.json`, the code wins.

## Canonical workflow

Build a single processor:

```shell
mkdir -p build && cd build
python3 -m venv integrationsdk && source integrationsdk/bin/activate
cmake ..
cmake --build . --target <processor-name>
```

Install it:

```shell
cmake --install . --component <processor-name>
```

Register it by adding an entry to `VMS_BASE/mediaserver/var/nx_ai_manager/postprocessors/external_postprocessors.json` (see Install paths below for `VMS_BASE`), then:

```shell
sudo service <vms-service> restart
```

Each example ships its own `external_postprocessors.json` next to its source — copy that as the schema reference. The registration file is a **JSON array** at the root; a v4-style `{"externalPostprocessors": [...]}` wrapper makes the AI Manager reject the whole file.

## Verifying on Windows

`verify-windows.cmd` in the repo root is the entry point for Windows work: it enters the MSVC
developer environment itself, configures, builds the C++ targets and runs the `ctest` suite,
exiting non-zero on any failing case. Start it from any shell — do not wrap it in `vcvars`,
and do not substitute a bare `cmake --build`, which packages every Python processor with
Nuitka for nothing. The C++ scenarios, the launcher's own tests and the Python scenarios all
run, and every enabled case must pass. `e2e-python-preproc-image-realloc-multi` is
`DISABLED`: it sends three 4K frames back to back, the first reply grows the output
channel, and the harness does not acknowledge that reallocation in time, so frame 1's send
times out and frame 2's is refused with `IPC_ERROR_BUSY`. A sender must not write over an
unread payload, and the contract for what it does after `BUSY` is not settled; keep the case
disabled until it is.

## Development shortcut

For fast iteration, replace the `Command` field with **`"Port": <n>`** in the JSON entry. The AI Manager then does not start the processor; it connects to `127.0.0.1:<n>` and sends data there each frame. You start the processor yourself with `--port <n>`, edit, kill, restart — no rebuild-and-reinstall cycle. An entry with neither is skipped; if both are present, `Port` wins without a warning. Set exactly one. See the root `README.md` under "Development".

## Gotchas that cost hours

- **The registration JSON root is an array.** `[{...}, {...}]`, not `{"externalPostprocessors": [...]}`. The v4 wrapper does not fail one entry — it fails the entire file, so every processor disappears at once.
- **Register new classes in the JSON.** Any new class name a post-processor adds to `msg.objects` must also appear under `Objects: [{ID, Name}]` in its JSON entry, or the UI silently drops the box.
- **Register new event IDs in the JSON.** Same rule for `Events: [{ID, Name}]` — undeclared event IDs are dropped.
- **Emit event ids exactly as declared — no `nx.nxai.` prefix.** The AI Manager copies `Events[].ID` into the DeviceAgent manifest verbatim and relays emitted ids verbatim, so JSON and emit must match. A prefixed emitted id is dropped by the mediaserver. Object classes are the opposite: those the engine still prefixes with `nx.nxai.`.
- **The first two CLI arguments are the IPC sync descriptors.** The AI Manager launches the processor as `<input_sync> <output_sync> [--Config-key value ...] --ProcessorName <name>`. There is no socket path anywhere — call `initialize_from_args()` with no argument (it reads `sys.argv[1:]`), pass the full `sys.argv` to `load_config()`, and read the processor's own name from the `ProcessorName` config key, not from a positional argument.
- **Bounding boxes are normalized to `[0,1]`, not pixels.** Each object is a map with `x`/`y`/`width`/`height` under `msg.objects[class_name]`; v4's flat pixel `BBoxes_xyxy` array and its parallel `Counts` / `ObjectsMetaData` arrays are gone. The exception is a `bboxes-format:xyxysc` tensor that a `TENSOR`-schedule postprocessor returns: its corners are pixels of the model's input image, and the engine maps them to the frame.
- **Extra input blocks are opt-in flags, not extra messages.** `ReceiveInputImage: true` adds an `ImageData` block and `ReceiveBinaryData: true` a `Tensors` array — both inside the *same* message. There is no second per-frame message to read, and no `ReceiveInputTensor` field; when a processor runs is set by `Schedule` (`OBJECTS` / `TENSOR`, and `IMAGE` for preprocessors).
- **Shutdown arrives as an IPC `EXIT` signal, not an `{"EXIT": ...}` message.** `wait_for_message()` returns `None` for it exactly as it does for a timeout. The SDK sets `exit_requested` to `True` when it reads the signal, so read `processor.exit_requested` after each `None` and leave the loop when it is `True`. A loop that only `continue`s on `None` will be terminated by the engine after its grace period rather than exiting cleanly. The C++ SDK draws the same line: `nxai_ipc_receive()` returns `IPC_EXIT_REQUESTED` for a graceful `EXIT` (arriving directly, or as the follow-up to a shared-memory reallocation) and `IPC_UNEXPECTED_SIGNAL` only for a real protocol violation, so a C++ loop leaves quietly on the first and reports an error on the second. The Python `exit_requested` flag and `IPC_EXIT_REQUESTED` mean the same thing: a graceful EXIT.

## Picking an example to copy

- Modify bounding boxes or filter detections → `postprocessor-python-example/` (Python), `postprocessor-cpp-objects-example/` (C++).
- Attach per-object attributes (labels in the UI) → `postprocessor-python-confidences-example/`.
- Emit events → `postprocessor-python-events-example/`.
- Expose settings in the NX plugin UI → `postprocessor-python-settings-example/` (see also its `settings_model.md`).
- Read the raw input image → `postprocessor-python-image-example/`.
- Consume raw model outputs (logits, embeddings) → `postprocessor-cpp-tensors-example/`, `postprocessor-python-anpr-example/`.
- Semantic similarity search with CLIP embeddings → `postprocessor-python-clip-example/`.
- Multi-frame tracking / speed analytics → `postprocessor-python-measure-average-car-speed/`.
- Event when the object count reaches a settable threshold → `postprocessor-python-object-counter-threshold/`.
- Tag a tracked object with a best shot → `postprocessor-python-bestshots-example/`.
- Estimate person height from the bounding box, event above a limit → `postprocessor-python-height-estimator/`.
- Forward detections to a companion web app (objects pass through unchanged) → `postprocessor-python-web-dashboard/`.
- Map detections to geo-coordinates and send GeoJSON over HTTP → `postprocessor-python-objects-sender/`.
- Offload inference to a cloud endpoint → `postprocessor-cloud-inference-example/`.
- Fire-and-forget (no response) → `postprocessor-python-noresponse-example/`, `postprocessor-python-edgeimpulse-example/`.
- Preprocessing (image stage) → `preprocessor-python-image-example/`.
- Preprocessing (tensor stage) → `preprocessor-python-tensor-example/`.
- Preprocessing with CLIP tokenization → `preprocessor-python-clip-example/`.

## Install paths (reference)

Both NX Meta and NX Witness are supported. Every install path is written relative to
`VMS_BASE`, the install root of the VMS product — resolve it from this table when
reading or writing a path (there is no shell here to export it into):

| Product | `VMS_BASE` | Service |
|---------|------------|---------|
| NX Meta | `/opt/networkoptix-metavms` | `networkoptix-metavms-mediaserver` |
| NX Witness | `/opt/networkoptix` | `networkoptix-mediaserver` |

- Post-processor binaries: `VMS_BASE/mediaserver/var/nx_ai_manager/postprocessors/`
- Post-processor JSON: `.../postprocessors/external_postprocessors.json`
- Preprocessor binaries and JSON: analogous `.../preprocessors/` directory.
- CMake auto-detects the active product at configure time; override with `-DINSTALL_DEST_POSTPROCESSORS=<path>`.

On Windows `VMS_BASE` is the mediaserver service profile: `C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix MetaVMS Media Server` for NX Meta, `...\Network Optix\Network Optix Media Server` (same parent) for NX Witness. CMake picks the one that already has an `nx_ai_manager` directory, MetaVMS when neither has. That tree has no `mediaserver\var` level — the paths above become `VMS_BASE\nx_ai_manager\postprocessors\` and `VMS_BASE\nx_ai_manager\preprocessors\`.

## When in doubt

Open `nxai-utilities/nxai_postprocessor_sdk.py` (message classes, IPC setup, config parsing) and the example from the list above that's closest to the target use case, together with its `external_postprocessors.json`. The SDK is small enough that reading two examples end-to-end is often faster than guessing. Consult `.claude/docs/*.md` for field-by-field detail.
