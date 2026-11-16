---
name: migrate-to-v5
description: Port a v4 pre- or post-processor (Unix-socket transport) to the AI Manager v5 SDK.
argument-hint: "<processor name>"
disable-model-invocation: true
---

Migrate a v4 pre- or post-processor (Unix-socket transport) to the AI Manager v5 SDK (IPC transport).

This skill covers the v4 check, the wire-format changes, and the rewrite of the script, JSON, CMake and README. The calling process does the tests, review and commit.

`.claude/docs/writing_a_processor.md` gives the v5 target: the template table, the main script, the registration JSON, the CMake file, the README and the tests, for both sides. This skill gives what changes on the way from v4 to that target.

## Arguments

The processor name is the text after the skill name, or the processor the request names. If neither gives one, ask. It is the directory name of a v4 **postprocessor or preprocessor**, full or short (`<name>` or `postprocessor-python-<name>`). Resolve `postprocessor-python-<name>/` first, then `preprocessor-python-<name>/`, then the literal argument.

## Steps

1. **Gate.** Read the source and decide by the transport:
   - A Unix-socket listener, a socket path from `argv[1]`, or a `SocketPath` registration key → v4. Continue.
   - `import nxai_postprocessor_sdk` → already v5. Stop.

   The module name of the v4 socket helpers varies: `nxai_communication_utils` in the v4 SDK, `communication_utils` in older layouts and forks. Decide by the transport only.

2. **Read `.claude/docs/writing_a_processor.md` and the wire-format files it names, in full, before you change code.**

3. **Map each v4 feature to v5.**
   - **Raw model outputs (`BinaryOutputs`).** Set `ReceiveBinaryData: true`. The outputs then arrive as a root `Tensors` array, and the SDK parses it into `msg.tensors` (`name -> TensorMetadata`). `postprocessor-python-clip-example` is the worked example: it reads the `scores` tensor from `msg.tensors`.
   - **Pixel `Width`/`Height`.** v5 bounding boxes are frame-relative. Change the logic to normalized units, as `frame_fraction_per_meter` in `postprocessor-python-height-estimator` does. Turn on `ReceiveInputImage` only if the logic needs the frame itself.
   - **Live settings.** The engine sends GUI settings as a root `UserSettings` map, to postprocessors and preprocessors. The SDK exposes it as `msg.user_settings`. Declare typed controls in the JSON `UserSettings` array.
   - **Fire-and-forget.** The engine sends messages and waits for no response. Set `"NoResponse": true`.

4. **Pick a v5 sibling as the template** from the table in `writing_a_processor.md`.

5. **Rewrite the main script** as `writing_a_processor.md` ("The main script") says. Also:
   - Delete the socket bootstrap, the `argv[1]` socket path and the `os.unlink` socket cleanup. Add `import nxai_postprocessor_sdk`.
   - **Postprocessor objects.** `input_object["BBoxes_xyxy"]` and its parallel `ObjectsMetaData` and `Counts` arrays become `msg.objects[class_name]`.
   - **Postprocessor events.** `input_object["Events"].append({...})` becomes `msg.events.append(EventMetadata())`.
   - **`Caption`.** v5 events have no `Caption`. Remove it and name the removal in the module docstring, as `postprocessor-python-height-estimator` does.

6. **Write the registration JSON** as `writing_a_processor.md` ("Registration JSON") says. Also:
   - **The v4 source has its own JSON.** Keep its content and convert it:
     1. Unwrap the `{"externalPostprocessors": [...]}` or `{"externalPreprocessors": [...]}` root to a bare array.
     2. Rename `Settings` to `UserSettings`.
     3. Remove the `externalprocessor.` prefix from each control `name`, and from each field that refers to a control by name, such as `filledCheckItems`. The script reads the same bare names from `msg.user_settings`.
     4. Delete the v4 keys `SocketPath`, `ReceiveInputTensor` and `ReceiveConfidenceData`.
   - Delete `plugin.<name>.ini.example`, unless the script still reads the ini file, as `postprocessor-python-edgeimpulse-example` does.

7. **Replace `CMakeLists.txt`** as `writing_a_processor.md` ("CMakeLists.txt") says. Also:
   - Discard the v4 file completely. No v4 build step works on v5: `DEPENDS nxai-c-utilities-shared`, copy steps for the socket helpers or the shared C library, PyInstaller calls. `../nxai-utilities/python-utilities/` does not exist.
   - In `requirements.txt`, replace the v4 bundler (such as `pyinstaller`) with `nuitka`.

8. **Replace `README.md`** as `writing_a_processor.md` ("README.md") says. Also:
   - Delete the v4 text completely. Its sections are wrong for v5: "Postprocessors Control Flow", "MessagePack schema", "Download the integration SDK", "Preparation of dependencies", "Restarting the server", "Output logging", and the wrapped `{"externalPostprocessors": [...]}` example.
   - Add `## Behavior change from v4` after `## What it does`, only if step 3 or 5 changed behavior, for example a removed `Caption` or a unit change.

9. **Hand back to the calling process** for tests, review and commit. Tell it:
   - Name each behavior change from step 3 or 5 in the commit message.
   - Write the tests that `writing_a_processor.md` ("Tests") requires.
