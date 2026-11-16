# Writing a v5 processor — Reference

This page gives the target shape of a Python processor for AI Manager v5: the names, the main script, the registration JSON, the CMake file, the README and the tests. The `/migrate-to-v5` skill ports a v4 processor to this shape. Start from a v5 sibling and change only what differs.

Read the wire format before you write code:

- `input_from_ai_manager.md` — what the processor receives.
- `output_to_ai_manager.md` — what the processor returns.
- `nxai-utilities/nxai_postprocessor_sdk.py` — the message classes and codecs. If the docs and the SDK disagree, the SDK is correct.

One module, `nxai_postprocessor_sdk`, serves both sides. There is no `nxai_preprocessor_sdk`. The sides differ only in the codec calls, the registration file and the install destination.

---

## Pick a template

Keep the template open while you write.

| Processor does | Template |
|---|---|
| Changes boxes or attributes | `postprocessor-python-example` |
| Adds attributes or events | `postprocessor-python-height-estimator`, `postprocessor-python-object-counter-threshold` |
| Reads live settings | `postprocessor-python-settings-example` |
| Sends results to a network service | `postprocessor-python-web-dashboard`, `postprocessor-python-objects-sender` |
| Reads the input frame | `postprocessor-python-image-example` |
| Fire-and-forget | `postprocessor-python-noresponse-example`, `postprocessor-python-edgeimpulse-example` |
| Preprocessor, `Schedule: "TENSOR"` | `preprocessor-python-tensor-example` |
| Preprocessor, `Schedule: "IMAGE"` | `preprocessor-python-image-example` |

---

## Names

Give a new processor these names. Some older examples do not obey them. The processor directory, `<processor-dir>`, sets all other names. Example: `postprocessor-python-vehicle-counter`.

- **`<name>`**, the short name: `<processor-dir>` without a leading `postprocessor-python-` or `preprocessor-python-`. If the directory has neither prefix, remove a leading `postprocessor-` or `preprocessor-`. Keep the case and the hyphens. Example: `vehicle-counter`.
- **The script** is `<processor-dir>/<processor-dir>.py`.
- **The unit test** is `<processor-dir>/test_<processor_dir>.py`, where `<processor_dir>` is `<processor-dir>` with `-` replaced by `_`.
- **`PROCESSOR_NAME` in the script** is `"<name>"`. It is only the fallback name in the startup log line. The running processor reads its name from the `ProcessorName` config key.
- **`PROCESSOR_NAME` in `CMakeLists.txt`** is `<processor-dir>`. It is also the build target and the install component.
- **`Name` in the registration JSON** is `<processor-dir>`.
- **`Command`** in the colocated registration JSON is a placeholder. `/install-processor` tells the user to replace it with the path of the installed binary.
  - Linux: `/path/to/build/<processor-dir>/<processor-dir>`, as in the shipped examples.
  - Windows: `C:/path/to/build/<processor-dir>/<processor-dir>.exe`. On Windows, Nuitka and the C++ compiler write the binary with the `.exe` suffix. Forward slashes need no escape in JSON.
- **`LogFilepath`** is the file that gets the stdout and stderr of the processor. The AI Manager writes to it. `<side>` is `postprocess` for a postprocessor and `preprocess` for a preprocessor:
  - Linux: `/tmp/<side>.<name>.stdout.log`.
  - Windows: `"C:\\Windows\\Temp\\<side>.<name>.stdout.log"`, as written in the JSON file. Write each backslash two times: a single backslash makes the JSON invalid. The mediaserver service runs as LocalSystem, and Windows has no `/tmp`.
- **A new object class** (`INJECTED_CLASS` in the templates) has the ID `<name>` with `-` replaced by `_`. Example: `vehicle_counter`. Its `Objects[].Name` is `<name>` in title case with `-` replaced by a space. Example: `Vehicle Counter`. Do not keep the class ID of the template, and do not use the class ID of another processor. The AI Manager merges the `Objects` of all registered processors and keeps duplicates. The manifest then reports duplicate object types.

---

## The main script

- Copy the `__main__` block and `run_message_loop(processor)` from the template without changes. Keep the alias the template uses, `ExternalPostprocessor()` or `ExternalPreprocessor()`. Both name the same `ExternalProcessor` class.
  - `__main__`: `setup_logging()` → `load_config(sys.argv)` → `apply_config_logging()` → `initialize_from_args()` → `run_message_loop(processor)`.
  - The loop: `wait_for_message()` → parse → `handle_message()` → write → `send_response()`.
  - `wait_for_message()` returns `None` for a timeout and for an EXIT signal. On `None`, return if `processor.exit_requested` is set, else continue. If the loop ignores the flag, the AI Manager kills the process.
- Log through the SDK logger to stdout and stderr. The AI Manager writes both to `LogFilepath` of the registration entry.
- Use the codec pair for the message type:
  - Postprocessor: `parse_objects_message` / `write_objects_message`.
  - Preprocessor, `TENSOR`: `parse_tensor_message` / `write_tensor_message`.
  - Preprocessor, `IMAGE`: `parse_image_message` / `write_image_message`.
- Put the business logic in one pure function, `handle_message(msg) -> msg`, typed on the message class (`ObjectsPostProcessorMessage`, `TensorMessage` or `ImagePreprocessorMessage`). Tests then call it without IPC.
- `handle_message` changes the message it receives and returns that same object. A postprocessor response must carry the `DeviceID` and `Timestamp` of its request. A new message has them empty, and the AI Manager drops that response without a log line.
- **Postprocessor objects.** `msg.objects[class_name]` is a list of `ObjectMetadata` with normalized xywh. Set attributes directly: `obj.attributes[key] = value`.
- **Postprocessor events.** Append an `EventMetadata()` to `msg.events`, with `.type_id` and `.description`. Emit `.type_id` exactly as it is in `Events[].ID` in the JSON, with no `nx.nxai.` prefix.
- **Live settings.** Read the bare control names from `msg.user_settings` on each frame. The values arrive as strings (see `postprocessor-python-settings-example/settings_model.md`). Parse each value (for example with `int()`), catch the parse error, and check the declared range. Use the hardcoded default when the key is absent, the value does not parse, or the value is out of range.
- **Fire-and-forget postprocessor.** End each loop pass after the side effect. Call neither `write_objects_message` nor `send_response`, and build no response message.
- **Preprocessor.** Always respond, also with an unchanged message. The engine needs the message back to continue the pipeline.

---

## Registration JSON

Write the registration JSON next to the script. The root is a bare array. Copy the file of the nearest sibling and change only what differs. The field reference is `external_postprocessors_json.md`.

- **Postprocessor**, `external_postprocessors.json`: `Name`, `Command`, `Schedule: "OBJECTS"`, `LogFilepath`, `Config`. Add the fields the processor needs:
  - `Objects` and `Events` for each class name and event id the processor adds. The UI drops undeclared ones.
  - `UserSettings` for live settings.
  - `NoResponse: true` for fire-and-forget.
  - `ReceiveInputImage` for the frame, `ReceiveBinaryData` for raw tensors.
- **Preprocessor**, `external_preprocessors.json`: `Name`, `Command`, `Schedule` (`"TENSOR"` or `"IMAGE"`), `LogFilepath`, `Config`, and `UserSettings` for live settings. A preprocessor adds no objects or events, so it has no `Objects` or `Events`.

---

## CMakeLists.txt

- Copy `postprocessor-python-height-estimator/CMakeLists.txt`. Set `PROCESSOR_NAME`. For a preprocessor, change `INSTALL_DEST_POSTPROCESSORS` to `INSTALL_DEST_PREPROCESSORS`. With the wrong one, the processor installs to a directory the plugin does not read.
- The template builds with Nuitka. Put `nuitka` in `requirements.txt`.
- Add an `add_subdirectory` line for the directory to the root `CMakeLists.txt`. Without it, the target does not exist and `cmake --build . --target <processor-dir>` fails with "No rule to make target".
- From the build directory, run `cmake ..` and `cmake --build . --target <processor-dir>`. The build must exit 0 and leave the binary `<build dir>/<processor-dir>/<processor-dir>`, with `.exe` on Windows. Ninja, the generator of `build-windows/`, prints no "Built target" line.
- Run the wiring scan: `python tests/build-wiring/build_wiring_scan.py --root .`. CTest runs the same check as `build-wiring-scan`.

---

## README.md

Write at most about 125 lines. The content sets the length: a simple processor gets a short README. Use the section order of the height-estimator, object-counter-threshold, settings-example and tensor-example READMEs:

- `# <Title>`
- `## What it does` — the logic in v5 message terms: `msg.objects[cls]`, `.attributes`, `.events`, and for preprocessors `msg.tensors` or `msg.image`.
- `## Registration` — the registration JSON, verbatim. Then one bullet for each field the processor uses. The full field reference stays in the docs file.
- `## Build, test, install` — `cmake ..`, `cmake --build . --target <processor-dir>`, `cmake --install . --component <processor-dir>`, and `PYTHONPATH=../nxai-utilities python -m unittest test_<processor_dir> -v` for the test next to the script.
- Processor-specific sections, where they read best. For example, settings-example puts `## The UserSettings contract` before Registration, and web-dashboard puts `## Running the web app` after Build.
- `## Licence`

---

## Tests

- **Unit tests** are required for every processor. They call `handle_message` directly, with no IPC.
- **External services** (a cloud endpoint, an HTTP bridge, a vendor SDK): unit tests make no live call and do not need the service. Mock at the boundary:
  - If the service is behind its own module, put a `MagicMock` into `sys.modules` under that module name before you import the processor. `postprocessor-cloud-inference-example` does this for `aws_utils`, and `postprocessor-python-edgeimpulse-example` for `edgeimpulse`.
  - If the script calls a client library directly, patch that call (for example `requests.request`) and assert on the request it would send.
- **A processor in this SDK repo** also gets a harness scenario: a `--check` case in `tests/e2e/postprocessor_harness.cpp` and an `add_test` in `tests/e2e/CMakeLists.txt`. Templates: `e2e-python-counter` (postprocessor), `e2e-python-preproc-tensor` (preprocessor). Give each setting the processor reads (`Config` or `UserSettings`) a value different from its default, so that the test exercises the read. A processor outside this repo has no harness, so unit tests are its only layer.
