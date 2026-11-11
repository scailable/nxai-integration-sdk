Migrate a v4 pre- or post-processor (Unix-socket transport) to the AI Manager v5 SDK (IPC transport). One module, `nxai_postprocessor_sdk`, serves both sides. There is no `nxai_preprocessor_sdk`.

This command covers the v4 check, the wire-format changes, and the rewrite of the script, JSON, CMake and README. The calling process does the tests, review and commit.

## Arguments

`$ARGUMENTS` — the directory name of a v4 **postprocessor or preprocessor**, full or short (`<name>` or `postprocessor-python-<name>`). Resolve `postprocessor-python-<name>/` first, then `preprocessor-python-<name>/`, then the literal argument.

Both sides share every step. Only these steps split them: codec calls (step 5), registration file (step 6), install destination (step 7).

## Steps

1. **Gate.** Read the source and decide by the transport:
   - A Unix-socket listener, a socket path from `argv[1]`, or a `SocketPath` registration key → v4. Continue.
   - `import nxai_postprocessor_sdk` → already v5. Stop.

   The module name of the v4 socket helpers varies: `nxai_communication_utils` in the v4 SDK, `communication_utils` in older layouts and forks. Decide by the transport only.

2. **Read the v5 wire format in full before you change code:**
   - `.claude/docs/input_from_ai_manager.md` — what the processor receives.
   - `.claude/docs/output_to_ai_manager.md` — what the processor returns.
   - `nxai-utilities/nxai_postprocessor_sdk.py` — the message classes and codecs. If the docs and the SDK disagree, the SDK is correct.

3. **Map each v4 feature to v5.**
   - **Raw model outputs (`BinaryOutputs`).** Set `ReceiveBinaryData: true`. The outputs then arrive as a root `Tensors` array, and the SDK parses it into `msg.tensors` (`name -> TensorMetadata`). `postprocessor-python-clip-example` is the worked example: it reads the `scores` tensor from `msg.tensors`.
   - **Pixel `Width`/`Height`.** v5 bounding boxes are frame-relative. Change the logic to normalized units, as `frame_fraction_per_meter` in `postprocessor-python-height-estimator` does. Turn on `ReceiveInputImage` only if the logic needs the frame itself.
   - **Live settings.** The engine sends GUI settings as a root `UserSettings` map, to postprocessors and preprocessors. The SDK exposes it as `msg.user_settings`. Declare typed controls in the JSON `UserSettings` array. Read the bare control names from `msg.user_settings` on each frame, with a hardcoded default for an absent key.
   - **Fire-and-forget.** The engine sends messages and waits for no response. Set `"NoResponse": true`.

4. **Pick a v5 sibling as the template.** Keep it open while you rewrite.

   | Processor does | Template |
   |---|---|
   | Changes boxes or attributes | `postprocessor-python-example` |
   | Adds attributes or events | `postprocessor-python-height-estimator`, `postprocessor-python-object-counter-threshold` |
   | Reads live settings | `postprocessor-python-settings-example` |
   | Reads the input frame | `postprocessor-python-image-example` |
   | Fire-and-forget | `postprocessor-python-noresponse-example`, `postprocessor-python-edgeimpulse-example` |
   | Preprocessor, `Schedule: "TENSOR"` | `preprocessor-python-tensor-example` |
   | Preprocessor, `Schedule: "IMAGE"` | `preprocessor-python-image-example` |

5. **Rewrite the main script.**
   - Delete the socket bootstrap, the `argv[1]` socket path and the `os.unlink` socket cleanup. Add `import nxai_postprocessor_sdk`.
   - Copy the `__main__` block and `run_message_loop(processor)` from the template without changes. Keep the alias the template uses, `ExternalPostprocessor()` or `ExternalPreprocessor()`. Both name the same `ExternalProcessor` class.
     - `__main__`: `setup_logging()` → `load_config(sys.argv)` → `apply_config_logging()` → `initialize_from_args()` → `run_message_loop(processor)`.
     - The loop: `wait_for_message()` → parse → `handle_message()` → write → `send_response()`.
     - `wait_for_message()` returns `None` for a timeout and for an EXIT signal. On `None`, return if `processor.exit_requested` is set, else continue. If the loop ignores the flag, the AI Manager kills the process.
   - Use the codec pair for the message type:
     - Postprocessor: `parse_objects_message` / `write_objects_message`.
     - Preprocessor, `TENSOR`: `parse_tensor_message` / `write_tensor_message`.
     - Preprocessor, `IMAGE`: `parse_image_message` / `write_image_message`.
   - Put the business logic in one pure function, `handle_message(msg) -> msg`, typed on the message class (`ObjectsPostProcessorMessage`, `TensorMessage` or `ImagePreprocessorMessage`). Tests then call it without IPC.
   - `handle_message` changes the message it receives and returns that same object. A postprocessor response must carry the `DeviceID` and `Timestamp` of its request. A new message has them empty, and the AI Manager drops that response without a log line.
   - **Postprocessor objects.** `input_object["BBoxes_xyxy"]` and its parallel `ObjectsMetaData` and `Counts` arrays become `msg.objects[class_name]`. That is a list of `ObjectMetadata` with normalized xywh. Set attributes directly: `obj.attributes[key] = value`.
   - **Postprocessor events.** `input_object["Events"].append({...})` becomes `msg.events.append(EventMetadata())` with `.type_id` and `.description`. Emit `.type_id` exactly as it is in `Events[].ID` in the JSON, with no `nx.nxai.` prefix.
   - **`Caption`.** v5 events have no `Caption`. Remove it and name the removal in the module docstring, as `postprocessor-python-height-estimator` does.
   - **Fire-and-forget postprocessor.** End each loop pass after the side effect. Call neither `write_objects_message` nor `send_response`, and build no response message.
   - **Preprocessor.** Always respond, also with an unchanged message. The engine needs the message back to continue the pipeline.

6. **Write the registration JSON next to the script.** The root is a bare array, with no `SocketPath`. Copy the file of the nearest sibling and change only what differs. The field reference is `.claude/docs/external_postprocessors_json.md`.
   - **The v4 source has its own JSON.** Keep its content and convert it:
     1. Unwrap the `{"externalPostprocessors": [...]}` or `{"externalPreprocessors": [...]}` root to a bare array.
     2. Rename `Settings` to `UserSettings`.
     3. Remove the `externalprocessor.` prefix from each control `name`, and from each field that refers to a control by name, such as `filledCheckItems`. The script reads the same bare names from `msg.user_settings`.
     4. Delete the v4 keys `SocketPath`, `ReceiveInputTensor` and `ReceiveConfidenceData`.
   - **Postprocessor**, `external_postprocessors.json`: `Name`, `Command`, `Schedule: "OBJECTS"`, `LogFilepath`, `Config`. Add the fields the processor needs:
     - `Objects` and `Events` for each class name and event id the processor adds. The UI drops undeclared ones.
     - `UserSettings` for live settings.
     - `NoResponse: true` for fire-and-forget.
     - `ReceiveInputImage` for the frame, `ReceiveBinaryData` for raw tensors.
   - **Preprocessor**, `external_preprocessors.json`: `Name`, `Command`, `Schedule` (`"TENSOR"` or `"IMAGE"`, as in the source), `LogFilepath`, `Config`, and `UserSettings` for live settings. A preprocessor adds no objects or events, so it has no `Objects` or `Events`.
   - Delete `plugin.<name>.ini.example`, unless the script still reads the ini file, as `postprocessor-python-edgeimpulse-example` does.

7. **Replace `CMakeLists.txt`.**
   - Copy `postprocessor-python-height-estimator/CMakeLists.txt`. Set `PROCESSOR_NAME`. For a preprocessor, change `INSTALL_DEST_POSTPROCESSORS` to `INSTALL_DEST_PREPROCESSORS`. With the wrong one, the processor installs to a directory the plugin does not read.
   - Discard the v4 file completely. No v4 build step works on v5: `DEPENDS nxai-c-utilities-shared`, copy steps for the socket helpers or the shared C library, PyInstaller calls. `../nxai-utilities/python-utilities/` does not exist.
   - The template builds with Nuitka. In `requirements.txt`, replace the v4 bundler (such as `pyinstaller`) with `nuitka`.
   - Add an `add_subdirectory` line for the directory to the root `CMakeLists.txt`. Without it, the target does not exist and `cmake --build . --target <name>` fails with "No rule to make target".
   - From the build directory, run `cmake ..` and `cmake --build . --target <name>`. The output must show "Built target".
   - Run the wiring scan: `python tests/build-wiring/build_wiring_scan.py --root .`. CTest runs the same check as `build-wiring-scan`.

8. **Replace `README.md`.** Delete the v4 text completely. Its sections are wrong for v5: "Postprocessors Control Flow", "MessagePack schema", "Download the integration SDK", "Preparation of dependencies", "Restarting the server", "Output logging", and the wrapped `{"externalPostprocessors": [...]}` example. Write about 90 to 125 lines, in the section order of the height-estimator, object-counter-threshold, settings-example and tensor-example READMEs:
   - `# <Title>`
   - `## What it does` — the logic in v5 message terms: `msg.objects[cls]`, `.attributes`, `.events`, and for preprocessors `msg.tensors` or `msg.image`.
   - `## Behavior change from v4` — only if step 3 or 5 changed behavior, for example a removed `Caption` or a unit change. Otherwise omit the section.
   - `## Registration` — the step 6 JSON, verbatim. Then one bullet for each field the processor uses. The full field reference stays in the docs file.
   - `## Build, test, install` — `cmake ..`, `cmake --build . --target <name>`, `cmake --install . --component <name>`, and `PYTHONPATH=../nxai-utilities python -m unittest test_<name> -v` for the test next to the script.
   - Processor-specific sections, where they read best. For example, settings-example puts `## The UserSettings contract` before Registration, and web-dashboard puts `## Running the web app` after Build.
   - `## Licence`

9. **Hand back to the calling process** for tests, review and commit. Tell it:
   - Name each behavior change from step 3 or 5 in the commit message.
   - **Unit tests** are required for every processor. They call `handle_message` directly, with no IPC.
   - **External services** (a cloud endpoint, an HTTP bridge, a vendor SDK): unit tests make no live call and do not need the service. Mock at the boundary:
     - If the service is behind its own module, put a `MagicMock` into `sys.modules` under that module name before you import the processor. `postprocessor-cloud-inference-example` does this for `aws_utils`, and `postprocessor-python-edgeimpulse-example` for `edgeimpulse`.
     - If the script calls a client library directly, patch that call (for example `requests.request`) and assert on the request it would send.
   - **A processor in this SDK repo** also gets a harness scenario: a `--check` case in `tests/e2e/postprocessor_harness.cpp` and an `add_test` in `tests/e2e/CMakeLists.txt`. Templates: `e2e-python-counter` (postprocessor), `e2e-python-preproc-tensor` (preprocessor). Give each setting the processor reads (`Config` or `UserSettings`) a value different from its default, so that the test exercises the read. A processor outside this repo has no harness, so unit tests are its only layer.
