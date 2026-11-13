Scaffold a new Python post-processor for the NX AI Integration SDK.

## Arguments

`$ARGUMENTS` — format: `<name> [type]`

- **name**: short name, e.g. `vehicle-counter`. Directory and file will be named `postprocessor-python-<name>`.
- **type** (optional, default: `simple`):

  | type | purpose | template |
  |------|---------|----------|
  | `simple` | Modify bounding boxes | `postprocessor-python-example` |
  | `events` | Emit events | `postprocessor-python-events-example` |
  | `settings` | Expose UI settings | `postprocessor-python-settings-example` |
  | `binary` | Consume raw model output tensors | `postprocessor-python-example` |
  | `noresponse` | Fire-and-forget side effect | `postprocessor-python-noresponse-example` |
  | `image` | Read the raw input image | `postprocessor-python-image-example` |

## Steps

1. **Parse** name and type from the arguments. If the name already starts with `postprocessor-python-`, strip that prefix before re-applying it.

2. **Check for an existing directory.** If `postprocessor-python-<name>/` already exists, stop immediately and print:
   ```
   ❌ postprocessor-python-<name>/ already exists. Choose a different name or delete it first.
   ```

3. **Confirm** before touching anything: print the full processor name, the chosen type, and the template being used.

4. **Create the processor directory** `postprocessor-python-<name>/`.

5. **Create the Python script** `postprocessor-python-<name>/postprocessor-python-<name>.py`:
   - Copy the template `.py` file.
   - Set `PROCESSOR_NAME` to `"<name>"`. It is only the fallback name in the startup log line: the running processor reads its name from the `ProcessorName` config key (`load_config(sys.argv)`). The SDK logs to stdout and stderr, and the AI Manager writes both to the entry's `LogFilepath`. Do not add a name, a socket path or a file path of your own.
   - If the template defines `INJECTED_CLASS`, set it to `"<name>"` with `-` replaced by `_`. The AI Manager merges the `Objects` of all registered processors without deduplication, so a class shared with the template makes the manifest report duplicate object types.
   - Make this comment the first statement of `handle_message()`, directly after its docstring:
     ```python
     # TODO: add your processing logic here
     ```
     Leave the template's logic in `handle_message()` in place: it is a working example, which the user later replaces with their own logic. Keep `run_message_loop()` and the `__main__` block unchanged. They already wait for messages, stop on the IPC `EXIT` signal, parse, and send the response.
   - If type is `binary`, use this comment block instead:
     ```python
     # TODO: read raw model outputs from msg.tensors (needs "ReceiveBinaryData": true).
     # msg.tensors maps a tensor name to a TensorMetadata: .shape (list of int),
     # .data_type (nxai_postprocessor_sdk.DATA_TYPE_* code) and .data (bytes).
     # See .claude/docs/input_from_ai_manager.md, "Raw model outputs: Tensors".
     ```

6. **Create the unit test** `postprocessor-python-<name>/test_postprocessor_python_<name_with_underscores>.py`, where `<name_with_underscores>` is `<name>` with `-` replaced by `_`:
   - Copy the template's `test_*.py` file (`test_postprocessor_python_example.py` for `simple` and `binary`).
   - Replace every occurrence of the template's directory name (e.g. `postprocessor-python-example`) with `postprocessor-python-<name>`. This changes the module the test loads.
   - Replace every occurrence of the template's directory name with `-` replaced by `_` (e.g. `postprocessor_python_events_example`) with `postprocessor_python_<name_with_underscores>`.
   - If the template defines `INJECTED_CLASS`, replace every quoted string literal of its old value (e.g. `"python_image_example"`), docstrings included, with the new `INJECTED_CLASS` from step 5.
   - Replace every quoted string literal of the template's old `PROCESSOR_NAME` value (e.g. `"python-example"`) with `"<name>"`.
   - Leave everything else unchanged. The test then checks the template logic that step 5 keeps in place; when the user replaces that logic, they update the test with it.

7. **Create `CMakeLists.txt`** `postprocessor-python-<name>/CMakeLists.txt`:
   - Read the template's `CMakeLists.txt`.
   - Replace the `PROCESSOR_NAME` value with `postprocessor-python-<name>`.
   - Leave everything else unchanged.

8. **Create `requirements.txt`**:
   - If the template has a `requirements.txt`, copy it verbatim.
   - Otherwise create a minimal one containing only `msgpack`.

9. **Create `external_postprocessors.json`**: copy the template's `external_postprocessors.json`, then in its one element:
   - Set `Name` to `postprocessor-python-<name>`.
   - In `Command`, replace both occurrences of the template's directory name with `postprocessor-python-<name>`.
   - Set `LogFilepath` to `/tmp/postprocess.<name>.stdout.log`.
   - If it has an `Objects` entry whose `ID` is the template's old `INJECTED_CLASS`, set that `ID` to the new `INJECTED_CLASS` from step 5 and its `Name` to `<name>` in title case with `-` replaced by a space (e.g. `Vehicle Counter`).
   - If type is `binary`, add `"ReceiveBinaryData": true` after `LogFilepath`.
   - Leave everything else unchanged.

10. **Wire into root `CMakeLists.txt`**: insert
    ```cmake
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-<name>)
    ```
    directly after the last `add_subdirectory` line that names a processor directory (`postprocessor-*` or `preprocessor-*`), not after the `tests/` ones.
    - Then run the build wiring scan: `python tests/build-wiring/build_wiring_scan.py --root .` — the same check as CTest case `build-wiring-scan`; it asserts root-CMake reachability and a registration JSON in every wired processor directory, so the new directory can't ship unwired.

11. **Print a summary**:
    ```
    Created postprocessor-python-<name>/
      postprocessor-python-<name>.py   ← add your logic at the TODO
      test_postprocessor_python_<name_with_underscores>.py   ← update with your logic
      CMakeLists.txt
      requirements.txt
      external_postprocessors.json

    Next steps:
    1. Edit the .py file — search for "TODO" to find where to add logic.
    2. Declare any new event IDs or object class names in external_postprocessors.json
       before restarting.
    3. Run /install-processor <name> to build and install.
    ```
