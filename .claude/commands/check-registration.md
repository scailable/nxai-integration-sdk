Audit that `external_postprocessors.json` and `external_preprocessors.json` declarations are consistent with what each processor's source code actually does.

## Arguments

`$ARGUMENTS` — (optional) short or full processor name. If omitted, all registered post- and preprocessors are checked.

## Steps

1. **Detect the active NX VMS installation** by running these checks in order, stopping at the first match:
   - `systemctl is-active networkoptix-metavms-mediaserver` → `VMS_BASE=/opt/networkoptix-metavms`
   - `systemctl is-active networkoptix-mediaserver` → `VMS_BASE=/opt/networkoptix`
   - `ls /opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/` exists → use metavms
   - `ls /opt/networkoptix/mediaserver/var/nx_ai_manager/` exists → use witness
   - Neither found → print warning, continue without production path check

   **Windows detection.** The platform is Windows when `$env:OS` is `Windows_NT` or `uname -s` starts with `MINGW` or `MSYS`. `VMS_BASE` is the mediaserver service profile. Check these directories in order and stop at the first with an `nx_ai_manager\` subdirectory:
   - `C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix MetaVMS Media Server` (NX Meta)
   - `C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix Media Server` (NX Witness)
   - Neither found → print warning, continue without production path check

   There is no `mediaserver\var` level.

2. **Locate the registration files.** Try the installed files first:
   ```
   <VMS_BASE>/mediaserver/var/nx_ai_manager/postprocessors/external_postprocessors.json
   <VMS_BASE>/mediaserver/var/nx_ai_manager/preprocessors/external_preprocessors.json
   ```
   On Windows these are `<VMS_BASE>\nx_ai_manager\postprocessors\external_postprocessors.json` and `<VMS_BASE>\nx_ai_manager\preprocessors\external_preprocessors.json`.
   For each kind with no installed file, use the files in the repo instead: every `postprocessor-*/external_postprocessors.json` and every `preprocessor-*/external_preprocessors.json`. Treat their arrays as one list. These colocated files are templates. Report if no file is found.

3. **Parse** the JSON. Each file is a bare array of entries. If a name argument was given, keep only the entries whose `Command` basename or processor directory matches it (with the prefix rule of `/install-processor`).

4. **For each entry**, locate its source directory in the repo. Match the basename of `Command` (split on `/` and `\`, without `.exe`) to a `postprocessor-*/` or `preprocessor-*/` directory. `Name` is a display name and often differs from the directory, so do not match on it. In a colocated file the directory is the one that holds the file. Read its `.py` / `.cpp` source and run the checks in step 5.

5. **Run these checks.** The message field names are the same in both SDKs:

   | Check | Detect in code | Compare against JSON |
   |-------|---------------|---------------------|
   | **Events** | `EventMetadata()` / `EventData` with `type_id = "id"`, appended to `msg.events`; the first argument of `add_event(` | IDs in code vs `"Events"[].ID` — must match verbatim, no `nx.nxai.` prefix on either side |
   | **Objects** | `msg.objects["id"]` assignments (`msg.objects[CLASS] = [...]`, `setdefault`, `emplace`) — the key is often a module constant, resolve it to its literal. In a C++ TENSOR processor, the class is also encoded in a tensor name such as `"bboxes-format:xyxysc;0:<class>"`: the text after the last `:` is a new class. | Class names in code vs `"Objects"[].ID` |
   | **UserSettings** | `msg.user_settings.get("key")` / `msg.user_settings["key"]`; the same reads on `message.user_settings` or on a function parameter named `user_settings` that receives it; f-string keys such as `f"prompt{index}"` | Keys in code vs `"UserSettings"[].name`. A Repeater entry has a `template` whose `name` has a `#` (for example `prompt#`); it covers the keys `prompt1`, `prompt2`, ... so match `prompt{i}` to it. |
   | **Schedule** | which parse the loop calls: `parse_objects_message`, `ObjectsPostProcessorMessage.from_dict`, a `create_*_from_bytes` factory that builds an objects message, or C++ `parseObjectsPostProcessorMessage` → `OBJECTS`; `parse_tensor_message` or C++ `parseTensorMessage` → `TENSOR`; `parse_image_message` → `IMAGE` (preprocessors only) | Matches `"Schedule"`; also flag `ReceiveInputImage` on a `TENSOR` schedule — that message never carries `ImageData` |
   | **ReceiveBinaryData** | any of: `msg.tensors` referenced; the raw `"Tensors"` key read off the unpacked msgpack dict; the message parsed by a `message_processing_utils` factory (`create_anpr_message_from_bytes`, `create_speed_message_from_bytes`), which reads `Tensors` in `parse_typed_message` | Code usage matches JSON flag. A `TENSOR` schedule always carries tensors, so the flag is not required there: do not report "in code, not declared" for it. |
   | **ReceiveInputImage** | `msg.image_data` referenced | Code usage matches JSON flag |
   | **Command path** | — | If `Command` is present, the file exists and is executable. On Windows the path ends in `.exe`, uses `/` or `\`, and the file only has to exist (`Test-Path -LiteralPath`). Skip this check for entries from colocated template files: their `Command` is a placeholder. |

6. **Classify** each check:
   - ✅ **OK** — JSON and code agree
   - ⚠️ **In code, not declared** — processor uses it but JSON doesn't declare it (will be silently dropped by the UI)
   - ℹ️ **Declared, not used** — JSON declares it but code doesn't use it (harmless but noisy)
   - ❌ **Path missing** — `Command` path does not exist on disk

7. **Print** one block per processor:
   ```
   <processor-dir>
   ─────────────────────────────────────────────
   Events            ✅ OK
   Objects           ⚠️  "car" used in code but not declared in JSON
   UserSettings      ✅ OK
   Schedule          ⚠️  code calls parse_objects_message but JSON declares "TENSOR"
   ReceiveBinaryData ✅ OK
   ReceiveInputImage ✅ OK
   Command path      ❌ file not found: /opt/.../postprocessor-python-<name>
   ```

8. For any ⚠️ or ❌ items, **print the exact fix** (e.g. the JSON snippet to add) and offer to apply it.
