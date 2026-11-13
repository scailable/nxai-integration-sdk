Generate the `external_postprocessors.json` registration entry for an existing post-processor by inspecting its source code.

## Arguments

`$ARGUMENTS` — short name (e.g. `vehicle-counter`) or full directory name (e.g. `postprocessor-python-vehicle-counter`).

## Steps

1. **Locate** the processor directory. Try `postprocessor-python-<name>/` first, then the literal argument.

2. **Read** the main `.py` file.

3. **Detect** what the processor needs by scanning the source. Count only code, not comments, docstrings or log strings. A key or ID is often a module constant (e.g. `msg.objects[INJECTED_CLASS]`); resolve it to its string literal.

   | Flag | Detect when |
   |------|-------------|
   | `ReceiveBinaryData: true` | `tensors` is read off the inbound message (e.g. `msg.tensors`); or the raw `"Tensors"` key is read off the unpacked msgpack dict (e.g. `raw_data.get("Tensors")`); or the message is parsed by a `message_processing_utils` factory (`create_anpr_message_from_bytes`, `create_speed_message_from_bytes`), which reads the `Tensors` block in `parse_typed_message` |
   | `ReceiveInputImage: true` | `image_data` is read off the inbound message (e.g. `msg.image_data`) |
   | `NoResponse: true` | there is no `send_response(` call on the IPC processor object. An HTTP handler's `self.send_response(...)` does not count. |

   **Objects** — all class keys the processor writes to the message's `objects` map (`msg.objects[...] = ...`, `msg.objects[...].append(`, `msg.objects.setdefault(...`). Count only writes in the processor's own file. A key that is only read (e.g. the detector's own class in `msg.objects.get(...)`), or a class whose existing objects only get attributes (e.g. through `update_original_object(...)`), is not a new class and is not declared.

   **Events** — all values assigned to `type_id` on an `EventMetadata()` (e.g. `event.type_id = "..."`), and the first argument of `add_event(` calls. Declare each ID verbatim, with no `nx.nxai.` prefix.

   **`UserSettings` keys** — all string literals accessed via `user_settings["..."]`, `user_settings.get("..."`, or `"..." in <msg>.user_settings` on the message's `user_settings` map. These are names only; the caption, description, control type and default value can't be inferred from code, so they still need a human to write the actual `UserSettings` control declarations (see `postprocessor-python-settings-example/settings_model.md` for the control grammar).

   **`Config` keys** — all string literals read off the variable assigned from `nxai_postprocessor_sdk.load_config(sys.argv)` (subscript, `.get(...)`, or `in <var>`), excluding `ProcessorName` and `LogLevel` (SDK-internal).

4. **Generate** the JSON element. `Schedule` and `LogFilepath` are always included; boolean flags default to `false` and are omitted when `false`; empty arrays are omitted.

   Derive the names from the directory name found in step 1, the same way `/new-processor` step 9 does:

   - `<processor-dir>` is the directory name, e.g. `postprocessor-python-anpr-example`.
   - `<name>` is `<processor-dir>` with a leading `postprocessor-python-` removed (e.g. `anpr-example`). If the directory does not start with that prefix, remove a leading `postprocessor-` instead. Keep the case and the hyphens.
   - `Name` is `<processor-dir>`.
   - `LogFilepath` is `/tmp/postprocess.<name>.stdout.log`.

   ```json
   {
       "Name": "<processor-dir>",
       "Command": "/path/to/build/<processor-dir>/<processor-dir>",
       "Schedule": "OBJECTS",
       "LogFilepath": "/tmp/postprocess.<name>.stdout.log",
       <detected flags>
       "Objects": [ { "ID": "<id>", "Name": "<Human Name>" } ],
       "Events": [ { "ID": "<id>", "Name": "<Human Name>" } ],
       "Config": {
           "LogLevel": "DEBUG",
           "<config key>": "<value>"
       }
   }
   ```

   Field order is fixed: `Name`, `Command`, `Schedule`, `LogFilepath`, the detected flags (in the order of the step 3 table), `Objects`, `Events`, `Config`. Put one `"Flag": true,` line per detected flag, each followed by a comma, and no line when there are no flags. Inside `Config`, `LogLevel` comes first, then the detected keys in the order of their first read in the source; put a comma after every entry except the last.

   **`Config` values.** Launch arguments are strings, so every value is a JSON string.
   - If the read has a default (`.get(key, <default>)`), use the default. Resolve a module constant to its literal. Write a string as is, and write a number as it appears in the source, inside quotes (`300.0` becomes `"300.0"`). Ignore a wrapper such as `float(...)` around the read.
   - If the read has no default (a subscript, an `in` test, or `.get(key)`), use the empty string `""`.

   **`UserSettings`.** Do not write a `UserSettings` key in this step, not even an empty array. A control cannot be derived from code, so a human adds the array later; the step 6 reminder tells them to.

   For `Objects` and `Events`, derive `Name` from the ID: take the last dot-separated segment, replace `_` with a space and title-case it (e.g. `anpr.plate_recognized` → `Plate Recognized`, `python_example` → `Python Example`).

5. **Add** the element to the array in `<processor-dir>/external_postprocessors.json` (create the file with `[ ... ]` if it doesn't exist yet). This is the file every migrated processor in this repo carries colocated with its source — treat it as the canonical registration for the processor.

6. **Print** the snippet and this reminder:

   ```
   Added to <processor-dir>/external_postprocessors.json.

   If any UserSettings keys were detected, declare their controls by hand in the
   "UserSettings" array (type, caption, description, defaultValue) — see
   postprocessor-python-settings-example/settings_model.md for the control grammar.

   Development mode (no build needed): replace the "Command" field with
   "Port": <n> and start the processor manually with --port <n>. The AI Manager
   then connects to 127.0.0.1:<n> instead of starting the processor. An entry
   with neither "Command" nor "Port" is skipped.

   For a production install, copy this element into the bare array in:
     <VMS_BASE>/mediaserver/var/nx_ai_manager/postprocessors/external_postprocessors.json
   and point "Command" at the installed binary path, e.g.:
     <VMS_BASE>/mediaserver/var/nx_ai_manager/postprocessors/<processor-dir>

   After editing the installed JSON:
     sudo chmod -R a+x <VMS_BASE>/mediaserver/var/nx_ai_manager/postprocessors/
     sudo service <VMS_SERVICE> restart
   ```
