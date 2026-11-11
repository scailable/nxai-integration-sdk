Find and display the log file for a processor.

## Arguments

`$ARGUMENTS` — a short name (e.g. `events-example`) or a full directory name (e.g. `postprocessor-python-events-example` or `preprocessor-python-image-example`). If the argument starts with neither `postprocessor-` nor `preprocessor-`, prepend `postprocessor-python-`. The C++ and cloud examples (`postprocessor-cpp-*`, `postprocessor-cloud-*`) need the full directory name.

## Background

A v5 processor writes its logs to stdout and stderr. The AI Manager writes that output to the file named by the `LogFilepath` field of the processor's registration entry. There is no log file that the processor opens itself.

The entry is in `external_postprocessors.json` (postprocessors) or `external_preprocessors.json` (preprocessors).

## Steps

1. **Detect the active NX VMS installation.** Run these checks in order and stop at the first match:
   - `systemctl is-active networkoptix-metavms-mediaserver` → `VMS_BASE=/opt/networkoptix-metavms`
   - `systemctl is-active networkoptix-mediaserver` → `VMS_BASE=/opt/networkoptix`
   - `/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/` exists → use metavms
   - `/opt/networkoptix/mediaserver/var/nx_ai_manager/` exists → use witness
   - Neither found → skip the installed file and use only the colocated one.

   **Windows detection.** The platform is Windows when `$env:OS` is `Windows_NT` or `uname -s` starts with `MINGW` or `MSYS`. `VMS_BASE` is the mediaserver service profile. Check these directories in order and stop at the first with an `nx_ai_manager\` subdirectory:
   - `C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix MetaVMS Media Server` (NX Meta)
   - `C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix Media Server` (NX Witness)
   - Neither found → skip the installed file and use only the colocated one.

   There is no `mediaserver\var` level.

2. **Find the registration file and the entry.** The kind comes from the name prefix. Search in this order:
   - Installed: `<VMS_BASE>/mediaserver/var/nx_ai_manager/<postprocessors|preprocessors>/external_<postprocessors|preprocessors>.json` (on Windows `<VMS_BASE>\nx_ai_manager\<postprocessors|preprocessors>\external_<postprocessors|preprocessors>.json`)
   - Colocated: `<processor-dir>/external_<postprocessors|preprocessors>.json` in the repo.

   The file is a bare JSON array. Take the element whose `Command` ends with the processor directory name, with or without a `.exe` suffix, after a `/` or a `\` (for example `.../preprocessor-python-image-example`, or on Windows `C:/.../postprocessors/preprocessor-python-image-example.exe`). The `Name` field is a display name and often differs from the directory name. If no file has such an element, print that the processor is not registered and stop.

3. **Read `LogFilepath`** from the entry.

4. **If `LogFilepath` is missing**, print:
   ```
   ⚠️  The entry for <Name> has no LogFilepath. The processor's stdout and stderr are lost.
       Add "LogFilepath": "/tmp/<postprocess|preprocess>.<short name>.stdout.log" to the entry and restart the AI Manager.
   ```
   On Windows, suggest `C:\Windows\Temp\<postprocess|preprocess>.<short name>.stdout.log`, because the service runs as LocalSystem and `/tmp` does not exist. Then go to step 6.

5. **If `LogFilepath` is set**, show the last 50 lines of that file:
   ```shell
   tail -n 50 <LogFilepath>
   ```
   On Windows, read the file with `Get-Content -Tail 50 -LiteralPath <LogFilepath>`. If the file does not exist, say that the processor has not run yet or the path is not writable. Suggest `tail -f <LogFilepath>` for live output (on Windows `Get-Content -Wait -Tail 50 -LiteralPath <LogFilepath>`).

6. **Also check** the mediaserver's main log. Skip this step when step 1 found no VMS. The AI Manager writes its own lines there, with the tag `nx_ai_manager_plugin`. They include processor startup errors, for example `Failed to start external postprocessor <Name>`:
   ```shell
   <VMS_BASE>/mediaserver/var/log/main.log
   ```
   On Windows the file is `<VMS_BASE>\log\main.log`.
   Search it for lines that contain the entry's `Name` and print any found.
