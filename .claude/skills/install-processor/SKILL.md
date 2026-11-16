---
name: install-processor
description: Build and install a processor into the local NX VMS, then offer a service restart.
argument-hint: "<processor name>"
disable-model-invocation: true
---

Build and install a named post-processor or preprocessor using the project's CMake workflow.

## Arguments

The processor name is the text after the skill name, or the processor the request names. If neither gives one, ask. It is a short name (e.g. `events-example`) or a full CMake target name (e.g. `postprocessor-python-events-example` or `preprocessor-python-image-example`). If the argument starts with neither `postprocessor-` nor `preprocessor-`, prepend `postprocessor-python-`. The C++ and cloud examples (`postprocessor-cpp-*`, `postprocessor-cloud-*`) need the full directory name.

## Steps

1. **Detect the active NX VMS installation.** The platform is Windows when `$env:OS` is `Windows_NT` or `uname -s` starts with `MINGW` or `MSYS`. On Windows, use the Windows detection below. On Linux, run these checks in order, stopping at the first match:
   - `systemctl is-active networkoptix-metavms-mediaserver` → `VMS_BASE=/opt/networkoptix-metavms`, `VMS_SERVICE=networkoptix-metavms-mediaserver`
   - `systemctl is-active networkoptix-mediaserver` → `VMS_BASE=/opt/networkoptix`, `VMS_SERVICE=networkoptix-mediaserver`
   - `/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/` exists → use metavms values
   - `/opt/networkoptix/mediaserver/var/nx_ai_manager/` exists → use witness values
   - Neither found → print a warning and use the NX Meta values. This is the default of the CMake install destination. The build works. The install needs an existing destination directory and root rights to write to it, and the restart in step 7 has no service to restart.

   **Windows detection.** `VMS_BASE` is the mediaserver service profile. Check these directories in order and stop at the first with an `nx_ai_manager\` subdirectory:
   - `C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix MetaVMS Media Server` (NX Meta)
   - `C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix Media Server` (NX Witness)
   - Neither found → print a warning and use the NX Meta path. This is the default of the CMake install destination. The build works. The install needs an existing destination directory, and the restart in step 7 has no service to restart.

   There is no `mediaserver\var` level: the install directories are `<VMS_BASE>\nx_ai_manager\postprocessors\` and `<VMS_BASE>\nx_ai_manager\preprocessors\`. Find the service by its binary, because its name is not fixed (a default install uses `defaultMediaServer`):
   ```powershell
   Get-CimInstance Win32_Service | Where-Object { $_.PathName -match 'Network Optix' } | Format-Table Name, State, StartName
   ```
   Set `VMS_SERVICE` to its `Name`. With no match, leave it empty.

2. **Determine the CMake target name and the kind.** If the argument starts with `postprocessor-` or `preprocessor-`, use it as is. Otherwise prepend `postprocessor-python-`. The prefix gives the kind: `postprocessor-` is a postprocessor, `preprocessor-` is a preprocessor. Verify the source directory exists; stop with an error if not.

3. **Check the build directory.** On Windows the build directory is `build-windows/`, and `build-windows.cmd` in step 4 configures it when `build-windows/CMakeCache.txt` is missing. On Linux: if `build/` does not exist at the repo root, or `build/CMakeCache.txt` is missing, run the full configure step first:
   ```shell
   mkdir -p build
   cd build
   python3 -m venv integrationsdk
   source integrationsdk/bin/activate
   cmake ..
   ```
   If the configure fails because a tool is missing (`cmake`, a C++ compiler, `python3-venv`, the MSVC toolset; on Windows `build-windows.cmd` exits 2), stop and tell the user to run `/setup`. It checks the toolchain and, on Linux, installs what is missing.

4. **Build** the target. On Windows, run `build-windows.cmd` by absolute path, from PowerShell or Git Bash. It enters the MSVC developer environment itself, configures `build-windows/` if needed, and builds only the named target:
   ```shell
   cmd /c <repo root>\build-windows.cmd <target-name>
   ```
   In Git Bash write `cmd //c`: Git Bash rewrites `/c` as a path, so `cmd /c` builds nothing and exits 0. Check that the target binary exists after the build.

   On Linux:
   ```shell
   cd build && cmake --build . --target <target-name>
   ```
   If it fails, print the last 40 lines of build output and stop.

5. **Install** the target:
   ```shell
   cd build && sudo cmake --install . --component <target-name>
   ```
   Use `sudo` when the destination is a root-owned directory under `/opt/`, as in step 6.
   On Windows, run `cmake --install build-windows --component <target-name>` with no `sudo`, from an elevated PowerShell. Without elevation it works only when the profile's ACL lets your user write; on `Access is denied`, tell the user to repeat the command in an elevated shell. Read the cached destination with `findstr INSTALL_DEST build-windows\CMakeCache.txt`.
   CMake installs to the cached `INSTALL_DEST_POSTPROCESSORS` or `INSTALL_DEST_PREPROCESSORS` value, which can differ from `<VMS_BASE>`. Read the value with `grep INSTALL_DEST build/CMakeCache.txt`. To use another destination, configure with `-DINSTALL_DEST_POSTPROCESSORS=<path>` or `-DINSTALL_DEST_PREPROCESSORS=<path>`.

6. **Fix permissions.** On Windows, skip this step with the reason `no execute bit on Windows`. On Linux, fix permissions on the directory CMake installed to (the cached `INSTALL_DEST_*` value for the kind; by default `<VMS_BASE>/mediaserver/var/nx_ai_manager/<postprocessors|preprocessors>/`):
   ```shell
   sudo chmod -R a+x <install destination>
   ```

7. **Ask** the user whether to restart the AI Manager service now. Only run the restart command if the user explicitly confirms. On Windows, with an empty `VMS_SERVICE`, skip this step and print `no mediaserver service installed`; otherwise the restart needs an elevated PowerShell:
   ```powershell
   Restart-Service -Name <VMS_SERVICE>
   ```
   On Linux:
   ```shell
   sudo service <VMS_SERVICE> restart
   ```

8. **Print** the installed binary path and the registration entry's `Name`. The entry is in `<processor-dir>/external_<postprocessors|preprocessors>.json`, next to the source. Tell the user to set its `Command` placeholder to the installed binary path (on Windows, the `.exe` in the install directory, for example `<VMS_BASE>\nx_ai_manager\postprocessors\<target-name>.exe`) and to add the entry to the installed `external_<postprocessors|preprocessors>.json` in the install directory. If that installed file does not exist, tell the user to run `/setup`, which creates it as an empty array.
