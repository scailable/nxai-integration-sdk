Build and install a named post-processor or preprocessor using the project's CMake workflow.

## Arguments

`$ARGUMENTS` — a short name (e.g. `events-example`) or a full CMake target name (e.g. `postprocessor-python-events-example` or `preprocessor-python-image-example`). If the argument starts with neither `postprocessor-` nor `preprocessor-`, prepend `postprocessor-python-`. The C++ and cloud examples (`postprocessor-cpp-*`, `postprocessor-cloud-*`) need the full directory name.

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

3. **Check the build directory.** On Windows the build directory is `build-windows/`. If `build-windows/CMakeCache.txt` is missing, configure it with `cmd /c <repo root>\verify-windows.cmd` (by absolute path, from any shell; it enters the MSVC developer environment itself). It also builds the C++ targets and runs the `ctest` suite, and prints `verify-windows: PASS` on success. CLAUDE.md ("Verifying on Windows") says every enabled case must pass; `e2e-python-preproc-image-realloc-multi` is `DISABLED` and does not run. Stop on any failing case. On Linux: if `build/` does not exist at the repo root, or `build/CMakeCache.txt` is missing, run the full configure step first:
   ```shell
   mkdir -p build
   cd build
   python3 -m venv integrationsdk
   source integrationsdk/bin/activate
   cmake ..
   ```

4. **Build** the target. On Windows, `postprocessor-cpp-objects-example` and `postprocessor-cpp-tensors-example` are already built by `verify-windows.cmd`. Build any other target inside the MSVC developer environment, and always with `--target`: a bare `cmake --build` packages every Python processor with Nuitka.
   ```shell
   cmd /c ""<VS_PATH>\VC\Auxiliary\Build\vcvars64.bat" && cmake --build <repo root>\build-windows --target <target-name>"
   ```
   `<VS_PATH>` is the output of `vswhere.exe -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`, with `vswhere.exe` in `%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\`.

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

8. **Print** the installed binary path and the registration entry's `Name`. The entry is in `<processor-dir>/external_<postprocessors|preprocessors>.json`, next to the source. Tell the user to set its `Command` placeholder to the installed binary path (on Windows, the `.exe` in the install directory, for example `<VMS_BASE>\nx_ai_manager\postprocessors\<target-name>.exe`) and to add the entry to the installed `external_<postprocessors|preprocessors>.json` in the install directory.
