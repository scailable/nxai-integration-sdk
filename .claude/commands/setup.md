Set up the development environment for the NX AI Integration SDK from scratch. Run this once on a fresh machine or after cloning the repo.

## Steps

### 1. Identify the installed NX VMS and detect active product

This SDK supports **NX Meta** (`networkoptix-metavms`) and **NX Witness** (`networkoptix`). The install paths differ only in the base directory. Other Network Optix-based products (DW Spectrum, Hanwha Wisenet WAVE, etc.) are not supported — they use different paths and package names.

**a. Check for unsupported products first**

On Windows, skip this check (`dpkg` does not exist there); the profile directories of step 1b are the check.

```shell
dpkg -l | grep -E "digitalwatchdog|hanwha|wisenet|dwspectrum"
```
If any match is found, stop and print:
```
❌ Unsupported VMS detected: <package name>
   This SDK supports networkoptix-metavms (NX Meta) and networkoptix (NX Witness) only.
```

**b. Detect which NX product is active.** The platform is Windows when `$env:OS` is `Windows_NT` or `uname -s` starts with `MINGW` or `MSYS`. On Windows, use the Windows detection below. On Linux, run these checks in order and stop at the first match:

1. `systemctl is-active networkoptix-metavms-mediaserver` → if active, set `VMS_BASE=/opt/networkoptix-metavms`, `VMS_SERVICE=networkoptix-metavms-mediaserver`, `VMS_NAME="NX Meta"`
2. `systemctl is-active networkoptix-mediaserver` → if active, set `VMS_BASE=/opt/networkoptix`, `VMS_SERVICE=networkoptix-mediaserver`, `VMS_NAME="NX Witness"`
3. `ls /opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/` → if exists, use metavms values above
4. `ls /opt/networkoptix/mediaserver/var/nx_ai_manager/` → if exists, use witness values above
5. None found → print warning and continue (build steps still work without a VMS install):
   ```
   ⚠️  No NX VMS installation found.
       Build steps will work, but registration and install steps require it.
       Set VMS_BASE=/opt/networkoptix-metavms or /opt/networkoptix manually if needed.
   ```

**Windows detection.** `VMS_BASE` is the mediaserver service profile. Check these directories in order and stop at the first with an `nx_ai_manager\` subdirectory:

1. `C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix MetaVMS Media Server` → `VMS_NAME="NX Meta"`
2. `C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix Media Server` → `VMS_NAME="NX Witness"`
3. Neither found → print the same warning as Linux case 5, with `VMS_BASE` set manually to one of these two paths.

There is no `mediaserver\var` level: the AI Manager directories are `<VMS_BASE>\nx_ai_manager\postprocessors\` and `<VMS_BASE>\nx_ai_manager\preprocessors\`.

Find the service by its binary, because its name is not fixed (a default install uses `defaultMediaServer`):
```powershell
Get-CimInstance Win32_Service | Where-Object { $_.PathName -match 'Network Optix' } | Format-Table Name, State, StartName
```
Set `VMS_SERVICE` to its `Name`. If no service matches, leave `VMS_SERVICE` empty. Later steps then skip the restart and print `no mediaserver service installed`.

Print the detected product:
```
✅ Detected: NX Meta  →  /opt/networkoptix-metavms  (service: networkoptix-metavms-mediaserver)
```
or
```
✅ Detected: NX Witness  →  /opt/networkoptix  (service: networkoptix-mediaserver)
```

or on Windows
```
✅ Detected: NX Witness  →  C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix Media Server  (service: <VMS_SERVICE or none>)
```

Use `VMS_BASE` and `VMS_SERVICE` for all paths in the remaining steps.

### 2. Check nxai-utilities

Verify that the Python SDK is present. `nxai-utilities/` is part of this repository, and without it nothing imports:

```shell
ls nxai-utilities/nxai_postprocessor_sdk.py
```

If the file is missing, the checkout is incomplete: restore it with `git checkout -- nxai-utilities`.

Report if this succeeds or fails and stop on failure.

### 3. Install system dependencies

**Windows:** check these instead of the Linux packages. `patchelf`, `apt` and `sudo` do not exist there.

```powershell
cmake --version
python --version
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
```
If one is missing, tell the user to install it (CMake, Python 3, Visual Studio with the C++ toolset) and stop. Then continue at step 4.

**Linux:** check whether each required package is already installed before running apt:

```shell
cmake --version
g++ --version
python3 --version
pip3 --version
python3 -m venv --help
patchelf --version
```

Install any that are missing:

```shell
sudo apt install cmake g++ python3-pip python3-venv patchelf
```

### 4. Configure the build directory

**Windows:** the build directory is `build-windows/`, and `verify-windows.cmd` in the repo root configures it. The script enters the MSVC developer environment itself. Run it by absolute path from any shell; do not wrap it in `vcvars`:
```shell
cmd /c <repo root>\verify-windows.cmd
```
It builds the C++ targets and runs the `ctest` suite, so it also proves the toolchain. It prints `verify-windows: PASS` and exits 0 when every case passes. CLAUDE.md ("Verifying on Windows") says every enabled case must pass; `e2e-python-preproc-image-realloc-multi` is `DISABLED` and does not run. Stop on any failing case. If `build-windows/CMakeCache.txt` already exists, skip the run and tell the user the build is configured. Check the cached destinations (`findstr INSTALL_DEST build-windows\CMakeCache.txt`) against `VMS_BASE`. Then continue at step 5.

**Linux:** if `build/` does not exist or `build/CMakeCache.txt` is missing, run the full configure sequence:

```shell
mkdir -p build
cd build
python3 -m venv integrationsdk
source integrationsdk/bin/activate
cmake ..
```

The `integrationsdk` venv gives CMake a Python it can install packages into. Recent Ubuntu servers refuse `pip install` into the system Python. Each Python processor then builds its own venv in `build/venv`.

If CMakeCache.txt already exists, skip the configure and tell the user the build is already configured. Tell them to check the cached `INSTALL_DEST_POSTPROCESSORS` and `INSTALL_DEST_PREPROCESSORS` values (`grep INSTALL_DEST build/CMakeCache.txt`) against the detected `VMS_BASE`. If they differ, re-configure with a new build directory, or pass `-DINSTALL_DEST_POSTPROCESSORS=<path> -DINSTALL_DEST_PREPROCESSORS=<path>`.

### 5. Check that the NX AI Manager is present

If step 1 found no VMS, skip steps 5, 6 and 7.

Verify the postprocessors and preprocessors directories exist at the detected `VMS_BASE`:

```shell
ls <VMS_BASE>/mediaserver/var/nx_ai_manager/postprocessors/
ls <VMS_BASE>/mediaserver/var/nx_ai_manager/preprocessors/
```
On Windows the paths are `<VMS_BASE>\nx_ai_manager\postprocessors\` and `<VMS_BASE>\nx_ai_manager\preprocessors\`.

If one does not exist, print this warning and continue:

```
⚠️  The NX AI Manager <postprocessors|preprocessors> directory was not found under <VMS_BASE>.
    The AI Manager must be installed before processors can be registered or run.
    Re-run /setup after installing the mediaserver package.
```

### 6. Ensure the processor directories are executable

Skip this step when step 1 found no VMS. On Windows, skip it with the reason `no execute bit on Windows`: `chmod` and `sudo` do not exist. An elevated PowerShell replaces `sudo`: writing to the service profile needs one unless the profile's ACL already lets your user write, and restarting the service always needs one. Check which shell you have:
```powershell
([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
```
`False` is a note for the summary, not a failure.

On Linux, for each directory that exists:

```shell
sudo chmod -R a+x <VMS_BASE>/mediaserver/var/nx_ai_manager/postprocessors/
sudo chmod -R a+x <VMS_BASE>/mediaserver/var/nx_ai_manager/preprocessors/
```

### 7. Create the registration files if they do not exist

Skip this step when step 1 found no VMS.

Paths:
- `<VMS_BASE>/mediaserver/var/nx_ai_manager/postprocessors/external_postprocessors.json`
- `<VMS_BASE>/mediaserver/var/nx_ai_manager/preprocessors/external_preprocessors.json`

On Windows, drop the `mediaserver\var` level: `<VMS_BASE>\nx_ai_manager\postprocessors\external_postprocessors.json` and `<VMS_BASE>\nx_ai_manager\preprocessors\external_preprocessors.json`.

If a file is absent and its directory exists, create it with an empty array:

```json
[]
```

The root of a registration file is a bare JSON array. A v4-style
`{"externalPostprocessors": [...]}` wrapper does not fail one entry — the AI Manager
rejects the whole file, so every registered processor disappears at once.

If a file already exists, leave it untouched and report its current content.

### 8. Print a summary and next steps

```
✅ VMS:             <VMS_NAME> (<VMS_BASE>)
✅ nxai-utilities:  nxai_postprocessor_sdk.py present
✅ System packages: cmake, g++, python3, pip3, python3-venv, patchelf
✅ Build dir:       build/ configured (CMakeCache.txt present)
✅ AI Manager dirs: postprocessors/ and preprocessors/ found
✅ Permissions:     postprocessors/ and preprocessors/ executable
✅ JSON files:      external_postprocessors.json and external_preprocessors.json ready
```

Print the warning variant (⚠️) instead of ✅ for each line whose step failed or was skipped. With no VMS, the VMS, AI Manager dirs, Permissions and JSON files lines are ⚠️ "skipped, no VMS found". On Windows, the System packages line lists cmake, python and the MSVC toolset, the Build dir line names `build-windows/`, and the Permissions line is ⚠️ "skipped, no execute bit on Windows".

You're set up. Typical next steps:

  /new-processor <name> [type]     — scaffold a new post-processor
  /register-processor <name>       — generate its JSON registration entry
  /install-processor <name>        — build and install it
  /check-registration              — verify JSON matches source code

Development shortcut (no build needed):
  Replace "Command" with "Port": <n> in the JSON entry, then start the
  processor yourself with --port <n>. The AI Manager connects to
  127.0.0.1:<n>. An entry with neither "Command" nor "Port" is skipped.
  See .claude/docs/external_postprocessors_json.md for details.
```
