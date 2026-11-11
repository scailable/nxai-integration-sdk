NX Integration SDK
==================

The NX Integration SDK provides tools and examples for clients to create applications which seamlessly integrates with the NX software stack.

See the subdirectories for further documentation.

Instructions that differ between Linux and Windows are in collapsible sections. Expand the one for your platform. Commands outside those sections are the same on both.

# Requirements

## The AI Manager

The examples need the AI Manager v5 plugin in the NX VMS. The public v4 engine does not run them. A v4 engine starts a processor from this repository, but the processor does not signal that it is ready, and the AI Manager drops it. If you have a v4 processor, the `/migrate-to-v5` command in `.claude/commands/` gives the steps to port it to v5.

## Download the integration SDK

You probably have the integration SDK already if you're looking at this readme, the command to get the full integration SDK is as follows:

```shell
git clone https://github.com/scailable/nxai-integration-sdk.git
```

On Windows, clone into a short directory such as `C:\s` instead (`git clone https://github.com/scailable/nxai-integration-sdk.git C:\s`). Windows limits a path to 260 characters, and the Nuitka build of a Python processor makes long paths, so a long clone path breaks the build.

If you have downloaded the sdk previously, you can also update to the latest version of the integration SDK while in the directory of the downloaded git repository.

```shell
git pull
```

## Requirements to compile the pre/postprocessors

These applications can be compiled natively on Linux, on any architecture, and on Windows.

<details open>
<summary><b>Linux</b></summary>

Install the compiler and CMake:

```shell
sudo apt install cmake g++
```

For the Python pre/postprocessors the following is also required:

```shell
sudo apt install python3-pip python3-venv patchelf
```

</details>

<details>
<summary><b>Windows</b></summary>

You need:

- Visual Studio 2022 or later (for example Visual Studio 2026) with the "Desktop development with C++" workload. It also supplies CMake and Ninja.
- Python 3.12 or later on `PATH`, for the Python processors.

On first use, Nuitka downloads a compiler cache and asks for permission in an interactive prompt. A build that CMake starts can hang on that prompt. Download the cache once, before the first build, in a scratch virtual environment:

```bat
python -m venv C:\nuitka-seed
C:\nuitka-seed\Scripts\pip install nuitka
cd C:\nuitka-seed
echo print(1) > seed.py
Scripts\python -m nuitka --assume-yes-for-downloads seed.py
```

</details>

## Requirements to run the pre/postprocessors

These applications can be run on any platform on which they can be compiled.

# How to use

## Getting Started

This project is CMake based, and all its modules can be compiled or gathered with CMake commands.

Because the different pre/postprocessors must be compiled for each hardware architecture this repository does not include pre-built binaries. All processors can be compiled manually.

First create a *build* directory in the project directory and configure CMake in it.

<details open>
<summary><b>Linux</b></summary>

On recent Ubuntu servers pip refuses to install into the system Python, so run `cmake` from a virtual environment. CMake creates its own environment, `build/venv`, for the processors.

```shell
cd nxai-integration-sdk/
mkdir -p build
cd build
python3 -m venv integrationsdk
source integrationsdk/bin/activate
cmake ..
```

</details>

<details>
<summary><b>Windows</b></summary>

Open the "x64 Native Tools Command Prompt" for your Visual Studio version (in the Start menu, "x64 Native Tools Command Prompt for VS 2022" for Visual Studio 2022, "x64 Native Tools Command Prompt for VS" for Visual Studio 2026) and run every command in it. CMake creates its own virtual environment in the build directory, so do not activate another one. The build directory is named `b` to keep paths short.

```bat
cd C:\s
mkdir b
cd b
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
```

</details>

The rest of this section is the same on both platforms. Run the commands in the build directory.

Build a single processor:

```shell
cmake --build . --target <processor-name>
```

On Windows the executable is `b\<processor-name>\<processor-name>.exe`.

Build all targets:

```shell
cmake --build .
```

This will build the default target, which includes all the example applications that are active in the `CMakeLists.txt`. On Windows this packages every Python processor with Nuitka, which takes a long time.

Build serially. A parallel build (`cmake --build . -j`) races on the shared Python virtual environment.

It is possible to build only specific examples, refer to the readme files in the subdirectories of those examples for specific instructions.

### Run the test suite on Windows

`verify-windows.cmd` in the repository root is the entry point. It enters the MSVC environment itself, configures, builds the C++ targets and runs the `ctest` suite. It builds only the C++ targets and does not package the Python processors. You can start it from any shell:

```bat
verify-windows.cmd
```

## Install the pre/postprocessors

All install paths below are relative to `VMS_BASE`, the install root of your VMS product. Declare it once for your shell session and every later command in this document reuses it.

<details open>
<summary><b>Linux</b></summary>

```shell
# NX Meta
export VMS_BASE=/opt/networkoptix-metavms
# NX Witness
# export VMS_BASE=/opt/networkoptix
```

Before installing make sure the target directories are writable:

```shell
sudo chmod 777 $VMS_BASE/mediaserver/var/nx_ai_manager/postprocessors/
sudo chmod 777 $VMS_BASE/mediaserver/var/nx_ai_manager/preprocessors/
```

</details>

<details>
<summary><b>Windows</b></summary>

`VMS_BASE` is the mediaserver service profile. This tree has no `mediaserver\var` level: the AI Manager directory sits directly under `VMS_BASE`, so the Windows counterpart of `$VMS_BASE/mediaserver/var/nx_ai_manager/postprocessors/` is `%VMS_BASE%\nx_ai_manager\postprocessors\`.

```bat
:: NX Meta
set "VMS_BASE=C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix MetaVMS Media Server"
:: NX Witness
:: set "VMS_BASE=C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix Media Server"
```

The service profile is normally writable only by administrators. Run the install commands below in an elevated prompt: right-click the "x64 Native Tools Command Prompt" and choose "Run as administrator".

</details>

CMake does not read `VMS_BASE`. At configure time it finds the installed product itself and sets the default install folders from it. If it finds no product, it uses the NX Meta paths. To install to a different folder, set both folders when you configure:

```shell
cmake .. -DINSTALL_DEST_POSTPROCESSORS=<path> -DINSTALL_DEST_PREPROCESSORS=<path>
```

To install a single processor to the default folder:

```shell
cmake --install . --component <processor-name>
```

To install the generated pre/postprocessor examples to the default pre/postprocessors folder:

```shell
cmake --build . --target install
```

## Defining the pre/postprocessors

Create a configuration file and add one entry per pre/postprocessor. The file is `external_postprocessors.json` in the postprocessors directory, or `external_preprocessors.json` in the preprocessors directory:

| Platform | Postprocessors directory | Preprocessors directory |
|---|---|---|
| Linux | `$VMS_BASE/mediaserver/var/nx_ai_manager/postprocessors/` | `$VMS_BASE/mediaserver/var/nx_ai_manager/preprocessors/` |
| Windows | `%VMS_BASE%\nx_ai_manager\postprocessors\` | `%VMS_BASE%\nx_ai_manager\preprocessors\` |

**The root of the file is a JSON array.** If the root is anything else the AI Manager logs `External postprocessors JSON root must be an array` and ignores the whole file, so a single malformed root disables every processor at once.

Each entry must define `Name`, and one of `Command` (the AI Manager launches the processor) or `Port` (the AI Manager connects to an already-running processor — see [Development](#development)). An entry with neither is skipped; an entry with both uses `Port`, without a warning, so set exactly one. Everything else is optional:

| Field | Meaning |
|---|---|
| `Schedule` | When the processor runs. Postprocessors: `OBJECTS` (default) or `TENSOR`. Preprocessors: `IMAGE` or `TENSOR`. |
| `ReceiveInputImage` | Adds an `ImageData` block to the message the processor receives. |
| `ReceiveBinaryData` | Adds a `Tensors` array of raw model outputs to that same message. |
| `NoResponse` | Fire-and-forget: the AI Manager does not wait for a reply. |
| `Objects` | `[{"ID", "Name"}]` — every object class the processor may add. Undeclared classes are dropped by the UI. |
| `Events` | `[{"ID", "Name"}]` — every event id the processor may emit, declared exactly as it is emitted (no `nx.nxai.` prefix on either side, unlike object classes). Undeclared ids are dropped. |
| `UserSettings` | Controls to expose in the NX plugin UI; their values are forwarded to the processor in each message. |
| `Config` | `{"key": "value"}` pairs handed to the processor as `--key value` on the command line at launch. |
| `LogFilepath` | File to capture the processor's merged stdout/stderr into. |
| `TimeoutMs` | How long the AI Manager waits for a response. |

`Command` is the full path to the executable. The AI Manager does not expand shell variables in it, so write the path out in full. The examples below use the NX Meta paths on Linux. For NX Witness, start each path with `/opt/networkoptix` instead. On Windows, `Command` is the full path of the `.exe` in the install directory, with each backslash doubled as JSON requires, for example `"C:\\Windows\\System32\\config\\systemprofile\\AppData\\Local\\Network Optix\\Network Optix MetaVMS Media Server\\nx_ai_manager\\postprocessors\\postprocessor-python-events-example.exe"`. A `LogFilepath` needs a Windows path too.

For example, the following file registers two post-processors:

```json
[
    {
        "Name": "Events-Example",
        "Command": "/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/postprocessors/postprocessor-python-events-example",
        "Schedule": "OBJECTS",
        "LogFilepath": "/tmp/postprocess.events-example.stdout.log",
        "Config": {
            "LogLevel": "DEBUG"
        },
        "Events": [
            {
                "ID": "ex.example.event",
                "Name": "Example event"
            }
        ]
    },
    {
        "Name": "Image-Example",
        "Command": "/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/postprocessors/postprocessor-python-image-example",
        "Schedule": "OBJECTS",
        "ReceiveInputImage": true,
        "Objects": [
            {
                "ID": "python_image_example",
                "Name": "Python Image Example"
            }
        ]
    }
]
```

Preprocessors use the same shape in `external_preprocessors.json`:

``` json
[
    {
        "Name": "Image-Mirror-Example",
        "Command": "/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/preprocessors/preprocessor-python-image-example",
        "Schedule": "IMAGE",
        "UserSettings": [
            {
                "type": "CheckBox",
                "name": "mirrorimage",
                "caption": "Mirror Image",
                "description": "Mirror the input image before the rest of the pipeline sees it",
                "defaultValue": true
            }
        ]
    }
]
```

Every post-processor example in this repository ships the `external_postprocessors.json` it expects next to its source, and every preprocessor example ships an `external_preprocessors.json`; copying that file is the quickest way to get a valid entry.

## Restarting the server

To (re)load your new pre/postprocessor, restart the NX Server.

<details open>
<summary><b>Linux</b></summary>

Before the restart, make sure the NX AI Manager can execute the pre/postprocessors:

```shell
sudo chmod -R a+x $VMS_BASE/mediaserver/var/nx_ai_manager/postprocessors/
sudo chmod -R a+x $VMS_BASE/mediaserver/var/nx_ai_manager/preprocessors/
```

Then restart the service. Its name follows the same product choice as `VMS_BASE` above.

```shell
# NX Meta
sudo service networkoptix-metavms-mediaserver restart
# NX Witness
# sudo service networkoptix-mediaserver restart
```

</details>

<details>
<summary><b>Windows</b></summary>

Windows has no execute bit to set. The service name is not fixed (a default install uses `defaultMediaServer`), so look it up by its binary, then restart it. Run both commands in an elevated PowerShell:

```powershell
Get-CimInstance Win32_Service | Where-Object { $_.PathName -match 'Network Optix' } | Format-Table Name, State
Restart-Service -Name <service-name>
```

</details>

## Selecting the pre/postprocessor

If the pre/postprocessor is defined correctly, its name should appear in the list of pre/postprocessors in the NX Cloud Pipelines UI. If it is selected in the pipeline settings then the NxAI Manager will send data to the pre/postprocessor and wait for its output.

# Development

For fast iterative development it is often desired not to fully recompile the postprocessor to have it reloaded.

Functionality has therefore been added to instruct the AI Manager not to start the external processor, but instead assume it's running.
The AI Manager will then attempt to send data to the external processor each frame, and wait for a response.
You can then start the processor yourself to immediately see its effect.
You can also print/manipulate data easily to get familiar with the format of the data that the processor receives.

Once you are satisfied with how your processor behaves, you can then compile and package it.

To use this mode, replace the `Command` field with a `Port` field in the processor definition:

```json
[
    {
        "Name": "Events-Example",
        "Port": 5010,
        "Schedule": "OBJECTS"
    }
]
```

The AI Manager then connects to `127.0.0.1:5010` instead of launching anything. Install the Python dependency of the processor, then start the processor yourself against the same port, from the repository root. The `PYTHONPATH` setting makes the SDK module in `nxai-utilities` visible to the processor.

<details open>
<summary><b>Linux</b></summary>

Activate the `integrationsdk` virtual environment from [Getting Started](#getting-started) first.

```shell
pip install msgpack
PYTHONPATH=nxai-utilities python3 postprocessor-python-events-example/postprocessor-python-events-example.py --port 5010
```

</details>

<details>
<summary><b>Windows</b></summary>

Use a plain Command Prompt and a virtual environment outside the build directory:

```bat
python -m venv C:\nxai-dev
C:\nxai-dev\Scripts\activate.bat
pip install msgpack
set PYTHONPATH=nxai-utilities
python postprocessor-python-events-example\postprocessor-python-events-example.py --port 5010
```

</details>

In the normal (`Command`) mode the AI Manager launches the processor with two shared-memory IPC sync descriptors as the first two arguments, followed by any `Config` pairs as `--key value` and finally `--ProcessorName <name>`; the SDK's `initialize_from_args()` handles both modes, so the same binary works either way.

# Licence

Copyright 2026, Network Optix, All rights reserved.
