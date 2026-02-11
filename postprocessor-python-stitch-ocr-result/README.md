# Postprocessor Python Stitch OCR Result

This postprocessor is designed to work with OCR models (like CCT - Compact Convolutional Transformer) that output raw character logits. It converts these logits into readable text and stitches the results back into the metadata of the original objects.

## Overview

The postprocessor handles two types of messages in a high-performance asynchronous flow:
1. **OCR Messages**: Receives raw logits (e.g., 9×37 float32 tensor), decodes them into text using argmax, and stores the result in a local cache.
2. **Detector Messages**: Receives object detections, retrieves the latest decoded OCR text from the cache for each object ID, and updates the message metadata with `recognized_text` and `confidence` attributes.

## Key Features

- **Asynchronous Processing**: Uses a worker pool for OCR decoding to prevent blocking the main communication loop.
- **Caching Logic**: Implements an `OcrCache` to store recognition results, allowing immediate responses to detector messages with the most recent OCR data.
- **Robust Configuration**: Optional INI at `../etc/plugin.stitch-ocr-result.ini`; first CLI argument is always the socket path (same as other postprocessors).
- **Nuitka Integration**: Fully compatible with Nuitka for compiling into a high-performance standalone binary.

## Model Output Format

By default, the postprocessor expects:
- **Shape**: (9, 37) — 9 character positions, 37 classes.
- **Character Mapping**: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ` (37 classes).
- **Tensor Name**: `Identity:0` (configurable via `.ini`).

## Requirements

To build and install this postprocessor, you need the following dependencies:

**Linux:**
- cmake (3.30 or higher)
- g++
- python3-pip
- python3-venv
- patchelf (required for Nuitka)

**Windows:**
- cmake (3.30 or higher)
- MSVC (Visual Studio) or MinGW
- python3-pip
- python3-venv

Install dependencies on Linux:
```bash
sudo apt install cmake g++ python3-pip python3-venv patchelf
```

## Installation and Build

This postprocessor is integrated into the SDK's CMake build system.

### Build Steps

```bash
# From the project root
mkdir -p build
cd build
cmake ..
cmake --build . --target postprocessor-python-stitch-ocr-result
```

**Note**: The `cmake --build` command is cross-platform and works on both Linux and Windows. On Linux, you can alternatively use `make postprocessor-python-stitch-ocr-result` if preferred.

The compiled standalone binary will be located at:
`build/postprocessor-python-stitch-ocr-result/postprocessor-python-stitch-ocr-result`

### Installation

Once compiled, install the postprocessor and its dependencies to the AI Manager's postprocessors directory:

```bash
# From the build directory
cmake --install . --component postprocessor-python-stitch-ocr-result
```

This command installs:
- The postprocessor binary: `postprocessor-python-stitch-ocr-result` (or `.exe` on Windows)
- The shared library required for communication:
  - **Linux**: `libnxai-c-utilities-shared.so`
  - **Windows**: `nxai-c-utilities-shared.dll`

Both files are installed to the same directory:
- **Linux**: `/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/postprocessors/`
- **Windows**: `C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix MetaVMS Media Server\nx_ai_manager\nxai_manager\postprocessors\`

The shared library is automatically installed alongside the binary and is required for `nxai_communication_utils` to function properly. It provides the low-level socket/pipe communication functions used by the postprocessor.

## Setting Permissions

**Linux only**: The application and library files must be readable and executable by the NX AI Manager. Set the appropriate permissions:

```bash
sudo chmod -R a+x /opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/postprocessors/
```

## Configuration

The postprocessor can be started with the **socket path as an optional first argument** (e.g., `postprocessor-python-stitch-ocr-result.exe C:\path\to\socket.sock`). Additional configuration options are read from a fixed path: `../etc/plugin.stitch-ocr-result.ini` (relative to the postprocessor binary). If the socket path is provided as the first argument, it takes priority over the value set in the .ini configuration file. Copy the example configuration file to the AI Manager's configuration directory:

**Linux:**
```bash
cp postprocessor-python-stitch-ocr-result/plugin.stitch-ocr-result.ini.example \
   /opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/etc/plugin.stitch-ocr-result.ini
```

**Windows:**
Copy the file to:
```
C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix MetaVMS Media Server\nx_ai_manager\nxai_manager\etc\plugin.stitch-ocr-result.ini
```

### Configuration Options (`.ini`)

- `[common]`:
    - `log_level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR).
    - `socket_path`: Unix socket path for AI Manager communication.
    - `log_file`: Path to the log file.
- `[ocr]`:
    - `worker_count`: Number of parallel threads for OCR decoding.
    - `output_name`: The name of the output tensor to extract from the inference result.

## Message Flow and Formats

The postprocessor communicates using the MessagePack protocol. It handles different message types as follows:

### 1. OCR Messages & Generic Messages
For **OCR Messages** (containing raw logits) and any other **Generic Messages** not explicitly handled, the postprocessor **immediately returns the original, unmodified message**. 
OCR decoding is performed asynchronously in the background to update the internal cache.

**Example OCR Message Structure (Input/Output):**
```json
{
  "OriginalObjectID": "uuid-1234",
  "BinaryOutputs": [
    {
      "Name": "Identity:0",
      "Data": "<binary_float32_logits>"
    }
  ]
}
```

### 2. Detector Messages
When a **Detector Message** (containing bounding boxes) is received, the postprocessor looks up the `ObjectIDs` in its cache. If a match is found, it adds the recognized text and confidence to the object's metadata.

**Example Detector Message (Input):**
```json
{
  "BBoxes_xyxy": {
    "license_plate": [100, 150, 300, 200]
  },
  "ObjectsMetaData": {
    "license_plate": {
      "ObjectIDs": ["uuid-1234"],
      "AttributeKeys": [[]],
      "AttributeValues": [[]]
    }
  }
}
```

**Example Detector Message (Output with OCR data):**
```json
{
  "BBoxes_xyxy": {
    "license_plate": [100, 150, 300, 200]
  },
  "ObjectsMetaData": {
    "license_plate": {
      "ObjectIDs": ["uuid-1234"],
      "AttributeKeys": [["recognized_text", "confidence"]],
      "AttributeValues": [["ABC1234", "0.985"]]
    }
  }
}
```

## AI Manager Integration

To enable this postprocessor, you must configure it in the AI Manager pipeline settings for **both models** (the Detector and the OCR model). This is necessary because the postprocessor needs to receive messages from both to maintain its cache and update metadata.

Add the postprocessor to the `externalPostprocessors` section in `external_postprocessors.json`:

**Linux:**
```json
{
  "Name": "Stitch-OCR-Result",
  "Command": "/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/postprocessors/postprocessor-python-stitch-ocr-result",
  "SocketPath": "/tmp/postprocessor-stitch-ocr-result.sock",
  "ReceiveBinaryData": true
}
```

**Windows:**
```json
{
  "Name": "Stitch-OCR-Result",
  "Command": "C:\\Windows\\System32\\config\\systemprofile\\AppData\\Local\\Network Optix\\Network Optix MetaVMS Media Server\\nx_ai_manager\\nxai_manager\\postprocessors\\postprocessor-python-stitch-ocr-result.exe",
  "SocketPath": "C:\\Windows\\SystemTemp\\postprocessor-stitch-ocr-result.sock",
  "ReceiveBinaryData": true
}

```

**Note**: The shared library (`libnxai-c-utilities-shared.so` on Linux or `nxai-c-utilities-shared.dll` on Windows) must be in the same directory as the postprocessor binary. This is handled automatically by the installation process.

### Restarting the Server

After installation and configuration, restart the NX Server to load the new postprocessor:

**Linux:**
```bash
sudo service networkoptix-metavms-mediaserver restart
```

**Windows:**
Restart the NX Server using the Tray Tool (icon in the Windows system tray). Right-click the icon and select "Restart Server" or use the appropriate menu option.

## Viewing Logs

The postprocessor writes logs to the file specified in the configuration. To view logs in real-time:

**Linux:**
```bash
tail -f /opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/etc/plugin.stitch-ocr-result.log
```

**Windows:**
Open the log file in a text editor or use PowerShell:
```powershell
Get-Content "C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix MetaVMS Media Server\nx_ai_manager\nxai_manager\etc\plugin.stitch-ocr-result.log" -Wait -Tail 50
```

Or check the log file path configured in your `plugin.stitch-ocr-result.ini` file.

## Development and Testing

A comprehensive test suite is provided in `test_postprocessor_python_stitch_ocr_result.py`. To run tests:

```bash
python3 -m unittest test_postprocessor_python_stitch_ocr_result.py
```

## Dependencies

**Python packages** (installed automatically during build):
- Python 3.10+
- numpy
- msgpack
- nuitka (for building)

**System libraries** (installed via CMake):
- `libnxai-c-utilities-shared.so` (Linux) or `nxai-c-utilities-shared.dll` (Windows) - automatically installed with the postprocessor
