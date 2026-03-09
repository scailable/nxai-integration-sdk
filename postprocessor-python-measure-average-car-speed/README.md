# Postprocessor Python Measure Average Car Speed

This postprocessor extends the ANPR (Automatic Number Plate Recognition) functionality to measure average car speed by tracking license plates across two cameras. It uses OCR processing to recognize license plates and calculates speed when the same plate is detected on both cameras.

## Overview

The postprocessor handles messages in a high-performance asynchronous flow:
1. **OCR Messages**: Receives raw logits (e.g., 9×37 float32 tensor), decodes them into text using argmax, and stores the result in a local cache (same as ANPR example).
2. **Detector Messages**: Receives object detections, retrieves the latest decoded OCR text from the cache for each object ID, and updates the message metadata with `recognized_text` attribute. Additionally, it tracks license plates across two cameras and calculates average speed when the same plate is detected on both cameras.
3. **Event Generation**: When speed is calculated, generates an event with license plate and speed information that can be used in Camera Rules.

## Key Features

- **Two-Camera License Plate Tracking**: Tracks license plates detected on two configured cameras and calculates average speed when the same plate appears on both.
- **Speed Calculation**: Calculates average speed using the distance between cameras and the time delta between detections.
- **Event Generation**: Generates events when speed is calculated, containing license plate number and speed in km/h.
- **Asynchronous OCR Processing**: Uses a worker pool for OCR decoding to prevent blocking the main communication loop (inherited from ANPR functionality).
- **Caching Logic**: Implements an `OcrCache` to store recognition results and a `SpeedMeasurementCache` to track license plates across cameras.
- **Automatic Cache Cleanup**: Automatically removes expired cache entries to prevent memory leaks.
- **Robust Configuration**: Optional INI at `../etc/plugin.measure-average-car-speed.ini`; first CLI argument is always the socket path (same as other postprocessors).
- **Nuitka Integration**: Fully compatible with Nuitka for compiling into a high-performance standalone binary.

## Model Output Format

By default, the postprocessor expects:
- **Shape**: (9, 37) — 9 character positions, 37 classes.
- **Character Mapping**: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ` (37 classes).
- **Tensor Name**: `Identity:0` (configurable via `.ini`).

## Example Models

The `example_models` directory at the repository root contains two ONNX models that can be used with this postprocessor in the AI Manager pipeline (same as the ANPR example):

- **License_Plate_Detector.onnx**: Detector model that produces bounding boxes for license plates. Use it as the detector in the pipeline.
- **License_Plate_OCR.onnx**: OCR model that produces raw character logits. Use it as the OCR model in the pipeline; the postprocessor decodes the logits and caches results by object ID.

When building the pipeline in the AI Manager, chain the **License_Plate_Detector** model with the **License_Plate_OCR** model in **Feature Extraction** mode, and set the extracted feature name to **LP** (License Plate). Pipeline configuration is described in **AI Manager Integration**.

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

Use **Release** configuration so that the installed binary and shared library match what the install step expects (nxai-utilities and the postprocessor should be built in the same configuration).

**Linux** (single-config):
```bash
# From the project root
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target postprocessor-python-measure-average-car-speed
```

**Windows** (multi-config; use `--config Release` for build and install):
```bash
# From the project root
mkdir -p build
cd build
cmake ..
cmake --build . --config Release --target postprocessor-python-measure-average-car-speed
```

On Linux you can alternatively use `make postprocessor-python-measure-average-car-speed` after configuring with `CMAKE_BUILD_TYPE=Release`.

The compiled standalone binary will be located at:
`build/postprocessor-python-measure-average-car-speed/postprocessor-python-measure-average-car-speed`

### Installation

Once compiled in **Release**, install the postprocessor and its dependencies to the AI Manager's postprocessors directory:

**Linux:**
```bash
# From the build directory
cmake --install . --component postprocessor-python-measure-average-car-speed
```

**Windows** (must use Release for install):
```bash
# From the build directory
cmake --install . --config Release --component postprocessor-python-measure-average-car-speed
```

This command installs:
- The postprocessor binary: `postprocessor-python-measure-average-car-speed` (or `.exe` on Windows)
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

The postprocessor can be started with the **socket path as an optional first argument** (e.g., `postprocessor-python-measure-average-car-speed.exe C:\path\to\socket.sock`). Speed measurement parameters (camera IDs, distance, timeout) can be set in two ways:

- **INI file**: `../etc/plugin.measure-average-car-speed.ini` (relative to the postprocessor binary). If the file is missing, the postprocessor still starts with default values and can receive configuration from the plugin UI.
- **Plugin UI**: Settings defined in `external_postprocessors.json` (see **AI Manager Integration** below) appear in the plugin UI. Values from the UI are sent with every inference message and **override** INI values for that message. You can use either INI, or UI, or both (UI overrides INI).

Create the configuration file in the AI Manager's configuration directory:

**Linux:**
```bash
nano /opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/etc/plugin.measure-average-car-speed.ini
```

**Windows:**
Create the file at:
```
C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix MetaVMS Media Server\nx_ai_manager\nxai_manager\etc\plugin.measure-average-car-speed.ini
```

### Configuration Options (`.ini`)

- `[common]`:
    - `log_level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR).
    - `socket_path`: Unix socket path for AI Manager communication.
    - `log_file`: Path to the log file.
- `[ocr]`:
    - `worker_count`: Number of parallel threads for OCR decoding.
    - `output_name`: The name of the output tensor to extract from the inference result.
- `[speed_measurement]` (optional if UI settings are used):
    - `camera_1_id`: Device ID (UUID or string) of the first camera.
    - `camera_2_id`: Device ID (UUID or string) of the second camera.
    - `distance_between_cameras_m`: Distance between the two cameras in meters.
    - `second_appearance_timeout_sec`: Timeout in seconds for cache entries. If a license plate is detected on one camera but not on the other within this timeout, the entry is removed from the cache.

At least one of INI or plugin UI must provide valid `camera_1_id`, `camera_2_id`, and `distance_between_cameras_m` for speed measurement to run; otherwise detector messages are still processed (e.g. OCR metadata) but speed is not calculated.

**Example configuration file:**
```ini
[common]
log_level=INFO
socket_path=/tmp/postprocessor-measure-average-car-speed.sock
log_file=/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/etc/plugin.measure-average-car-speed.log

[ocr]
worker_count=4
output_name=Identity:0

[speed_measurement]
camera_1_id=3645c7ee-ca91-e579-e753-1d85af1fd08c
camera_2_id=e3e9a385-7fe0-3ba5-5482-a86cde7faf48
distance_between_cameras_m=100.0
second_appearance_timeout_sec=30.0
```

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
When a **Detector Message** (containing bounding boxes) is received, the postprocessor:
1. Looks up the `ObjectIDs` in its OCR cache. If a match is found, it adds the recognized text to the object's metadata (same as ANPR example).
2. Extracts license plates from the message metadata (looking for `recognized_text` attribute).
3. For each license plate, adds it to the speed measurement cache with the camera ID and timestamp.
4. If the same license plate was previously detected on the other camera, calculates the average speed.
5. If speed is calculated, generates an event with the license plate and speed information.

**Example Detector Message (Input):**
```json
{
  "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
  "Timestamp": 1234567890,
  "BBoxes_xyxy": {
    "license_plate": [100, 150, 300, 200]
  },
  "ObjectsMetaData": {
    "license_plate": {
      "ObjectIDs": ["uuid-1234"],
      "AttributeKeys": [["recognized_text", "confidence"]],
      "AttributeValues": [["ABC123", "0.9850"]]
    }
  }
}
```

**Example Detector Message (Output with OCR data and event):**
```json
{
  "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
  "Timestamp": 1234567890,
  "BBoxes_xyxy": {
    "license_plate": [100, 150, 300, 200]
  },
  "ObjectsMetaData": {
    "license_plate": {
      "ObjectIDs": ["uuid-1234"],
      "AttributeKeys": [["recognized_text", "confidence"]],
      "AttributeValues": [["ABC123", "0.9850"]]
    }
  },
  "Events": [
    {
      "ID": "speed.measurement",
      "Caption": "Speed Measurement",
      "Description": "License plate ABC123 detected with average speed 72.0 km/h"
    }
  ]
}
```

### Speed Calculation

The postprocessor calculates average speed using the formula:
```
speed = distance_between_cameras_m / time_delta_seconds
```

Where:
- `distance_between_cameras_m` is the configured distance between the two cameras in meters.
- `time_delta_seconds` is the time difference between when the license plate was first detected on camera 1 and when it was detected on camera 2.

The speed is calculated in m/s and converted to km/h for display in events (multiplied by 3.6).

## AI Manager Integration

To enable this postprocessor, you must configure it in the AI Manager pipeline settings for **both models** (the Detector and the OCR model). This is necessary because the postprocessor needs to receive messages from both to maintain its cache and update metadata. Example ONNX models are provided in the `example_models` directory at the repository root (see **Example Models**).

Add the postprocessor to `external_postprocessors.json`. You can optionally add a **`Settings`** array so that camera IDs, distance, and timeout are editable in the plugin UI. Setting names must use the prefix `externalprocessor.`. Example with UI settings:

**Linux:**
```json
{
  "externalPostprocessors": [
    {
      "Name": "Measure-Average-Car-Speed",
      "Command": "/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/postprocessors/postprocessor-python-measure-average-car-speed",
      "SocketPath": "/tmp/postprocessor-measure-average-car-speed.sock",
      "ReceiveBinaryData": true,
      "Events": [
        {
          "ID": "speed.measurement",
          "Name": "Speed Measurement"
        }
      ],
      "Settings": [
        {
          "type": "TextField",
          "name": "externalprocessor.camera_1_id",
          "caption": "Camera 1 ID",
          "description": "Device ID (UUID) of the first camera.",
          "defaultValue": ""
        },
        {
          "type": "TextField",
          "name": "externalprocessor.camera_2_id",
          "caption": "Camera 2 ID",
          "description": "Device ID (UUID) of the second camera.",
          "defaultValue": ""
        },
        {
          "type": "SpinBox",
          "name": "externalprocessor.distance_between_cameras_m",
          "caption": "Distance between cameras (m)",
          "description": "Distance between the two cameras in meters.",
          "defaultValue": 100,
          "minValue": 1,
          "maxValue": 10000
        },
        {
          "type": "SpinBox",
          "name": "externalprocessor.second_appearance_timeout_sec",
          "caption": "Second appearance timeout (sec)",
          "description": "Timeout in seconds for matching a plate on the other camera.",
          "defaultValue": 60,
          "minValue": 1,
          "maxValue": 3600
        }
      ]
    }
  ]
}
```

**Windows:** use the same structure with your Windows `Command` and `SocketPath`. For example:
```json
{
  "externalPostprocessors": [
    {
      "Name": "Measure-Average-Car-Speed",
      "Command": "C:\\Windows\\System32\\config\\systemprofile\\AppData\\Local\\Network Optix\\Network Optix MetaVMS Media Server\\nx_ai_manager\\nxai_manager\\postprocessors\\postprocessor-python-measure-average-car-speed.exe",
      "SocketPath": "C:\\Windows\\Temp\\postprocessor-measure-average-car-speed.sock",
      "ReceiveBinaryData": true,
      "Events": [
        { "ID": "speed.measurement", "Name": "Speed Measurement" }
      ],
      "Settings": [
        {
          "type": "TextField",
          "name": "externalprocessor.camera_1_id",
          "caption": "Camera 1 ID",
          "description": "Device ID (UUID) of the first camera.",
          "defaultValue": ""
        },
        {
          "type": "TextField",
          "name": "externalprocessor.camera_2_id",
          "caption": "Camera 2 ID",
          "description": "Device ID (UUID) of the second camera.",
          "defaultValue": ""
        },
        {
          "type": "SpinBox",
          "name": "externalprocessor.distance_between_cameras_m",
          "caption": "Distance between cameras (m)",
          "description": "Distance between the two cameras in meters.",
          "defaultValue": 100,
          "minValue": 1,
          "maxValue": 10000
        },
        {
          "type": "SpinBox",
          "name": "externalprocessor.second_appearance_timeout_sec",
          "caption": "Second appearance timeout (sec)",
          "description": "Timeout in seconds for matching a plate on the other camera.",
          "defaultValue": 60,
          "minValue": 1,
          "maxValue": 3600
        }
      ]
    }
  ]
}
```

**Important Notes:**
- The `Events` section defines the events this postprocessor can generate. If an event is not listed here, it will not show up in the Nx Client Analytics Events.
- If you add the `Settings` array, the runtime sends current UI values in every inference message as `ExternalProcessorSettings`. The postprocessor merges them with INI (UI overrides INI) and updates the speed cache when values change. Changing camera IDs in the UI clears the in-memory plate cache; changing only distance or timeout does not.
- The shared library (`libnxai-c-utilities-shared.so` on Linux or `nxai-c-utilities-shared.dll` on Windows) must be in the same directory as the postprocessor binary. This is handled automatically by the installation process.
- `ReceiveBinaryData` must be set to `true` to receive OCR logits for processing.

### Event Format

When speed is calculated, the postprocessor generates an event with the following structure:
- **ID**: `"speed.measurement"` (fixed)
- **Caption**: `"Speed Measurement"`
- **Description**: `"License plate {license_plate} detected with average speed {speed} km/h"`

The speed is displayed in km/h for better readability, even though it's calculated in m/s internally.

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
tail -f /opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/etc/plugin.measure-average-car-speed.log
```

**Windows:**
Open the log file in a text editor or use PowerShell:
```powershell
Get-Content "C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix MetaVMS Media Server\nx_ai_manager\nxai_manager\etc\plugin.measure-average-car-speed.log" -Wait -Tail 50
```

Or check the log file path configured in your `plugin.measure-average-car-speed.ini` file.

## Development and Testing

Comprehensive test suites are provided:
- `test_postprocessor_python_measure_average_car_speed.py` - Tests for the main postprocessor logic
- `test_speed_cache.py` - Tests for the speed measurement cache

To run tests:
```bash
python3 -m unittest postprocessor-python-measure-average-car-speed.test_postprocessor_python_measure_average_car_speed postprocessor-python-measure-average-car-speed.test_speed_cache
```

## Dependencies

**Python packages** (installed automatically during build):
- Python 3.10+
- numpy
- msgpack
- nuitka (for building)

**System libraries** (installed via CMake):
- `libnxai-c-utilities-shared.so` (Linux) or `nxai-c-utilities-shared.dll` (Windows) - automatically installed with the postprocessor

**Internal packages** (from SDK):
- `inference` package - provides general message classes and utilities
- `message_processing_utils.general_detector` - provides DetectorMessage class
- `message_processing_utils.general_ocr` - provides OCR processing functionality (LogitsOcrEngine, OcrWorkerPool, OcrCache)
