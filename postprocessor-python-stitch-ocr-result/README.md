# Postprocessor Python Stitch OCR Result

This postprocessor is designed to work with OCR models (like CCT - Compact Convolutional Transformer) that output raw character logits. It converts these logits into readable text and stitches the results back into the metadata of the original objects.

## Overview

The postprocessor handles two types of messages in a high-performance asynchronous flow:
1. **OCR Messages**: Receives raw logits (e.g., 9×37 float32 tensor), decodes them into text using argmax, and stores the result in a local cache.
2. **Detector Messages**: Receives object detections, retrieves the latest decoded OCR text from the cache for each object ID, and updates the message metadata with `recognized_text` and `confidence` attributes.

## Key Features

- **Asynchronous Processing**: Uses a worker pool for OCR decoding to prevent blocking the main communication loop.
- **Caching Logic**: Implements an `OcrCache` to store recognition results, allowing immediate responses to detector messages with the most recent OCR data.
- **Robust Configuration**: Supports INI-based configuration for logging levels, socket paths, and worker counts.
- **Nuitka Integration**: Fully compatible with Nuitka for compiling into a high-performance standalone binary.

## Model Output Format

By default, the postprocessor expects:
- **Shape**: (9, 37) — 9 character positions, 37 classes.
- **Character Mapping**: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ` (37 classes).
- **Tensor Name**: `Identity:0` (configurable via `.ini`).

## Installation and Build

This postprocessor is integrated into the SDK's CMake build system.

### Build Steps

```bash
# From the project root
mkdir -p build
cd build
cmake ..
make postprocessor-python-stitch-ocr-result
```

The compiled standalone binary will be located at:
`build/postprocessor-python-stitch-ocr-result/postprocessor-python-stitch-ocr-result`

## Configuration

The postprocessor uses an `.ini` file for configuration. Copy the example file to the AI Manager's configuration directory:

```bash
cp postprocessor-python-stitch-ocr-result/plugin.stitch-ocr-result.ini.example \
   /opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/etc/plugin.stitch-ocr-result.ini
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

```json
{
  "Name": "Stitch-OCR-Result",
  "Command": "/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/postprocessors/postprocessor-python-stitch-ocr-result",
  "SocketPath": "/tmp/postprocessor-stitch-ocr-result.sock",
  "ReceiveBinaryData": true
}
```

## Development and Testing

A comprehensive test suite is provided in `test_postprocessor_python_stitch_ocr_result.py`. To run tests:

```bash
python3 -m unittest test_postprocessor_python_stitch_ocr_result.py
```

## Dependencies

- Python 3.10+
- numpy
- msgpack
- nuitka (for building)
