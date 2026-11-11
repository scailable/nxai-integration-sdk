"""
Image Postprocessor (v5 SDK)

Reads the input frame delivered at the root `ImageData` key (gated by
`ReceiveInputImage` in the registration JSON), decodes it with Pillow, and
injects a single "python_image_example" detection carrying the decode result
as attributes.

Behavior-preserving port of the v4 example's byte-sum check, adapted to the
v5 wire: no more separate image-header message / SHM read, and the injected
box is normalized xywh (v5 objects are frame-relative) rather than pixel xyxy.

Summing the frame's bytes, as the original v4-ported check did, is a checksum
over the blob: it is identical for a clean frame and one with its rows
shifted, its channels swapped, or garbage spliced into the middle, as long as
the total is unchanged. Decoding the buffer into a real image with
`_decode_frame` below catches the wrong-shape cases outright (the reshape
raises); the pixel-level ones aren't wrong by any number, so they need a human
looking at the frame -- see the preview stream below for that.
"""

import io
import logging
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import numpy as np
from PIL import Image

import nxai_postprocessor_sdk

# ============================================================================
# Configuration
# ============================================================================

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "image-example"

# Object type this example injects. Unique to this example rather than a shared
# "test", so a readback can tell which shipped example produced a track.
INJECTED_CLASS = "python_image_example"

# The frame arrives as uint8 RGB, laid out HWC or CHW per `image.layout`.
_CHANNELS = 3

# 0 lets the OS pick a free port, printed once the server is listening; fixed
# via the "stream_port" launch-time Config key if a stable URL is wanted.
DEFAULT_STREAM_PORT = 0
stream_port = DEFAULT_STREAM_PORT

# UUID of the injected object's track, per device: assigned on the first
# handle_message call for a device and reused on every subsequent call for it, so
# the track persists across frames. Keyed by device because a track belongs to
# the device that minted it -- the plugin resolves an incoming ID against that
# device's own track table, so reusing one camera's ID on another is rejected
# with "Could not find object ID" and the object is dropped.
global test_track_ids
test_track_ids = {}

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(
    msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage,
) -> nxai_postprocessor_sdk.ObjectsPostProcessorMessage:
    """Decode the input frame, publish it to the preview stream, and inject a detection.

    Args:
        msg: Parsed ObjectsPostProcessorMessage from NXAI runtime, with
            `image_data` populated (ReceiveInputImage is set in the JSON).

    Returns:
        The same message, with `msg.objects[INJECTED_CLASS]` set to a single
        normalized detection whose `ImageDecodeOk` attribute records whether
        the frame decoded cleanly (and `ImageDecodeError` why not, if so).
    """
    try:
        frame = _decode_frame(msg.image_data)
        decode_error = None
    except ValueError as e:
        # Only the shape-mismatch case _decode_frame itself raises: a malformed but
        # present frame. A missing ImageData block (ReceiveInputImage declared but no
        # block arrived) is a different failure entirely and is intentionally left to
        # propagate -- see tests/e2e/CMakeLists.txt's e2e-python-negative-no-image for
        # why examples shouldn't quietly paper over that one.
        frame = None
        decode_error = str(e)
        logger.error(f"Failed to decode input frame: {decode_error}")

    if frame is not None:
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG", quality=85)
        _publish_frame(buffer.getvalue())

    obj = nxai_postprocessor_sdk.ObjectMetadata()
    test_track_id = test_track_ids.get(msg.device_id)
    if test_track_id is None:
        test_track_id = str(uuid.uuid4())
        test_track_ids[msg.device_id] = test_track_id
        obj.new_track_id = test_track_id  # create the track with our chosen UUID
    else:
        obj.track_id = test_track_id  # continue this device's existing track
    obj.x = 0.1
    obj.y = 0.1
    obj.width = 0.1
    obj.height = 0.1
    obj.confidence = 1.0
    obj.attributes["ImageDecodeOk"] = str(decode_error is None)
    if decode_error is not None:
        obj.attributes["ImageDecodeError"] = decode_error

    msg.objects[INJECTED_CLASS] = [obj]
    logger.debug(f"Injected '{INJECTED_CLASS}' object with ImageDecodeOk={decode_error is None}")

    return msg


def run_message_loop(processor: nxai_postprocessor_sdk.ExternalPostprocessor) -> None:
    """Serve messages until the engine sends a graceful EXIT signal.

    `wait_for_message()` returns None both on a plain timeout and on an IPC EXIT
    signal; only `processor.exit_requested` tells them apart. A loop that just
    `continue`s on None regardless of that flag never returns, and the AI Manager
    kills the process after its grace period instead of it exiting cleanly.

    Args:
        processor: An initialized ExternalPostprocessor.
    """
    while True:
        logger.debug("Waiting for input message")
        try:
            message_bytes = processor.wait_for_message(timeout_ms=5000)
            if message_bytes is None:
                if processor.exit_requested:
                    logger.info("Received graceful exit signal, shutting down")
                    return
                logger.debug("Timeout waiting for message")
                continue
            logger.debug(f"Received message ({len(message_bytes)} bytes)")
            objects_message = processor.parse_objects_message(message_bytes)
            if objects_message is None:
                logger.error("Failed to parse message")
                continue
            processed_msg = handle_message(objects_message)
            response_bytes = processor.write_objects_message(processed_msg)
            if response_bytes is None:
                logger.error("Failed to serialize message")
                continue
            if not processor.send_response(response_bytes):
                logger.error("Failed to send response")
                break
            logger.debug("Response sent successfully")
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            break


# ============================================================================
# Image preview stream (MJPEG)
# ============================================================================
#
# Writing each decoded frame to its own file would mean thousands of files
# per minute and no way to just watch the feed. Instead the latest decoded
# frame is kept in memory and served over HTTP as an MJPEG stream
# (`multipart/x-mixed-replace`): open the printed URL in a browser and the
# image updates live, with no polling and no per-frame disk write. The
# format is natively understood by every mainstream browser, so this needs
# nothing beyond the standard library's `http.server` plus Pillow for the
# JPEG encode -- no platform-specific code, so it behaves the same on
# Windows and Linux.


def start_stream_server(port: int = DEFAULT_STREAM_PORT) -> ThreadingHTTPServer:
    """Start the MJPEG preview server on a background thread and log its URL.

    Binds to loopback only -- this is a local debugging aid, not something to
    expose on the network.
    """
    server = ThreadingHTTPServer(("127.0.0.1", port), _MjpegRequestHandler)
    threading.Thread(target=server.serve_forever, name="mjpeg-preview", daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    logger.info(f"Image preview stream: {url} (open in a browser to watch frames live)")
    return server


# ============================================================================
# Private helpers
# ============================================================================


def _decode_frame(image_data: nxai_postprocessor_sdk.ImageData) -> Image.Image:
    """Decode a raw uint8 RGB ImageData frame into a Pillow RGB image.

    Raises:
        ValueError: The buffer's size doesn't match height x width x 3 uint8
            elements -- a truncated or misaligned frame. This is the check a
            plain byte sum can never make: the sum is identical whether the
            bytes that are there form a valid frame or not.
    """
    expected_bytes = image_data.height * image_data.width * _CHANNELS
    pixels = np.frombuffer(image_data.bytes, dtype=np.uint8)
    if pixels.size != expected_bytes:
        raise ValueError(
            f"frame has {pixels.size} uint8 bytes, expected {expected_bytes} "
            f"for a {image_data.width}x{image_data.height}x{_CHANNELS} image"
        )

    if image_data.layout == nxai_postprocessor_sdk.ImageOrder.CHW:
        pixels = pixels.reshape(_CHANNELS, image_data.height, image_data.width).transpose(1, 2, 0)
    else:
        pixels = pixels.reshape(image_data.height, image_data.width, _CHANNELS)

    return Image.fromarray(pixels, mode="RGB")


_latest_frame_jpeg: Optional[bytes] = None
_frame_version = 0
_frame_condition = threading.Condition()


def _publish_frame(jpeg_bytes: bytes) -> None:
    """Make jpeg_bytes the frame served to any connected stream client."""
    global _latest_frame_jpeg, _frame_version
    with _frame_condition:
        _latest_frame_jpeg = jpeg_bytes
        _frame_version += 1
        _frame_condition.notify_all()


class _MjpegRequestHandler(BaseHTTPRequestHandler):
    """Serves the latest published frame as a live multipart/x-mixed-replace stream."""

    def log_message(self, format_str, *args):
        logger.debug(f"stream: {format_str % args}")

    def do_GET(self):
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        last_sent_version = -1
        try:
            while True:
                with _frame_condition:
                    _frame_condition.wait_for(lambda: _frame_version != last_sent_version, timeout=5.0)
                    frame, last_sent_version = _latest_frame_jpeg, _frame_version
                if frame is None:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.debug("Preview stream client disconnected")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger = nxai_postprocessor_sdk.setup_logging()
    processor_config = nxai_postprocessor_sdk.load_config(sys.argv)
    nxai_postprocessor_sdk.apply_config_logging(processor_config, logger)
    logger.info(f"Initializing '{processor_config.get('ProcessorName', PROCESSOR_NAME)}' postprocessor")
    logger.debug(f"Command-line arguments: {sys.argv}")

    stream_port = int(processor_config.get("stream_port", DEFAULT_STREAM_PORT))
    start_stream_server(stream_port)

    processor = nxai_postprocessor_sdk.ExternalPostprocessor()
    if not processor.initialize_from_args():
        logger.error("Failed to initialize postprocessor from arguments")
        sys.exit(1)
    logger.info(f"Postprocessor '{processor.name}' initialized successfully")
    logger.info("Waiting for messages from NXAI runtime...")
    try:
        run_message_loop(processor)
    except KeyboardInterrupt:
        logger.info("Exited with keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Postprocessor shutdown")
