"""
Cloud Inference Postprocessor (v5 SDK)

Crops every detected `face` out of the input frame, sends the crop to AWS
Rekognition, and attaches the returned human-readable description to the object.

Behavior-preserving port of the v4 example, with two changes forced by the v5
wire format rather than chosen:

* v4 moved a classified face into a *new* class named after the Rekognition
  description. That description is free text, so the class set was unbounded,
  and v5 requires every class in `msg.objects` to be pre-declared in the
  registration JSON's `Objects: [{ID, Name}]`. The description is therefore
  attached as a `description` attribute on the existing `face` object instead.
* Consequently the classified face is no longer deleted from the `face` class
  (v4 removed it once it had been re-homed under the description class).

The frame arrives at the root `ImageData` key (gated by `ReceiveInputImage`)
rather than through a separate image-header message plus a raw SHM read, and
object boxes are normalized to [0,1], so they are scaled to pixels before the
crop.
"""

import sys
import logging
import numpy as np
from PIL import Image
from aws_utils import classify_faces, create_session
import nxai_postprocessor_sdk

# ============================================================================
# Configuration
# ============================================================================

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "cloud-inference"

# Class whose objects are sent to Rekognition, and the attribute the returned
# description is written to.
FACE_CLASS = "face"
DESCRIPTION_ATTRIBUTE = "description"

# Classify at most this many faces per frame, to avoid affecting the FPS rate.
MAX_FACES_PER_FRAME = 2

# Where the cropped face is written before being handed to Rekognition. Set from
# the `image_path` Config key at startup; a launch-time path, not a live setting.
image_path = None

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(
    msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage,
) -> nxai_postprocessor_sdk.ObjectsPostProcessorMessage:
    """Classify the frame's faces and describe them.

    Args:
        msg: Parsed ObjectsPostProcessorMessage from NXAI runtime, with
            `image_data` populated (ReceiveInputImage is set in the JSON).

    Returns:
        The same message, with a `description` attribute set on each face that
        Rekognition described, up to MAX_FACES_PER_FRAME per frame.
    """
    faces = msg.objects.get(FACE_CLASS, [])
    if not faces:
        return msg

    image = _image_from_frame(msg.image_data)
    classified = 0
    for face in faces:
        cropped_image = image.crop(_pixel_box(face, image.width, image.height))
        cropped_image.save(image_path)

        logger.info(f"Classifying image {image_path}")
        description = classify_faces(image_path, logger)
        if description is None:
            logger.info("No description for this face.")
            continue

        face.attributes[DESCRIPTION_ATTRIBUTE] = description
        classified += 1
        if classified >= MAX_FACES_PER_FRAME:
            break

    logger.debug(f"Described {classified} of {len(faces)} '{FACE_CLASS}' object(s)")
    return msg


def run_message_loop(processor: nxai_postprocessor_sdk.ExternalPostprocessor) -> None:
    """Serve messages until the engine sends a graceful EXIT signal.

    `wait_for_message()` returns None both on a plain timeout and on an IPC EXIT
    signal; only `processor.exit_requested` tells them apart. A loop that just
    `continue`s on None regardless of that flag never returns, and the AI Manager
    kills the process after its grace period instead of it exiting cleanly.

    The Rekognition call inside `handle_message` is a synchronous step of the loop
    body, so no request is ever left in flight across the return: the frame that
    made it completes before the next message is asked for.

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
# Utilities
# ============================================================================


def _image_from_frame(image_data: nxai_postprocessor_sdk.ImageData) -> Image.Image:
    """Rebuild a PIL image from the frame's raw bytes.

    The engine delivers the frame as uint8 RGB, one byte per channel.
    """
    array = np.frombuffer(image_data.bytes, dtype=np.uint8)
    if image_data.layout == nxai_postprocessor_sdk.ImageOrder.CHW:
        array = array.reshape((_CHANNELS, image_data.height, image_data.width)).transpose(1, 2, 0)
    else:
        array = array.reshape((image_data.height, image_data.width, _CHANNELS))
    return Image.fromarray(array)


# RGB, as the v4 example's `Image.frombytes("RGB", ...)` also assumed.
_CHANNELS = 3


def _pixel_box(
    obj: nxai_postprocessor_sdk.ObjectMetadata, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """Scale a normalized xywh object box to a pixel xyxy crop box."""
    x1 = round(obj.x * image_width)
    y1 = round(obj.y * image_height)
    x2 = round((obj.x + obj.width) * image_width)
    y2 = round((obj.y + obj.height) * image_height)
    return x1, y1, x2, y2


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger = nxai_postprocessor_sdk.setup_logging()
    processor_config = nxai_postprocessor_sdk.load_config(sys.argv)
    nxai_postprocessor_sdk.apply_config_logging(processor_config, logger)
    logger.info(f"Initializing '{processor_config.get('ProcessorName', PROCESSOR_NAME)}' postprocessor")
    logger.debug(f"Command-line arguments: {sys.argv}")

    image_path = processor_config.get("image_path")
    if image_path is None:
        logger.error("No 'image_path' in the processor Config; nowhere to write the cropped face")
        sys.exit(1)

    if create_session(logger) is None:
        logger.error("AWS session failed")
        sys.exit(1)
    logger.info("AWS session started")

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
