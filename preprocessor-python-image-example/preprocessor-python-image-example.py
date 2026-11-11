"""
Image Preprocessor (v5 SDK)

Mirrors the input image at full resolution before it reaches the rest of
the AI Manager's own preprocessing pipeline (resize/normalize/cast).
Disabled, the image passes through unchanged.

The payload is uint8 RGB at source resolution, laid out HWC or CHW per
`image.layout` -- a live v5 engine delivers CHW. Decoding it into a Pillow
image and mirroring with `ImageOps.mirror` keeps the pixel logic in one
well-tested library call instead of hand-rolled byte slicing; Pillow's and
numpy's own operations run in C, so this stays well inside the engine's IPC
window at full resolution.
"""

import logging
import sys

import numpy as np
from PIL import Image, ImageOps

import nxai_postprocessor_sdk

# ============================================================================
# Configuration
# ============================================================================

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "image-example"
DEFAULT_MIRROR_IMAGE = True
_CHANNELS = 3

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(msg: nxai_postprocessor_sdk.ImagePreprocessorMessage) -> nxai_postprocessor_sdk.ImagePreprocessorMessage:
    """Mirror `msg.image` horizontally at its input resolution.

    Args:
        msg: Parsed ImagePreprocessorMessage from NXAI runtime.

    Returns:
        The same message. If the forwarded `mirrorimage` setting is enabled
        (falling back to `DEFAULT_MIRROR_IMAGE` when the engine forwards no
        settings), `msg.image` is replaced with a horizontal mirror at
        unchanged width and height; otherwise the image passes through
        unchanged.
    """
    logger.debug(f"UserSettings received: {msg.user_settings!r}")
    if "mirrorimage" in msg.user_settings:
        mirror_image = str(msg.user_settings["mirrorimage"]).lower() == "true"
    else:
        mirror_image = DEFAULT_MIRROR_IMAGE

    if not mirror_image:
        return msg

    image = msg.image
    pixels = np.frombuffer(image.bytes, dtype=np.uint8)
    chw = image.layout == nxai_postprocessor_sdk.ImageOrder.CHW
    if chw:
        # CHW is channels-planes-of-rows; Pillow wants interleaved HWC, so
        # transpose to a view in that order and transpose back afterward.
        pixels = pixels.reshape(_CHANNELS, image.height, image.width).transpose(1, 2, 0)
    else:
        pixels = pixels.reshape(image.height, image.width, _CHANNELS)

    mirrored = np.asarray(ImageOps.mirror(Image.fromarray(pixels, mode="RGB")))
    if chw:
        mirrored = mirrored.transpose(2, 0, 1)

    image.bytes = mirrored.tobytes()
    logger.debug(f"Mirrored image: {image.width}x{image.height}, layout {image.layout!r}")

    return msg


def run_message_loop(processor: nxai_postprocessor_sdk.ExternalPreprocessor) -> None:
    """Serve messages until the engine sends a graceful EXIT signal.

    `wait_for_message()` returns None both on a plain timeout and on an IPC EXIT
    signal; only `processor.exit_requested` tells them apart. A loop that just
    `continue`s on None regardless of that flag never returns, and the AI Manager
    kills the process after its grace period instead of it exiting cleanly.

    Args:
        processor: An initialized ExternalPreprocessor.
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
            image_message = processor.parse_image_message(message_bytes)
            if image_message is None:
                logger.error("Failed to parse message")
                continue
            processed_msg = handle_message(image_message)
            response_bytes = processor.write_image_message(processed_msg)
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
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger = nxai_postprocessor_sdk.setup_logging()
    processor_config = nxai_postprocessor_sdk.load_config(sys.argv)
    nxai_postprocessor_sdk.apply_config_logging(processor_config, logger)
    logger.info(f"Initializing '{processor_config.get('ProcessorName', PROCESSOR_NAME)}' preprocessor")
    logger.debug(f"Command-line arguments: {sys.argv}")

    processor = nxai_postprocessor_sdk.ExternalPreprocessor()
    if not processor.initialize_from_args():
        logger.error("Failed to initialize preprocessor from arguments")
        sys.exit(1)
    logger.info(f"Preprocessor '{processor.name}' initialized successfully")
    logger.info("Waiting for messages from NXAI runtime...")
    try:
        run_message_loop(processor)
    except KeyboardInterrupt:
        logger.info("Exited with keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Preprocessor shutdown")
