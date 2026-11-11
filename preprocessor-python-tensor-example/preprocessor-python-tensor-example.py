"""
Tensor Preprocessor (v5 SDK)

Overrides the NMS-sensitivity tensor ("nms_sensitivity-") with a live
UserSettings value before the frame reaches the model. All other tensors
pass through unchanged.
"""

import struct
import sys
import logging
import nxai_postprocessor_sdk

# ============================================================================
# Configuration
# ============================================================================

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "tensor-example"
NMS_TENSOR_NAME = "nms_sensitivity-"
DEFAULT_NMS_OVERRIDE = 0.8

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(msg: nxai_postprocessor_sdk.TensorMessage) -> nxai_postprocessor_sdk.TensorMessage:
    """Overwrite the "nms_sensitivity-" tensor's data with the forwarded `nmsoverride` setting.

    Args:
        msg: Parsed TensorMessage from NXAI runtime.

    Returns:
        The same message, with the NMS-sensitivity tensor's data replaced if
        present. Absent -> no-op.
    """
    tensor = msg.tensors.get(NMS_TENSOR_NAME)
    if tensor is None:
        logger.debug(f"No '{NMS_TENSOR_NAME}' tensor in message; nothing to override")
        return msg

    # A single guard means the plugin no-ops cleanly on a not-yet-patched engine
    # that forwards no settings (empty user_settings).
    if "nmsoverride" in msg.user_settings:
        nms_override = float(msg.user_settings["nmsoverride"])
    else:
        nms_override = DEFAULT_NMS_OVERRIDE

    tensor.data = struct.pack("f", nms_override)
    logger.debug(f"Overrode '{NMS_TENSOR_NAME}' with {nms_override}")

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
            tensor_message = processor.parse_tensor_message(message_bytes)
            if tensor_message is None:
                logger.error("Failed to parse message")
                continue
            processed_msg = handle_message(tensor_message)
            response_bytes = processor.write_tensor_message(processed_msg)
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
