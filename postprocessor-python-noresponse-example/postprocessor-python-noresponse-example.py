"""
Example NXAI postprocessor that never responds.

This postprocessor receives detected objects and events from the NXAI
runtime, logs them, and returns nothing: it is fire-and-forget. The
colocated `external_postprocessors.json` sets `"NoResponse": true` so the
AI Manager does not wait for output after sending the message.

Configuration:
  - Logs: written to stdout/stderr; the AI Manager writes them to the entry's LogFilepath
  - Config: the Config keys of the registration entry, passed as launch arguments
"""

import sys
import logging
import nxai_postprocessor_sdk

# ============================================================================
# Configuration
# ============================================================================

# Module-level logger
logger = logging.getLogger(__name__)

PROCESSOR_NAME = "noresponse-example"

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage) -> None:
    """Log a received message. Returns nothing: this processor never responds.

    Args:
        msg: Parsed ObjectsPostProcessorMessage from NXAI runtime
    """

    logger.info(
        f"Processing: device={msg.device_id}, "
        f"{len(msg.objects)} object classes, {len(msg.events)} events"
    )
    logger.debug(f"Device: {msg.device_name}, Timestamp: {msg.timestamp}")
    logger.debug(f"Objects: {msg.objects}")
    logger.debug(f"Events: {msg.events}")


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

            # Parse msgpack message into ObjectsPostProcessorMessage
            objects_message = processor.parse_objects_message(message_bytes)
            if objects_message is None:
                logger.error("Failed to parse message")
                continue

            # Fire-and-forget: perform the side effect and end the loop
            # here. No write_objects_message/send_response call - the AI
            # Manager is configured with "NoResponse": true and does not
            # wait for one.
            handle_message(objects_message)

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            break


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    # Set up logging using SDK utility
    logger = nxai_postprocessor_sdk.setup_logging()

    # Load configuration using SDK utility
    processor_config = nxai_postprocessor_sdk.load_config(sys.argv)

    # Apply logging configuration if available
    nxai_postprocessor_sdk.apply_config_logging(processor_config, logger)

    # Log startup information
    logger.info(f"Initializing '{processor_config.get('ProcessorName', PROCESSOR_NAME)}' postprocessor")
    logger.debug(f"Command-line arguments: {sys.argv}")

    # Initialize ExternalPostprocessor from command-line arguments
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
