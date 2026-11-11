"""
Example NXAI postprocessor demonstrating forwarded GUI settings.

The AI Manager forwards a camera's GUI settings as a root `UserSettings` map on
the objects message. This postprocessor reads two of them,
`attributeName` and `attributeValue`, and writes that key/value pair as an
attribute onto every detected object, so the forwarded setting becomes visible
in the NX client.

The settings are declared as unprefixed TextField controls in
`external_postprocessors.json`, so a control name equals the forwarded
`UserSettings` key verbatim.
"""

import sys
import logging
import nxai_postprocessor_sdk

# ============================================================================
# Configuration
# ============================================================================

# Module-level logger
logger = logging.getLogger(__name__)

PROCESSOR_NAME = "settings-example"

# Config received from AI Manager
global processor_config

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(
    msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage,
) -> nxai_postprocessor_sdk.ObjectsPostProcessorMessage:
    """Apply the forwarded GUI setting as an attribute on every object.

    Args:
        msg: Parsed ObjectsPostProcessorMessage from NXAI runtime

    Returns:
        Modified ObjectsPostProcessorMessage to send back to runtime
    """

    logger.info(
        f"Processing: device={msg.device_id}, "
        f"{len(msg.objects)} object classes, {len(msg.user_settings)} forwarded settings"
    )
    logger.debug(f"Device: {msg.device_name}, Timestamp: {msg.timestamp}")

    # A single guard means the plugin no-ops cleanly on a not-yet-patched engine
    # that forwards no settings (empty user_settings).
    if "attributeName" in msg.user_settings and "attributeValue" in msg.user_settings:
        attribute_name = msg.user_settings["attributeName"]
        attribute_value = msg.user_settings["attributeValue"]

        for objects in msg.objects.values():
            for obj in objects:
                obj.attributes[attribute_name] = attribute_value

        logger.info(
            f"Set attribute '{attribute_name}' = '{attribute_value}' on all objects"
        )
    else:
        logger.debug("No forwarded settings to apply; leaving objects unchanged")

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
            # Wait for message from server (timeout in milliseconds)
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

            # Process message using business logic
            processed_msg = handle_message(objects_message)

            # Serialize message back to bytes
            response_bytes = processor.write_objects_message(processed_msg)
            if response_bytes is None:
                logger.error("Failed to serialize message")
                continue

            # Send response back to server
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

    # Main message processing loop
    try:
        run_message_loop(processor)
    except KeyboardInterrupt:
        logger.info("Exited with keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Postprocessor shutdown")
