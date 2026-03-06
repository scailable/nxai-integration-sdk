#!/usr/bin/env python3
"""
Postprocessor Python ANPR Example

This example postprocessor receives OCR results from the CCT model (NxM float32 logits)
and converts them to readable text by finding argmax for each character position.
"""

import os
import sys
import logging

# Add message_processing_utils package to path
script_location = os.path.dirname(os.path.abspath(__file__))
if os.path.join(script_location, "..") not in sys.path:
    sys.path.insert(0, os.path.join(script_location, ".."))

from config_utils import (
    setup_logging,
    get_nxai_utilities_library_path,
    get_postprocessor_base_dir,
)
from message_processing_utils import create_anpr_message_from_bytes
from message_processing_utils.general.ocr import (
    LogitsOcrEngine,
    OcrWorkerPool,
    OcrCache,
    load_ocr_config,
)

logger = logging.getLogger(__name__)


def main(settings, engine, ocr_pool=None):
    """Main postprocessor loop"""
    logger.info("=== STARTING ANPR POSTPROCESSOR ===")
    logger.info("Socket path: %s", settings["socket_path"])
    logger.info("Current working directory: %s", os.getcwd())
    
    # Add nxai_utilities to path
    if settings["nxai_utilities_path"] not in sys.path:
        sys.path.append(settings["nxai_utilities_path"])
    
    import nxai_communication_utils
    lib_path = get_nxai_utilities_library_path()
    if lib_path is not None:
        nxai_communication_utils.initializeLibrary(lib_path)
    server = nxai_communication_utils.SocketListener(settings["socket_path"])
    ocr_cache = OcrCache(engine, settings["ocr_output_name"], ocr_pool)
    while True:
        # Wait for input message from runtime
        logger.debug("Waiting for input message")
        connection = None
        try:
            connection, input_message = server.accept()
            logger.debug("Received input message")
        except nxai_communication_utils.SocketTimeout:
            # Request timed out. Continue waiting
            continue
        try:
            message = create_anpr_message_from_bytes(input_message)
            logger.debug("Processing message: %s", message.__class__.__name__)
            message.handle(ocr_cache)
            try:
                connection.send(message.to_bytes())
            except Exception as e:
                logger.warning("Failed to send response: %s", e)
        except Exception as e:
            logger.error("Error processing message: %s", e, exc_info=True)
            try:
                # Return original message on error
                connection.send(input_message)
            except Exception:
                pass
        finally:
            if connection is not None:
                connection.close()


if __name__ == "__main__":
    # Read configuration from optional INI at fixed path (../etc/plugin.anpr.ini)
    script_location = get_postprocessor_base_dir()
    config_file = os.path.join(script_location, "..", "etc", "plugin.anpr.ini")
    config_path = config_file if os.path.exists(config_file) else None
    
    settings = load_ocr_config(config_path, processor_name="anpr-example")
    
    setup_logging(settings["log_level"], settings["log_file"], processor_name="anpr-example")
    if len(sys.argv) > 1:
        settings["socket_path"] = sys.argv[1]
    logger.debug("Input parameters: %s", sys.argv)
    logger.info("Configuration loaded:")
    for key, val in settings.items():
        logger.info("  %s = %s", key, val)

    if settings["nxai_utilities_path"] not in sys.path:
        sys.path.append(settings["nxai_utilities_path"])

    import nxai_communication_utils
    logger.info(
        "nxai_communication_utils loaded from %s", nxai_communication_utils.__file__)
    try:
        engine = LogitsOcrEngine(
            expected_logits_shape=(9, 37),
            char_map="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
        )
        pool = OcrWorkerPool(engine, settings["ocr_worker_count"])
        main(settings, engine, pool)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        if "pool" in locals():
            pool.stop()
