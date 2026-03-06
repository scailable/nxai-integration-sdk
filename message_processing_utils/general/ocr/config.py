"""
OCR-specific configuration utilities.

This module extends common configuration with OCR-specific settings.
"""

import os
import logging
import configparser
from config_utils import load_common_config


def load_ocr_config(config_path=None, processor_name="postprocessor"):
    """
    Read configuration from file, including OCR-specific settings.

    This function extends load_common_config() with OCR-specific settings.

    Args:
        config_path: Path to INI configuration file. If None, returns defaults.
        processor_name: Name of the processor for default paths.

    Returns:
        Dictionary with configuration settings including common and OCR-specific settings.
    """
    # Get common settings
    settings = load_common_config(config_path, processor_name)

    # Add OCR-specific defaults
    settings["ocr_worker_count"] = max(1, min(4, os.cpu_count() or 1))
    settings["ocr_output_name"] = "Identity:0"

    if config_path is None:
        return settings

    logger = logging.getLogger(__name__)
    try:
        configuration = configparser.ConfigParser()
        configuration.read(config_path)
    except Exception as e:
        logger.error("Failed to read configuration: %s", e)
        return settings

    if "ocr" in configuration:
        settings["ocr_worker_count"] = configuration.getint(
            "ocr", "worker_count", fallback=settings["ocr_worker_count"]
        )
        settings["ocr_output_name"] = configuration.get(
            "ocr", "output_name", fallback=settings["ocr_output_name"]
        )

    logger.debug("Read OCR configuration done")
    return settings
