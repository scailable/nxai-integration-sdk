"""
Configuration and logging utilities for postprocessors.

This module provides common configuration utilities that can be used by
all postprocessors.
"""

import os
import sys
import logging
import configparser
import tempfile
from pathlib import Path


def get_postprocessor_base_dir():
    """
    Directory to use for resolving config/data paths (INI, etc.).
    When running as a frozen exe (e.g. Nuitka onefile), returns the executable's
    directory; otherwise returns the script directory (from sys.argv[0]).
    Use this so that paths like os.path.join(base_dir, "..", "etc", "plugin.ini")
    work both when run as script and when run as onefile exe.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(sys.argv[0]))


def get_nxai_utilities_library_path():
    """
    Return the full path to the nxai-c-utilities shared library for
    initializeLibrary(library_path=...), so the loader resolves dependencies correctly.
    When frozen (exe): library next to the executable. When run as script: next to the script.
    """
    if getattr(sys, "frozen", False):
        base_dir = os.path.dirname(os.path.abspath(sys.executable))
    else:
        base_dir = get_postprocessor_base_dir()
    if sys.platform == "win32":
        return os.path.join(base_dir, "nxai-c-utilities-shared.dll")
    return os.path.join(base_dir, "libnxai-c-utilities-shared.so")


def load_common_config(config_path=None, processor_name="postprocessor"):
    """
    Read common configuration from file.

    Args:
        config_path: Path to INI configuration file. If None, returns defaults.
        processor_name: Name of the processor for default paths.

    Returns:
        Dictionary with common configuration settings (log_level, socket_path, log_file, nxai_utilities_path).
    """
    # Default settings
    script_location = get_postprocessor_base_dir()

    # Ensure processor_name has "postprocessor-" prefix for socket and log file names
    if not processor_name.startswith("postprocessor-"):
        socket_name = f"postprocessor-{processor_name}"
    else:
        socket_name = processor_name
    temp_dir = Path(tempfile.gettempdir())
    settings = {
        "log_level": "INFO",
        "socket_path": str(temp_dir / f"{socket_name}.sock"),
        "log_file": str(temp_dir / f"{socket_name}.log"),
        "nxai_utilities_path": os.path.join(script_location, "..", "nxai-utilities", "python-utilities"),
    }
    if config_path is None:
        return settings

    logger = logging.getLogger(__name__)
    logger.info("Reading configuration from: %s", config_path)
    try:
        configuration = configparser.ConfigParser()
        configuration.read(config_path)
    except Exception as e:
        logger.error("Failed to read configuration: %s", e)
        return settings

    if "common" in configuration:
        settings["log_level"] = configuration.get("common", "log_level", fallback=settings["log_level"])
        settings["socket_path"] = configuration.get("common", "socket_path", fallback=settings["socket_path"])
        settings["log_file"] = configuration.get("common", "log_file", fallback=settings["log_file"])
        settings["nxai_utilities_path"] = configuration.get("common", "nxai_utilities_path", fallback=settings["nxai_utilities_path"])

    logger.debug("Read configuration done")
    return settings


def setup_logging(level_str: str, log_file: str, processor_name: str = "postprocessor"):
    """
    Set up logging configuration.

    Args:
        level_str: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to the log file.
        processor_name: Name of the processor for log format.
    """
    numeric_level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format=f"%(asctime)s - %(levelname)s - {processor_name} - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w")
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized at level %s", level_str)
