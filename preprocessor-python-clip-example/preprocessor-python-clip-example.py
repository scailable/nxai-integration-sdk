#!/usr/bin/env python3
import numpy as np
import numpy.core.multiarray
import instant_clip_tokenizer

import os
import sys
import socket
import signal
import logging
import logging.handlers
import msgpack
import configparser
import struct

# Add the nxai-utilities python utilities
script_location = os.path.dirname(sys.argv[0])
sys.path.append(os.path.join(script_location, "../nxai-utilities/python-utilities"))
import nxai_communication_utils

CONFIG_FILE = os.path.join(script_location, "..", "etc", "plugin.tensor.pre.ini")

# Set up logging
if os.path.exists(os.path.join(script_location, "..", "etc")):
    LOG_FILE = os.path.join(script_location, "..", "etc", "plugin.clip.pre.log")
else:
    LOG_FILE = os.path.join(script_location, "plugin.clip.pre.log")

# Initialize plugin and logging, script makes use of INFO and DEBUG levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - example - %(message)s",
    filename=LOG_FILE,
    filemode="w",
)

# The socket this preprocessor will listen on.
# This is always given as the first argument when the process is started
# But it can be manually defined as well, as long as it is the same as the socket path in the runtime settings
Preprocessor_Socket_Path = "/tmp/example-clip-preprocessor.sock"

# Define a single SHM object to share images back to AI Manager
global output_shm
output_shm = None

tokenizer = instant_clip_tokenizer.Tokenizer()


def parseTensorFromSHM(shm_key: str, external_settings: dict):

    ######### Get input tensor from SHM
    logger.info("Got shm key: " + shm_key)
    try:
        shared_memory = nxai_communication_utils.SharedMemory(key=shm_key)
        tensor_raw_data = shared_memory.read()
    except Exception:
        logger.error("Could not read SHM!")
        return 0
    tensor_data = msgpack.unpackb(tensor_raw_data)

    logger.info("Got external_settings: " + str(external_settings))
    if tensor_data is None or isinstance(tensor_data, dict) == False or "Tensors" not in tensor_data:
        logger.error("Invalid input tensor received. Ignoring.")
        return 0

    for tensor_name, _ in tensor_data["Tensors"].items():
        logger.info("Got tensor name: " + str(tensor_name))
        if tensor_name == "text":
            # Get text classes from settings
            prompts = []
            settings_names = sorted(list(external_settings.keys()))
            for setting_name in settings_names:
                if setting_name.startswith("externalprocessor.prompt"):
                    prompts.append(external_settings[setting_name])
            # Make sure there are enough prompts
            while len(prompts) != 5:
                prompts.append("")
            logger.info("Got prompts: " + str(prompts))
            # Tokenize prompts
            text_tokens_np = np.array(tokenizer.tokenize_batch(prompts, context_length=77), dtype=np.int32)
            # Pack the numpy array into a binary structure.
            # Use the flattened array so that we pack all integers.
            text_tokens_np_struct = struct.pack(f"{text_tokens_np.size}i", *text_tokens_np.flatten())
            logger.info("After struct")
            tensor_data["Tensors"][tensor_name] = text_tokens_np_struct

    ######## Write modified tensor to SHM

    output_data = msgpack.packb(tensor_data)

    global output_shm
    if output_shm is None:
        # Can reuse SHM ( if data is smaller or equal size ) or create new SHM and return ID
        output_data_size = len(output_data)
        output_shm = output_shm = nxai_communication_utils.SharedMemory(size=output_data_size)
        logger.debug("Created SHM with ID: " + str(output_shm.id) + " and size: " + str(output_shm.size))
    output_shm.write(output_data)

    return output_shm.id


def main():
    # Start socket listener to receive messages from NXAI runtime
    server = server = nxai_communication_utils.SocketListener(Preprocessor_Socket_Path)
    # Wait for messages in a loop
    while True:
        # Wait for input message from runtime
        try:
            connection, input_message = server.accept()
        except nxai_communication_utils.SocketTimeout:
            # Request timed out. Continue waiting
            continue

        tensor_header = msgpack.unpackb(input_message)
        print("EXAMPLE PREPROCESSOR: Received input message: ", tensor_header)

        external_settings = {}
        if "ExternalProcessorSettings" in tensor_header:
            logger.info("Got settings: " + str(tensor_header["ExternalProcessorSettings"]))
            external_settings = tensor_header["ExternalProcessorSettings"]

        # Process image
        output_shm_id = parseTensorFromSHM(tensor_header["SHMKEY"], external_settings)

        if output_shm_id != 0:
            tensor_header["SHMID"] = output_shm_id

        # Write header to respond
        output_message = msgpack.packb(tensor_header)

        # Send message back to runtime
        connection.send(output_message)
        connection.close()


def signalHandler(sig, _):
    print("EXAMPLE PREPROCESSOR: Received interrupt signal: ", sig)
    logger.info("EXAMPLE PREPROCESSOR: Received interrupt signal: " + str(sig))
    # Detach and destroy output shm
    if output_shm is not None:
        output_shm.detach()
        output_shm.remove()
    sys.exit(0)


def config():
    logger.info("Reading configuration from:" + CONFIG_FILE)

    try:
        configuration = configparser.ConfigParser()
        configuration.read(CONFIG_FILE)

        configured_log_level = configuration.get("common", "debug_level", fallback="INFO")
        set_log_level(configured_log_level)

        for section in configuration.sections():
            logger.info("config section: " + section)
            for key in configuration[section]:
                logger.info("config key: " + key + " = " + configuration[section][key])

    except Exception as e:
        logger.error(e, exc_info=True)

    logger.debug("Read configuration done")


def set_log_level(level):
    try:
        logger.setLevel(level)
    except Exception as e:
        logger.error(e, exc_info=True)


if __name__ == "__main__":
    print("EXAMPLE PREPROCESSOR: Input parameters: ", sys.argv)
    # Parse input arguments
    if len(sys.argv) > 1:
        Preprocessor_Socket_Path = sys.argv[1]
    # Handle interrupt signals
    signal.signal(signal.SIGTERM, signalHandler)

    ## initialize the logger
    logger = logging.getLogger(__name__)

    ## read configuration file if it's available
    config()

    logger.info("Initializing example plugin")
    logger.debug("Input parameters: " + str(sys.argv))

    # Start program
    try:
        main()
    except Exception as e:
        logger.error(e, exc_info=True)
    except KeyboardInterrupt:
        logger.info("Exited with keyboard interrupt")

    try:
        os.unlink(Preprocessor_Socket_Path)
    except OSError:
        if os.path.exists(Preprocessor_Socket_Path):
            logger.error("Could not remove socket file: " + Preprocessor_Socket_Path)
