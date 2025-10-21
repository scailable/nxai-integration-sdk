import os
import sys
import socket
import signal
import logging
import logging.handlers
import configparser
from pprint import pformat
import msgpack

# Add the nxai-utilities python utilities
script_location = os.path.dirname(sys.argv[0])
sys.path.append(os.path.join(script_location, "../sclbl-utilities/python-utilities"))


CONFIG_FILE = os.path.join(script_location, "..", "etc", "plugin.image.ini")

LOG_FILE = os.path.join(script_location, "..", "etc", "plugin.image.log")

# Initialize plugin and logging, script makes use of INFO and DEBUG levels
handler_stream = logging.StreamHandler(sys.stdout)
handler_stream.setLevel(logging.DEBUG)
handler_file = logging.FileHandler(filename=LOG_FILE,mode="w")
handler_file.setLevel(logging.INFO)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - example - %(message)s",
    handlers=[handler_stream,handler_file]
)

import nxai_communication_utils

# The name of the postprocessor.
# This is used to match the definition of the postprocessor with routing.
Postprocessor_Name = "Python-Image-Example-Postprocessor"

# The socket this postprocessor will listen on.
# This is always given as the first argument when the process is started
# But it can be manually defined as well, as long as it is the same as the socket path in the runtime settings
Postprocessor_Socket_Path = "/tmp/python-image-postprocessor.sock"

global shared_memory
shared_memory = None
def parse_image_from_shm(shm_key: int, width: int, height: int, channels: int):
    global shared_memory
    if shared_memory is None:
        shared_memory = nxai_communication_utils.SharedMemory(key=shm_key)
    image_data = shared_memory.read()

    cumulative = 0
    for b in image_data:
        cumulative += b

    return cumulative


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


def signal_handler(sig, _):
    logger.info("Received interrupt signal: " + str(sig))
    sys.exit(0)


def main():
    # Start socket listener to receive messages from NXAI runtime
    server = nxai_communication_utils.SocketListener(Postprocessor_Socket_Path)
    # Wait for messages in a loop
    while True:
        # Wait for input message from runtime
        logger.debug("Waiting for input message")

        try:
            connection, input_message = server.accept()
            logger.debug("Received input message")
            formatted_input_message = pformat(input_message)
            logger.debug(f"Input message: :\n\n{formatted_input_message}\n\n")

        except nxai_communication_utils.SocketTimeout:
            # Request timed out. Continue waiting
            continue

        # Since we're also expecting an image, receive the image header
        try:
            image_header = connection.receive()
        except nxai_communication_utils.SocketTimeout:
            # Did not receive image header
            logger.warning("Did not receive image header. Are the settings correct?")
            continue

        # Parse input message
        input_object = nxai_communication_utils.parseInferenceResults(input_message)
        formatted_unpacked_object = pformat(input_object)
        logger.info(f"Unpacked input image:\n\n{formatted_unpacked_object}\n\n")

        image_header = msgpack.unpackb(image_header)
        formatted_image_object = pformat(image_header)
        logger.info(f"Image header:\n\n{formatted_image_object}\n\n")

        # Convert SHM Key to integer from string
        shm_key = image_header["SHMKEY"]

        cumulative = parse_image_from_shm(
            shm_key,
            image_header["Width"],
            image_header["Height"],
            image_header["Channels"],
        )

        # Add extra bbox
        if "BBoxes_xyxy" not in input_object:
            input_object["BBoxes_xyxy"] = {}
        input_object["BBoxes_xyxy"]["test"] = [100.0, 100.0, 200.0, 200.0]

        # Add Counts
        if "Counts" not in input_object:
            input_object["Counts"] = {}
        input_object["Counts"]["ImageBytesCumulative"] = cumulative

        formatted_packed_object = pformat(input_object)
        logger.info(f"Returning packed object:\n\n{formatted_packed_object}\n\n")

        # Write object back to string
        output_message = nxai_communication_utils.writeInferenceResults(input_object)

        # Send message back to runtime
        connection.send(output_message)
        connection.close()


if __name__ == "__main__":
    ## initialize the logger
    logger = logging.getLogger(__name__)

    ## read configuration file if it's available
    config()

    logger.info("Initializing image plugin")
    logger.debug("Input parameters: " + str(sys.argv))

    # Parse input arguments
    if len(sys.argv) > 1:
        Postprocessor_Socket_Path = sys.argv[1]
    # Handle interrupt signals
    signal.signal(signal.SIGTERM, signal_handler)
    # Start program
    try:
        main()
    except Exception as e:
        logging.error(e, exc_info=True)
