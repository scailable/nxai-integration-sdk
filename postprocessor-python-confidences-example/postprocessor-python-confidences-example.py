import os
import sys
import socket
import signal
import logging
import logging.handlers
import configparser

# Add the nxai-utilities python utilities
script_location = os.path.dirname(sys.argv[0])
sys.path.append(os.path.join(script_location, "../nxai-utilities/python-utilities"))
import nxai_communication_utils


CONFIG_FILE = os.path.join(script_location, "..", "etc", "plugin.confidence.ini")

# Set up logging
if os.path.exists(os.path.join(script_location, "..", "etc")):
    LOG_FILE = os.path.join(script_location, "..", "etc", "plugin.confidence.log")
else:
    LOG_FILE = os.path.join(script_location, "plugin.confidence.log")

# Initialize plugin and logging, script makes use of INFO and DEBUG levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - example - %(message)s",
    filename=LOG_FILE,
    filemode="w",
)

# The name of the postprocessor.
# This is used to match the definition of the postprocessor with routing.
Postprocessor_Name = "Python-Example-Confidences-Postprocessor"

# The socket this postprocessor will listen on.
# This is always given as the first argument when the process is started
# But it can be manually defined as well, as long as it is the same as the socket path in the runtime settings
Postprocessor_Socket_Path = "/tmp/python-example-confidences-postprocessor.sock"

# Data Types
# 1:  //FLOAT
# 2:  //UINT8
# 3:  //INT8
# 4:  //UINT16
# 5:  //INT16
# 6:  //INT32
# 7:  //INT64
# 8:  //STRING
# 9:  //BOOL
# 11: //DOUBLE
# 12: //UINT32
# 13: //UINT64


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
        except nxai_communication_utils.SocketTimeout:
            # Request timed out. Continue waiting
            continue

        # Parse input message
        input_object = nxai_communication_utils.parseInferenceResults(input_message)

        # Use pformat to format the deep object
        # formatted_unpacked_object = pformat(input_object)
        # logging.debug(f'Unpacked:\n\n{formatted_unpacked_object}\n\n')

        # Add the confidence of each object as attributes
        for _, class_data in input_object["ObjectsMetaData"].items():
            for object_index in range(len(class_data["Confidences"])):
                confidence_string = str(round(class_data["Confidences"][object_index], 2))
                class_data["AttributeKeys"][object_index].append("Confidence")
                class_data["AttributeValues"][object_index].append(confidence_string)

        logger.info("Added test bounding box to output")

        # Write object back to string
        output_message = nxai_communication_utils.writeInferenceResults(input_object)

        # Send message back to runtime
        connection.send(output_message)
        connection.close()


if __name__ == "__main__":
    ## initialize the logger
    logger = logging.getLogger(__name__)

    logger.info("Location: " + str(script_location))

    ## read configuration file if it's available
    config()

    logger.info("Initializing example plugin")
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
        logger.error(e, exc_info=True)
    except KeyboardInterrupt:
        logger.info("Exited with keyboard interrupt")

    try:
        os.unlink(Postprocessor_Socket_Path)
    except OSError:
        if os.path.exists(Postprocessor_Socket_Path):
            logger.error("Could not remove socket file: " + Postprocessor_Socket_Path)
