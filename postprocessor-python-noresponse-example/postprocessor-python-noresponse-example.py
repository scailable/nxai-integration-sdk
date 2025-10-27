import os
import sys
import logging
import configparser
from pprint import pformat

# Add the nxai-utilities python utilities
script_location = os.path.dirname(sys.argv[0])
sys.path.append(os.path.join(script_location, "../nxai-utilities/python-utilities"))
import nxai_communication_utils


CONFIG_FILE = os.path.join(script_location, "..", "etc", "plugin.noresponse.ini")

LOG_FILE = os.path.join(script_location, "..", "etc", "plugin.noresponse.log")


# Initialize plugin and logging, script makes use of INFO and DEBUG levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - noresponse - %(message)s",
    filename=LOG_FILE,
    filemode="w",
)

# The name of the postprocessor.
# This is used to match the definition of the postprocessor with routing.
Postprocessor_Name = "Python-NoResponse-Postprocessor"

# The socket this postprocessor will listen on.
# This is always given as the first argument when the process is started
# But it can be manually defined as well, as long as it is the same as the socket path in the runtime settings
Postprocessor_Socket_Path = "/tmp/python-noresponse-postprocessor.sock"

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


def main():
    # Start socket listener to receive messages from NXAI runtime
    server = nxai_communication_utils.SocketListener(Postprocessor_Socket_Path)

    logging.debug("Starting main" + str(Postprocessor_Socket_Path))
    # Wait for messages in a loop
    while True:
        # Wait for input message from runtime
        logging.debug("Starting loop")

        try:
            input_message, connection = server.accept()
            logging.debug("Received input message")
            formatted_input_message = pformat(input_message)
            logger.debug(f"Input message: :\n\n{formatted_input_message}\n\n")
        except nxai_communication_utils.SocketTimeout:
            # Request timed out. Continue waiting
            continue

        # Parse input message
        input_object = nxai_communication_utils.parseInferenceResults(input_message)
        if isinstance(input_object, nxai_communication_utils.ExitSignal):
            logger.info("Received exit signal.")
            connection.close()
            break

        formatted_unpacked_object = pformat(input_object)
        logging.info(f"Unpacked object:\n\n{formatted_unpacked_object}\n\n")


if __name__ == "__main__":
    ## initialize the logger
    logger = logging.getLogger(__name__)

    ## read configuration file if it's available
    config()

    logger.info("Initializing noresponse plugin")
    logger.debug("Input parameters: " + str(sys.argv))

    # Parse input arguments
    if len(sys.argv) > 1:
        Postprocessor_Socket_Path = sys.argv[1]

    # Start program
    try:
        main()
    except Exception as e:
        logging.error(e, exc_info=True)
