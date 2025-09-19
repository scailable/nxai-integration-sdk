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
import communication_utils

CONFIG_FILE = os.path.join(script_location, "..", "etc", "plugin.tensor.pre.ini")

# Set up logging
if os.path.exists(os.path.join(script_location, "..", "etc")):
    LOG_FILE = os.path.join(script_location, "..", "etc", "plugin.tensor.pre.log")
else:
    LOG_FILE = os.path.join(script_location, "plugin.tensor.pre.log")

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
Preprocessor_Socket_Path = "/tmp/python-tensor-example-preprocessor.sock"

# Define a single SHM object to share images back to AI Manager
global output_shm
output_shm = None


def parseTensorFromSHM(shm_key: int, external_settings: dict):

    ######### Get input tensor from SHM
    logger.info("Got shm key: " + str(shm_key))
    tensor_raw_data = communication_utils.read_shm(shm_key)
    tensor_data = msgpack.unpackb(tensor_raw_data)

    if tensor_data is None or isinstance(tensor_data, dict) == False or "Tensors" not in tensor_data:
        logger.error("Invalid input tensor received. Ignoring.")
        return 0

    ######## Get nms setting ( if any )

    new_nms_value = 0.8
    if "externalprocessor.nmsoverride" in external_settings:
        try:
            new_nms_value = float(external_settings["externalprocessor.nmsoverride"])
        except:
            pass

    ######## Modify tensor as example

    for tensor_name, _ in tensor_data["Tensors"].items():
        logger.info("Got tensor name: " + str(tensor_name))
        if tensor_name == "nms_sensitivity-":
            # Set nms to new_nms_value
            tensor_data["Tensors"][tensor_name] = struct.pack("f", new_nms_value)

    ######## Write modified tensor to SHM

    output_data = msgpack.packb(tensor_data)

    global output_shm
    if output_shm is None:
        # Can reuse SHM ( if data is smaller or equal size ) or create new SHM and return ID
        output_data_size = len(output_data)
        output_shm = communication_utils.create_shm(output_data_size)
        logger.debug("Created SHM with ID: " + str(output_shm.id) + " and size: " + str(output_shm.size))

    communication_utils.write_shm(output_shm, output_data)

    return output_shm.id


def main():
    # Start socket listener to receive messages from NXAI runtime
    server = communication_utils.startUnixSocketServer(Preprocessor_Socket_Path)
    # Wait for messages in a loop
    while True:
        # Wait for input message from runtime
        try:
            input_message, connection = communication_utils.waitForSocketMessage(server)
        except socket.timeout:
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
        communication_utils.sendMessageOverConnection(connection, output_message)


def signalHandler(sig, _):
    print("EXAMPLE PREPROCESSOR: Received interrupt signal: ", sig)
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
