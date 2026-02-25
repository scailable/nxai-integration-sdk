import os
import sys
import socket
import logging
import logging.handlers
import configparser
from pprint import pformat
from collections import OrderedDict
import time

# Add the nxai-utilities python utilities
script_location = os.path.dirname(sys.argv[0])
sys.path.append(os.path.join(script_location, "../nxai-utilities/python-utilities"))
import nxai_communication_utils


CONFIG_FILE = os.path.join(script_location, "..", "etc", "plugin.clip.post.ini")

# Set up logging
if os.path.exists(os.path.join(script_location, "..", "etc")):
    LOG_FILE = os.path.join(script_location, "..", "etc", "plugin.clip.post.log")
else:
    LOG_FILE = os.path.join(script_location, "plugin.clip.post.log")

# Initialize plugin and logging, script makes use of INFO and DEBUG levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - example - %(message)s",
    filename=LOG_FILE,
    filemode="w",
)

# The socket this postprocessor will listen on.
# This is always given as the first argument when the process is started
# But it can be manually defined as well, as long as it is the same as the socket path in the runtime settings
import tempfile
Postprocessor_Socket_Path = os.path.join(tempfile.gettempdir(),"example-clip-postprocessor.sock")

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


objects_attributes = OrderedDict()
# Keep track of the last states and times we generated an event per camera
previous_state = {}
previous_event_time = {}


def main():
    # Start socket listener to receive messages from NXAI runtime
    logger.debug("Creating socket at " + Postprocessor_Socket_Path)
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
        if isinstance(input_object, nxai_communication_utils.ExitSignal):
            logger.info("Received exit signal.")
            connection.close()
            break

        # Use pformat to format the deep object
        # formatted_unpacked_object = pformat(input_object)
        # logging.info(f"Unpacked:\n\n{formatted_unpacked_object}\n\n")

        if "ObjectsMetaData" in input_object:
            logger.info("Found objects ")
            # This is the output of the object detector
            # Add prompts as attributes if present
            for class_data in input_object["ObjectsMetaData"].values():
                for index, id in enumerate(class_data["ObjectIDs"]):
                    if id in objects_attributes:
                        logger.info("Found ID " + objects_attributes[id])
                        class_data["AttributeKeys"][index].append(objects_attributes[id])
                        class_data["AttributeValues"][index].append(objects_attributes[id])

        # Get prompts from settings
        prompts = {}
        settings_names = sorted(list(input_object["ExternalProcessorSettings"].keys()))
        for setting_name in settings_names:
            if setting_name.startswith("externalprocessor.prompt"):
                prompts[setting_name.replace("externalprocessor.", "")] = input_object["ExternalProcessorSettings"][setting_name]
        logger.info("Got prompts: " + str(prompts))

        # Get event cooldown setting
        device_id = input_object["DeviceID"]
        event_cooldown = 0
        if "externalprocessor.eventCooldown" in input_object["ExternalProcessorSettings"]:
            try:
                event_cooldown = int(input_object["ExternalProcessorSettings"]["externalprocessor.eventCooldown"])
            except:
                pass

        if "Scores" in input_object:
            # This is the output of the clip model
            # Replace the prompts with the appropriate text
            score_names = list(input_object["Scores"].keys())
            top_score = ("", 0.0)
            prompt_found = False
            for score_name in score_names:
                if score_name in prompts:
                    prompt_found = True
                    if top_score[1] < input_object["Scores"][score_name]:
                        top_score = (
                            prompts[score_name],
                            input_object["Scores"][score_name],
                        )
            if prompt_found == True:
                # Remove original scores
                del input_object["Scores"]
                if top_score[0] != "":
                    # Check if event should be added
                    add_event = True
                    if device_id in previous_state and previous_state[device_id] == top_score[0]:
                        # State has not changed, check if timed out
                        if (time.time() - previous_event_time[device_id]) < event_cooldown:
                            add_event = False

                    if add_event is True:
                        # Add event to output
                        if "Events" not in input_object:
                            input_object["Events"] = []
                        input_object["Events"].append(
                            {
                                "ID": "nx.clip.event",
                                "Caption": "CLIP Prompt Recognized",
                                "Description": top_score[0],
                            }
                        )
                        # Set previous state to handle event cooldowns
                        previous_event_time[device_id] = time.time()
                        previous_state[device_id] = top_score[0]

            # Check if object is feature extracted
            if "OriginalObjectID" in input_object:
                if top_score[0] != "":
                    objects_attributes[input_object["OriginalObjectID"]] = top_score[0]

        # formatted_unpacked_object = pformat(input_object)
        # logging.info(f"Packing:\n\n{formatted_unpacked_object}\n\n")

        # Write object back to string
        output_message = nxai_communication_utils.writeInferenceResults(input_object)

        # Send message back to runtime
        connection.send(output_message)
        connection.close()

        # Remove objects
        while len(objects_attributes) > 100:
            logger.info("Popping item from objects_attributes " + str(len(objects_attributes)))
            objects_attributes.popitem(False)


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
