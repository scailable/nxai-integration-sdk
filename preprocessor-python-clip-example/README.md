Python Clip Example
=========================

This example application provides an example on how to create a Python based preprocessor that can be integrated with the NXAI Edge AI Manager.

This example is meant to be integrated with an OpenAI CLIP model. The CLIP model takes in an image and a number of text prompts. It will then output a list of scores, one for each prompt, on how well the prompt matches the image.

This preprocessor assists with gathering prompts and feeding them to the CLIP model as a separate input tensor. The preprocessor makes use of UI settings to receive the prompts, and will convert them a tokenized input tensor which the CLIP model expects.

This preprocessor can be combined with the CLIP postprocessor to provide better output.

# Tensor Preprocessors Control Flow

The normal control flow of a preprocessor is to receive a MessagePack binary message header from the AI Manager. This message will contain information on how to connect to a Shared Memory segment which contains a MessagePack encoded tensor structure which will be given to the model.

The AI Manager will send the tensor message after its internal preprocessing steps.

The external preprocessor could write back an altered tensor to the shared memory segment sent by the AI Manager, or create a new shared memory segment and write a new tensor to the new segment. 

Finally, the external preprocessor sends back a message to the AI Manager, signalling that it's completed processing. This message contains the new or same shared memory ID. The AI Manager will read the tensor from the shared memory segment ID sent back to the AI Manager, and pass it to the model.

# Message schema

Communication between the AI Manager and Tensor Preprocessor consists of two messages. A small header message is sent over socket, and a large tensor message is made available over SHM. When the header message is sent, the processor can assume that the tensor message is ready on the SHM. When the preprocessor responds with a header message over socket, the AI Manager will assume that processing is completed, and will copy back the SHM message.

## Header message

The header message sent to the external postprocessor is MessagePack encoded and contains information about the image and how to retrieve it through shared memory. In Json, this schema would look like:

```json
{
    "SHMKEY": <Key>,
    "SHMID": <ID>,
}
```

The header message sent back to the AI Manager should contain at least the SHMID field, whether the ID was changed or not.

## Tensor message

The tensor message can contain large amounts of binary data, and is therefore sent over shared memory as a MessagePack encoded tensor structure. In Json, this schema would look like:

```json
{
    "Tensors": {"output_name1": [...], "output_name2": [...], ...},
    "TensorsRanks": [rank1, rank2, ...],
    "TensorShapes": [[shape1], [shape2], ...],
    "TensorDataTypes": [dtype1, dtype2, ...]
}
```

The tensor preprocessor is free to add, remove or edit tensors, as long as the tensors are compatible with the assigned model.

# How to use

Once compiled, copy the executable to an accessible directory. A convenience directory within the Edge AI Manager installation is created for this purpose at `/opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/preprocessors`.

It's a good idea to make sure the application and settings file you add is readable and executable by the NXAI Edge AI Manager. This can be achieved by running:

```
sudo chmod -R 777 /opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/preprocessors
```

## Defining the preprocessor

Create a configuration file at `/opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/preprocessors/external_preprocessors.json` and add the details of your preprocessor to the root object of that file. For example: 

``` json
{
    "externalPreprocessors": [
        {
            "Name": "Example-Preprocessor-Clip",
            "Command": "/opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/preprocessors/preprocessor-python-clip-example",
            "SocketPath": "/tmp/example-clip-preprocessor.sock",
            "Schedule": "TENSOR",
            "Settings": [
                {
                    "type": "Repeater",
                    "count": 5,
                    "startIndex": 1,
                    "template": {
                        "type": "TextField",
                        "name": "externalprocessor.prompt#",
                        "caption": "Prompt #",
                        "description": "Prompt #",
                        "defaultValue": ""
                    }
                }
            ]
        }
    ]
}
```

This tells the Edge AI Manager about the preprocessor:
- **Name** gives the preprocesor a name so it can be selected later
- **Command** defines how to start the preprocessor
- **SocketPath** tells the AI Manager where to send data to so the external preprocessor will receive it
- **Schedule** controls during which stage of the preprocessing pipeline the data is passed to the external processor. See [Schedules](#schedules)

The socket path is always given as the first command line argument when the application is started. It is therefore best practice for the external preprocessor application to read its socket path from here, instead of defining the data twice.

## Schedules

Schedules control during which stage of the processing pipeline the data will be passed to the external processor.

Available schedules are:

- **IMAGE**: The data will be passed to the external processor before any processing is done. The external processor will receive the raw image as it is received by the AI Manager. 
The returned image, if any, will then be resized, normalized, and cast to the specifications of the model.
- **TENSOR**: The data will be passed to the external processor after the image is processed. The external processor will receive a tensor structure containing the data of the processed image. 
The external processor has the option of adding/removing/renaming tensors which will be passed to the model.


## Selecting to the preprocessor

If the preprocessor is defined correctly, its name should appear in the list of preprocessors in the NX Plugin settings. If it is selected in the plugin settings then the Edge AI Runtime will send data to the preprocessor and wait for its output.

# Licence

Copyright 2025, Network Optix, All rights reserved.