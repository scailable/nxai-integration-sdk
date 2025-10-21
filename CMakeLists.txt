cmake_minimum_required(VERSION 3.31)

project(nxai-integration-examples)

# Define common installation destinations
if (WIN32)
    set(INSTALL_DEST_POSTPROCESSORS "C:\\Program Files\\Network Optix\\Nx Meta\\MediaServer\\plugins\\nx_ai_manager_plugin\\nxai_manager\\postprocessors")
    set(INSTALL_DEST_PREPROCESSORS "C:\\Program Files\\Network Optix\\Nx Meta\\MediaServer\\plugins\\nx_ai_manager_plugin\\nxai_manager\\preprocessors")
else()
    set(INSTALL_DEST_POSTPROCESSORS "/opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/postprocessors/")
    set(INSTALL_DEST_PREPROCESSORS "/opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/preprocessors/")
endif()

# Check if the compiler is MSVC
if(MSVC)
    # Apply the flag specifically to the 'my_app' target
    # PRIVATE: The option is only for building this target, not for consumers.
    add_compile_options(/Zc:preprocessor)
    link_libraries(ws2_32)
endif()

# Add Scailable C Utilities for all subprojects
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nxai-utilities)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nxai-utilities/include)

# Add C Postprocessor project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-c-example)

## Add C Image Postprocessor project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-c-image-example)

## Add C Raw Postprocessor project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-c-raw-example)

## Add Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-example)

## Add Image Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-image-example)

## Add Confidence Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-confidences-example)

## Add Settings Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-settings-example)

## Add Postprocessor Python Events project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-events-example)

## Add Postprocessor Python NoResponse project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-noresponse-example)

## Add Cloud Inference Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-cloud-inference-example)

# Add Clip Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-clip-example)

# Add Image Preprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/preprocessor-python-image-example)

# Add Tensor Preprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/preprocessor-python-tensor-example)

# Add Clip Preprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/preprocessor-python-clip-example)
