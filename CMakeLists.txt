cmake_minimum_required(VERSION 3.30)

project(nxai-integration-examples)

# Define common installation destinations
if(WIN32)
    if(NOT DEFINED CACHE{INSTALL_DEST_POSTPROCESSORS})
        set(INSTALL_DEST_POSTPROCESSORS "C:\\Windows\\System32\\config\\systemprofile\\AppData\\Local\\Network Optix\\Network Optix MetaVMS Media Server\\nx_ai_manager\\nxai_manager\\postprocessors" CACHE PATH "Installation destination for postprocessors")
    endif()
    if(NOT DEFINED CACHE{INSTALL_DEST_PREPROCESSORS})
        set(INSTALL_DEST_PREPROCESSORS "C:\\Windows\\System32\\config\\systemprofile\\AppData\\Local\\Network Optix\\Network Optix MetaVMS Media Server\\nx_ai_manager\\nxai_manager\\preprocessors" CACHE PATH "Installation destination for preprocessors")
    endif()
else()
    # Auto-detect NX VMS installation. Supports both networkoptix-metavms (NX Meta) and
    # networkoptix (NX Witness). Override with:
    #   cmake -DINSTALL_DEST_POSTPROCESSORS=<path> -DINSTALL_DEST_PREPROCESSORS=<path>
    if(NOT DEFINED CACHE{INSTALL_DEST_POSTPROCESSORS} OR NOT DEFINED CACHE{INSTALL_DEST_PREPROCESSORS})
        execute_process(COMMAND systemctl is-active networkoptix-metavms-mediaserver
            OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE _nxmeta_active)
        execute_process(COMMAND systemctl is-active networkoptix-mediaserver
            OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE _nxwitness_active)

        if(_nxmeta_active EQUAL 0)
            set(_NX_BASE "/opt/networkoptix-metavms")
        elseif(_nxwitness_active EQUAL 0)
            set(_NX_BASE "/opt/networkoptix")
        elseif(EXISTS "/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/")
            set(_NX_BASE "/opt/networkoptix-metavms")
        elseif(EXISTS "/opt/networkoptix/mediaserver/var/nx_ai_manager/")
            set(_NX_BASE "/opt/networkoptix")
        else()
            set(_NX_BASE "/opt/networkoptix-metavms")
        endif()
        message(STATUS "NX VMS base detected: ${_NX_BASE}")
    endif()

    if(NOT DEFINED CACHE{INSTALL_DEST_POSTPROCESSORS})
        set(INSTALL_DEST_POSTPROCESSORS "${_NX_BASE}/mediaserver/var/nx_ai_manager/nxai_manager/postprocessors/" CACHE PATH "Installation destination for postprocessors")
    endif()
    if(NOT DEFINED CACHE{INSTALL_DEST_PREPROCESSORS})
        set(INSTALL_DEST_PREPROCESSORS "${_NX_BASE}/mediaserver/var/nx_ai_manager/nxai_manager/preprocessors/" CACHE PATH "Installation destination for preprocessors")
    endif()
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

## Add ANPR Example Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-anpr-example)

## Add Confidence Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-confidences-example)

## Add Settings Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-settings-example)

## Add Postprocessor Python Events project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-events-example)

## Add Postprocessor Python Measure Average Car Speed project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-measure-average-car-speed)

## Add Postprocessor Python NoResponse project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-noresponse-example)

## Add EdgeImpulse Upload Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-edgeimpulse-example)

## Add Cloud Inference Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-cloud-inference-example)

# Add Clip Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-clip-example)

## Add Height Estimator Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-height-estimator)

## Add Object Counter Threshold Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-object-counter-threshold)

## Add Web Dashboard Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-web-dashboard)

# Add Image Preprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/preprocessor-python-image-example)

# Add Tensor Preprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/preprocessor-python-tensor-example)

# Add Clip Preprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/preprocessor-python-clip-example)
