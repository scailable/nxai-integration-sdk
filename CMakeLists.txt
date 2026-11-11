cmake_minimum_required(VERSION 3.28)

project(nxai-integration-examples)

# Define common installation destinations
if(WIN32)
    # The Windows install tree is the mediaserver service account's local app data,
    # keyed on the product's display names rather than its linux company id, and with
    # no mediaserver/var level. Both products are supported, detected the same way the
    # linux branch below detects them; override with:
    #   cmake -DINSTALL_DEST_POSTPROCESSORS=<path> -DINSTALL_DEST_PREPROCESSORS=<path>
    if(NOT DEFINED CACHE{INSTALL_DEST_POSTPROCESSORS} OR NOT DEFINED CACHE{INSTALL_DEST_PREPROCESSORS})
        set(_NX_WIN_APP_DATA "C:\\Windows\\System32\\config\\systemprofile\\AppData\\Local")
        set(_NX_METAVMS_BASE "${_NX_WIN_APP_DATA}\\Network Optix\\Network Optix MetaVMS Media Server")
        set(_NX_WITNESS_BASE "${_NX_WIN_APP_DATA}\\Network Optix\\Network Optix Media Server")

        if(EXISTS "${_NX_METAVMS_BASE}\\nx_ai_manager")
            set(_NX_BASE "${_NX_METAVMS_BASE}")
        elseif(EXISTS "${_NX_WITNESS_BASE}\\nx_ai_manager")
            set(_NX_BASE "${_NX_WITNESS_BASE}")
        else()
            set(_NX_BASE "${_NX_METAVMS_BASE}")
        endif()
        message(STATUS "NX VMS base detected: ${_NX_BASE}")
    endif()

    if(NOT DEFINED CACHE{INSTALL_DEST_POSTPROCESSORS})
        set(INSTALL_DEST_POSTPROCESSORS "${_NX_BASE}\\nx_ai_manager\\postprocessors" CACHE PATH "Installation destination for postprocessors")
    endif()
    if(NOT DEFINED CACHE{INSTALL_DEST_PREPROCESSORS})
        set(INSTALL_DEST_PREPROCESSORS "${_NX_BASE}\\nx_ai_manager\\preprocessors" CACHE PATH "Installation destination for preprocessors")
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
        set(INSTALL_DEST_POSTPROCESSORS "${_NX_BASE}/mediaserver/var/nx_ai_manager/postprocessors/" CACHE PATH "Installation destination for postprocessors")
    endif()
    if(NOT DEFINED CACHE{INSTALL_DEST_PREPROCESSORS})
        set(INSTALL_DEST_PREPROCESSORS "${_NX_BASE}/mediaserver/var/nx_ai_manager/preprocessors/" CACHE PATH "Installation destination for preprocessors")
    endif()
endif()

# Check if the compiler is MSVC
if(MSVC)
    # Apply the flag specifically to the 'my_app' target
    # PRIVATE: The option is only for building this target, not for consumers.
    add_compile_options(/Zc:preprocessor)
    link_libraries(ws2_32)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nxai-utilities)

# Add CPP Objects Postprocessor project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-cpp-objects-example)

# Add CPP Tensors Postprocessor project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-cpp-tensors-example)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-example)

# Add Settings Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-settings-example)

# Add Object Counter Threshold Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-object-counter-threshold)

# Add BestShots Example Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-bestshots-example)

# Add Height Estimator Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-height-estimator)

# Add Web Dashboard Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-web-dashboard)

# Add Image Example Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-image-example)

# Add Confidences Example Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-confidences-example)

# Add Events Example Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-events-example)

# Add NoResponse Example Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-noresponse-example)

# Add Tensor Preprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/preprocessor-python-tensor-example)

# Add Image Preprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/preprocessor-python-image-example)

# Add CLIP Preprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/preprocessor-python-clip-example)

# Add ANPR Example Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-anpr-example)

# Add CLIP Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-clip-example)

# Add Measure Average Car Speed Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-measure-average-car-speed)

# Add Cloud Inference Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-cloud-inference-example)

# Add Edge Impulse Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-edgeimpulse-example)

# Add Objects Sender Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-objects-sender)

# End-to-end test harness. Builds on Linux and on Windows/MSVC, and every case inside it
# is offered on both platforms; none is gated by platform.
enable_testing()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e)

# The tests/vms-e2e directory is optional and Linux-only: it is added only where it exists.
if(UNIX AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/vms-e2e)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests/vms-e2e)
endif()
