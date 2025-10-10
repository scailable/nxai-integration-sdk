# Add target to build everything at once
set(PROCESSOR_NAME "postprocessor-python-settings-example")

# Create virtual environment in build directory
if(NOT EXISTS "${CMAKE_BINARY_DIR}/venv")
    execute_process(
        COMMAND python3 -m venv ${CMAKE_BINARY_DIR}/venv
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

# Activate virtual environment for subsequent commands
set(ENV{VIRTUAL_ENV} "${CMAKE_BINARY_DIR}/venv")
set(ENV{PATH} "${CMAKE_BINARY_DIR}/venv/bin:$ENV{PATH}")
set(PIP_PATH "$ENV{VIRTUAL_ENV}/bin/pip")
set(Python3_Executable "$ENV{VIRTUAL_ENV}/bin/python3")

add_custom_target(${PROCESSOR_NAME} ALL
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS nxai-c-utilities-shared
    COMMAND ${PIP_PATH} install -r ../nxai-utilities/python-utilities/requirements.txt
    COMMAND ${PIP_PATH} install -r requirements.txt
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../nxai-utilities/python-utilities/nxai_communication_utils.py ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${Python3_Executable} -m nuitka ${CMAKE_CURRENT_SOURCE_DIR}/${PROCESSOR_NAME}.py --onefile --output-filename=${PROCESSOR_NAME} --output-dir=${CMAKE_CURRENT_BINARY_DIR}
    COMMAND rm ${CMAKE_CURRENT_SOURCE_DIR}/nxai_communication_utils.py
)

# Add the installation rule for this program
install(
    PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${PROCESSOR_NAME}
    DESTINATION ${INSTALL_DEST_POSTPROCESSORS}
    COMPONENT ${PROCESSOR_NAME}
)

install(
    TARGETS nxai-c-utilities-shared
    DESTINATION ${INSTALL_DEST_POSTPROCESSORS}
    COMPONENT ${PROCESSOR_NAME}
)