cmake_minimum_required(VERSION 3.16)
project(postprocessor-cpp-tensors-example CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(NXAI_UTILS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../nxai-utilities")

add_executable(${PROJECT_NAME}
    main.cpp
    "${NXAI_UTILS_DIR}/nxai_utils.cpp"
    "${NXAI_UTILS_DIR}/nxai_ipc_utils.cpp"
    "${NXAI_UTILS_DIR}/external_processor_utilities.cpp"
    "${NXAI_UTILS_DIR}/mpack.c"
)

target_include_directories(${PROJECT_NAME} PRIVATE
    "${NXAI_UTILS_DIR}"
)

if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE rt)
endif()

install(TARGETS ${PROJECT_NAME}
    DESTINATION ${INSTALL_DEST_POSTPROCESSORS}
    COMPONENT ${PROJECT_NAME}
)
