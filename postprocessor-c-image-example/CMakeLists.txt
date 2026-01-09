cmake_minimum_required(VERSION 3.30)

project(postprocessor-c-image-example)

# Generate compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PROCESSOR_NAME "postprocessor-c-image-example")

add_executable(${PROCESSOR_NAME}
  src/main.c
  src/data_utils.c

  # Add local dependencies
  deps/mpack.c
)

target_include_directories(${PROCESSOR_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/deps)

# Add compile options for all targets

if (MSVC)
  target_compile_options(${PROCESSOR_NAME} PRIVATE /wd5045 /wd4061 /wd4774 /wd4820)
else()
  target_compile_options(${PROCESSOR_NAME} PUBLIC
    -funsafe-math-optimizations -s -fno-delete-null-pointer-checks -fno-math-errno
    -fPIC -Wall -fno-math-errno
    -fdata-sections -ffunction-sections -Wno-unknown-pragmas -Wextra -Wshadow -Wmissing-declarations)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(${PROCESSOR_NAME} PUBLIC
    -O3 -Ofast
  )
endif()

# Link generic libraries
if (NOT MSVC)
  target_link_options(${PROCESSOR_NAME} PUBLIC -Wl,--gc-sections -Wl,--as-needed -static -pthread)
  target_link_libraries(${PROCESSOR_NAME} m dl rt)
endif()

# Link utilities libraries
target_link_libraries(${PROCESSOR_NAME} nxai-c-utilities)

# Add the installation rule for this specific target
install(TARGETS ${PROCESSOR_NAME}
    DESTINATION ${INSTALL_DEST_POSTPROCESSORS}
    COMPONENT ${PROCESSOR_NAME}
)