
cmake_minimum_required(VERSION 3.10.2)

project(nxai-c-raw-postprocessor-example)

# Generate compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(postprocessor-c-raw-example
  src/main.c
  src/data_utils.c

  # Add local dependencies
  deps/mpack.c
)

target_include_directories(postprocessor-c-raw-example PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/deps)

# Add compile options for all targets
target_compile_options(postprocessor-c-raw-example PUBLIC
  -funsafe-math-optimizations -s -fno-delete-null-pointer-checks -fno-math-errno -Wno-unknown-pragmas -ffunction-sections
  -fPIC -Wall -Wextra -Wshadow -Wmissing-declarations -fno-math-errno
  -fdata-sections -ffunction-sections
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(postprocessor-c-raw-example PUBLIC
    -O3 -Ofast
  )
endif()

# Add linker options
target_link_options(postprocessor-c-raw-example PUBLIC -Wl,--gc-sections -Wl,--as-needed -static -pthread)

# Link generic libraries
target_link_libraries(postprocessor-c-raw-example m dl rt)

# Link utilities libraries
target_link_libraries(postprocessor-c-raw-example nxai-c-utilities)

# Add the installation rule for this specific target
install(TARGETS postprocessor-c-example
    DESTINATION ${INSTALL_DEST_POSTPROCESSORS}
)