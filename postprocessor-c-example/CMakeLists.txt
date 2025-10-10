
cmake_minimum_required(VERSION 3.10.2)

project(nxai-c-postprocessor-example)

# Generate compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(postprocessor-c-example
  src/main.c
  src/data_utils.c

  # Add local dependencies
  deps/mpack.c
)

target_include_directories(postprocessor-c-example PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/deps)

# Add compile options for all targets
target_compile_options(postprocessor-c-example PUBLIC
  -funsafe-math-optimizations -s -fno-delete-null-pointer-checks -fno-math-errno -Wno-unknown-pragmas -ffunction-sections
  -fPIC -Wall -Wextra -Wshadow -Wmissing-declarations -fno-math-errno
  -fdata-sections -ffunction-sections
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(postprocessor-c-example PUBLIC
    -O3 -Ofast
  )
endif()

# Add linker options
target_link_options(postprocessor-c-example PUBLIC -Wl,--gc-sections -Wl,--as-needed -static -pthread)

# Link generic libraries
target_link_libraries(postprocessor-c-example m dl rt)

# Link utilities libraries
target_link_libraries(postprocessor-c-example nxai-c-utilities)

# Add the installation rule for this specific target
install(TARGETS postprocessor-c-example
    DESTINATION ${INSTALL_DEST_POSTPROCESSORS}
)