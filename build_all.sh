# Create build folder
mkdir -p build
# Make sure build folder is empty
rm build/* -r
# Configure CMake
cmake -B build -S .
# Build install target
cmake --build build --target install
# Copy created binaries to local
mkdir -p build/postprocessors
mkdir -p build/preprocessors
cp /opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/postprocessors/* build/postprocessors
cp /opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/preprocessors/* build/preprocessors
