cd "$(dirname $0)"

# Create build folder
mkdir -p build
# Make sure build folder is empty
rm build/* -r
# Configure CMake
cmake -B build -S . -DINSTALL_DEST_PREPROCESSORS=${PWD}/build/install/ -DINSTALL_DEST_POSTPROCESSORS=${PWD}/build/install/
# Build install target
cmake --build build --target install
