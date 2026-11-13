/**
 * Example NXAI external postprocessor in C++ for the TENSOR schedule.
 *
 * Receives the raw output tensors of the model from the NXAI runtime, replaces them
 * with one box tensor of the class "cpp_tensors_example", and sends the result back.
 * The engine turns that tensor into an object, as it does for the box output of a model.
 *
 * Usage: <binary> <input_sync> <output_sync> [--key value ...]
 */

#include <iostream>

#include <signal.h>

#include "external_processor_utilities.h"
#include "nxai_ipc_utils.h"
#include "oaax/interface.h"
#include "processor_loop.h"

// ============================================================================
// Constants
// ============================================================================

static constexpr int kReceiveTimeoutMs = 5000;

// ============================================================================
// Globals
// ============================================================================

// Must be volatile sig_atomic_t for async-signal-safety.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static volatile sig_atomic_t g_running = 1;

static void on_signal(int /*signum*/)
{
    g_running = 0;
}

// ============================================================================
// Business logic
// ============================================================================

static void handle_message(TensorsMessage& msg)
{
    std::cout << "Processing: device=" << msg.device_id << ", " << msg.tensors.size()
              << " tensors\n";

    // Iterate through tensors
    std::cout << "Message has tensors:\n";
    for (auto& tensor_data: msg.tensors)
    {
        std::cout << "\t" << tensor_data.name << "\n";
    }
    // Replace the tensors with a single bbox tensor of this example's own class.
    // The class encoded in the tensor name must match the sole Objects[].ID in this
    // example's external_postprocessors.json, and must be unique across every
    // processor registered on the server: the AI Manager merges all registered
    // processors' Objects into every device's manifest without deduplicating, so two
    // processors sharing an ID make the manifest report duplicate types.
    msg.tensors.clear();
    // One xyxysc row: x1, y1, x2, y2, score, class index. The corners are pixels of the
    // model's input image, not of the frame and not normalized: the engine maps them to
    // the frame. If all four corners are below 1, the engine reads them as fractions of
    // the model input instead.
    float binary_object[] = {100.0f, 100.0f, 200.0f, 200.0f, 1.0f, 0.0f};
    TensorData new_tensor = {
        .name = "bboxes-format:xyxysc;0:cpp_tensors_example",
        .data_type = DATA_TYPE_FLOAT,
        .shape = {1, 6},
        .data = std::vector<char>(
            reinterpret_cast<char*>(binary_object),
            reinterpret_cast<char*>(binary_object + 6))};
    msg.tensors.push_back(new_tensor);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[])
{
    std::cout << "External postprocessor started with " << argc << " arguments.\n";

    if (argc < 3)
    {
        std::cerr << "Error: too few arguments.\n"
                  << "Usage: " << argv[0] << " <input_sync> <output_sync> [--key value ...]\n";
        return 1;
    }

    const char* ipc_input_str = argv[1];
    const char* ipc_output_str = argv[2];
    const std::string processor_name = processor_name_from_args(argc, argv);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    // Initialise IPC channels from strings passed by the AI Manager.
    // ipc_from_str() sets IPC_SYNC_HOST; override to IPC_SYNC_CLIENT because
    // this process is the downstream consumer of the channels.
    nxai_ipc_t input_channel = ipc_from_str(ipc_input_str);
    if (input_channel == IPC_SYNC_INVALID)
    {
        std::cerr << "Error: failed to initialise input IPC channel\n";
        return 1;
    }
    input_channel->role = IPC_SYNC_CLIENT;

    nxai_ipc_t output_channel = ipc_from_str(ipc_output_str);
    if (output_channel == IPC_SYNC_INVALID)
    {
        std::cerr << "Error: failed to initialise output IPC channel\n";
        ipc_sync_close(input_channel);
        return 1;
    }
    output_channel->role = IPC_SYNC_CLIENT;

    // Signal readiness on the input channel only: the AI Manager consumes it there and
    // never reads a READY on the output channel, where it would be mistaken for a
    // response signal.
    if (nxai_ipc_send_ready(input_channel) != IPC_ERROR_NONE)
        std::cerr << "Warning: failed to signal READY on input channel\n";

    std::cout << "Postprocessor '" << processor_name << "' initialised. Waiting for messages...\n";

    // Main message processing loop; closes both channels on the way out.
    ShmChannel input(input_channel);
    ShmChannel output(output_channel);
    const LoopExit exit_reason = run_receive_loop(
        input,
        output,
        g_running,
        kReceiveTimeoutMs,
        [](const unsigned char* data, uint32_t size)
        {
            return parseTensorMessage(data, size);
        },
        handle_message,
        writeTensorMessage);

    std::cout << "Postprocessor '" << processor_name << "' shutdown\n";
    return exit_reason == LoopExit::Error ? 1 : 0;
}
