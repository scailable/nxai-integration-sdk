/**
 * Example NXAI external postprocessor in C++.
 *
 * Mirrors the Python example: receives detected objects and events from the
 * NXAI runtime, adds a "cpp_objects_example" object to demonstrate processing, and sends
 * the modified results back.
 *
 * Usage: <binary> <input_sync> <output_sync> [--key value ...]
 */

#include <iostream>

#include <signal.h>

#include "external_processor_utilities.h"
#include "nxai_ipc_utils.h"
#include "processor_loop.h"

// ============================================================================
// Constants
// ============================================================================

static constexpr int kReceiveTimeoutMs = 5000;

static constexpr const char* kInjectedClass = "cpp_objects_example";

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

static void handle_message(ObjectsPostProcessorMessage& msg)
{
    std::cout << "Processing: device=" << msg.device_id << ", " << msg.objects.size()
              << " object classes"
              << ", " << msg.events.size() << " events\n";

    // Add a test object under this example's own class to demonstrate postprocessing.
    // kInjectedClass must match the sole Objects[].ID in this example's
    // external_postprocessors.json, and must be unique across every processor
    // registered on the server: the AI Manager merges all registered processors'
    // Objects into every device's manifest without deduplicating, so two processors
    // sharing an ID make the manifest report duplicate object types.
    ObjectData test_obj;
    test_obj.new_track_id = ""; // empty string: VMS generates a new UUID
    test_obj.x = 0.1F;
    test_obj.y = 0.1F;
    test_obj.width = 0.3F;
    test_obj.height = 0.3F;
    test_obj.confidence = 0.95F;
    test_obj.attributes["source"] = "cpp-postprocessor-example";

    msg.objects[kInjectedClass].push_back(test_obj);

    std::cout << "Added \"" << kInjectedClass << "\" object\n";
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
            return parseObjectsPostProcessorMessage(data, size);
        },
        handle_message,
        writeObjectsPostprocessorMessage);

    std::cout << "Postprocessor '" << processor_name << "' shutdown\n";
    return exit_reason == LoopExit::Error ? 1 : 0;
}
