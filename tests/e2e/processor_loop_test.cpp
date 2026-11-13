/*
 * Tests for the receive loop both C++ examples run: how it leaves the loop and what
 * it does with the channels when it does.
 *
 * The channels are fakes, so each scenario scripts exactly the receive outcomes the loop
 * sees. Each scenario runs once with the Objects example's message functions and once with
 * the Tensors example's, since each example instantiates the loop with its own pair.
 */

#include <csignal>
#include <iostream>
#include <string>
#include <vector>

#include "external_processor_utilities.h"
#include "nxai_ipc_utils.h"
#include "processor_loop.h"

namespace {

bool g_failed = false;

void check(bool condition, const std::string& what)
{
    if (condition)
        return;
    std::cerr << "FAIL: " << what << "\n";
    g_failed = true;
}

// Replays a fixed list of receive outcomes, then reports NOT_READY so a loop that keeps
// running past its script ends as an error instead of hanging.
struct FakeInput
{
    std::vector<ipc_error_t> script;
    size_t received = 0;
    int closed = 0;

    ipc_error_t receive(unsigned char** data, uint32_t* size, int /*timeout_ms*/)
    {
        *data = nullptr;
        *size = 0;
        if (received >= script.size())
            return IPC_ERROR_NOT_READY;
        return script[received++];
    }

    void release() {}
    void close() { ++closed; }
};

struct FakeOutput
{
    int closed = 0;

    ipc_error_t send(const unsigned char* /*data*/, uint32_t /*size*/) { return IPC_ERROR_NONE; }
    void close() { ++closed; }
};

template<typename Parse, typename Handle, typename Write>
void run_scenarios(const std::string& example, Parse parse, Handle handle, Write write)
{
    const volatile sig_atomic_t running = 1;

    {
        FakeInput input{{IPC_EXIT_REQUESTED}};
        FakeOutput output;
        const auto result = run_receive_loop(input, output, running, 5000, parse, handle, write);
        check(result == LoopExit::ExitRequested, example + ": graceful exit is reported as such");
        check(
            input.closed == 1 && output.closed == 1,
            example + ": graceful exit closes both channels");
    }
    {
        FakeInput input{{IPC_UNEXPECTED_SIGNAL}};
        FakeOutput output;
        const auto result = run_receive_loop(input, output, running, 5000, parse, handle, write);
        check(
            result == LoopExit::Error,
            example + ": unexpected signal leaves the loop as an error");
        check(
            input.closed == 1 && output.closed == 1,
            example + ": unexpected signal closes both channels");
    }
    {
        FakeInput input{{IPC_ERROR_TIMEOUT, IPC_ERROR_TIMEOUT, IPC_EXIT_REQUESTED}};
        FakeOutput output;
        const auto result = run_receive_loop(input, output, running, 5000, parse, handle, write);
        check(input.received == 3, example + ": timeouts keep the loop running");
        check(
            result == LoopExit::ExitRequested,
            example + ": loop still honours an exit request after timeouts");
    }
    {
        const volatile sig_atomic_t stopped = 0;
        FakeInput input{{IPC_ERROR_TIMEOUT}};
        FakeOutput output;
        const auto result = run_receive_loop(input, output, stopped, 5000, parse, handle, write);
        check(result == LoopExit::Stopped, example + ": a cleared running flag stops the loop");
        check(input.received == 0, example + ": a stopped loop does not receive again");
        check(
            input.closed == 1 && output.closed == 1,
            example + ": a stopped loop closes both channels");
    }
}

std::string name_from(std::vector<const char*> args)
{
    return processor_name_from_args(
        static_cast<int>(args.size()),
        const_cast<char**>(args.data()));
}

void run_processor_name_scenarios()
{
    check(
        name_from({"bin", "in", "out", "--LogLevel", "DEBUG", "--ProcessorName", "p1", "--x", "y"})
            == "p1",
        "name: found among other --key value pairs");
    check(
        name_from({"bin", "in", "out", "--LogLevel", "DEBUG"}).empty(),
        "name: absent key gives empty");
    check(
        name_from({"bin", "in", "out", "--ProcessorName"}).empty(),
        "name: key without a value gives empty");
    check(
        name_from({"bin", "--ProcessorName", "out", "--LogLevel", "DEBUG"}).empty(),
        "name: argv[1] is not scanned");
    check(
        name_from({"bin", "in", "--ProcessorName", "--LogLevel", "DEBUG"}).empty(),
        "name: argv[2] is not scanned");
}

} // namespace

int main()
{
    run_processor_name_scenarios();
    run_scenarios(
        "objects",
        parseObjectsPostProcessorMessage,
        [](ObjectsPostProcessorMessage&) {},
        writeObjectsPostprocessorMessage);
    run_scenarios(
        "tensors",
        [](const unsigned char* data, uint32_t size)
        {
            return parseTensorMessage(data, size);
        },
        [](TensorsMessage&) {},
        writeTensorMessage);

    if (g_failed)
        return 1;
    std::cout << "processor_loop_test: all checks passed\n";
    return 0;
}
