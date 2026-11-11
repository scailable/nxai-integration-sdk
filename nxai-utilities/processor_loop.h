/**
 * The receive loop shared by the C++ example processors.
 *
 * The loop is a template over the two channels so a test can drive it with fakes. Each
 * channel type provides:
 *   input:  ipc_error_t receive(unsigned char** data, uint32_t* size, int timeout_ms);
 *           void release();
 *           void close();
 *   output: ipc_error_t send(const unsigned char* data, uint32_t size);
 *           void close();
 */

#pragma once

#include <csignal>
#include <cstring>
#include <iostream>
#include <string>

#include "nxai_ipc_utils.h"

enum class LoopExit
{
    Stopped, // the running flag was cleared (SIGINT / SIGTERM)
    ExitRequested, // the AI Manager asked for a graceful shutdown
    Error, // an unexpected IPC signal or a failed send
};

// Runs until one of the LoopExit conditions, then closes both channels.
// parse(data, size) -> optional message, called while the input buffer is valid;
// handle(message&) mutates it; write(message) -> optional buffer with .data and .size.
template<typename Input, typename Output, typename Parse, typename Handle, typename Write>
LoopExit run_receive_loop(
    Input& input,
    Output& output,
    const volatile sig_atomic_t& running,
    int receive_timeout_ms,
    Parse parse,
    Handle handle,
    Write write)
{
    LoopExit result = LoopExit::Stopped;

    while (running != 0)
    {
        unsigned char* raw_data = nullptr;
        uint32_t raw_size = 0;

        const ipc_error_t err = input.receive(&raw_data, &raw_size, receive_timeout_ms);

        if (err == IPC_ERROR_TIMEOUT)
            continue;

        if (err == IPC_EXIT_REQUESTED)
        {
            std::cout << "Received exit request. Shutting down.\n";
            result = LoopExit::ExitRequested;
            break;
        }

        if (err != IPC_ERROR_NONE)
        {
            std::cerr << "Error: unexpected IPC signal or failure (" << err
                      << "). Shutting down.\n";
            result = LoopExit::Error;
            break;
        }

        std::cout << "Received message (" << raw_size << " bytes)\n";

        // Parse while the SHM buffer is valid, then immediately release it.
        auto msg = parse(raw_data, raw_size);
        input.release();

        if (!msg.has_value())
        {
            std::cerr << "Error: failed to parse message, skipping\n";
            continue;
        }

        handle(*msg);

        auto response = write(*msg);
        if (response.has_value() == false)
        {
            std::cerr << "Error: failed to serialise response, skipping\n";
            continue;
        }

        const ipc_error_t send_err = output.send(
            reinterpret_cast<const unsigned char*>(response->data.get()),
            static_cast<uint32_t>(response->size));
        if (send_err != IPC_ERROR_NONE)
        {
            std::cerr << "Error: failed to send response (" << send_err << ")\n";
            result = LoopExit::Error;
            break;
        }

        std::cout << "Response sent (" << response->size << " bytes)\n";
    }

    input.close();
    output.close();
    return result;
}

// The AI Manager passes the processor's own name as "--ProcessorName <name>" after the
// two sync descriptors. Returns an empty string when the key is absent.
inline std::string processor_name_from_args(int argc, char* argv[])
{
    for (int i = 3; i + 1 < argc; ++i)
    {
        if (std::strcmp(argv[i], "--ProcessorName") == 0)
            return argv[i + 1];
    }
    return {};
}

// The real channels: thin wrappers over the shared-memory IPC calls.
class ShmChannel
{
public:
    explicit ShmChannel(nxai_ipc_t channel):
        m_channel(channel)
    {
    }

    ipc_error_t receive(unsigned char** data, uint32_t* size, int timeout_ms)
    {
        return nxai_ipc_receive(m_channel, data, size, timeout_ms);
    }

    void release() { release_buffer(m_channel); }

    ipc_error_t send(const unsigned char* data, uint32_t size)
    {
        return nxai_ipc_send(m_channel, data, size);
    }

    void close() { ipc_sync_close(m_channel); }

private:
    nxai_ipc_t m_channel;
};
