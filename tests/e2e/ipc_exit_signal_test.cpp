/*
 * Unit tests for the outcome nxai_ipc_receive() reports per incoming signal.
 *
 * Before IPC_EXIT_REQUESTED existed, a graceful EXIT from the AI Manager and a genuine
 * protocol violation returned the identical IPC_UNEXPECTED_SIGNAL, so the C++ examples
 * exited for a reason nothing could assert -- and the Python SDK, which does distinguish the
 * two, disagreed with the C++ binding on the shutdown contract. These tests pin the
 * distinction at the receive call, which is the only place it can be observed.
 *
 * The signals are driven through ipc_sync_signal() rather than nxai_ipc_send(): the framed
 * sender always follows a reallocation with NOTICE, so an EXIT arriving as the follow-up
 * signal after a reallocation -- the second place the receive path decides this -- is
 * unreachable through the public API.
 */

#include "nxai_ipc_utils.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

namespace
{

bool g_failed = false;

void check(bool condition, const std::string& what)
{
    if (condition)
        return;
    std::cerr << "FAIL: " << what << "\n";
    g_failed = true;
}

// The host and the client end of one channel, both in this process. The host is the AI
// Manager's side: it signals downstream, which is the direction the client receives from.
struct ChannelPair
{
    nxai_ipc_t host = IPC_SYNC_INVALID;
    nxai_ipc_t client = IPC_SYNC_INVALID;

    bool open()
    {
        host = nxai_ipc_create(1024);
        if (host == IPC_SYNC_INVALID)
            return false;
        char* descriptor = ipc_sync_to_str(host);
        client = ipc_from_str(descriptor);
        free(descriptor);
        if (client == IPC_SYNC_INVALID)
            return false;
        // ipc_from_str() leaves the loaded channel as IPC_SYNC_HOST, so both ends would
        // signal downstream and wait upstream and neither would hear the other. Both C++
        // examples override the role the same way right after loading their channels.
        client->role = IPC_SYNC_CLIENT;
        return true;
    }

    ~ChannelPair()
    {
        ipc_sync_close(client);
        ipc_sync_close(host);
    }
};

// How long a receive may wait for a signal that is already pending, or for the thread below
// to produce one. Generous: nothing here is timing-sensitive except the timeout case.
const int WAIT_MS = 2000;

// A graceful shutdown request must be reported as itself, so a receive loop can leave
// without logging an error.
void test_exit_is_reported_as_a_shutdown_request()
{
    ChannelPair channel;
    if (!channel.open())
    {
        check(false, "could not open a channel pair");
        return;
    }

    check(
        ipc_sync_signal(channel.host, IPC_SIG_EXIT) == IPC_ERROR_NONE,
        "could not send EXIT downstream");

    unsigned char* data = nullptr;
    uint32_t data_size = 0;
    check(
        nxai_ipc_receive(channel.client, &data, &data_size, WAIT_MS) == IPC_EXIT_REQUESTED,
        "nxai_ipc_receive() did not report an EXIT signal as a shutdown request");
}

// A signal that has no meaning at this point in the protocol stays an error, and must not be
// swept into the shutdown outcome.
void test_unexpected_signal_stays_an_error()
{
    ChannelPair channel;
    if (!channel.open())
    {
        check(false, "could not open a channel pair");
        return;
    }

    check(
        ipc_sync_signal(channel.host, IPC_SIG_ERROR) == IPC_ERROR_NONE,
        "could not send ERROR downstream");

    unsigned char* data = nullptr;
    uint32_t data_size = 0;
    check(
        nxai_ipc_receive(channel.client, &data, &data_size, WAIT_MS) == IPC_UNEXPECTED_SIGNAL,
        "nxai_ipc_receive() did not report an out-of-protocol signal as unexpected");
}

// An idle channel must still time out rather than report either of the above: a processor
// polling for work distinguishes "nothing yet" from "stop" from "broken".
void test_no_signal_times_out()
{
    ChannelPair channel;
    if (!channel.open())
    {
        check(false, "could not open a channel pair");
        return;
    }

    unsigned char* data = nullptr;
    uint32_t data_size = 0;
    check(
        nxai_ipc_receive(channel.client, &data, &data_size, 100) == IPC_ERROR_TIMEOUT,
        "nxai_ipc_receive() did not report an idle channel as a timeout");
}

// The second place the receive path reads a signal: after a reallocation it acknowledges,
// re-attaches and waits again. A shutdown request arriving there must be reported the same
// way as one arriving directly -- the AI Manager is free to stop the processor at that point.
//
// This needs a second thread. The signal travels as one byte in shared memory, so both
// signals cannot be queued up front: the second write would overwrite the first before the
// receive path read it.
void test_exit_after_a_reallocation_is_reported_as_a_shutdown_request()
{
    ChannelPair channel;
    if (!channel.open())
    {
        check(false, "could not open a channel pair");
        return;
    }

    ipc_signal_t acknowledgement = IPC_SIG_NOTICE;
    std::thread host_side(
        [&]()
        {
            ipc_sync_signal(channel.host, IPC_SIG_REALLOC);
            if (ipc_sync_wait(channel.host, &acknowledgement, WAIT_MS) != IPC_ERROR_NONE)
                return;
            ipc_sync_signal(channel.host, IPC_SIG_EXIT);
        });

    unsigned char* data = nullptr;
    uint32_t data_size = 0;
    const ipc_error_t error = nxai_ipc_receive(channel.client, &data, &data_size, WAIT_MS);
    host_side.join();

    check(
        acknowledgement == IPC_SIG_ACK,
        "the receive path did not acknowledge the reallocation, so the EXIT below was never "
        "sent and this case proves nothing");
    check(
        error == IPC_EXIT_REQUESTED,
        "nxai_ipc_receive() did not report an EXIT arriving as the follow-up signal after a "
        "reallocation as a shutdown request");
}

} // namespace

int main()
{
    test_exit_is_reported_as_a_shutdown_request();
    test_unexpected_signal_stays_an_error();
    test_no_signal_times_out();
    test_exit_after_a_reallocation_is_reported_as_a_shutdown_request();

    if (g_failed)
        return 1;
    std::cout << "PASS\n";
    return 0;
}
