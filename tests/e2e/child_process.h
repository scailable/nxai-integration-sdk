/*
 * Child-process launcher for the offline SHM harness.
 *
 * The harness's only platform dependence is starting the processor under test and tearing it
 * down; everything else goes through the nxai_ipc_* API. This module owns that dependence
 * behind three calls -- spawn, request_termination, wait -- and hides pid-versus-HANDLE from
 * its callers entirely.
 */

#ifndef E2E_CHILD_PROCESS_H
#define E2E_CHILD_PROCESS_H

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace e2e
{

// Whether the child inherits the parent's inheritable descriptors/handles. Explicit at the
// interface rather than a per-platform default: the IPC sync objects encoded in the child's
// argv only reach it by inheritance, and an earlier version made that decision silently
// and inconsistently between an event object and the shared mapping beside it.
enum class HandleInheritance
{
    none,
    inherit,
};

// How a child ended. Windows has no signals, so `signalled` occurs only on POSIX.
struct ChildExit
{
    enum class Kind
    {
        exited,
        signalled,
    };

    Kind kind;
    int value; // exit code, or the terminating signal number

    std::string describe() const;
};

class ChildProcess
{
public:
    // Passed to wait() to block until the child ends instead of polling.
    static constexpr std::chrono::milliseconds kBlockUntilExit = std::chrono::milliseconds::max();
    // How long request_termination() lets a child honour the polite request before killing it.
    static constexpr std::chrono::milliseconds kTerminationGrace = std::chrono::milliseconds(500);

    // argv is the execvp-shaped, nullptr-terminated vector; argv[0] is the program to run and
    // is resolved against PATH. Returns nullopt with `error` set if the child cannot start.
    static std::optional<ChildProcess> spawn(
        const std::vector<char*>& argv, HandleInheritance inheritance, std::string& error);

    ChildProcess(ChildProcess&& other) noexcept;
    ChildProcess& operator=(ChildProcess&& other) noexcept;
    ChildProcess(const ChildProcess&) = delete;
    ChildProcess& operator=(const ChildProcess&) = delete;
    ~ChildProcess();

    // Ask the child to end, and make sure it does: a child that ignores the polite request is
    // killed outright after a short grace period.
    void request_termination();

    // Returns how the child ended, or nullopt if it is still running when `timeout` expires.
    // A zero timeout is a liveness poll; kBlockUntilExit blocks. Once the child has ended the
    // status is remembered, so repeated waits answer immediately instead of hanging.
    std::optional<ChildExit> wait(std::chrono::milliseconds timeout);

private:
    explicit ChildProcess(std::uintptr_t handle);

    // Opaque process identity -- a pid on POSIX, a process HANDLE on Windows. Never exposed.
    std::uintptr_t handle_;
    // Remembered once observed, so that a second wait() answers instead of hanging (on POSIX
    // a reaped child is gone and a further waitpid would fail rather than report it again).
    std::optional<ChildExit> exit_status_;
};

} // namespace e2e

#endif // E2E_CHILD_PROCESS_H
