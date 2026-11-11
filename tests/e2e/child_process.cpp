#include "child_process.h"

#include <cerrno>
#include <cstring>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace e2e
{

#ifndef _WIN32
namespace
{

// POSIX has no waitpid() with a timeout, so a bounded wait polls at this interval.
constexpr std::chrono::milliseconds kPollInterval = std::chrono::milliseconds(10);

} // namespace
#endif

std::string ChildExit::describe() const
{
    return kind == Kind::exited ? "exited with code " + std::to_string(value)
                                : "killed by signal " + std::to_string(value);
}

ChildProcess::ChildProcess(std::uintptr_t handle): handle_(handle) {}

ChildProcess::ChildProcess(ChildProcess&& other) noexcept
    : handle_(other.handle_), exit_status_(other.exit_status_)
{
    other.handle_ = 0;
}

ChildProcess& ChildProcess::operator=(ChildProcess&& other) noexcept
{
    if (this != &other)
    {
        handle_ = other.handle_;
        exit_status_ = other.exit_status_;
        other.handle_ = 0;
    }
    return *this;
}

#ifndef _WIN32

std::optional<ChildProcess> ChildProcess::spawn(
    const std::vector<char*>& argv, HandleInheritance inheritance, std::string& error)
{
    const pid_t pid = fork();
    if (pid < 0)
    {
        error = std::string("fork failed: ") + std::strerror(errno);
        return std::nullopt;
    }
    if (pid == 0)
    {
        // Child. Descriptors are inherited across fork/exec unless they carry FD_CLOEXEC, so
        // "none" is expressed by closing everything the child was not meant to keep. The
        // harness always inherits -- the IPC sync objects reach the child no other way.
        if (inheritance == HandleInheritance::none)
        {
            for (int fd = STDERR_FILENO + 1; fd < static_cast<int>(sysconf(_SC_OPEN_MAX)); ++fd)
                close(fd);
        }
        execvp(argv[0], const_cast<char* const*>(argv.data()));
        // Reached only if exec failed; the parent sees it as exit code 127, as a shell would
        // report it.
        _exit(127);
    }
    return ChildProcess(static_cast<std::uintptr_t>(pid));
}

ChildProcess::~ChildProcess() = default;

void ChildProcess::request_termination()
{
    if (exit_status_.has_value())
        return;

    kill(static_cast<pid_t>(handle_), SIGTERM);
    if (wait(kTerminationGrace).has_value())
        return;

    // The child ignored SIGTERM (processors may install their own handler, or none at all
    // while blocked in a read), so stop asking.
    kill(static_cast<pid_t>(handle_), SIGKILL);
    wait(kBlockUntilExit);
}

std::optional<ChildExit> ChildProcess::wait(std::chrono::milliseconds timeout)
{
    if (exit_status_.has_value())
        return exit_status_;

    const auto reap = [this](int flags) -> bool
    {
        int status = 0;
        const pid_t reaped = waitpid(static_cast<pid_t>(handle_), &status, flags);
        if (reaped != static_cast<pid_t>(handle_))
            return false;
        exit_status_ = WIFSIGNALED(status)
            ? ChildExit{ChildExit::Kind::signalled, WTERMSIG(status)}
            : ChildExit{ChildExit::Kind::exited, WEXITSTATUS(status)};
        return true;
    };

    if (timeout == kBlockUntilExit)
    {
        reap(0);
        return exit_status_;
    }

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true)
    {
        if (reap(WNOHANG))
            return exit_status_;
        if (std::chrono::steady_clock::now() >= deadline)
            return std::nullopt;
        std::this_thread::sleep_for(kPollInterval);
    }
}

#else // _WIN32

namespace
{

// CreateProcess takes one command line, not an argv, and re-splits it by the CRT rules the
// child will apply. Quote any argument that would otherwise split, doubling the backslashes
// that precede the closing quote so a trailing backslash cannot escape it.
std::string quote_argument(const char* argument)
{
    const std::string value(argument);
    if (!value.empty() && value.find_first_of(" \t\"") == std::string::npos)
        return value;

    std::string quoted = "\"";
    size_t pending_backslashes = 0;
    for (const char c: value)
    {
        if (c == '\\')
        {
            ++pending_backslashes;
            continue;
        }
        if (c == '"')
        {
            quoted.append(pending_backslashes * 2 + 1, '\\');
            pending_backslashes = 0;
        }
        else
        {
            quoted.append(pending_backslashes, '\\');
            pending_backslashes = 0;
        }
        quoted.push_back(c);
    }
    quoted.append(pending_backslashes * 2, '\\');
    quoted.push_back('"');
    return quoted;
}

std::string format_last_error(DWORD code)
{
    char* text = nullptr;
    const DWORD length = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER
            | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, 0, reinterpret_cast<char*>(&text), 0, nullptr);
    std::string message = length == 0 ? "error " + std::to_string(code) : std::string(text, length);
    if (text != nullptr)
        LocalFree(text);
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r'))
        message.pop_back();
    return message;
}

} // namespace

std::optional<ChildProcess> ChildProcess::spawn(
    const std::vector<char*>& argv, HandleInheritance inheritance, std::string& error)
{
    std::string command_line;
    for (const char* argument: argv)
    {
        if (argument == nullptr) // The execvp-shaped terminator.
            break;
        if (!command_line.empty())
            command_line.push_back(' ');
        command_line += quote_argument(argument);
    }

    STARTUPINFOA startup_info{};
    startup_info.cb = sizeof(startup_info);
    PROCESS_INFORMATION process_info{};

    // lpApplicationName is null so that argv[0] is resolved against PATH, as execvp does.
    // bInheritHandles carries the caller's decision: the harness's sync objects are created
    // inheritable and reach the child only when this is TRUE.
    if (CreateProcessA(nullptr, command_line.data(), nullptr, nullptr,
            inheritance == HandleInheritance::inherit ? TRUE : FALSE, 0, nullptr, nullptr,
            &startup_info, &process_info)
        == 0)
    {
        error = "CreateProcess failed: " + format_last_error(GetLastError());
        return std::nullopt;
    }

    // Only the process handle is kept; the initial thread's is of no use here.
    CloseHandle(process_info.hThread);
    return ChildProcess(reinterpret_cast<std::uintptr_t>(process_info.hProcess));
}

ChildProcess::~ChildProcess()
{
    if (handle_ != 0)
        CloseHandle(reinterpret_cast<HANDLE>(handle_));
}

void ChildProcess::request_termination()
{
    if (exit_status_.has_value())
        return;

    // Windows has no polite request a console process can decline the way SIGTERM can be
    // ignored, so there is no grace period to observe here: the process is ended outright and
    // the wait below only collects the status.
    TerminateProcess(reinterpret_cast<HANDLE>(handle_), 1);
    wait(kBlockUntilExit);
}

std::optional<ChildExit> ChildProcess::wait(std::chrono::milliseconds timeout)
{
    if (exit_status_.has_value())
        return exit_status_;

    const DWORD wait_ms = timeout == kBlockUntilExit
        ? INFINITE
        : static_cast<DWORD>(timeout.count());
    if (WaitForSingleObject(reinterpret_cast<HANDLE>(handle_), wait_ms) != WAIT_OBJECT_0)
        return std::nullopt;

    DWORD code = 0;
    GetExitCodeProcess(reinterpret_cast<HANDLE>(handle_), &code);
    exit_status_ = ChildExit{ChildExit::Kind::exited, static_cast<int>(code)};
    return exit_status_;
}

#endif // _WIN32

} // namespace e2e
