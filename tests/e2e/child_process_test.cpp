/*
 * Unit tests for the child-process launcher.
 *
 * The launcher is the harness's only platform-dependent part, so its tests must run on
 * both POSIX and Windows. To stay self-contained the test executable is its own child:
 * invoked with a `--child-*` mode it behaves as the child under test and never runs the
 * assertions, so no fixture binary and no shell is involved.
 *
 * Usage: <test> [--child-exit <code> | --child-ignore-termination]
 */

#include "child_process.h"

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace
{

using namespace std::chrono_literals;

// Long enough for a freshly spawned child to reach its own first statement.
constexpr auto kChildHandlerSetupDelay = 200ms;

bool g_failed = false;

void check(bool condition, const std::string& what)
{
    if (condition)
        return;
    std::cerr << "FAIL: " << what << "\n";
    g_failed = true;
}

// A trivial child must start and its exit status must be observable through wait().
void test_spawns_a_child_and_observes_its_exit(const char* self)
{
    std::string error;
    auto child = e2e::ChildProcess::spawn(
        {const_cast<char*>(self), const_cast<char*>("--child-exit"), const_cast<char*>("7"),
            nullptr},
        e2e::HandleInheritance::inherit, error);
    if (!child.has_value())
    {
        check(false, "spawn failed: " + error);
        return;
    }

    const auto exit_status = child->wait(e2e::ChildProcess::kBlockUntilExit);
    if (!exit_status.has_value())
    {
        check(false, "blocking wait returned while the child was still running");
        return;
    }
    check(exit_status->kind == e2e::ChildExit::Kind::exited,
        "child did not exit normally: " + exit_status->describe());
    check(exit_status->value == 7, "child exit code != 7: " + exit_status->describe());
}

// A processor that ignores the polite request must still be gone when the launcher says it
// asked -- otherwise a hung processor leaves the harness blocked in teardown.
void test_terminates_a_child_that_ignores_the_request(const char* self)
{
    std::string error;
    auto child = e2e::ChildProcess::spawn(
        {const_cast<char*>(self), const_cast<char*>("--child-ignore-termination"), nullptr},
        e2e::HandleInheritance::inherit, error);
    if (!child.has_value())
    {
        check(false, "spawn failed: " + error);
        return;
    }

    // Let the child install its SIGTERM handler first: a child still inside exec() carries the
    // default disposition and would die on the polite signal, which would pass this test
    // without ever exercising the escalation it is about.
    std::this_thread::sleep_for(kChildHandlerSetupDelay);

    const auto started = std::chrono::steady_clock::now();
    child->request_termination();
    const auto elapsed = std::chrono::steady_clock::now() - started;

    check(child->wait(0ms).has_value(), "child survived request_termination()");
#ifndef _WIN32
    // Only POSIX has a polite request to ignore; TerminateProcess is unconditional, so on
    // Windows there is no grace period to observe.
    check(elapsed >= e2e::ChildProcess::kTerminationGrace,
        "child ended before the grace period expired, so SIGTERM was not actually ignored "
        "and the escalation path went untested");
#endif
}

// Waiting again on a child already known to have ended must answer from the remembered
// status: on POSIX the child has been reaped and a second waitpid would report failure, and
// the harness's teardown waits unconditionally after its liveness polls already saw the exit.
void test_waiting_on_an_already_exited_child_returns(const char* self)
{
    std::string error;
    auto child = e2e::ChildProcess::spawn(
        {const_cast<char*>(self), const_cast<char*>("--child-exit"), const_cast<char*>("3"),
            nullptr},
        e2e::HandleInheritance::inherit, error);
    if (!child.has_value())
    {
        check(false, "spawn failed: " + error);
        return;
    }

    const auto first = child->wait(e2e::ChildProcess::kBlockUntilExit);
    const auto second = child->wait(e2e::ChildProcess::kBlockUntilExit);
    if (!second.has_value())
    {
        check(false, "second wait() on an exited child reported it as still running");
        return;
    }
    check(first.has_value() && first->kind == second->kind && first->value == second->value,
        "second wait() reported a different status: " + second->describe());

    // request_termination() on an already-exited child must be a no-op rather than signalling
    // a pid that may since have been reused.
    child->request_termination();
    check(child->wait(0ms).has_value(), "status was lost after request_termination()");
}

// Ignore the polite request and outlive the launcher's grace period; the test's own timeout
// bounds this if termination is broken.
int run_ignoring_termination()
{
#ifndef _WIN32
    std::signal(SIGTERM, SIG_IGN);
#endif
    std::this_thread::sleep_for(60s);
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    const std::string mode = argc > 1 ? argv[1] : "";

    if (mode == "--child-exit")
        return argc > 2 ? std::atoi(argv[2]) : 0;
    if (mode == "--child-ignore-termination")
        return run_ignoring_termination();

    test_spawns_a_child_and_observes_its_exit(argv[0]);
    test_terminates_a_child_that_ignores_the_request(argv[0]);
    test_waiting_on_an_already_exited_child_returns(argv[0]);

    if (g_failed)
        return 1;
    std::cout << "PASS\n";
    return 0;
}
