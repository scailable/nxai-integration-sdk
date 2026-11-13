/*
 * Runtime evidence for the two nxai_ipc_utils.cpp failure paths.
 *
 * Of six defects found in these paths, four are proven by a passing happy path. Two are not:
 * the cleanup branch in nxai_ipc_create() and the descriptor-parsing branch in
 * ipc_from_str(). Both referenced a variable that did not exist, so neither had ever
 * executed on any platform; compiling proves the names resolve and nothing more.
 *
 * These tests drive both branches deliberately and assert the observable consequence -- the
 * call reports failure, the process survives, and repeating the failure accumulates no
 * handles or descriptors -- rather than asserting which platform call was made.
 */

#include "nxai_ipc_utils.h"

#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
    #include <psapi.h>
#else
    #include <dirent.h>
    #include <unistd.h>
#endif

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

// How many kernel resources this process currently holds. A leak on a failure path shows up
// here as growth across repetitions; what exactly leaked (an event, a file mapping, a
// descriptor) is deliberately not distinguished, since that is the platform detail these
// tests must not depend on.
size_t held_resource_count()
{
#ifdef _WIN32
    DWORD count = 0;
    if (!GetProcessHandleCount(GetCurrentProcess(), &count))
        return 0;
    return count;
#else
    DIR* fd_dir = opendir("/proc/self/fd");
    if (!fd_dir)
        return 0;
    size_t count = 0;
    while (readdir(fd_dir) != nullptr)
        count++;
    closedir(fd_dir);
    return count;
#endif
}

// Repetitions are cheap and a single leaked resource per attempt has to be visible above the
// noise of the runtime's own allocations, which the slack below absorbs.
const int REPETITIONS = 200;
const size_t LEAK_SLACK = 8;

void check_no_accumulation(size_t before, const std::string& what)
{
    const size_t after = held_resource_count();
    check(
        after <= before + LEAK_SLACK,
        what + ": held resources grew from " + std::to_string(before) + " to "
            + std::to_string(after) + " over " + std::to_string(REPETITIONS) + " attempts");
}

// A channel whose second sync object could not be created must be reported as a failure, and
// must leave nothing behind -- not the sync object that did get created, and not the two
// shared-memory segments created before the validation runs.
void test_create_cleans_up_after_partial_failure()
{
    for (int failing_object = 1; failing_object <= 2; failing_object++)
    {
        nxai_ipc_test_fail_sync_obj = failing_object;

        // One attempt first, so the baseline is taken with whatever one-off allocations the
        // first call makes already done.
        check(
            nxai_ipc_create(1024) == IPC_SYNC_INVALID,
            "nxai_ipc_create() reported success although sync object "
                + std::to_string(failing_object) + " could not be created");

        const size_t before = held_resource_count();
        for (int i = 0; i < REPETITIONS; i++)
        {
            if (nxai_ipc_create(1024) != IPC_SYNC_INVALID)
            {
                check(false, "nxai_ipc_create() reported success under injected failure");
                break;
            }
        }
        check_no_accumulation(
            before,
            "failed nxai_ipc_create() with sync object " + std::to_string(failing_object)
                + " failing");

        nxai_ipc_test_fail_sync_obj = 0;
    }
}

// Every shape the parser guards against. The separator cases stop before any resource is
// acquired; the unresolvable-key cases stop after the parser has already started opening
// segments, which is where a cleanup omission would cost a handle per attempt.
std::vector<std::string> malformed_descriptors()
{
#ifdef _WIN32
    const std::string missing_key = "Global_RandomSHM_00000000";
#else
    const std::string missing_key = "/nxai_shm_00000000";
#endif
    return {
        "",
        "no-separators-at-all",
        "one|separator-only",
        "two|separators|only",
        "0|0|" + missing_key + "|" + missing_key + "_data",
    };
}

void test_from_str_rejects_malformed_descriptors()
{
    check(ipc_from_str(nullptr) == IPC_SYNC_INVALID, "ipc_from_str(NULL) returned a channel");

    for (const std::string& descriptor : malformed_descriptors())
    {
        check(
            ipc_from_str(descriptor.c_str()) == IPC_SYNC_INVALID,
            "ipc_from_str() returned a channel for the malformed descriptor \"" + descriptor
                + "\"");
    }
}

void test_from_str_does_not_accumulate()
{
    for (const std::string& descriptor : malformed_descriptors())
    {
        ipc_from_str(descriptor.c_str());
        const size_t before = held_resource_count();
        for (int i = 0; i < REPETITIONS; i++)
            ipc_from_str(descriptor.c_str());
        check_no_accumulation(before, "rejected descriptor \"" + descriptor + "\"");
    }
}

// The one branch a hand-written descriptor cannot reach: a descriptor whose signal segment
// resolves and whose data segment does not, so the parser fails with a segment already open.
void test_from_str_releases_a_half_opened_descriptor()
{
    nxai_ipc_t channel = nxai_ipc_create(1024);
    if (channel == IPC_SYNC_INVALID)
    {
        check(false, "could not create a real channel to derive a descriptor from");
        return;
    }

    char* descriptor = ipc_sync_to_str(channel);
    std::string half_broken(descriptor);
    free(descriptor);
    half_broken = half_broken.substr(0, half_broken.rfind('|') + 1) + "no_such_data_segment";

    ipc_from_str(half_broken.c_str());
    const size_t before = held_resource_count();
    bool accepted = false;
    for (int i = 0; i < REPETITIONS; i++)
        accepted = accepted || ipc_from_str(half_broken.c_str()) != IPC_SYNC_INVALID;
    check(
        !accepted,
        "ipc_from_str() returned a channel although its data segment does not exist");
    check_no_accumulation(before, "descriptor with a resolvable signal segment only");

    ipc_sync_close(channel);
}

} // namespace

int main()
{
    test_create_cleans_up_after_partial_failure();
    test_from_str_rejects_malformed_descriptors();
    test_from_str_does_not_accumulate();
    test_from_str_releases_a_half_opened_descriptor();

    if (g_failed)
        return 1;
    std::cout << "PASS\n";
    return 0;
}
