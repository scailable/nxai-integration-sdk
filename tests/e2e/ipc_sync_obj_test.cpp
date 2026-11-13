/*
 * Unit tests for the IPC sync-object validity predicate.
 *
 * The predicate is what decides whether nxai_ipc_create() succeeded and whether
 * ipc_sync_close() may close a handle. Its Windows branch used to compare against
 * INVALID_HANDLE_VALUE, which is the CreateFile convention -- CreateEventA reports failure
 * by returning NULL, so a failed event creation passed validation and was later handed to
 * CloseHandle. The failure value is the whole contract here, hence a test of its own.
 */

#include "nxai_ipc_utils.h"

#include <iostream>
#include <string>

#ifndef _WIN32
    #include <sys/eventfd.h>
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

// The value the platform's creation call actually returns on failure must be rejected.
void test_rejects_the_creation_failure_value()
{
#ifdef _WIN32
    check(!nxai_ipc_sync_obj_is_valid(NULL),
        "NULL accepted as a valid sync object -- CreateEventA reports failure this way, so "
        "nxai_ipc_create() would return a channel carrying a null event handle");
#else
    check(!nxai_ipc_sync_obj_is_valid(-1), "-1 accepted as a valid sync object");
#endif
}

// A successfully created object must be accepted, or every channel would be reported broken.
void test_accepts_a_real_sync_object()
{
#ifdef _WIN32
    HANDLE event = CreateEventA(NULL, FALSE, FALSE, NULL);
    if (event == NULL)
    {
        check(false, "could not create an event to test with");
        return;
    }
    check(nxai_ipc_sync_obj_is_valid(event), "a live event handle was rejected");
    CloseHandle(event);
#else
    int fd = eventfd(0, EFD_SEMAPHORE);
    if (fd == -1)
    {
        check(false, "could not create an eventfd to test with");
        return;
    }
    check(nxai_ipc_sync_obj_is_valid(fd), "a live eventfd was rejected");
    close(fd);
#endif
}

#ifdef _WIN32
// INVALID_HANDLE_VALUE is not a value CreateEventA can return, but it is the value the old
// code guarded against and it names no object either way -- keep rejecting it so the fix
// cannot be read as loosening the check.
void test_also_rejects_the_file_api_sentinel()
{
    check(!nxai_ipc_sync_obj_is_valid(INVALID_HANDLE_VALUE),
        "INVALID_HANDLE_VALUE accepted as a valid sync object");
}
#endif

} // namespace

int main()
{
    test_rejects_the_creation_failure_value();
    test_accepts_a_real_sync_object();
#ifdef _WIN32
    test_also_rejects_the_file_api_sentinel();
#endif

    if (g_failed)
        return 1;
    std::cout << "PASS\n";
    return 0;
}
