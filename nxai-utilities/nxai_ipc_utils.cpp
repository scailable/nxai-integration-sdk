#include <cstdint>
#include <random>
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <errno.h>
    #include <fcntl.h>
    #include <poll.h>
    #include <stdint.h>
    #include <sys/eventfd.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <unistd.h>
#endif

#include "nxai_ipc_utils.h"
#include "nxai_utils.h"

////////////////////// SHM /////////////////////////////////
#define POSIX_SHM_PREFIX "/nxai_shm_"
#define HEADER_SIZE sizeof(uint32_t)

#define SHM_MAX_SIZE 200 * 1024 * 1024 // 200 MB

// Defined below and not declared in the header, but called from nxai_shm_create_random() above
// its definition. C++ requires the declaration; in C the implicit declaration hid the omission.
bool nxai_shm_realloc(nxai_shm_t* shm, size_t new_size, size_t data_to_preserve);

static nxai_shm_t nxai_shm_create_random(size_t size)
{
    static std::mt19937 rng{std::random_device{}()};
    static std::uniform_int_distribution<unsigned int> dist;
#if defined(_MSC_VER)
    // Windows implementation
    nxai_shm_t new_shm;
    new_shm.id = NULL;
    new_shm.key = (char*) malloc(sizeof(char) * MAX_PATH);
    new_shm.attached = false;
    new_shm.pointer = NULL;
    new_shm.capacity = 0;

    SECURITY_ATTRIBUTES saAttr;
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL;

    for (size_t retry_counter = 0; retry_counter < 5; retry_counter++)
    {
        // Generate random name for the mapping. Note "Global_" is a literal part of the name,
        // not the "Global\" kernel namespace prefix: this object stays in the caller's session
        // namespace and needs no SeCreateGlobalPrivilege. Whether it should move to the global
        // namespace depends on which account starts a processor and is not decided here.
        sprintf_s(new_shm.key, MAX_PATH, "Global_RandomSHM_%08X", dist(rng));

        // Create file mapping object
        HANDLE hMapFile = CreateFileMappingA(
            INVALID_HANDLE_VALUE, // Use paging file
            &saAttr, // Default security attributes
            PAGE_READWRITE | SEC_RESERVE, // Read/write access
            0, // High DWORD of size
            SHM_MAX_SIZE, // Low DWORD of size
            new_shm.key // Name of mapping object
        );

        if (hMapFile == NULL)
        {
            char error_string[1024];
            DWORD error_length = get_windows_error(GetLastError(), error_string, 1024);
            nxai_vlog(
                "Warning: Could not create SHM \"%s\" (requested size %zu): %.*s\n",
                new_shm.key,
                size,
                error_length,
                error_string);
            break;
        }
        else if (GetLastError() == ERROR_ALREADY_EXISTS)
        {
            // SHM with key exists, regenerate key and try again
            nxai_vlog_verbose(
                "SHM with key %s already exists (handle %p). Trying again...\n",
                new_shm.key,
                hMapFile);
            continue;
        }
        else
        {
            // Creation succesful
            new_shm.id = hMapFile;
            nxai_vlog_verbose(
                "Created SHM \"%s\", handle %p, reserved size %d\n",
                new_shm.key,
                hMapFile,
                SHM_MAX_SIZE);
            break;
        }
    }

    // Commit memory
    nxai_shm_realloc(&new_shm, size, 0); // No data preservation for new SHM

    // Set capacity if SHM was successfully created
    if (new_shm.id != NULL)
    {
        new_shm.capacity = size;
    }

    return new_shm;
#else
    // Linux POSIX implementation
    nxai_shm_t new_shm;
    new_shm.id = -1;
    new_shm.key = NULL;
    new_shm.attached = false;
    new_shm.pointer = NULL;
    new_shm.capacity = 0;

    // Keep trying random names until unused is found
    for (int retry_counter = 0; retry_counter < 10; retry_counter++)
    {
        // Generate random name for POSIX shared memory
        unsigned int random_val = dist(rng);
        char shm_name[256];
        snprintf(shm_name, sizeof(shm_name), "%s%08X", POSIX_SHM_PREFIX, random_val);

        // Try to create shared memory with O_EXCL to ensure uniqueness
        int fd = shm_open(shm_name, O_CREAT | O_EXCL | O_RDWR, 0666);

        if (fd != -1)
        {
            // Successfully created
            // Set size using ftruncate
            if (ftruncate(fd, size) == -1)
            {
                // printf("Warning: Could not set SHM size: %s\n", strerror(errno));
                close(fd);
                shm_unlink(shm_name);
                continue;
            }

            new_shm.id = fd;
            new_shm.key = (char*) malloc(strlen(shm_name) + 1);
            strcpy(new_shm.key, shm_name);
            new_shm.capacity = size;
            break;
        }
        else if (errno == EEXIST)
        {
            // Name already exists, try again
            continue;
        }
        else
        {
            // printf("Warning: Could not create POSIX SHM: %s\n", strerror(errno));
            break;
        }
    }

    return new_shm;
#endif
}

void nxai_shm_attach(nxai_shm_t* shm)
{
    if (!shm)
    {
        return;
    }

#if defined(_MSC_VER)
    // Windows implementation
    LPVOID shm_pointer = MapViewOfFile(
        shm->id, // Handle to map object
        FILE_MAP_ALL_ACCESS, // Desired access
        0, // File offset (high DWORD)
        0, // File offset (low DWORD)
        0 // Number of bytes to map
    );
    if (shm_pointer == NULL)
    {
        char error_string[1024];
        DWORD error_length = get_windows_error(GetLastError(), error_string, 1024);
        nxai_vlog(
            "Warning: Could not attach shared memory (handle %p, was already attached: %d, "
            "capacity %zu): %.*s\n",
            shm->id,
            shm->attached,
            shm->capacity,
            error_length,
            error_string);
        return;
    }

    // Learn the segment's usable size from the OS, exactly as the Linux branch below takes it
    // from fstat(). The mapping spans the whole SHM_MAX_SIZE reservation, but only the
    // committed prefix may be touched, so the committed region -- the first one with uniform
    // state, which is what VirtualQuery reports here -- is the real capacity. Without this a
    // process that only ever attaches (every child: ipc_from_str() malloc()s the channel and
    // no other Windows path assigns capacity) keeps whatever the heap happened to hold, so it
    // skips the reallocation handshake it needed and writes past the committed pages.
    MEMORY_BASIC_INFORMATION region_info;
    if (VirtualQuery(shm_pointer, &region_info, sizeof(region_info)) == 0)
    {
        char error_string[1024];
        DWORD error_length = get_windows_error(GetLastError(), error_string, 1024);
        nxai_vlog(
            "Warning: Could not query shared memory size (handle %p): %.*s\n",
            shm->id,
            error_length,
            error_string);
        UnmapViewOfFile(shm_pointer);
        return;
    }

    nxai_vlog_verbose(
        "Attached shared memory (handle %p, capacity %zu) at %p\n",
        shm->id,
        (size_t) region_info.RegionSize,
        shm_pointer);

    shm->pointer = (unsigned char*) shm_pointer;
    shm->capacity = region_info.RegionSize;
    shm->attached = true;
#else
    // Linux POSIX implementation
    // Map the shared memory file descriptor to process address space using mmap
    // Get file size first
    struct stat sb;
    if (fstat(shm->id, &sb) == -1)
    {
        // printf("Warning: Could not get POSIX SHM size: %s\n", strerror(errno));
        return;
    }

    void* result = mmap(NULL, sb.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm->id, 0);
    if (result == MAP_FAILED)
    {
        // printf("Warning: Could not mmap shared memory: %s\n", strerror(errno));
        return;
    }

    shm->pointer = (unsigned char*) result;
    shm->capacity = sb.st_size;
    shm->attached = true;
#endif
}

char* nxai_shm_key_to_string(nxai_shm_t shm)
{
#if defined(_MSC_VER)
    // Windows implementation
    // Copy string so it can be freed
    char* shm_key_string = (char*) malloc(strlen(shm.key) + 1);
    strcpy(shm_key_string, shm.key);
#else
    // Linux POSIX implementation
    // Key is already a string name, just copy it
    char* shm_key_string = (char*) malloc(strlen(shm.key) + 1);
    strcpy(shm_key_string, shm.key);
#endif
    return shm_key_string;
}

void nxai_shm_key_from_string(nxai_shm_t* shm, const char* str)
{
#if defined(_MSC_VER)
    // Windows implementation
    shm->key = (char*) malloc(sizeof(char) * MAX_PATH);
    strcpy(shm->key, str);
#else
    // Linux POSIX implementation
    // Store string name for POSIX shared memory
    shm->key = (char*) malloc(strlen(str) + 1);
    strcpy(shm->key, str);
#endif
}

char* nxai_shm_id_to_string(nxai_shm_t shm)
{
#if defined(_MSC_VER)
    // Windows implementation
    char* id_string = nxai_pointer_to_string(shm.id);
#else
    // Linux POSIX implementation - fd as decimal string
    char* id_string = nxai_sprintf(32, "%d", shm.id);
#endif
    return id_string;
}

nxai_shm_t nxai_shm_id_from_string(const char* str)
{
    nxai_shm_t shm;

#if defined(_MSC_VER)
    // Windows implementation
    sscanf(str, "%p", &shm.id);
#else
    // Linux implementation
    shm.id = atoi(str);
#endif

    return shm;
}

bool nxai_shm_is_valid(const nxai_shm_t* shm)
{
#if defined(_MSC_VER)
    // Windows implementation
    if (shm->id == NULL)
    {
        return false;
    }
    else
    {
        return true;
    }
#else
    // Linux implementation
    if (shm->id == -1)
    {
        return false;
    }
    else
    {
        return true;
    }
#endif
}

bool nxai_shm_get_id(nxai_shm_t* shm)
{
#if defined(_MSC_VER)
    // Windows implementation
    HANDLE shm_id = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, shm->key);
    if (shm_id == NULL)
    {
        char error_string[1024];
        DWORD error_length = get_windows_error(GetLastError(), error_string, 1024);
        nxai_vlog("Warning: Could not get SHM ID: %.*s\n", error_length, error_string);
        return false;
    }
    else
    {
        shm->id = shm_id;
        return true;
    }
#else
    // Linux POSIX implementation - open existing shared memory
    int fd = shm_open(shm->key, O_RDWR, 0666);
    if (fd == -1)
    {
        // printf("Could not open POSIX SHM %s: %s\n", shm->key, strerror(errno));
        return false;
    }
    shm->id = fd;
    return true;
#endif
}

bool nxai_shm_is_attached(const nxai_shm_t* shm)
{
    if (!shm)
    {
        return false;
    }

    // Check the attached flag - this is now the source of truth
    if (!shm->attached)
    {
        return false;
    }
    return true;
}

void nxai_shm_detach(nxai_shm_t* shm)
{
    if (!shm || !shm->attached)
    {
        return;
    }

#if defined(_MSC_VER)
    // Windows implementation
    if (shm->pointer)
    {
        UnmapViewOfFile(shm->pointer);
    }
#else
    // Linux POSIX implementation
    if (shm->pointer && shm->capacity > 0)
    {
        munmap(shm->pointer, shm->capacity);
    }
#endif

    // Clear the structure
    shm->pointer = NULL;
    shm->capacity = 0;
    shm->attached = false;
}

int nxai_shm_destroy(const nxai_shm_t* shm)
{
#if defined(_MSC_VER)
    // Windows implementation
    int result = CloseHandle(shm->id);
    return result ? 0 : -1;
#else
    // Linux POSIX implementation
    // Close the file descriptor
    if (close(shm->id) == -1)
    {
        // printf("Warning: Could not close SHM fd: %s\n", strerror(errno));
        return -1;
    }

    // Unlink the shared memory object
    if (shm_unlink(shm->key) == -1)
    {
        // It's not an error if the object doesn't exist
        if (errno != ENOENT)
        {
            // printf("Warning: Could not unlink POSIX SHM: %s\n", strerror(errno));
            return -1;
        }
    }

    return 0;
#endif
}

// Put a segment back in the state a never-created one has. The failure paths below reach
// their cleanup with the segments in every state -- never created, created, attached -- and
// releasing one twice would close a handle number the process no longer owns.
static void nxai_shm_invalidate(nxai_shm_t* shm)
{
#if defined(_MSC_VER)
    shm->id = NULL;
#else
    shm->id = -1;
#endif
    shm->key = NULL;
    shm->attached = false;
    shm->pointer = NULL;
    shm->capacity = 0;
}

// Release a segment this process created, destroying the segment itself along with the
// mapping and the key.
static void nxai_shm_close_owned(nxai_shm_t* shm)
{
    nxai_shm_detach(shm);
    if (nxai_shm_is_valid(shm))
    {
        nxai_shm_destroy(shm);
    }
    free(shm->key);
    nxai_shm_invalidate(shm);
}

// Release a segment this process only attached to. The descriptor or handle it opened is
// this process's to close, but the segment belongs to whoever created it and must not be
// destroyed -- nxai_shm_destroy() would unlink it out from under that process.
static void nxai_shm_close_attached(nxai_shm_t* shm)
{
    nxai_shm_detach(shm);
    if (nxai_shm_is_valid(shm))
    {
#if defined(_MSC_VER)
        CloseHandle(shm->id);
#else
        close(shm->id);
#endif
    }
    free(shm->key);
    nxai_shm_invalidate(shm);
}

bool nxai_shm_realloc(nxai_shm_t* shm, size_t new_size, size_t data_to_preserve)
{
#if defined(_MSC_VER)
    // Windows implementation
    // Preserve existing data if requested
    char* preserved_data = NULL;
    if (data_to_preserve > 0)
    {
        nxai_shm_attach(shm);
        if (nxai_shm_is_attached(shm))
        {
            preserved_data = (char*) malloc(data_to_preserve);
            memcpy(preserved_data, (char*) shm->pointer, data_to_preserve);
            UnmapViewOfFile(shm->pointer);
            shm->attached = false;
            shm->pointer = NULL;
        }
    }

    // Commit memory to ensure space on disk
    nxai_shm_attach(shm);
    if (!nxai_shm_is_attached(shm))
    {
        if (preserved_data)
            free(preserved_data);
        return false;
    }
    if (VirtualAlloc(shm->pointer, new_size, MEM_COMMIT, PAGE_READWRITE) == NULL)
    {
        char error_string[1024];
        DWORD error_length = get_windows_error(GetLastError(), error_string, 1024);
        nxai_vlog(
            "Error: Could not realloc shared memory (handle %p, view %p, new_size %zu): %.*s\n",
            shm->id,
            shm->pointer,
            new_size,
            error_length,
            error_string);
        UnmapViewOfFile(shm->pointer);
        if (preserved_data)
            free(preserved_data);
        return false;
    }
    nxai_vlog_verbose("Committed %zu bytes at %p (handle %p)\n", new_size, shm->pointer, shm->id);

    // Restore preserved data to new segment
    if (preserved_data != NULL)
    {
        memcpy((char*) shm->pointer, preserved_data, data_to_preserve);
        free(preserved_data);
    }

    nxai_vlog_verbose("Detaching handle %p after realloc (view %p)\n", shm->id, shm->pointer);
    UnmapViewOfFile(shm->pointer);
    shm->attached = false;
    shm->pointer = NULL;
    // Unlike the Linux branch below, nothing else sets shm->capacity after a runtime resize
    // (nxai_shm_create_random() sets it separately, but only for a brand-new SHM) - without this,
    // the channel keeps thinking it's still at the old, too-small size and reruns the whole
    // REALLOC handshake on every subsequent send, not just the first oversized one.
    shm->capacity = new_size;
    return true;
#else
    // Linux POSIX implementation
    // Preserve existing data before resizing
    char* preserved_data = NULL;
    if (data_to_preserve > 0)
    {
        nxai_shm_attach(shm);
        if (nxai_shm_is_attached(shm))
        {
            preserved_data = (char*) malloc(data_to_preserve);
            if (preserved_data != NULL)
            {
                memcpy(preserved_data, (char*) shm->pointer, data_to_preserve);
            }
        }
    }

    // Leave the segment detached, the same contract the Windows branch above returns under.
    // ftruncate() below grows the object, but an existing mmap() keeps the length it was made
    // with, so a caller that went on using the old mapping would write past its end. Detaching
    // forces a fresh mmap() at the new size.
    nxai_shm_detach(shm);

    // Resize existing SHM using ftruncate
    if (ftruncate(shm->id, new_size) == -1)
    {
        nxai_error_log("Error! Could not truncate POSIX SHM: %s\n", strerror(errno));
        if (preserved_data)
            free(preserved_data);
        return false;
    }

    // Remap at the new size, both to confirm the resize actually took and to restore any
    // preserved data.
    nxai_shm_attach(shm);
    if (!nxai_shm_is_attached(shm))
    {
        if (preserved_data)
            free(preserved_data);
        return false;
    }

    if (preserved_data != NULL)
    {
        memcpy((char*) shm->pointer, preserved_data, data_to_preserve);
        free(preserved_data);
    }

    nxai_shm_detach(shm);
    // Set this explicitly rather than relying on the next nxai_shm_attach() to learn it from
    // fstat(): a caller that checks shm->capacity before its next attach (nxai_ipc_send() does)
    // must see the post-resize value immediately, matching the Windows branch's guarantee.
    shm->capacity = new_size;
    return true;
#endif
}

////////////////////////////////// IPC ////////////////////////////////////

#ifdef _WIN32
bool nxai_ipc_sync_obj_is_valid(HANDLE sync_obj)
{
    // CreateEventA reports failure by returning NULL, not INVALID_HANDLE_VALUE -- that is the
    // CreateFile convention and a different value, so guarding against it let a failed event
    // creation through and then handed the null handle to CloseHandle. Both are
    // rejected: neither names an object.
    return sync_obj != NULL && sync_obj != INVALID_HANDLE_VALUE;
}
#else
bool nxai_ipc_sync_obj_is_valid(int sync_obj)
{
    return sync_obj != -1;
}
#endif

int nxai_ipc_test_fail_sync_obj = 0;

nxai_ipc_t nxai_ipc_create(size_t initial_size)
{
    nxai_ipc_t new_ipc_channel = (nxai_ipc_t) malloc(sizeof(ipc_sync_internal_t));
    if (!new_ipc_channel)
    {
        return IPC_SYNC_INVALID;
    }

    // Since this process creates the IPC, it is upstream, meaning it will always send downstream
    new_ipc_channel->role = IPC_SYNC_HOST;

    new_ipc_channel->is_ready = false;
    new_ipc_channel->notice_outstanding = false;

    // Create separate Sync Primitives for each direction
#ifdef _WIN32
    // The event handles are passed to the child process on the command line, so they must be
    // marked inheritable at creation time - bInheritHandles=TRUE on CreateProcess cannot make an
    // already non-inheritable handle inheritable after the fact. Omitting this leaves the
    // child with numbers that name nothing in its own handle table and it never manages to signal
    // READY.
    SECURITY_ATTRIBUTES event_security_attributes;
    event_security_attributes.nLength = sizeof(SECURITY_ATTRIBUTES);
    event_security_attributes.bInheritHandle = TRUE;
    event_security_attributes.lpSecurityDescriptor = NULL;

    new_ipc_channel->sync_obj_downstream = (nxai_ipc_test_fail_sync_obj == 1)
        ? NULL
        : CreateEventA(&event_security_attributes, FALSE, FALSE, NULL);
    new_ipc_channel->sync_obj_upstream = (nxai_ipc_test_fail_sync_obj == 2)
        ? NULL
        : CreateEventA(&event_security_attributes, FALSE, FALSE, NULL);
#else
    // Note: Not using EFD_CLOEXEC because we need these FDs to be inherited by child processes
    // The FD numbers are explicitly passed to child processes via command-line arguments
    new_ipc_channel->sync_obj_downstream =
        (nxai_ipc_test_fail_sync_obj == 1) ? -1 : eventfd(0, EFD_SEMAPHORE);
    new_ipc_channel->sync_obj_upstream =
        (nxai_ipc_test_fail_sync_obj == 2) ? -1 : eventfd(0, EFD_SEMAPHORE);
#endif

    // Create and attach 2 bytes of Shared Memory (one for each direction)
    new_ipc_channel->signal_shm = nxai_shm_create_random(2);
    nxai_shm_attach(&new_ipc_channel->signal_shm);

    if (nxai_shm_is_attached(&new_ipc_channel->signal_shm))
    {
        new_ipc_channel->data_ptr_downstream = (int8_t*) &new_ipc_channel->signal_shm.pointer[0];
        new_ipc_channel->data_ptr_upstream = (int8_t*) &new_ipc_channel->signal_shm.pointer[1];

        // Initialize both bytes to 0
        new_ipc_channel->signal_shm.pointer[0] = 0;
        new_ipc_channel->signal_shm.pointer[1] = 0;
    }
    else
    {
        new_ipc_channel->data_ptr_downstream = NULL;
        new_ipc_channel->data_ptr_upstream = NULL;
    }

    // Create and attach the data SHM ( Add space for header )
    new_ipc_channel->data_shm = nxai_shm_create_random(HEADER_SIZE + initial_size);
    nxai_shm_attach(&new_ipc_channel->data_shm);

    // Validate creation
    if (!nxai_ipc_sync_obj_is_valid(new_ipc_channel->sync_obj_downstream)
        || !nxai_ipc_sync_obj_is_valid(new_ipc_channel->sync_obj_upstream)
        || !nxai_shm_is_attached(&new_ipc_channel->signal_shm)
        || !nxai_shm_is_attached(&new_ipc_channel->data_shm))
    {
        // Cleanup on failure
#ifdef _WIN32
        if (nxai_ipc_sync_obj_is_valid(new_ipc_channel->sync_obj_downstream))
            CloseHandle(new_ipc_channel->sync_obj_downstream);
        if (nxai_ipc_sync_obj_is_valid(new_ipc_channel->sync_obj_upstream))
            CloseHandle(new_ipc_channel->sync_obj_upstream);
#else
        if (nxai_ipc_sync_obj_is_valid(new_ipc_channel->sync_obj_downstream))
            close(new_ipc_channel->sync_obj_downstream);
        if (nxai_ipc_sync_obj_is_valid(new_ipc_channel->sync_obj_upstream))
            close(new_ipc_channel->sync_obj_upstream);
#endif
        // Both segments were created and attached above, before this validation runs, so
        // whichever of them got that far leaks unless it is released here.
        nxai_shm_close_owned(&new_ipc_channel->signal_shm);
        nxai_shm_close_owned(&new_ipc_channel->data_shm);
        free(new_ipc_channel);
        return IPC_SYNC_INVALID;
    }

    return new_ipc_channel;
}

ipc_error_t ipc_sync_signal(nxai_ipc_t sync, ipc_signal_t val)
{
    if (!sync)
        return IPC_INVALID_SYNC;

    int8_t* data_ptr;
#ifdef _WIN32
    HANDLE sync_obj;
#else
    int sync_obj;
#endif

    // Select the correct handle and data pointer for this direction
    if (sync->role == IPC_SYNC_CLIENT)
    {
        // The client always sends signals upstream
        data_ptr = sync->data_ptr_upstream;
        sync_obj = sync->sync_obj_upstream;
    }
    else
    {
        // The host always sends signals downstream
        data_ptr = sync->data_ptr_downstream;
        sync_obj = sync->sync_obj_downstream;
    }

    if (!data_ptr)
        return IPC_INVALID_SYNC;

    // Write the byte to shared memory
    *data_ptr = (int8_t) val;

    // Signal the primitive
#ifdef _WIN32
    int error = SetEvent(sync_obj) ? 0 : -1;
    if (error == 0)
    {
        return IPC_ERROR_NONE;
    }
    return IPC_SIGNAL_ERROR;
#else
    uint64_t signal_val = 1;
    int error = (write(sync_obj, &signal_val, sizeof(signal_val)) == sizeof(signal_val)) ? 0 : -1;
    if (error == 0)
    {
        return IPC_ERROR_NONE;
    }
    return IPC_SIGNAL_ERROR;

#endif
}

ipc_error_t ipc_sync_wait(nxai_ipc_t sync, ipc_signal_t* signal, int timeout_ms)
{
    if (!sync)
        return IPC_INVALID_SYNC;

#ifdef _WIN32
    HANDLE sync_obj;
#else
    int sync_obj;
#endif

    int8_t* data_ptr;

    // Select the correct handle and data pointer for this direction
    if (sync->role == IPC_SYNC_HOST)
    {
        // Host always waits for signals coming upstream
        data_ptr = sync->data_ptr_upstream;
        sync_obj = sync->sync_obj_upstream;
    }
    else
    {
        // Client always waits for signals coming downstream
        data_ptr = sync->data_ptr_downstream;
        sync_obj = sync->sync_obj_downstream;
    }

    if (!data_ptr)
        return IPC_INVALID_SYNC;

#ifdef _WIN32
    DWORD dwTimeout = (timeout_ms < 0) ? INFINITE : (DWORD) timeout_ms;
    if (WaitForSingleObject(sync_obj, dwTimeout) == WAIT_OBJECT_0)
    {
        *signal = (ipc_signal_t) *data_ptr;
        return IPC_ERROR_NONE;
    }
#else
    struct pollfd pfd = {.fd = sync_obj, .events = POLLIN};
    if (poll(&pfd, 1, timeout_ms) > 0)
    {
        uint64_t val;
        read(sync_obj, &val, sizeof(val)); // Consume eventfd
        *signal = (ipc_signal_t) *data_ptr;
        return IPC_ERROR_NONE;
    }
#endif
    return IPC_ERROR_TIMEOUT;
}

char* ipc_sync_to_str(nxai_ipc_t sync)
{
    if (!sync)
        return NULL;

    char* signal_shm_key_str = nxai_shm_key_to_string(sync->signal_shm);
    char* data_shm_key_str = nxai_shm_key_to_string(sync->data_shm);

    // Format each field on its own, single-argument call instead of one multi-argument
    // "%p|%p|%s|%s" vsnprintf(), so a variadic argument-type mismatch can no longer silently
    // corrupt the string the child process reads from its command line.
#ifdef _WIN32
    char* downstream_str = nxai_pointer_to_string(sync->sync_obj_downstream);
    char* upstream_str = nxai_pointer_to_string(sync->sync_obj_upstream);
#else
    char* downstream_str = nxai_sprintf(32, "%d", sync->sync_obj_downstream);
    char* upstream_str = nxai_sprintf(32, "%d", sync->sync_obj_upstream);
#endif

    std::string combined = std::string(downstream_str) + "|" + upstream_str + "|"
        + signal_shm_key_str + "|" + data_shm_key_str;

    char* out_str = (char*) malloc(combined.size() + 1);
    if (out_str != nullptr)
        memcpy(out_str, combined.c_str(), combined.size() + 1);

    free(downstream_str);
    free(upstream_str);
    free(signal_shm_key_str);
    free(data_shm_key_str);
    return out_str;
}

nxai_ipc_t ipc_from_str(const char* str)
{
    if (!str)
        return IPC_SYNC_INVALID;

    nxai_ipc_t new_ipc_channel = (nxai_ipc_t) malloc(sizeof(ipc_sync_internal_t));
    if (!new_ipc_channel)
        return IPC_SYNC_INVALID;

    // Since this process is loading the IPC channel, it is downstream, meaning it will always send
    // upstream
    new_ipc_channel->role = IPC_SYNC_HOST;
    new_ipc_channel->is_ready = true;
    new_ipc_channel->notice_outstanding = false;

    // malloc() leaves both segments holding whatever the heap did. The parsing failures below
    // are reached with neither segment opened, so they would otherwise pass an arbitrary
    // handle number to nxai_shm_attach(), and the cleanup could not tell what to release.
    nxai_shm_invalidate(&new_ipc_channel->signal_shm);
    nxai_shm_invalidate(&new_ipc_channel->data_shm);

    char* input = strdup(str);
    char* sep1 = strchr(input, '|');
    if (!sep1)
    {
        free(input);
        free(new_ipc_channel);
        return IPC_SYNC_INVALID;
    }
    *sep1 = '\0';

    char* sep2 = strchr(sep1 + 1, '|');
    if (!sep2)
    {
        free(input);
        free(new_ipc_channel);
        return IPC_SYNC_INVALID;
    }
    *sep2 = '\0';
    char* signal_shm_part = sep2 + 1;

    char* sep3 = strchr(sep2 + 1, '|');
    if (!sep3)
    {
        free(input);
        free(new_ipc_channel);
        return IPC_SYNC_INVALID;
    }
    *sep3 = '\0';
    char* data_shm_part = sep3 + 1;

#ifdef _WIN32
    new_ipc_channel->sync_obj_downstream = (HANDLE) strtoll(input, NULL, 16);
    new_ipc_channel->sync_obj_upstream = (HANDLE) strtoll(sep1 + 1, NULL, 16);
#else
    new_ipc_channel->sync_obj_downstream = atoi(input);
    new_ipc_channel->sync_obj_upstream = atoi(sep1 + 1);
#endif

    nxai_shm_key_from_string(&new_ipc_channel->signal_shm, signal_shm_part);
    nxai_shm_get_id(&new_ipc_channel->signal_shm);
    nxai_shm_attach(&new_ipc_channel->signal_shm);

    if (!nxai_shm_is_attached(&new_ipc_channel->signal_shm))
    {
        nxai_shm_close_attached(&new_ipc_channel->signal_shm);
        free(input);
        free(new_ipc_channel);
        return IPC_SYNC_INVALID;
    }

    nxai_shm_key_from_string(&new_ipc_channel->data_shm, data_shm_part);
    nxai_shm_get_id(&new_ipc_channel->data_shm);
    nxai_shm_attach(&new_ipc_channel->data_shm);

    if (!nxai_shm_is_attached(&new_ipc_channel->data_shm))
    {
        nxai_shm_close_attached(&new_ipc_channel->signal_shm);
        nxai_shm_close_attached(&new_ipc_channel->data_shm);
        free(input);
        free(new_ipc_channel);
        return IPC_SYNC_INVALID;
    }

    new_ipc_channel->data_ptr_downstream = (int8_t*) &new_ipc_channel->signal_shm.pointer[0];
    new_ipc_channel->data_ptr_upstream = (int8_t*) &new_ipc_channel->signal_shm.pointer[1];

    free(input);
    return new_ipc_channel;
}

void ipc_sync_close(nxai_ipc_t sync)
{
    if (!sync)
        return;

    // Close the SHM attachment (tracks pointer and capacity)
    nxai_shm_detach(&sync->signal_shm);
    nxai_shm_destroy(&sync->signal_shm);
    nxai_shm_detach(&sync->data_shm);
    nxai_shm_destroy(&sync->data_shm);

#ifdef _WIN32
    if (nxai_ipc_sync_obj_is_valid(sync->sync_obj_downstream))
        CloseHandle(sync->sync_obj_downstream);
    if (nxai_ipc_sync_obj_is_valid(sync->sync_obj_upstream))
        CloseHandle(sync->sync_obj_upstream);
#else
    if (nxai_ipc_sync_obj_is_valid(sync->sync_obj_downstream))
        close(sync->sync_obj_downstream);
    if (nxai_ipc_sync_obj_is_valid(sync->sync_obj_upstream))
        close(sync->sync_obj_upstream);
#endif
    free(sync);
}

////////////////////////////////// API ////////////////////////////////////
int ipc_timeout_ms = 1000;

// Discards any signal already pending on this channel, without blocking. A handshake this
// process gave up on after a timeout does not stop existing: whatever it wrote (data + a
// signal) or the peer's late reply to it can still show up here well after the exchange it
// belonged to was abandoned. Call this immediately before starting a *new* handshake that this
// process initiates, i.e. before a send -- nothing for that new exchange has been sent yet at
// that point, so anything already sitting on the channel cannot belong to it and is safe to
// discard. Without this, a fresh nxai_ipc_send() call can consume a stale leftover instead of
// the ACK it actually asked for, permanently shifting which reply belongs to which request from
// then on.
//
// Never call this before a receive: there the peer initiates the exchange, and a signal already
// pending is most likely the NOTICE or EXIT it legitimately posted before this process got
// around to waiting.
//
// Returns whether one of the discarded signals was an ACK, so nxai_ipc_send() can tell a late
// acknowledgement of a NOTICE it gave up waiting on apart from unrelated leftovers, and only
// then treat data_shm as free to overwrite.
static bool ipc_sync_drain_stale(nxai_ipc_t ipc_channel)
{
    bool saw_ack = false;
    ipc_signal_t stale_signal;
    while (ipc_sync_wait(ipc_channel, &stale_signal, 0) == IPC_ERROR_NONE)
    {
        if (stale_signal == IPC_SIG_ACK)
            saw_ack = true;
        nxai_vlog(
            "Warning! Discarded stale signal %d left over from an abandoned exchange.\n",
            (int) stale_signal);
    }
    return saw_ack;
}

ipc_error_t nxai_ipc_send(nxai_ipc_t ipc_channel, const unsigned char* data, uint32_t data_size)
{
    if (ipc_channel->is_ready == false)
    {
        // Not ready yet
        return IPC_ERROR_NOT_READY;
    }
    const bool late_ack_arrived = ipc_sync_drain_stale(ipc_channel);
    if (ipc_channel->notice_outstanding && !late_ack_arrived)
    {
        // The previous NOTICE this channel sent is still unacknowledged, so the processor may
        // still be reading the payload currently sitting in data_shm. Overwriting it here would
        // silently lose that unread frame. Leave the buffer untouched and let the
        // caller retry once the processor catches up.
        return IPC_ERROR_BUSY;
    }
    ipc_channel->notice_outstanding = false;

    // Check if data SHM large enough for message
    nxai_vlog_verbose("Sending message with size: %u\n", data_size);
    if (ipc_channel->data_shm.capacity < data_size + HEADER_SIZE)
    {
        nxai_vlog_verbose(
            "Reallocating buffer from %zu to %zu\n",
            ipc_channel->data_shm.capacity,
            data_size + HEADER_SIZE);
        // Reallocate data SHM to accommodate message
        nxai_shm_realloc(&ipc_channel->data_shm, data_size + HEADER_SIZE, 0);
        // nxai_shm_realloc() returns with the segment detached on both platforms. Re-attach so
        // .pointer is valid, and at the new size, before it is written to below -- guarded,
        // since re-attaching an already-attached segment would map it a second time without
        // releasing the first.
        if (!nxai_shm_is_attached(&ipc_channel->data_shm))
        {
            nxai_shm_attach(&ipc_channel->data_shm);
            if (!nxai_shm_is_attached(&ipc_channel->data_shm))
            {
                nxai_vlog("Warning! Could not attach to reallocated data SHM!\n");
                return IPC_INVALID_SYNC;
            }
        }
        // Notify other process of reallocation
        ipc_error_t error = ipc_sync_signal(ipc_channel, IPC_SIG_REALLOC);
        if (error != IPC_ERROR_NONE)
        {
            nxai_vlog("Warning! Could not send reallocation signal to external process!\n");
            return error;
        }
        // Wait for other process to acknowledge
        nxai_vlog_verbose("Sent realloc signal, waiting for ack\n");
        ipc_signal_t response;
        error = ipc_sync_wait(ipc_channel, &response, ipc_timeout_ms);
        nxai_vlog_verbose("Got response: %d %d\n", error, response);
        if (error != IPC_ERROR_NONE)
        {
            nxai_vlog("Warning! Could not receive reallocation signal from external process!\n");
            return error;
        }
        if (response != IPC_SIG_ACK)
        {
            nxai_vlog("Warning! External process did not acknowledge reallocation!\n");
            return IPC_ERROR_REALLOC_NO_ACK;
        }
    }

    // Write header
    memcpy(ipc_channel->data_shm.pointer, &data_size, HEADER_SIZE);
    // Write data
    memcpy(((char*) ipc_channel->data_shm.pointer) + HEADER_SIZE, data, data_size);

    // Notify other process of input message
    ipc_sync_signal(ipc_channel, IPC_SIG_NOTICE);
    // From here until the ACK below is observed, the processor may be reading this payload out
    // of data_shm, so the next send must not overwrite it.
    ipc_channel->notice_outstanding = true;
    // Wait for acknowledge/read
    ipc_signal_t response;
    ipc_error_t error = ipc_sync_wait(ipc_channel, &response, ipc_timeout_ms);
    if (error != IPC_ERROR_NONE)
    {
        // Distinct wording from the realloc-ack wait above: same log text for both timeouts
        // made it impossible to tell which handshake actually stalled from the log alone.
        // notice_outstanding is left set: the ACK may still arrive later, and until it does the
        // payload just written must be treated as still in use by the processor.
        nxai_vlog(
            "Warning! Could not receive data-notice acknowledgement from external "
            "process!\n");
        return error;
    }
    if (response != IPC_SIG_ACK)
    {
        nxai_vlog("Warning! External process did not acknowledge data notice!\n");
        return IPC_ERROR_REALLOC_NO_ACK;
    }
    ipc_channel->notice_outstanding = false;
    return IPC_ERROR_NONE;
}

ipc_error_t nxai_ipc_receive(
    nxai_ipc_t ipc_channel,
    unsigned char** data,
    uint32_t* data_size,
    int timeout_ms)
{
    // Wait for signal
    ipc_signal_t signal;
    ipc_error_t error = ipc_sync_wait(ipc_channel, &signal, timeout_ms);
    if (error != IPC_ERROR_NONE)
    {
        return error;
    }
    while (signal == IPC_SIG_READY)
    {
        // A stray READY signal on this channel (e.g. an external process announcing
        // itself on both the input and output channels) carries no payload. Consume it
        // and keep waiting instead of failing the exchange.
        error = ipc_sync_wait(ipc_channel, &signal, timeout_ms);
        if (error != IPC_ERROR_NONE)
        {
            return error;
        }
    }
    if (signal == IPC_SIG_REALLOC)
    {
        // Other process sent realloc signal. Reattach SHM and wait again
        nxai_shm_detach(&(ipc_channel->data_shm));
        nxai_shm_get_id(&(ipc_channel->data_shm)); // TODO
        nxai_shm_attach(&(ipc_channel->data_shm));
        // Acknowledge reallocation
        ipc_sync_signal(ipc_channel, IPC_SIG_ACK);
        // Wait for follow up signal
        error = ipc_sync_wait(ipc_channel, &signal, timeout_ms);
        if (error != IPC_ERROR_NONE)
        {
            return error;
        }
    }
    // A shutdown request is not a protocol violation, so it gets an outcome of its own -- and
    // this one site covers it arriving directly as well as arriving as the follow-up to a
    // reallocation, since the branch above reassigns `signal` and falls through to here.
    if (signal == IPC_SIG_EXIT)
    {
        return IPC_EXIT_REQUESTED;
    }
    if (signal != IPC_SIG_NOTICE)
    {
        return IPC_UNEXPECTED_SIGNAL;
    }

    // The first 4 bytes of the shared memory is always the size of the data
    memcpy(data_size, ipc_channel->data_shm.pointer, HEADER_SIZE);
    // Return pointer to the payload data after the size header
    *data = ipc_channel->data_shm.pointer + HEADER_SIZE;

    return IPC_ERROR_NONE;
}

ipc_error_t release_buffer(nxai_ipc_t ipc_channel)
{
    return ipc_sync_signal(ipc_channel, IPC_SIG_ACK);
}

ipc_error_t nxai_ipc_wait_ready(nxai_ipc_t ipc_channel, int timeout_ms)
{
    ipc_signal_t signal;
    ipc_error_t error = ipc_sync_wait(ipc_channel, &signal, timeout_ms);
    if (error != IPC_ERROR_NONE)
    {
        return error;
    }
    if (signal != IPC_SIG_READY)
    {
        return IPC_UNEXPECTED_SIGNAL;
    }
    ipc_channel->is_ready = true;
    return IPC_ERROR_NONE;
}

ipc_error_t nxai_ipc_send_ready(nxai_ipc_t ipc_channel)
{
    return ipc_sync_signal(ipc_channel, IPC_SIG_READY);
}

ipc_error_t nxai_ipc_send_kill(nxai_ipc_t ipc_channel)
{
    return ipc_sync_signal(ipc_channel, IPC_SIG_EXIT);
}
