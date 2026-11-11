#ifndef IPC_SYNC_H
#define IPC_SYNC_H

#include <stdbool.h>
#include <stdint.h>

// Platform imports
#if defined(_MSC_VER)
    #include <basetsd.h>

    #include "windows.h"
typedef SSIZE_T ssize_t;
#else
    #include <sys/mman.h>
    #include <sys/types.h>
#endif

// Platform types
#if defined(_MSC_VER)
typedef HANDLE shm_id_t;
typedef LPSTR shm_key_t;
#else
// Linux: Use string-based POSIX shared memory names instead of SysV integer keys
typedef int shm_id_t;
typedef char* shm_key_t;
#endif
typedef struct
{
    shm_id_t id;
    shm_key_t key;
    bool attached;
    unsigned char* pointer;
    size_t capacity;
} nxai_shm_t;

/**
 * Bidirectional IPC sync direction.
 * UPSTREAM: From child/inference engine to parent/AI manager
 * DOWNSTREAM: From parent/AI manager to child/inference engine
 */
typedef enum
{
    IPC_SYNC_CLIENT = 0,
    IPC_SYNC_HOST = 1
} ipc_sync_role_t;

// Define the internal handle structure with separate handles for each direction
typedef struct
{
#ifdef _WIN32
    HANDLE sync_obj_downstream;
    HANDLE sync_obj_upstream;
#else
    int sync_obj_downstream;
    int sync_obj_upstream;
#endif
    nxai_shm_t signal_shm;
    nxai_shm_t data_shm;
    int8_t* data_ptr_downstream; // Cached pointer to downstream data byte
    int8_t* data_ptr_upstream; // Cached pointer to upstream data byte
    int8_t* data; // Cached pointer to upstream data byte
    ipc_sync_role_t role;
    bool is_ready;
    // Set while a NOTICE sent by nxai_ipc_send() has not yet been ACK'd. The peer may still be
    // reading data_shm for as long as this is true, so a send that finds it still set must not
    // overwrite the buffer.
    bool notice_outstanding;
} ipc_sync_internal_t;
typedef ipc_sync_internal_t* nxai_ipc_t;
#define IPC_SYNC_INVALID NULL

typedef enum
{
    IPC_ERROR_NONE,
    IPC_ERROR_TIMEOUT,
    IPC_INVALID_SYNC,
    IPC_SIGNAL_ERROR,
    IPC_ERROR_NOT_READY,
    IPC_ERROR_REALLOC_NO_ACK,
    IPC_UNEXPECTED_SIGNAL,
    /**
     * The other side asked for a graceful shutdown. Distinct from IPC_UNEXPECTED_SIGNAL so a
     * receive loop can leave on request and report an error only for a real protocol
     * violation; the Python SDK draws the same line with its exit_requested flag.
     */
    IPC_EXIT_REQUESTED,
    /**
     * The previous NOTICE on this channel is still unacknowledged, so data_shm may still be in
     * use by the peer. Nothing was sent or touched; the caller should retry later.
     */
    IPC_ERROR_BUSY
} ipc_error_t;

/**
 * The one-byte signal exchanged over the sync objects.
 *
 * Prefixed enumerators: the unprefixed ERROR collides with wingdi.h's `#define ERROR 0`,
 * which arrives through windows.h, so MSVC sees `0 = 3`. Prefixing keeps the collision from
 * recurring the next time an enumerator is added, which an `#undef ERROR` would not.
 */
typedef enum
{
    IPC_SIG_ACK = 0,
    IPC_SIG_NOTICE = 1,
    IPC_SIG_REALLOC = 2,
    IPC_SIG_ERROR = 3,
    IPC_SIG_READY = 4,
    IPC_SIG_EXIT = 5
} ipc_signal_t;

/**
 * Whether a sync object names something usable, i.e. whether its creation succeeded.
 * The failure value is platform-specific and easy to get wrong -- on Windows it is the one
 * CreateEventA returns, which is not the sentinel the file APIs use -- so the comparison
 * lives here rather than being repeated at each creation and close site.
 */
#ifdef _WIN32
bool nxai_ipc_sync_obj_is_valid(HANDLE sync_obj);
#else
bool nxai_ipc_sync_obj_is_valid(int sync_obj);
#endif

/**
 * Test-only failure injection for nxai_ipc_create()'s partial-creation cleanup branch.
 * Zero -- the value it holds in every shipped processor -- creates both sync objects normally;
 * 1 or 2 makes that sync object take the platform's creation-failure value instead, so the
 * cleanup branch runs. The branch is otherwise reachable only under handle or descriptor
 * exhaustion, which cannot be provoked on Windows at all.
 */
extern int nxai_ipc_test_fail_sync_obj;

nxai_ipc_t nxai_ipc_create(size_t initial_size);

char* ipc_sync_to_str(nxai_ipc_t sync);

nxai_ipc_t ipc_from_str(const char* str);

ipc_error_t nxai_ipc_wait_ready(nxai_ipc_t ipc_channel, int timeout_ms);

ipc_error_t nxai_ipc_send(nxai_ipc_t ipc_channel, const unsigned char* data, uint32_t data_size);

ipc_error_t nxai_ipc_receive(
    nxai_ipc_t ipc_channel,
    unsigned char** data,
    uint32_t* data_size,
    int timeout_ms);

ipc_error_t release_buffer(nxai_ipc_t ipc_channel);

ipc_error_t nxai_ipc_send_ready(nxai_ipc_t ipc_channel);

ipc_error_t nxai_ipc_send_kill(nxai_ipc_t ipc_channel);

/**
 * Raw signalling, below the framed nxai_ipc_send()/nxai_ipc_receive() pair. Exported so a test
 * can drive signal sequences the framed API cannot produce -- nxai_ipc_send() always follows a
 * reallocation with NOTICE, so an EXIT arriving as the follow-up signal is otherwise
 * unreachable.
 */
ipc_error_t ipc_sync_signal(nxai_ipc_t sync, ipc_signal_t val);

ipc_error_t ipc_sync_wait(nxai_ipc_t sync, ipc_signal_t* signal, int timeout_ms);

void ipc_sync_close(nxai_ipc_t sync);

#endif
