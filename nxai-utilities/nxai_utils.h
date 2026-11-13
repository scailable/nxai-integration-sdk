#pragma once

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

#include <stdbool.h>
#include <stdint.h>

// Platform imports
#if defined(_MSC_VER)
    #define NOMINMAX //< Needed to prevent windows.h define macros min() and max().
    #include <basetsd.h>

    #include "windows.h"
#else
    #include <sys/types.h>
#endif

using nxai_smart_ptr = std::unique_ptr<char, decltype(&free)>;

struct nxai_buffer_t
{
    nxai_smart_ptr data{nullptr, free};
    size_t size{0};

    nxai_buffer_t(char* ptr, size_t size): data{ptr, free}, size{size} {}
};

enum class ImageOrder
{
    CHW,
    HWC
};

char* nxai_sprintf(size_t initial_size, const char* fmt, ...);

char* nxai_pointer_to_string(void* pointer);

void nxai_initialize_logging(
    const char* start_log_filepath,
    const char* rotating_log_filepath,
    const char* log_prefix,
    bool log_to_console,
    bool log_to_file,
    int log_verbosity_level);

void nxai_finalise_logging();

void nxai_vlog_profiling(const char* fmt, ...);

void nxai_vlog_verbose(const char* fmt, ...);

void nxai_error_log(const char* fmt, ...);

void nxai_vlog(const char* fmt, ...);

bool nxai_get_file_size(const char* filepath, size_t* file_size);

void nxai_sleep_ms(int milliseconds);

uint64_t nxai_current_timestamp_ms();

uint64_t nxai_current_timestamp_us();

void nxai_chdir(const char* path);

void nxai_chmod(const char* filepath, int mode);

int nxai_strcasecmp(const char* str1, const char* str2);

// Sets the OS-visible name of a thread (visible in debuggers, top, etc.).
// Name is silently truncated to 15 characters on Linux.
void nxai_set_thread_name(std::thread& thr, const char* name);

// Lightweight scope timer for profiling hot paths.
// Construct with a name to start the clock; call lap(label) to log microseconds
// since the last lap (or construction) and since construction. The destructor
// logs a final lap under the timer's own name.
class NxAITimer
{
public:
    static void setEnabled(bool enabled);
    explicit NxAITimer(const char* name);
    ~NxAITimer();
    NxAITimer(const NxAITimer&) = delete;
    NxAITimer& operator=(const NxAITimer&) = delete;
    NxAITimer(NxAITimer&&) = delete;
    NxAITimer& operator=(NxAITimer&&) = delete;
    void lap(const char* label);

private:
    static bool s_enabled;
    const char* m_name;
    bool m_enabled;
    std::chrono::steady_clock::time_point m_start;
    std::chrono::steady_clock::time_point m_last;
};

#if defined(_MSC_VER)
DWORD get_windows_error(DWORD errorCode, char* buffer, DWORD bufferSize);
#endif
