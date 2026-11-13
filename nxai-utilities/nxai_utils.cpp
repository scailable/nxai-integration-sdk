#include "nxai_utils.h"

#include <mutex>
#include <stdexcept>

#include <errno.h>
#include <inttypes.h> // PRIu64
#include <limits.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_MSC_VER)
    // Windows stuff
    #include <basetsd.h>
    #include <direct.h>
    #include <errno.h>
    #include <io.h>

    #include "windows.h"
#else
    // Linux stuff
    #include <sys/stat.h>
    #include <unistd.h>
#endif

// Variable definitions
char* _start_log_filepath = NULL;
char* _rotating_log_filepath = NULL;
char* _log_prefix = NULL;
char* _old_logfile_path = NULL;

static uint64_t last_timestamp = 0;
size_t logfile_max_size_mb = 10;
static bool start_logfile_full = false;
static size_t logfile_last_size = 0;
static bool _log_to_console = false;
static bool _log_to_file = true;
static int _log_verbosity_level = 1;
static bool log_mutex_initialized = false;
FILE* start_logfile;
FILE* rotating_logfile;
std::mutex log_file_mutex;

// Forward declarations
static void nxai_vvlog(const char* fmt, va_list* args);

#if defined(_MSC_VER)
    #define NXAI_SET_FILE_PERMS _chmod
#else
    #define NXAI_SET_FILE_PERMS chmod
#endif

void nxai_initialize_logging(
    const char* start_log_filepath,
    const char* rotating_log_filepath,
    const char* log_prefix,
    bool log_to_console,
    bool log_to_file,
    int log_verbosity_level)
{
    start_logfile_full = false;
    logfile_last_size = 0;
    _start_log_filepath = strdup(start_log_filepath);
    _rotating_log_filepath = strdup(rotating_log_filepath);
    _log_prefix = strdup(log_prefix);
    _log_to_console = log_to_console;
    _log_verbosity_level = log_verbosity_level;
    _log_to_file = log_to_file;
    log_mutex_initialized = true;
    if (_log_to_file == true)
    {
        // Create and clear log files
        start_logfile = fopen(_start_log_filepath, "w");
        if (start_logfile == NULL)
        {
            printf("Failed to initialise logfile: %s\n", _start_log_filepath);
        }
        NXAI_SET_FILE_PERMS(_start_log_filepath, 0666);
        rotating_logfile = fopen(_rotating_log_filepath, "w");
        if (rotating_logfile == NULL)
        {
            printf("Failed to initialise logfile: %s\n", _rotating_log_filepath);
        }
        NXAI_SET_FILE_PERMS(_rotating_log_filepath, 0666);
        // Create old log file path
        size_t old_filepath_length = strlen(_rotating_log_filepath) + 5 + 1;
        _old_logfile_path = (char*) malloc(old_filepath_length);
        strcpy(_old_logfile_path, _rotating_log_filepath);
        strcat(_old_logfile_path, ".old");
    }
}

void nxai_finalise_logging()
{
    if (log_mutex_initialized)
    {
        log_file_mutex.lock();
    }

    _log_to_console = false;
    _log_to_file = false;
    _log_verbosity_level = 0;

    if (start_logfile != NULL)
    {
        fclose(start_logfile);
        start_logfile = NULL;
    }
    if (rotating_logfile != NULL)
    {
        fclose(rotating_logfile);
        rotating_logfile = NULL;
    }

    start_logfile_full = true;
    logfile_last_size = 0;

    if (log_mutex_initialized)
    {
        log_file_mutex.unlock();
    }
    log_mutex_initialized = false;

    free(_start_log_filepath);
    _start_log_filepath = NULL;
    free(_rotating_log_filepath);
    _rotating_log_filepath = NULL;
    free(_log_prefix);
    _log_prefix = NULL;
    free(_old_logfile_path);
    _old_logfile_path = NULL;
}

void nxai_vlog_profiling(const char* fmt, ...)
{
    if (_log_verbosity_level > 2)
    {
        va_list args;
        va_start(args, fmt);
        nxai_vvlog(fmt, &args);
        va_end(args);
    }
}

void nxai_vlog_verbose(const char* fmt, ...)
{
    if (_log_verbosity_level > 1)
    {
        va_list args;
        va_start(args, fmt);
        nxai_vvlog(fmt, &args);
        va_end(args);
    }
}

void nxai_error_log(const char* fmt, ...)
{
    va_list args;
    va_list args_copy;

    // Initialize both va_lists at once
    va_start(args, fmt);
    va_copy(args_copy, args);

    // First, process arguments for stderr printing
    vfprintf(stderr, fmt, args);

    // Process the same arguments for vvlog
    nxai_vvlog(fmt, &args_copy);

    // Clean up both va_lists
    va_end(args);
    va_end(args_copy);
}

void nxai_vlog(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    nxai_vvlog(fmt, &args);
    va_end(args);
}

bool nxai_get_file_size(const char* filepath, size_t* file_size)
{
#if defined(_MSC_VER)
    // Windows specific implementation
    WIN32_FIND_DATAA fileData;
    HANDLE hFile = FindFirstFileA(filepath, &fileData);

    if (hFile != INVALID_HANDLE_VALUE)
    {
        *file_size = (size_t) fileData.nFileSizeLow;
        FindClose(hFile);
        return true;
    }
    else
    {
        printf(
            "An unexpected error occurred accessing log file: %s %s\n",
            filepath,
            strerror(errno));
        return false;
    }

#else
    // Linux specific implementation
    struct stat file_stat;
    if (stat(filepath, &file_stat) < 0)
    {
        switch (errno)
        {
            case EACCES: // Permission denied
                printf("Permission denied trying to open logfile: %s.\n", filepath);
                break;
            default:
                printf(
                    "An unexpected error occurred accessing log file: %s %s\n",
                    filepath,
                    strerror(errno));
        }
        return false;
    }
    *file_size = file_stat.st_size;
    return true;
#endif
}

static void nxai_vvlog(const char* fmt, va_list* args)
{
    if (_log_verbosity_level == 0 || (_log_to_file == false && _log_to_console == false))
    {
        // Logging is turned off. Return immediately
        return;
    }

    // Copy argument list
    va_list copied_args;
    va_copy(copied_args, *args);

    // Get the current timestamp
    uint64_t timestamp = nxai_current_timestamp_us();

    uint64_t duration = 0;
    if (last_timestamp == 0)
    {
        last_timestamp = timestamp;
    }

    duration = timestamp - last_timestamp;
    last_timestamp = timestamp;

    const char* log_prefix = _log_prefix;
    if (log_prefix == NULL)
    {
        log_prefix = "";
    }

    if (_log_to_console == true)
    {
        // Print to console
        printf("%s%" PRIu64 " %09" PRIu64 ": ", log_prefix, (uint64_t) timestamp / 1000, duration);
        vprintf(fmt, *args);
    }

    if (_start_log_filepath == NULL || _rotating_log_filepath == NULL || _log_to_file == false
        || log_mutex_initialized == false)
    {
        va_end(copied_args);
        return;
    }

    log_file_mutex.lock();

    if (_start_log_filepath == NULL || _rotating_log_filepath == NULL || _log_to_file == false)
    {
        log_file_mutex.unlock();
        va_end(copied_args);
        return;
    }

    FILE* flogfile = NULL;

    // Determine which file to log to
    if (start_logfile_full == false)
    {
        // Check if start logfile is full
        if ((size_t) logfile_last_size < logfile_max_size_mb * 1000000)
        {
            // Write to start_log
            flogfile = start_logfile;
        }
        else
        {
            start_logfile_full = true;
            logfile_last_size = 0;
            if (start_logfile != NULL)
            {
                fclose(start_logfile);
                start_logfile = NULL;
            }
        }
    }

    if (flogfile == NULL)
    {
        // Start logfile was full or unavailable, open rotating logfile
        if (rotating_logfile == NULL)
        {
            rotating_logfile = fopen(_rotating_log_filepath, "a");
            if (rotating_logfile == NULL)
            {
                fprintf(stderr, "Error creating log file");
                log_file_mutex.unlock();
                va_end(copied_args);
                return;
            }
        }
        if ((size_t) logfile_last_size > logfile_max_size_mb * 1000000)
        {
            // Rotating logfile is full, rename to ".old"
            if (rotating_logfile != NULL)
            {
                fclose(rotating_logfile);
                rotating_logfile = NULL;
                remove(_old_logfile_path);
                int result = rename(_rotating_log_filepath, _old_logfile_path);
                if (result != 0)
                {
                    fprintf(stderr, "Error renaming file: %s\n", strerror(errno));
                    log_file_mutex.unlock();
                    va_end(copied_args);
                    return;
                }
            }
            // Create new log file
            rotating_logfile = fopen(_rotating_log_filepath, "w");
            if (rotating_logfile == NULL)
            {
                fprintf(stderr, "Error creating log file");
                log_file_mutex.unlock();
                va_end(copied_args);
                return;
            }
            logfile_last_size = 0;
        }
        // Write to rotating log
        flogfile = rotating_logfile;
    }

    if (flogfile == NULL)
    {
        log_file_mutex.unlock();
        va_end(copied_args);
        return;
    }

    // Write to logfile
    int bytes_written = fprintf(
        flogfile,
        "%s%" PRIu64 " %09" PRIu64 ": ",
        log_prefix,
        (uint64_t) timestamp / 1000,
        duration);

    if (bytes_written < 0)
    {
        printf("Failed to write to log file!\n");
        log_file_mutex.unlock();
        va_end(copied_args);
        return;
    }

    logfile_last_size += bytes_written;

    bytes_written = vfprintf(flogfile, fmt, copied_args);

    if (bytes_written < 0)
    {
        printf("Failed to write to log file!\n");
        log_file_mutex.unlock();
        va_end(copied_args);
        return;
    }
    logfile_last_size += bytes_written;
    if (_log_verbosity_level > 1)
    {
        fflush(flogfile); // Flush writing file to make sure latest prints are logged
        fflush(stdout);
    }
    log_file_mutex.unlock();
    va_end(copied_args);
}

void nxai_sleep_ms(int milliseconds)
{
#if defined(_MSC_VER)
    // Windows implementation
    Sleep(milliseconds);
#else
    // Linux implementation
    usleep(milliseconds * 1000);
#endif
}

uint64_t nxai_current_timestamp_ms()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    uint64_t time_ms = ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    return time_ms;
}

uint64_t nxai_current_timestamp_us()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    uint64_t time_us = ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
    return time_us;
}

void nxai_chdir(const char* path)
{
#if defined(_MSC_VER)
    // Windows implementation
    _chdir(path);
#else
    // Linux implementation
    chdir(path);
#endif
}

int nxai_strcasecmp(const char* str1, const char* str2)
{
#if defined(_MSC_VER)
    // Windows implementation
    return _stricmp(str1, str2);
#else
    // Linux implementation
    return strcasecmp(str1, str2);
#endif
}

void nxai_chmod(const char* filepath, int mode)
{
#if defined(_MSC_VER)
    // Windows implementation
    _chmod(filepath, mode);
#else
    // Linux implementation
    chmod(filepath, mode);
#endif
}

#if defined(_MSC_VER)
DWORD get_windows_error(DWORD errorCode, char* buffer, DWORD bufferSize)
{
    return FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        errorCode,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        buffer,
        bufferSize,
        NULL);
}
#endif

char* nxai_sprintf(size_t initial_size, const char* fmt, ...)
{
    // Initial allocation
    char* return_string = (char*) malloc(initial_size);
    if (!return_string)
    {
        return NULL;
    }

    va_list args;
    va_start(args, fmt);

    // First attempt to format string
    size_t len = vsnprintf(return_string, initial_size, fmt, args);
    va_end(args);

    // Check if buffer was too small
    if (len >= initial_size)
    {
        // Need larger buffer
        return_string = (char*) realloc(return_string, len + 1);
        if (!return_string)
        {
            return NULL;
        }

        // Restart va_list for second formatting attempt
        va_start(args, fmt);

        // Format string again with larger buffer
        vsnprintf(return_string, len + 1, fmt, args);
        va_end(args);
    }

    return return_string;
}

char* nxai_pointer_to_string(void* pointer)
{
    return nxai_sprintf(32, "%p", pointer); // 32 bytes is typically enough for pointers
}

bool NxAITimer::s_enabled = false;

void NxAITimer::setEnabled(bool enabled)
{
    s_enabled = enabled;
}

NxAITimer::NxAITimer(const char* name):
    m_name(name),
    m_enabled(s_enabled)
{
    if (m_enabled)
    {
        m_start = std::chrono::steady_clock::now();
        m_last = m_start;
    }
}

NxAITimer::~NxAITimer()
{
    lap(m_name);
}

void NxAITimer::lap(const char* label)
{
    if (!m_enabled)
    {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto since_last =
        std::chrono::duration_cast<std::chrono::microseconds>(now - m_last).count();
    const auto since_start =
        std::chrono::duration_cast<std::chrono::microseconds>(now - m_start).count();
    nxai_vlog(
        "[Timer] %s::%s | +%lld us | total %lld us\n",
        m_name,
        label,
        since_last,
        since_start);
    m_last = now;
}

void nxai_set_thread_name(std::thread& thr, const char* name)
{
#if defined(_MSC_VER)
    using SetThreadDescriptionFn = HRESULT(WINAPI*)(HANDLE, PCWSTR);
    static const auto fn = reinterpret_cast<SetThreadDescriptionFn>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription"));
    if (fn)
    {
        wchar_t wname[16];
        MultiByteToWideChar(CP_UTF8, 0, name, -1, wname, 16);
        fn(thr.native_handle(), wname);
    }
#else
    pthread_setname_np(thr.native_handle(), name); // silently truncated to 15 chars
#endif
}
