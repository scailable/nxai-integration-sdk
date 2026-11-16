# Templates

Material that only `new-processor` needs. The shape of a Python processor (template table, names, main script, JSON, CMake, README, tests) is in `.claude/docs/writing_a_processor.md`. Read it first. This file adds the C++ processors, the C++ preprocessor (which has no template), and the background worker.

## C++ templates

| Processor | Template |
|---|---|
| Postprocessor, `OBJECTS` | `postprocessor-cpp-objects-example` |
| Postprocessor, `TENSOR` | `postprocessor-cpp-tensors-example` |
| Preprocessor, `IMAGE` or `TENSOR` | none, see "C++ preprocessor" |

The templates have one `main.cpp`. Make the business logic testable without IPC:

- **`handler.h` / `handler.cpp`**: `void handle_message(<Message>& msg);` and the constants it uses (`kInjectedClass`). The message type is `ObjectsPostProcessorMessage`, `TensorsMessage` or `ImagePreprocessorMessage` from `nxai-utilities/external_processor_utilities.h`.
- **`main.cpp`**: the `main` of the template without changes, apart from `#include "handler.h"` and the parse/write pair. It passes `handle_message` to `run_receive_loop` (`nxai-utilities/processor_loop.h`).
- **`test_<processor_dir>.cpp`**: builds messages, calls `handle_message`, and checks the result. Use the `check(condition, what)` pattern of `tests/e2e/processor_loop_test.cpp`: print `FAIL: <what>` for each failed check, and return 1 from `main` if one failed.
- **`CMakeLists.txt`**: copy the template, set `project(<processor-dir> CXX C)`, add `handler.cpp` to the sources. Add a second executable `test_${PROJECT_NAME}` with `test_<processor_dir>.cpp`, `handler.cpp` and the same `nxai-utilities` sources, and the same `target_include_directories` and `rt` link. Install only the processor target. Run the test binary directly: the root `CMakeLists.txt` calls `enable_testing()` after the processor lines, so CTest does not see this directory.
- **Config**: read `--key value` pairs from `argv` the way `processor_name_from_args` does. Pass the values to the handler through `handler.h` (for example a `struct Settings` that `main` fills). For what C++ cannot do, see `design-tree.md`, branch 9.
- **Names**: the binary and the build target are `<processor-dir>`. On Windows CMake adds `.exe`. `<name>` follows "Names": for `postprocessor-cpp-blur-faces` it is `cpp-blur-faces`, and the new class ID is `cpp_blur_faces`.

## C++ preprocessor

There is no template. The skill assembles it from three parts. Write them into `DESIGN.md`, "Template":

1. **Loop, CMake, test**: `postprocessor-cpp-objects-example`, in the shape of "C++ templates", with the preprocessor install destination of `writing_a_processor.md`, "CMakeLists.txt".
2. **Semantics**: `preprocessor-python-image-example` (`IMAGE`) or `preprocessor-python-tensor-example` (`TENSOR`). A preprocessor always responds. `run_receive_loop` does that.
3. **Codecs** from `nxai-utilities/external_processor_utilities.h`:
   - `IMAGE`: `parseImagePreprocessorMessage` / `writeImagePreprocessorMessage`. The parse takes a `std::span<char>`, so the parse lambda wraps the raw buffer: `std::span<char>(reinterpret_cast<char*>(const_cast<unsigned char*>(data)), size)`.
   - `TENSOR`: `parseTensorMessage` / `writeTensorMessage`.

The registration file is `external_preprocessors.json`, with `Schedule` `"IMAGE"` or `"TENSOR"`. Copy the entry of the Python preprocessor example and change `Name`, `Command` and `LogFilepath`.

## Background worker (Python)

For a slow step whose result can come some frames later (`design-tree.md`, branch 8). `handle_message` never waits. It reads the cache and gives new work to one daemon thread:

```python
JOB_QUEUE_SIZE = 32
RESULT_TTL_S = 30.0

_jobs: "queue.Queue[tuple]" = queue.Queue(maxsize=JOB_QUEUE_SIZE)
_results: dict[tuple[str, str], tuple[str, float]] = {}  # (device_id, track_id) -> (result, last seen)
_pending: set[tuple[str, str]] = set()
_lock = threading.Lock()


def handle_message(msg):
    now = time.monotonic()
    with _lock:
        for objects in msg.objects.values():
            for obj in objects:
                key = (msg.device_id, obj.track_id)
                if key in _results:
                    result, _ = _results[key]
                    _results[key] = (result, now)
                    obj.attributes[ATTRIBUTE] = result
                elif key not in _pending:
                    try:
                        _jobs.put_nowait((key, _job_input(msg, obj)))
                        _pending.add(key)
                    except queue.Full:
                        pass  # retried on a later frame
        _forget_older_than(now - RESULT_TTL_S)
    return msg


def run_worker(stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            key, job_input = _jobs.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            result = _slow_step(job_input)
            with _lock:
                _results[key] = (result, time.monotonic())
        except Exception:
            logger.exception("slow step failed for %s", key)  # retried on a later frame
        finally:
            with _lock:
                _pending.discard(key)
```

- **`__main__`**: this is the one change to the `__main__` block of the template. Start `threading.Thread(target=run_worker, args=(stop,), daemon=True)` before `run_message_loop(processor)`. After the loop, set `stop` and `join` the thread.
- **Bound everything**: the queue has a maximum size, and `_forget_older_than` removes cache entries that were not seen for `RESULT_TTL_S`. Record both values in `DESIGN.md`.
- **Tests**: call `_slow_step` and `handle_message` directly. Fill `_results` to test the cached path, and read `_jobs` to test the enqueue path. Mock the slow call at its boundary. The variant has no `_results`: test the enqueue path and the drop count. Test the thread only for "no thread is alive after the stop", as `postprocessor-python-web-dashboard` does.
- **No track id** (preprocessor, `TENSOR` schedule): key by `msg.device_id` only, and keep one result per camera.
- **Variant: a worker that returns no result to VMS** (forward, upload). Drop `_results`, `_pending` and `_forget_older_than`.
  - `handle_message` calls `_jobs.put_nowait` and returns `msg`.
  - A full queue drops the item. No later frame retries it, so count the drops and log the count.
  - To signal the loop (for example for an event), the worker sets a flag under `_lock`. `handle_message` reads the flag under `_lock`, clears it and appends the event.
  - The processor still responds if it adds an event. Only a processor with no output at all uses `NoResponse` (`design-tree.md`, branch 3).
