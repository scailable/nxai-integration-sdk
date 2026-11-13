# e2e post-processor harness

`postprocessor_harness.cpp` builds a single test binary that **emulates the AI
Manager**: it creates the two shared-memory IPC channels, launches a processor as a
child, drives it for one or more frames, and asserts on the response (or on its absence
for `no-response` and `edgeimpulse`). The binary *is* the
test — its exit code is the CTest verdict, and every assertion lives in the C++ harness, not in a script.

## Platforms

The harness builds and runs on Linux and on Windows/MSVC. Its only platform-dependent
part is starting the child processor and tearing it down, which lives behind the
launcher in `child_process.h` (`spawn` / `request_termination` / `wait`, with handle
inheritance an explicit argument of `spawn`); the launcher has its own CTest case,
`e2e-child-process`.

On Linux, configure and build the targets first, then run CTest from the repo root.
CTest runs prebuilt binaries and does not build them:

```
cmake -S . -B build
cmake --build build --target e2e-postprocessor-harness e2e-child-process-test e2e-ipc-sync-obj-test e2e-ipc-failure-paths-test e2e-ipc-exit-signal-test e2e-processor-loop-test postprocessor-cpp-objects-example postprocessor-cpp-tensors-example
ctest --test-dir build
```

A bare `cmake --build build` also packages every Python processor, which no case in the
suite uses.

All `e2e-python*` cases need Python 3.12 or later at configure time. Without it, CMake
does not register them. The `e2e-python-edgeimpulse` and `e2e-python-objects-sender`
cases also need the extras virtual environment that configure creates in
`build/tests/e2e/e2e-extras-venv`. This step installs packages with `pip`, so it needs
network access. If it fails, configure prints a warning and skips these two cases.

No case is gated by platform. Windows registers the same cases: the Python SDK's IPC
layer branches per platform, the SHM growth in `_ipc_send_response` included.

**On Windows, verify with the checked-in entry point — `verify-windows.cmd` in the repo
root.** It enters the MSVC developer environment itself (so start it from any shell),
configures, builds only the C++ targets the suite uses, runs CTest and exits non-zero on
any failing case:

```
.\verify-windows.cmd [build-dir]      # default build dir: build-windows
```

Do not run a bare `cmake --build` there: it would also package every Python processor
with Nuitka, which no case in the suite uses.

## Running

```
postprocessor_harness --check <scenario> <args...>
```

`--check` is the **sole scenario selector** and is mandatory; a missing or unknown
scenario is a usage error, not a silent default. Per-scenario `<args...>`:

| Scenario | `<args...>` |
|----------|-------------|
| `objects` | `<expected_class> <expected_source> <child_program> [prefix_args...]` |
| `realloc` | `<expected_class> <expected_source> <child_program> [prefix_args...]` |
| `tensors` | `<child_program> [prefix_args...]` |
| `settings` | `<child_program> [prefix_args...]` |
| `counter` | `<child_program> [prefix_args...]` |
| `height` | `<child_program> [prefix_args...]` |
| `objects-sender` | `<child_program> [prefix_args...]` |
| `dashboard` | `<child_program> [prefix_args...]` |
| `skip-feature-extraction` | `<child_program> [prefix_args...]` |
| `image` | `<child_program> [prefix_args...]` |
| `preproc-tensor` | `<child_program> [prefix_args...]` |
| `preproc-image` | `<child_program> [prefix_args...]` |
| `preproc-image-mirror` | `<child_program> [prefix_args...]` |
| `preproc-image-realloc` | `<child_program> [prefix_args...]` |
| `preproc-image-realloc-multi` | `<child_program> [prefix_args...]` |
| `no-response` | `<child_program> [prefix_args...]` |
| `confidences` | `<child_program> [prefix_args...]` |
| `events` | `<child_program> [prefix_args...]` |
| `bestshots` | `<child_program> [prefix_args...]` |
| `bestshots-no-emit` | `<child_program> [prefix_args...]` |
| `edgeimpulse` | `<child_program> [prefix_args...]` |
| `negative` | `<missing-keys\|truncated\|no-image> <recovers\|dies> <child_program> [prefix_args...]` |

CTest registers `e2e-python-preproc-image-realloc-multi` as DISABLED, so it does not
count as a passing case. A send that times out during an SHM reallocation leaves the
next frame's send failing with `BUSY`, and the contract for what a sender does after
`BUSY` is not settled.

The harness inserts the IPC sync strings and the `--ProcessorName` flag into the child's argv, right
after `<child_program>`:

```
<child_program> [pre_args...] <input_str> <output_str> --ProcessorName e2e-test [post_args...]
```

`pre_args` are tokens that must precede the sync pair positionally (a Python script
path: `python script.py <sync1> <sync2> ...`); `post_args` are scenario-specific
Config `--key value` flags (`load_config()` scans the whole argv for them regardless
of position, but the two sync strings must be the Python SDK's
`initialize_from_args()`'s first two positionals exactly --
`nxai_postprocessor_sdk.py` -- so a Config flag cannot sit ahead of them). A literal
`--` token in a scenario's `<args...>`, after `<child_program>`, splits the two:
`... postprocessor-python-bestshots-example.py -- --emit_best_shots true`. With no
`--` token every arg is a `pre_arg` -- the shape every scenario without extra Config
flags uses. CTest cases wire these invocations up in `tests/e2e/CMakeLists.txt`.

## How a scenario is built

A **check scenario** is a named seed/assert pair.
Each one is hardcoded in C++ — it is *not* a config flag. A scenario touches five
places in `postprocessor_harness.cpp`:

1. **Enum arm** — add to `enum class Check`.
2. **`--check` string map** — map the scenario name to the enum value in the `if`/`else if` chain in `main` that sets `check`.
3. **Seed** — in the request-building block of `main`, construct the
   `ObjectsPostProcessorMessage` / `TensorsMessage` the harness sends, using the
   `write*Message` codec for that message type.
4. **Assert function** — a free function returning `bool` that inspects the parsed
   response (e.g. `check_response`, `check_settings_response`, `check_tensors_response`,
   `check_counter_response`, `check_height_response`, `check_dashboard_response`).
5. **Dispatch** — call the right assert from the response-handling block
   of `main`, and add the scenario to the `usage` lambda and to the usage comment at the top of the file.

The message type a scenario speaks (objects / tensors / image) is fixed by the
scenario itself — it seeds with one codec and asserts the matching response.

## Asserting on events and attributes (objects scenarios)

The C++ side **already parses `Events`** —
`parseObjectsPostProcessorMessage` fills `parsed->events`
(`parseObjectsPostProcessorMessage` in `external_processor_utilities.cpp`, struct `EventData{type_id, ...}`).
An event-emission assert therefore needs **no struct/parser extension**, only a new
assert function reading `parsed->events`. Likewise per-object `attributes` are parsed
into the `ObjectData` struct.

Assertion granularity is chosen per scenario, by what regression it must catch:

- **Event presence** → assert on `type_id` (the event ID) only. The ID is the UI
  contract (it must also be declared in `external_postprocessors.json`); the
  `Description` is brittle prose not worth pinning.
- **Attribute formatting** → assert the **exact** attribute string when the format
  is user-visible and behavior-preserving migration must keep it byte-for-byte.
- **Value identity** (e.g. a forwarding processor) → assert field-for-field, with
  `approx` for the normalized float bbox fields.

## `skip-feature-extraction`: what it proves, and what it does not

`skip-feature-extraction` seeds an empty objects message and asserts the fixture
(`postprocessor-skipfeatureextraction-fixture.py`) emits a `"chain-test"` class with a
`role=gated` object carrying the `"skipFeatureExtraction": true` wire key and a
`role=ungated` object with the key entirely absent. The fixture sets the field through
`ObjectMetadata.skip_feature_extraction` (`nxai_postprocessor_sdk.py`), which writes the
key only when the attribute is set. `ObjectData.skip_feature_extraction`
(`external_processor_utilities.h`) is parsed read-only for this assertion; the C++ side
has no writer support.

**This is a wire-contract check only.** The harness *emulates* the AI Manager — it
never runs the real engine — so it cannot observe the actual behavior the field is for:
suppressing a chained `FEATURE_EXTRACTION` model's inference for the gated object on
that frame.
Proving that needs a real engine build and a real chained model, which this offline
harness cannot provide. This
scenario exists to regression-protect the wire shape the real engine's deserializer
depends on, not the chaining gate's runtime effect.
