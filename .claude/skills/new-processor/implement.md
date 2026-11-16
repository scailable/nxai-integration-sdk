# Implement

Step 4 of `new-processor`. `<processor-dir>/DESIGN.md` is the specification. Python processors follow `.claude/docs/writing_a_processor.md`, C++ processors follow `templates.md`.

## 1. Files from the template

Create `<processor-dir>/` and copy the files of the template that `DESIGN.md` names. Apply the names of `writing_a_processor.md`, "Names", to each file: the script, the unit test, `PROCESSOR_NAME` in the script and in CMake, the class ID. In the copied unit test, replace the template directory name, its `_` form and its old class ID and `PROCESSOR_NAME` literals, so that the test loads the new module.

Python: copy `requirements.txt` if the template has one, else write `msgpack` and `nuitka`. Add the packages of `DESIGN.md`, "Dependencies".

## 2. Red: the acceptance examples as tests

Replace the test cases of the template with one test for each acceptance example of `DESIGN.md`, in the same order, and one for each edge case. The test name says the expected behaviour. Each test builds the input message, calls `handle_message`, and checks the expected output, including the fields that must stay unchanged and the `DeviceID` and `Timestamp` of the request.
Stubs first: where the template has no handler (a C++ preprocessor), write `handle_message` as a pass-through. In Python, make `handle_message` return `msg` unchanged and add each new public name the tests use (function, constant) as a minimal stub. The tests then link or import, and fail on their assertions, not on a missing name.


Run the tests. Each new test fails on its assertion, or passes because the template already behaves as the example says (for example "the objects stay unchanged"). No test fails on an import, a link or a syntax error:

- Python, Linux: `cd <processor-dir> && PYTHONPATH=../nxai-utilities python3 -m unittest test_<processor_dir> -v`
- Python, Windows PowerShell: `cd <processor-dir>; $env:PYTHONPATH='..\nxai-utilities'; python -m unittest test_<processor_dir> -v`
- C++: the test target needs the build wiring, so do 5. and the configure of 6. first. Then build the target `test_<processor-dir>` and run its binary.

Install the packages of `requirements.txt` into the Python environment of the tests first.

## 3. Green: the logic

Write `handle_message` and its helpers until all tests pass. Public functions first, private helpers (leading `_`) at the bottom of the module, and the `__main__` block last. Change the template `__main__` block and the message loop only where `templates.md` says so (background worker).

## 4. Registration JSON

Write `<processor-dir>/external_postprocessors.json` or `external_preprocessors.json` from `DESIGN.md`, "Registration": a bare array with one entry. `Name`, `Command` and `LogFilepath` follow "Names" for the target OS. Declare each class the code adds under `Objects` and each event ID it emits under `Events`, verbatim.

Then load the file with a JSON parser (for example `python -c "import json,sys; json.load(open(sys.argv[1]))" <file>`). It must parse. A Windows path in `Command` or `LogFilepath` needs doubled backslashes.

## 5. CMake and wiring

1. Write `<processor-dir>/CMakeLists.txt` as `writing_a_processor.md` ("CMakeLists.txt") or `templates.md` (C++) says.
2. In the root `CMakeLists.txt`, insert `add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/<processor-dir>)` directly after the last `add_subdirectory` line of a processor directory (`postprocessor-*` or `preprocessor-*`), before `enable_testing()` and the `tests/` lines.
3. Run `python tests/build-wiring/build_wiring_scan.py --root .` from the repo root. It must pass.

## 6. Build

Build the processor target, and for C++ also `test_<processor-dir>`.

- Linux: in `build/`, configure as `.claude/skills/install-processor/SKILL.md` step 3 says if `build/CMakeCache.txt` is missing, then `cmake --build . --target <processor-dir>`.
- Windows: run `cmd /c <repo root>\build-windows.cmd <processor-dir>` from PowerShell, for C++ `cmd /c <repo root>\build-windows.cmd <processor-dir> test_<processor-dir>`. In Git Bash write `cmd //c`: Git Bash rewrites `/c` as a path, so `cmd /c` starts an interactive shell, builds nothing and exits 0. It configures `build-windows/` if needed.

The build must exit 0 and leave the binary `<build dir>/<processor-dir>/<processor-dir>`, with `.exe` on Windows. Ninja, the generator of `build-windows/`, prints no "Built target" line. For C++, run the test binary again from the build tree: `<build dir>/<processor-dir>/test_<processor-dir>`, with `.exe` on Windows.

## 7. Registration check

Read `.claude/skills/check-registration/SKILL.md` and run its steps 2 to 8 for the new processor. It is not installed, so the check uses the colocated JSON. Every row must be ✅. Fix each ⚠️ in the code or the JSON, then run the check again.

## Deviations from DESIGN.md

When the implementation cannot follow `DESIGN.md`:

- **Behaviour or an acceptance example changes**: stop and ask the user one question. Continue after the answer.
- **Only the implementation changes** (a library, a template, a helper): decide yourself.

In both cases, update `DESIGN.md` at once (the decision table or the example, and one line under "Deviations"), and name the change in the final report.

## Done

All of these are true:

- every acceptance example and edge case has a test, and all tests pass;
- the build wiring scan passes;
- the registration JSON parses;
- the build exits 0 and the processor binary exists;
- the registration check shows ✅ in every row;
- `DESIGN.md` matches the code.
