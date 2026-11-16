---
name: new-processor
description: Design and build a new pre- or post-processor from a description, by interview.
argument-hint: "[description]"
disable-model-invocation: true
---

Turn the user's description into a working processor in this repo: interview, `DESIGN.md`, test-first code, registration JSON, CMake, README. The user's description is the text after the skill name, or the request that started the skill. The user can be a novice: explain each choice in VMS terms.

The skill makes no git commits and adds no scenario to `tests/e2e`.

## Steps

### 1. Intake

Read the description and `design-tree.md`. Infer every branch the description settles. If there is no description, ask: "What must the processor do? Describe what VMS must show or do."

Show the list "I understood it like this": one line per inferred branch, in tree order, each with its decision in VMS terms.

**Done when** the list is shown and the user has corrected it or accepted it.

### 2. Interview

Ask about each open branch of `design-tree.md`, in tree order, one question per turn, as that file says. Each answer can settle or skip later branches.

If the directory of the agreed name exists, stop and ask for a different name.

**Done when** each branch of `design-tree.md` has a decision or a reason why it does not apply.

### 3. DESIGN.md

Write `<processor-dir>/DESIGN.md` from `design-template.md`. Give at least one acceptance example for each output change, each parameter and each state rule, and the edge cases of the decisions. Show the file to the user and ask for a "yes".

**Done when** the user says "yes" to the `DESIGN.md`. Change it and ask again after each correction. If a correction changes the name, rename the directory. Start the code after the "yes".

### 4. Implement

Read `implement.md` and do its steps.

**Done when** each item of the "Done" list in `implement.md` is true.

### 5. README and report

Write `<processor-dir>/README.md` from `DESIGN.md`, for the operator who installs the processor, in the section order of `.claude/docs/writing_a_processor.md`, "README.md". Keep `DESIGN.md`. If the target OS is Windows or both, give the Windows `Command` and `LogFilepath` in the "Registration" section, and the Windows build (`build-windows.cmd <processor-dir>`) and the PowerShell unit-test command in "Build, test, install".

Then report:

- the files created, and the line added to the root `CMakeLists.txt`;
- the test, wiring scan, build and registration check results;
- each deviation from the first approved `DESIGN.md`;
- if the target OS is not the host: what is not verified, and the commands to run on the target;
- next: install with `/install-processor <processor-dir>`, or for fast iteration run it in dev mode with `Port` (`.claude/docs/external_postprocessors_json.md`, "The development pattern: use `Port`");
- for contributors to this SDK: a harness scenario in `tests/e2e` is a separate step (`writing_a_processor.md`, "Tests").

**Done when** the README exists and the report is shown.
