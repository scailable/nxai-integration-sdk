# DESIGN.md template

Write `<processor-dir>/DESIGN.md` from this skeleton in step 3. The reader is the agent that implements the processor, now or later. Keep each section short and concrete: decisions, values, examples. Write it in English.

````markdown
# <processor-dir>: design

<One or two sentences: what the processor does, in VMS terms.>

## Decisions

| # | Branch | Decision | Reason |
|---|---|---|---|
| 1 | Side | postprocessor | <from the description, or the user's answer> |
| 2 | Schedule | `OBJECTS` | ... |
| 3 | Response | response | ... |
| 4 | Output changes | attribute `<key>` on `<class>`; event `<id>` | ... |
| 5 | Extra inputs | none | ... |
| 6 | Parameters | `UserSettings`: `<control>` (default `<value>`) | ... |
| 7 | State | per (`DeviceID`, track id), removed after <N> s unseen | ... |
| 8 | Latency | synchronous, `TimeoutMs` <N> | ... |
| 9 | Language | Python | ... |
| 10 | Dependencies | `requests`; endpoint URL and API key from `Config` | ... |
| 11 | Target OS | Linux | ... |
| 12 | Name | `postprocessor-python-<short>` | ... |

A branch that does not apply has "not applicable" and the reason.

## Template

<The example directory to copy, and what changes. For a C++ preprocessor: the parts and where
each comes from, from templates.md.>

## Registration

<The planned JSON entry: Name, Command, Schedule, LogFilepath, Config, and each optional field
with its value. Each class ID and event ID with its display name.>

## Acceptance examples

Each example becomes one unit test.

1. **<short title>**
   - Input: <the message fields that matter: device, objects with class / box / track id,
     attributes, user settings, earlier frames for stateful logic>
   - Expected output: <the fields that change, and the ones that must stay unchanged>

## Edge cases

- <empty `objects`; absent setting key; unknown class; the first frame of a camera;
  a track that disappears; a slow or failing external call; ...> → <expected behaviour>

## Dependencies

<Each package, model file and network service: what it is, where it comes from, how the tests
replace it.>

## Target OS and verification

<Target OS. What the skill verifies on this host. If the target is not the host: what stays
unverified, and the commands to run on the target.>

## Deviations

<Empty at the start. Step 4 adds one line for each change against this file: what changed,
why, and whether the user approved it.>
````
