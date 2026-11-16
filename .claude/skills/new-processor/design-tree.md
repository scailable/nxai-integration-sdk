# Design tree

The branches are in dependency order: an earlier answer can settle or remove a later branch. Settle every branch for the `DESIGN.md`, by inference or by a question.

- **Infer** a branch from the description when its cues are there. Put the inference on the "I understood it like this" list, so that the user can correct it.
- **Ask** only about a branch that is still open. One question per turn. Ask about what the operator sees in VMS, with the SDK term in parentheses. Offer the options as choices when the agent has a choice mechanism (for example AskUserQuestion), and put the recommended option first.
- **Skip** a branch when its skip rule applies. Record it in `DESIGN.md` as "not applicable" with the reason.

Field and wire details: `.claude/docs/external_postprocessors_json.md`, `input_from_ai_manager.md`, `output_to_ai_manager.md`. Names: `.claude/docs/writing_a_processor.md`, "Names".

---

## 1. Side

**Ask:** "Must the processor change the camera frame before the model sees it, or work with what the model found?" (preprocessor / postprocessor)

**Cues:** crop, mask, resize, rotate, denoise, "before inference" → preprocessor. Boxes, detections, count, alarm, label, attribute, send results somewhere → postprocessor.

**Consequences:**
- Preprocessor: directory `preprocessor-<lang>-<short>`, `external_preprocessors.json`, `INSTALL_DEST_PREPROCESSORS`, `<side>` = `preprocess`. It always responds and adds no objects or events, so branches 3 and 4 do not apply.
- Postprocessor: directory `postprocessor-<lang>-<short>`, `external_postprocessors.json`, `INSTALL_DEST_POSTPROCESSORS`, `<side>` = `postprocess`.

## 2. Schedule

**Ask (postprocessor):** "Must the processor work with the detected objects (boxes the model already made), or with the raw output numbers of the model?" (`OBJECTS` / `TENSOR`)

**Ask (preprocessor):** "Must the processor get the full camera picture before the resize, or the prepared model input?" (`IMAGE` / `TENSOR`)

**Cues:** boxes, classes, tracks, counts → `OBJECTS`. Logits, embeddings, a custom model head, "decode the output" → postprocessor `TENSOR`. A picture operation → `IMAGE`. Normalization, channel order, a tensor operation → preprocessor `TENSOR`.

**Consequences:**
- Postprocessor `OBJECTS` (the default): `msg.objects[class]` with boxes normalized to `[0,1]`, track ids, `OriginalObjectID`.
- Postprocessor `TENSOR`: the processor replaces the model outputs. New boxes go out as one `bboxes-format:xyxysc;0:<class>` tensor in pixels of the model input image. It can add no attributes and no events, and it gets no track ids. A processor that needs raw tensors and tracks takes `OBJECTS` with `ReceiveBinaryData` (branch 5) instead.
- Preprocessor `IMAGE`: `msg.image`, uint8 RGB, 3 bytes per pixel, `CHW` or `HWC` as `layout` says.
- Preprocessor `TENSOR`: `msg.tensors`, the model input after the conversion.

## 3. Response (postprocessor only)

**Ask:** "Must the result of the processor show in VMS, or does the processor only send data to a different place (a file, a dashboard, a cloud service)?" (response / `NoResponse`)

**Skip:** preprocessor.

**Cues:** log, export, upload, forward, dashboard, "VMS shows nothing new" → `NoResponse`. If VMS must also show something (an event, a box, a label), for example "forward the data and raise an alarm when the server is unreachable", the processor responds: `NoResponse` cannot emit events.

**Consequences:** `NoResponse: true`. The processor sends nothing back, and the objects go on unchanged. Branch 4 does not apply. Python only: the C++ loop always sends a response.

## 4. Output changes (postprocessor with a response)

**Ask:** "What must change in VMS?" Several answers are possible:
- the boxes move, disappear or change class (change `msg.objects`);
- a label on an object (an attribute, `obj.attributes[key]`);
- a new kind of box (a new object class, `Objects`);
- an alarm in the event log (an event, `Events`).

**Skip:** preprocessor, `NoResponse`.

**Consequences:**
- Each new class gets an `Objects[]` entry. One class: the ID and display name rules of "Names". More classes: `<name>_<suffix>`, unique on the server.
- Each event gets an `Events[]` entry. Emit the ID exactly as declared, with no `nx.nxai.` prefix. Ask when the event fires: on each frame, once per object, or on a threshold crossing. "Once" needs state (branch 7).
- Details for `DESIGN.md`: the model classes the processor acts on, each attribute key and value, and each event display name. For each change to the boxes: which boxes it changes, by which criterion (class, confidence, size or zone), and which box field it reads or changes (`x`, `y`, `width`, `height`, `confidence`). Propose them from the description, and the user confirms them with the `DESIGN.md` in step 3.
- Postprocessor `TENSOR`: only new box tensors. The class in the tensor name is declared under `Objects`.

## 5. Extra inputs (postprocessor `OBJECTS` only)

**Ask:** "Does the processor need the camera picture itself?" (`ReceiveInputImage`) and "Does it need the raw model output next to the objects?" (`ReceiveBinaryData`)

**Skip:** preprocessor (the `IMAGE` message already holds the frame, the `TENSOR` message the tensors), postprocessor `TENSOR` (it gets tensors and never `ImageData`).

**Cues:** crop, colour, OCR, snapshot, "look at the object" → picture. Embedding, logits, a second model head → raw output.

**Consequences:** the flag in the JSON. Both blocks arrive in the same message: `msg.image_data`, `msg.tensors`.

## 6. Parameters

**Ask:** "Must an operator change a value in the camera settings of VMS while the processor runs, or is a value set at installation enough?" (`UserSettings` / `Config`, or none)

**Cues:** threshold, zone, list of classes, "the operator sets" → `UserSettings`. Endpoint, API key, model path, a value set once → `Config`.

**Consequences:**
- `UserSettings`: the JSON gets the controls, see `postprocessor-python-settings-example/settings_model.md`. The code reads them as `writing_a_processor.md`, "Live settings", says. Python only: the C++ codecs do not parse `UserSettings`.
- `Config`: launch-time `--key value` pairs. Python: `load_config(sys.argv)`. C++: read the pairs from `argv` the way `processor_name_from_args` in `nxai-utilities/processor_loop.h` reads `--ProcessorName`.
- Secrets (API keys, passwords) go in `Config`, not in the code.
- Details for `DESIGN.md`, for each parameter: the caption, the unit, the default, the range, and the behaviour when the value is not set or out of range. Propose them, and the user confirms them with the `DESIGN.md` in step 3.

## 7. State between frames

**Ask:** "Does the result for a frame depend on earlier frames: counting, dwell time, speed, 'only once per object'?"

**Skip:** the description is clearly frame by frame.

**Consequences:**
- One process serves all cameras that use the pipeline. Key the state by `msg.device_id`, and by `obj.track_id` for per-object state (postprocessor `OBJECTS` only).
- Bound the store: remove an entry that was not seen for a fixed time. Record the time in `DESIGN.md`.
- Acceptance examples for this branch are sequences of messages, not one message.

## 8. Latency budget

**Ask:** "A step of the processor is slow (a cloud call, a large model). Is a result that comes some frames later acceptable?"

**Skip:** the work is arithmetic on the message, with no I/O and no model.

**Consequences:**
- A later result is acceptable → a background worker with a per-track cache. `handle_message` responds at once with the cached result. See `templates.md`, "Background worker".
- The result is necessary on the same frame → synchronous. Set `TimeoutMs` in the JSON to the worst call time plus a margin. Tell the user that the pipeline waits for each response, so the analytics frame rate of the camera is at most one frame per call time.
- No result goes back to VMS → `NoResponse` (branch 3), unless VMS must also show something (an event, a box, a label). Then the processor responds. Use the variant "a worker that returns no result to VMS" in `templates.md`, "Background worker".

## 9. Language

**Infer:** Python, and put it on the "I understood it like this" list so that the user can correct it. **Ask** "Python or C++?" only when the user names C++ or a decision needs C++ (postprocessor `TENSOR`).

**Consequences:**
- C++ is not available with `UserSettings` (branch 6) or `NoResponse` (branch 3).
- Postprocessor `TENSOR` is C++ only: the SDK has no Python template for it.
- If the decisions conflict, ask which decision to change.
- C++ preprocessor: there is no template. Tell the user that the skill assembles it from parts, see `templates.md`, "C++ preprocessor".

## 10. External dependencies

**Ask:** "Does the processor need a software package, a model file, or a server on the network?" (library / model file / network service) Ask for each one the description names: which, where it comes from, and which credentials.

**Consequences:**
- Python packages go in `requirements.txt`, with `nuitka`. Each package must build into the Nuitka binary on the target OS.
- A model file: its path comes from `Config`. The build does not copy it.
- A network service: the URL and the credentials come from `Config`. Unit tests make no live call, see `writing_a_processor.md`, "Tests".

## 11. Target OS

**Ask:** "On which operating system does the VMS server run that must run the processor: Linux, Windows or both?"

**Cues:** "Ubuntu", "Jetson", "Windows server". If there are none, recommend the host OS.

**Consequences:**
- `Command` and `LogFilepath` per target OS, see "Names". Linux: Linux values in the colocated JSON. Windows: Windows values in the colocated JSON. Both: Linux values in the colocated JSON, Windows values in the README.
- Dependencies: each one must exist for the target OS and CPU.
- Code: build paths with `pathlib` or `os.path`, use no OS-specific calls, and write no fixed `/tmp` path in the code.
- Build and tests run on the host. If the target is not the host, say in the report that nothing is verified on the target, and give the commands to run there (the unit test, the build).

## 12. Name

**Ask:** propose `<side>-<lang>-<short>` from the description, for example `postprocessor-python-vehicle-counter`, and ask for a "yes" or a different short name.

**Consequences:** all other names follow from the directory, see "Names".
