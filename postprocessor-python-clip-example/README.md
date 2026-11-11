# CLIP Postprocessor

## What it does

This postprocessor is meant to be paired with `preprocessor-python-clip-example`
and a chained CLIP model. CLIP scores an image against up to five text prompts;
this postprocessor picks the winning prompt and surfaces it as an attribute and
an event.

On the `OBJECTS` schedule, with `ReceiveBinaryData: true`, this postprocessor
reads the raw `scores` tensor from `msg.tensors`. The SDK parses the `Tensors`
array of the message into this map. The processor computes an `argmax` over the
non-empty prompt slots configured in `UserSettings`, then:

- caches the winning prompt text, keyed by `OriginalObjectID` (module-level
  `OrderedDict`, FIFO-evicted past 100 entries, no TTL — this is a literal port
  of v4's `objects_attributes`), because the CLIP-scoring message carries
  `OriginalObjectID` but an empty `Objects` map (CLIP is not a detector);
- appends a `clip.event` if the winning prompt differs from the last one for this device, or if
  `event_cooldown_sec` has elapsed since the last identical one.

On a later `OBJECTS` message from the detector node — the same registration
entry runs on every node in the chain — any object
whose track id matches a cached `OriginalObjectID` gets
`obj.attributes["clip_prompt"] = <cached winning prompt text>`.

### The `scores` tensor guard

- **Absent** `Tensors`, or no entry named `scores` — the **normal** case for a
  non-CLIP frame (e.g. the detector node itself never carries a `scores`
  tensor): `debug`-logged, no-op.
- **Present but the wrong shape/dtype** — an operator/model-contract mismatch:
  `error`-logged with the expected vs. actual shape and dtype, no-op.
- **Present and matching the contract** (`[1, 5]` float32) — scored normally.

### Empty prompt slots

A `promptN` slot left blank in the settings dialog can never win the `argmax`,
even if the model happens to score that slot's fixed padding highest — matching
v4 exactly (v4's engine-computed `Scores` map never contained an entry for a
slot outside the operator's typed prompts in the first place).

## The model this example expects

| Tensor | Shape | Type | Notes |
|---|---|---|---|
| input `image-` | `[1, 3, 224, 224]` | float32 | RGB in `[0, 1]` |
| input `text` | `[5, 77]` | int32 | written by `preprocessor-python-clip-example` |
| output `scores` | `[1, 5]` | **float32** | one score per prompt slot, higher is better — what this postprocessor reads |

See `preprocessor-python-clip-example/README.md` for the `image`/`text` side of
the contract and `tools/export_clip_prompt_scorer.py` for a reference model
satisfying this table (LAION CLIP ViT-B/32, exported with `logits_per_image` as
`scores`).

**Prompt order matters.** This postprocessor's `prompt1`..`prompt5` `UserSettings`
must contain the **same prompt texts, in the same order**, as the paired
preprocessor's `prompt1`..`prompt5`. v5 matches `scores[i]` back to a prompt
purely by index — there is no name carried on the wire like v4's `Scores` map
keys. A prompt typed into one dialog but not the other (or in a different slot)
is scored silently against the wrong text; nothing detects the divergence.

## Behavior change from v4

- **Scoring moves processor-side.** v4's engine computed the `argmax` over an
  engine-delivered `Scores: {name: score}` map. v5 has no `Scores` key; the
  `argmax` is computed here, over the raw `scores` tensor.
- **`event_cooldown_sec`** is a `UserSettings` `DoubleSpinBox` (replacing v4's
  `externalprocessor.eventCooldown`), and the five prompts are a `UserSettings`
  `Repeater` (replacing v4's `externalprocessor.prompt*` keys) — matching the
  paired preprocessor's dialog style.
- **Attribute key.** v4 wrote the winning prompt text as *both* the attribute
  key and its value (`AttributeKeys[i].append(prompt)` /
  `AttributeValues[i].append(prompt)`). v5's `attributes` is a `dict[str, str]`
  rendered verbatim in the object details panel, so a literal port would make
  the attribute *name* change every time an operator edits a prompt. This
  postprocessor instead writes a stable key, `obj.attributes["clip_prompt"]`,
  with the winning prompt text as the value — a deliberate, narrow departure
  from v4, whose behavior is not expressible in the v5 attribute type
  without an observably worse UI result.
- The emitted event drops v4's `Caption` field (v5 `EventMetadata` carries only
  `type_id` + `description`) and carries the `clip.event` id registered in
  `external_postprocessors.json` verbatim.

## Registration

```json
[
    {
        "Name": "CLIP-Post-Example",
        "Command": "/path/to/build/postprocessor-python-clip-example/postprocessor-python-clip-example",
        "Schedule": "OBJECTS",
        "ReceiveBinaryData": true,
        "LogFilepath": "/tmp/postprocess.clip-example.stdout.log",
        "Config": {
            "LogLevel": "DEBUG"
        },
        "UserSettings": [
            {
                "type": "DoubleSpinBox",
                "name": "event_cooldown_sec",
                "caption": "Event Cooldown (seconds)",
                "description": "Minimum time between repeated CLIP prompt-recognized events for the same winning prompt on a device",
                "defaultValue": 0
            },
            {
                "type": "Repeater",
                "count": 5,
                "startIndex": 1,
                "template": {
                    "type": "TextField",
                    "name": "prompt#",
                    "caption": "Prompt #",
                    "description": "Text prompt # to score the frame against. Must match preprocessor-python-clip-example's prompt # in the same order.",
                    "defaultValue": ""
                }
            }
        ],
        "Events": [
            {
                "ID": "clip.event",
                "Name": "CLIP prompt recognized"
            }
        ]
    }
]
```

- `Schedule: "OBJECTS"` + `ReceiveBinaryData: true` — this postprocessor runs on
  every node of the pipeline (detector and chained CLIP node alike) and needs
  the raw `Tensors` array on each message.
- `UserSettings` — a `DoubleSpinBox` (`event_cooldown_sec`) and a `Repeater` of
  five `TextField` controls, `prompt1` through `prompt5`.
- `Events` — the id `clip.event`, which the code emits verbatim.

## Build, test, install

```shell
cmake ..
cmake --build . --target postprocessor-python-clip-example
cmake --install . --component postprocessor-python-clip-example
```

Run the tests from the `postprocessor-python-clip-example/` directory:

```shell
PYTHONPATH=../nxai-utilities python -m unittest test_postprocessor_python_clip_example -v
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
