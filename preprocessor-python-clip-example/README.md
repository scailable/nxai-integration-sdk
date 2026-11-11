# CLIP Text Preprocessor

## What it does

This preprocessor is meant to be paired with a chained CLIP model. CLIP takes an
image tensor and a text-token tensor and scores how well each prompt matches the
image.

On the `TENSOR` schedule, this preprocessor tokenizes up to five text prompts
(`instant_clip_tokenizer`) and writes the resulting CLIP token ids into the
`msg.tensors["text"]` tensor's `.data`, replacing whatever placeholder bytes
the engine sent. Every other tensor in `msg.tensors` passes through unchanged.

The engine builds input tensors by manifest name and only ever materializes
`image-`, `mask-` and `nms_sensitivity-`, so on a live pipeline the `"text"`
tensor a chained CLIP model declares never arrives. When it is absent, this
preprocessor **appends** a `[5, 77]` int32 `"text"` tensor instead of doing
nothing: the engine rebuilds the whole tensor array from the preprocessor's
response, so an appended tensor does reach the runtime.

## The model this example expects

The example does not adapt to the model it is chained with — it requires a model
with exactly this interface:

| Tensor | Shape | Type | Notes |
|---|---|---|---|
| input `image-` | `[1, 3, 224, 224]` | float32 | RGB in `[0, 1]`; CLIP's mean/std normalization baked into the graph. The trailing hyphen is required: the engine finds a multi-input model's image input only by that literal name |
| input `text` | `[5, 77]` | **int32** | 5 prompt slots × CLIP's 77-token context, written by this preprocessor |
| output `scores` | `[1, 5]` | float32 | one score per prompt slot, higher is better |

`[5, 77]` int32 is a requirement this example places on the model, **not** a
parameter it reads from the tensor: the prompt count and context length are module
constants (`PROMPT_COUNT`, `TOKEN_CONTEXT_LENGTH`), and the registration JSON's
`Repeater` pins the number of prompts an operator can type independently of any
model. A `text` tensor whose shape or type differs is logged and left untouched
rather than filled with tokens the model cannot read. Deriving these from the
tensor instead is not implemented.

Slot order is part of the contract. Empty prompts are tokenized and padded rather
than skipped, so slot `i` always corresponds to the operator's `prompt{i+1}` — and
the paired post-processor maps `scores[i]` back to a prompt purely by that index.

Only the `text` input is this preprocessor's business; what the paired
post-processor does with `scores` is documented in
[`postprocessor-python-clip-example/README.md`](../postprocessor-python-clip-example/README.md).

To bring your own model, either match the table above or edit the two constants
(and the `Repeater` `count`) to match your text tower.

### Reference model

`tools/export_clip_prompt_scorer.py` exports a graph satisfying the table above
from `open_clip`'s LAION ViT-B/32, and verifies the result ranks prompts the same
way the reference float32 CLIP pipeline does:

```shell
pip install torch open_clip_torch onnxruntime instant-clip-tokenizer
python tools/export_clip_prompt_scorer.py --out clip_vit_b32_prompt_scorer.onnx
```

Its module docstring explains the non-obvious choices (int32 rather than int64
tokens, `open_clip` rather than `transformers`, LAION rather than OpenAI weights,
fp16 towers) — read it before adapting the exporter to another checkpoint.

### Registering the model in the AI catalogue

Because the normalization is baked into the graph, the `image-` input wants values
in `[0, 1]`, while the engine feeds float RGB `0..255` and applies
`(raw - mean) / var` before handing the tensor to the model. Register the model
with:

```
means = [0, 0, 0]
vars  = [255, 255, 255]
```

Copying a detector's identity `vars = [1, 1, 1]` is the mistake to avoid: the
model then sees `0..255` pixels, scores are garbage, and nothing logs an error.

## Behavior change from v4

The v4 version read prompts off `ExternalProcessorSettings` under keys
prefixed `externalprocessor.prompt`, sorted lexically, then padded to exactly
five entries. The v5 version reads `msg.user_settings["prompt1"]` through
`msg.user_settings["prompt5"]` directly, defaulting missing
entries to `""`. Same five-prompt, empty-padded behavior — different key
shape.

## Registration

```json
[
    {
        "Name": "CLIP-Text-Prompts",
        "Command": "/path/to/build/preprocessor-python-clip-example/preprocessor-python-clip-example",
        "Schedule": "TENSOR",
        "LogFilepath": "/tmp/preprocess.clip-example.stdout.log",
        "Config": {
            "LogLevel": "DEBUG"
        },
        "UserSettings": [
            {
                "type": "Repeater",
                "count": 5,
                "startIndex": 1,
                "template": {
                    "type": "TextField",
                    "name": "prompt#",
                    "caption": "Prompt #",
                    "description": "Text prompt # to score the frame against",
                    "defaultValue": ""
                }
            }
        ]
    }
]
```

- `Schedule: "TENSOR"` — this preprocessor runs after the AI Manager's own
  tensor preparation, and can add/remove/rename tensors before the model
  runs.
- `UserSettings` — a `Repeater` of five `TextField` controls, `prompt1`
  through `prompt5`, forwarded per-frame as `msg.user_settings`.
- `Config.LogLevel` — passed on the command line as `--LogLevel DEBUG`.

## Build, test, install

```shell
cmake ..
cmake --build . --target preprocessor-python-clip-example
cmake --install . --component preprocessor-python-clip-example
```

```shell
PYTHONPATH=../nxai-utilities python -m unittest test_preprocessor_python_clip_example -v
```

## Licence

Copyright 2026, Network Optix, All rights reserved.
