#!/usr/bin/env python3
"""Export a CLIP prompt-scorer to an ONNX graph the AI Manager can ingest.

The graph does not embed a crop; it answers "which of the operator's five
prompts fits this frame best" with one score per prompt -- the contract
``preprocessor-python-clip-example`` and ``postprocessor-python-clip-example``
were written against:

- ``preprocessor-python-clip-example.py`` tokenizes exactly
  ``PROMPT_COUNT = 5`` prompts at ``TOKEN_CONTEXT_LENGTH = 77`` and writes them
  into the ``text`` tensor as **int32** token ids, padding empty slots so slot
  ``i`` always means ``prompt{i+1}``. Hence the fixed ``text`` shape ``[5, 77]``
  and the int32 (not the usual int64) input dtype -- the graph must consume the
  bytes the preprocessor actually packs.
- The post-processor only ever takes an ``argmax`` over the per-prompt scores
  (``_argmax_prompt`` in ``postprocessor-python-clip-example.py``), so any
  monotone score works. This emits CLIP's own ``logits_per_image`` (scaled
  cosine), the quantity both towers are trained to rank by.

CLIP's mean/std normalization is baked into the graph, so the image input is a
plain RGB tensor in [0, 1].

Weights come from ``open_clip`` rather than ``transformers``: transformers 5.x
builds its text-tower causal mask in a way the TorchScript exporter cannot trace
(``masking_utils.sdpa_mask`` indexes a scalar), and the dynamo exporter that does
handle it only emits opset 18 with the weights in a sidecar ``.onnx.data`` --
neither of which the catalogue accepts. ``open_clip`` traces cleanly at opset 16
into a single file, and it is the upstream of the LAION weights anyway.

The default checkpoint is LAION's MIT-licensed ViT-B/32 rather than OpenAI's,
whose model card declares *any* deployed use out of scope and surveillance use
out of scope unconditionally, and which ships no license in its HF metadata. Same
architecture and the same BPE vocabulary, so ``instant_clip_tokenizer`` -- the
preprocessor's runtime tokenizer -- tokenizes for it unchanged.

Usage:
    pip install torch open_clip_torch onnxruntime instant-clip-tokenizer
    python tools/export_clip_prompt_scorer.py --out clip_vit_b32_prompt_scorer.onnx
"""

import argparse
import pathlib

import torch

# The CLIP preprocessing constants, shared by every ViT-B/32 CLIP checkpoint.
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
_SIDE = 224
_PROMPT_COUNT = 5
_CONTEXT_LENGTH = 77
_ARCH = 'ViT-B-32'
_PRETRAINED = 'laion2b_s34b_b79k'
# The engine (nxai_preprocessor.cpp) locates the image input of a multi-input
# model only by this literal manifest name, then passes it straight through to
# the OAAX runtime -- an ONNX graph input named plain "image" can never be fed.
_IMAGE_INPUT_NAME = 'image-'


class ClipPromptScorer(torch.nn.Module):
    """Both CLIP towers, scoring one image against a fixed batch of prompts.

    `compute_dtype` is the precision the towers run at; the module's own image
    input and score output stay float32 either way, which is what AIM's ONNX
    requirements ask for. `text` is int32 because that is what the preprocessor
    packs.
    """

    def __init__(self, clip, compute_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.clip = clip
        self.compute_dtype = compute_dtype
        self.register_buffer('mean', torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(_CLIP_STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        pixel_values = ((image - self.mean) / self.std).to(self.compute_dtype)
        image_features = self.clip.encode_image(pixel_values, normalize=True).float()
        # The tokens stay int32 all the way into the text tower. `encode_text`
        # locates each prompt's EOT slot with `text.argmax(dim=-1)`, and the
        # engine's ONNX runtime registers ArgMax for int8/uint8/int32/float/double
        # only -- an int64 token tensor traces an ArgMax the runtime cannot find an
        # implementation for, and the whole model then fails to load ("Could not
        # find an implementation for ArgMax(13) node with name '/ArgMax'"), which
        # takes the entire inference engine down with it. Embedding lookup accepts
        # int32 indices, so nothing needs the wider type.
        text_features = self.clip.encode_text(text, normalize=True).float()
        return self.clip.logit_scale.float().exp() * image_features @ text_features.t()


def export(out_path: pathlib.Path, arch: str, pretrained: str, opset: int) -> None:
    import open_clip

    # Float32 weights put both towers at ~577 MiB, well over the catalogue's
    # 300 MiB upload limit. fp16 halves that and is CLIP's own training precision;
    # casting inside the torch module rather than rewriting the exported graph
    # keeps the fp32/fp16 boundary traced and consistent by construction.
    clip = open_clip.create_model(arch, pretrained=pretrained).eval().half()
    scorer = ClipPromptScorer(clip, compute_dtype=torch.float16).eval()
    # open_clip's text tower is a stock `nn.MultiheadAttention`, whose fused
    # inference kernel traces as `aten::_native_multi_head_attention` -- an op no
    # ONNX opset exports. Disabling the fast path traces the equivalent unfused
    # attention instead.
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.no_grad():
        torch.onnx.export(
            scorer,
            (
                torch.rand(1, 3, _SIDE, _SIDE),
                torch.zeros(_PROMPT_COUNT, _CONTEXT_LENGTH, dtype=torch.int32),
            ),
            str(out_path),
            input_names=[_IMAGE_INPUT_NAME, 'text'],
            output_names=['scores'],
            opset_version=opset,
            dynamo=False,
        )
    print(f'exported {out_path} ({out_path.stat().st_size / 2**20:.1f} MiB)')


def verify(out_path: pathlib.Path, arch: str, pretrained: str) -> None:
    """Assert the graph ranks prompts the way the reference CLIP pipeline does."""
    import instant_clip_tokenizer
    import numpy as np
    import onnxruntime
    import open_clip

    clip = open_clip.create_model(arch, pretrained=pretrained).eval()

    rng = np.random.default_rng(0)
    image_uint8 = rng.integers(0, 256, size=(_SIDE, _SIDE, 3), dtype=np.uint8)
    prompts = ['a photo of a cat', 'a photo of a dog', 'a car', 'random noise', '']

    # Tokenize with the tokenizer the preprocessor actually uses at runtime, so
    # this checks the token ids on the wire and not just the graph in isolation.
    tokens = np.array(
        instant_clip_tokenizer.Tokenizer().tokenize_batch(
            prompts, context_length=_CONTEXT_LENGTH),
        dtype=np.int32,
    )

    # Reference: float32 towers, normalization applied outside the graph.
    image = (image_uint8.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    mean = np.array(_CLIP_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(_CLIP_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    with torch.no_grad():
        image_features = clip.encode_image(
            torch.from_numpy((image - mean) / std), normalize=True)
        text_features = clip.encode_text(
            torch.from_numpy(tokens).long(), normalize=True)
        reference = (clip.logit_scale.exp() * image_features @ text_features.t()).numpy()

    session = onnxruntime.InferenceSession(str(out_path), providers=['CPUExecutionProvider'])
    actual = session.run(['scores'], {_IMAGE_INPUT_NAME: image, 'text': tokens})[0]

    print(f'shape={actual.shape}\nreference={reference[0]}\nactual   ={actual[0]}')
    assert actual.shape == (1, _PROMPT_COUNT), actual.shape
    # The post-processor only argmaxes, so ranking agreement is the contract; the
    # absolute values additionally have to stay within fp16 noise.
    assert int(actual[0].argmax()) == int(reference[0].argmax()), 'prompt ranking differs'
    assert np.allclose(actual[0], reference[0], atol=0.5), 'score drift beyond fp16 noise'
    print('OK: ONNX graph reproduces the reference CLIP prompt ranking')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', default=_ARCH)
    parser.add_argument('--pretrained', default=_PRETRAINED)
    parser.add_argument(
        '--out', type=pathlib.Path, default=pathlib.Path('clip_vit_b32_prompt_scorer.onnx')
    )
    parser.add_argument('--opset', type=int, default=16)
    args = parser.parse_args()

    export(args.out, args.arch, args.pretrained, args.opset)
    verify(args.out, args.arch, args.pretrained)


if __name__ == '__main__':
    main()
