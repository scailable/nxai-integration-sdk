"""Unit tests for the CLIP prompt-scorer exporter's graph semantics.

The exporter's own `verify()` is the end-to-end oracle, but it downloads the real
checkpoint. These tests pin what the wrapper module itself contributes -- folded-in
mean/std normalization, the int32 token input the preprocessor packs, and a
per-prompt score row whose ranking is the post-processor's only contract --
against stub towers, so a regression is caught without a 600 MB download.
"""

import contextlib
import io
import pathlib
import sys
import types
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from export_clip_prompt_scorer import (  # noqa: E402
    _CLIP_MEAN, _CLIP_STD, ClipPromptScorer, export,
)

_FEATURES = 8
_PROMPTS = 5
_CONTEXT = 77


class _StubClip(torch.nn.Module):
    """Stands in for an open_clip model: records its inputs, returns fixed features."""

    def __init__(self, image_features, text_features):
        super().__init__()
        self.image_features = image_features
        self.text_features = text_features
        self.logit_scale = torch.tensor(0.0)  # exp(0) == 1, so scores are plain cosines.

    def encode_image(self, pixel_values, normalize):
        self.received_pixel_values = pixel_values
        self.image_normalize = normalize
        return self.image_features.to(pixel_values.dtype)

    def encode_text(self, text, normalize):
        self.received_text = text
        self.text_normalize = normalize
        return self.text_features


def _make_scorer(image_features=None, text_features=None, **kwargs):
    image_features = torch.zeros(1, _FEATURES) if image_features is None else image_features
    text_features = torch.zeros(_PROMPTS, _FEATURES) if text_features is None else text_features
    return ClipPromptScorer(_StubClip(image_features, text_features), **kwargs).eval()


class TestClipPromptScorer(unittest.TestCase):
    def setUp(self):
        self.image = torch.rand(1, 3, 2, 2)
        self.text = torch.zeros(_PROMPTS, _CONTEXT, dtype=torch.int32)

    def test_mean_std_normalization_is_folded_into_the_graph(self):
        scorer = _make_scorer()
        with torch.no_grad():
            scorer(self.image, self.text)

        mean = torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD).view(1, 3, 1, 1)
        torch.testing.assert_close(
            scorer.clip.received_pixel_values, (self.image - mean) / std
        )

    def test_int32_tokens_reach_the_text_tower_unwidened(self):
        """int64 tokens trace an ArgMax the engine's runtime cannot implement.

        `encode_text` argmaxes the token tensor to find each prompt's EOT slot,
        and the runtime registers ArgMax for int8/uint8/int32/float/double only.
        Widening the tokens makes the exported model fail to load outright.
        """
        scorer = _make_scorer()
        with torch.no_grad():
            scorer(self.image, self.text)

        self.assertEqual(scorer.clip.received_text.dtype, torch.int32)

    def test_both_towers_are_asked_to_normalize(self):
        """Unnormalized features would make the scores dot products, not cosines."""
        scorer = _make_scorer()
        with torch.no_grad():
            scorer(self.image, self.text)

        self.assertTrue(scorer.clip.image_normalize)
        self.assertTrue(scorer.clip.text_normalize)

    def test_scores_are_one_row_of_per_prompt_cosines(self):
        image_features = torch.zeros(1, _FEATURES)
        image_features[0, 0] = 1.0
        text_features = torch.zeros(_PROMPTS, _FEATURES)
        for prompt in range(_PROMPTS):
            text_features[prompt, 0] = prompt / (_PROMPTS - 1)
        scorer = _make_scorer(image_features, text_features)

        with torch.no_grad():
            scores = scorer(self.image, self.text)

        self.assertEqual(scores.shape, (1, _PROMPTS))
        torch.testing.assert_close(scores[0], text_features[:, 0])
        # Slot order is the contract the preprocessor's padding upholds: score i
        # belongs to prompt i+1, and the post-processor reads it by argmax.
        self.assertEqual(int(scores[0].argmax()), _PROMPTS - 1)

    def test_compute_dtype_stays_inside_the_module(self):
        """The towers may run in fp16, but the module's own boundary stays float32."""
        scorer = _make_scorer(compute_dtype=torch.float16)
        with torch.no_grad():
            scores = scorer(self.image, self.text)

        self.assertEqual(scorer.clip.received_pixel_values.dtype, torch.float16)
        self.assertEqual(scores.dtype, torch.float32)


class TestExportImageInputName(unittest.TestCase):
    """The engine finds the image input by manifest name, so the exported graph
    must name it `image-`, not `image` -- see the comment at `_IMAGE_INPUT_NAME`.
    """

    def test_export_names_the_image_input_with_the_trailing_hyphen(self):
        stub_clip_module = types.SimpleNamespace(
            create_model=lambda *a, **k: _make_scorer().clip
        )
        with mock.patch.dict(sys.modules, {'open_clip': stub_clip_module}), \
                mock.patch('torch.onnx.export') as mock_export, \
                mock.patch('pathlib.Path.stat', return_value=types.SimpleNamespace(st_size=0)), \
                contextlib.redirect_stdout(io.StringIO()):
            export(pathlib.Path('unused.onnx'), 'ViT-B-32', 'laion2b_s34b_b79k', opset=16)

        _, kwargs = mock_export.call_args
        self.assertEqual(kwargs['input_names'], ['image-', 'text'])


if __name__ == '__main__':
    unittest.main()
