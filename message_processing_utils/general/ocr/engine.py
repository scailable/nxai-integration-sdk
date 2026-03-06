"""
OCR engine for decoding logits into text.
"""

import logging
import numpy as np


logger = logging.getLogger(__name__)


def _calculate_geometric_mean(*confidences: float) -> float:
    """Return geometric mean of per-character confidences, or 0.0 if none."""
    if not confidences:
        return 0.0
    product = 1.0
    for c in confidences:
        product *= float(c)
    return product ** (1.0 / len(confidences))


class LogitsOcrEngine:

    def __init__(self, expected_logits_shape, char_map: str):
        self._expected_logits_shape = expected_logits_shape
        self._char_map = char_map
        self._validate_config()

    def _validate_config(self):
        if (
            not isinstance(self._expected_logits_shape, tuple)
            or len(self._expected_logits_shape) != 2
        ):
            raise ValueError(
                f"expected_logits_shape must be (rows, cols), got {self._expected_logits_shape}"
            )
        if self._expected_logits_shape[0] <= 0 or self._expected_logits_shape[1] <= 0:
            raise ValueError(
                f"expected_logits_shape must be positive, got {self._expected_logits_shape}"
            )
        if not isinstance(self._char_map, str) or not self._char_map:
            raise ValueError("char_map must be a non-empty string")
        if len(self._char_map) != self._expected_logits_shape[1]:
            raise ValueError(
                f"char_map length {len(self._char_map)} does not match "
                f"expected_logits_shape[1] {self._expected_logits_shape[1]}"
            )

    def apply(self, processed_image) -> str:
        return self.decode_logits(np.array(processed_image)) or ""

    def decode_logits(self, ocr_logits):
        if ocr_logits is None:
            logger.warning("No OCR logits found in BinaryOutputs or InferenceData")
            return None
        logger.debug("OCR logits shape: %s", ocr_logits.shape)
        logger.debug("OCR logits dtype: %s", ocr_logits.dtype)
        logger.debug("OCR logits sample: %s", ocr_logits.flatten()[:20])
        if ocr_logits.shape != self._expected_logits_shape:
            logger.warning(
                "Unexpected OCR logits shape: %s, expected %s",
                ocr_logits.shape, self._expected_logits_shape
            )
            return None
        recognized_text = ""
        confidences = []
        for i in range(self._expected_logits_shape[0]):
            char_logits = ocr_logits[i]
            char_index = np.argmax(char_logits)
            confidence = char_logits[char_index]

            if char_index < len(self._char_map):
                char = self._char_map[char_index]
            else:
                char = "?"

            recognized_text += char
            confidences.append(float(confidence))

            logger.debug(
                "Position %d: '%s' (index %d, confidence %.4f)",
                i, char, char_index, confidence
            )
        recognized_text = recognized_text.rstrip()
        overall_confidence = _calculate_geometric_mean(*confidences)
        logger.debug("Recognized text: '%s'", recognized_text)
        logger.debug("Confidences: %s", confidences)
        return recognized_text, overall_confidence
