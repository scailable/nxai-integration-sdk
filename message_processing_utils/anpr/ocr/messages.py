"""
ANPR OCR message (CCT model).
"""

import logging
from message_processing_utils.general.ocr.messages import OcrMessage


logger = logging.getLogger(__name__)


class CctOcrMessage(OcrMessage):
    """OCR message for CCT model (NxM logits)."""

    def __init__(self, sdk_message, raw_tensors=None, expected_logits_shape=(9, 37)):
        self.expected_logits_shape = expected_logits_shape  # (N characters, M classes)
        super().__init__(sdk_message, raw_tensors)

    def decode_ocr_logits(self, engine, output_name):
        """
        Decode OCR inference results from CCT model.
        Expected input shape is defined by expected_logits_shape.
        """
        ocr_logits = self.extract_logits_array(output_name)
        return engine.decode_logits(ocr_logits)

    def extract_logits_array(self, output_name):
        ocr_logits = None
        binary_logits = self.get_binary_output_f32(output_name)
        expected_size = self.expected_logits_shape[0] * self.expected_logits_shape[1]
        if binary_logits is None:
            ocr_logits = None
        elif binary_logits.size != expected_size:
            logger.warning(
                "Tensors %s size %d does not match expected %s",
                output_name, binary_logits.size, self.expected_logits_shape
            )
            ocr_logits = None
        else:
            ocr_logits = binary_logits.reshape(self.expected_logits_shape)
        return ocr_logits
