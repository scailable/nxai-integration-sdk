import os
import sys
import unittest

_nxai_utilities = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "nxai-utilities")
)
if _nxai_utilities not in sys.path:
    sys.path.insert(0, _nxai_utilities)

from message_processing_utils.general.ocr.messages import OcrMessage


class TestOcrMessageContract(unittest.TestCase):
    """`handle` reads logits through extract_logits_array on the pool path.

    A subclass that does not provide it silently disables OCR — no recognitions,
    no log lines — so it must be abstract rather than a `pass` stub.
    """

    def test_subclass_without_extract_logits_array_cannot_be_instantiated(self):
        class Incomplete(OcrMessage):
            def decode_ocr_logits(self, engine, output_name):
                return None

        with self.assertRaises(TypeError) as ctx:
            Incomplete(sdk_message=None)
        self.assertIn("extract_logits_array", str(ctx.exception))

    def test_subclass_implementing_both_can_be_instantiated(self):
        class Complete(OcrMessage):
            def decode_ocr_logits(self, engine, output_name):
                return None

            def extract_logits_array(self, output_name):
                return None

        self.assertIsInstance(Complete(sdk_message=None), OcrMessage)


if __name__ == "__main__":
    unittest.main()
