"""Tests for the single `ExternalProcessor` class and its role-specific aliases.

The high-level SDK endpoint is one concrete class `ExternalProcessor`;
`ExternalPostprocessor` and `ExternalPreprocessor` are role-specific names for it.
These tests pin that contract.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nxai_postprocessor_sdk
from nxai_postprocessor_sdk import (
    ExternalPostprocessor,
    ExternalPreprocessor,
    ExternalProcessor,
)


class TestExternalProcessorAliases(unittest.TestCase):
    def test_external_processor_is_a_concrete_class(self):
        processor = ExternalProcessor()
        self.assertFalse(processor.is_initialized)

    def test_postprocessor_alias_is_the_same_class(self):
        self.assertIs(ExternalPostprocessor, ExternalProcessor)

    def test_preprocessor_alias_is_the_same_class(self):
        self.assertIs(ExternalPreprocessor, ExternalProcessor)

    def test_module_exposes_external_processor(self):
        self.assertIs(nxai_postprocessor_sdk.ExternalProcessor, ExternalProcessor)


if __name__ == "__main__":
    unittest.main()
