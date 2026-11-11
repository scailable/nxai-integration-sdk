"""initialize_from_args must take the processor's name from --ProcessorName.

The v5 engine never passes the name positionally: it appends the
registration entry's Config as --key value pairs, then --ProcessorName
<name>. The old code read args[2]
unconditionally, which is always a flag, not the name.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nxai_postprocessor_sdk import ExternalProcessor


class TestInitializeFromArgsProcessorName(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(
            ExternalProcessor, "_initialize_from_keys", return_value=True
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_name_parsed_from_processor_name_flag_with_config(self):
        processor = ExternalProcessor()
        args = [
            "input_sync",
            "output_sync",
            "--LogLevel",
            "DEBUG",
            "--ProcessorName",
            "Events-Example",
        ]

        self.assertTrue(processor.initialize_from_args(args))
        self.assertEqual(processor.name, "Events-Example")

    def test_name_parsed_from_processor_name_flag_without_config(self):
        processor = ExternalProcessor()
        args = ["input_sync", "output_sync", "--ProcessorName", "X"]

        self.assertTrue(processor.initialize_from_args(args))
        self.assertEqual(processor.name, "X")

    def test_falls_back_to_constructor_name_when_flag_absent(self):
        processor = ExternalProcessor(name="Constructor-Name")
        args = ["input_sync", "output_sync"]

        self.assertTrue(processor.initialize_from_args(args))
        self.assertEqual(processor.name, "Constructor-Name")

    def test_falls_back_to_unknown_when_flag_and_constructor_name_absent(self):
        processor = ExternalProcessor()
        args = ["input_sync", "output_sync"]

        self.assertTrue(processor.initialize_from_args(args))
        self.assertEqual(processor.name, "unknown")

    def test_port_mode_is_unaffected(self):
        processor = ExternalProcessor()
        with patch.object(
            ExternalProcessor, "_initialize_from_tcp", return_value=True
        ) as tcp_init:
            self.assertTrue(processor.initialize_from_args(["--port", "9999"]))
        tcp_init.assert_called_once_with(9999)


if __name__ == "__main__":
    unittest.main()
