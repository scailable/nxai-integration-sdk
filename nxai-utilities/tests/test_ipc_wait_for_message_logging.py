"""DEBUG-log size regression test for the IPC read path.

`_ipc_wait_for_message` used to log the full raw channel payload
(`logging.debug(f"Read data: {data}")`). That is harmless for small object
messages but catastrophic for image-carrying channels: a single frame's raw
bytes stringified and written to disk at DEBUG level ballooned a live e2e
run's log file past 5 GB, causing enough I/O pressure that the postprocessor
missed its response-ack deadline.
This pins the fix: the debug log must stay a fixed, small size regardless of
payload size.
"""

import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nxai_postprocessor_sdk import ExternalProcessor, IpcSignal


class _StubDataShm:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data


class _StubInputChannel:
    def __init__(self, data):
        self.data_shm = _StubDataShm(data)

    def wait(self, timeout_ms):
        return IpcSignal.NOTICE

    def signal(self, signal):
        pass


class TestIpcWaitForMessageLogging(unittest.TestCase):
    def test_debug_log_size_does_not_scale_with_payload_size(self):
        processor = ExternalProcessor()
        large_payload = bytes(10_000_000)
        processor.input_channel = _StubInputChannel(large_payload)

        with self.assertLogs(level="DEBUG") as captured:
            result = processor._ipc_wait_for_message()

        self.assertEqual(result, large_payload)
        read_data_lines = [line for line in captured.output if "Read data" in line]
        self.assertEqual(len(read_data_lines), 1)
        self.assertLess(len(read_data_lines[0]), 200)


if __name__ == "__main__":
    unittest.main()
