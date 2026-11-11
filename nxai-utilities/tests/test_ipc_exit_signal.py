"""EXIT vs timeout must be distinguishable by callers.

`_ipc_wait_for_message` returns `None` for a graceful `IpcSignal.EXIT`, the same
value it returns for an ordinary timeout. A processor's message loop treats
`None` as "keep waiting" (`continue`), so today it cannot tell the two apart
and never runs its shutdown path on EXIT. This pins the additive fix: an
`exit_requested` flag on `ExternalProcessor` that gets set when EXIT is
observed, so a loop can check it alongside the existing `None` check.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nxai_postprocessor_sdk import ExternalProcessor, IpcSignal


class _StubInputChannel:
    def __init__(self, signal):
        self._signal = signal

    def wait(self, timeout_ms):
        return self._signal

    def signal(self, signal):
        pass


class TestIpcExitSignal(unittest.TestCase):
    def test_exit_signal_sets_exit_requested_flag(self):
        processor = ExternalProcessor()
        processor.input_channel = _StubInputChannel(IpcSignal.EXIT)

        result = processor._ipc_wait_for_message()

        self.assertIsNone(result)
        self.assertTrue(processor.exit_requested)

    def test_timeout_does_not_set_exit_requested_flag(self):
        processor = ExternalProcessor()
        processor.input_channel = _StubInputChannel(None)

        result = processor._ipc_wait_for_message()

        self.assertIsNone(result)
        self.assertFalse(processor.exit_requested)


if __name__ == "__main__":
    unittest.main()
