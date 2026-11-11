"""A signal the peer posted before the receiver started waiting must be delivered.

The AI Manager posts a frame and its NOTICE as soon as the processor is READY, or while
the processor is still inside handle_message(); the EXIT it sends at shutdown lands the
same way. `_ipc_wait_for_message()` is called only afterwards, so the signal is already
pending on the input channel when the wait starts. 0f8d1fe drained pending signals at the
top of the receive path and threw exactly these away: the frame was lost and the AI
Manager's send timed out. Draining is only sound before a *send*, where nothing for the
new exchange exists yet; `test_stale_signal_on_output_channel_is_discarded_before_send`
pins that half so the fix for the receive side does not take it down too.
"""

import os
import sys
import unittest
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nxai_postprocessor_sdk import ExternalProcessor, IpcSignal


class _StubDataShm:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def write(self, data):
        return True


class _StubChannel:
    """A channel with signals already posted on it. Each wait() consumes one.

    `reply` is what the peer posts in answer to every signal this side sends, so it
    becomes pending only after signal() -- like a real ACK, it cannot be there earlier.
    """

    def __init__(self, pending_signals, data=None, reply=None):
        self._pending = deque(pending_signals)
        self._reply = reply
        self.data_shm = _StubDataShm(data)
        self.sent = []

    def wait(self, timeout_ms):
        if self._pending:
            return self._pending.popleft()
        return None

    def signal(self, signal):
        self.sent.append(signal)
        if self._reply is not None:
            self._pending.append(self._reply)
        return True


class TestIpcSignalPostedBeforeWait(unittest.TestCase):
    def test_notice_posted_before_wait_is_delivered(self):
        processor = ExternalProcessor()
        processor.input_channel = _StubChannel([IpcSignal.NOTICE], data=b"frame")

        result = processor._ipc_wait_for_message()

        self.assertEqual(result, b"frame")
        self.assertEqual(processor.input_channel.sent, [IpcSignal.ACK])

    def test_exit_posted_before_wait_requests_shutdown(self):
        processor = ExternalProcessor()
        processor.input_channel = _StubChannel([IpcSignal.EXIT])

        result = processor._ipc_wait_for_message()

        self.assertIsNone(result)
        self.assertTrue(processor.exit_requested)

    def test_stale_signal_on_output_channel_is_discarded_before_send(self):
        processor = ExternalProcessor()
        # A leftover from an exchange the peer abandoned; the peer ACKs whatever is sent next.
        processor.output_channel = _StubChannel([IpcSignal.NOTICE], reply=IpcSignal.ACK)

        with self.assertLogs(level="WARNING") as captured:
            sent = processor._ipc_send_response(b"response")

        self.assertTrue(sent)
        self.assertEqual(processor.output_channel.sent, [IpcSignal.NOTICE])
        self.assertTrue(any("Discarded stale signal" in line for line in captured.output))


if __name__ == "__main__":
    unittest.main()
