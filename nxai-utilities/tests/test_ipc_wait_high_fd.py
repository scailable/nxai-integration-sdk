"""The Linux wait must work on an eventfd numbered at or above FD_SETSIZE.

The mediaserver hands the processor eventfd numbers from its own descriptor table, and on a
busy server those exceed 1024. `select()` cannot watch such a descriptor and raises
`ValueError: filedescriptor out of range in select()`, which killed every Python processor
on that server. The wait must use `poll()`, which has no such limit.
"""

import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nxai_postprocessor_sdk import IPCSync, IpcSignal

HIGH_FD = 2000


class _StubShm:
    def __init__(self):
        self.buf = bytearray(2)


class _StubSyncShm:
    def __init__(self):
        self.shm = _StubShm()


@unittest.skipUnless(sys.platform.startswith("linux"), "eventfd is Linux-only")
class TestIpcWaitHighFd(unittest.TestCase):
    def setUp(self):
        import resource  # POSIX-only module; the class is skipped elsewhere

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY and hard <= HIGH_FD:
            self.skipTest(f"RLIMIT_NOFILE hard limit {hard} does not allow fd {HIGH_FD}")
        if soft != resource.RLIM_INFINITY and soft <= HIGH_FD:
            resource.setrlimit(resource.RLIMIT_NOFILE, (HIGH_FD + 1, hard))
            self.addCleanup(resource.setrlimit, resource.RLIMIT_NOFILE, (soft, hard))

        eventfd = os.eventfd(0)
        os.dup2(eventfd, HIGH_FD)
        os.close(eventfd)
        self.addCleanup(os.close, HIGH_FD)

        self.sync = IPCSync()
        self.sync.role = "client"
        self.sync.downstream_sync = HIGH_FD
        self.sync.sync_shm = _StubSyncShm()

    def test_signal_on_high_fd_is_delivered(self):
        self.sync.sync_shm.shm.buf[0] = IpcSignal.NOTICE
        os.write(HIGH_FD, struct.pack("<Q", 1))

        self.assertEqual(self.sync.wait(timeout_ms=1000), IpcSignal.NOTICE)

    def test_timeout_on_high_fd_returns_none(self):
        self.assertIsNone(self.sync.wait(timeout_ms=10))


if __name__ == "__main__":
    unittest.main()
