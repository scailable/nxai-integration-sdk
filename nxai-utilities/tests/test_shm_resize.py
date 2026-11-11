"""Growing the output segment must work on Windows too.

`_ipc_send_response` grew the segment through `/dev/shm` + `ftruncate`, unguarded
by any platform check: the last POSIX-only path in the Python SDK's IPC layer.
On Windows `os.open` raised, the exception was swallowed, and the processor
silently failed to answer that frame.

These cases pin the platform-agnostic contract of `SharedMemory.resize`, against
a segment created the way the engine's C++ side creates it -- on Windows a named
`SEC_RESERVE` mapping over the whole `SHM_MAX_SIZE` reservation, of which only a
committed prefix may be touched, so growing it is a `MEM_COMMIT` rather than a
resize.
"""

import ctypes
import os
import sys
import unittest
from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nxai_postprocessor_sdk import (
    HEADER_BYTES,
    IS_WINDOWS,
    SHM_MAX_SIZE,
    SharedMemory,
)

_INITIAL_SIZE = 1024  # What the AI Manager creates its channels at.


class TestSharedMemoryResize(unittest.TestCase):
    def setUp(self):
        self._segment = _EngineSegment(_INITIAL_SIZE)
        self.addCleanup(self._segment.close)

        self.shm = SharedMemory()
        self.assertTrue(self.shm.attach(self._segment.name))
        self.addCleanup(self.shm.detach)

    def test_oversized_write_asks_for_a_resize(self):
        with self.assertRaises(BufferError):
            self.shm.write(b"x" * (self.shm.capacity + 1))

    def test_resize_grows_the_segment_so_the_write_lands(self):
        payload = b"x" * (self.shm.capacity + 1)

        self.assertTrue(self.shm.resize(HEADER_BYTES + len(payload)))

        self.assertGreaterEqual(self.shm.capacity, len(payload))
        self.assertTrue(self.shm.write(payload))
        self.assertEqual(self.shm.read(), payload)

    def test_resize_beyond_the_protocol_maximum_fails(self):
        self.assertFalse(self.shm.resize(SHM_MAX_SIZE + 1))

    def test_resize_of_an_unattached_segment_fails(self):
        self.assertFalse(SharedMemory().resize(HEADER_BYTES + 1))


class _EngineSegment:
    """A segment created the way `nxai_shm_create_random()` creates it."""

    def __init__(self, size):
        if IS_WINDOWS:
            self.name = f"Global_TestSHM_{os.getpid():08X}"
            self._handle = _create_reserved_mapping(self.name, size)
        else:
            self._shm = BuiltinSharedMemory(create=True, size=size)
            self.name = self._shm.name

    def close(self):
        if IS_WINDOWS:
            ctypes.windll.kernel32.CloseHandle(self._handle)
        else:
            self._shm.close()
            self._shm.unlink()


def _create_reserved_mapping(name, committed_size):
    kernel32 = ctypes.windll.kernel32
    PAGE_READWRITE = 0x04
    SEC_RESERVE = 0x4000000
    MEM_COMMIT = 0x1000
    FILE_MAP_ALL_ACCESS = 0xF001F

    kernel32.CreateFileMappingA.restype = ctypes.c_void_p
    handle = kernel32.CreateFileMappingA(
        ctypes.c_void_p(-1),
        None,
        PAGE_READWRITE | SEC_RESERVE,
        0,
        SHM_MAX_SIZE,
        name.encode(),
    )
    if not handle:
        raise OSError(f"CreateFileMappingA failed: {ctypes.get_last_error()}")

    kernel32.MapViewOfFile.restype = ctypes.c_void_p
    view = kernel32.MapViewOfFile(ctypes.c_void_p(handle), FILE_MAP_ALL_ACCESS, 0, 0, 0)
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualAlloc(
        ctypes.c_void_p(view), ctypes.c_size_t(committed_size), MEM_COMMIT, PAGE_READWRITE
    )
    kernel32.UnmapViewOfFile(ctypes.c_void_p(view))
    return handle


if __name__ == "__main__":
    unittest.main()
