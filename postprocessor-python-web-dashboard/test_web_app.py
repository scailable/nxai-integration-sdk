"""Unit tests for the web app's shutdown path.

The web app is the standalone companion program: an HTTP server thread plus a
periodic flush thread, both daemons, torn down by SIGTERM/SIGINT. These tests
cover that teardown — the signal handler, the flush thread's final flush, and
the bounded wait that lets that flush finish before the interpreter exits.
No dashboard rendering, no SQLite schema, no request handling.
"""

import atexit
import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

script_dir = os.path.dirname(os.path.abspath(__file__))

# The module writes its log file next to sys.argv[0]; under a test runner that
# would be the runner's own directory, so point it at a scratch directory that
# is discarded when the interpreter exits.
_scratch = tempfile.mkdtemp(prefix="web-app-tests-")
atexit.register(shutil.rmtree, _scratch, ignore_errors=True)

_saved_argv0 = sys.argv[0]
sys.argv[0] = os.path.join(_scratch, "web_app.py")
try:
    spec = importlib.util.spec_from_file_location(
        "web_app", os.path.join(script_dir, "web_app.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.argv[0] = _saved_argv0


class _RecordingStore:
    """Stands in for DetectionStore: records every flush, optionally fails."""

    def __init__(self, raises=False, on_flush=None):
        self.raises = raises
        self.on_flush = on_flush
        self.flushes = 0

    def flush_meta(self):
        self.flushes += 1
        if self.on_flush:
            self.on_flush()
        if self.raises:
            raise RuntimeError("flush failed")


class _WebAppTestCase(unittest.TestCase):
    """Isolates the module's globals; joins whatever a test started."""

    def setUp(self):
        module.shutdown_event.clear()
        self._saved_store = module.store
        self._saved_server = module.web_server
        self._saved_logger = module.logger
        module.logger = logging.getLogger("web-app-tests")
        module.logger.addHandler(logging.NullHandler())
        module.logger.propagate = False
        self.threads = []

    def tearDown(self):
        module.shutdown_event.set()
        for thread in self.threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive(), "test left a thread running")
        module.shutdown_event.clear()
        module.store = self._saved_store
        module.web_server = self._saved_server
        module.logger = self._saved_logger


class TestSignalHandler(_WebAppTestCase):
    def test_sets_the_shutdown_event(self):
        module.web_server = None
        module.signal_handler(module.signal.SIGTERM, None)
        self.assertTrue(module.shutdown_event.is_set())

    def test_shuts_the_web_server_down(self):
        class _Server:
            def __init__(self):
                self.shutdowns = 0

            def shutdown(self):
                self.shutdowns += 1

        module.web_server = _Server()
        module.signal_handler(module.signal.SIGINT, None)
        self.assertEqual(module.web_server.shutdowns, 1)

    def test_survives_a_signal_before_the_server_is_bound(self):
        module.web_server = None
        module.signal_handler(module.signal.SIGTERM, None)  # must not raise
        self.assertTrue(module.shutdown_event.is_set())


class TestStartFlushThread(_WebAppTestCase):
    def test_returns_a_started_daemon_thread(self):
        module.store = _RecordingStore()
        thread = module.start_flush_thread(60)
        self.threads.append(thread)
        self.assertIsInstance(thread, threading.Thread)
        self.assertTrue(thread.is_alive())
        self.assertTrue(thread.daemon)

    def test_flushes_once_more_on_shutdown(self):
        module.store = _RecordingStore()
        thread = module.start_flush_thread(60)  # never fires periodically
        module.shutdown_event.set()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(module.store.flushes, 1)

    def test_a_raising_final_flush_is_not_fatal(self):
        module.store = _RecordingStore(raises=True)
        thread = module.start_flush_thread(60)
        module.shutdown_event.set()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(module.store.flushes, 1)

    def test_a_raising_periodic_flush_keeps_the_thread_alive(self):
        flushed = threading.Event()
        module.store = _RecordingStore(raises=True, on_flush=flushed.set)
        thread = module.start_flush_thread(0.01)
        self.threads.append(thread)
        self.assertTrue(flushed.wait(timeout=5))
        self.assertTrue(thread.is_alive())


class TestWaitForShutdown(_WebAppTestCase):
    def test_returns_only_once_the_shutdown_event_is_set(self):
        module.store = _RecordingStore()
        thread = module.start_flush_thread(60)
        returned = threading.Event()

        waiter = threading.Thread(
            target=lambda: (module.wait_for_shutdown(thread), returned.set()),
            daemon=True,
        )
        waiter.start()
        self.threads.append(waiter)
        self.assertFalse(returned.wait(timeout=0.2))
        module.shutdown_event.set()
        self.assertTrue(returned.wait(timeout=5))

    def test_the_final_flush_has_completed_before_it_returns(self):
        # The flush takes long enough that a shutdown which does not wait for
        # the thread would return while it is still in progress.
        module.store = _RecordingStore(on_flush=lambda: time.sleep(0.3))
        thread = module.start_flush_thread(60)
        module.shutdown_event.set()

        module.wait_for_shutdown(thread)

        self.assertFalse(thread.is_alive())
        self.assertEqual(module.store.flushes, 1)

    def test_the_wait_for_the_flush_thread_is_bounded(self):
        blocked = threading.Event()
        thread = threading.Thread(target=blocked.wait, daemon=True, name="stuck")
        thread.start()
        module.shutdown_event.set()
        started = time.monotonic()
        try:
            module.wait_for_shutdown(thread, join_timeout=0.1)
            elapsed = time.monotonic() - started
            # It gave up on a flush thread that never finishes, rather than
            # blocking the process shutdown on it.
            self.assertTrue(thread.is_alive())
            self.assertLess(elapsed, 2)
        finally:
            blocked.set()
        thread.join(timeout=5)


class TestStartWebServer(_WebAppTestCase):
    def test_serves_on_a_daemon_thread(self):
        module.start_web_server(0)  # ephemeral port
        server = module.web_server
        self.addCleanup(server.server_close)
        http = [t for t in threading.enumerate() if t.name == "http"]
        try:
            self.assertEqual(len(http), 1)
            self.assertTrue(http[0].daemon)
        finally:
            server.shutdown()
            for thread in http:
                thread.join(timeout=5)


class TestConfig(_WebAppTestCase):
    def test_defaults_without_flags(self):
        args = module.parse_args([])
        self.assertEqual(
            module.config(args),
            (module.DEFAULT_PORT, module.DEFAULT_TIMELINE_CAP,
             module.DEFAULT_SCATTER_CAP, module.DEFAULT_DB),
        )

    def test_flags_set_every_value(self):
        args = module.parse_args([
            "--port", "9000", "--db", "/tmp/x.db",
            "--timeline-capacity", "100", "--scatter-capacity", "20",
        ])
        self.assertEqual(module.config(args), (9000, 100, 20, "/tmp/x.db"))

    def test_out_of_range_values_fall_back_to_defaults(self):
        args = module.parse_args([
            "--port", "70000", "--timeline-capacity", "0", "--scatter-capacity", "2000000",
        ])
        port, tc, sc, _ = module.config(args)
        self.assertEqual(port, module.DEFAULT_PORT)
        self.assertEqual(tc, module.DEFAULT_TIMELINE_CAP)
        self.assertEqual(sc, module.DEFAULT_SCATTER_CAP)

    def test_no_ini_file_is_read(self):
        self.assertFalse(hasattr(module, "CONFIG_FILE"))


if __name__ == "__main__":
    unittest.main()
