"""Unit tests for the objects-sender business logic.

External behavior only: a built ObjectsPostProcessorMessage goes in, and the
per-object Coordinates attribute and the POST to the HTTP-to-MQTT bridge are
asserted on the way out. No IPC. `requests.request` is patched, so no test makes
a network call.

The reference points of every fixture lie on one affine map from the normalized
frame to degrees, lat = 10 + 2 * x and lon = 20 + 4 * y, so the lat/lon of any
bbox center is known in closed form.
"""

import importlib.util
import json
import os
import sys
import unittest
from unittest import mock

import msgpack

script_dir = os.path.dirname(os.path.abspath(__file__))
# nxai_postprocessor_sdk lives one level up, in nxai-utilities/
sys.path.insert(0, os.path.join(script_dir, "../nxai-utilities"))

from nxai_postprocessor_sdk import ObjectMetadata, ObjectsPostProcessorMessage

spec = importlib.util.spec_from_file_location(
    "postprocessor_python_objects_sender",
    os.path.join(script_dir, "postprocessor-python-objects-sender.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


ECHOED_DEVICE_ID = "cam-echo"
ECHOED_TIMESTAMP = 1726000000000000

# Box centers of the reference points, in normalized frame coordinates.
FOUR_CORNERS = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]


def _lat_lon(cx, cy):
    """The reference map every fixture's points are built on, in degrees."""
    return 10 + 2 * cx, 20 + 4 * cy


def _point_settings(centers, first_index=0, lat_lon=_lat_lon):
    """UserSettings for one reference point per center, as the engine sends them.

    The engine writes every UserSettings value as a string: the BoxFigure value is
    its JSON document, and the spin boxes carry degrees times 1000.
    """
    settings = {}
    for i, (cx, cy) in enumerate(centers, start=first_index):
        lat, lon = lat_lon(cx, cy)
        settings[f"point{i}.figure"] = json.dumps(
            {
                "figure": {
                    "color": "#536dfe",
                    "points": [[cx - 0.02, cy - 0.03], [cx + 0.02, cy + 0.03]],
                },
                "label": "",
                "showOnCamera": False,
            }
        )
        settings[f"point{i}.latitude"] = str(lat * 1000)
        settings[f"point{i}.longitude"] = str(lon * 1000)
    return settings


def _object(cx, cy, track_id="0b8d2f4e-5c1a-4f7e-9d3b-2a6c8e1f0a11"):
    obj = ObjectMetadata()
    obj.x = cx - 0.05
    obj.y = cy - 0.1
    obj.width = 0.1
    obj.height = 0.2
    obj.track_id = track_id
    return obj


def _make_message(objects, user_settings=None, device_id=ECHOED_DEVICE_ID, timestamp=ECHOED_TIMESTAMP):
    msg = ObjectsPostProcessorMessage()
    msg.objects = objects
    msg.user_settings = user_settings or {}
    msg.device_id = device_id
    msg.timestamp = timestamp
    return msg


def _coordinates(obj):
    lat, lon = obj.attributes["Coordinates"].split(", ")
    return float(lat), float(lon)


class _ProcessorTestCase(unittest.TestCase):
    def setUp(self):
        # The per-device homography cache is module state: start every test empty.
        module._homographies.clear()
        module._known_points_cache.clear()
        patcher = mock.patch.object(module.requests, "request")
        self.request = patcher.start()
        self.addCleanup(patcher.stop)


class TestCoordinates(_ProcessorTestCase):
    def test_bbox_center_maps_to_the_lat_lon_of_the_reference_map(self):
        msg = _make_message({"car": [_object(0.5, 0.25)]}, _point_settings(FOUR_CORNERS))

        module.handle_message(msg)

        lat, lon = _coordinates(msg.objects["car"][0])
        self.assertAlmostEqual(11.0, lat, places=5)
        self.assertAlmostEqual(21.0, lon, places=5)

    def test_every_object_of_every_class_gets_coordinates(self):
        msg = _make_message(
            {"car": [_object(0.2, 0.3), _object(0.7, 0.6)], "person": [_object(0.4, 0.8)]},
            _point_settings(FOUR_CORNERS),
        )

        module.handle_message(msg)

        for cls, index, center in [
            ("car", 0, (0.2, 0.3)),
            ("car", 1, (0.7, 0.6)),
            ("person", 0, (0.4, 0.8)),
        ]:
            lat, lon = _coordinates(msg.objects[cls][index])
            expected_lat, expected_lon = _lat_lon(*center)
            self.assertAlmostEqual(expected_lat, lat, places=5)
            self.assertAlmostEqual(expected_lon, lon, places=5)

    def test_more_than_four_points_fit_one_homography(self):
        centers = FOUR_CORNERS + [(0.5, 0.5), (0.3, 0.7)]
        msg = _make_message({"car": [_object(0.6, 0.4)]}, _point_settings(centers))

        module.handle_message(msg)

        lat, lon = _coordinates(msg.objects["car"][0])
        self.assertAlmostEqual(11.2, lat, places=5)
        self.assertAlmostEqual(21.6, lon, places=5)

    def test_a_gap_in_the_point_indexes_is_skipped(self):
        # Point 0 has no figure; points 1..4 are set. The lat/lon of each point
        # belongs to that point, not to its position in the list.
        settings = _point_settings(FOUR_CORNERS, first_index=1)
        settings["point0.figure"] = json.dumps({"figure": None})
        msg = _make_message({"car": [_object(0.5, 0.25)]}, settings)

        module.handle_message(msg)

        lat, lon = _coordinates(msg.objects["car"][0])
        self.assertAlmostEqual(11.0, lat, places=5)
        self.assertAlmostEqual(21.0, lon, places=5)

    def test_fewer_than_four_points_add_no_coordinates(self):
        msg = _make_message({"car": [_object(0.5, 0.5)]}, _point_settings(FOUR_CORNERS[:3]))

        module.handle_message(msg)

        self.assertEqual({}, msg.objects["car"][0].attributes)
        self.assertIsNone(module._homographies[ECHOED_DEVICE_ID])

    def test_no_points_pass_the_message_through_unchanged(self):
        msg = _make_message({"car": [_object(0.5, 0.5)]})

        result = module.handle_message(msg)

        self.assertIs(msg, result)
        self.assertEqual({}, msg.objects["car"][0].attributes)
        self.assertNotIn(ECHOED_DEVICE_ID, module._homographies)


class TestHomographyCache(_ProcessorTestCase):
    def _handle(self, centers, device_id=ECHOED_DEVICE_ID):
        msg = _make_message({"car": [_object(0.5, 0.5)]}, _point_settings(centers), device_id)
        module.handle_message(msg)
        return msg

    def test_unchanged_points_do_not_recompute_the_homography(self):
        with mock.patch.object(module, "compute_homography", wraps=module.compute_homography) as spy:
            self._handle(FOUR_CORNERS)
            self._handle(FOUR_CORNERS)

        self.assertEqual(1, spy.call_count)

    def test_changed_points_recompute_the_homography(self):
        moved = FOUR_CORNERS[:3] + [(0.2, 0.8)]
        with mock.patch.object(module, "compute_homography", wraps=module.compute_homography) as spy:
            self._handle(FOUR_CORNERS)
            msg = self._handle(moved)

        self.assertEqual(2, spy.call_count)
        lat, lon = _coordinates(msg.objects["car"][0])
        self.assertAlmostEqual(11.0, lat, places=5)
        self.assertAlmostEqual(22.0, lon, places=5)

    def test_dropping_below_four_points_removes_the_homography(self):
        self._handle(FOUR_CORNERS)
        msg = self._handle(FOUR_CORNERS[:3])

        self.assertEqual({}, msg.objects["car"][0].attributes)

    def test_each_device_keeps_its_own_homography(self):
        # Device B has the other map: lat = 10 + 2 * y, lon = 20 + 4 * x.
        def swapped(cx, cy):
            return _lat_lon(cy, cx)

        def handle(device_id, lat_lon):
            msg = _make_message(
                {"car": [_object(0.25, 0.75)]},
                _point_settings(FOUR_CORNERS, lat_lon=lat_lon),
                device_id,
            )
            module.handle_message(msg)
            return _coordinates(msg.objects["car"][0])

        with mock.patch.object(module, "compute_homography", wraps=module.compute_homography) as spy:
            coordinates_a = handle("cam-a", _lat_lon)
            coordinates_b = handle("cam-b", swapped)
            coordinates_a_again = handle("cam-a", _lat_lon)

        self.assertEqual(2, spy.call_count, "one fit per device, none on the repeated frame")
        for expected, actual in [
            ((10.5, 23.0), coordinates_a),
            ((11.5, 21.0), coordinates_b),
            ((10.5, 23.0), coordinates_a_again),
        ]:
            self.assertAlmostEqual(expected[0], actual[0], places=5)
            self.assertAlmostEqual(expected[1], actual[1], places=5)

TRACK_A = "0b8d2f4e-5c1a-4f7e-9d3b-2a6c8e1f0a11"
TRACK_B = "7f3e9c1d-2b4a-4e6f-8a0b-1c2d3e4f5a6b"


class TestPost(_ProcessorTestCase):
    def _settings(self, **extra):
        settings = _point_settings(FOUR_CORNERS)
        settings.update(extra)
        return settings

    def test_post_carries_one_geojson_feature_per_object(self):
        msg = _make_message(
            {"car": [_object(0.5, 0.25, TRACK_A)], "person": [_object(0.25, 0.75, TRACK_B)]},
            self._settings(server_address="http://bridge:8080", topic_path="cameras/geo"),
        )

        module.handle_message(msg)

        self.request.assert_called_once()
        args, kwargs = self.request.call_args
        self.assertEqual(("POST", "http://bridge:8080/send"), args)
        self.assertEqual(10, kwargs["timeout"])
        body = kwargs["json"]
        self.assertEqual("cameras/geo", body["topic"])
        self.assertEqual(1, body["QoS"])
        features = body["message"]
        self.assertEqual(2, len(features))
        for feature, cls, track_id, obj in [
            (features[0], "car", TRACK_A, msg.objects["car"][0]),
            (features[1], "person", TRACK_B, msg.objects["person"][0]),
        ]:
            self.assertEqual("Feature", feature["type"])
            self.assertEqual("Point", feature["geometry"]["type"])
            self.assertEqual(
                obj.attributes["Coordinates"].split(", "), feature["geometry"]["coordinates"]
            )
            self.assertEqual(
                {
                    "object_type": cls,
                    "object_id": track_id,
                    "timestamp": str(float(ECHOED_TIMESTAMP)),
                },
                feature["properties"],
            )

    def test_absent_address_and_topic_fall_back_to_the_defaults(self):
        msg = _make_message({"car": [_object(0.5, 0.5)]}, self._settings())

        module.handle_message(msg)

        args, kwargs = self.request.call_args
        self.assertEqual(("POST", "http://127.0.0.1:5000/send"), args)
        self.assertEqual("topic1", kwargs["json"]["topic"])

    def test_no_post_without_objects(self):
        module.handle_message(_make_message({}, self._settings()))

        self.request.assert_not_called()

    def test_no_post_with_fewer_than_four_points(self):
        msg = _make_message({"car": [_object(0.5, 0.5)]}, _point_settings(FOUR_CORNERS[:3]))

        module.handle_message(msg)

        self.request.assert_not_called()

    def test_no_post_without_points(self):
        module.handle_message(_make_message({"car": [_object(0.5, 0.5)]}))

        self.request.assert_not_called()

    def test_request_failures_are_logged_and_the_message_is_still_returned(self):
        http_error = module.requests.exceptions.HTTPError(response=mock.Mock(status_code=503))
        failures = [
            (module.requests.exceptions.ConnectionError(), "Failed to connect to the server."),
            (module.requests.exceptions.Timeout(), "The request timed out."),
            (http_error, "HTTP error occurred: 503"),
            (module.requests.exceptions.RequestException("boom"), "An unexpected request error occurred: boom"),
        ]
        for failure, logged in failures:
            with self.subTest(failure=type(failure).__name__):
                self.request.reset_mock()
                if isinstance(failure, module.requests.exceptions.HTTPError):
                    self.request.return_value.raise_for_status.side_effect = failure
                    self.request.side_effect = None
                else:
                    self.request.side_effect = failure
                msg = _make_message({"car": [_object(0.5, 0.25)]}, self._settings())

                with self.assertLogs(module.logger, level="ERROR") as captured:
                    result = module.handle_message(msg)

                self.assertIs(msg, result)
                self.assertIn("Coordinates", msg.objects["car"][0].attributes)
                self.assertEqual([logged], [record.getMessage() for record in captured.records])


class TestResponseEchoesCorrelationFields(_ProcessorTestCase):
    def test_serialized_response_carries_the_requests_device_id_and_timestamp(self):
        # The AI Manager routes a response back to the camera that asked for it by
        # the key DeviceID + "|" + Timestamp. A response that does not echo both
        # fields back unchanged is dropped silently -- no log line anywhere -- so
        # nothing but this assertion catches a processor that answers with a freshly
        # built message. The comparison is against the serialized frame, which is
        # what the plugin actually reads.
        msg = _make_message({"car": [_object(0.5, 0.25)]}, _point_settings(FOUR_CORNERS))

        result = module.handle_message(msg)

        serialized = msgpack.unpackb(result.to_bytes(), raw=False)
        self.assertEqual(ECHOED_DEVICE_ID, serialized["DeviceID"])
        self.assertEqual(ECHOED_TIMESTAMP, serialized["Timestamp"])


EXIT = object()  # Sentinel script item: simulates IpcSignal.EXIT via exit_requested.
UNPARSEABLE = b"unparseable-frame"  # Script item for which parse_objects_message returns None.


class _ScriptExhausted(BaseException):
    """Raised when the loop asks for more messages than the script holds.

    Derives from BaseException rather than Exception on purpose. run_message_loop wraps
    its body in `except Exception: break`, which would otherwise swallow this guard and
    exit the loop normally -- turning every assertion below into one that cannot fail.
    An EXIT-swallowing regression, exactly the defect these tests exist to catch, would
    then still pass.
    """


class _FakeProcessor:
    """Replays a scripted sequence of wait_for_message() outcomes.

    `calls` counts wait_for_message() invocations, which is what pins down where the
    loop stopped: asserting on exit_requested alone cannot distinguish "returned on
    EXIT" from "kept spinning past it". `sent` collects the serialized responses, so a
    frame that was received but never annotated and returned is visible too.
    """

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0
        self.exit_requested = False
        self.sent = []

    def wait_for_message(self, timeout_ms=None):
        if not self._script:
            raise _ScriptExhausted("wait_for_message called past the end of the script")
        self.calls += 1
        item = self._script.pop(0)
        if item is EXIT:
            self.exit_requested = True
            return None
        if isinstance(item, Exception):
            raise item
        return item

    def parse_objects_message(self, message_bytes):
        if message_bytes == UNPARSEABLE:
            return None
        return _make_message({"car": [_object(0.5, 0.25)]}, _point_settings(FOUR_CORNERS))

    def write_objects_message(self, msg):
        # Stand in for msgpack serialization: hand back the annotated objects so the
        # test can assert the frame really went through handle_message.
        return msg

    def send_response(self, response_bytes):
        self.sent.append(response_bytes)
        return True


class TestMessageLoopSurvivability(_ProcessorTestCase):
    """The engine keeps sending frames; a processor that exits is dropped."""

    def test_exit_signal_ends_the_loop_cleanly(self):
        # A graceful EXIT must return, not be swallowed as a plain timeout that
        # keeps the loop spinning until the AI Manager kills the process. The call
        # count is the assertion that catches the swallow: a loop that treats EXIT
        # as a timeout asks for a second message.
        processor = _FakeProcessor([EXIT])

        with self.assertLogs(module.logger, level="INFO") as captured:
            module.run_message_loop(processor)

        self.assertEqual(1, processor.calls, "loop must return on EXIT, not ask for more")
        self.assertTrue(
            any("graceful exit signal" in record.getMessage() for record in captured.records)
        )

    def test_timeout_does_not_end_the_loop(self):
        processor = _FakeProcessor([None, None, b"frame-bytes", EXIT])

        module.run_message_loop(processor)

        self.assertEqual(4, processor.calls, "loop must survive both timeouts and the frame")
        self.assertTrue(processor.exit_requested)
        # The one frame was actually handled: a loop that skipped it would still
        # reach EXIT with the same call count.
        self.assertEqual(1, len(processor.sent))
        lat, lon = _coordinates(processor.sent[0].objects["car"][0])
        self.assertAlmostEqual(11.0, lat, places=5)
        self.assertAlmostEqual(21.0, lon, places=5)

    def test_unparseable_frame_does_not_end_the_loop(self):
        # A frame the SDK cannot parse is logged and skipped, not fatal.
        processor = _FakeProcessor([UNPARSEABLE, b"frame-bytes", EXIT])

        module.run_message_loop(processor)

        self.assertEqual(3, processor.calls, "loop must survive an unparseable frame")
        self.assertEqual(1, len(processor.sent), "only the parseable frame is answered")
        self.assertTrue(processor.exit_requested)

    def test_raising_message_ends_the_loop(self):
        # Unlike a plain timeout, an unexpected exception from wait_for_message
        # is not survived: the loop's except clause breaks rather than continues.
        processor = _FakeProcessor([RuntimeError("transport hiccup"), EXIT])

        module.run_message_loop(processor)

        self.assertEqual(1, processor.calls, "loop must break on the exception, before EXIT")
        self.assertFalse(processor.exit_requested)


if __name__ == "__main__":
    unittest.main()
