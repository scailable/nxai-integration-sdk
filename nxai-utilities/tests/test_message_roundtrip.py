"""Round-trip serialization tests for the Python postprocessor SDK message types.

Each message class owns a msgpack `to_bytes()`/`from_bytes()` pair. These tests
build a populated message, serialize it, deserialize it, and assert the result
is equal field-by-field — validating the msgpack foundation that every
processor and the test harness depend on.
"""

import os
import sys
import unittest

import msgpack

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nxai_postprocessor_sdk import (
    DATA_TYPE_FLOAT,
    DATA_TYPE_UINT8,
    BestShotMetadata,
    EventMetadata,
    ImageData,
    ImageOrder,
    ImagePreprocessorMessage,
    ObjectMetadata,
    ObjectsPostProcessorMessage,
    TensorMessage,
    TensorMetadata,
)


class TestObjectsPostProcessorMessageRoundTrip(unittest.TestCase):
    def test_populated_message_round_trips(self):
        msg = ObjectsPostProcessorMessage()
        msg.timestamp = 1717334400
        msg.device_id = "device-42"
        msg.device_name = "Front Door Camera"

        obj = ObjectMetadata()
        obj.x = 0.1
        obj.y = 0.2
        obj.width = 0.3
        obj.height = 0.4
        obj.confidence = 0.95
        obj.attributes = {"source": "postprocessor-example"}
        msg.objects = {"person": [obj]}

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(restored.timestamp, msg.timestamp)
        self.assertEqual(restored.device_id, msg.device_id)
        self.assertEqual(restored.device_name, msg.device_name)
        self.assertEqual(list(restored.objects.keys()), ["person"])

        restored_obj = restored.objects["person"][0]
        self.assertEqual(restored_obj.x, obj.x)
        self.assertEqual(restored_obj.y, obj.y)
        self.assertEqual(restored_obj.width, obj.width)
        self.assertEqual(restored_obj.height, obj.height)
        self.assertEqual(restored_obj.confidence, obj.confidence)
        self.assertEqual(restored_obj.attributes, obj.attributes)

    def test_track_id_round_trips_when_present(self):
        msg = ObjectsPostProcessorMessage()
        obj = ObjectMetadata()
        obj.track_id = "550e8400-e29b-41d4-a716-446655440000"
        msg.objects = {"car": [obj]}

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(restored.objects["car"][0].track_id, obj.track_id)

    def test_track_id_absent_when_not_set(self):
        msg = ObjectsPostProcessorMessage()
        msg.objects = {"car": [ObjectMetadata()]}

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertIsNone(restored.objects["car"][0].track_id)

    def test_new_track_id_round_trips_when_present(self):
        msg = ObjectsPostProcessorMessage()
        obj = ObjectMetadata()
        obj.new_track_id = "111e8400-e29b-41d4-a716-446655440aaa"
        msg.objects = {"car": [obj]}

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(restored.objects["car"][0].new_track_id, obj.new_track_id)

    def test_new_track_id_absent_when_not_set(self):
        msg = ObjectsPostProcessorMessage()
        msg.objects = {"car": [ObjectMetadata()]}

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertIsNone(restored.objects["car"][0].new_track_id)

    def test_events_round_trip(self):
        msg = ObjectsPostProcessorMessage()
        event = EventMetadata()
        event.type_id = "intrusion"
        event.description = "Person entered restricted zone"
        msg.events = [event]

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(len(restored.events), 1)
        self.assertEqual(restored.events[0].type_id, event.type_id)
        self.assertEqual(restored.events[0].description, event.description)

    def test_empty_message_round_trips(self):
        msg = ObjectsPostProcessorMessage()

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(restored.timestamp, 0)
        self.assertEqual(restored.device_id, "")
        self.assertEqual(restored.device_name, "")
        self.assertEqual(restored.objects, {})
        self.assertEqual(restored.events, [])

    def test_multiple_classes_round_trip(self):
        msg = ObjectsPostProcessorMessage()
        msg.objects = {
            "person": [ObjectMetadata()],
            "car": [ObjectMetadata()],
        }

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(set(restored.objects.keys()), {"person", "car"})

    def test_multiple_objects_per_class_round_trip(self):
        msg = ObjectsPostProcessorMessage()
        first = ObjectMetadata()
        first.confidence = 0.5
        second = ObjectMetadata()
        second.confidence = 0.9
        msg.objects = {"person": [first, second]}

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(len(restored.objects["person"]), 2)
        self.assertEqual(restored.objects["person"][0].confidence, 0.5)
        self.assertEqual(restored.objects["person"][1].confidence, 0.9)

    def test_user_settings_parsed_from_root_map(self):
        # The engine forwards GUI settings as a root `UserSettings` map,
        # a flat {settingName: stringValue} sibling of Timestamp/Objects/Events.
        inbound = msgpack.packb(
            {
                "Timestamp": 1717334400,
                "DeviceID": "device-42",
                "DeviceName": "Front Door Camera",
                "Objects": {},
                "Events": [],
                "UserSettings": {"attributeName": "reviewed", "attributeValue": "true"},
            },
            use_bin_type=True,
        )

        restored = ObjectsPostProcessorMessage.from_bytes(inbound)

        self.assertEqual(
            restored.user_settings,
            {"attributeName": "reviewed", "attributeValue": "true"},
        )

    def test_user_settings_empty_when_absent(self):
        msg = ObjectsPostProcessorMessage()

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(restored.user_settings, {})

    def test_to_bytes_never_emits_user_settings(self):
        # UserSettings is inbound-only: the processor reads forwarded settings but
        # never echoes them back, so the outbound root map keeps its five keys.
        msg = ObjectsPostProcessorMessage()
        msg.user_settings = {"attributeName": "reviewed", "attributeValue": "true"}

        root = msgpack.unpackb(msg.to_bytes(), raw=False)

        self.assertNotIn("UserSettings", root)
        self.assertEqual(
            set(root.keys()),
            {"Timestamp", "DeviceID", "DeviceName", "Objects", "Events"},
        )

    def test_image_data_parsed_from_root_map(self):
        # ReceiveInputImage delivers the frame as a root `ImageData` map (same
        # shape as ImagePreprocessorMessage's), a sibling of Timestamp/Objects/Events.
        inbound = msgpack.packb(
            {
                "Timestamp": 1717334400,
                "DeviceID": "device-42",
                "DeviceName": "Front Door Camera",
                "Objects": {},
                "Events": [],
                "ImageData": {
                    "Height": 2,
                    "Width": 2,
                    "Layout": int(ImageOrder.CHW),
                    "Data": bytes([10, 20, 30, 40]),
                },
            },
            use_bin_type=True,
        )

        restored = ObjectsPostProcessorMessage.from_bytes(inbound)

        self.assertIsNotNone(restored.image_data)
        self.assertEqual(restored.image_data.height, 2)
        self.assertEqual(restored.image_data.width, 2)
        self.assertEqual(restored.image_data.layout, ImageOrder.CHW)
        self.assertEqual(restored.image_data.bytes, bytes([10, 20, 30, 40]))

    def test_image_data_none_when_absent(self):
        msg = ObjectsPostProcessorMessage()

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertIsNone(restored.image_data)

    def test_to_bytes_never_emits_image_data(self):
        # ImageData is inbound-only on the objects message: the processor reads
        # the frame the engine sent but never echoes it back.
        msg = ObjectsPostProcessorMessage()
        image = ImageData()
        image.height = 2
        image.width = 2
        image.bytes = bytes([1, 2, 3, 4])
        msg.image_data = image

        root = msgpack.unpackb(msg.to_bytes(), raw=False)

        self.assertNotIn("ImageData", root)
        self.assertEqual(
            set(root.keys()),
            {"Timestamp", "DeviceID", "DeviceName", "Objects", "Events"},
        )

    def test_original_object_id_parsed_from_root_map(self):
        # A chained (FEATURE_EXTRACTION) inference's output
        # message carries the parent object's track_id as a root `OriginalObjectID`
        # string, a sibling of Timestamp/Objects/Events.
        inbound = msgpack.packb(
            {
                "Timestamp": 1717334400,
                "DeviceID": "device-42",
                "DeviceName": "Front Door Camera",
                "Objects": {},
                "Events": [],
                "OriginalObjectID": "550e8400-e29b-41d4-a716-446655440000",
            },
            use_bin_type=True,
        )

        restored = ObjectsPostProcessorMessage.from_bytes(inbound)

        self.assertEqual(
            restored.original_object_id, "550e8400-e29b-41d4-a716-446655440000"
        )

    def test_original_object_id_none_when_absent(self):
        msg = ObjectsPostProcessorMessage()

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertIsNone(restored.original_object_id)

    def test_to_bytes_never_emits_original_object_id(self):
        # OriginalObjectID is inbound-only: it identifies the parent for a
        # chained model's output, and is never echoed back on write.
        msg = ObjectsPostProcessorMessage()
        msg.original_object_id = "550e8400-e29b-41d4-a716-446655440000"

        root = msgpack.unpackb(msg.to_bytes(), raw=False)

        self.assertNotIn("OriginalObjectID", root)
        self.assertEqual(
            set(root.keys()),
            {"Timestamp", "DeviceID", "DeviceName", "Objects", "Events"},
        )


class TestSkipFeatureExtractionRoundTrip(unittest.TestCase):
    """skipFeatureExtraction (v5 engine opt-out): an optional per-object
    bool gating whether the FEATURE_EXTRACTION chained stage runs for that object.
    Absent on the wire means the engine's default (do not skip); when the
    postprocessor sets it, the explicit bool is emitted."""

    def test_skip_feature_extraction_true_round_trips(self):
        msg = ObjectsPostProcessorMessage()
        obj = ObjectMetadata()
        obj.skip_feature_extraction = True
        msg.objects = {"person": [obj]}

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertIs(restored.objects["person"][0].skip_feature_extraction, True)

    def test_skip_feature_extraction_false_round_trips(self):
        # Explicit False is a distinct signal from unset: it must survive the
        # round trip as False, not collapse to None/absent.
        msg = ObjectsPostProcessorMessage()
        obj = ObjectMetadata()
        obj.skip_feature_extraction = False
        msg.objects = {"person": [obj]}

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertIs(restored.objects["person"][0].skip_feature_extraction, False)

    def test_skip_feature_extraction_false_emitted_on_the_wire(self):
        msg = ObjectsPostProcessorMessage()
        obj = ObjectMetadata()
        obj.skip_feature_extraction = False
        msg.objects = {"person": [obj]}

        root = msgpack.unpackb(msg.to_bytes(), raw=False)

        self.assertIn("skipFeatureExtraction", root["Objects"]["person"][0])
        self.assertIs(root["Objects"]["person"][0]["skipFeatureExtraction"], False)

    def test_skip_feature_extraction_unset_by_default(self):
        msg = ObjectsPostProcessorMessage()
        msg.objects = {"person": [ObjectMetadata()]}

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertIsNone(restored.objects["person"][0].skip_feature_extraction)

    def test_skip_feature_extraction_omitted_from_the_wire_when_unset(self):
        # Absent key means "engine default" (do not skip); an untouched object
        # must not change the per-object wire shape existing processors rely on.
        msg = ObjectsPostProcessorMessage()
        msg.objects = {"person": [ObjectMetadata()]}

        root = msgpack.unpackb(msg.to_bytes(), raw=False)

        self.assertNotIn("skipFeatureExtraction", root["Objects"]["person"][0])


class TestAttributeUpdateWireShape(unittest.TestCase):
    """removeAttributes and attribute-only updates: an object continuing a track
    may leave out its bounding box and confidence (the engine keeps its own), and
    may name attributes for the engine to delete."""

    def test_remove_attributes_emitted_on_the_wire(self):
        msg = ObjectsPostProcessorMessage()
        obj = ObjectMetadata()
        obj.track_id = "obj-1"
        obj.remove_attributes = ["Color", "Size"]
        msg.objects = {"person": [obj]}

        root = msgpack.unpackb(msg.to_bytes(), raw=False)

        self.assertEqual(root["Objects"]["person"][0]["removeAttributes"], ["Color", "Size"])

    def test_remove_attributes_omitted_from_the_wire_when_empty(self):
        msg = ObjectsPostProcessorMessage()
        msg.objects = {"person": [ObjectMetadata()]}

        root = msgpack.unpackb(msg.to_bytes(), raw=False)

        self.assertNotIn("removeAttributes", root["Objects"]["person"][0])

    def test_remove_attributes_round_trips(self):
        msg = ObjectsPostProcessorMessage()
        obj = ObjectMetadata()
        obj.remove_attributes = ["Color"]
        msg.objects = {"person": [obj]}

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(restored.objects["person"][0].remove_attributes, ["Color"])

    def test_attribute_update_carries_only_id_and_attributes(self):
        msg = ObjectsPostProcessorMessage()
        obj = ObjectMetadata.attribute_update("obj-1")
        obj.attributes["Plate"] = "ABC123"
        msg.objects = {"person": [obj]}

        root = msgpack.unpackb(msg.to_bytes(), raw=False)

        self.assertEqual(
            root["Objects"]["person"][0], {"ID": "obj-1", "attributes": {"Plate": "ABC123"}}
        )

    def test_default_object_still_emits_bounding_box_and_confidence(self):
        # Existing processors build NewID objects from a default ObjectMetadata;
        # the engine rejects a new track without these, so they stay on the wire.
        msg = ObjectsPostProcessorMessage()
        msg.objects = {"person": [ObjectMetadata()]}

        root = msgpack.unpackb(msg.to_bytes(), raw=False)

        for key in ("x", "y", "width", "height", "confidence"):
            self.assertIn(key, root["Objects"]["person"][0])


class TestBestShotsRoundTrip(unittest.TestCase):
    """BestShots (wire shape): optional top-level array, each entry a
    required TrackId (same UUID format as ObjectMetadata's track_id/new_track_id)
    plus an optional nested BoundingBox."""

    def test_best_shot_with_bounding_box_round_trips(self):
        msg = ObjectsPostProcessorMessage()
        best_shot = BestShotMetadata()
        best_shot.track_id = "550e8400-e29b-41d4-a716-446655440000"
        best_shot.x = 0.1
        best_shot.y = 0.2
        best_shot.width = 0.3
        best_shot.height = 0.4
        msg.best_shots = [best_shot]

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(len(restored.best_shots), 1)
        restored_best_shot = restored.best_shots[0]
        self.assertEqual(restored_best_shot.track_id, best_shot.track_id)
        self.assertEqual(restored_best_shot.x, best_shot.x)
        self.assertEqual(restored_best_shot.y, best_shot.y)
        self.assertEqual(restored_best_shot.width, best_shot.width)
        self.assertEqual(restored_best_shot.height, best_shot.height)

    def test_best_shot_without_bounding_box_round_trips(self):
        msg = ObjectsPostProcessorMessage()
        best_shot = BestShotMetadata()
        best_shot.track_id = "550e8400-e29b-41d4-a716-446655440000"
        msg.best_shots = [best_shot]

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(restored.best_shots[0].track_id, best_shot.track_id)
        self.assertIsNone(restored.best_shots[0].x)

    def test_best_shot_without_bounding_box_omits_it_on_the_wire(self):
        msg = ObjectsPostProcessorMessage()
        best_shot = BestShotMetadata()
        best_shot.track_id = "550e8400-e29b-41d4-a716-446655440000"
        msg.best_shots = [best_shot]

        root = msgpack.unpackb(msg.to_bytes(), raw=False)

        self.assertNotIn("BoundingBox", root["BestShots"][0])

    def test_empty_best_shots_omits_the_key_on_the_wire(self):
        # A missing/absent "BestShots" key is a no-op; a
        # processor that never touches best_shots must not change the wire
        # shape existing processors already rely on.
        msg = ObjectsPostProcessorMessage()

        root = msgpack.unpackb(msg.to_bytes(), raw=False)

        self.assertNotIn("BestShots", root)

    def test_empty_best_shots_round_trips_to_empty_list(self):
        msg = ObjectsPostProcessorMessage()

        restored = ObjectsPostProcessorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(restored.best_shots, [])


class TestObjectsMessageChainedFields(unittest.TestCase):
    """The two fields an OBJECTS-schedule processor needs to relay a chained shot.

    Both are engine-injected and parse-only. `OriginalObjectID` is written by the
    engine whenever the inference input came from a chained FEATURE_EXTRACTION
    stage, and reaches an external postprocessor on this schedule *only* — the
    TENSOR-schedule root is a fixed 6/7-key map that cannot carry it. `Tensors` is
    written on the same schedule under `ReceiveBinaryData: true` and, unlike the
    TENSOR-schedule form, is an array of self-describing maps rather than a
    name→binary map with parallel shape/type arrays.
    """

    @staticmethod
    def _inbound(**extra) -> bytes:
        root = {
            "Timestamp": 1717334400,
            "DeviceID": "device-42",
            "DeviceName": "Front Door Camera",
            "Objects": {},
            "Events": [],
        }
        root.update(extra)
        return msgpack.packb(root, use_bin_type=True)

    def test_original_object_id_parsed_from_root_map(self):
        restored = ObjectsPostProcessorMessage.from_bytes(
            self._inbound(OriginalObjectID="550e8400-e29b-41d4-a716-446655440000")
        )

        self.assertEqual(
            restored.original_object_id, "550e8400-e29b-41d4-a716-446655440000"
        )

    def test_original_object_id_none_when_absent(self):
        restored = ObjectsPostProcessorMessage.from_bytes(self._inbound())

        self.assertIsNone(restored.original_object_id)

    def test_tensor_array_parsed_into_named_tensors(self):
        restored = ObjectsPostProcessorMessage.from_bytes(
            self._inbound(
                Tensors=[
                    {
                        "Name": "embedding",
                        "DataType": DATA_TYPE_UINT8,
                        "Shape": [1, 3],
                        "Data": bytes([1, 2, 3]),
                    }
                ]
            )
        )

        self.assertEqual(list(restored.tensors.keys()), ["embedding"])
        tensor = restored.tensors["embedding"]
        self.assertEqual(tensor.name, "embedding")
        self.assertEqual(tensor.shape, [1, 3])
        self.assertEqual(tensor.data_type, DATA_TYPE_UINT8)
        self.assertEqual(tensor.data, bytes([1, 2, 3]))

    def test_multiple_tensors_keep_their_own_shape_and_type(self):
        # Each entry is self-describing, so nothing depends on positional
        # alignment with sibling arrays the way the TENSOR-schedule form does.
        restored = ObjectsPostProcessorMessage.from_bytes(
            self._inbound(
                Tensors=[
                    {
                        "Name": "a",
                        "DataType": DATA_TYPE_UINT8,
                        "Shape": [2],
                        "Data": bytes([7, 8]),
                    },
                    {
                        "Name": "b",
                        "DataType": DATA_TYPE_FLOAT,
                        "Shape": [1, 1],
                        "Data": bytes([9]),
                    },
                ]
            )
        )

        self.assertEqual(restored.tensors["a"].shape, [2])
        self.assertEqual(restored.tensors["a"].data_type, DATA_TYPE_UINT8)
        self.assertEqual(restored.tensors["b"].shape, [1, 1])
        self.assertEqual(restored.tensors["b"].data_type, DATA_TYPE_FLOAT)

    def test_tensors_empty_when_absent(self):
        # ReceiveBinaryData: false — the key is simply not written.
        restored = ObjectsPostProcessorMessage.from_bytes(self._inbound())

        self.assertEqual(restored.tensors, {})

    def test_to_bytes_emits_neither_field(self):
        msg = ObjectsPostProcessorMessage()
        msg.original_object_id = "550e8400-e29b-41d4-a716-446655440000"
        tensor = TensorMetadata()
        tensor.name = "embedding"
        tensor.data = bytes([1, 2, 3])
        msg.tensors = {"embedding": tensor}

        root = msgpack.unpackb(msg.to_bytes(), raw=False)

        self.assertEqual(
            set(root.keys()),
            {"Timestamp", "DeviceID", "DeviceName", "Objects", "Events"},
        )


class TestTensorMessageRoundTrip(unittest.TestCase):
    def test_populated_tensor_message_round_trips(self):
        msg = TensorMessage()
        msg.timestamp = 1717334400
        msg.device_id = "device-7"
        msg.device_name = "Rear Camera"

        tensor = TensorMetadata()
        tensor.name = "logits"
        tensor.shape = [1, 3]
        tensor.data_type = DATA_TYPE_UINT8
        tensor.data = bytes([1, 2, 3])
        msg.tensors = {"logits": tensor}

        restored = TensorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(restored.timestamp, msg.timestamp)
        self.assertEqual(restored.device_id, msg.device_id)
        self.assertEqual(restored.device_name, msg.device_name)
        self.assertEqual(list(restored.tensors.keys()), ["logits"])

        restored_tensor = restored.tensors["logits"]
        self.assertEqual(restored_tensor.name, tensor.name)
        self.assertEqual(restored_tensor.shape, tensor.shape)
        self.assertEqual(restored_tensor.data_type, tensor.data_type)
        self.assertEqual(restored_tensor.data, tensor.data)

    def test_empty_tensor_message_round_trips(self):
        msg = TensorMessage()

        restored = TensorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(restored.tensors, {})

    def test_user_settings_parsed_from_root_map(self):
        # The engine forwards preprocessor-schedule UserSettings the same way
        # postprocessors already get them: a flat {name: stringValue}
        # sibling of Timestamp/Tensors on the TENSOR-schedule message.
        inbound = msgpack.packb(
            {
                "Timestamp": 1717334400,
                "DeviceID": "device-7",
                "DeviceName": "Rear Camera",
                "Tensors": {},
                "TensorShapes": [],
                "TensorDataTypes": [],
                "UserSettings": {"nmsoverride": "0.3"},
            },
            use_bin_type=True,
        )

        restored = TensorMessage.from_bytes(inbound)

        self.assertEqual(restored.user_settings, {"nmsoverride": "0.3"})

    def test_user_settings_empty_when_absent(self):
        msg = TensorMessage()

        restored = TensorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(restored.user_settings, {})

    def test_to_bytes_never_emits_user_settings(self):
        # UserSettings is inbound-only, same as on ObjectsPostProcessorMessage: the
        # preprocessor reads the forwarded value but never echoes it back.
        msg = TensorMessage()
        msg.user_settings = {"nmsoverride": "0.3"}

        root = msgpack.unpackb(msg.to_bytes(), raw=False)

        self.assertNotIn("UserSettings", root)
        self.assertEqual(
            set(root.keys()),
            {"Timestamp", "DeviceID", "DeviceName", "Tensors", "TensorShapes", "TensorDataTypes"},
        )

    def test_multiple_tensors_keep_shape_and_type_alignment(self):
        msg = TensorMessage()
        first = TensorMetadata()
        first.name = "a"
        first.shape = [2]
        first.data_type = DATA_TYPE_UINT8
        first.data = bytes([7, 8])
        second = TensorMetadata()
        second.name = "b"
        second.shape = [1, 1]
        second.data_type = DATA_TYPE_UINT8
        second.data = bytes([9])
        msg.tensors = {"a": first, "b": second}

        restored = TensorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(restored.tensors["a"].shape, [2])
        self.assertEqual(restored.tensors["a"].data, bytes([7, 8]))
        self.assertEqual(restored.tensors["b"].shape, [1, 1])
        self.assertEqual(restored.tensors["b"].data, bytes([9]))


class TestImagePreprocessorMessageRoundTrip(unittest.TestCase):
    def test_populated_image_message_round_trips(self):
        msg = ImagePreprocessorMessage()
        msg.timestamp = 1717334400
        msg.device_id = "device-9"
        msg.device_name = "Lobby Camera"

        image = ImageData()
        image.height = 2
        image.width = 2
        image.layout = ImageOrder.CHW
        image.bytes = bytes([10, 20, 30, 40])
        msg.image = image

        restored = ImagePreprocessorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(restored.timestamp, msg.timestamp)
        self.assertEqual(restored.device_id, msg.device_id)
        self.assertEqual(restored.device_name, msg.device_name)
        self.assertEqual(restored.image.height, image.height)
        self.assertEqual(restored.image.width, image.width)
        self.assertEqual(restored.image.layout, image.layout)
        self.assertEqual(restored.image.bytes, image.bytes)

    def test_user_settings_parsed_from_root_map(self):
        # The engine forwards preprocessor-schedule UserSettings the same way
        # postprocessors already get them and the TENSOR-schedule
        # message already does: a flat {name: stringValue} sibling of
        # Timestamp/ImageData on the IMAGE-schedule message.
        inbound = msgpack.packb(
            {
                "Timestamp": 1717334400,
                "DeviceID": "device-9",
                "DeviceName": "Lobby Camera",
                "ImageData": ImageData().to_dict(),
                "UserSettings": {"mirrorimage": "false"},
            },
            use_bin_type=True,
        )

        restored = ImagePreprocessorMessage.from_bytes(inbound)

        self.assertEqual(restored.user_settings, {"mirrorimage": "false"})

    def test_user_settings_empty_when_absent(self):
        msg = ImagePreprocessorMessage()

        restored = ImagePreprocessorMessage.from_bytes(msg.to_bytes())

        self.assertEqual(restored.user_settings, {})

    def test_to_bytes_never_emits_user_settings(self):
        # UserSettings is inbound-only, same as on TensorMessage: the
        # preprocessor reads the forwarded value but never echoes it back.
        msg = ImagePreprocessorMessage()
        msg.user_settings = {"mirrorimage": "false"}

        root = msgpack.unpackb(msg.to_bytes(), raw=False)

        self.assertNotIn("UserSettings", root)
        self.assertEqual(
            set(root.keys()), {"Timestamp", "DeviceID", "DeviceName", "ImageData"}
        )


if __name__ == "__main__":
    unittest.main()
