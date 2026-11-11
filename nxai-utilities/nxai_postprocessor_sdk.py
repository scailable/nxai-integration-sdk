"""
Cross-platform Python library for interfacing with NXAI external postprocessor IPC/SHM.

Supports both Linux (eventfd + POSIX shared memory) and Windows (Events + file mapping).
"""

import ctypes
import os
import struct
import sys
import platform
import msgpack
import logging
import socket
from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory
from typing import Optional, Dict, List

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

if not (IS_WINDOWS or IS_LINUX):
    raise OSError(f"Unsupported platform: {platform.system()}")

# Constants
HEADER_BYTES = 4  # Size header used by message protocol
SHM_MAX_SIZE = 200 * 1024 * 1024

# Platform-specific for IPC sync
if IS_WINDOWS:
    kernel32 = ctypes.windll.kernel32
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    WAIT_OBJECT_0 = 0
    WAIT_TIMEOUT = 258
    INFINITE = 0xFFFFFFFF
    # Growing a segment (see SharedMemory.resize)
    PAGE_READWRITE = 0x04
    MEM_COMMIT = 0x1000
    FILE_MAP_ALL_ACCESS = 0xF001F
    # Handles and addresses do not fit ctypes' default c_int return type on x64.
    kernel32.OpenFileMappingA.restype = ctypes.c_void_p
    kernel32.MapViewOfFile.restype = ctypes.c_void_p
    kernel32.VirtualAlloc.restype = ctypes.c_void_p


# ============================================================================
# Logging and Configuration Utilities
# ============================================================================


def setup_logging(
    log_level: int = logging.DEBUG,
) -> logging.Logger:
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.addFilter(lambda r: r.levelno < logging.ERROR)
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    return logger


def load_config(args: list[str]) -> dict[str, str]:
    result = {}
    it = iter(args)
    for token in it:
        if token.startswith("--"):
            key = token[2:]
            value = next(it, None)
            if value is not None and not value.startswith("--"):
                result[key] = value
    return result


def apply_config_logging(config: dict[str, str], logger: logging.Logger) -> None:
    """Apply logging level from config to logger.

    Looks for [logging] section with 'level' option.

    Args:
        config: ConfigParser object
        logger: Logger instance to configure
    """
    try:
        if "LogLevel" in config:
            level_str = config["LogLevel"]
            level = getattr(logging, level_str.upper(), logging.INFO)
            logger.setLevel(level)
            logger.info(f"Logging level set to {level_str}")
    except Exception as e:
        logger.warning(f"Error applying config logging: {e}")


# ============================================================================
# Data Type Definitions
# ============================================================================

# Tensor data types (from enum nxai_data_type)
DATA_TYPE_FLOAT = 1
DATA_TYPE_UINT8 = 2
DATA_TYPE_INT8 = 3
DATA_TYPE_UINT16 = 4
DATA_TYPE_INT16 = 5
DATA_TYPE_INT32 = 6
DATA_TYPE_INT64 = 7
DATA_TYPE_STRING = 8
DATA_TYPE_BOOL = 9
DATA_TYPE_DOUBLE = 11
DATA_TYPE_UINT32 = 12
DATA_TYPE_UINT64 = 13

DATA_TYPE_NAMES = {
    DATA_TYPE_FLOAT: "float",
    DATA_TYPE_UINT8: "uint8",
    DATA_TYPE_INT8: "int8",
    DATA_TYPE_UINT16: "uint16",
    DATA_TYPE_INT16: "int16",
    DATA_TYPE_INT32: "int32",
    DATA_TYPE_INT64: "int64",
    DATA_TYPE_STRING: "string",
    DATA_TYPE_BOOL: "bool",
    DATA_TYPE_DOUBLE: "double",
    DATA_TYPE_UINT32: "uint32",
    DATA_TYPE_UINT64: "uint64",
}


# ============================================================================
# Message Type Classes
# ============================================================================


class ObjectMetadata:
    """Represents a detected object in a frame.

    ``skip_feature_extraction`` gates the v5 engine's chained FEATURE_EXTRACTION
    stage for this object, serialized as the optional per-object bool
    key ``skipFeatureExtraction``:

    - ``None`` (default) — the key is omitted; the engine applies its own default
      (do not skip, i.e. feature-extract this object).
    - ``True`` — the engine skips the FEATURE_EXTRACTION chain for this object.
    - ``False`` — the engine runs the chain for this object.

    Semantics are per-frame transient / unconditional-overwrite: the engine reads
    the field fresh each frame (absent means default, never "keep last value"), so
    a processor gating chaining must set it every frame it wants the gate applied.

    ``remove_attributes`` names attributes the engine deletes from the object,
    serialized as ``removeAttributes`` when non-empty. It is applied after
    ``attributes``, so a name in both ends up removed.

    For an object continuing an existing track (``track_id`` set), the bounding
    box and ``confidence`` may be left as ``None`` to keep the engine's current
    values; ``None`` fields are omitted from the wire. A new track (``NewID``)
    still needs all of them. See ``attribute_update``.
    """

    def __init__(self):
        self.track_id: str = None  # UUID string
        self.new_track_id: str = None  # UUID string
        self.x: Optional[float] = 0.0
        self.y: Optional[float] = 0.0
        self.width: Optional[float] = 0.0
        self.height: Optional[float] = 0.0
        self.confidence: Optional[float] = 0.0
        self.attributes: Dict[str, str] = {}
        self.remove_attributes: List[str] = []
        self.skip_feature_extraction: Optional[bool] = None

    @staticmethod
    def attribute_update(track_id: str) -> "ObjectMetadata":
        """An object that only edits the attributes of the existing track ``track_id``.

        The bounding box and confidence are left out of the wire, so the engine
        keeps its own. This is how a processor on a model chained after feature
        extraction edits the original object it was given as ``OriginalObjectID``.
        """
        obj = ObjectMetadata()
        obj.track_id = track_id
        obj.x = obj.y = obj.width = obj.height = obj.confidence = None
        return obj

    @staticmethod
    def from_dict(data: dict) -> "ObjectMetadata":
        """Create ObjectMetadata from dictionary (msgpack deserialized)."""
        obj = ObjectMetadata()
        obj.track_id = data.get("ID")
        obj.new_track_id = data.get("NewID")
        obj.x = float(data.get("x", 0.0))
        obj.y = float(data.get("y", 0.0))
        obj.width = float(data.get("width", 0.0))
        obj.height = float(data.get("height", 0.0))
        obj.confidence = float(data.get("confidence", 0.0))
        obj.attributes = dict(data.get("attributes", {}))
        obj.remove_attributes = list(data.get("removeAttributes", []))
        skip = data.get("skipFeatureExtraction")
        obj.skip_feature_extraction = bool(skip) if skip is not None else None
        return obj

    def to_dict(self) -> dict:
        """Convert ObjectMetadata to dictionary for msgpack serialization."""
        return_dict = {"attributes": self.attributes}
        for key in ("x", "y", "width", "height", "confidence"):
            value = getattr(self, key)
            if value is not None:
                return_dict[key] = value
        if self.remove_attributes:
            return_dict["removeAttributes"] = list(self.remove_attributes)
        if self.track_id is not None:
            return_dict["ID"] = self.track_id
        if self.new_track_id is not None:
            return_dict["NewID"] = self.new_track_id
        if self.skip_feature_extraction is not None:
            return_dict["skipFeatureExtraction"] = self.skip_feature_extraction
        return return_dict


class EventMetadata:
    """Represents an event detected in a frame."""

    def __init__(self):
        self.type_id: str = ""
        self.description: str = ""

    @staticmethod
    def from_dict(data: dict) -> "EventMetadata":
        """Create EventMetadata from dictionary (msgpack deserialized)."""
        event = EventMetadata()
        event.type_id = data.get("ID", "")
        event.description = data.get("Description", "")
        return event

    def to_dict(self) -> dict:
        """Convert EventMetadata to dictionary for msgpack serialization."""
        return {
            "ID": self.type_id,
            "Description": self.description,
        }


class BestShotMetadata:
    """Represents a postprocessor-nominated best shot for an already-tracked object.

    Mirrors the v5 engine's top-level optional "BestShots" response array
    array: each entry names an existing track (``track_id``, required,
    same UUID format as ``ObjectMetadata``'s ``track_id``/``new_track_id``) and
    optionally a bounding box to crop the shot to. A missing/absent top-level
    ``BestShots`` key is a no-op, so an empty list is never serialized.
    """

    def __init__(self):
        self.track_id: str = None  # required: UUID string of an existing track
        self.x: float = None
        self.y: float = None
        self.width: float = None
        self.height: float = None

    @staticmethod
    def from_dict(data: dict) -> "BestShotMetadata":
        """Create BestShotMetadata from dictionary (msgpack deserialized)."""
        best_shot = BestShotMetadata()
        best_shot.track_id = data.get("TrackId")
        bounding_box = data.get("BoundingBox")
        if bounding_box is not None:
            best_shot.x = float(bounding_box.get("x", 0.0))
            best_shot.y = float(bounding_box.get("y", 0.0))
            best_shot.width = float(bounding_box.get("width", 0.0))
            best_shot.height = float(bounding_box.get("height", 0.0))
        return best_shot

    def to_dict(self) -> dict:
        """Convert BestShotMetadata to dictionary for msgpack serialization."""
        entry = {"TrackId": self.track_id}
        if self.x is not None:
            entry["BoundingBox"] = {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            }
        return entry


class ObjectsPostProcessorMessage:
    """Represents the full message exchanged with the postprocessor."""

    def __init__(self):
        self.timestamp: int = 0
        self.device_id: str = ""
        self.device_name: str = ""
        self.objects: Dict[str, List[ObjectMetadata]] = {}  # class_name -> [objects]
        self.events: List[EventMetadata] = []
        # Postprocessor-nominated best shots, write-only: a
        # missing/absent "BestShots" key on the wire is a no-op, so this stays
        # empty unless the processor appends to it.
        self.best_shots: List[BestShotMetadata] = []
        # Forwarded GUI settings, parse-only: a flat {name: value} map
        # the engine delivers under the root `UserSettings` key. Never serialized back.
        self.user_settings: Dict[str, str] = {}
        # Input frame (ReceiveInputImage), parse-only: the engine delivers it under
        # the root `ImageData` key, absent unless the JSON opts in. Never serialized back.
        self.image_data: Optional["ImageData"] = None
        # Root-level tracker UUID on chained FEATURE_EXTRACTION-stage messages
        # Engine-injected and parse-only. This schedule is the *only*
        # one that carries it: the TENSOR-schedule root is a fixed 6/7-key map with
        # no room for the key, so a processor keying on it must run on OBJECTS.
        self.original_object_id: Optional[str] = None
        # Raw model output tensors, delivered under `ReceiveBinaryData: true` and
        # parse-only. On this schedule the engine writes `Tensors` as an array of
        # self-describing maps ({Name, DataType, Shape, Data}), not the TENSOR
        # schedule's name->binary map with parallel shape/type arrays.
        self.tensors: Dict[str, "TensorMetadata"] = {}

    @staticmethod
    def from_bytes(data: bytes) -> "ObjectsPostProcessorMessage":
        """Parse msgpack bytes into ObjectsPostProcessorMessage."""
        try:
            raw_data = msgpack.unpackb(data, raw=False)
        except Exception as e:
            print(f"Error parsing msgpack message: {e}")
            return None
        return ObjectsPostProcessorMessage.from_dict(raw_data)

    @staticmethod
    def from_dict(raw_data: dict) -> "ObjectsPostProcessorMessage":
        """Build an ObjectsPostProcessorMessage from an already-unpacked dict."""
        msg = ObjectsPostProcessorMessage()
        try:
            msg.timestamp = int(raw_data.get("Timestamp", 0))
            msg.device_id = raw_data.get("DeviceID", "")
            msg.device_name = raw_data.get("DeviceName", "")

            # Parse objects (organized by class name)
            objects_data = raw_data.get("Objects", {})
            for class_name, objects_array in objects_data.items():
                msg.objects[class_name] = [
                    ObjectMetadata.from_dict(obj) for obj in objects_array
                ]

            # Parse events
            events_data = raw_data.get("Events", [])
            msg.events = [EventMetadata.from_dict(evt) for evt in events_data]

            # Parse best shots (absent key -> empty list)
            best_shots_data = raw_data.get("BestShots", [])
            msg.best_shots = [BestShotMetadata.from_dict(bs) for bs in best_shots_data]

            # Parse forwarded GUI settings (absent key -> empty dict)
            msg.user_settings = dict(raw_data.get("UserSettings", {}))

            # Parse input frame (ReceiveInputImage; absent key -> None)
            image_data = raw_data.get("ImageData")
            msg.image_data = ImageData.from_dict(image_data) if image_data is not None else None

            # Parse the chained-stage tracker UUID (absent key -> None)
            msg.original_object_id = raw_data.get("OriginalObjectID")

            # Parse raw output tensors (absent key -> empty dict)
            for tensor_entry in raw_data.get("Tensors", []):
                name = tensor_entry.get("Name", "")
                tensor_data = tensor_entry.get("Data") or b""
                if not isinstance(tensor_data, bytes):
                    tensor_data = bytes(tensor_data)
                msg.tensors[name] = TensorMetadata.from_msgpack_pair(
                    name,
                    tensor_data,
                    tensor_entry.get("Shape", []),
                    tensor_entry.get("DataType", DATA_TYPE_FLOAT),
                )

            return msg
        except Exception as e:
            print(f"Error parsing msgpack message: {e}")
            return None

    def to_bytes(self) -> bytes:
        """Serialize ProcessorMessage to msgpack bytes."""
        try:
            # Build objects dictionary
            objects_dict = {}
            for class_name, objects_list in self.objects.items():
                objects_dict[class_name] = [obj.to_dict() for obj in objects_list]

            # Build events array
            events_array = [evt.to_dict() for evt in self.events]

            # Build root message
            message_dict = {
                "Timestamp": self.timestamp,
                "DeviceID": self.device_id,
                "DeviceName": self.device_name,
                "Objects": objects_dict,
                "Events": events_array,
            }
            # A missing/absent "BestShots" key is a no-op, so it
            # is only emitted when the processor actually nominated one.
            if self.best_shots:
                message_dict["BestShots"] = [bs.to_dict() for bs in self.best_shots]
            logging.debug("Packing message: %s", message_dict)

            return msgpack.packb(message_dict, use_bin_type=True)
        except Exception as e:
            print(f"Error serializing to msgpack: {e}")
            return None


class TensorMetadata:
    """Represents a tensor (multi-dimensional array) with metadata."""

    def __init__(self):
        self.name: str = ""
        self.shape: List[int] = []
        self.data_type: int = DATA_TYPE_FLOAT
        self.data: bytes = b""

    @property
    def rank(self) -> int:
        return len(self.shape)

    @staticmethod
    def from_msgpack_pair(
        name: str, tensor_data: bytes, shape: List[int], data_type: int
    ) -> "TensorMetadata":
        """Create TensorMetadata from msgpack parsed values."""
        tensor = TensorMetadata()
        tensor.name = name
        tensor.shape = list(shape) if shape else []
        tensor.data_type = data_type
        tensor.data = tensor_data if isinstance(tensor_data, bytes) else b""
        return tensor


class TensorMessage:
    """Represents tensor data exchanged with postprocessor (typically model input/output)."""

    def __init__(self):
        self.timestamp: int = 0
        self.device_id: str = ""
        self.device_name: str = ""
        self.tensors: Dict[str, TensorMetadata] = {}  # name -> TensorMetadata
        # Forwarded GUI settings, parse-only: a flat {name: value} map
        # the engine delivers under the root `UserSettings` key on the TENSOR-schedule
        # message, mirroring the postprocessor contract. Never serialized back.
        self.user_settings: Dict[str, str] = {}

    @staticmethod
    def from_bytes(data: bytes) -> Optional["TensorMessage"]:
        msg = TensorMessage()
        try:
            raw_data = msgpack.unpackb(data, raw=False)

            msg.timestamp = int(raw_data.get("Timestamp", 0))
            msg.device_id = raw_data.get("DeviceID", "")
            msg.device_name = raw_data.get("DeviceName", "")

            # Parse forwarded GUI settings (absent key -> empty dict)
            msg.user_settings = dict(raw_data.get("UserSettings", {}))

            tensors_map = raw_data.get("Tensors", {})
            shapes_array = raw_data.get("TensorShapes", [])
            types_array = raw_data.get("TensorDataTypes", [])

            # Get tensor names in order (msgpack preserves order in dicts)
            tensor_names = list(tensors_map.keys())

            for idx, name in enumerate(tensor_names):
                tensor_data = tensors_map[name]
                shape = shapes_array[idx] if idx < len(shapes_array) else []
                data_type = (
                    types_array[idx] if idx < len(types_array) else DATA_TYPE_FLOAT
                )

                # Handle binary data - may be None/nil if not present
                if tensor_data is None:
                    tensor_data = b""
                elif not isinstance(tensor_data, bytes):
                    tensor_data = bytes(tensor_data)

                msg.tensors[name] = TensorMetadata.from_msgpack_pair(
                    name, tensor_data, shape, data_type
                )

            return msg
        except Exception as e:
            print(f"Error parsing tensor msgpack message: {e}")
            return None

    def to_bytes(self) -> bytes:
        """Serialize TensorMessage to msgpack bytes (matches parseTensorMessage C++ format)."""
        try:
            tensors_map = {}
            shapes_array = []
            types_array = []

            for name, tensor in self.tensors.items():
                tensors_map[name] = tensor.data
                shapes_array.append(tensor.shape)
                types_array.append(tensor.data_type)

            message_dict = {
                "Timestamp": self.timestamp,
                "DeviceID": self.device_id,
                "DeviceName": self.device_name,
                "Tensors": tensors_map,
                "TensorShapes": shapes_array,
                "TensorDataTypes": types_array,
            }

            return msgpack.packb(message_dict, use_bin_type=True)
        except Exception as e:
            print(f"Error serializing tensor message to msgpack: {e}")
            return None


# ============================================================================
# Shared Memory Management
# ============================================================================

from enum import IntEnum


class ImageOrder(IntEnum):
    CHW = 0
    HWC = 1


class ImageData:
    """Represents a raw image frame."""

    def __init__(self):
        self.height: int = 0
        self.width: int = 0
        self.layout: ImageOrder = ImageOrder.HWC
        self.bytes: bytes = b""

    @classmethod
    def from_dict(cls, data: dict) -> "ImageData":
        img = cls()
        img.height = int(data.get("Height", 0))
        img.width = int(data.get("Width", 0))
        img.layout = ImageOrder(data.get("Layout", ImageOrder.HWC))
        raw = data.get("Data", b"")
        img.bytes = bytes(raw) if not isinstance(raw, bytes) else raw
        return img

    def to_dict(self) -> dict:
        return {
            "Height": self.height,
            "Width": self.width,
            "Layout": int(self.layout),
            "Data": self.bytes,
        }


class ImagePreprocessorMessage:
    """Represents the image message exchanged with an external preprocessor."""

    def __init__(self):
        self.timestamp: int = 0
        self.device_id: str = ""
        self.device_name: str = ""
        self.image: ImageData = ImageData()
        # Forwarded GUI settings, parse-only: a flat {name: value} map
        # the engine delivers under the root `UserSettings` key on the IMAGE-schedule
        # message, mirroring the tensor-schedule contract. Never serialized back.
        self.user_settings: Dict[str, str] = {}

    @staticmethod
    def from_bytes(data: bytes) -> Optional["ImagePreprocessorMessage"]:
        """Parse msgpack bytes into ImagePreprocessorMessage."""
        msg = ImagePreprocessorMessage()
        try:
            raw_data = msgpack.unpackb(data, raw=False)
            msg.timestamp = int(raw_data.get("Timestamp", 0))
            msg.device_id = raw_data.get("DeviceID", "")
            msg.device_name = raw_data.get("DeviceName", "")
            image_data = raw_data.get("ImageData", {})
            msg.image = ImageData.from_dict(image_data)
            msg.user_settings = dict(raw_data.get("UserSettings", {}))
            return msg
        except Exception as e:
            print(f"Error parsing image preprocessor msgpack message: {e}")
            return None

    def to_bytes(self) -> Optional[bytes]:
        """Serialize ImagePreprocessorMessage to msgpack bytes."""
        try:
            message_dict = {
                "Timestamp": self.timestamp,
                "DeviceID": self.device_id,
                "DeviceName": self.device_name,
                "ImageData": self.image.to_dict(),
            }
            return msgpack.packb(message_dict, use_bin_type=True)
        except Exception as e:
            print(f"Error serializing image preprocessor message to msgpack: {e}")
            return None


class IpcSignal(IntEnum):
    ACK = 0
    NOTICE = 1
    REALLOC = 2
    ERROR = 3
    READY = 4
    EXIT = 5


class SharedMemory:
    """Cross-platform shared memory interface using built-in multiprocessing module."""

    def __init__(self):
        self.shm = None
        self.shm_key = None
        self.capacity = 0  # User-facing capacity (excludes HEADER_BYTES)

    def attach(self, key: str) -> bool:
        """Attach to existing shared memory."""
        try:
            # Strip leading '/' if present (POSIX shared memory names)
            shm_name = key.lstrip("/")

            self.shm = BuiltinSharedMemory(name=shm_name)
            self.shm_key = key
            # Capacity is the buffer size minus the 4-byte header
            self.capacity = len(self.shm.buf) - HEADER_BYTES
            return True
        except Exception as e:
            print(f"Error attaching to SHM {key}: {e}")
            return False

    def read(self) -> Optional[bytes]:
        """Read message from shared memory (size header + payload)."""
        if not self.shm:
            logging.warning("Tried to read from a SHM that is not initialized.")
            return None

        try:
            # Read 4-byte size header
            header = bytes(self.shm.buf[0:4])
            size = struct.unpack("<I", header)[0]

            if size == 0 or size > SHM_MAX_SIZE:
                return None

            # Read payload
            payload = bytes(self.shm.buf[HEADER_BYTES : HEADER_BYTES + size])
            return payload
        except Exception as e:
            print(f"Error reading from SHM: {e}")
            return None

    def write(self, data: bytes) -> bool:
        """Write message to shared memory (size header + payload).

        Returns: True on success, False on buffer too small or error
        Raises: BufferError if buffer is too small (caller should reallocate)
        """
        if not self.shm:
            return False

        try:
            size = len(data)
            if size > SHM_MAX_SIZE:
                return False

            # Check if buffer is large enough
            required_size = HEADER_BYTES + size
            if required_size > len(self.shm.buf):
                # Signal that reallocation is needed by raising BufferError
                raise BufferError(
                    f"SHM buffer too small ({len(self.shm.buf)} bytes), need {required_size} bytes"
                )

            # Write by converting entire buffer to bytearray, modifying it, then writing back
            # This avoids memoryview structure issues
            buf_data = bytearray(self.shm.buf)

            # Write size header (4 bytes, little-endian)
            size_bytes = struct.pack("<I", size)
            buf_data[0:4] = size_bytes

            # Write payload
            buf_data[HEADER_BYTES : HEADER_BYTES + size] = data

            # Write modified buffer back to shared memory
            self.shm.buf[:] = buf_data

            return True
        except BufferError:
            # Re-raise BufferError to let caller handle reallocation
            raise
        except Exception as e:
            logging.error(f"Error writing to SHM: {e}", exc_info=True)
            return False

    def resize(self, required_size: int) -> bool:
        """Grow the segment to hold `required_size` bytes, header included.

        The engine owns the segment; this only enlarges it under the same key,
        so the other side keeps its handle and only has to re-map. Returns True
        with `capacity` updated, or False with the reason logged.
        """
        if required_size > SHM_MAX_SIZE:
            logging.error(
                f"Cannot resize SHM to {required_size} bytes: over the {SHM_MAX_SIZE}-byte protocol maximum"
            )
            return False

        shm_name = self.shm_key.lstrip("/") if self.shm_key else None
        if not shm_name or not self.shm:
            logging.error("Cannot resize SHM: not attached")
            return False

        logging.info(
            f"Resizing SHM '{shm_name}' from {len(self.shm.buf)} to {required_size} bytes"
        )
        try:
            self.shm.close()
            if IS_WINDOWS:
                self._commit_windows(shm_name, required_size)
            else:
                self._truncate_posix(shm_name, required_size)
        except Exception as e:
            logging.error(f"Failed to resize SHM '{shm_name}': {e}", exc_info=True)
            return False

        # Re-attach to pick up the new size: a mapping keeps the length it was
        # made with, on either platform.
        if not self.attach(self.shm_key):
            logging.error(f"Failed to reattach to resized SHM '{shm_name}'")
            return False

        logging.info(f"Reattached to resized SHM, new capacity: {self.capacity} bytes")
        return True

    def detach(self) -> bool:
        """Detach from shared memory."""
        if not self.shm:
            return True

        try:
            self.shm.close()
            self.shm = None
            self.capacity = 0
            return True
        except Exception as e:
            print(f"Error detaching from SHM: {e}")
            return False

    @staticmethod
    def _commit_windows(shm_name: str, required_size: int) -> None:
        """Commit `required_size` bytes of the mapping's reservation.

        The engine's nxai_shm_create_random() creates the mapping with
        SEC_RESERVE over the whole SHM_MAX_SIZE, so it is never resized on
        Windows: only its committed prefix grows, and only that prefix may be
        touched. This is the same MEM_COMMIT the C++ nxai_shm_realloc() does --
        the segment cannot be resized from here (multiprocessing.shared_memory
        cannot resize on either platform), and it does not need to be.
        """
        handle = kernel32.OpenFileMappingA(
            FILE_MAP_ALL_ACCESS, False, shm_name.encode()
        )
        if not handle:
            raise OSError(
                f"OpenFileMappingA('{shm_name}') failed: {ctypes.WinError()}"
            )
        try:
            view = kernel32.MapViewOfFile(
                ctypes.c_void_p(handle), FILE_MAP_ALL_ACCESS, 0, 0, 0
            )
            if not view:
                raise OSError(f"MapViewOfFile failed: {ctypes.WinError()}")
            try:
                if not kernel32.VirtualAlloc(
                    ctypes.c_void_p(view),
                    ctypes.c_size_t(required_size),
                    MEM_COMMIT,
                    PAGE_READWRITE,
                ):
                    raise OSError(f"VirtualAlloc failed: {ctypes.WinError()}")
            finally:
                kernel32.UnmapViewOfFile(ctypes.c_void_p(view))
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))

    @staticmethod
    def _truncate_posix(shm_name: str, required_size: int) -> None:
        """Grow the POSIX segment through its /dev/shm object."""
        fd = os.open(f"/dev/shm/{shm_name}", os.O_RDWR)
        try:
            os.ftruncate(fd, required_size)
        finally:
            os.close(fd)


# ============================================================================
# IPC Sync Management
# ============================================================================


class IPCSync:
    """Cross-platform IPC synchronization interface using events/signals."""

    def __init__(self):
        self.downstream_sync = None
        self.upstream_sync = None
        self.sync_shm = None
        self.data_shm = None
        self.data_ptr_downstream = None
        self.data_ptr_upstream = None
        self.role = None

    def initialize(self, sync_str: str) -> bool:
        """Initialize IPC sync from serialized string.

        Format: "fd_downstream|fd_upstream|shm_key"
        Linux: decimal FD numbers
        Windows: hex pointers
        """
        try:
            # Since we are initializing and not creating, we are "client"
            self.role = "client"

            parts = sync_str.split("|")
            print("Initializing channel with parts:", parts)
            if len(parts) != 4:
                print(f"Error: Invalid IPC sync format: {sync_str}")
                return False

            self.sync_shm = SharedMemory()
            if not self.sync_shm.attach(parts[2]):
                print(f"Error: Could not attach to sync SHM")
                return False

            self.data_shm = SharedMemory()
            if not self.data_shm.attach(parts[3]):
                print(f"Error: Could not attach to sync SHM")
                return False

            if IS_LINUX:
                self.downstream_sync = int(parts[0])
                self.upstream_sync = int(parts[1])
            else:
                self.downstream_sync = int(parts[0], 16)
                self.upstream_sync = int(parts[1], 16)

            return True
        except Exception as e:
            print(f"Error initializing IPC sync: {e}")
            return False

    def signal(self, value: int) -> bool:
        """Send a signal.

        role: "host" or "client"
        value: signal value (-5 to 127)
        """
        if not self.sync_shm or not self.sync_shm.shm:
            return False

        try:
            # Write signal value to shared memory
            if self.role == "client":
                sync_fd = self.upstream_sync
                offset = 1
            else:
                sync_fd = self.downstream_sync
                offset = 0

            # Write byte value to SHM
            self.sync_shm.shm.buf[offset] = struct.pack("b", value)[0]

            # Signal the event
            if IS_LINUX:
                signal_val = 1
                written = os.write(sync_fd, struct.pack("<Q", signal_val))
                return written == 8
            else:
                return kernel32.SetEvent(ctypes.c_void_p(sync_fd)) != 0
        except Exception as e:
            print(f"Error signaling: {e}")
            return False

    def wait(self, timeout_ms: int = 1000) -> Optional[int]:
        """Wait for a signal and return its value.

        role: "host" or "client"
        timeout_ms: timeout in milliseconds (-1 for infinite)

        Returns: signal value (int8) or None on timeout/error
        """
        if not self.sync_shm or not self.sync_shm.shm:
            return None

        try:
            if self.role == "client":
                sync_fd = self.downstream_sync
                offset = 0
            else:
                sync_fd = self.upstream_sync
                offset = 1

            if IS_LINUX:
                return self._wait_linux(sync_fd, offset, timeout_ms)
            else:
                return self._wait_windows(sync_fd, offset, timeout_ms)
        except Exception as e:
            raise RuntimeError(f"Error waiting for signal: {e}") from e

    def _wait_linux(self, fd: int, offset: int, timeout_ms: int) -> Optional[int]:
        """Wait on Linux using poll."""
        import select

        poller = select.poll()
        poller.register(fd, select.POLLIN)
        ready = poller.poll(None if timeout_ms < 0 else timeout_ms)
        if not ready:
            return None  # Timeout

        # Read from eventfd to consume the signal
        try:
            data = os.read(fd, 8)
            # Read signal value from SHM
            sig_byte = self.sync_shm.shm.buf[offset]
            if isinstance(sig_byte, bytes):
                sig_byte = sig_byte[0]
            return struct.unpack("b", bytes([sig_byte]))[0]
        except Exception:
            return None

    def _wait_windows(self, handle: int, offset: int, timeout_ms: int) -> Optional[int]:
        """Wait on Windows using WaitForSingleObject."""
        if timeout_ms < 0:
            timeout_val = INFINITE
        else:
            timeout_val = timeout_ms

        result = kernel32.WaitForSingleObject(
            ctypes.c_void_p(handle), ctypes.c_uint(timeout_val)
        )

        if result == WAIT_OBJECT_0:
            # Read signal value from SHM
            sig_byte = self.sync_shm.shm.buf[offset]
            if isinstance(sig_byte, bytes):
                sig_byte = sig_byte[0]
            return struct.unpack("b", bytes([sig_byte]))[0]

        return None  # Timeout or error

    def close(self) -> bool:
        """Close IPC sync resources."""
        if self.sync_shm:
            self.sync_shm.detach()
        if self.data_shm:
            self.data_shm.detach()
        return True


# ============================================================================
# High-level API
# ============================================================================


class ExternalProcessor:
    """High-level interface for an external processor (pre- or post-).

    A single generic IPC endpoint: pre- and post-processors share the same
    SHM/sync transport, so they are one class here. The axis that varies is the
    wire codec (objects / tensors / image), exposed via the parse_*/write_*
    helpers below, not the pipeline side.
    """

    def __init__(self, name: str = None):
        self.name = name
        self.input_channel = IPCSync()
        self.output_channel = IPCSync()
        self.is_initialized = False  # Track successful initialization
        self.tcp_socket = None
        self.use_tcp = False
        self.tcp_connection = None
        self.exit_requested = False  # Set when IpcSignal.EXIT is observed

    def initialize_from_args(self, args: list = None) -> bool:
        """Initialize from command-line arguments.

        Supports two modes:
        1. Direct args: [input_sync_str, output_sync_str, --Config-key value ..., --ProcessorName <name>]
        2. TCP mode: [--port, <port>]

        If args is None, uses sys.argv[1:]
        """
        if args is None:
            args = sys.argv[1:]

        # Check for TCP mode (--port <port>)
        if len(args) >= 2 and args[0] == "--port":
            try:
                port = int(args[1])
                return self._initialize_from_tcp(port)
            except (ValueError, IndexError):
                print("Error: Invalid port argument")
                print(f"Usage: {sys.argv[0]} --port <port>")
                return False

        # Direct args mode
        if len(args) < 2:
            print(f"Error: Expected at least 2 arguments, got {len(args)}")
            print(
                f"Usage: {sys.argv[0]} <input_sync> <output_sync> "
                "[--Config-key value ...] [--ProcessorName <name>]"
            )
            print(f"   or: {sys.argv[0]} --port <port>")
            return False

        input_sync_str = args[0]
        output_sync_str = args[1]
        config = load_config(args[2:])
        self.name = config.get("ProcessorName") or self.name or "unknown"

        return self._initialize_from_keys(input_sync_str, output_sync_str)

    def _initialize_from_keys(
        self, input_channel_str: str, output_channel_str: str
    ) -> bool:
        """Initialize using explicit IPC/SHM keys.

        Args:
            input_sync_str: Input sync string
            output_sync_str: Output sync string

        Returns: True on success, False on error
        """
        # Initialize IPC syncs
        if not self.input_channel.initialize(input_channel_str):
            print("Error: Failed to initialize input sync")
            return False

        if not self.output_channel.initialize(output_channel_str):
            print("Error: Failed to initialize output sync")
            self.input_channel.close()
            return False

        print("Postprocessor initialized successfully")
        self.is_initialized = True

        # READY goes on the input channel only. The AI Manager consumes it there
        # (sendMessageIPC), but never reads a READY on the output channel: the signal
        # would sit in the semaphore and be returned to the engine's first
        # waitForOutput as an unexpected signal, costing one whole exchange.
        self.input_channel.signal(IpcSignal.READY)
        return True

    def _initialize_from_tcp(self, port: int) -> bool:
        """Initialize by listening for JSON configuration over TCP.

        Args:
            port: TCP port to listen on

        Returns: True on success, False on error
        """
        print(f"Initializing TCP listener on port: {port}")
        try:
            # Create and bind server socket
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(("127.0.0.1", port))
            server_socket.listen(1)
            server_socket.settimeout(1000)
            self.use_tcp = True
            self.tcp_socket = server_socket
            print("Postprocessor TCP initialized successfully")
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Error during TCP initialization: {e}")
            self.tcp_socket = None
            return False

    def wait_for_message(self, timeout_ms: int = 1000) -> Optional[bytes]:
        """Wait for incoming message from server.

        Returns: message bytes, None on timeout/error, or raises exception on connection loss
        """
        if self.use_tcp is True:
            return self._tcp_wait_for_message(timeout_ms)
        else:
            return self._ipc_wait_for_message(timeout_ms)

    def _tcp_wait_for_message(self, timeout_ms: int = 1000) -> Optional[bytes]:

        if self.tcp_connection is None:
            try:
                self.tcp_connection, _ = self.tcp_socket.accept()
                self.tcp_connection.settimeout(timeout_ms / 1000)
                self.tcp_connection.setsockopt(
                    socket.IPPROTO_TCP, socket.TCP_NODELAY, 1
                )
            except TimeoutError:
                # no connection arrived within timeout
                return None  # Timeout (normal, no data)

        recv_data = b""
        message_size = None
        received_bytes = 0
        try:
            header = self.tcp_connection.recv(4)
            if len(header) < 4:
                # Connection closed gracefully
                self.tcp_connection.close()
                self.tcp_connection = None
                return None
            message_size = struct.unpack("<I", header)[0]
            received_bytes = 0
            while received_bytes < message_size:
                chunk = self.tcp_connection.recv(
                    min(4096, message_size - received_bytes)
                )
                received_bytes += len(chunk)
                if not chunk:
                    break
                recv_data += chunk
        except (ConnectionResetError, OSError):
            print("Warning! TCP connection closed. Will wait for new connection.")
            self.tcp_connection.close()
            self.tcp_connection = None
            return None
        except socket.timeout:
            if message_size is None:
                print("Error: Timeout receiving header.")
            else:
                print(
                    "Error: Timed out receiving message. Received",
                    received_bytes,
                    "/",
                    message_size,
                )
                self.tcp_connection.close()
            return False
        # Data received succesfully
        return recv_data

    def _drain_stale_signals(self, sync: "IPCSync", channel_name: str) -> None:
        """Discard any signal already pending on a channel, without blocking.

        A handshake this process gave up on after a timeout does not stop existing: whatever it
        wrote (data + a signal) or the peer's late reply to it can still show up here well after
        the exchange it belonged to was abandoned. Call this immediately before starting a *new*
        handshake that this process initiates, i.e. before a send -- nothing for that new
        exchange has been sent yet at that point, so anything already sitting on the channel
        cannot belong to it and is safe to discard. Without this, a fresh send_response() call
        can consume a stale leftover instead of the ACK it actually asked for, permanently
        shifting which reply belongs to which request from then on.

        Never call this before a receive: there the peer initiates the exchange, and a signal
        already pending is most likely the NOTICE or EXIT it legitimately posted before this
        process got around to waiting.
        """
        while True:
            stale_signal = sync.wait(0)
            if stale_signal is None:
                return
            logging.warning(
                f"Discarded stale signal {stale_signal} on {channel_name} channel "
                "left over from an abandoned exchange."
            )

    def _ipc_wait_for_message(self, timeout_ms: int = 1000) -> Optional[bytes]:
        # Wait for signal from server
        input_sync = self.input_channel
        signal = input_sync.wait(timeout_ms)
        logging.debug(f"Got signal on input channel: {signal}")
        if signal is None:
            return None  # Timeout (normal, no data)

        if signal == -10:
            # Connection error
            raise RuntimeError(
                "IPC connection lost - C++ process may have crashed or disconnected"
            )

        if signal == IpcSignal.REALLOC:
            # SHM reallocation signal
            logging.info("Received SHM reallocation signal from server")

            # Close current input_sync.data_shm attachment
            self.input_channel.data_shm.detach()

            # Try to reattach to input_sync.data_shm (key should be the same, just size changed)
            logging.debug(f"Reattaching with key {self.input_channel.data_shm.shm_key}")
            if not self.input_channel.data_shm.attach(
                self.input_channel.data_shm.shm_key
            ):
                logging.error(
                    "Failed to reattach to input_sync.data_shm after reallocation"
                )
                raise RuntimeError(
                    "Failed to reattach shared memory after reallocation"
                )

            # Acknowledge reallocation
            input_sync.signal(IpcSignal.ACK)

            # Continue waiting for actual data
            logging.debug("Reattached to input_sync.data_shm, waiting for data")
            return self._ipc_wait_for_message(timeout_ms)

        if signal == IpcSignal.NOTICE:
            # Data ready, read from recv SHM
            data = self.input_channel.data_shm.read()
            if data is None:
                logging.warning(
                    "External Processor couldn't receive message from AI Manager!"
                )
                # Signal AI Manager that response is ready
                input_sync.signal(IpcSignal.ACK)
                return None

            logging.debug(f"Read data: {len(data)} bytes")
            # Acknowledge receipt
            input_sync.signal(IpcSignal.ACK)
            return data
        elif signal == IpcSignal.EXIT:
            # Graceful exit signal. Still returns None like a timeout (unchanged
            # contract for existing callers), but sets exit_requested so a loop
            # can tell the two apart and run its shutdown path instead of
            # looping until the AI Manager kills the process.
            logging.info("Received graceful exit signal from server")
            self.exit_requested = True
            return None
        else:
            logging.warning(f"Unexpected signal: {signal}")
            return None

    def parse_objects_message(
        self, data: bytes
    ) -> Optional[ObjectsPostProcessorMessage]:
        return ObjectsPostProcessorMessage.from_bytes(data)

    def write_objects_message(
        self, objects_message: ObjectsPostProcessorMessage
    ) -> Optional[bytes]:
        return objects_message.to_bytes()

    def parse_tensor_message(self, data: bytes) -> Optional[TensorMessage]:
        return TensorMessage.from_bytes(data)

    def write_tensor_message(self, msg: TensorMessage) -> Optional[bytes]:
        return msg.to_bytes()

    def write_image_message(self, msg: ImagePreprocessorMessage) -> Optional[bytes]:
        return msg.to_bytes()

    def parse_image_message(self, data: bytes) -> Optional[ImagePreprocessorMessage]:
        return ImagePreprocessorMessage.from_bytes(data)

    def send_response(self, data: bytes) -> bool:
        if self.use_tcp is True:
            return self._tcp_send_response(data)
        else:
            return self._ipc_send_response(data)

    def _tcp_send_response(self, data: bytes) -> bool:
        try:
            # Send header and payload
            header_message = struct.pack("<I", len(data))
            self.tcp_connection.sendall(header_message + data)
        except (ConnectionResetError, OSError):
            print("Warning! TCP connection lost. Could not send response message.")
            self.tcp_connection.close()
            self.tcp_connection = None
            return False
        return True

    def _ipc_send_response(self, data: bytes) -> bool:
        """Send response message to server.

        Handles SHM reallocation if needed.

        Returns: True on success
        """
        output_channel = self.output_channel
        self._drain_stale_signals(output_channel, "output")

        # Try to write to send SHM, handling reallocation if needed
        try:
            if not self.output_channel.data_shm.write(data):
                logging.error(f"Failed to write {len(data)}-byte response to SHM")
                return False
        except BufferError as e:
            # Buffer too small, grow our output_sync.data_shm in place under the same key
            logging.info(f"SHM reallocation needed: {e}")

            if not self.output_channel.data_shm.resize(HEADER_BYTES + len(data)):
                logging.error("Failed to reallocate SHM, response not sent")
                return False

            # Write data to resized SHM
            try:
                if not self.output_channel.data_shm.write(data):
                    logging.error("Failed to write response to resized SHM")
                    return False
            except Exception as e:
                logging.error(f"Failed to write to resized SHM: {e}")
                return False

            # Send REALLOC signal to notify C++ of reallocation (same key, just resized)
            logging.debug("Notifying C++ of SHM reallocation via REALLOC signal")
            if not output_channel.signal(IpcSignal.REALLOC):
                logging.error("Failed to send reallocation signal")
                return False

            # Wait for C++ acknowledgment (signal 0)
            response = output_channel.wait(5000)
            if response != IpcSignal.ACK:
                logging.error(f"Expected ACK (0) for reallocation, got {response}")
                return False

            logging.info("C++ acknowledged SHM reallocation")

        # Signal server that data is ready
        if not output_channel.signal(IpcSignal.NOTICE):
            return False

        # Wait for server acknowledgement
        while True:
            signal = output_channel.wait(1000)

            if signal is None:
                logging.warning("Timeout waiting for acknowledgement")
                return False

            if signal == -10:
                logging.error("Connection lost while waiting for acknowledgement")
                raise RuntimeError("IPC connection lost during send_response")

            if signal == IpcSignal.ACK:
                # Acknowledgement received
                logging.debug("Server acknowledged message received.")
                return True
            else:
                logging.warning(f"Unexpected acknowledgement signal: {signal}")
                return False

    def cleanup(self):
        """Clean up resources."""
        self.input_channel.close()
        self.output_channel.close()


# Role-specific names for the one ExternalProcessor class. Pre- and post-processors
# are the same concrete class; each example uses the name that matches its role.
ExternalPostprocessor = ExternalProcessor
ExternalPreprocessor = ExternalProcessor
