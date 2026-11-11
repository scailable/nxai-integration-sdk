# What the Post-processor Returns to the AI Manager

A post-processor sends back a single MessagePack-encoded message after processing each frame — an `ObjectsPostProcessorMessage` built from the parsed input message. The AI Manager merges it downstream into the pipeline.

If `NoResponse: true` is set in the processor's JSON entry, no response is expected and the processor must not send one.

---

## Transport

The response is serialized and sent back on the same long-lived IPC channel pair the input arrived on (see `input_from_ai_manager.md` for the transport itself):

```python
response_bytes = processor.write_objects_message(processed_msg)
if not processor.send_response(response_bytes):
    logger.error("Failed to send response")
```

`write_objects_message()` calls `ObjectsPostProcessorMessage.to_bytes()`, which handles MessagePack encoding. `send_response()` writes the bytes to shared memory and signals the AI Manager (or, in `--port` dev mode, sends them length-prefixed over the TCP socket) — the same call regardless of which transport `initialize_from_args()` selected.

---

## The response must echo `DeviceID` and `Timestamp`

A response must carry the `DeviceID` and `Timestamp` of the message it answers, **unchanged**. A response that carries a different pair is discarded by the AI Manager *silently* — no log line on either side. (Only a response the AI Manager cannot read the two fields out of at all is logged, as `Could not read DeviceID/Timestamp from postprocessor <name> response`. A well-formed response carrying the wrong values produces nothing at all.)

This is how the AI Manager knows which camera a response belongs to. Every camera assigned to the same post-processor shares one process, so several cameras' requests are in flight on the one channel pair at once. The AI Manager matches each response to its request on the pair `DeviceID` + `Timestamp`; a response whose pair matches no outstanding request is dropped. Echo both fields.

Returning the message that was parsed satisfies this for free, which is what every example here does:

```python
objects_message = processor.parse_objects_message(message_bytes)
# ... mutate objects_message ...
processor.send_response(processor.write_objects_message(objects_message))
```

The mistake to avoid is answering with a **freshly constructed** message. `ObjectsPostProcessorMessage()` starts at `timestamp = 0` and `device_id = ""`, and neither is filled in for you on the way out — the serializer writes back whatever the attributes hold:

```python
# WRONG: the AI Manager drops this, without logging why
response = nxai_postprocessor_sdk.ObjectsPostProcessorMessage()
response.objects["person"] = detections
processor.send_response(processor.write_objects_message(response))
```

If a processor does need to build its response from a new object, copy both fields across first:

```python
response.timestamp = objects_message.timestamp
response.device_id = objects_message.device_id
```

This applies to post-processor responses. Pre-processor responses are not matched this way: the AI Manager sends a frame to a pre-processor and blocks on that one answer, so nothing is correlated.

---

## What the processor can add or modify

`ObjectsPostProcessorMessage` (`nxai_postprocessor_sdk.py`) is the same class the input message was parsed into (see `input_from_ai_manager.md`). The processor is free to modify any field before sending it back. The fields processors most commonly write are described below.

### Bounding boxes and attributes: `Objects`

`msg.objects` is a `dict[str, list[ObjectMetadata]]` — class name to a list of per-object structs. Each `ObjectMetadata` carries normalized `[0,1]` coordinates, not pixels:

```python
obj = nxai_postprocessor_sdk.ObjectMetadata()
obj.x = 0.1
obj.y = 0.1
obj.width = 0.3
obj.height = 0.3
obj.confidence = 0.95
obj.attributes["source"] = "postprocessor-example"

msg.objects.setdefault("test", []).append(obj)
```

- **`x`, `y`, `width`, `height`** (float) — normalized `[0,1]`, relative to frame `Width`/`Height`. There is no pixel-space form on v5.
- **`confidence`** (float) — detection confidence.
- **`attributes`** (`dict[str, str]`) — arbitrary key-value pairs attached to this one object, displayed in the NX object details panel. Unlike v4's `ObjectsMetaData` (parallel `AttributeKeys`/`AttributeValues` lists), each object carries its own attribute dict directly — no index bookkeeping.
- **`track_id`** / **`new_track_id`** (UUID string, `ID`/`NewID` on the wire) — set exactly one: `new_track_id` to start a new track with a chosen UUID, `track_id` to continue an existing one across frames. Leaving both `None` serializes an object with no `ID`/`NewID` at all.
- **`remove_attributes`** (`list[str]`, `removeAttributes` on the wire, omitted when empty) — attribute names the engine deletes from the object. Applied after `attributes`, so a name in both ends up removed.
- **Omitting the bounding box and confidence** — on an object with `track_id` set, any of `x`, `y`, `width`, `height`, `confidence` left as `None` is left off the wire and the engine keeps its current value (the bounding box is only applied when all four are present). A `new_track_id` object still needs all five, or the engine drops it. `ObjectMetadata.attribute_update(track_id)` builds such an attribute-only entry.

#### Editing the original object from a Feature Extraction stage

A processor assigned to a model chained in Feature Extraction mode gets the parent object's track ID as `OriginalObjectID`, and nothing else about it. To change that object's attributes, return an attribute-only entry with `track_id` set to `OriginalObjectID`. The engine matches it to the original object by ID even when it is returned under a different class key, though using the parent's class keeps it consistent with the pipeline's visualisation settings. `InferenceMessage.update_original_object()` in `message_processing_utils` does this; see `postprocessor-python-anpr-example/` for the worked example.

Any class name added to `msg.objects` must be declared in the `Objects` array of the processor's JSON entry, or the UI will silently drop it.

To remove all objects for a class, set its list to empty (`msg.objects["car"] = []`) or delete the key. To remove a single object, delete it from the class's list — `ObjectMetadata` instances aren't parallel-array indexed, so no companion structure needs updating.

See `postprocessor-python-example/postprocessor-python-example.py` for the worked example (adds a `python_example`-class object, reusing its `track_id` across frames so it doesn't flicker as a new track every call).

### Events: `Events`

`msg.events` is a `list[EventMetadata]`. Each has two string fields — **no `Caption`**, unlike v4:

```python
event = nxai_postprocessor_sdk.EventMetadata()
event.type_id = "ex.example.event"
event.description = "Person detected in restricted area"
msg.events.append(event)
```

- **`type_id`** (`ID` on the wire) — the event id. Must be declared in the `Events` array of the processor's JSON entry (e.g. `"ex.example.event"`), or the event is dropped.
  **Emit it verbatim.** The AI Manager copies `Events[].ID` into the DeviceAgent manifest unprefixed and relays emitted ids verbatim, so the JSON id and the emitted `type_id` are the same bare string. This is the opposite of object classes, which the engine prefixes with `nx.nxai.`. Adding the prefix yourself puts an id in the response that is not in the manifest, and the mediaserver drops it while logging "Unable to find Event Type with id ...".
- **`description`** (`Description` on the wire) — displayed as the event body. Can include dynamic information (detected plate number, count, etc.).

Multiple events can be appended in a single frame.

See `postprocessor-python-events-example/postprocessor-python-events-example.py` for the worked example, and its colocated `external_postprocessors.json` for the bare-id declaration.

### Best shots: `BestShots`

`msg.best_shots` is a `list[BestShotMetadata]` — the only way a post-processor can nominate the thumbnail the VMS shows for a track:

```python
best_shot = nxai_postprocessor_sdk.BestShotMetadata()
best_shot.track_id = target.track_id
best_shot.x, best_shot.y, best_shot.width, best_shot.height = 0.125, 0.0625, 0.03125, 0.03125
msg.best_shots.append(best_shot)
```

- **`track_id`** (`TrackId` on the wire) — required, the UUID of an **already-tracked** object, i.e. one whose `track_id` is set because it was seen on a previous frame. An entry names an existing tracker-assigned UUID; it never mints one, so an object detected for the first time this frame (`new_track_id` only) is not eligible. An entry whose `TrackId` is missing or not a string is skipped with `Warning! BestShots entry missing or invalid TrackId. Skipping.` in the plugin log — the other entries still apply.
- **`x`, `y`, `width`, `height`** (float) — optional, normalized `[0,1]` like `ObjectMetadata`'s. Set all four to crop the shot to that rectangle; leave them `None` and the engine leaves the packet's bounding box unset.

`BestShots` is optional and additive: an empty `msg.best_shots` serializes no `BestShots` key at all, and an absent key is a no-op. The engine builds one best-shot packet per entry, stamped with the current frame's timestamp. Nothing has to be declared in the processor's JSON entry for this — unlike object classes and event ids.

**Sending `BestShots` is not what gives a track a thumbnail.** Every track already gets a real best shot with a non-empty bounding box, generated by the *server*, whether or not a post-processor ever sends an entry — the AI Manager does not set `noAutoBestShots`. What `BestShots` buys is control over **which** rectangle and frame becomes the thumbnail, in place of the server's own choice. A processor that merely wants thumbnails to exist needs to do nothing.

See `postprocessor-python-bestshots-example/postprocessor-python-bestshots-example.py` for the worked example (tags an already-tracked `car`, gated by an `emit_best_shots` Config flag).

---

## Minimal response example

The simplest valid response is to send the parsed message back with no changes:

```python
objects_message = processor.parse_objects_message(message_bytes)
response_bytes = processor.write_objects_message(objects_message)
processor.send_response(response_bytes)
```

A response that uses both output types:

```python
# 1. Add a bounding box for a new class
zone = nxai_postprocessor_sdk.ObjectMetadata()
zone.new_track_id = str(uuid.uuid4())
zone.x, zone.y, zone.width, zone.height = 0.2, 0.1, 0.5, 0.5
zone.confidence = 1.0
msg.objects.setdefault("zone_violation", []).append(zone)

# 2. Attach an attribute to every detected "person"
for person in msg.objects.get("person", []):
    person.attributes["Zone"] = "Restricted"

# 3. Emit an event
event = nxai_postprocessor_sdk.EventMetadata()
event.type_id = "security.zone_violation"
event.description = "Person detected in restricted area"
msg.events.append(event)

response_bytes = processor.write_objects_message(msg)
processor.send_response(response_bytes)
```

---

## Fire-and-forget (`NoResponse: true`)

When the processor's JSON entry has `NoResponse: true`, skip `write_objects_message()`/`send_response()` entirely — the AI Manager does not wait for a reply:

```python
objects_message = processor.parse_objects_message(message_bytes)
handle_message(objects_message)  # side effect only, no response sent
```

See `postprocessor-python-noresponse-example/postprocessor-python-noresponse-example.py`.

---

## Evidence in the repo

- Message class and encoding: `nxai-utilities/nxai_postprocessor_sdk.py` (`ObjectsPostProcessorMessage.to_bytes`, `ObjectMetadata.to_dict`, `EventMetadata.to_dict`, `ExternalProcessor.write_objects_message`/`send_response`).
- Object mutation: `postprocessor-python-example/postprocessor-python-example.py`.
- Events emission and the verbatim event-id rule: `postprocessor-python-events-example/postprocessor-python-events-example.py` and its `external_postprocessors.json`.
- Best shots: `nxai-utilities/nxai_postprocessor_sdk.py` (`BestShotMetadata`), `postprocessor-python-bestshots-example/`.
- Attribute attachment: `postprocessor-python-confidences-example/`, `postprocessor-python-settings-example/`.
- Fire-and-forget: `postprocessor-python-noresponse-example/`, `postprocessor-python-edgeimpulse-example/`.
- `DeviceID`/`Timestamp` echo: `nxai-utilities/nxai_postprocessor_sdk.py` (`ObjectsPostProcessorMessage.from_dict` reads both into plain attributes, `to_bytes` writes them back from those attributes) and `nxai-utilities/external_processor_utilities.cpp` for the C++ binding.
