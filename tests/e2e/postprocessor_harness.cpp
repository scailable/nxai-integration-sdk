/**
 * End-to-end test harness ("AI Manager emulator") for the reference postprocessors.
 *
 * Plays the role the NX AI Manager plays at runtime: it is the IPC HOST. It creates the
 * shared-memory channels, launches a postprocessor as a child over the real transport,
 * sends it a request message, reads the response, and asserts the processor applied the
 * reference example's mutation. A mandatory --check <name> selects the scenario; each
 * scenario fully owns the wire codec it sends, parses, and asserts:
 *   objects — ObjectsPostProcessorMessage; expects one <expected_class> object with known
 *     fields and attributes["source"] == <expected_source>.
 *   realloc — the objects scenario over channels created at the engine's own 1024-byte
 *     initial capacity, seeded with enough "person" objects that both the request and the
 *     echoed response exceed it. This is the only scenario that drives the SHM
 *     reallocation handshake in either direction: every other one fits inside the initial
 *     buffer, which is why the whole suite stayed green while the Windows realloc paths
 *     were broken (on both the C++ and the Python side).
 *   tensors — TensorsMessage; expects the tensors replaced with the single known
 *     "bboxes-format:xyxysc;0:cpp_tensors_example" float tensor.
 *   settings — ObjectsPostProcessorMessage seeded with a UserSettings map and one
 *     "person" object; expects the forwarded setting applied as attributes["reviewed"] ==
 *     "true" on that object. Exercises the engine->processor forwarding path.
 *   counter — ObjectsPostProcessorMessage seeded with a live min_object_count of 3
 *     (UserSettings) and exactly 3 "person" objects — below the processor's hardcoded
 *     default (5), so the event only fires if the live setting is read; expects an event
 *     with type_id == "object.counter.threshold" (declared and emitted
 *     unprefixed) in the response. Exercises the object-counter-threshold postprocessor's
 *     revert to live UserSettings.
 *   height — ObjectsPostProcessorMessage seeded with live frame_fraction_per_meter=0.5 and
 *     max_height_cm=100 (UserSettings, both distinct from the processor's hardcoded
 *     defaults 0.25 / 250) and one "person" object of normalized height 0.7 (-> 1.4 m at
 *     the live 0.5 fraction, above the live 1.0 m limit); expects attributes["Height"] ==
 *     "1.40 m" on that object and a "person.height.exceeded" event (declared and
 *     emitted unprefixed). Exercises the height-estimator postprocessor's
 *     revert to live UserSettings.
 *   objects-sender — ObjectsPostProcessorMessage seeded with 4 reference points (UserSettings
 *     point0..3.figure/latitude/longitude, the figures as the JSON strings the engine sends),
 *     a server_address nothing listens on and a non-default topic_path, plus one "car" and
 *     one "person" object; expects attributes["Coordinates"] on both objects equal to the
 *     lat/lon of their bbox centers under the seeded points. Exercises the objects-sender
 *     postprocessor's homography over the real IPC channels. Its POST to the bridge fails
 *     with a refused connection, which the processor logs and ignores, as in dashboard.
 *   dashboard — ObjectsPostProcessorMessage seeded with one "test" object of known fields;
 *     expects the same object returned field-for-field unmodified (no objects, attributes,
 *     or events added). Exercises the web-dashboard postprocessor's "forwards unmodified"
 *     contract; the processor's web-app POST side-effect fails harmlessly (no server) and
 *     is ignored.
 *   skip-feature-extraction — ObjectsPostProcessorMessage seeded with no objects; expects a
 *     "chain-test" class with two objects: attributes["role"] == "gated" carrying the
 *     "skipFeatureExtraction": true wire field, and attributes["role"] == "ungated" with the
 *     field entirely absent (parsed skip_feature_extraction defaults to false). This is a
 *     WIRE-CONTRACT check only: it proves the
 *     fixture postprocessor emits the exact per-object shape the real engine's deserializer
 *     reads — it does NOT run the real engine and
 *     cannot observe whether chained FEATURE_EXTRACTION inference actually gets skipped (that
 *     behavior lives entirely in the engine).
 *   image — ObjectsPostProcessorMessage seeded with a root ImageData block carrying a
 *     well-formed 2x2 RGB uint8 raster (12 bytes); expects one "python_image_example" object
 *     at normalized box (0.1, 0.1, 0.1, 0.1), confidence 1.0, and
 *     attributes["ImageDecodeOk"] == "True". Exercises the ReceiveInputImage contract end to
 *     end.
 *   preproc-tensor — TensorsMessage seeded with a known "nms_sensitivity-" float tensor
 *     (0.5), an unrelated "input" tensor (0.25), and a forwarded UserSettings map
 *     of {"nmsoverride": "0.3"}; expects "nms_sensitivity-" replaced with that
 *     value (0.3, distinct from the compiled-in 0.8 default, so the assertion covers the
 *     processor's live-settings read site) and "input" returned unchanged. Exercises the
 *     tensor preprocessor.
 *   preproc-image — ImagePreprocessorMessage seeded with a known 4x4 three-channel uint8 CHW
 *     image (elements 0..47) and a forwarded UserSettings map of {"mirrorimage": "false"}
 *     (distinct from the compiled-in `true` default, so the assertion covers the processor's
 *     live-settings read site) and expects the image returned byte-for-byte and
 *     dimension-for-dimension unchanged -- proving the disabled flag was actually read, not
 *     just accepted on the wire. Exercises the image preprocessor.
 *   preproc-image-mirror — same seeded 4x4 three-channel uint8 CHW image, but with no
 *     UserSettings seeded at all, relying on the compiled-in `true` default, and expects a
 *     true horizontal mirror at unchanged (full) resolution: each row of each channel plane
 *     reversed, elements {3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17,
 *     16, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28, 35, 34, 33, 32, 39, 38, 37, 36, 43,
 *     42, 41, 40, 47, 46, 45, 44} and Data length == Height * Width * channels for the
 *     unchanged dimensions. Exercises the mirroring algorithm itself, which
 *     the disabled-path preproc-image scenario cannot reach.
 *   preproc-image-realloc — the mirror scenario at the size a live IMAGE preprocessor
 *     actually runs at: a 3840x2160 RGB uint8 CHW frame (~24.9 MB) over the same
 *     1024-byte channels the AI Manager creates, so the reallocation handshake runs in
 *     both directions on a payload of real size. The 4x4 mirror scenario reallocates
 *     nothing at all -- it fits inside the initial channel -- so nothing else in the
 *     suite covers an oversized IMAGE payload.
 *   preproc-image-realloc-multi — the same 4K frame, but three of them sent back to back
 *     without draining the response in between, which is how the AI Manager drives a
 *     preprocessor: nxai_ipc_send() waits only for the input acknowledgement, and once a
 *     send has failed the plugin returns early and never reads a response at all. The
 *     single-frame scenario above always finds the processor parked in its wait, so it
 *     cannot see a processor that is still busy with the previous frame when the next
 *     frame's reallocation signal arrives -- the shape a live run dies in.
 *     Reports the outcome of every send and receive rather than stopping at the first
 *     failure, so the frame the two sides desynchronise on is visible.
 *   confidences — ObjectsPostProcessorMessage seeded with one "person" object of
 *     confidence 0.876; expects attributes["Confidence"] == "0.88" on that object
 *     (bare `str(round(confidence, 2))`, no fixed decimal width -- the v4-preserving
 *     format the confidences postprocessor must not regress).
 *   events — ObjectsPostProcessorMessage seeded with no objects; expects exactly one
 *     event with type_id == "ex.example.event", byte-identical to the `Events[].ID`
 *     declared in the processor's external_postprocessors.json (the
 *     AI Manager relays emitted event ids verbatim, so a prefixed id
 *     is silently dropped). Also asserts the id does NOT carry an "nx.nxai." prefix --
 *     object classes are the opposite (engine-prefixed) -- pinning the asymmetry.
 *   bestshots — ObjectsPostProcessorMessage seeded with one already-tracked "car"
 *     object (track_id set -- BestShots names an existing track, it
 *     never mints one) and the processor launched with `--emit_best_shots true`;
 *     expects exactly one entry in the response's top-level BestShots array naming
 *     that track_id with the fixture's sentinel bounding box (0.125, 0.0625, 0.03125,
 *     0.03125). Exercises the emit path of the BestShots fixture.
 *   bestshots-no-emit — identical seed, but the processor launched with no
 *     `emit_best_shots` flag (default "false"); expects an empty BestShots array.
 *     Mirrors the two live Check scenarios' emit/no-emit split.
 *   edgeimpulse — sends 3 consecutive ObjectsPostProcessorMessage frames, each
 *     carrying a root ImageData block (a real 2x2 RGB uint8 raster) and one object of
 *     confidence 0.1 (below the fixture ini's p_value=0.4, so every frame queues a
 *     training sample -- exercising the tracker-confidence upload-trigger path and
 *     the ImageData -> JPEG conversion) and, after each one, asserts nothing is
 *     written to the output channel (NoResponse: true) within a short wait; after the
 *     third frame, asserts the child process is still alive. The multi-frame
 *     requirement mirrors no-response: a processor that crashes converting the
 *     seeded image would die silently with no output-channel evidence otherwise. The
 *     processor is launched with `--config_file <fixture ini>` rather than the
 *     shipped plugin.edgeimpulse.ini.example, mainly because the example enables
 *     auto_generator, whose timer would queue samples even with the path under test
 *     broken; see edgeimpulse-fixture.ini's header.
 *   no-response — sends 3 consecutive ObjectsPostProcessorMessage frames (the multi-frame
 *     count is the point: a processor that dies after exactly one frame, as an OBJECTS
 *     no-response path once did, would pass a single-frame version of this check)
 *     and, after each one, asserts nothing is written to the output channel within a short
 *     wait. After the third frame, asserts the child process is still alive. Exercises the
 *     NoResponse contract (postprocessor-python-noresponse-example), which has no return
 *     channel at all -- not even the SHM-realloc ACK -- so a processor that mistakenly
 *     replies, or that dies silently, both need a multi-frame probe to be caught.
 *
 *   negative — the only scenario that feeds a processor input it cannot process.
 *     Sends one malformed frame of the selected <kind>, then one well-formed frame, and
 *     asserts the <outcome> the processor is currently observed to have:
 *       missing-keys — structurally valid MessagePack ({"DeviceID": "e2e"}) in which every
 *         key the processor reads is absent.
 *       truncated — the first half of a well-formed frame, stopping mid-structure.
 *       no-image — a well-formed frame with no ImageData block, sent to a processor that
 *         declared ReceiveInputImage: true.
 *     The response to the malformed frame is reported, never asserted; the assertion is
 *     "recovers" (the follow-up frame is answered with a parseable message AND the child is
 *     still running) versus "dies". The expected outcome is an argument because it PINS
 *     today's behavior rather than prescribing it: the v5 main loop is
 *     copy-pasted across the examples and one unprocessable frame currently terminates
 *     several of them. A deliberate change to the error policy will
 *     flip the pinned outcomes -- do not "fix" a processor to turn a "dies"
 *     entry into a "recovers" one here.
 *
 * Because it drives the production SHM/sync path, a single harness works for both the C++
 * and the Python reference processor: both parse the same sync string the host produces.
 *
 * The child is started through the launcher in child_process.h, which owns the platform
 * dependence; the child inherits the sync objects encoded in its argv, and the shared memory
 * is opened by name.
 *
 * Usage: <harness> --check
 *            <objects|tensors|settings|counter|height|objects-sender|dashboard|
 *             skip-feature-extraction|image|
 *             preproc-tensor|preproc-image|preproc-image-mirror|preproc-image-realloc|
 *             preproc-image-realloc-multi|no-response|
 *             confidences|events|bestshots|bestshots-no-emit|edgeimpulse|negative>
 *                  <args...>
 *   objects:              <args...> = <expected_class> <expected_source> <child_program>
 *                                      [child_prefix_args...]
 *   tensors:              <args...> = <child_program> [child_prefix_args...]
 *   settings:             <args...> = <child_program> [child_prefix_args...]
 *   counter:              <args...> = <child_program> [child_prefix_args...]
 *   height:               <args...> = <child_program> [child_prefix_args...]
 *   objects-sender:       <args...> = <child_program> [child_prefix_args...]
 *   dashboard:            <args...> = <child_program> [child_prefix_args...]
 *   skip-feature-extraction: <args...> = <child_program> [child_prefix_args...]
 *   image:                <args...> = <child_program> [child_prefix_args...]
 *   preproc-tensor:       <args...> = <child_program> [child_prefix_args...]
 *   preproc-image:        <args...> = <child_program> [child_prefix_args...]
 *   preproc-image-mirror: <args...> = <child_program> [child_prefix_args...]
 *   preproc-image-realloc:<args...> = <child_program> [child_prefix_args...]
 *   preproc-image-realloc-multi: <args...> = <child_program> [child_prefix_args...]
 *   no-response:          <args...> = <child_program> [child_prefix_args...]
 *   confidences:          <args...> = <child_program> [child_prefix_args...]
 *   events:               <args...> = <child_program> [child_prefix_args...]
 *   bestshots:            <args...> = <child_program> [child_prefix_args...]
 *   bestshots-no-emit:    <args...> = <child_program> [child_prefix_args...]
 *   edgeimpulse:          <args...> = <child_program> [child_prefix_args...]
 *   negative:             <args...> = <missing-keys|truncated|no-image>
 *                                     <recovers|dies> <child_program> [child_prefix_args...]
 *   The harness inserts "<input_sync> <output_sync> --ProcessorName e2e-test"
 *   into <args...> right after <child_program>: everything before a literal "--" token
 *   stays ahead of that insertion (e.g. a Python script path, which must precede it
 *   positionally); everything after "--" is appended behind it instead (scenario
 *   Config flags, e.g. "-- --emit_best_shots true", which the Python SDK's
 *   initialize_from_args() would misread as the sync strings if placed ahead of them).
 *   With no "--" token, every arg stays ahead of the insertion (the shape every
 *   existing scenario without extra Config flags uses).
 *   --check is mandatory; a missing or unknown scenario is a usage error.
 * Exit codes: 0 = pass, 1 = assertion failed, 2 = harness/setup error.
 */

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "child_process.h"
#include "external_processor_utilities.h"
#include "nxai_ipc_utils.h"
#include "oaax/interface.h"

namespace
{

constexpr size_t kInitialShmSize = 64 * 1024;
// realloc scenario: the AI Manager creates its channels at 1024 bytes, so every real
// message of any size crosses the reallocation handshake. Reproducing that number here is
// the point of the scenario -- at kInitialShmSize the seeded frame would have to be
// implausibly large to get there.
constexpr size_t kReallocInitialShmSize = 1024;
// preproc-image-realloc scenario: a 4K RGB uint8 frame, the payload a live IMAGE
// preprocessor gets from a 3840x2160 camera. At ~24.9 MB it crosses the reallocation
// handshake in both directions off the 1024-byte channels, and it is the size at which
// the handshake has to hold up against real page commits rather than a few kilobytes.
constexpr size_t kReallocImageWidth = 3840;
constexpr size_t kReallocImageHeight = 2160;
constexpr size_t kReallocImageChannels = 3;
constexpr size_t kReallocImageBytes
    = kReallocImageWidth * kReallocImageHeight * kReallocImageChannels * sizeof(unsigned char);
// preproc-image-realloc-multi scenario: how many 4K frames to push back to back. Three is
// enough for the second send to land while the processor is still working on the first,
// with a third to show whether the two sides ever resynchronise.
constexpr int kReallocImageFrameCount = 3;
// Enough seeded objects that both the request and the echoed-back response clear 1024
// bytes with no room for doubt: each serialized object costs well over a byte, so this is
// tens of kilobytes in either direction.
constexpr int kReallocObjectCount = 2000;
constexpr int kReadyTimeoutMs = 10000;
// edgeimpulse is the one processor whose import graph (numpy, pillow, the edgeimpulse
// client) costs more than the standard ready window before its first line runs:
// measured at 12.8s on ext4 and 52s on a /mnt/c checkout under WSL. The wait is a test
// bound, not an engine one -- the AI Manager's own window is a separate number -- so
// widening it here only stops the harness from calling a slow import a dead processor.
constexpr int kHeavyImportReadyTimeoutMs = 90000;
constexpr int kReceiveTimeoutMs = 10000;
constexpr float kEpsilon = 1e-4F;

// no-response scenario: how many frames to drive (the multi-frame count is the point,
// see the scenario doc comment above) and how long to wait, per frame, for a response
// that must never arrive.
constexpr int kNoResponseFrameCount = 3;
constexpr int kNoResponseWaitMs = 2000;

// negative scenario: how long to wait for a response after each of the two frames. Both
// waits are "may or may not arrive" waits, so they are short -- a processor killed by the
// malformed frame would otherwise cost the full receive timeout twice.
constexpr int kNegativeWaitMs = 3000;

bool approx(float a, float b)
{
    return std::fabs(a - b) < kEpsilon;
}

// Assert the response carries exactly the object the reference example appends. The class
// is a parameter because each shipped example injects its own -- there is no shared "test".
bool check_response(const ObjectsPostProcessorMessage& msg, const std::string& expected_class,
    const std::string& expected_source)
{
    const auto it = msg.objects.find(expected_class);
    if (it == msg.objects.end())
    {
        std::cerr << "FAIL: no \"" << expected_class << "\" object class in response\n";
        return false;
    }
    if (it->second.size() != 1)
    {
        std::cerr << "FAIL: expected exactly 1 \"" << expected_class << "\" object, got "
                  << it->second.size() << "\n";
        return false;
    }

    const ObjectData& obj = it->second.front();
    bool ok = true;
    const auto check = [&](bool cond, const char* what)
    {
        if (!cond)
        {
            std::cerr << "FAIL: " << what << "\n";
            ok = false;
        }
    };

    check(approx(obj.x, 0.1F), "x != 0.1");
    check(approx(obj.y, 0.1F), "y != 0.1");
    check(approx(obj.width, 0.3F), "width != 0.3");
    check(approx(obj.height, 0.3F), "height != 0.3");
    check(approx(obj.confidence, 0.95F), "confidence != 0.95");

    const auto attr = obj.attributes.find("source");
    check(attr != obj.attributes.end() && attr->second == expected_source,
        "attributes[\"source\"] mismatch");

    return ok;
}

// Assert the oversized response survived the reallocation intact: the injected object is
// there, and so is every seeded "person". The count matters -- a response written into a
// buffer that was grown on one side only would come back truncated rather than absent, and
// checking the injected class alone would pass on the truncated remainder.
bool check_realloc_response(const ObjectsPostProcessorMessage& msg,
    const std::string& expected_class, const std::string& expected_source)
{
    if (!check_response(msg, expected_class, expected_source))
        return false;

    const auto it = msg.objects.find("person");
    const size_t returned = it == msg.objects.end() ? 0 : it->second.size();
    if (returned != static_cast<size_t>(kReallocObjectCount))
    {
        std::cerr << "FAIL: expected " << kReallocObjectCount << " seeded \"person\" objects "
                  << "back, got " << returned << "\n";
        return false;
    }
    return true;
}

// Assert the forwarded setting landed as an attribute on the seeded "person" object.
bool check_settings_response(const ObjectsPostProcessorMessage& msg)
{
    const auto it = msg.objects.find("person");
    if (it == msg.objects.end())
    {
        std::cerr << "FAIL: no \"person\" object class in response\n";
        return false;
    }
    if (it->second.size() != 1)
    {
        std::cerr << "FAIL: expected exactly 1 \"person\" object, got " << it->second.size()
                  << "\n";
        return false;
    }

    const ObjectData& obj = it->second.front();
    const auto attr = obj.attributes.find("reviewed");
    if (attr == obj.attributes.end() || attr->second != "true")
    {
        std::cerr << "FAIL: attributes[\"reviewed\"] != \"true\"\n";
        return false;
    }
    return true;
}

// Assert the response carries exactly the tensor the C++ tensors example writes.
bool check_tensors_response(const TensorsMessage& msg)
{
    if (msg.tensors.size() != 1)
    {
        std::cerr << "FAIL: expected exactly 1 tensor in response, got " << msg.tensors.size()
                  << "\n";
        return false;
    }

    const TensorData& tensor = msg.tensors.front();
    bool ok = true;
    const auto check = [&](bool cond, const char* what)
    {
        if (!cond)
        {
            std::cerr << "FAIL: " << what << "\n";
            ok = false;
        }
    };

    check(tensor.name == "bboxes-format:xyxysc;0:cpp_tensors_example", "tensor name mismatch");
    check(tensor.data_type == DATA_TYPE_FLOAT, "tensor data_type != float");
    check(tensor.shape == std::vector<uint32_t>{1, 6}, "tensor shape != {1, 6}");

    constexpr float kExpected[] = {100.0F, 100.0F, 200.0F, 200.0F, 1.0F, 0.0F};
    if (tensor.data.size() != sizeof(kExpected))
    {
        std::cerr << "FAIL: tensor data size != " << sizeof(kExpected) << "\n";
        return false;
    }
    float values[6];
    std::memcpy(values, tensor.data.data(), sizeof(values));
    for (int i = 0; i < 6; ++i)
        check(approx(values[i], kExpected[i]), "tensor value mismatch");

    return ok;
}

// Assert the response contains an event with type_id == "object.counter.threshold"
// (postprocessors emit the id exactly as their JSON declares it).
bool check_counter_response(const ObjectsPostProcessorMessage& msg)
{
    for (const auto& event : msg.events)
    {
        if (event.type_id == "object.counter.threshold")
            return true;
    }
    std::cerr << "FAIL: no event with type_id \"object.counter.threshold\" in response\n";
    return false;
}

// Assert the seeded over-limit person got attributes["Height"] == "1.40 m" (proves both
// live UserSettings values were read, not the processor's hardcoded defaults) and the
// response carries a "person.height.exceeded" event (unprefixed id).
bool check_height_response(const ObjectsPostProcessorMessage& msg)
{
    const auto it = msg.objects.find("person");
    if (it == msg.objects.end())
    {
        std::cerr << "FAIL: no \"person\" object class in response\n";
        return false;
    }
    if (it->second.size() != 1)
    {
        std::cerr << "FAIL: expected exactly 1 \"person\" object, got " << it->second.size()
                  << "\n";
        return false;
    }

    bool ok = true;
    const ObjectData& obj = it->second.front();
    const auto attr = obj.attributes.find("Height");
    if (attr == obj.attributes.end() || attr->second != "1.40 m")
    {
        std::cerr << "FAIL: attributes[\"Height\"] != \"1.40 m\"\n";
        ok = false;
    }

    bool fired = false;
    for (const auto& event : msg.events)
    {
        if (event.type_id == "person.height.exceeded")
            fired = true;
    }
    if (!fired)
    {
        std::cerr << "FAIL: no event with type_id \"person.height.exceeded\" in response\n";
        ok = false;
    }
    return ok;
}

// Assert the one seeded object of `cls` carries attributes["Coordinates"] == "<lat>, <lon>"
// with the expected values.
bool check_coordinates(
    const ObjectsPostProcessorMessage& msg,
    const std::string& cls,
    double expected_lat,
    double expected_lon)
{
    const auto it = msg.objects.find(cls);
    if (it == msg.objects.end() || it->second.size() != 1)
    {
        std::cerr << "FAIL: expected exactly 1 \"" << cls << "\" object in response\n";
        return false;
    }
    const auto& attributes = it->second.front().attributes;
    const auto attr = attributes.find("Coordinates");
    if (attr == attributes.end())
    {
        std::cerr << "FAIL: no attributes[\"Coordinates\"] on the \"" << cls << "\" object\n";
        return false;
    }
    double lat = 0.0;
    double lon = 0.0;
    if (std::sscanf(attr->second.c_str(), "%lf, %lf", &lat, &lon) != 2
        || std::fabs(lat - expected_lat) > kEpsilon || std::fabs(lon - expected_lon) > kEpsilon)
    {
        std::cerr << "FAIL: \"" << cls << "\" attributes[\"Coordinates\"] == \"" << attr->second
                  << "\", expected " << expected_lat << ", " << expected_lon << "\n";
        return false;
    }
    return true;
}

// The seed points map the normalized frame to lat = 10 + 2 * x, lon = 20 + 4 * y, so the
// "car" centered at (0.5, 0.25) lands on (11, 21) and the "person" at (0.25, 0.75) on
// (10.5, 23).
bool check_objects_sender_response(const ObjectsPostProcessorMessage& msg)
{
    const bool car_ok = check_coordinates(msg, "car", 11.0, 21.0);
    const bool person_ok = check_coordinates(msg, "person", 10.5, 23.0);
    return car_ok && person_ok;
}

// Assert the dashboard returned the seeded "test" object field-for-field unmodified and
// added no objects, attributes, or events ("forwards unmodified").
bool check_dashboard_response(const ObjectsPostProcessorMessage& msg)
{
    if (msg.objects.size() != 1)
    {
        std::cerr << "FAIL: expected exactly 1 object class, got " << msg.objects.size() << "\n";
        return false;
    }
    const auto it = msg.objects.find("test");
    if (it == msg.objects.end())
    {
        std::cerr << "FAIL: no \"test\" object class in response\n";
        return false;
    }
    if (it->second.size() != 1)
    {
        std::cerr << "FAIL: expected exactly 1 \"test\" object, got " << it->second.size() << "\n";
        return false;
    }

    const ObjectData& obj = it->second.front();
    bool ok = true;
    const auto check = [&](bool cond, const char* what)
    {
        if (!cond)
        {
            std::cerr << "FAIL: " << what << "\n";
            ok = false;
        }
    };

    check(approx(obj.x, 0.1F), "x != 0.1");
    check(approx(obj.y, 0.2F), "y != 0.2");
    check(approx(obj.width, 0.3F), "width != 0.3");
    check(approx(obj.height, 0.4F), "height != 0.4");
    check(approx(obj.confidence, 0.95F), "confidence != 0.95");
    check(obj.attributes.empty(), "object attributes were modified");
    check(msg.events.empty(), "events were added");

    return ok;
}

// Assert the "chain-test" class carries a gated object with skip_feature_extraction == true
// and an ungated object whose absent skipFeatureExtraction field parsed to the false default.
// Wire-contract only -- see the skip-feature-extraction scenario doc comment above.
bool check_skip_feature_extraction_response(const ObjectsPostProcessorMessage& msg)
{
    const auto it = msg.objects.find("chain-test");
    if (it == msg.objects.end())
    {
        std::cerr << "FAIL: no \"chain-test\" object class in response\n";
        return false;
    }
    if (it->second.size() != 2)
    {
        std::cerr << "FAIL: expected exactly 2 \"chain-test\" objects, got " << it->second.size()
                  << "\n";
        return false;
    }

    const ObjectData* gated = nullptr;
    const ObjectData* ungated = nullptr;
    for (const ObjectData& obj: it->second)
    {
        const auto role = obj.attributes.find("role");
        if (role == obj.attributes.end())
            continue;
        if (role->second == "gated")
            gated = &obj;
        else if (role->second == "ungated")
            ungated = &obj;
    }

    bool ok = true;
    if (gated == nullptr)
    {
        std::cerr << "FAIL: no object with attributes[\"role\"] == \"gated\"\n";
        ok = false;
    }
    else if (!gated->skip_feature_extraction)
    {
        std::cerr << "FAIL: gated object's skip_feature_extraction != true\n";
        ok = false;
    }

    if (ungated == nullptr)
    {
        std::cerr << "FAIL: no object with attributes[\"role\"] == \"ungated\"\n";
        ok = false;
    }
    else if (ungated->skip_feature_extraction)
    {
        std::cerr
            << "FAIL: ungated object's skip_feature_extraction != false (absent field must "
               "default false)\n";
        ok = false;
    }

    return ok;
}

// Assert the response carries one "python_image_example" object at the fixed normalized box
// the image example injects, with attributes["ImageDecodeOk"] == "True" -- the processor
// decoded the seeded raster cleanly.
bool check_image_response(const ObjectsPostProcessorMessage& msg)
{
    const auto it = msg.objects.find("python_image_example");
    if (it == msg.objects.end())
    {
        std::cerr << "FAIL: no \"python_image_example\" object class in response\n";
        return false;
    }
    if (it->second.size() != 1)
    {
        std::cerr << "FAIL: expected exactly 1 \"python_image_example\" object, got "
                  << it->second.size() << "\n";
        return false;
    }

    const ObjectData& obj = it->second.front();
    bool ok = true;
    const auto check = [&](bool cond, const char* what)
    {
        if (!cond)
        {
            std::cerr << "FAIL: " << what << "\n";
            ok = false;
        }
    };

    check(approx(obj.x, 0.1F), "x != 0.1");
    check(approx(obj.y, 0.1F), "y != 0.1");
    check(approx(obj.width, 0.1F), "width != 0.1");
    check(approx(obj.height, 0.1F), "height != 0.1");
    check(approx(obj.confidence, 1.0F), "confidence != 1.0");

    const auto attr = obj.attributes.find("ImageDecodeOk");
    check(attr != obj.attributes.end() && attr->second == "True",
        "attributes[\"ImageDecodeOk\"] != \"True\"");
    check(obj.attributes.find("ImageDecodeError") == obj.attributes.end(),
        "attributes[\"ImageDecodeError\"] present on a well-formed frame");

    // A single-shot request has no prior track: the engine only creates (and later
    // reports) a track when new_track_id is set. Neither field being set is the
    // exact bug that made the real v5 engine silently drop this object.
    check(obj.new_track_id.has_value() && !obj.new_track_id->empty(),
        "new_track_id missing or empty on first-seen object");
    check(!obj.track_id.has_value(), "track_id set on first-seen object");

    return ok;
}

// Assert the seeded "person" object got attributes["Confidence"] == "0.88" -- the
// bare `str(round(confidence, 2))` format (no fixed decimal width) the confidences
// postprocessor must preserve from v4.
bool check_confidences_response(const ObjectsPostProcessorMessage& msg)
{
    const auto it = msg.objects.find("person");
    if (it == msg.objects.end())
    {
        std::cerr << "FAIL: no \"person\" object class in response\n";
        return false;
    }
    if (it->second.size() != 1)
    {
        std::cerr << "FAIL: expected exactly 1 \"person\" object, got " << it->second.size()
                  << "\n";
        return false;
    }

    const ObjectData& obj = it->second.front();
    const auto attr = obj.attributes.find("Confidence");
    if (attr == obj.attributes.end() || attr->second != "0.88")
    {
        std::cerr << "FAIL: attributes[\"Confidence\"] != \"0.88\" (got \""
                  << (attr == obj.attributes.end() ? "<absent>" : attr->second) << "\")\n";
        return false;
    }
    return true;
}

// Assert exactly one event with type_id == "ex.example.event", byte-identical to the
// `Events[].ID` declared in the events example's external_postprocessors.json
// (the AI Manager relays emitted event ids verbatim).
// Also asserts the id carries no "nx.nxai." prefix -- object classes are the opposite
// (engine-prefixed) -- pinning that asymmetry in the same case.
bool check_events_response(const ObjectsPostProcessorMessage& msg)
{
    if (msg.events.size() != 1)
    {
        std::cerr << "FAIL: expected exactly 1 event, got " << msg.events.size() << "\n";
        return false;
    }

    bool ok = true;
    const std::string& type_id = msg.events.front().type_id;
    if (type_id != "ex.example.event")
    {
        std::cerr << "FAIL: event type_id \"" << type_id << "\" != \"ex.example.event\"\n";
        ok = false;
    }
    if (type_id.rfind("nx.nxai.", 0) == 0)
    {
        std::cerr << "FAIL: event type_id carries an \"nx.nxai.\" prefix -- events must be"
                     " emitted exactly as declared, unlike object classes\n";
        ok = false;
    }
    return ok;
}

// Assert the response's top-level BestShots array names the seeded track with the
// fixture's sentinel bounding box (BestShots names an existing track, so the
// track_id must round-trip the one the harness seeded, not a freshly minted one).
bool check_bestshots_response(const ObjectsPostProcessorMessage& msg)
{
    if (msg.best_shots.size() != 1)
    {
        std::cerr << "FAIL: expected exactly 1 best shot, got " << msg.best_shots.size() << "\n";
        return false;
    }

    const BestShotData& best_shot = msg.best_shots.front();
    bool ok = true;
    const auto check = [&](bool cond, const char* what)
    {
        if (!cond)
        {
            std::cerr << "FAIL: " << what << "\n";
            ok = false;
        }
    };

    check(best_shot.track_id == "e2e-track-1", "best shot track_id != seeded track");
    check(best_shot.x.has_value() && approx(*best_shot.x, 0.125F), "best shot x != 0.125");
    check(best_shot.y.has_value() && approx(*best_shot.y, 0.0625F), "best shot y != 0.0625");
    check(best_shot.width.has_value() && approx(*best_shot.width, 0.03125F),
        "best shot width != 0.03125");
    check(best_shot.height.has_value() && approx(*best_shot.height, 0.03125F),
        "best shot height != 0.03125");

    return ok;
}

// Sibling of check_bestshots_response: with no --emit_best_shots flag (default "false"),
// the fixture must leave BestShots empty even though an eligible tracked car is present.
bool check_bestshots_no_emit_response(const ObjectsPostProcessorMessage& msg)
{
    if (!msg.best_shots.empty())
    {
        std::cerr << "FAIL: expected 0 best shots with emit_best_shots unset, got "
                  << msg.best_shots.size() << "\n";
        return false;
    }
    return true;
}

// Assert the "nms_sensitivity-" tensor was overridden to the value the harness forwarded
// via UserSettings (0.3, not the compiled-in 0.8 default -- so this covers the processor's
// live-settings read site), that the unrelated "input" tensor came back with its own
// distinct value intact, and that the round-trip preserved the tensor set and its shapes.
bool check_preproc_tensor_response(const TensorsMessage& msg)
{
    constexpr float kExpectedOverride = 0.3F;
    constexpr float kUnchangedInput = 0.25F;

    const TensorData* nms = nullptr;
    const TensorData* input = nullptr;
    for (const TensorData& tensor: msg.tensors)
    {
        if (tensor.name == "nms_sensitivity-")
            nms = &tensor;
        else if (tensor.name == "input")
            input = &tensor;
    }

    bool ok = true;
    if (msg.tensors.size() != 2)
    {
        std::cerr << "FAIL: response has " << msg.tensors.size() << " tensors, expected 2\n";
        ok = false;
    }

    // Both tensors are a single float of shape {1}; the Python SDK re-derives the shape and
    // data-type maps from the parsed metadata on write, so assert them on the way back.
    const auto check_tensor = [&ok](const TensorData* tensor, const char* name, float expected)
    {
        if (tensor == nullptr)
        {
            std::cerr << "FAIL: no \"" << name << "\" tensor in response\n";
            ok = false;
            return;
        }
        float value = 0.0F;
        if (tensor->data.size() != sizeof(value))
        {
            std::cerr << "FAIL: \"" << name << "\" tensor data size " << tensor->data.size()
                      << " != " << sizeof(value) << "\n";
            ok = false;
        }
        else
        {
            std::memcpy(&value, tensor->data.data(), sizeof(value));
            if (!approx(value, expected))
            {
                std::cerr << "FAIL: \"" << name << "\" value " << value << " != " << expected
                          << "\n";
                ok = false;
            }
        }
        if (tensor->shape != std::vector<uint32_t>{1})
        {
            std::cerr << "FAIL: \"" << name << "\" shape != {1}\n";
            ok = false;
        }
        if (tensor->data_type != DATA_TYPE_FLOAT)
        {
            std::cerr << "FAIL: \"" << name << "\" data_type != DATA_TYPE_FLOAT\n";
            ok = false;
        }
    };

    check_tensor(nms, "nms_sensitivity-", kExpectedOverride);
    check_tensor(input, "input", kUnchangedInput);

    return ok;
}

// Serialize image elements as the raw uint8 raster the v5 IMAGE path carries: `Data` is
// uint8, one byte per element, so its length is simply Height * Width * channels.
std::vector<char> uint8_raster(const std::vector<unsigned char>& elements)
{
    return std::vector<char>(elements.begin(), elements.end());
}

// Assert the image came back unchanged: the harness seeded a forwarded UserSettings
// mirrorimage=false (distinct from the compiled-in `true` default), so an unmodified
// image proves the processor actually read the live setting rather than always mirroring.
bool check_preproc_image_response(const ImagePreprocessorMessage& msg)
{
    const std::vector<char> kSeededBytes = uint8_raster(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
            45, 46, 47});

    bool ok = true;
    const auto check = [&](bool cond, const char* what)
    {
        if (!cond)
        {
            std::cerr << "FAIL: " << what << "\n";
            ok = false;
        }
    };

    check(msg.image.height == 4, "image height != 4");
    check(msg.image.width == 4, "image width != 4");
    check(msg.image.data == kSeededBytes, "image bytes changed with mirrorimage disabled");

    return ok;
}

// Assert a true horizontal mirror at unchanged (full) resolution: for the seeded 4x4
// three-channel uint8 CHW raster 0..47, each row of each channel plane reverses in place,
// and the response's Data length equals Height * Width * channels for the unchanged
// dimensions -- the v5 invariant that a truncated buffer would violate.
bool check_preproc_image_mirror_response(const ImagePreprocessorMessage& msg)
{
    const std::vector<char> kExpectedBytes = uint8_raster(
        {3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20,
            27, 26, 25, 24, 31, 30, 29, 28, 35, 34, 33, 32, 39, 38, 37, 36, 43, 42, 41, 40, 47,
            46, 45, 44});

    bool ok = true;
    const auto check = [&](bool cond, const char* what)
    {
        if (!cond)
        {
            std::cerr << "FAIL: " << what << "\n";
            ok = false;
        }
    };

    check(msg.image.height == 4, "image height != 4");
    check(msg.image.width == 4, "image width != 4");
    check(msg.image.data.size() == 48, "Data length != Height * Width * channels (48)");
    check(msg.image.data == kExpectedBytes, "mirrored bytes != row-reversed raster");

    return ok;
}

// A column index wrapped into a uint8 byte: the realloc raster's width (3840) does not fit
// in a single byte, so each pixel carries its column index modulo 256 rather than the index
// itself. Shared between the seed and the check so the two never drift apart.
constexpr unsigned char column_value(size_t x)
{
    return static_cast<unsigned char>(x % 256);
}

// A full 4K CHW raster in which every pixel of a row carries its own (wrapped) column
// value, so the mirrored response is checkable without holding a second copy of the
// expected raster.
std::vector<char> seed_realloc_image_raster()
{
    std::vector<char> raster(kReallocImageBytes);
    for (size_t plane = 0; plane < kReallocImageChannels * kReallocImageHeight; ++plane)
    {
        for (size_t x = 0; x < kReallocImageWidth; ++x)
            raster[plane * kReallocImageWidth + x] = static_cast<char>(column_value(x));
    }
    return raster;
}

// The realloc variant seeds a full 4K CHW frame, the shape a live IMAGE preprocessor
// actually receives: every pixel of a row carries its own (wrapped) column value, so a true
// horizontal mirror turns column x into column kReallocImageWidth - 1 - x. Comparing all
// ~24.9 MB would dwarf the rest of the suite, so the assertion samples the two ends of the
// first and last row of every channel -- enough to catch a truncated buffer, a row that
// came back unmirrored, or channels that got swapped.
bool check_preproc_image_realloc_response(const ImagePreprocessorMessage& msg)
{
    bool ok = true;
    const auto check = [&](bool cond, const char* what)
    {
        if (!cond)
        {
            std::cerr << "FAIL: " << what << "\n";
            ok = false;
        }
    };

    check(msg.image.height == kReallocImageHeight, "image height changed");
    check(msg.image.width == kReallocImageWidth, "image width changed");
    check(msg.image.data.size() == kReallocImageBytes, "Data length != Height * Width * channels");
    if (!ok)
        return false;

    const auto element_at = [&](size_t index)
    { return static_cast<unsigned char>(msg.image.data[index]); };

    for (size_t channel = 0; channel < kReallocImageChannels; ++channel)
    {
        for (const size_t row: {size_t{0}, kReallocImageHeight - 1})
        {
            const size_t row_start
                = (channel * kReallocImageHeight + row) * kReallocImageWidth;
            check(element_at(row_start) == column_value(kReallocImageWidth - 1),
                "first pixel of a row is not the mirrored last column");
            check(element_at(row_start + kReallocImageWidth - 1) == column_value(0),
                "last pixel of a row is not the mirrored first column");
        }
    }

    return ok;
}

} // namespace

int main(int argc, char* argv[])
{
    const auto usage = [&]
    {
        std::cerr << "Usage: " << argv[0]
                  << " --check "
                     "<objects|realloc|tensors|settings|counter|height|objects-sender|dashboard|"
                     "skip-feature-extraction|image|"
                     "preproc-tensor|preproc-image|preproc-image-mirror|preproc-image-realloc|"
                     "preproc-image-realloc-multi|no-response|"
                     "confidences|events|bestshots|bestshots-no-emit|edgeimpulse|"
                     "negative> "
                     "<args...>\n"
                     "  objects:              <args...> = <expected_class> <expected_source> <child_program> [prefix_args...]\n"
                     "  realloc:              <args...> = <expected_class> <expected_source> <child_program> [prefix_args...]\n"
                     "  tensors:              <args...> = <child_program> [prefix_args...]\n"
                     "  settings:             <args...> = <child_program> [prefix_args...]\n"
                     "  counter:              <args...> = <child_program> [prefix_args...]\n"
                     "  height:               <args...> = <child_program> [prefix_args...]\n"
                     "  objects-sender:       <args...> = <child_program> [prefix_args...]\n"
                     "  dashboard:            <args...> = <child_program> [prefix_args...]\n"
                     "  skip-feature-extraction: <args...> = <child_program> [prefix_args...]\n"
                     "  image:                <args...> = <child_program> [prefix_args...]\n"
                     "  preproc-tensor:       <args...> = <child_program> [prefix_args...]\n"
                     "  preproc-image:        <args...> = <child_program> [prefix_args...]\n"
                     "  preproc-image-mirror: <args...> = <child_program> [prefix_args...]\n"
                     "  preproc-image-realloc:<args...> = <child_program> [prefix_args...]\n"
                     "  preproc-image-realloc-multi: <args...> = <child_program> "
                     "[prefix_args...]\n"
                     "  no-response:          <args...> = <child_program> [prefix_args...]\n"
                     "  confidences:          <args...> = <child_program> [prefix_args...]\n"
                     "  events:               <args...> = <child_program> [prefix_args...]\n"
                     "  bestshots:            <args...> = <child_program> [prefix_args...]\n"
                     "  bestshots-no-emit:    <args...> = <child_program> [prefix_args...]\n"
                     "  edgeimpulse:          <args...> = <child_program> [prefix_args...]\n";
        return 2;
    };

    enum class Check
    {
        objects,
        realloc,
        tensors,
        settings,
        counter,
        height,
        objects_sender,
        dashboard,
        skip_feature_extraction,
        image,
        preproc_tensor,
        preproc_image,
        preproc_image_mirror,
        preproc_image_realloc,
        preproc_image_realloc_multi,
        no_response,
        confidences,
        events,
        bestshots,
        bestshots_no_emit,
        edgeimpulse,
        negative,
    };

    // --check is the sole scenario selector and is mandatory: a missing or unknown
    // scenario is a usage error rather than a silent default.
    int argi = 1;
    if (argi + 1 >= argc || std::string(argv[argi]) != "--check")
        return usage();
    Check check;
    const std::string scenario = argv[argi + 1];
    if (scenario == "objects")
        check = Check::objects;
    else if (scenario == "realloc")
        check = Check::realloc;
    else if (scenario == "tensors")
        check = Check::tensors;
    else if (scenario == "settings")
        check = Check::settings;
    else if (scenario == "counter")
        check = Check::counter;
    else if (scenario == "height")
        check = Check::height;
    else if (scenario == "objects-sender")
        check = Check::objects_sender;
    else if (scenario == "dashboard")
        check = Check::dashboard;
    else if (scenario == "skip-feature-extraction")
        check = Check::skip_feature_extraction;
    else if (scenario == "image")
        check = Check::image;
    else if (scenario == "preproc-tensor")
        check = Check::preproc_tensor;
    else if (scenario == "preproc-image")
        check = Check::preproc_image;
    else if (scenario == "preproc-image-mirror")
        check = Check::preproc_image_mirror;
    else if (scenario == "preproc-image-realloc")
        check = Check::preproc_image_realloc;
    else if (scenario == "preproc-image-realloc-multi")
        check = Check::preproc_image_realloc_multi;
    else if (scenario == "no-response")
        check = Check::no_response;
    else if (scenario == "confidences")
        check = Check::confidences;
    else if (scenario == "events")
        check = Check::events;
    else if (scenario == "bestshots")
        check = Check::bestshots;
    else if (scenario == "bestshots-no-emit")
        check = Check::bestshots_no_emit;
    else if (scenario == "edgeimpulse")
        check = Check::edgeimpulse;
    else if (scenario == "negative")
        check = Check::negative;
    else
        return usage();
    argi += 2;

    std::string expected_class;
    std::string expected_source;
    if (check == Check::objects || check == Check::realloc)
    {
        if (argi + 1 >= argc)
            return usage();
        expected_class = argv[argi++];
        expected_source = argv[argi++];
    }

    std::string negative_kind;
    std::string negative_expected;
    if (check == Check::negative)
    {
        if (argi + 1 >= argc)
            return usage();
        negative_kind = argv[argi++];
        negative_expected = argv[argi++];
        if (negative_kind != "missing-keys" && negative_kind != "truncated"
            && negative_kind != "no-image")
            return usage();
        if (negative_expected != "recovers" && negative_expected != "dies")
            return usage();
    }
    if (argi >= argc)
        return usage();

    // Create the two IPC channels exactly as the AI Manager (HOST) would.
    const size_t initial_shm_size
        = (check == Check::realloc || check == Check::preproc_image_realloc
              || check == Check::preproc_image_realloc_multi)
        ? kReallocInitialShmSize
        : kInitialShmSize;
    nxai_ipc_t input_channel = nxai_ipc_create(initial_shm_size);
    nxai_ipc_t output_channel = nxai_ipc_create(initial_shm_size);
    if (input_channel == IPC_SYNC_INVALID || output_channel == IPC_SYNC_INVALID)
    {
        std::cerr << "FAIL: could not create IPC channels\n";
        return 2;
    }

    char* input_str = ipc_sync_to_str(input_channel);
    char* output_str = ipc_sync_to_str(output_channel);
    if (input_str == nullptr || output_str == nullptr)
    {
        std::cerr << "FAIL: could not serialise sync strings\n";
        return 2;
    }

    // Child argv: <child_program> [pre_args...] <input_str> <output_str>
    // --ProcessorName e2e-test [post_args...]. pre_args are whatever must precede the
    // sync pair positionally (e.g. a Python script path: `python script.py <sync1>
    // <sync2> ...`); post_args are scenario-specific Config `--key value` flags
    // (load_config() scans the whole argv regardless of position, but the Python SDK's
    // initialize_from_args() requires the two sync strings at args[0]/args[1] exactly
    // -- nxai_postprocessor_sdk.py -- so Config flags cannot sit ahead of them). A
    // literal "--" token in <args...> after <child_program> splits the two; absent,
    // every remaining token is a pre_arg (this is the original, still-supported shape
    // every existing scenario uses).
    std::vector<char*> pre_args;
    std::vector<char*> post_args;
    {
        bool seen_separator = false;
        for (int i = argi + 1; i < argc; ++i)
        {
            if (!seen_separator && std::string(argv[i]) == "--")
            {
                seen_separator = true;
                continue;
            }
            (seen_separator ? post_args : pre_args).push_back(argv[i]);
        }
    }

    std::vector<char*> child_argv;
    child_argv.push_back(argv[argi]);
    child_argv.insert(child_argv.end(), pre_args.begin(), pre_args.end());
    child_argv.push_back(input_str);
    child_argv.push_back(output_str);
    child_argv.push_back(const_cast<char*>("--ProcessorName"));
    child_argv.push_back(const_cast<char*>("e2e-test"));
    child_argv.insert(child_argv.end(), post_args.begin(), post_args.end());
    child_argv.push_back(nullptr);

    // The child must inherit: the sync objects encoded in its argv reach it no other way,
    // and the shared memory is opened by name.
    std::string spawn_error;
    auto child = e2e::ChildProcess::spawn(
        child_argv, e2e::HandleInheritance::inherit, spawn_error);
    if (!child.has_value())
    {
        std::cerr << "FAIL: could not start " << child_argv[0] << ": " << spawn_error << "\n";
        return 2;
    }

    // Parent: drive the processor like the AI Manager does, for one frame.
    bool passed = false;
    do
    {
        // The AI Manager waits for READY on the input channel only (sendMessageIPC in
        // nxai_external_process.cpp); it never consumes a READY on the output channel.
        // Waiting for it here made the harness more forgiving than the engine.
        const int ready_timeout_ms =
            check == Check::edgeimpulse ? kHeavyImportReadyTimeoutMs : kReadyTimeoutMs;
        if (nxai_ipc_wait_ready(input_channel, ready_timeout_ms) != IPC_ERROR_NONE)
        {
            std::cerr << "FAIL: processor did not signal ready\n";
            break;
        }

        if (check == Check::negative)
        {
            // Two frames: one the processor cannot process, then a well-formed one. The
            // response to the FIRST frame is only reported, never asserted -- what this
            // scenario pins is whether the processor is still able to serve the SECOND.
            // See the scenario doc comment above for why the observed
            // outcome is an argument rather than a fixed expectation.
            const bool image_processor = negative_kind == "no-image";
            const auto make_good_frame = [&]
            {
                ObjectsPostProcessorMessage request;
                request.device_id = "e2e-device";
                request.device_name = "e2e";
                request.objects["person"].push_back(ObjectData{.confidence = 0.9F});
                if (image_processor)
                {
                    // The ReceiveInputImage processor needs the block to answer at all,
                    // so the recovery frame carries the same well-formed raster the image
                    // scenario seeds.
                    constexpr char kImageBytes[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
                    request.image_data = ImageData{.height = 2,
                        .width = 2,
                        .layout = ImageOrder::HWC,
                        .data = std::vector<char>(kImageBytes, kImageBytes + sizeof(kImageBytes))};
                }
                return writeObjectsPostprocessorMessage(request);
            };

            std::vector<unsigned char> malformed;
            if (negative_kind == "missing-keys")
            {
                // Structurally valid MessagePack -- a one-entry map {"DeviceID": "e2e"} --
                // in which every key the processor reads is simply absent.
                const std::string raw = "\x81\xa8"
                                        "DeviceID"
                                        "\xa3"
                                        "e2e";
                malformed.assign(raw.begin(), raw.end());
            }
            else
            {
                // truncated: the first half of a well-formed frame, i.e. bytes that stop
                // mid-structure. no-image: a well-formed frame that just omits the
                // ImageData block the processor declared ReceiveInputImage for.
                ObjectsPostProcessorMessage request;
                request.device_id = "e2e-device";
                request.device_name = "e2e";
                request.objects["person"].push_back(ObjectData{.confidence = 0.9F});
                const auto seed = writeObjectsPostprocessorMessage(request);
                if (!seed.has_value())
                {
                    std::cerr << "FAIL: could not serialise the malformed frame\n";
                    break;
                }
                const auto* first = reinterpret_cast<const unsigned char*>(seed->data.get());
                malformed.assign(
                    first, first + (negative_kind == "truncated" ? seed->size / 2 : seed->size));
            }

            unsigned char* resp = nullptr;
            uint32_t resp_size = 0;
            if (nxai_ipc_send(input_channel, malformed.data(),
                    static_cast<uint32_t>(malformed.size()))
                != IPC_ERROR_NONE)
            {
                std::cerr << "FAIL: could not send the malformed frame\n";
                break;
            }
            if (nxai_ipc_receive(output_channel, &resp, &resp_size, kNegativeWaitMs)
                == IPC_ERROR_NONE)
            {
                std::cerr << "note: the processor answered the malformed frame with "
                          << resp_size << " bytes\n";
                release_buffer(output_channel);
            }
            else
            {
                std::cerr << "note: no response to the malformed frame\n";
            }

            bool answered_follow_up = false;
            const auto good = make_good_frame();
            if (!good.has_value())
            {
                std::cerr << "FAIL: could not serialise the follow-up frame\n";
                break;
            }
            if (nxai_ipc_send(input_channel,
                    reinterpret_cast<const unsigned char*>(good->data.get()),
                    static_cast<uint32_t>(good->size))
                == IPC_ERROR_NONE)
            {
                resp = nullptr;
                resp_size = 0;
                if (nxai_ipc_receive(output_channel, &resp, &resp_size, kNegativeWaitMs)
                    == IPC_ERROR_NONE)
                {
                    answered_follow_up
                        = parseObjectsPostProcessorMessage(resp, resp_size).has_value();
                    release_buffer(output_channel);
                }
            }

            // Same liveness rule as no-response: ask the launcher whether the child has
            // actually ended -- a signal-zero probe would still find an exited-but-unreaped
            // one alive.
            const bool alive = !child->wait(std::chrono::milliseconds(0)).has_value();
            const std::string observed = (answered_follow_up && alive) ? "recovers" : "dies";
            std::cerr << "observed: " << observed
                      << " (follow-up answered: " << (answered_follow_up ? "yes" : "no")
                      << ", still running: " << (alive ? "yes" : "no") << ")\n";
            if (observed != negative_expected)
            {
                std::cerr << "FAIL: expected \"" << negative_expected << "\", observed \""
                          << observed << "\"\n";
                break;
            }
            passed = true;
            break;
        }

        if (check == Check::no_response || check == Check::edgeimpulse)
        {
            // See the scenario doc comment above for why more than one frame is driven.
            passed = true;
            for (int frame = 0; frame < kNoResponseFrameCount; ++frame)
            {
                ObjectsPostProcessorMessage request;
                request.device_id = "e2e-device";
                request.device_name = "e2e";
                if (check == Check::edgeimpulse)
                {
                    // Confidence below the fixture ini's p_value (0.4) so every frame
                    // queues a training sample, exercising the ImageData -> JPEG path
                    // a crash there would only surface as the child dying (checked
                    // below), since this processor never responds either way.
                    request.objects["person"].push_back(ObjectData{.confidence = 0.1F});
                    constexpr char kImageBytes[] = {
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; // 2x2 RGB uint8
                    request.image_data = ImageData{
                        .height = 2,
                        .width = 2,
                        .layout = ImageOrder::HWC,
                        .data = std::vector<char>(kImageBytes, kImageBytes + sizeof(kImageBytes))};
                }
                else
                {
                    request.objects["person"].push_back(ObjectData{.confidence = 0.9F});
                }
                const auto bytes = writeObjectsPostprocessorMessage(request);
                if (!bytes.has_value())
                {
                    std::cerr << "FAIL: could not serialise request\n";
                    passed = false;
                    break;
                }
                if (nxai_ipc_send(input_channel,
                        reinterpret_cast<const unsigned char*>(bytes->data.get()),
                        static_cast<uint32_t>(bytes->size))
                    != IPC_ERROR_NONE)
                {
                    std::cerr << "FAIL: could not send request " << frame << "\n";
                    passed = false;
                    break;
                }

                unsigned char* resp = nullptr;
                uint32_t resp_size = 0;
                const ipc_error_t recv_result
                    = nxai_ipc_receive(output_channel, &resp, &resp_size, kNoResponseWaitMs);
                if (recv_result == IPC_ERROR_NONE)
                {
                    std::cerr << "FAIL: response arrived on frame " << frame
                              << " for a NoResponse processor\n";
                    release_buffer(output_channel);
                    passed = false;
                    break;
                }
                if (recv_result != IPC_ERROR_TIMEOUT)
                {
                    std::cerr << "FAIL: unexpected IPC error waiting on frame " << frame << "\n";
                    passed = false;
                    break;
                }
            }

            // A processor that died without ever writing to the output channel passes every
            // assertion above, so check liveness explicitly: a zero-timeout wait reports an
            // exited child, which a signal-zero probe would still see as alive.
            if (passed)
            {
                if (const auto ended = child->wait(std::chrono::milliseconds(0)))
                {
                    std::cerr << "FAIL: processor " << ended->describe() << " after "
                              << kNoResponseFrameCount
                              << " frames instead of continuing to run\n";
                    passed = false;
                }
            }
            break;
        }

        if (check == Check::preproc_image_realloc_multi)
        {
            // See the scenario doc comment above: the sends go out back to back, and the
            // responses are only drained afterwards, because that is what the AI Manager
            // does -- nxai_ipc_send() waits for the input acknowledgement alone.
            ImagePreprocessorMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            request.image = ImageData{.height = uint32_t(kReallocImageHeight),
                .width = uint32_t(kReallocImageWidth),
                .layout = ImageOrder::CHW,
                .data = seed_realloc_image_raster()};
            const auto frame_bytes = writeImagePreprocessorMessage(request);
            if (!frame_bytes.has_value())
            {
                std::cerr << "FAIL: could not serialise request\n";
                break;
            }

            passed = true;
            for (int frame = 0; frame < kReallocImageFrameCount; ++frame)
            {
                const ipc_error_t send_result = nxai_ipc_send(input_channel,
                    reinterpret_cast<const unsigned char*>(frame_bytes->data.get()),
                    static_cast<uint32_t>(frame_bytes->size));
                if (send_result != IPC_ERROR_NONE)
                {
                    std::cerr << "FAIL: could not send frame " << frame << ", IPC error "
                              << send_result << "\n";
                    passed = false;
                }
            }

            for (int frame = 0; frame < kReallocImageFrameCount; ++frame)
            {
                unsigned char* resp = nullptr;
                uint32_t resp_size = 0;
                if (nxai_ipc_receive(output_channel, &resp, &resp_size, kReceiveTimeoutMs)
                    != IPC_ERROR_NONE)
                {
                    std::cerr << "FAIL: no response to frame " << frame << "\n";
                    passed = false;
                    continue;
                }
                const auto parsed = parseImagePreprocessorMessage(
                    std::span<char>(reinterpret_cast<char*>(resp), resp_size));
                release_buffer(output_channel);
                if (!parsed.has_value())
                {
                    std::cerr << "FAIL: could not parse the response to frame " << frame << "\n";
                    passed = false;
                    continue;
                }
                if (!check_preproc_image_realloc_response(*parsed))
                {
                    std::cerr << "FAIL: wrong response to frame " << frame << "\n";
                    passed = false;
                }
            }

            // A processor that died mid-handshake and one that is still waiting for an
            // acknowledgement nobody will send are the same missing response otherwise.
            if (const auto ended = child->wait(std::chrono::milliseconds(0)))
                std::cerr << "note: processor " << ended->describe() << "\n";
            else
                std::cerr << "note: processor is still running\n";
            break;
        }

        std::optional<nxai_buffer_t> bytes;
        if (check == Check::objects)
        {
            ObjectsPostProcessorMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            bytes = writeObjectsPostprocessorMessage(request);
        }
        else if (check == Check::realloc)
        {
            // Same message the objects scenario sends, padded with enough objects to
            // outgrow the 1024-byte channels created above -- in both directions, since the
            // reference examples echo what they receive.
            ObjectsPostProcessorMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            for (int i = 0; i < kReallocObjectCount; ++i)
                request.objects["person"].push_back(ObjectData{.confidence = 0.9F});
            bytes = writeObjectsPostprocessorMessage(request);
            if (bytes.has_value() && bytes->size <= kReallocInitialShmSize)
            {
                std::cerr << "FAIL: seeded request is " << bytes->size
                          << " bytes, which fits inside the " << kReallocInitialShmSize
                          << "-byte channel -- the scenario would not reallocate anything\n";
                break;
            }
        }
        else if (check == Check::settings)
        {
            // Seed the bytes the patched engine will send: a forwarded UserSettings map
            // plus a "person" object the plugin should tag with attributes["reviewed"].
            ObjectsPostProcessorMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            request.user_settings = {{"attributeName", "reviewed"}, {"attributeValue", "true"}};
            request.objects["person"].push_back(ObjectData{.confidence = 0.9F});
            bytes = writeObjectsPostprocessorMessage(request);
        }
        else if (check == Check::counter)
        {
            // Seed a live min_object_count of 3 and exactly 3 "person" objects: below
            // the processor's hardcoded default (5), so the event only fires if the
            // live UserSettings value is actually read.
            ObjectsPostProcessorMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            request.user_settings = {{"min_object_count", "3"}};
            for (int i = 0; i < 3; ++i)
                request.objects["person"].push_back(ObjectData{.confidence = 0.9F});
            bytes = writeObjectsPostprocessorMessage(request);
        }
        else if (check == Check::height)
        {
            // Seed live frame_fraction_per_meter=0.5 and max_height_cm=100 (both
            // distinct from the processor's hardcoded defaults 0.25 / 250) and one
            // "person" of normalized height 0.7 -> 1.4 m at the live fraction, above
            // the live 1.0 m limit, so the processor tags it "1.40 m" and fires
            // person.height.exceeded only if both live settings are actually read.
            ObjectsPostProcessorMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            request.user_settings = {
                {"frame_fraction_per_meter", "0.5"}, {"max_height_cm", "100"}};
            request.objects["person"].push_back(ObjectData{.height = 0.7F, .confidence = 0.9F});
            bytes = writeObjectsPostprocessorMessage(request);
        }
        else if (check == Check::objects_sender)
        {
            // Seed 4 reference points at the frame corners on the map lat = 10 + 2 * x,
            // lon = 20 + 4 * y (latitude/longitude in degrees times 1000, as the spin boxes
            // send them), and one object of each of two classes. Every setting differs from
            // the processor's default; server_address points at a port nothing listens on.
            ObjectsPostProcessorMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            request.user_settings = {
                {"point0.figure",
                 R"({"figure":{"color":"#536dfe","points":[[0.08,0.07],[0.12,0.13]]},)"
                 R"("label":"","showOnCamera":false})"},
                {"point0.latitude", "10200"},
                {"point0.longitude", "20400"},
                {"point1.figure",
                 R"({"figure":{"color":"#536dfe","points":[[0.88,0.07],[0.92,0.13]]},)"
                 R"("label":"","showOnCamera":false})"},
                {"point1.latitude", "11800"},
                {"point1.longitude", "20400"},
                {"point2.figure",
                 R"({"figure":{"color":"#536dfe","points":[[0.88,0.87],[0.92,0.93]]},)"
                 R"("label":"","showOnCamera":false})"},
                {"point2.latitude", "11800"},
                {"point2.longitude", "23600"},
                {"point3.figure",
                 R"({"figure":{"color":"#536dfe","points":[[0.08,0.87],[0.12,0.93]]},)"
                 R"("label":"","showOnCamera":false})"},
                {"point3.latitude", "10200"},
                {"point3.longitude", "23600"},
                {"server_address", "http://127.0.0.1:1"},
                {"topic_path", "e2e/objects"},
            };
            request.objects["car"].push_back(
                ObjectData{
                    .x = 0.45F,
                    .y = 0.15F,
                    .width = 0.1F,
                    .height = 0.2F,
                    .confidence = 0.9F});
            request.objects["person"].push_back(
                ObjectData{
                    .x = 0.2F,
                    .y = 0.65F,
                    .width = 0.1F,
                    .height = 0.2F,
                    .confidence = 0.9F});
            bytes = writeObjectsPostprocessorMessage(request);
        }
        else if (check == Check::dashboard)
        {
            // Seed one "test" object of known fields; the dashboard must return it
            // field-for-field unmodified (its web-app POST side-effect fails harmlessly
            // with no server listening and is ignored).
            ObjectsPostProcessorMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            request.objects["test"].push_back(ObjectData{
                .x = 0.1F, .y = 0.2F, .width = 0.3F, .height = 0.4F, .confidence = 0.95F});
            bytes = writeObjectsPostprocessorMessage(request);
        }
        else if (check == Check::skip_feature_extraction)
        {
            // The fixture ignores incoming objects and always emits its own fixed
            // "chain-test" pair; an empty seed is sufficient.
            ObjectsPostProcessorMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            bytes = writeObjectsPostprocessorMessage(request);
        }
        else if (check == Check::image)
        {
            // Seed a root ImageData block with a well-formed 2x2 RGB uint8 raster (12
            // bytes, HWC): exactly what the processor's _decode_frame expects, so the
            // response is expected to report a clean decode.
            ObjectsPostProcessorMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            constexpr char kImageBytes[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
            request.image_data = ImageData{
                .height = 2,
                .width = 2,
                .layout = ImageOrder::HWC,
                .data = std::vector<char>(kImageBytes, kImageBytes + sizeof(kImageBytes))};
            bytes = writeObjectsPostprocessorMessage(request);
        }
        else if (check == Check::confidences)
        {
            // Seed one "person" object of confidence 0.876; round(0.876, 2) == 0.88,
            // distinct in both decimal places so a truncation-vs-rounding bug would
            // not accidentally pass.
            ObjectsPostProcessorMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            request.objects["person"].push_back(ObjectData{.confidence = 0.876F});
            bytes = writeObjectsPostprocessorMessage(request);
        }
        else if (check == Check::events)
        {
            // The processor emits its event unconditionally; an empty seed is
            // sufficient (see the scenario doc comment above for what is asserted).
            ObjectsPostProcessorMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            bytes = writeObjectsPostprocessorMessage(request);
        }
        else if (check == Check::bestshots || check == Check::bestshots_no_emit)
        {
            // Seed one already-tracked "car" object (track_id set) --
            // eligibility the fixture requires before it will ever tag a best shot.
            // Whether it actually does depends on the child's --emit_best_shots flag
            // (set by the CTest entry, not here), not on anything in the request.
            ObjectsPostProcessorMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            request.objects["car"].push_back(
                ObjectData{.track_id = "e2e-track-1", .confidence = 0.9F});
            bytes = writeObjectsPostprocessorMessage(request);
        }
        else if (check == Check::preproc_tensor)
        {
            // Seed a known "nms_sensitivity-" float (0.5) the preprocessor must override to
            // the forwarded UserSettings value (0.3) plus an unrelated "input"
            // tensor of a distinct value (0.25) that must pass through unchanged -- distinct
            // so copying the wrong tensor's data over it cannot pass.
            TensorsMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            request.user_settings = {{"nmsoverride", "0.3"}};
            float nms_value = 0.5F;
            request.tensors.push_back(TensorData{
                .name = "nms_sensitivity-",
                .data_type = DATA_TYPE_FLOAT,
                .shape = {1},
                .data = std::vector<char>(reinterpret_cast<char*>(&nms_value),
                    reinterpret_cast<char*>(&nms_value) + sizeof(nms_value))});
            float input_value = 0.25F;
            request.tensors.push_back(TensorData{
                .name = "input",
                .data_type = DATA_TYPE_FLOAT,
                .shape = {1},
                .data = std::vector<char>(reinterpret_cast<char*>(&input_value),
                    reinterpret_cast<char*>(&input_value) + sizeof(input_value))});
            bytes = writeTensorMessage(request);
        }
        else if (check == Check::preproc_image || check == Check::preproc_image_mirror)
        {
            // Seed a known 4x4 three-channel uint8 CHW image (elements 0..47, 48 bytes --
            // the v5 IMAGE `Data` is uint8, matching what a live engine delivers). preproc_image
            // also seeds a forwarded UserSettings map of {"mirrorimage": "false"},
            // so the processor must return it unchanged; preproc_image_mirror seeds no
            // UserSettings at all, relies on the compiled-in `true` default, and expects a
            // true horizontal mirror at unchanged (full) resolution.
            ImagePreprocessorMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            request.image = ImageData{.height = 4,
                .width = 4,
                .layout = ImageOrder::CHW,
                .data = uint8_raster({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47})};
            if (check == Check::preproc_image)
                request.user_settings = {{"mirrorimage", "false"}};
            bytes = writeImagePreprocessorMessage(request);
        }
        else if (check == Check::preproc_image_realloc)
        {
            // A full 4K CHW frame over the 1024-byte channels.
            std::vector<char> raster = seed_realloc_image_raster();
            ImagePreprocessorMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            request.image = ImageData{.height = uint32_t(kReallocImageHeight),
                .width = uint32_t(kReallocImageWidth),
                .layout = ImageOrder::CHW,
                .data = std::move(raster)};
            bytes = writeImagePreprocessorMessage(request);
        }
        else
        {
            TensorsMessage request;
            request.device_id = "e2e-device";
            request.device_name = "e2e";
            float input[] = {0.5F, 0.5F};
            request.tensors.push_back(TensorData{
                .name = "input",
                .data_type = DATA_TYPE_FLOAT,
                .shape = {1, 2},
                .data = std::vector<char>(reinterpret_cast<char*>(input),
                    reinterpret_cast<char*>(input) + sizeof(input))});
            bytes = writeTensorMessage(request);
        }
        if (!bytes.has_value())
        {
            std::cerr << "FAIL: could not serialise request\n";
            break;
        }
        if (nxai_ipc_send(input_channel,
                reinterpret_cast<const unsigned char*>(bytes->data.get()),
                static_cast<uint32_t>(bytes->size))
            != IPC_ERROR_NONE)
        {
            std::cerr << "FAIL: could not send request\n";
            break;
        }

        unsigned char* resp = nullptr;
        uint32_t resp_size = 0;
        if (nxai_ipc_receive(output_channel, &resp, &resp_size, kReceiveTimeoutMs)
            != IPC_ERROR_NONE)
        {
            // Say whether the processor is still there. A silent death and a live processor
            // that simply never answered are the same "did not receive response" otherwise,
            // and they point at completely different halves of the transport.
            if (const auto ended = child->wait(std::chrono::milliseconds(0)))
                std::cerr << "FAIL: did not receive response; processor " << ended->describe()
                          << "\n";
            else
                std::cerr << "FAIL: did not receive response; processor is still running\n";
            break;
        }

        if (check == Check::objects || check == Check::realloc || check == Check::settings
            || check == Check::counter
            || check == Check::height || check == Check::objects_sender
            || check == Check::dashboard || check == Check::skip_feature_extraction
            || check == Check::image || check == Check::confidences || check == Check::events
            || check == Check::bestshots || check == Check::bestshots_no_emit)
        {
            const auto parsed = parseObjectsPostProcessorMessage(resp, resp_size);
            release_buffer(output_channel);
            if (!parsed.has_value())
            {
                std::cerr << "FAIL: could not parse response\n";
                break;
            }
            if (check == Check::objects)
                passed = check_response(*parsed, expected_class, expected_source);
            else if (check == Check::realloc)
                passed = check_realloc_response(*parsed, expected_class, expected_source);
            else if (check == Check::settings)
                passed = check_settings_response(*parsed);
            else if (check == Check::counter)
                passed = check_counter_response(*parsed);
            else if (check == Check::height)
                passed = check_height_response(*parsed);
            else if (check == Check::objects_sender)
                passed = check_objects_sender_response(*parsed);
            else if (check == Check::dashboard)
                passed = check_dashboard_response(*parsed);
            else if (check == Check::skip_feature_extraction)
                passed = check_skip_feature_extraction_response(*parsed);
            else if (check == Check::image)
                passed = check_image_response(*parsed);
            else if (check == Check::confidences)
                passed = check_confidences_response(*parsed);
            else if (check == Check::events)
                passed = check_events_response(*parsed);
            else if (check == Check::bestshots)
                passed = check_bestshots_response(*parsed);
            else
                passed = check_bestshots_no_emit_response(*parsed);
        }
        else if (check == Check::preproc_image || check == Check::preproc_image_mirror
            || check == Check::preproc_image_realloc)
        {
            const auto parsed = parseImagePreprocessorMessage(
                std::span<char>(reinterpret_cast<char*>(resp), resp_size));
            release_buffer(output_channel);
            if (!parsed.has_value())
            {
                std::cerr << "FAIL: could not parse response\n";
                break;
            }
            if (check == Check::preproc_image)
                passed = check_preproc_image_response(*parsed);
            else if (check == Check::preproc_image_mirror)
                passed = check_preproc_image_mirror_response(*parsed);
            else
                passed = check_preproc_image_realloc_response(*parsed);
        }
        else
        {
            const auto parsed = parseTensorMessage(resp, resp_size);
            release_buffer(output_channel);
            if (!parsed.has_value())
            {
                std::cerr << "FAIL: could not parse response\n";
                break;
            }
            if (check == Check::preproc_tensor)
                passed = check_preproc_tensor_response(*parsed);
            else
                passed = check_tensors_response(*parsed);
        }
    } while (false);

    // Tell the processor to exit, then make sure it is gone. Termination is requested in
    // addition to the IPC EXIT signal because processors may not exit cleanly on EXIT alone
    // (e.g. the Python reference processor's main loop treats EXIT the same as a timeout and
    // continues).
    nxai_ipc_send_kill(input_channel);
    child->request_termination();

    ipc_sync_close(input_channel);
    ipc_sync_close(output_channel);

    if (passed)
    {
        std::cout << "PASS\n";
        return 0;
    }
    return 1;
}
