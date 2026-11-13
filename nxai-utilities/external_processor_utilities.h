#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "nxai_utils.h"

// ============================================================================
// Message type definitions
// ============================================================================

struct ObjectData
{
    std::optional<std::string> track_id; //< Existing object UUID (absent = new object)
    std::optional<std::string> new_track_id; //< Requested new UUID (empty string = VMS assigns)
    float x = 0.0F;
    float y = 0.0F;
    float width = 0.0F;
    float height = 0.0F;
    float confidence = 0.0F;
    std::map<std::string, std::string> attributes;
    // Optional per-object chaining gate. Response-only: a
    // postprocessor may set it, the engine never sends it. Parsed here (read-only,
    // no writer support) purely for e2e assertions; mirrors the real engine
    // deserializer's unconditional-every-frame semantics: present -> its bool
    // value, absent -> false.
    bool skip_feature_extraction = false;
};

struct EventData
{
    std::string type_id;
    std::string description;
};

// Postprocessor-nominated best shot for an already-tracked object.
// Response-only: a postprocessor may append to ObjectsPostProcessorMessage::best_shots,
// the engine never sends it, so only parsing is needed here (mirrors
// ObjectData.skip_feature_extraction).
struct BestShotData
{
    std::string track_id; //< required: UUID of an existing track
    std::optional<float> x;
    std::optional<float> y;
    std::optional<float> width;
    std::optional<float> height;
};

struct ImageData
{
    uint64_t height = 0;
    uint64_t width = 0;
    ImageOrder layout;
    std::vector<char> data;
};

struct TensorData
{
    std::string name;
    int data_type = 0;
    std::vector<uint32_t> shape;
    std::vector<char> data;
};

struct TensorsMessage
{
    int64_t timestamp = 0;
    std::string device_id;
    std::string device_name;
    std::vector<TensorData> tensors;
    std::map<std::string, std::string> user_settings; //< forwarded GUI settings (write-only)
};

struct ImagePreprocessorMessage
{
    int64_t timestamp = 0;
    std::string device_id;
    std::string device_name;
    ImageData image;
    std::map<std::string, std::string> user_settings; //< forwarded GUI settings (write-only)
};

struct ObjectsPostProcessorMessage
{
    int64_t timestamp = 0;
    std::string device_id;
    std::string device_name;
    std::map<std::string, std::vector<ObjectData>> objects; //< class name -> detected objects
    std::vector<EventData> events;
    std::vector<BestShotData> best_shots; //< response-only (parse-only, see BestShotData)
    std::optional<ImageData> image_data;
    std::vector<TensorData> tensors;
    std::map<std::string, std::string> user_settings; //< forwarded GUI settings (write-only)
};

std::optional<ObjectsPostProcessorMessage> parseObjectsPostProcessorMessage(
    const unsigned char* data,
    uint32_t size);

std::optional<nxai_buffer_t> writeObjectsPostprocessorMessage(
    const ObjectsPostProcessorMessage& msg);

std::optional<nxai_buffer_t> writeImagePreprocessorMessage(
    const ImagePreprocessorMessage& image_message);

std::optional<ImagePreprocessorMessage> parseImagePreprocessorMessage(
    std::span<char> message_data);

std::optional<TensorsMessage> parseTensorMessage(const unsigned char* data, size_t size);

std::optional<nxai_buffer_t> writeTensorMessage(const TensorsMessage& tensor_message);
