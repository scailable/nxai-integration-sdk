#include "external_processor_utilities.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "mpack.h"
#include "nxai_utils.h"
#include "oaax/interface.h"

static void writeTensorsToMPackWriter(
    mpack_writer_t* writer,
    const std::vector<TensorData>& tensors);
// ============================================================================
// Parsing (msgpack -> structs)
// ============================================================================

static ObjectData parse_object(mpack_node_t node)
{
    ObjectData obj;

    const mpack_node_t id_node = mpack_node_map_cstr_optional(node, "ID");
    if (mpack_node_type(id_node) == mpack_type_str)
        obj.track_id = std::string(mpack_node_str(id_node), mpack_node_strlen(id_node));

    const mpack_node_t new_id_node = mpack_node_map_cstr_optional(node, "NewID");
    if (mpack_node_type(new_id_node) == mpack_type_str)
        obj.new_track_id =
            std::string(mpack_node_str(new_id_node), mpack_node_strlen(new_id_node));

    obj.x = mpack_node_float(mpack_node_map_cstr(node, "x"));
    obj.y = mpack_node_float(mpack_node_map_cstr(node, "y"));
    obj.width = mpack_node_float(mpack_node_map_cstr(node, "width"));
    obj.height = mpack_node_float(mpack_node_map_cstr(node, "height"));
    obj.confidence = mpack_node_float(mpack_node_map_cstr(node, "confidence"));

    const mpack_node_t attrs_node = mpack_node_map_cstr_optional(node, "attributes");
    if (mpack_node_type(attrs_node) == mpack_type_map)
    {
        const size_t count = mpack_node_map_count(attrs_node);
        for (size_t i = 0; i < count; ++i)
        {
            const mpack_node_t key = mpack_node_map_key_at(attrs_node, i);
            const mpack_node_t val = mpack_node_map_value_at(attrs_node, i);
            obj.attributes.emplace(
                std::string(mpack_node_str(key), mpack_node_strlen(key)),
                std::string(mpack_node_str(val), mpack_node_strlen(val)));
        }
    }

    // Optional response-only chaining gate. Absent ->
    // obj.skip_feature_extraction keeps its false default, matching the real engine
    // deserializer.
    const mpack_node_t skip_node = mpack_node_map_cstr_optional(node, "skipFeatureExtraction");
    if (mpack_node_type(skip_node) == mpack_type_bool)
        obj.skip_feature_extraction = mpack_node_bool(skip_node);

    return obj;
}

static EventData parse_event(mpack_node_t node)
{
    EventData evt;
    const mpack_node_t id_node = mpack_node_map_cstr(node, "ID");
    const mpack_node_t desc_node = mpack_node_map_cstr(node, "Description");
    evt.type_id = std::string(mpack_node_str(id_node), mpack_node_strlen(id_node));
    evt.description = std::string(mpack_node_str(desc_node), mpack_node_strlen(desc_node));
    return evt;
}

std::optional<ObjectsPostProcessorMessage> parseObjectsPostProcessorMessage(
    const unsigned char* data,
    uint32_t size)
{
    mpack_tree_t tree;
    mpack_tree_init_data(&tree, reinterpret_cast<const char*>(data), size);
    mpack_tree_parse(&tree);
    const mpack_node_t root = mpack_tree_root(&tree);

    if (mpack_tree_error(&tree) != mpack_ok)
    {
        std::cerr << "Error: failed to parse incoming msgpack message\n";
        mpack_tree_destroy(&tree);
        return std::nullopt;
    }

    ObjectsPostProcessorMessage msg;
    msg.timestamp = mpack_node_i64(mpack_node_map_cstr(root, "Timestamp"));

    const mpack_node_t device_id_node = mpack_node_map_cstr(root, "DeviceID");
    msg.device_id = std::string(mpack_node_str(device_id_node), mpack_node_strlen(device_id_node));

    const mpack_node_t device_name_node = mpack_node_map_cstr(root, "DeviceName");
    msg.device_name =
        std::string(mpack_node_str(device_name_node), mpack_node_strlen(device_name_node));

    const mpack_node_t objects_node = mpack_node_map_cstr(root, "Objects");
    const size_t class_count = mpack_node_map_count(objects_node);
    for (size_t i = 0; i < class_count; ++i)
    {
        const mpack_node_t key = mpack_node_map_key_at(objects_node, i);
        const mpack_node_t val = mpack_node_map_value_at(objects_node, i);

        std::string class_name(mpack_node_str(key), mpack_node_strlen(key));
        std::vector<ObjectData>& class_objects = msg.objects[class_name];

        const size_t obj_count = mpack_node_array_length(val);
        class_objects.reserve(obj_count);
        for (size_t j = 0; j < obj_count; ++j)
            class_objects.push_back(parse_object(mpack_node_array_at(val, j)));
    }

    const mpack_node_t events_node = mpack_node_map_cstr(root, "Events");
    const size_t event_count = mpack_node_array_length(events_node);
    msg.events.reserve(event_count);
    for (size_t i = 0; i < event_count; ++i)
        msg.events.push_back(parse_event(mpack_node_array_at(events_node, i)));

    // Parse optional BestShots: absent/missing key is a no-op, matching
    // the Python SDK's write side, which never serializes an empty list.
    const mpack_node_t best_shots_node = mpack_node_map_cstr_optional(root, "BestShots");
    if (mpack_node_type(best_shots_node) == mpack_type_array)
    {
        const size_t best_shot_count = mpack_node_array_length(best_shots_node);
        msg.best_shots.reserve(best_shot_count);
        for (size_t i = 0; i < best_shot_count; ++i)
        {
            const mpack_node_t entry = mpack_node_array_at(best_shots_node, i);
            BestShotData best_shot;

            const mpack_node_t track_id_node = mpack_node_map_cstr(entry, "TrackId");
            best_shot.track_id =
                std::string(mpack_node_str(track_id_node), mpack_node_strlen(track_id_node));

            const mpack_node_t bbox_node = mpack_node_map_cstr_optional(entry, "BoundingBox");
            if (mpack_node_type(bbox_node) == mpack_type_map)
            {
                best_shot.x = mpack_node_float(mpack_node_map_cstr(bbox_node, "x"));
                best_shot.y = mpack_node_float(mpack_node_map_cstr(bbox_node, "y"));
                best_shot.width = mpack_node_float(mpack_node_map_cstr(bbox_node, "width"));
                best_shot.height = mpack_node_float(mpack_node_map_cstr(bbox_node, "height"));
            }

            msg.best_shots.push_back(std::move(best_shot));
        }
    }

    // Parse optional ImageData (present when postprocessor has ReceiveInputImage=true)
    const mpack_node_t image_data_node = mpack_node_map_cstr_optional(root, "ImageData");
    if (mpack_node_type(image_data_node) == mpack_type_map)
    {
        ImageData img;
        img.height = mpack_node_u64(mpack_node_map_cstr(image_data_node, "Height"));
        img.width = mpack_node_u64(mpack_node_map_cstr(image_data_node, "Width"));
        img.layout = static_cast<ImageOrder>(
            mpack_node_int(mpack_node_map_cstr(image_data_node, "Layout")));

        const mpack_node_t data_node = mpack_node_map_cstr(image_data_node, "Data");
        const size_t data_size = mpack_node_bin_size(data_node);
        const char* data_ptr = mpack_node_bin_data(data_node);
        img.data.assign(
            reinterpret_cast<const uint8_t*>(data_ptr),
            reinterpret_cast<const uint8_t*>(data_ptr) + data_size);

        msg.image_data = std::move(img);
    }

    // Parse optional Tensors (present when postprocessor has ReceiveBinaryData=true)
    const mpack_node_t tensors_node = mpack_node_map_cstr_optional(root, "Tensors");
    if (mpack_node_type(tensors_node) == mpack_type_array)
    {
        const size_t tensor_count = mpack_node_array_length(tensors_node);
        msg.tensors.reserve(tensor_count);
        for (size_t i = 0; i < tensor_count; ++i)
        {
            const mpack_node_t t = mpack_node_array_at(tensors_node, i);
            TensorData tensor;

            const mpack_node_t name_node = mpack_node_map_cstr(t, "Name");
            tensor.name = std::string(mpack_node_str(name_node), mpack_node_strlen(name_node));

            tensor.data_type = mpack_node_int(mpack_node_map_cstr(t, "DataType"));

            const mpack_node_t shape_node = mpack_node_map_cstr(t, "Shape");
            const size_t rank = mpack_node_array_length(shape_node);
            tensor.shape.reserve(rank);
            for (size_t j = 0; j < rank; ++j)
                tensor.shape.push_back(mpack_node_uint(mpack_node_array_at(shape_node, j)));

            const mpack_node_t data_node = mpack_node_map_cstr(t, "Data");
            const size_t data_size = mpack_node_bin_size(data_node);
            const char* data_ptr = mpack_node_bin_data(data_node);
            tensor.data.assign(
                reinterpret_cast<const uint8_t*>(data_ptr),
                reinterpret_cast<const uint8_t*>(data_ptr) + data_size);

            msg.tensors.push_back(std::move(tensor));
        }
    }

    mpack_tree_destroy(&tree);
    return msg;
}

// ============================================================================
// Serialisation (structs -> msgpack)
// ============================================================================

static void write_object(mpack_writer_t* writer, const ObjectData& obj)
{
    uint32_t field_count = 5U + 1U; // x, y, width, height, confidence, attributes
    if (obj.track_id.has_value())
        ++field_count;
    if (obj.new_track_id.has_value())
        ++field_count;

    mpack_start_map(writer, field_count);

    if (obj.track_id.has_value())
    {
        mpack_write_cstr(writer, "ID");
        mpack_write_str(
            writer,
            obj.track_id->c_str(),
            static_cast<uint32_t>(obj.track_id->size()));
    }
    if (obj.new_track_id.has_value())
    {
        mpack_write_cstr(writer, "NewID");
        mpack_write_str(
            writer,
            obj.new_track_id->c_str(),
            static_cast<uint32_t>(obj.new_track_id->size()));
    }

    mpack_write_cstr(writer, "x");
    mpack_write_float(writer, obj.x);
    mpack_write_cstr(writer, "y");
    mpack_write_float(writer, obj.y);
    mpack_write_cstr(writer, "width");
    mpack_write_float(writer, obj.width);
    mpack_write_cstr(writer, "height");
    mpack_write_float(writer, obj.height);
    mpack_write_cstr(writer, "confidence");
    mpack_write_float(writer, obj.confidence);

    mpack_write_cstr(writer, "attributes");
    mpack_start_map(writer, static_cast<uint32_t>(obj.attributes.size()));
    for (const auto& [key, val]: obj.attributes)
    {
        mpack_write_str(writer, key.c_str(), static_cast<uint32_t>(key.size()));
        mpack_write_str(writer, val.c_str(), static_cast<uint32_t>(val.size()));
    }
    mpack_finish_map(writer);

    mpack_finish_map(writer);
}

static void write_event(mpack_writer_t* writer, const EventData& evt)
{
    mpack_start_map(writer, 2);
    mpack_write_cstr(writer, "ID");
    mpack_write_str(writer, evt.type_id.c_str(), static_cast<uint32_t>(evt.type_id.size()));
    mpack_write_cstr(writer, "Description");
    mpack_write_str(
        writer,
        evt.description.c_str(),
        static_cast<uint32_t>(evt.description.size()));
    mpack_finish_map(writer);
}

static void write_image_data(mpack_writer_t* writer, const ImageData& image)
{
    mpack_write_cstr(writer, "ImageData");
    mpack_start_map(writer, 4);
    mpack_write_cstr(writer, "Height");
    mpack_write_u64(writer, image.height);
    mpack_write_cstr(writer, "Width");
    mpack_write_u64(writer, image.width);
    mpack_write_cstr(writer, "Layout");
    mpack_write_int(writer, static_cast<int>(image.layout));
    mpack_write_cstr(writer, "Data");
    mpack_write_bin(writer, image.data.data(), static_cast<uint32_t>(image.data.size()));
    mpack_finish_map(writer);
}

std::optional<nxai_buffer_t> writeObjectsPostprocessorMessage(
    const ObjectsPostProcessorMessage& msg)
{
    char* message_data = nullptr;
    size_t message_size = 0;
    mpack_writer_t writer;
    mpack_writer_init_growable(&writer, &message_data, &message_size);

    // Root map: Timestamp, DeviceID, DeviceName, Objects, Events, plus UserSettings
    // when forwarded settings are present (the engine only sends it inbound; emitted
    // here so a test host can produce the bytes the engine will) and ImageData
    // when a test host seeds an input frame (ReceiveInputImage contract).
    const bool has_user_settings = !msg.user_settings.empty();
    const bool has_image_data = msg.image_data.has_value();
    mpack_start_map(&writer, 5U + (has_user_settings ? 1U : 0U) + (has_image_data ? 1U : 0U));

    mpack_write_cstr(&writer, "Timestamp");
    mpack_write_i64(&writer, msg.timestamp);

    mpack_write_cstr(&writer, "DeviceID");
    mpack_write_str(&writer, msg.device_id.c_str(), static_cast<uint32_t>(msg.device_id.size()));

    mpack_write_cstr(&writer, "DeviceName");
    mpack_write_str(
        &writer,
        msg.device_name.c_str(),
        static_cast<uint32_t>(msg.device_name.size()));

    mpack_write_cstr(&writer, "Objects");
    mpack_start_map(&writer, static_cast<uint32_t>(msg.objects.size()));
    for (const auto& [class_name, class_objects]: msg.objects)
    {
        mpack_write_str(&writer, class_name.c_str(), static_cast<uint32_t>(class_name.size()));
        mpack_start_array(&writer, static_cast<uint32_t>(class_objects.size()));
        for (const ObjectData& obj: class_objects)
            write_object(&writer, obj);
        mpack_finish_array(&writer);
    }
    mpack_finish_map(&writer);

    mpack_write_cstr(&writer, "Events");
    mpack_start_array(&writer, static_cast<uint32_t>(msg.events.size()));
    for (const EventData& evt: msg.events)
        write_event(&writer, evt);
    mpack_finish_array(&writer);

    if (has_user_settings)
    {
        mpack_write_cstr(&writer, "UserSettings");
        mpack_start_map(&writer, static_cast<uint32_t>(msg.user_settings.size()));
        for (const auto& [name, value]: msg.user_settings)
        {
            mpack_write_str(&writer, name.c_str(), static_cast<uint32_t>(name.size()));
            mpack_write_str(&writer, value.c_str(), static_cast<uint32_t>(value.size()));
        }
        mpack_finish_map(&writer);
    }

    if (has_image_data)
        write_image_data(&writer, *msg.image_data);

    mpack_finish_map(&writer);

    if (mpack_writer_destroy(&writer) != mpack_ok)
    {
        free(message_data);
        return {};
    }

    return nxai_buffer_t(message_data, message_size);
}

std::optional<nxai_buffer_t> writeImagePreprocessorMessage(
    const ImagePreprocessorMessage& image_message)
{
    // Create messagepack message with growable buffer
    mpack_writer_t writer;
    char* message_data = nullptr;
    size_t message_length = 0;
    mpack_writer_init_growable(&writer, &message_data, &message_length);

    // Start root object: Timestamp, DeviceID, DeviceName, ImageData, plus UserSettings
    // when forwarded settings are present (the engine only sends it inbound; emitted
    // here so a test host can produce the bytes the engine will).
    const bool has_user_settings = !image_message.user_settings.empty();
    mpack_start_map(&writer, 4U + (has_user_settings ? 1U : 0U));

    //      Write timestamp
    mpack_write_cstr(&writer, "Timestamp");
    mpack_write_i64(&writer, image_message.timestamp);

    //      Write device ID
    mpack_write_cstr(&writer, "DeviceID");
    mpack_write_cstr(&writer, image_message.device_id.c_str());

    //      Write device name
    mpack_write_cstr(&writer, "DeviceName");
    mpack_write_cstr(&writer, image_message.device_name.c_str());

    if (has_user_settings)
    {
        mpack_write_cstr(&writer, "UserSettings");
        mpack_start_map(&writer, static_cast<uint32_t>(image_message.user_settings.size()));
        for (const auto& [name, value]: image_message.user_settings)
        {
            mpack_write_str(&writer, name.c_str(), static_cast<uint32_t>(name.size()));
            mpack_write_str(&writer, value.c_str(), static_cast<uint32_t>(value.size()));
        }
        mpack_finish_map(&writer);
    }

    //      Write image data
    mpack_write_cstr(&writer, "ImageData");
    mpack_start_map(&writer, 4);

    mpack_write_cstr(&writer, "Height");
    mpack_write_u64(&writer, image_message.image.height);

    mpack_write_cstr(&writer, "Width");
    mpack_write_u64(&writer, image_message.image.width);

    mpack_write_cstr(&writer, "Layout");
    mpack_write_int(&writer, static_cast<int>(image_message.image.layout));

    mpack_write_cstr(&writer, "Data");
    mpack_write_bin(&writer, image_message.image.data.data(), image_message.image.data.size());

    //      Finish "ImageData" map
    mpack_finish_map(&writer);

    // Finish root map
    mpack_finish_map(&writer);

    // Check for errors
    if (mpack_writer_destroy(&writer) != mpack_ok)
    {
        nxai_vlog("Error: Failed to serialize image for preprocessor.\n");
        free(message_data);
        return {};
    }

    return nxai_buffer_t(message_data, message_length);
}

std::optional<ImagePreprocessorMessage> parseImagePreprocessorMessage(std::span<char> message_data)
{
    mpack_tree_t tree;
    mpack_tree_init_data(&tree, message_data.data(), message_data.size());
    mpack_tree_parse(&tree);
    const mpack_node_t root = mpack_tree_root(&tree);

    if (mpack_tree_error(&tree) != mpack_ok)
    {
        std::cerr << "Error: failed to parse incoming msgpack message\n";
        mpack_tree_destroy(&tree);
        return {};
    }

    // TODO: Safety

    ImagePreprocessorMessage image_message;
    image_message.timestamp = mpack_node_i64(mpack_node_map_cstr(root, "Timestamp"));

    const mpack_node_t device_id_node = mpack_node_map_cstr(root, "DeviceID");
    image_message.device_id =
        std::string(mpack_node_str(device_id_node), mpack_node_strlen(device_id_node));

    const mpack_node_t device_name_node = mpack_node_map_cstr(root, "DeviceName");
    image_message.device_name =
        std::string(mpack_node_str(device_name_node), mpack_node_strlen(device_name_node));

    const mpack_node_t image_data_node = mpack_node_map_cstr(root, "ImageData");
    ImageData image_data;
    image_data.height = mpack_node_u64(mpack_node_map_cstr(image_data_node, "Height"));
    image_data.width = mpack_node_u64(mpack_node_map_cstr(image_data_node, "Width"));
    image_data.layout =
        static_cast<ImageOrder>(mpack_node_int(mpack_node_map_cstr(image_data_node, "Layout")));

    const mpack_node_t data_node = mpack_node_map_cstr(image_data_node, "Data");
    const size_t data_size = mpack_node_bin_size(data_node);
    const char* data_ptr = mpack_node_bin_data(data_node);
    image_data.data.assign(
        reinterpret_cast<const uint8_t*>(data_ptr),
        reinterpret_cast<const uint8_t*>(data_ptr) + data_size);

    image_message.image = std::move(image_data);

    if (mpack_tree_destroy(&tree) != mpack_ok)
    {
        std::cerr << "Error: failed to parse incoming image message\n";
        return {};
    }

    return image_message;
}

std::optional<TensorsMessage> parseTensorMessage(const unsigned char* data, size_t size)
{
    mpack_tree_t tree;
    mpack_tree_init_data(&tree, (const char*) data, size);
    mpack_tree_parse(&tree);
    mpack_node_t root = mpack_tree_root(&tree);

    TensorsMessage parsed_message;

    parsed_message.timestamp = mpack_node_i64(mpack_node_map_cstr(root, "Timestamp"));

    const mpack_node_t device_id_node = mpack_node_map_cstr(root, "DeviceID");
    parsed_message.device_id =
        std::string(mpack_node_str(device_id_node), mpack_node_strlen(device_id_node));

    const mpack_node_t device_name_node = mpack_node_map_cstr(root, "DeviceName");
    parsed_message.device_name =
        std::string(mpack_node_str(device_name_node), mpack_node_strlen(device_name_node));

    mpack_node_t outputs_map = mpack_node_map_cstr(root, "Tensors");
    size_t num_tensors = mpack_node_map_count(outputs_map);
    // This will construct num_tensors new tensors in the array
    parsed_message.tensors.resize(num_tensors);

    // Read output shapes ([num tensors][rank]i64)
    mpack_node_t outputs_shapes_outer_array = mpack_node_map_cstr(root, "TensorShapes");
    for (size_t tensor_index = 0; tensor_index < num_tensors; tensor_index++)
    {
        mpack_node_t outputs_shapes_inner_array =
            mpack_node_array_at(outputs_shapes_outer_array, tensor_index);
        size_t rank = mpack_node_array_length(outputs_shapes_inner_array);
        auto& tensor_data = parsed_message.tensors.at(tensor_index);
        tensor_data.shape.reserve(rank);
        for (size_t rank_index = 0; rank_index < rank; rank_index++)
        {
            tensor_data.shape.push_back(
                mpack_node_i64(mpack_node_array_at(outputs_shapes_inner_array, rank_index)));
        }
    }

    // Read output data types ([num tensors]i32)
    mpack_node_t output_data_types_array = mpack_node_map_cstr(root, "TensorDataTypes");
    for (size_t index = 0; index < num_tensors; index++)
    {
        parsed_message.tensors.at(index).data_type = (TensorElementType) mpack_node_i32(
            mpack_node_array_at(output_data_types_array, index));
    }

    // Read outputs ({OutputName:bin})
    for (size_t tensor_index = 0; tensor_index < num_tensors; tensor_index++)
    {
        auto& tensor_data = parsed_message.tensors.at(tensor_index);
        mpack_node_t key_node = mpack_node_map_key_at(outputs_map, tensor_index);
        tensor_data.name = std::string(mpack_node_str(key_node), mpack_node_strlen(key_node));
        mpack_node_t tensor_data_node = mpack_node_map_value_at(outputs_map, tensor_index);
        size_t num_bytes = mpack_node_bin_size(tensor_data_node);
        const char* bin_data = mpack_node_bin_data(tensor_data_node);
        tensor_data.data.assign(bin_data, bin_data + num_bytes);
    }

    if (mpack_tree_destroy(&tree) != mpack_ok)
    {
        nxai_vlog("Warning! Could not properly destroy tree.\n");
        return {};
    }

    return parsed_message;
}

void writeTensorsToMPackWriter(mpack_writer_t* writer, const std::vector<TensorData>& tensors)
{
    mpack_write_cstr(writer, "Tensors");
    mpack_start_map(writer, tensors.size());
    for (const TensorData& tensor: tensors)
    {
        mpack_write_cstr(writer, tensor.name.c_str());
        if (tensor.data.empty() == false)
            mpack_write_bin(writer, (const char*) tensor.data.data(), tensor.data.size());
        else
            mpack_write_nil(writer);
    }
    mpack_finish_map(writer);

    mpack_write_cstr(writer, "TensorShapes");
    mpack_start_array(writer, tensors.size());
    for (const TensorData& tensor: tensors)
    {
        mpack_start_array(writer, tensor.shape.size());
        for (auto shape: tensor.shape)
        {
            mpack_write_i64(writer, shape);
        }
        mpack_finish_array(writer);
    }
    mpack_finish_array(writer);

    mpack_write_cstr(writer, "TensorDataTypes");
    mpack_start_array(writer, tensors.size());
    for (const TensorData& tensor: tensors)
    {
        mpack_write_i32(writer, tensor.data_type);
    }
    mpack_finish_array(writer);

    mpack_finish_map(writer);
}

std::optional<nxai_buffer_t> writeTensorMessage(const TensorsMessage& tensor_message)
{
    // Initialize growable writer
    mpack_writer_t writer;
    char* message_data = nullptr;
    size_t message_length = 0;
    mpack_writer_init_growable(&writer, &message_data, &message_length);

    // Root map: Timestamp, DeviceID, DeviceName, Tensors, TensorShapes, TensorDataTypes,
    // plus UserSettings when forwarded settings are present (the engine only sends it
    // inbound; emitted here so a test host can produce the bytes the engine will).
    const bool has_user_settings = !tensor_message.user_settings.empty();
    mpack_start_map(&writer, 6U + (has_user_settings ? 1U : 0U));

    mpack_write_cstr(&writer, "Timestamp");
    mpack_write_i64(&writer, tensor_message.timestamp);

    mpack_write_cstr(&writer, "DeviceID");
    mpack_write_str(
        &writer,
        tensor_message.device_id.c_str(),
        static_cast<uint32_t>(tensor_message.device_id.size()));

    mpack_write_cstr(&writer, "DeviceName");
    mpack_write_str(
        &writer,
        tensor_message.device_name.c_str(),
        static_cast<uint32_t>(tensor_message.device_name.size()));

    if (has_user_settings)
    {
        mpack_write_cstr(&writer, "UserSettings");
        mpack_start_map(&writer, static_cast<uint32_t>(tensor_message.user_settings.size()));
        for (const auto& [name, value]: tensor_message.user_settings)
        {
            mpack_write_str(&writer, name.c_str(), static_cast<uint32_t>(name.size()));
            mpack_write_str(&writer, value.c_str(), static_cast<uint32_t>(value.size()));
        }
        mpack_finish_map(&writer);
    }

    writeTensorsToMPackWriter(&writer, tensor_message.tensors);

    // Finish writing
    if (mpack_writer_destroy(&writer) != mpack_ok)
    {
        // Free buffer since it was not succesful
        free(writer.buffer);
        return {};
    }

    return nxai_buffer_t(message_data, message_length);
}
