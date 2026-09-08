#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <queue>
#include <type_traits>
#include <vector>
#include <ros/serialization.h>

struct InputQueueStats
{
    size_t payload_bytes = 0;
    size_t peak_payload_bytes = 0;
    size_t memory_limit_bytes = 0;
    size_t spilled_messages = 0;
    size_t spilled_bytes = 0;
    double write_seconds = 0;
    double read_seconds = 0;
};

// Externally synchronized, like the ROS queues it replaces.
template <class MessagePtr>
class InputQueue
{
public:
    explicit InputQueue(InputQueueStats& stats) : stats_(stats) {}

    void setCacheDirectory(const std::filesystem::path& path) { cache_directory_ = path; }

    void push(const MessagePtr& message)
    {
        Entry entry;
        if (stats_.memory_limit_bytes &&
            stats_.payload_bytes + message->data.size() > stats_.memory_limit_bytes)
        {
            const auto start = Clock::now();
            entry.path = cache_directory_ / (std::to_string(next_file_++) + ".bin");
            entry.serialized_size = ros::serialization::serializationLength(*message);
            std::vector<uint8_t> bytes(entry.serialized_size);
            ros::serialization::OStream stream(bytes.data(), bytes.size());
            ros::serialization::serialize(stream, *message);
            std::filesystem::create_directories(cache_directory_);
            std::ofstream file;
            file.exceptions(std::ios::failbit | std::ios::badbit);
            file.open(entry.path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            file.close();
            ++stats_.spilled_messages;
            stats_.spilled_bytes += bytes.size();
            stats_.write_seconds += std::chrono::duration<double>(Clock::now() - start).count();
        }
        else
        {
            entry.message = message;
            addResidentBytes(message->data.size());
        }
        messages_.push(std::move(entry));
    }

    void pop()
    {
        const auto& entry = messages_.front();
        if (entry.message) stats_.payload_bytes -= entry.message->data.size();
        if (!entry.path.empty()) std::filesystem::remove(entry.path);
        messages_.pop();
    }

    const MessagePtr& front()
    {
        auto& entry = messages_.front();
        if (!entry.message)
        {
            const auto start = Clock::now();
            std::vector<uint8_t> bytes(entry.serialized_size);
            std::ifstream file;
            file.exceptions(std::ios::failbit | std::ios::badbit);
            file.open(entry.path, std::ios::binary);
            file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
            using Message = std::remove_const_t<typename MessagePtr::element_type>;
            boost::shared_ptr<Message> message(new Message);
            ros::serialization::IStream stream(bytes.data(), bytes.size());
            ros::serialization::deserialize(stream, *message);
            entry.message = message;
            // One loaded head per queue may temporarily exceed the incoming-message budget.
            addResidentBytes(message->data.size());
            stats_.read_seconds += std::chrono::duration<double>(Clock::now() - start).count();
        }
        return entry.message;
    }
    bool empty() const { return messages_.empty(); }

private:
    using Clock = std::chrono::steady_clock;
    struct Entry
    {
        MessagePtr message;
        std::filesystem::path path;
        uint32_t serialized_size = 0;
    };

    void addResidentBytes(size_t bytes)
    {
        stats_.payload_bytes += bytes;
        stats_.peak_payload_bytes = std::max(stats_.peak_payload_bytes, stats_.payload_bytes);
    }

    InputQueueStats& stats_;
    std::queue<Entry> messages_;
    std::filesystem::path cache_directory_;
    size_t next_file_ = 0;
};
