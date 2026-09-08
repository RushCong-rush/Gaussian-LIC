#pragma once

#include <algorithm>
#include <cstddef>
#include <queue>

struct InputQueueStats
{
    size_t payload_bytes = 0;
    size_t peak_payload_bytes = 0;
};

// Externally synchronized, like the ROS queues it replaces.
template <class MessagePtr>
class InputQueue
{
public:
    explicit InputQueue(InputQueueStats& stats) : stats_(stats) {}

    void push(const MessagePtr& message)
    {
        messages_.push(message);
        stats_.payload_bytes += message->data.size();
        stats_.peak_payload_bytes = std::max(stats_.peak_payload_bytes, stats_.payload_bytes);
    }

    void pop()
    {
        stats_.payload_bytes -= messages_.front()->data.size();
        messages_.pop();
    }

    const MessagePtr& front() const { return messages_.front(); }
    bool empty() const { return messages_.empty(); }

private:
    InputQueueStats& stats_;
    std::queue<MessagePtr> messages_;
};
