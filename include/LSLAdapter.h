#pragma once

#include <lsl_cpp.h>
#include <vector>
#include <string>
#include <thread>
#include <atomic>

class RingBuffer;  // Forward declaration

class LSLAdapter {
public:
    LSLAdapter();
    ~LSLAdapter();

    bool connect(const std::string& stream_name, double timeout = 5.0);
    std::size_t pull_chunk(float* dst, std::size_t max_elements, double timeout = 0.2);

    void set_output_buffer(RingBuffer* buf);
    void start();  // Start acquisition thread
    void stop();   // Stop acquisition thread

    int channel_count() const { return channel_count_; }
    float sampling_rate() const { return sampling_rate_; }

private:
    void run();  // Internal acquisition loop

    lsl::stream_inlet* inlet_ = nullptr;
    int channel_count_ = 0;
    float sampling_rate_ = 0.0f;

    std::thread thread_;
    std::atomic<bool> running_{false};
    RingBuffer* ring_ = nullptr;
};
