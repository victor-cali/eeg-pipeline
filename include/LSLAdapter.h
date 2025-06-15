#pragma once

#include <lsl_cpp.h>
#include <vector>
#include <string>

class LSLAdapter {
public:
    LSLAdapter();
    ~LSLAdapter();

    // Connects to an LSL stream by name and initializes inlet
    bool connect(const std::string& stream_name, double timeout = 5.0);

    // Pulls a chunk of data into a preallocated float buffer.
    // Returns number of samples read
    std::size_t pull_chunk(float* dst, std::size_t max_elements, double timeout = 0.2);

    int channel_count() const { return channel_count_; }
    float sampling_rate() const { return sampling_rate_; }

private:
    lsl::stream_inlet* inlet_ = nullptr;
    int channel_count_ = 0;
    float sampling_rate_ = 0.0f;
};
