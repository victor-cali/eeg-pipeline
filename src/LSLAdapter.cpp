#include "LSLAdapter.h"
#include "RingBuffer.h"  // include RingBuffer implementation
#include <iostream>

LSLAdapter::LSLAdapter() = default;

LSLAdapter::~LSLAdapter() {
    stop();
    if (inlet_) {
        delete inlet_;
        inlet_ = nullptr;
    }
}

bool LSLAdapter::connect(const std::string& stream_name, double timeout) {
    std::cout << "Resolving stream with name: " << stream_name << "...\n";
    auto results = lsl::resolve_stream("name", stream_name, 1, timeout);
    if (results.empty()) {
        std::cerr << "No stream found with name: " << stream_name << "\n";
        return false;
    }

    lsl::stream_info info = results[0];
    channel_count_ = info.channel_count();
    sampling_rate_ = static_cast<float>(info.nominal_srate());

    std::cout << "Found stream: " << info.name()
              << ", channels: " << channel_count_
              << ", rate: " << sampling_rate_ << "\n";

    inlet_ = new lsl::stream_inlet(info);
    inlet_->set_postprocessing(lsl::post_ALL);
    return true;
}

std::size_t LSLAdapter::pull_chunk(float* dst, std::size_t max_elements, double timeout) {
    if (!inlet_) {
        std::cerr << "Inlet not initialized.\n";
        return 0;
    }

    std::vector<double> timestamps(max_elements / channel_count_);
    std::size_t n_read = inlet_->pull_chunk_multiplexed(
        dst, timestamps.data(), max_elements, timestamps.size(), timeout
    );

    return n_read / channel_count_;
}

void LSLAdapter::set_output_buffer(RingBuffer* buf) {
    ring_ = buf;
}

void LSLAdapter::start() {
    if (running_) return;
    running_ = true;
    thread_ = std::thread(&LSLAdapter::run, this);
}

void LSLAdapter::stop() {
    if (running_) {
        running_ = false;
        if (thread_.joinable())
            thread_.join();
    }
}

void LSLAdapter::run() {
    if (!inlet_ || !ring_) {
        std::cerr << "[LSLAdapter] Inlet or output buffer not set!\n";
        return;
    }

    std::vector<float> chunk(channel_count_ * 32);
    std::vector<double> timestamps(32);

    while (running_) {
        std::size_t n_elems = inlet_->pull_chunk_multiplexed(
            chunk.data(), timestamps.data(), chunk.size(), timestamps.size(), 0.5
        );

        std::size_t n_samples = n_elems / channel_count_;
        if (n_samples > 0) {
            if (!ring_->push(chunk.data(), n_samples)) {
                std::cerr << "[LSLAdapter] Buffer full, dropped " << n_samples << " samples\n";
            } else {
                std::cout << "[LSLAdapter] Pushed " << n_samples << " samples\n";
            }
        }
    }

    std::cout << "[LSLAdapter] Acquisition thread exiting\n";
}
