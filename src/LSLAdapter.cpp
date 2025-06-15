#include "LSLAdapter.h"
#include <iostream>

LSLAdapter::LSLAdapter() = default;

// Destructor definition
LSLAdapter::~LSLAdapter() {
    // If inlet_ was successfully created (not nullptr), delete it.
    // The lsl::stream_inlet destructor will automatically handle disconnection.
    if (inlet_ != nullptr) {
        delete inlet_; 
        inlet_ = nullptr; // Good practice to nullify after deleting
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
    std::size_t n_read = inlet_->pull_chunk_multiplexed(dst, timestamps.data(), max_elements, timestamps.size(), timeout);
    return n_read / channel_count_; // return number of samples
}
