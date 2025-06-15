#include "LSLOutput.h"
#include <iostream>

LSLOutput::LSLOutput(const std::string& name, const std::string& type, const std::vector<std::string>& channel_names, double sampling_rate) {
    const int num_channels = channel_names.size();

    // 1. Create stream info using the C API
    info_ = lsl_create_streaminfo(name.c_str(), type.c_str(), num_channels, sampling_rate, cft_double64, "myuid_features1234");

    // 2. Add channel metadata for clear labeling in LSL clients
    lsl_xml_ptr desc = lsl_get_desc(info_);
    lsl_xml_ptr channels_node = lsl_append_child(desc, "channels");
    for (const auto& ch_name : channel_names) {
        lsl_xml_ptr channel_node = lsl_append_child(channels_node, "channel");
        lsl_append_child_value(channel_node, "label", ch_name.c_str());
        lsl_append_child_value(channel_node, "unit", "feature_value");
    }

    // 3. Create the outlet
    outlet_ = lsl_create_outlet(info_, 0, 360); // Use a chunk size of 0 for variable size, 360s buffer
    if (!outlet_) {
        std::cerr << "Error: Could not create LSL outlet." << std::endl;
        return;
    }
    std::cout << "LSL output stream '" << name << "' created and broadcasting.\n";

    // 4. Start the dedicated output thread
    output_thread_ = std::thread(&LSLOutput::output_thread_func, this);
}

LSLOutput::~LSLOutput() {
    // Signal the thread to stop and clean up
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_thread_ = true;
    }
    cv_.notify_one(); // Wake up the thread so it can exit
    if (output_thread_.joinable()) {
        output_thread_.join();
    }

    // Destroy the outlet (this also destroys the streaminfo)
    if (outlet_) {
        lsl_destroy_outlet(outlet_);
        std::cout << "LSL output stream destroyed.\n";
    }
}

void LSLOutput::send_features(const std::vector<double>& features) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        data_queue_.push(features);
    }
    cv_.notify_one(); // Notify the output thread that new data is available
}

void LSLOutput::output_thread_func() {
    while (true) {
        std::vector<double> features_to_send;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            // Wait until the queue is not empty or the thread is stopped
            cv_.wait(lock, [this] { return !data_queue_.empty() || stop_thread_; });

            if (stop_thread_ && data_queue_.empty()) {
                break; // Exit loop if stopped and no more data to send
            }

            features_to_send = data_queue_.front();
            data_queue_.pop();
        }

        // Send the data using the LSL C API
        if (outlet_ && !features_to_send.empty()) {
            lsl_push_sample_d(outlet_, features_to_send.data());
        }
    }
}