#ifndef LSL_OUTPUT_H
#define LSL_OUTPUT_H

#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include "lsl_c.h" // Using the C API

/**
 * @class LSLOutput
 * @brief Manages a dedicated LSL output stream on a separate thread.
 *
 * This class creates an LSL outlet and runs a background thread to send feature
 * data with low latency, preventing the main processing loop from blocking.
 */
class LSLOutput {
public:
    /**
     * @brief Constructs the LSL output stream.
     * @param name The name of the stream (e.g., "ProcessedEEGFeatures").
     * @param type The type of the stream (e.g., "Features").
     * @param channel_names A vector of names for each feature channel.
     * @param sampling_rate The effective sampling rate of the feature stream.
     */
    LSLOutput(const std::string& name, const std::string& type, const std::vector<std::string>& channel_names, double sampling_rate);

    /**
     * @brief Destructor that stops the streaming thread and cleans up resources.
     */
    ~LSLOutput();

    /**
     * @brief Pushes a vector of features to the output queue to be sent. This method is thread-safe.
     * @param features A vector of doubles representing the features for one time point.
     */
    void send_features(const std::vector<double>& features);

private:
    /**
     * @brief The main function for the dedicated output thread.
     */
    void output_thread_func();

    lsl_streaminfo info_;
    lsl_outlet outlet_;

    std::thread output_thread_;
    std::queue<std::vector<double>> data_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    bool stop_thread_ = false;
};

#endif // LSL_OUTPUT_H
