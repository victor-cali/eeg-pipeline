#include "LSLAdapter.h"
#include "FeatureExtractor.h"
#include "LSLOutput.h" // For sending processed features
#include "RingBuffer.h"
#include "BandPassFilter.h" // Include the header for CudaBandpassFilter
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <omp.h> // Required for OpenMP functions

// Helper function to find the largest power of 2 less than or equal to n
size_t largest_power_of_2(size_t n) {
    if (n == 0) return 0;
    return static_cast<size_t>(std::pow(2, std::floor(std::log2(n))));
}

int main() {
    // --- LSL Input Setup ---
    LSLAdapter adapter;
    std::cout << "Attempting to connect to LSL stream 'SimpleStream'...\n";
    if (!adapter.connect("SimpleStream")) {
        return 1;
    }
    const int C = adapter.channel_count();

    const int num_channels = adapter.channel_count();
    const double sampling_rate =  128.0f;//adapter.sampling_rate();
    const int max_chunk_size = 512;
    std::vector<float> buffer(num_channels * max_chunk_size);

    CudaBandpassFilter filter(
        128.0f, //adapter.sampling_rate(), // fs
        4.0f,                    // low cutoff in Hz
        30.0f                    // high cutoff in Hz
    );

    // --- LSL Output Setup ---
    std::vector<std::string> feature_channel_names;
    const std::vector<std::string> feature_names = {"RMS", "Var", "Ent", "Alpha", "Beta"};
    for (int c = 0; c < num_channels; ++c) {
        for (const auto& feature_name : feature_names) {
            feature_channel_names.push_back("Ch" + std::to_string(c) + "_" + feature_name);
        }
    }
    const double feature_sampling_rate = 2.0; 
    LSLOutput output_stream("ProcessedEEGFeatures", "Features", feature_channel_names, feature_sampling_rate);
    
    std::cout << "\n--- Starting Live Parallel Feature Extraction ---" << std::endl;

    // Main loop
    for (int i = 0; i < 100; ++i) {
        std::size_t samples_read = adapter.pull_chunk(buffer.data(), buffer.size());

        if (samples_read > 0) {
            size_t window_size = largest_power_of_2(samples_read);
            
            if (window_size == 0) {
                std::cout << "Chunk too small for FFT (" << samples_read << " samples). Skipping." << std::endl;
                continue;
            }

            std::cout << "--- Received Chunk, processing " << window_size << " samples ---\n";

            // ✅ Apply CUDA filter
            filter.process(buffer.data(), adapter.channel_count(), window_size);

            // Optionally inspect or pass filtered data downstream
            std::cout << "[Filter] Applied bandpass filter on " << adapter.channel_count()
                      << " channels × " << window_size << " samples\n";

            std::vector<FeatureExtractor::Features> all_features(num_channels);

            #pragma omp parallel for
            for (int c = 0; c < num_channels; ++c) {
                std::vector<double> channel_window;
                channel_window.reserve(window_size);
                for (size_t s = 0; s < window_size; ++s) {
                    channel_window.push_back(buffer[s * num_channels + c]);
                }
                all_features[c] = FeatureExtractor::compute_features(channel_window, sampling_rate);
            }

            // --- Flatten and Send Features via LSL Output ---
            std::vector<double> features_flat;
            features_flat.reserve(num_channels * feature_names.size());
            for (int c = 0; c < num_channels; ++c) {
                features_flat.push_back(all_features[c].rms);
                features_flat.push_back(all_features[c].variance);
                features_flat.push_back(all_features[c].entropy);
                features_flat.push_back(all_features[c].alpha_power);
                features_flat.push_back(all_features[c].beta_power);
            }
            output_stream.send_features(features_flat);
            
            // v-- THIS IS THE PART THAT DISPLAYS THE FEATURES --v
            std::cout << "Features computed and sent to LSL. Values:\n";
            for (int c = 0; c < num_channels; ++c) {
                const auto& features = all_features[c];
                std::cout << "  Channel " << c 
                          << ": RMS=" << features.rms 
                          << ", Var=" << features.variance
                          << ", Ent=" << features.entropy
                          << ", Alpha=" << features.alpha_power
                          << ", Beta=" << features.beta_power
                          << "\n";
            }
            std::cout << std::endl;

        } else {
            std::cout << "[Consumer] Waiting for data...\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    std::cout << "Loop finished. Waiting a moment for final features to be sent...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    return 0;
}
