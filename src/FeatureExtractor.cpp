#include "FeatureExtractor.h"
#include "fft.hpp" // For FFT
#include <iostream>
#include <numeric>
#include <map>
#include <algorithm>
#include <complex> // For FFT

FeatureExtractor::Features FeatureExtractor::compute_features(const std::vector<double>& window_data, double sampling_rate) {
    if (window_data.empty()) {
        std::cerr << "Warning: Window data is empty. Cannot compute features." << std::endl;
        return {};
    }

    Features features;
    features.rms = calculate_rms(window_data);
    features.variance = calculate_variance(window_data);
    features.entropy = calculate_entropy(window_data);
    
    calculate_band_powers(window_data, sampling_rate, features);

    return features;
}

double FeatureExtractor::calculate_rms(const std::vector<double>& window_data) {
    double sum_of_squares = 0.0;
    for (double val : window_data) {
        sum_of_squares += val * val;
    }
    return std::sqrt(sum_of_squares / window_data.size());
}

double FeatureExtractor::calculate_variance(const std::vector<double>& window_data) {
    double sum = std::accumulate(window_data.begin(), window_data.end(), 0.0);
    double mean = sum / window_data.size();

    double squared_diff_sum = 0.0;
    for (double val : window_data) {
        squared_diff_sum += (val - mean) * (val - mean);
    }
    
    return squared_diff_sum / window_data.size();
}

double FeatureExtractor::calculate_entropy(const std::vector<double>& window_data) {
    //Create a histogram to count occurrences of each value
    
    std::map<int, int> counts;
    for (double val : window_data) {
        counts[static_cast<int>(val * 100)]++;
    }

    //Calculate probabilities and sum for entropy
    double entropy = 0.0;
    const double total_samples = window_data.size();
    for (auto const& [value_bin, count] : counts) {
        if (count > 0) {
            double probability = static_cast<double>(count) / total_samples;
            entropy -= probability * std::log2(probability);
        }
    }
    
    return entropy;
}


void FeatureExtractor::calculate_band_powers(const std::vector<double>& window_data, double sampling_rate, Features& features) {
    size_t n = window_data.size();

    if (n == 0 || (n & (n - 1)) != 0) {
        std::cerr << "Warning: Data size (" << n << ") is not a power of 2. FFT results may be inaccurate." << std::endl;
        return;
    }

    // 1. Prepare data for FFT
    std::vector<std::complex<double>> fft_input(n);
    for (size_t i = 0; i < n; ++i) {
        fft_input[i] = std::complex<double>(window_data[i], 0);
    }

    // 2. Perform FFT
    Fft::transform(fft_input);

    // 3. Calculate Power Spectrum
    std::vector<double> power_spectrum(n / 2);
    for (size_t i = 0; i < n / 2; ++i) {
        power_spectrum[i] = std::norm(fft_input[i]);
    }

    // 4. Define frequency bands
    const double freq_resolution = sampling_rate / n;
    const double theta_low = 4.0, theta_high = 8.0;
    const double alpha_low = 8.0, alpha_high = 13.0;
    const double beta_low = 13.0, beta_high = 30.0;

    // 5. Sum power in each band
    for (size_t i = 0; i < n / 2; ++i) {
        double freq = i * freq_resolution;
        if (freq >= theta_low && freq < theta_high) {
            features.theta_power += power_spectrum[i];
        }
        if (freq >= alpha_low && freq < alpha_high) {
            features.alpha_power += power_spectrum[i];
        }
        if (freq >= beta_low && freq < beta_high) {
            features.beta_power += power_spectrum[i];
        }
    }
}
