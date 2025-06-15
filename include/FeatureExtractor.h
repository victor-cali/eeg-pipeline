#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <vector>
#include <cmath>
#include <numeric>
#include <map>

class FeatureExtractor {
public:
    struct Features {
        double rms = 0.0;
        double variance = 0.0;
        double entropy = 0.0;
        double alpha_power = 0.0;
        double beta_power = 0.0;  
        double theta_power = 0.0; 
    };

    /**
     * @brief Computes features for a single window of EEG data from one channel.
     * @param window_data A vector of doubles representing the signal in a time window.
     * @param sampling_rate The sampling rate of the signal in Hz.
     * @return A Features struct containing the calculated feature values.
     */
    static Features compute_features(const std::vector<double>& window_data, double sampling_rate);

private:
    static double calculate_rms(const std::vector<double>& window_data);
    static double calculate_variance(const std::vector<double>& window_data);
    static double calculate_entropy(const std::vector<double>& window_data);

    /**
     * @brief Calculates band powers (alpha, beta, theta) using FFT.
     * @param window_data The signal data for the window.
     * @param sampling_rate The sampling rate of the signal in Hz.
     * @param features A reference to the features struct to be filled.
     */
    static void calculate_band_powers(const std::vector<double>& window_data, double sampling_rate, Features& features);
};

#endif // FEATURE_EXTRACTOR_H
