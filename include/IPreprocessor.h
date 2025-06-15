// include/IPreprocessor.h
#pragma once
#include <cstddef>
#include <vector>

/**
 * Common interface for every processing stage in the real-time pipeline.
 *
 * All stages receive the same memory layout:
 *   • data  – contiguous buffer of floats, channel-major
 *             [ch0-s0, … ch0-sN-1, ch1-s0, …]
 *   • C     – number of channels
 *   • N     – samples per channel in this window
 *
 * A stage may:
 *   1) modify the buffer in place (e.g., filtering)
 *   2) leave the buffer untouched but produce a side-vector of features
 *   3) both (rare but allowed)
 */
class IPreprocessor {
public:
    virtual ~IPreprocessor() = default;

    /**
     * Process the buffer (in place) or extract information from it.
     *
     * Implementations must return quickly; heavy work should already be
     * parallelised internally (CUDA, OpenMP, std::thread, etc.).
     *
     * @param data      Pointer to the first float in the window
     * @param channels  Number of EEG channels (rows)
     * @param samples   Samples per channel (cols)
     */
    virtual void process(float* data,
                         std::size_t channels,
                         std::size_t samples) = 0;

    /** True if this stage produces a feature vector. Default = false. */
    virtual bool has_features() const { return false; }

    /**
     * Return a read-only reference to the latest features.
     * Only safe to call if has_features()==true.
     *
     * Default implementation returns an empty static vector.
     */
    virtual const std::vector<float>& features() const {
        static const std::vector<float> empty;
        return empty;
    }

    // Non-copyable, non-movable to avoid slicing and dangling GPU handles
    IPreprocessor(const IPreprocessor&)            = delete;
    IPreprocessor& operator=(const IPreprocessor&) = delete;
    IPreprocessor(IPreprocessor&&)                 = delete;
    IPreprocessor& operator=(IPreprocessor&&)      = delete;

protected:
    IPreprocessor() = default;
};
