#pragma once

#include <vector>
#include <atomic>
#include <cstddef>

class RingBuffer {
public:
    RingBuffer(std::size_t capacitySamples, std::size_t channels);

    bool push(const float* src, std::size_t nSamples);   // samples in
    bool pop(float* dst, std::size_t nSamples);          // samples out

    std::size_t freeSamples() const;
    std::size_t usedSamples() const;

private:
    std::size_t index(std::size_t sampleIndex) const;

    std::size_t capacitySamples_;
    std::size_t channels_;
    std::size_t capacityElements_;

    std::vector<float> buffer_;               // raw float buffer: size = channels × capacitySamples
    std::atomic<std::size_t> head_;           // next write position (in samples)
    std::atomic<std::size_t> tail_;           // next read position (in samples)
};
