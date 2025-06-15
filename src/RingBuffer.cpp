#include "RingBuffer.h"
#include <cstring>
#include <iostream>

RingBuffer::RingBuffer(std::size_t capacitySamples, std::size_t channels)
    : capacitySamples_(capacitySamples),
      channels_(channels),
      capacityElements_(capacitySamples * channels),
      head_(0),
      tail_(0)
{
    buffer_.resize(capacityElements_, 0.0f);  //safe explicit allocation
    std::cout << "[RingBuffer::ctor] capacitySamples = " << capacitySamples_ << "\n";
    std::cout << "[RingBuffer::ctor] channels = " << channels_ << "\n";
    std::cout << "[RingBuffer::ctor] capacityElements = " << capacityElements_ << "\n";
    std::cout << "[RingBuffer::ctor] buffer_.size() = " << buffer_.size() << "\n";
}

bool RingBuffer::push(const float* src, std::size_t nSamples) {
    std::size_t nElems = nSamples * channels_;
    if (nSamples == 0 || nElems > capacityElements_) {
        std::cerr << "[RingBuffer::push] ERROR: Invalid push size\n";
        std::terminate();
    }

    std::size_t head = head_.load(std::memory_order_relaxed);
    std::size_t tail = tail_.load(std::memory_order_acquire);
    std::size_t used = head - tail;

    if ((capacitySamples_ - used) < nSamples) {
        std::cout << "[RingBuffer::push] Buffer full\n";
        return false;
    }

    std::size_t start = index(head);
    std::size_t end = start + nElems;

    std::cout << "[RingBuffer::push] head=" << head << ", tail=" << tail
              << ", start=" << start << ", end=" << end << ", elems=" << nElems << "\n";

    if (end <= capacityElements_) {
        std::memcpy(&buffer_[start], src, nElems * sizeof(float));
    } else {
        std::size_t firstPart = capacityElements_ - start;
        std::memcpy(&buffer_[start], src, firstPart * sizeof(float));
        std::memcpy(&buffer_[0], src + firstPart, (nElems - firstPart) * sizeof(float));
    }

    head_.store(head + nSamples, std::memory_order_release);
    return true;
}

bool RingBuffer::pop(float* dst, std::size_t nSamples) {
    std::cout << "[RingBuffer::pop] Called with nSamples = " << nSamples << "\n";

    std::size_t nElems = nSamples * channels_;
    if (nSamples == 0 || nElems > capacityElements_) {
        std::cerr << "[RingBuffer::pop] ERROR: Invalid pop size\n";
        std::terminate();
    }

    std::size_t head = head_.load(std::memory_order_acquire);
    std::size_t tail = tail_.load(std::memory_order_relaxed);
    std::size_t available = head - tail;

    if (available < nSamples) {
        std::cout << "[RingBuffer::pop] Skipping read, not enough data.\n";
        return false;
    }

    std::size_t start = index(tail);
    std::size_t end = start + nElems;

    std::cout << "[RingBuffer::pop] head=" << head << ", tail=" << tail
              << ", start=" << start << ", end=" << end << ", elems=" << nElems << "\n";

    if (end <= capacityElements_) {
        std::memcpy(dst, &buffer_[start], nElems * sizeof(float));
    } else {
        std::size_t firstPart = capacityElements_ - start;
        std::memcpy(dst, &buffer_[start], firstPart * sizeof(float));
        std::memcpy(dst + firstPart, &buffer_[0], (nElems - firstPart) * sizeof(float));
    }

    tail_.store(tail + nSamples, std::memory_order_release);
    return true;
}

std::size_t RingBuffer::freeSamples() const {
    std::size_t head = head_.load(std::memory_order_acquire);
    std::size_t tail = tail_.load(std::memory_order_acquire);
    return capacitySamples_ - (head - tail);
}

std::size_t RingBuffer::usedSamples() const {
    std::size_t head = head_.load(std::memory_order_acquire);
    std::size_t tail = tail_.load(std::memory_order_acquire);
    return head - tail;
}

std::size_t RingBuffer::index(std::size_t sampleIndex) const {
    return (sampleIndex % capacitySamples_) * channels_;
}
