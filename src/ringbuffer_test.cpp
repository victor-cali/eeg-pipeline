#include "RingBuffer.h"
#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <atomic>
#include <chrono>

// 🧠 These must be defined BEFORE use
const size_t channels = 16;
const size_t capacitySamples = 256;

std::atomic<bool> running{true};

void producer_thread(RingBuffer& ring) {
    std::vector<float> buffer(channels * 16);
    size_t sample_count = 0;

    while (running.load()) {
        for (size_t s = 0; s < 16; ++s) {
            for (size_t c = 0; c < channels; ++c) {
                buffer[s * channels + c] = std::sin(0.01f * float(sample_count + s)) + float(c);
            }
        }

        if (ring.push(buffer.data(), 16)) {
            std::cout << "[Producer] Pushed 16 samples\n";
        } else {
            std::cout << "[Producer] Buffer full!\n";
        }

        sample_count += 16;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

void consumer_thread(RingBuffer& ring) {
    std::vector<float> buffer(channels * 16);

    while (running.load()) {
        if (ring.pop(buffer.data(), 16)) {
            float sum = 0.0f;
            for (size_t i = 0; i < channels * 16; ++i)
                sum += buffer[i];

            std::cout << "[Consumer] Got 16 samples, sum = " << sum << "\n";
        } else {
            std::cout << "[Consumer] Buffer empty\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

int main() {
    std::cout << "[Test] Initializing ring buffer...\n";

    // ✅ Pass valid values
    RingBuffer ring(capacitySamples, channels);

    std::thread prod(producer_thread, std::ref(ring));
    std::thread cons(consumer_thread, std::ref(ring));

    std::this_thread::sleep_for(std::chrono::seconds(5));
    running.store(false);

    prod.join();
    cons.join();

    std::cout << "[Test] Finished.\n";
    return 0;
}
