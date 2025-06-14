#include "LSLAdapter.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

int main() {
    LSLAdapter adapter;

    if (!adapter.connect("FakeEEG")) {
        return 1;
    }

    const int C = adapter.channel_count();
    const int max_samples = 32;
    std::vector<float> buffer(C * max_samples);

    for (int i = 0; i < 10; ++i) {
        std::size_t samples = adapter.pull_chunk(buffer.data(), buffer.size());

        if (samples > 0) {
            std::cout << "Received " << samples << " samples:\n";
            for (std::size_t s = 0; s < samples; ++s) {
                std::cout << "  Sample " << s << ": ";
                for (int c = 0; c < C; ++c) {
                    std::cout << buffer[s * C + c] << " ";
                }
                std::cout << "\n";
            }
        } else {
            std::cout << "No samples received.\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}
