#include "LSLAdapter.h"
#include "RingBuffer.h"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

int main() {
    LSLAdapter adapter;
    if (!adapter.connect("FakeEEG")) return 1;

    const int C = adapter.channel_count();
    const int bufferSamples = 256;
    RingBuffer ring(bufferSamples, C);

    // Launch producer thread
    std::thread producer([&]() {
        const size_t chunkSamples = 32;
        std::vector<float> chunk(C * chunkSamples);

        while (true) {
            std::cout << "[Producer] Pulling chunk...\n";
            std::size_t nSamples = adapter.pull_chunk(chunk.data(), chunk.size());
            std::cout << "[Producer] Got " << nSamples << " samples\n";

            if (nSamples > 0) {
                std::cout << "[Producer] Trying push (samples=" << nSamples << ", elems=" << nSamples * C << ")\n";
                bool ok = ring.push(chunk.data(), nSamples);
                std::cout << "[Producer] Push " << (ok ? "OK" : "FAILED") << "\n";
            } else {
                std::cout << "[Producer] No data available.\n";
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    // Consumer thread in main
    std::vector<float> popBuf(C * 32);
    while (true) {
        if (ring.pop(popBuf.data(), 32)) {
            std::cout << "[Consumer] Popped 32 samples, Ch 0 of first sample = " 
                    << popBuf[0] << "\n";
        } else {
            std::cout << "[Consumer] Waiting for data...\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    producer.join(); // never reached
    return 0;
}
