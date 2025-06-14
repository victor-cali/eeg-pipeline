#include "LSLAdapter.h"
#include <iostream>

int main() {
    LSLAdapter inlet;
    inlet.open_inlet("TestStream");
    std::vector<std::vector<double>> sample;
    if (inlet.pull_sample(sample)) {
        std::cout << "Received " << sample.size() << " channels\n";
    }
    return 0;
}
