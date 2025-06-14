#pragma once
#include <vector>
#include <string>
class LSLAdapter {
public:
    LSLAdapter();
    ~LSLAdapter();
    void open_inlet(const std::string& name);
    bool pull_sample(std::vector<std::vector<double>>& buffer);
private:
    // opaque LSL inlet pointer
    void* inlet_;
};
