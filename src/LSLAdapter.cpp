#include "LSLAdapter.h"
#include <lsl_c.h>
#include <iostream>

LSLAdapter::LSLAdapter(): inlet_(nullptr) {}

LSLAdapter::~LSLAdapter() {
    if (inlet_) liblsl_destroy_inlet(inlet_);
}

void LSLAdapter::open_inlet(const std::string& name) {
    inlet_ = liblsl_create_inlet(liblsl_streaminfo(name.c_str(), "", 8, 0.0, liblsl_channel_format_t::cft_double, "uid"));
}

bool LSLAdapter::pull_sample(std::vector<std::vector<double>>& buffer) {
    // simple pull implementation...
    double temp[8];
    int got = liblsl_pull_sample_d(static_cast<liblsl_inlet>(inlet_), temp, 0.0);
    if (got > 0) {
        buffer.assign(8, std::vector<double>(1));
        for (int i = 0; i < 8; ++i) buffer[i][0] = temp[i];
        return true;
    }
    return false;
}
