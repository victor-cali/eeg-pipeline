#pragma once

#include "IPreprocessor.h"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

#define FILTER_LEN 63 

__global__ void fir_filter(const float* signal, const float* kernel, float* output, int N, int M);

class CudaBandpassFilter : public IPreprocessor {
public:
    CudaBandpassFilter(float sampling_frequency, float low_cutoff_freq, float high_cutoff_freq,
                       int filter_length = FILTER_LEN, int block_size = 256);
    ~CudaBandpassFilter() override;

    void process(float* data, std::size_t channels, std::size_t samples) override;

private:
    float fs;
    float f1;
    float f2;
    int M;
    int blockSize;
    int sharedMemSize;

    std::vector<float> h_kernel;

    float* d_kernel;
    float* d_signal_channel;

    size_t d_signal_channel_capacity_bytes;

    void design_bandpass_filter();
    void checkCudaError(cudaError_t err, const char* msg);
};
