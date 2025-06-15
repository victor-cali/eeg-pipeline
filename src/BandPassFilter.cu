#include "../include/BandPassFilter.h"
#include <cstdio>
#include <cmath>

__global__ void fir_filter(const float* signal, const float* kernel, float* output, int N, int M) {
    extern __shared__ float shared[];

    float* s_kernel = shared;
    float* s_signal = shared + M;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int halo_len = M - 1;

    if (tid < M) {
        s_kernel[tid] = kernel[tid];
    }

    if (gid < N) {
        s_signal[tid + halo_len] = signal[gid];
    } else {
        s_signal[tid + halo_len] = 0.0f;
    }

    if (tid < halo_len) {
        int global_halo_idx = gid - halo_len;
        s_signal[tid] = (global_halo_idx >= 0 && global_halo_idx < N) ? signal[global_halo_idx] : 0.0f;
    }

    __syncthreads();

    if (gid < N) {
        float sum = 0.0f;
        int base = tid + halo_len;
        for (int i = 0; i < M; ++i) {
            sum += s_kernel[i] * s_signal[base - i];
        }
        output[gid] = sum;
    }
}

void CudaBandpassFilter::design_bandpass_filter() {
    float fc1 = f1 / fs;
    float fc2 = f2 / fs;

    h_kernel.resize(M);
    for (int n = 0; n < M; ++n) {
        int m = n - (M - 1) / 2;
        float w = 0.54f - 0.46f * cosf(2.0f * M_PI * n / (M - 1));
        if (m == 0) {
            h_kernel[n] = 2.0f * (fc2 - fc1);
        } else {
            h_kernel[n] = (sinf(2.0f * M_PI * fc2 * m) - sinf(2.0f * M_PI * fc1 * m)) / (M_PI * m);
        }
        h_kernel[n] *= w;
    }
}

void CudaBandpassFilter::checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        throw std::runtime_error(msg);
    }
}

CudaBandpassFilter::CudaBandpassFilter(float sampling_frequency, float low_cutoff_freq, float high_cutoff_freq,
                                       int filter_length, int block_size)
    : fs(sampling_frequency), f1(low_cutoff_freq), f2(high_cutoff_freq),
      M(filter_length), blockSize(block_size), d_kernel(0), d_signal_channel(0),
      d_signal_channel_capacity_bytes(0) {

    if (M % 2 == 0) {
        throw std::invalid_argument("Filter length (M) must be odd.");
    }

    design_bandpass_filter();

    checkCudaError(cudaMalloc((void**)&d_kernel, M * sizeof(float)), "Failed to allocate d_kernel");
    checkCudaError(cudaMemcpy(d_kernel, h_kernel.data(), M * sizeof(float), cudaMemcpyHostToDevice), "Copy to d_kernel failed");

    sharedMemSize = (M + blockSize + M - 1) * sizeof(float);
}

CudaBandpassFilter::~CudaBandpassFilter() {
    if (d_kernel) cudaFree(d_kernel);
    if (d_signal_channel) cudaFree(d_signal_channel);
}

void CudaBandpassFilter::process(float* data, std::size_t channels, std::size_t samples) {
    int N_channel = static_cast<int>(samples);
    size_t required_bytes = N_channel * sizeof(float);

    if (!d_signal_channel || d_signal_channel_capacity_bytes < required_bytes) {
        if (d_signal_channel) cudaFree(d_signal_channel);
        checkCudaError(cudaMalloc((void**)&d_signal_channel, required_bytes), "Failed to allocate d_signal_channel");
        d_signal_channel_capacity_bytes = required_bytes;
    }

    int gridSize = (N_channel + blockSize - 1) / blockSize;

    for (std::size_t c = 0; c < channels; ++c) {
        float* current_channel = data + (c * N_channel);

        checkCudaError(cudaMemcpy(d_signal_channel, current_channel, required_bytes, cudaMemcpyHostToDevice), "Memcpy to device failed");
        fir_filter<<<gridSize, blockSize, sharedMemSize>>>(d_signal_channel, d_kernel, d_signal_channel, N_channel, M);
        checkCudaError(cudaGetLastError(), "Kernel launch failed");
        checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");

        checkCudaError(cudaMemcpy(current_channel, d_signal_channel, required_bytes, cudaMemcpyDeviceToHost), "Memcpy back to host failed");
    }
}
