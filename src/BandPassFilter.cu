#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#define FILTER_LEN 64

__global__ void fir_filter(const float* __restrict__ signal, const float* __restrict__ kernel, float* output, int N, int M){
    extern __shared__ float shared[];

    float* s_kernel = shared;
    float* s_signal = shared + M;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (tid < M){
        s_kernel[tid] = kernel[tid];
    }

    __syncthreads();

    int halo = M - 1;
    int s_index = tid + halo;

    if (gid < N){
        s_signal[s_index] = signal[gid];
    } else {
        s_signal[s_index] = 0.0f;
    }

    if (tid < halo){
        int idx = gid - halo;
        s_signal[tid] = (idx >= 0) ? s_signal[idx] : 0.0f;
    }

    __syncthreads();

    if (gid < N){
        float sum = 0.0f;

        for (int i = 0; i < M; i++){
            sum += s_kernel[i] * s_signal[ts_index - i];
        }

        output[gid] = sum;
    }
}

void design_bandpass_filter(float *h, int M, float fs, float f1, float f2){
    float fc1 = f1 / fs;
    float fc2 = f2 / fs;

    for(int n = 0; n < M; n++){
        int m = n - (M - 1) / 2;
        float w = 0.54f - 0.46f * cosf(2.0f * M_PI * n / (M - 1));
        if(m == 0){
            h[n] = 2 * (fc2 - fc1);
        } else {
            h[n] = (sinf(2 * M_PI * fc2 * m) - sinf(2 * M_PI * fc1 * m)) / (M_PI * m);
        }
        h[n] *= w;
    }

}

extern "C" void bandpass_filter(float *input_host, float* output_host, int length, float fs){
    float *d_signal, *d_kernel, *d_output;
    float h_kernel[FILTER_LEN];

    design_bandpass_filter(h_kernel, FILTER_LEN, fs, 7.0f, 12.0f);

    cudaMalloc(&d_signal, length * sizeof(float));
    cudaMalloc(&d_output, length * sizeof(float));

    cudaMalloc(&d_kernel, FILTER_LEN * sizeof(float));

    cudaMemcpy(d_signal, input_host, length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output_host, FILTER_LEN * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int sharedMemSize = (FILTER_LEN + blockSize + (FILTER_LEN - 1)) * sizeof(float);

    fir_filter<<<(length + blockSize - 1)/blockSize, blockSize, sharedMemSize>>>(d_signal, d_kernel, d_output, length, FILTER_LEN);

    cudaMemcpy(output_host, d_output, length * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_signal);
    cudaFree(d_output);
    cudaFree(d_kernel);

}