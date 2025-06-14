#ifndef BPF_FILTER_H
#define BPF_FILTER_H

#ifdef __cplusplus
extern "C" {
#endif

void bandpass_filter_eeg(float* input_host, float* output_host, int length, float fs);

#ifdef __cplusplus
}
#endif

#endif 