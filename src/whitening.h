#ifndef WHITENING_H
#define WHITENING_H

#include "alw_math.h"
#include <cufft.h>

#ifdef __cplusplus
extern "C" {
#endif

bool estimate_psd(const double* d_signal, int N,
                  int segment_len, int overlap,
                  int fft_size, double* d_psd,
                  cudaStream_t stream = 0);

void design_whitening_filter(const double* d_psd, int fft_size,
                             int filter_len, double epsilon,
                             double* d_filter, cudaStream_t stream = 0);

void apply_whitening_filter(const double* d_signal, int N,
                            const double* d_filter, int filter_len,
                            double* d_output, cudaStream_t stream = 0);

void whiten_signal(double* d_signal, int N,
                   int segment_len = 256,
                   int overlap = 128,
                   int fft_size = 512,
                   int filter_len = 64,
                   double epsilon = 1e-12,
                   cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif

#endif // WHITENING_H
