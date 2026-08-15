// =============================================================================
// afc.cu — АДАПТИВНЫЙ ПОИСК ЧАСТОТЫ (AFC) С ИСПРАВЛЕНИЯМИ
// =============================================================================
#include "afc.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define PI 3.14159265358979323846
#define PHI 1.6180339887498948482

// =============================================================================
// ЯДРО ДЛЯ ВЫЧИСЛЕНИЯ ЭНЕРГИИ НА ОДНОЙ ЧАСТОТЕ (ИСПРАВЛЕНО)
// =============================================================================
__global__ void afc_energy_kernel(
    const double* __restrict__ y_hi,
    const double* __restrict__ y_lo,
    int N,
    double freq,
    double* __restrict__ energy_hi,
    double* __restrict__ energy_lo
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ double sh[];
    double* sh_dot_s_hi  = sh;
    double* sh_dot_s_lo  = sh + num_threads;
    double* sh_norm_s_hi = sh + 2 * num_threads;
    double* sh_norm_s_lo = sh + 3 * num_threads;
    double* sh_dot_c_hi  = sh + 4 * num_threads;
    double* sh_dot_c_lo  = sh + 5 * num_threads;
    double* sh_norm_c_hi = sh + 6 * num_threads;
    double* sh_norm_c_lo = sh + 7 * num_threads;

    double dot_s_hi = 0.0, dot_s_lo = 0.0;
    double norm_s_hi = 0.0, norm_s_lo = 0.0;
    double dot_c_hi = 0.0, dot_c_lo = 0.0;
    double norm_c_hi = 0.0, norm_c_lo = 0.0;

    const double T = 30.0;

    for (int i = tid; i < N; i += num_threads) {
        double t = ((double)i / (N - 1)) * T;
        double phase = 2.0 * PI * freq * t;
        double s = sin(phase);
        double c = cos(phase);

        double yh = y_hi[i];
        double yl = y_lo[i];

        double prod_s_hi, prod_s_lo;
        alw_mul_dd(yh, yl, s, 0.0, prod_s_hi, prod_s_lo);
        alw_add_dd(dot_s_hi, dot_s_lo, prod_s_hi, prod_s_lo, dot_s_hi, dot_s_lo);

        double sq_s_hi, sq_s_lo;
        alw_mul_dd(s, 0.0, s, 0.0, sq_s_hi, sq_s_lo);
        alw_add_dd(norm_s_hi, norm_s_lo, sq_s_hi, sq_s_lo, norm_s_hi, norm_s_lo);

        double prod_c_hi, prod_c_lo;
        alw_mul_dd(yh, yl, c, 0.0, prod_c_hi, prod_c_lo);
        alw_add_dd(dot_c_hi, dot_c_lo, prod_c_hi, prod_c_lo, dot_c_hi, dot_c_lo);

        double sq_c_hi, sq_c_lo;
        alw_mul_dd(c, 0.0, c, 0.0, sq_c_hi, sq_c_lo);
        alw_add_dd(norm_c_hi, norm_c_lo, sq_c_hi, sq_c_lo, norm_c_hi, norm_c_lo);
    }

    sh_dot_s_hi[tid] = dot_s_hi;  sh_dot_s_lo[tid] = dot_s_lo;
    sh_norm_s_hi[tid] = norm_s_hi; sh_norm_s_lo[tid] = norm_s_lo;
    sh_dot_c_hi[tid] = dot_c_hi;  sh_dot_c_lo[tid] = dot_c_lo;
    sh_norm_c_hi[tid] = norm_c_hi; sh_norm_c_lo[tid] = norm_c_lo;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double r_hi, r_lo;
            alw_add_dd(sh_dot_s_hi[tid], sh_dot_s_lo[tid], sh_dot_s_hi[tid + s], sh_dot_s_lo[tid + s], r_hi, r_lo);
            sh_dot_s_hi[tid] = r_hi; sh_dot_s_lo[tid] = r_lo;

            alw_add_dd(sh_norm_s_hi[tid], sh_norm_s_lo[tid], sh_norm_s_hi[tid + s], sh_norm_s_lo[tid + s], r_hi, r_lo);
            sh_norm_s_hi[tid] = r_hi; sh_norm_s_lo[tid] = r_lo;

            alw_add_dd(sh_dot_c_hi[tid], sh_dot_c_lo[tid], sh_dot_c_hi[tid + s], sh_dot_c_lo[tid + s], r_hi, r_lo);
            sh_dot_c_hi[tid] = r_hi; sh_dot_c_lo[tid] = r_lo;

            alw_add_dd(sh_norm_c_hi[tid], sh_norm_c_lo[tid], sh_norm_c_hi[tid + s], sh_norm_c_lo[tid + s], r_hi, r_lo);
            sh_norm_c_hi[tid] = r_hi; sh_norm_c_lo[tid] = r_lo;
        }
        __syncthreads();
    }

    if (tid == 0) {
        DD ds = {sh_dot_s_hi[0], sh_dot_s_lo[0]};
        DD dc = {sh_dot_c_hi[0], sh_dot_c_lo[0]};
        DD ds2 = alw_mul_dd(ds, ds);
        DD dc2 = alw_mul_dd(dc, dc);
        DD sum = alw_add_dd(ds2, dc2);
        double norm_factor = (double)N * N / 4.0;
        DD nf = {norm_factor, 0.0};
        DD energy = alw_div_dd(sum, nf);
        energy_hi[0] = energy.hi;
        energy_lo[0] = energy.lo;
    }
}

// =============================================================================
// ЯДРО ГРУБОГО ПОИСКА (МНОГО ЧАСТОТ) – ИСПРАВЛЕНО
// =============================================================================
__global__ void afc_coarse_kernel(
    const double* __restrict__ y_hi,
    const double* __restrict__ y_lo,
    int N,
    const double* __restrict__ freqs,
    int num_freqs,
    double* __restrict__ energies_hi,
    double* __restrict__ energies_lo
) {
    int freq_idx = blockIdx.x;
    if (freq_idx >= num_freqs) return;

    double freq = freqs[freq_idx];
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ double sh[];
    double* sh_dot_s_hi  = sh;
    double* sh_dot_s_lo  = sh + num_threads;
    double* sh_norm_s_hi = sh + 2 * num_threads;
    double* sh_norm_s_lo = sh + 3 * num_threads;
    double* sh_dot_c_hi  = sh + 4 * num_threads;
    double* sh_dot_c_lo  = sh + 5 * num_threads;
    double* sh_norm_c_hi = sh + 6 * num_threads;
    double* sh_norm_c_lo = sh + 7 * num_threads;

    double dot_s_hi = 0.0, dot_s_lo = 0.0;
    double norm_s_hi = 0.0, norm_s_lo = 0.0;
    double dot_c_hi = 0.0, dot_c_lo = 0.0;
    double norm_c_hi = 0.0, norm_c_lo = 0.0;

    const double T = 30.0;

    for (int i = tid; i < N; i += num_threads) {
        double t = ((double)i / (N - 1)) * T;
        double phase = 2.0 * PI * freq * t;
        double s = sin(phase);
        double c = cos(phase);

        double yh = y_hi[i];
        double yl = y_lo[i];

        double prod_s_hi, prod_s_lo;
        alw_mul_dd(yh, yl, s, 0.0, prod_s_hi, prod_s_lo);
        alw_add_dd(dot_s_hi, dot_s_lo, prod_s_hi, prod_s_lo, dot_s_hi, dot_s_lo);

        double sq_s_hi, sq_s_lo;
        alw_mul_dd(s, 0.0, s, 0.0, sq_s_hi, sq_s_lo);
        alw_add_dd(norm_s_hi, norm_s_lo, sq_s_hi, sq_s_lo, norm_s_hi, norm_s_lo);

        double prod_c_hi, prod_c_lo;
        alw_mul_dd(yh, yl, c, 0.0, prod_c_hi, prod_c_lo);
        alw_add_dd(dot_c_hi, dot_c_lo, prod_c_hi, prod_c_lo, dot_c_hi, dot_c_lo);

        double sq_c_hi, sq_c_lo;
        alw_mul_dd(c, 0.0, c, 0.0, sq_c_hi, sq_c_lo);
        alw_add_dd(norm_c_hi, norm_c_lo, sq_c_hi, sq_c_lo, norm_c_hi, norm_c_lo);
    }

    sh_dot_s_hi[tid] = dot_s_hi;  sh_dot_s_lo[tid] = dot_s_lo;
    sh_norm_s_hi[tid] = norm_s_hi; sh_norm_s_lo[tid] = norm_s_lo;
    sh_dot_c_hi[tid] = dot_c_hi;  sh_dot_c_lo[tid] = dot_c_lo;
    sh_norm_c_hi[tid] = norm_c_hi; sh_norm_c_lo[tid] = norm_c_lo;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double r_hi, r_lo;
            alw_add_dd(sh_dot_s_hi[tid], sh_dot_s_lo[tid], sh_dot_s_hi[tid + s], sh_dot_s_lo[tid + s], r_hi, r_lo);
            sh_dot_s_hi[tid] = r_hi; sh_dot_s_lo[tid] = r_lo;

            alw_add_dd(sh_norm_s_hi[tid], sh_norm_s_lo[tid], sh_norm_s_hi[tid + s], sh_norm_s_lo[tid + s], r_hi, r_lo);
            sh_norm_s_hi[tid] = r_hi; sh_norm_s_lo[tid] = r_lo;

            alw_add_dd(sh_dot_c_hi[tid], sh_dot_c_lo[tid], sh_dot_c_hi[tid + s], sh_dot_c_lo[tid + s], r_hi, r_lo);
            sh_dot_c_hi[tid] = r_hi; sh_dot_c_lo[tid] = r_lo;

            alw_add_dd(sh_norm_c_hi[tid], sh_norm_c_lo[tid], sh_norm_c_hi[tid + s], sh_norm_c_lo[tid + s], r_hi, r_lo);
            sh_norm_c_hi[tid] = r_hi; sh_norm_c_lo[tid] = r_lo;
        }
        __syncthreads();
    }

    if (tid == 0) {
        DD ds = {sh_dot_s_hi[0], sh_dot_s_lo[0]};
        DD dc = {sh_dot_c_hi[0], sh_dot_c_lo[0]};
        DD ds2 = alw_mul_dd(ds, ds);
        DD dc2 = alw_mul_dd(dc, dc);
        DD sum = alw_add_dd(ds2, dc2);
        double norm_factor = (double)N * N / 4.0;
        DD nf = {norm_factor, 0.0};
        DD energy = alw_div_dd(sum, nf);
        energies_hi[freq_idx] = energy.hi;
        energies_lo[freq_idx] = energy.lo;
    }
}

// =============================================================================
// ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ВЫЗОВА ЯДРА
// =============================================================================
static inline double eval_energy_fast(
    const double* d_y_hi, const double* d_y_lo, int N, double freq,
    double* d_e_hi, double* d_e_lo, int threads
) {
    size_t shmem_size = 8 * threads * sizeof(double);
    afc_energy_kernel<<<1, threads, shmem_size>>>(d_y_hi, d_y_lo, N, freq, d_e_hi, d_e_lo);
    
    double e_hi, e_lo;
    CUDA_CHECK(cudaMemcpy(&e_hi, d_e_hi, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&e_lo, d_e_lo, sizeof(double), cudaMemcpyDeviceToHost));
    return e_hi + e_lo;
}

// =============================================================================
// ОСНОВНАЯ ФУНКЦИЯ ПОИСКА ЧАСТОТЫ (С УМЕНЬШЕННЫМ ЧИСЛОМ ПОТОКОВ)
// =============================================================================
double afc_find_frequency(
    const double* d_y_hi,
    const double* d_y_lo,
    int N,
    double f_prev,
    const AFC_Params& params,
    bool verbose,
    double* d_energy_out
) {
    if (N < 2 || params.coarse_points < 2) return 0.0;

    const int coarse = params.coarse_points;
    const int refine_iter = params.refine_iter;
    const double alpha = params.inertia_alpha;
    const double f_min = params.freq_min;
    const double f_max = params.freq_max;

    // 1. Грубый поиск
    std::vector<double> freqs_host(coarse);
    for (int i = 0; i < coarse; ++i) {
        freqs_host[i] = f_min + (f_max - f_min) * i / (coarse - 1);
    }

    double *d_freqs, *d_energies_hi, *d_energies_lo;
    CUDA_CHECK(cudaMalloc(&d_freqs, coarse * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energies_hi, coarse * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energies_lo, coarse * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_freqs, freqs_host.data(), coarse * sizeof(double), cudaMemcpyHostToDevice));

    // *** УМЕНЬШИЛИ ЧИСЛО ПОТОКОВ С 128 ДО 64 ***
    int threads = 64;
    size_t shmem_size = 8 * threads * sizeof(double);
    afc_coarse_kernel<<<coarse, threads, shmem_size>>>(
        d_y_hi, d_y_lo, N, d_freqs, coarse, d_energies_hi, d_energies_lo
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> energies_hi_host(coarse), energies_lo_host(coarse);
    CUDA_CHECK(cudaMemcpy(energies_hi_host.data(), d_energies_hi, coarse * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(energies_lo_host.data(), d_energies_lo, coarse * sizeof(double), cudaMemcpyDeviceToHost));

    int best_idx = 0;
    double best_energy = energies_hi_host[0] + energies_lo_host[0];
    for (int i = 1; i < coarse; ++i) {
        double e = energies_hi_host[i] + energies_lo_host[i];
        if (e > best_energy) {
            best_energy = e;
            best_idx = i;
        }
    }
    double f_coarse = freqs_host[best_idx];

    // 2. Уточнение золотым сечением
    double delta = (f_max - f_min) / (coarse - 1);
    double a = fmax(f_min, f_coarse - delta);
    double b = fmin(f_max, f_coarse + delta);

    double *d_ref_hi, *d_ref_lo;
    CUDA_CHECK(cudaMalloc(&d_ref_hi, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ref_lo, sizeof(double)));

    for (int iter = 0; iter < refine_iter; ++iter) {
        double c = b - (b - a) / PHI;
        double d = a + (b - a) / PHI;

        double Ec = eval_energy_fast(d_y_hi, d_y_lo, N, c, d_ref_hi, d_ref_lo, threads);
        double Ed = eval_energy_fast(d_y_hi, d_y_lo, N, d, d_ref_hi, d_ref_lo, threads);

        if (Ec < Ed) {
            a = c;
        } else {
            b = d;
        }
    }
    double f_refined = (a + b) / 2.0;

    if (d_energy_out) {
        eval_energy_fast(d_y_hi, d_y_lo, N, f_refined, d_ref_hi, d_ref_lo, threads);
        CUDA_CHECK(cudaMemcpy(d_energy_out, d_ref_hi, sizeof(double), cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_ref_hi));
    CUDA_CHECK(cudaFree(d_ref_lo));

    // 3. Инерция
    double f_final = (f_prev > 0.0) ? (alpha * f_refined + (1.0 - alpha) * f_prev) : f_refined;
    f_final = fmax(f_min, fmin(f_max, f_final));

    CUDA_CHECK(cudaFree(d_freqs));
    CUDA_CHECK(cudaFree(d_energies_hi));
    CUDA_CHECK(cudaFree(d_energies_lo));

    return f_final;
}
