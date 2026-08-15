// =============================================================================
// amad_x.cu — МНОГОМАСШТАБНАЯ ОЦЕНКА ШУМА AMAD-X (ДЕМО-ВЕРСИЯ)
//
// ВНИМАНИЕ: Адаптивная логика (amad_x_estimate_adaptive) в этой версии
//           удалена и заменена на заглушку. Полная версия содержит
//           автоматический подбор параметров под сигнал.
// =============================================================================
#include "amad_x.h"
#include "alw_math.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#ifndef MIN_EPSILON
#define MIN_EPSILON 1e-30
#endif

// =============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (device)
// =============================================================================

__device__ inline bool dd_less(DD a, DD b) {
    return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
}

__device__ inline void dd_sort_descending(DD* arr, int n) {
    for (int i = 1; i < n; ++i) {
        DD key = arr[i];
        int j = i - 1;
        while (j >= 0 && dd_less(arr[j], key)) {
            arr[j+1] = arr[j];
            --j;
        }
        arr[j+1] = key;
    }
}

// ----------------------------------------------------------------------------
// БЛОЧНОЕ ВЫЧИСЛЕНИЕ IQR -> СИГМА
// ----------------------------------------------------------------------------
__global__ void amad_blocks_kernel(
    const double* __restrict__ d_res_hi,
    const double* __restrict__ d_res_lo,
    int N,
    int block_size,
    int overlap,
    double* __restrict__ d_sigma_hi,
    double* __restrict__ d_sigma_lo,
    double* __restrict__ d_weights_hi,
    double* __restrict__ d_weights_lo
) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int block_stride = block_size - overlap;
    int start = block_idx * block_stride;
    int end = min(start + block_size, N);
    int len = end - start;
    if (len < 4) {
        if (tid == 0) {
            d_sigma_hi[block_idx] = MIN_EPSILON;
            d_sigma_lo[block_idx] = 0.0;
            d_weights_hi[block_idx] = 1.0;
            d_weights_lo[block_idx] = 0.0;
        }
        return;
    }

    extern __shared__ double sh[];
    DD* s_data = (DD*)sh;

    for (int i = tid; i < len; i += num_threads) {
        int idx = start + i;
        s_data[i].hi = d_res_hi[idx];
        s_data[i].lo = d_res_lo[idx];
        if (isnan(s_data[i].hi) || isinf(s_data[i].hi)) {
            s_data[i].hi = 0.0;
            s_data[i].lo = 0.0;
        }
    }
    __syncthreads();

    if (tid == 0) {
        dd_sort_descending(s_data, len);
    }
    __syncthreads();

    if (tid == 0) {
        int idx_q1 = (int)(0.75 * len);
        int idx_q3 = (int)(0.25 * len);
        if (idx_q1 < 0) idx_q1 = 0;
        if (idx_q3 >= len) idx_q3 = len - 1;
        if (idx_q1 >= len) idx_q1 = len - 1;
        if (idx_q3 < 0) idx_q3 = 0;

        DD q1 = s_data[idx_q1];
        DD q3 = s_data[idx_q3];

        DD iqr = alw_sub_dd(q3, q1);
        if (iqr.hi < 0.0) {
            iqr.hi = -iqr.hi;
            iqr.lo = -iqr.lo;
        }

        DD sigma = alw_div_dd(iqr, {1.349, 0.0});
        if (sigma.hi < MIN_EPSILON || isnan(sigma.hi) || isinf(sigma.hi)) {
            sigma.hi = MIN_EPSILON;
            sigma.lo = 0.0;
        }

        d_sigma_hi[block_idx] = sigma.hi;
        d_sigma_lo[block_idx] = sigma.lo;
        d_weights_hi[block_idx] = 1.0;
        d_weights_lo[block_idx] = 0.0;
    }
}

// ----------------------------------------------------------------------------
// ВЫЧИСЛЕНИЕ ВЕСОВ (ОБРАТНАЯ ДИСПЕРСИЯ)
// ----------------------------------------------------------------------------
__global__ void amad_weights_kernel(
    const double* __restrict__ d_sigma_hi,
    const double* __restrict__ d_sigma_lo,
    int num_blocks,
    double* __restrict__ d_weights_hi,
    double* __restrict__ d_weights_lo
) {
    int tid = threadIdx.x;
    if (tid >= num_blocks) return;

    DD sigma = {d_sigma_hi[tid], d_sigma_lo[tid]};
    if (isnan(sigma.hi) || isinf(sigma.hi)) sigma.hi = MIN_EPSILON;

    DD sigma_sq = alw_mul_dd(sigma, sigma);
    DD denom = alw_add_dd(sigma_sq, {MIN_EPSILON, 0.0});
    DD w = alw_div_dd({1.0, 0.0}, denom);

    if (isnan(w.hi) || isinf(w.hi) || w.hi < 0.0) {
        w.hi = 1.0;
        w.lo = 0.0;
    }

    d_weights_hi[tid] = w.hi;
    d_weights_lo[tid] = w.lo;
}

// ----------------------------------------------------------------------------
// ИТЕРАТИВНОЕ УТОЧНЕНИЕ (ОТСЕЧЕНИЕ ВЫБРОСОВ)
// ----------------------------------------------------------------------------
__global__ void amad_refine_kernel(
    const double* __restrict__ d_res_hi,
    const double* __restrict__ d_res_lo,
    int N,
    int block_size,
    int overlap,
    double* __restrict__ d_sigma_hi,
    double* __restrict__ d_sigma_lo,
    double* __restrict__ d_weights_hi,
    double* __restrict__ d_weights_lo,
    int refine_iter
) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int block_stride = block_size - overlap;
    int start = block_idx * block_stride;
    int end = min(start + block_size, N);
    int len = end - start;
    if (len < 4) return;

    extern __shared__ double sh[];
    DD* s_data = (DD*)sh;

    for (int i = tid; i < len; i += num_threads) {
        int idx = start + i;
        s_data[i].hi = d_res_hi[idx];
        s_data[i].lo = d_res_lo[idx];
        if (isnan(s_data[i].hi) || isinf(s_data[i].hi)) {
            s_data[i].hi = 0.0;
            s_data[i].lo = 0.0;
        }
    }
    __syncthreads();

    DD sigma = {d_sigma_hi[block_idx], d_sigma_lo[block_idx]};
    if (isnan(sigma.hi) || isinf(sigma.hi) || sigma.hi < MIN_EPSILON) {
        sigma.hi = MIN_EPSILON;
        sigma.lo = 0.0;
    }

    for (int it = 0; it < refine_iter; ++it) {
        if (tid == 0) {
            dd_sort_descending(s_data, len);

            DD median = s_data[len/2];

            int new_len = 0;
            for (int i = 0; i < len; ++i) {
                DD diff = alw_sub_dd(s_data[i], median);
                double abs_diff = fabs(diff.hi + diff.lo);
                double threshold = 3.0 * (sigma.hi + sigma.lo);
                if (abs_diff < threshold) {
                    s_data[new_len] = s_data[i];
                    new_len++;
                }
            }
            len = new_len;
            if (len < 4) {
                return;
            }
            dd_sort_descending(s_data, len);

            int idx_q1 = (int)(0.75 * len);
            int idx_q3 = (int)(0.25 * len);
            if (idx_q1 < 0) idx_q1 = 0;
            if (idx_q3 >= len) idx_q3 = len - 1;
            if (idx_q1 >= len) idx_q1 = len - 1;
            if (idx_q3 < 0) idx_q3 = 0;

            DD q1 = s_data[idx_q1];
            DD q3 = s_data[idx_q3];

            DD iqr = alw_sub_dd(q3, q1);
            if (iqr.hi < 0.0) {
                iqr.hi = -iqr.hi;
                iqr.lo = -iqr.lo;
            }

            DD new_sigma = alw_div_dd(iqr, {1.349, 0.0});
            if (new_sigma.hi > MIN_EPSILON && !isnan(new_sigma.hi) && !isinf(new_sigma.hi)) {
                sigma = new_sigma;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_sigma_hi[block_idx] = sigma.hi;
        d_sigma_lo[block_idx] = sigma.lo;
    }
}

// ----------------------------------------------------------------------------
// ВЗВЕШЕННОЕ УСРЕДНЕНИЕ ПО ВСЕМ БЛОКАМ (МНОГОМАСШТАБНОСТЬ)
// ----------------------------------------------------------------------------
__global__ void amad_multiscale_kernel(
    const double* __restrict__ d_sigma_hi,
    const double* __restrict__ d_sigma_lo,
    const double* __restrict__ d_weights_hi,
    const double* __restrict__ d_weights_lo,
    int num_blocks,
    double* __restrict__ d_final_sigma_hi,
    double* __restrict__ d_final_sigma_lo
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ double sh[];
    double* sh_sum_hi = sh;
    double* sh_sum_lo = sh + blockDim.x;
    double* sh_wsum_hi = sh + 2 * blockDim.x;
    double* sh_wsum_lo = sh + 3 * blockDim.x;

    DD sum = {0.0, 0.0};
    DD wsum = {0.0, 0.0};

    for (int i = tid; i < num_blocks; i += num_threads) {
        DD sigma = {d_sigma_hi[i], d_sigma_lo[i]};
        DD w = {d_weights_hi[i], d_weights_lo[i]};
        if (isnan(sigma.hi) || isinf(sigma.hi)) sigma.hi = MIN_EPSILON;
        if (isnan(w.hi) || isinf(w.hi) || w.hi < 0.0) w.hi = 0.0;

        DD prod = alw_mul_dd(sigma, w);
        sum = alw_add_dd(sum, prod);
        wsum = alw_add_dd(wsum, w);
    }

    sh_sum_hi[tid] = sum.hi;
    sh_sum_lo[tid] = sum.lo;
    sh_wsum_hi[tid] = wsum.hi;
    sh_wsum_lo[tid] = wsum.lo;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            alw_add_dd(sh_sum_hi[tid], sh_sum_lo[tid],
                       sh_sum_hi[tid+s], sh_sum_lo[tid+s],
                       sh_sum_hi[tid], sh_sum_lo[tid]);
            alw_add_dd(sh_wsum_hi[tid], sh_wsum_lo[tid],
                       sh_wsum_hi[tid+s], sh_wsum_lo[tid+s],
                       sh_wsum_hi[tid], sh_wsum_lo[tid]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        double wsum_hi = sh_wsum_hi[0];
        double wsum_lo = sh_wsum_lo[0];
        if (wsum_hi < MIN_EPSILON || isnan(wsum_hi) || isinf(wsum_hi)) {
            wsum_hi = 1.0;
            wsum_lo = 0.0;
        }
        DD wsum_dd = {wsum_hi, wsum_lo};
        DD final_sigma = alw_div_dd({sh_sum_hi[0], sh_sum_lo[0]}, wsum_dd);
        if (isnan(final_sigma.hi) || isinf(final_sigma.hi) || final_sigma.hi < MIN_EPSILON) {
            final_sigma.hi = MIN_EPSILON;
            final_sigma.lo = 0.0;
        }
        d_final_sigma_hi[0] = final_sigma.hi;
        d_final_sigma_lo[0] = final_sigma.lo;
    }
}

// =============================================================================
// ОСНОВНАЯ ФУНКЦИЯ AMAD-X (ПАРАМЕТРИЗОВАННАЯ)
// =============================================================================
double amad_x_estimate(
    const double* d_res_hi,
    const double* d_res_lo,
    int N,
    int min_block_size,
    int max_block_size,
    int num_scales,
    int refine_iter,
    bool verbose
) {
    if (N < 4) {
        if (verbose) AEDS_LOGW("AMAD-X: N < 4, returning MIN_EPSILON");
        return MIN_EPSILON;
    }

    // Внутренние сетки параметров (могут быть переопределены через аргументы)
    int block_sizes[3] = {8, 32, 128};
    int overlaps[3] = {4, 16, 64};
    if (num_scales > 3) num_scales = 3;
    if (num_scales < 1) num_scales = 1;
    for (int s = 0; s < num_scales; ++s) {
        if (block_sizes[s] > N) block_sizes[s] = N;
        if (overlaps[s] >= block_sizes[s]) overlaps[s] = block_sizes[s] / 2;
    }

    if (min_block_size > 0 && num_scales > 0) {
        block_sizes[0] = min_block_size;
        overlaps[0] = min_block_size / 2;
    }
    if (max_block_size > 0 && num_scales > 1) {
        block_sizes[num_scales-1] = max_block_size;
        overlaps[num_scales-1] = max_block_size / 2;
    }

    DD final_sigma = {MIN_EPSILON, 0.0};
    DD total_weight = {0.0, 0.0};

    for (int s = 0; s < num_scales; ++s) {
        int block_size = block_sizes[s];
        int overlap = overlaps[s];
        int block_stride = block_size - overlap;
        if (block_stride <= 0) block_stride = block_size / 2;
        int num_blocks = (N - block_size + block_stride) / block_stride;
        if (num_blocks < 1) num_blocks = 1;

        if (verbose) {
            AEDS_LOGD("AMAD-X: scale %d block_size=%d overlap=%d num_blocks=%d",
                      s, block_size, overlap, num_blocks);
        }

        CudaPoolGuard<double> d_sigma_hi(num_blocks);
        CudaPoolGuard<double> d_sigma_lo(num_blocks);
        CudaPoolGuard<double> d_weights_hi(num_blocks);
        CudaPoolGuard<double> d_weights_lo(num_blocks);

        if (!d_sigma_hi.is_valid() || !d_sigma_lo.is_valid() ||
            !d_weights_hi.is_valid() || !d_weights_lo.is_valid()) {
            AEDS_LOGE("AMAD-X: memory allocation failed for scale %d", s);
            return MIN_EPSILON;
        }

        int threads = 256;
        int blocks = num_blocks;
        size_t shmem = threads * sizeof(DD);
        amad_blocks_kernel<<<blocks, threads, shmem>>>(
            d_res_hi, d_res_lo, N, block_size, overlap,
            d_sigma_hi.get(), d_sigma_lo.get(),
            d_weights_hi.get(), d_weights_lo.get()
        );
        cudaDeviceSynchronize();

        amad_weights_kernel<<<(num_blocks + 255) / 256, 256>>>(
            d_sigma_hi.get(), d_sigma_lo.get(), num_blocks,
            d_weights_hi.get(), d_weights_lo.get()
        );
        cudaDeviceSynchronize();

        if (refine_iter > 0) {
            amad_refine_kernel<<<blocks, threads, shmem>>>(
                d_res_hi, d_res_lo, N, block_size, overlap,
                d_sigma_hi.get(), d_sigma_lo.get(),
                d_weights_hi.get(), d_weights_lo.get(),
                refine_iter
            );
            cudaDeviceSynchronize();
            amad_weights_kernel<<<(num_blocks + 255) / 256, 256>>>(
                d_sigma_hi.get(), d_sigma_lo.get(), num_blocks,
                d_weights_hi.get(), d_weights_lo.get()
            );
            cudaDeviceSynchronize();
        }

        CudaPoolGuard<double> d_scale_sigma_hi(1);
        CudaPoolGuard<double> d_scale_sigma_lo(1);

        if (!d_scale_sigma_hi.is_valid() || !d_scale_sigma_lo.is_valid()) {
            AEDS_LOGE("AMAD-X: memory allocation for scale sigma failed");
            return MIN_EPSILON;
        }

        size_t shmem_mult = 4 * threads * sizeof(double);
        amad_multiscale_kernel<<<1, threads, shmem_mult>>>(
            d_sigma_hi.get(), d_sigma_lo.get(),
            d_weights_hi.get(), d_weights_lo.get(),
            num_blocks,
            d_scale_sigma_hi.get(), d_scale_sigma_lo.get()
        );
        cudaDeviceSynchronize();

        double scale_sigma_hi, scale_sigma_lo;
        cudaMemcpy(&scale_sigma_hi, d_scale_sigma_hi.get(), sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&scale_sigma_lo, d_scale_sigma_lo.get(), sizeof(double), cudaMemcpyDeviceToHost);
        DD scale_sigma = {scale_sigma_hi, scale_sigma_lo};

        if (isnan(scale_sigma.hi) || isinf(scale_sigma.hi) || scale_sigma.hi < MIN_EPSILON) {
            scale_sigma.hi = MIN_EPSILON;
            scale_sigma.lo = 0.0;
        }

        DD weight = alw_div_dd({1.0, 0.0}, alw_add_dd(scale_sigma, {MIN_EPSILON, 0.0}));
        if (isnan(weight.hi) || isinf(weight.hi)) weight.hi = 1.0;

        DD prod = alw_mul_dd(scale_sigma, weight);
        final_sigma = alw_add_dd(final_sigma, prod);
        total_weight = alw_add_dd(total_weight, weight);

        if (verbose) {
            AEDS_LOGD("AMAD-X: scale %d sigma=%.6f weight=%.6f", s, scale_sigma.hi, weight.hi);
        }
    }

    if (total_weight.hi > 0.0) {
        final_sigma = alw_div_dd(final_sigma, total_weight);
    } else {
        final_sigma.hi = MIN_EPSILON;
        final_sigma.lo = 0.0;
    }
    if (isnan(final_sigma.hi) || isinf(final_sigma.hi) || final_sigma.hi < MIN_EPSILON) {
        final_sigma.hi = MIN_EPSILON;
        final_sigma.lo = 0.0;
    }

    if (verbose) {
        AEDS_LOGI("AMAD-X: final sigma = %.6f", final_sigma.hi);
    }

    return final_sigma.hi;
}

// =============================================================================
// АДАПТИВНАЯ ОБЁРТКА (ДЕМО-ВЕРСИЯ — ЛОГИКА УДАЛЕНА)
// =============================================================================
//
// ВНИМАНИЕ: Эта функция является заглушкой для демонстрационных целей.
//           Полная версия содержит автоматический подбор параметров:
//           - размеров блоков (min_block_size, max_block_size)
//           - числа масштабов (num_scales)
//           - числа итераций уточнения (refine_iter)
//           - адаптивную коррекцию на основе статистик сигнала
//
//           В коммерческой версии эти параметры подбираются динамически
//           в зависимости от N, разброса сигнала и доли выбросов.
//           Для получения полной версии обратитесь к автору.
//
double amad_x_estimate_adaptive(
    const double* d_res_hi,
    const double* d_res_lo,
    int N,
    bool verbose
) {
    // Заглушка: просто вызываем базовую оценку с фиксированными параметрами
    // (в реальной версии здесь была бы адаптивная логика)
    if (verbose) {
        AEDS_LOGW("AMAD-X Adaptive: using demo stub (adaptive logic removed)");
    }

    // Параметры по умолчанию (подходят для многих сигналов)
    int min_block_size = 8;
    int max_block_size = 64;
    int num_scales = 3;
    int refine_iter = 2;

    return amad_x_estimate(
        d_res_hi, d_res_lo,
        N,
        min_block_size,
        max_block_size,
        num_scales,
        refine_iter,
        verbose
    );
}