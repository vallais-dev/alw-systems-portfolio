// aeds_kernel.cu — базовые ядра для AEDS (публичная версия, без адаптивного обновления)
#include "aeds_solver.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cstdio>

// ============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ============================================================================

__device__ __forceinline__ DD dd_sqrt(DD a) {
    if (a.hi <= 0.0) return DD{0.0, 0.0};
    double x = sqrt(a.hi);
    DD r = DD{x, 0.0};
    DD x2 = dd_mul(r, r);
    DD diff = dd_sub(a, x2);
    DD correction = dd_div(diff, dd_mul(r, DD{2.0, 0.0}));
    return dd_add(r, correction);
}

__device__ __forceinline__ DD matvec_dot_product_dd(
    const DD* __restrict__ d_A_cmaj,
    const DD* __restrict__ d_x,
    const int n, const int row)
{
    DD accum = {0.0, 0.0};
    for (int col = 0; col < n; ++col) {
        DD a = d_A_cmaj[col * n + row];
        DD xv = d_x[col];
        accum = alw_add_dd(accum, alw_mul_dd(a, xv));
    }
    return accum;
}

__device__ __forceinline__ double matvec_dot_product_double(
    const double* __restrict__ d_A_cmaj,
    const double* __restrict__ d_x,
    const int n, const int row)
{
    double accum = 0.0;
    for (int col = 0; col < n; ++col) {
        accum += d_A_cmaj[col * n + row] * d_x[col];
    }
    return accum;
}

// ============================================================================
// ЯДРА ДЛЯ DOUBLE
// ============================================================================

__global__ void compute_row_scales_kernel(
    const double* __restrict__ d_A_in,
    double* __restrict__ d_row_scales,
    const int m, const int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    double diag = (row < n) ? d_A_in[row * n + row] : 0.0;
    if (fabs(diag) < 1e-28) {
        double sign = (diag >= 0.0) ? 1.0 : -1.0;
        diag += sign * 1e-28;
    }
    d_row_scales[row] = diag;
}

__global__ void scale_b_vector_kernel(
    const double* __restrict__ d_b_in,
    double* __restrict__ d_b_out,
    const double* __restrict__ d_row_scales,
    const int m)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    d_b_out[row] = d_b_in[row] / d_row_scales[row];
}

__global__ void transpose_and_scale_kernel(
    const double* __restrict__ d_A_in,
    double* __restrict__ d_A_cmaj,
    double* __restrict__ d_L,
    const double* __restrict__ d_row_scales,
    const int m, const int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    double local_sum = 0.0;
    double* A_col = d_A_cmaj + col * m;
    for (int row = tid; row < m; row += num_threads) {
        double val = d_A_in[row * n + col];
        double scale = d_row_scales[row];
        double scaled = val / scale;
        A_col[row] = scaled;
        local_sum += scaled * scaled;
    }
    extern __shared__ double sh_transpose_double[];
    sh_transpose_double[tid] = local_sum;
    __syncthreads();
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) sh_transpose_double[tid] += sh_transpose_double[tid + s];
        __syncthreads();
    }
    if (tid == 0) d_L[col] = sh_transpose_double[0];
}

__global__ void matvec_kernel(
    const double* __restrict__ d_A_cmaj,
    const double* __restrict__ d_x,
    const double* __restrict__ d_b,
    double* __restrict__ d_r,
    const int m, const int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    double acc = 0.0;
    for (int col = 0; col < n; ++col) {
        acc += d_A_cmaj[col * m + row] * d_x[col];
    }
    d_r[row] = acc - d_b[row];
}

__global__ void compute_residual_norm_kernel_double(
    const double* __restrict__ d_r,
    double* __restrict__ d_norm,
    const int m)
{
    extern __shared__ double sh_norm_double[];
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    double local_sum = 0.0;
    for (int i = tid; i < m; i += num_threads) {
        double val = d_r[i];
        local_sum += val * val;
    }
    sh_norm_double[tid] = local_sum;
    __syncthreads();
    
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) sh_norm_double[tid] += sh_norm_double[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) {
        d_norm[0] = sqrt(sh_norm_double[0]);
    }
}

// ============================================================================
// ЯДРА ДЛЯ DD (с 64 потоками на блок)
// ============================================================================

__global__ void compute_row_scales_kernel_dd(
    const DD* __restrict__ d_A_in,
    DD* __restrict__ d_row_scales,
    const int m, const int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    DD diag = (row < n) ? d_A_in[row * n + row] : DD{0.0, 0.0};
    if (alw_is_critically_small(alw_abs_dd(diag), 1e-28)) {
        double sign = (diag.hi >= 0.0) ? 1.0 : -1.0;
        diag = alw_add_dd(diag, DD{sign * 1e-28, 0.0});
    }
    d_row_scales[row] = diag;
}

__global__ void scale_b_vector_kernel_dd(
    const DD* __restrict__ d_b_in,
    DD* __restrict__ d_b_out,
    const DD* __restrict__ d_row_scales,
    const int m)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    d_b_out[row] = alw_div_dd(d_b_in[row], d_row_scales[row]);
}

__global__ void transpose_and_scale_kernel_dd(
    const DD* __restrict__ d_A_in,
    DD* __restrict__ d_A_cmaj,
    DD* __restrict__ d_L,
    const DD* __restrict__ d_row_scales,
    const int m, const int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    DD local_sum = {0.0, 0.0};
    DD* A_col = d_A_cmaj + col * m;

    for (int row = tid; row < m; row += num_threads) {
        DD val = d_A_in[row * n + col];
        DD scale = d_row_scales[row];
        DD scaled = alw_div_dd(val, scale);
        A_col[row] = scaled;
        DD sq = alw_mul_dd(scaled, scaled);
        local_sum = alw_add_dd(local_sum, sq);
    }

    extern __shared__ DD sh_transpose_dd[];
    sh_transpose_dd[tid] = local_sum;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_transpose_dd[tid] = alw_add_dd(sh_transpose_dd[tid], sh_transpose_dd[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_L[col] = sh_transpose_dd[0];
    }
}

__global__ void matvec_kernel_dd(
    const DD* __restrict__ d_A_cmaj,
    const DD* __restrict__ d_x,
    const DD* __restrict__ d_b,
    DD* __restrict__ d_r,
    const int m, const int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    DD acc = {0.0, 0.0};
    for (int col = 0; col < n; ++col) {
        DD a = d_A_cmaj[col * m + row];
        DD xv = d_x[col];
        acc = alw_add_dd(acc, alw_mul_dd(a, xv));
    }
    d_r[row] = alw_sub_dd(acc, d_b[row]);
}

__global__ void compute_residual_norm_kernel_dd(
    const DD* __restrict__ d_r,
    DD* __restrict__ d_norm,
    const int m)
{
    extern __shared__ DD sh_norm_dd[];
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    DD local_sum = {0.0, 0.0};
    for (int i = tid; i < m; i += num_threads) {
        DD val = d_r[i];
        DD sq = alw_mul_dd(val, val);
        local_sum = alw_add_dd(local_sum, sq);
    }
    sh_norm_dd[tid] = local_sum;
    __syncthreads();
    
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_norm_dd[tid] = alw_add_dd(sh_norm_dd[tid], sh_norm_dd[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        d_norm[0] = dd_sqrt(sh_norm_dd[0]);
    }
}