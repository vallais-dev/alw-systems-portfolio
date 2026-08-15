// =============================================================================
// orthogonalization.cu — ГИБРИДНАЯ ОРТОГОНАЛИЗАЦИЯ (ДЕМО-ВЕРСИЯ)
// =============================================================================
//
// ВНИМАНИЕ: Данный файл содержит только CUDA-ядра и вспомогательные функции
//           для демонстрации технического уровня. Полная реализация
//           гибридной ортогонализации (MGS + Householder + SVD)
//           с автоматическим выбором режима удалена.
//
//           Оставлены:
//           - Все ядра (оценка cond, MGS, Householder, SVD, пересчёт R)
//           - Вспомогательные функции (арена, сохранение/загрузка базиса)
//           - Функция оценки ошибки ортогональности
//
//           Основные хост-функции orthogonalize_basis_hybrid и
//           orthogonalize_basis_batch заменены заглушками.
//
//           Для получения полной версии обратитесь к автору.
// =============================================================================

#include "alw_math.h"
#include "orthogonalization.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <mutex>

// =============================================================================
// ГЛОБАЛЬНЫЙ БУФЕР ДЛЯ ARENA (PINNED MEMORY)
// =============================================================================
static double* g_arena_hi = nullptr;
static double* g_arena_lo = nullptr;
static size_t g_arena_capacity = 0;
static std::mutex g_arena_mutex;

void init_orthogonal_arena(size_t max_size) {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    if (g_arena_hi != nullptr) return;
    cudaError_t err;
    err = cudaHostAlloc(&g_arena_hi, max_size * sizeof(double), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        ALW_LOG_ERROR("init_orthogonal_arena: cudaHostAlloc hi failed: %s", cudaGetErrorString(err));
        return;
    }
    err = cudaHostAlloc(&g_arena_lo, max_size * sizeof(double), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        ALW_LOG_ERROR("init_orthogonal_arena: cudaHostAlloc lo failed: %s", cudaGetErrorString(err));
        cudaFreeHost(g_arena_hi);
        g_arena_hi = nullptr;
        return;
    }
    g_arena_capacity = max_size;
    ALW_LOG_INFO("Orthogonal arena allocated: %zu doubles (%.2f MB)", max_size, max_size * sizeof(double) / (1024.0 * 1024.0));
}

static void free_orthogonal_arena() {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    if (g_arena_hi) { cudaFreeHost(g_arena_hi); g_arena_hi = nullptr; }
    if (g_arena_lo) { cudaFreeHost(g_arena_lo); g_arena_lo = nullptr; }
    g_arena_capacity = 0;
}

__attribute__((destructor)) static void cleanup_arena() {
    free_orthogonal_arena();
}

// =============================================================================
// ВСПОМОГАТЕЛЬНЫЕ СТРУКТУРЫ И ФУНКЦИИ
// =============================================================================

struct AlwXorshift {
    uint32_t state;
    AlwXorshift(uint32_t seed) : state(seed) {}
    double next_double() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (double)state / (double)0xFFFFFFFFu;
    }
};

struct OrthoStats {
    double time_reg = 0.0;
    double time_cond_est = 0.0;
    double time_mgs = 0.0;
    double time_householder = 0.0;
    double time_svd = 0.0;
    double time_recompute_R = 0.0;
    int householder_iterations = 0;
    bool svd_fallback = false;
    double final_error = 0.0;
    int num_basis = 0;
    int N = 0;
    int effective_mode = -1;
    int batch_size = 1;
    int num_mgs_frames = 0;
    int num_hybrid_frames = 0;
};

static void print_matrix_stats(const char* label, const double* d_mat_hi, const double* d_mat_lo,
                               int rows, int cols, bool verbose) {
    if (!verbose) return;
    alw_vector<double> h_mat_hi(rows * cols);
    cudaError_t err = cudaMemcpy(h_mat_hi.data(), d_mat_hi, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        ALW_LOG_ERROR("%s: cudaMemcpy failed: %s", label, cudaGetErrorString(err));
        return;
    }
    double max_val = 0.0;
    bool has_nan = false, has_inf = false;
    for (int i = 0; i < rows * cols; ++i) {
        double v = h_mat_hi[i];
        if (std::isnan(v)) has_nan = true;
        if (std::isinf(v)) has_inf = true;
        if (std::fabs(v) > max_val) max_val = std::fabs(v);
    }
    ALW_LOG_INFO("%s: max_abs=%f, NaN=%s, Inf=%s", label, max_val, has_nan ? "yes" : "no", has_inf ? "yes" : "no");
}

static void save_stats_to_file(const OrthoStats& stats) {
    std::fstream f("ortho_stats.json", std::ios::out | std::ios::app);
    if (!f.is_open()) {
        ALW_LOG_WARN("Cannot open ortho_stats.json for writing");
        return;
    }
    f << "{\n";
    f << "  \"num_basis\": " << stats.num_basis << ",\n";
    f << "  \"N\": " << stats.N << ",\n";
    f << "  \"batch_size\": " << stats.batch_size << ",\n";
    f << "  \"effective_mode\": " << stats.effective_mode << ",\n";
    f << "  \"final_error\": " << std::scientific << stats.final_error << ",\n";
    f << "  \"time_reg_ms\": " << stats.time_reg << ",\n";
    f << "  \"time_cond_est_ms\": " << stats.time_cond_est << ",\n";
    f << "  \"time_mgs_ms\": " << stats.time_mgs << ",\n";
    f << "  \"time_householder_ms\": " << stats.time_householder << ",\n";
    f << "  \"time_svd_ms\": " << stats.time_svd << ",\n";
    f << "  \"time_recompute_R_ms\": " << stats.time_recompute_R << ",\n";
    f << "  \"householder_iterations\": " << stats.householder_iterations << ",\n";
    f << "  \"svd_fallback\": " << (stats.svd_fallback ? "true" : "false") << ",\n";
    f << "  \"num_mgs_frames\": " << stats.num_mgs_frames << ",\n";
    f << "  \"num_hybrid_frames\": " << stats.num_hybrid_frames << "\n";
    f << "}\n";
    f.close();
}

// =============================================================================
// ЯДРА: ОЦЕНКА ЧИСЛА ОБУСЛОВЛЕННОСТИ
// =============================================================================

#define BATCH_COND_THREADS 256

__global__ void estimate_condition_batch_kernel(
    const double* __restrict__ A_hi,
    int num_basis, int N, int batch_size,
    double* __restrict__ cond_out)
{
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    const double* A = A_hi + batch_idx * num_basis * N;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    extern __shared__ double s_data[];
    double* s_max = s_data;
    double* s_min = s_data + blockDim.x;
    double my_max = 0.0;
    double my_min = 1e300;
    for (int col = tid; col < num_basis; col += num_threads) {
        double sum_sq = 0.0;
        for (int i = 0; i < N; ++i) {
            double val = A[col * N + i];
            if (isnan(val) || isinf(val)) val = 0.0;
            sum_sq += val * val;
        }
        double norm_hi, norm_lo;
        alw_sqrt_dd(sum_sq, 0.0, norm_hi, norm_lo);
        double norm = norm_hi;
        if (isnan(norm) || isinf(norm)) norm = 0.0;
        if (norm > my_max) my_max = norm;
        if (norm < my_min && norm > 1e-30) my_min = norm;
    }
    s_max[tid] = my_max;
    s_min[tid] = my_min;
    __syncthreads();
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
            if (s_min[tid + s] < s_min[tid] && s_min[tid + s] > 1e-30) s_min[tid] = s_min[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        double cond = (s_min[0] > 1e-30) ? s_max[0] / s_min[0] : 1e15;
        if (isnan(cond) || isinf(cond)) cond = 1e15;
        cond_out[batch_idx] = cond;
    }
}

__global__ void matvec_kernel(const double* __restrict__ A,
                              const double* __restrict__ x,
                              double* __restrict__ y,
                              int num_basis, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double sum = 0.0;
    for (int j = 0; j < num_basis; ++j) {
        double a = A[j * N + i];
        double xj = x[j];
        if (isnan(a) || isinf(a)) a = 0.0;
        if (isnan(xj) || isinf(xj)) xj = 0.0;
        sum += a * xj;
    }
    if (isnan(sum) || isinf(sum)) sum = 0.0;
    y[i] = sum;
}

__global__ void matvec_transpose_kernel(const double* __restrict__ A,
                                        const double* __restrict__ y,
                                        double* __restrict__ x,
                                        int num_basis, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_basis) return;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        double a = A[j * N + i];
        double yi = y[i];
        if (isnan(a) || isinf(a)) a = 0.0;
        if (isnan(yi) || isinf(yi)) yi = 0.0;
        sum += a * yi;
    }
    if (isnan(sum) || isinf(sum)) sum = 0.0;
    x[j] = sum;
}

__global__ void normalize_kernel(double* __restrict__ x, int n) {
    extern __shared__ double s_norm[];
    int tid = threadIdx.x, num_threads = blockDim.x;
    double sum_hi = 0.0, sum_lo = 0.0;
    for (int i = tid; i < n; i += num_threads) {
        double val = x[i];
        if (isnan(val) || isinf(val)) val = 0.0;
        double p_hi, p_lo;
        alw_mul_dd(val, 0.0, val, 0.0, p_hi, p_lo);
        alw_add_dd(sum_hi, sum_lo, p_hi, p_lo, sum_hi, sum_lo);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        double th_hi = __shfl_down_sync(0xffffffff, sum_hi, offset);
        double th_lo = __shfl_down_sync(0xffffffff, sum_lo, offset);
        alw_add_dd(sum_hi, sum_lo, th_hi, th_lo, sum_hi, sum_lo);
    }
    __shared__ double s_hi[32], s_lo[32];
    int lane = tid & 31;
    int wid = tid / 32;
    if (lane == 0) { s_hi[wid] = sum_hi; s_lo[wid] = sum_lo; }
    __syncthreads();
    int num_warps = (num_threads + 31) / 32;
    if (tid < num_warps) { sum_hi = s_hi[tid]; sum_lo = s_lo[tid]; }
    else { sum_hi = 0.0; sum_lo = 0.0; }
    if (tid < 32) {
        for (int offset = 16; offset > 0; offset /= 2) {
            double th_hi = __shfl_down_sync(0xffffffff, sum_hi, offset);
            double th_lo = __shfl_down_sync(0xffffffff, sum_lo, offset);
            alw_add_dd(sum_hi, sum_lo, th_hi, th_lo, sum_hi, sum_lo);
        }
    }
    __syncthreads();
    if (tid == 0) {
        double norm_hi, norm_lo;
        alw_sqrt_dd(sum_hi, sum_lo, norm_hi, norm_lo);
        if (norm_hi < 1e-30) {
            for (int i = tid; i < n; i += num_threads) x[i] = (i == 0) ? 1.0 : 0.0;
        } else {
            double inv_hi, inv_lo;
            alw_div_dd(1.0, 0.0, norm_hi, norm_lo, inv_hi, inv_lo);
            for (int i = tid; i < n; i += num_threads) {
                double val = x[i];
                double new_hi, new_lo;
                alw_mul_dd(val, 0.0, inv_hi, inv_lo, new_hi, new_lo);
                x[i] = new_hi;
            }
        }
    }
    __syncthreads();
}

__global__ void ata_matvec_kernel(const double* __restrict__ ATA,
                                  const double* __restrict__ x,
                                  double* __restrict__ y,
                                  int num_basis) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_basis) return;
    double sum = 0.0;
    for (int j = 0; j < num_basis; ++j) {
        double a = ATA[i * num_basis + j];
        double xj = x[j];
        if (isnan(a) || isinf(a)) a = 0.0;
        if (isnan(xj) || isinf(xj)) xj = 0.0;
        sum += a * xj;
    }
    if (isnan(sum) || isinf(sum)) sum = 0.0;
    y[i] = sum;
}

__global__ void solve_regularized_kernel(const double* __restrict__ ATA,
                                         const double* __restrict__ x,
                                         double* __restrict__ z,
                                         int num_basis, double lambda, int gd_steps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_basis) return;
    double zi = 0.0;
    for (int step = 0; step < gd_steps; ++step) {
        double Azi = lambda * zi;
        for (int j = 0; j < num_basis; ++j) {
            double a = ATA[i * num_basis + j];
            if (isnan(a) || isinf(a)) a = 0.0;
            Azi += a * zi;
        }
        double resid = Azi - x[i];
        if (isnan(resid) || isinf(resid)) resid = 0.0;
        zi -= 0.01 * resid;
    }
    if (isnan(zi) || isinf(zi)) zi = 0.0;
    z[i] = zi;
}

// =============================================================================
// ЯДРО: ПЕРЕСЧЁТ R = Q^T * A_orig
// =============================================================================

#define R_COMPUTE_THREADS 256

__global__ void recompute_R_gpu_kernel(
    const double* __restrict__ Q_hi, const double* __restrict__ Q_lo,
    const double* __restrict__ A_hi, const double* __restrict__ A_lo,
    int num_basis, int N, double* __restrict__ R_hi, double* __restrict__ R_lo) {
    int i = blockIdx.x, j = blockIdx.y;
    if (i >= num_basis || j >= num_basis) return;
    int tid = threadIdx.x;
    extern __shared__ double s_sum[];
    double my_sum = 0.0;
    for (int k = tid; k < N; k += blockDim.x) {
        double q = Q_hi[i * N + k];
        double a = A_hi[j * N + k];
        if (isnan(q) || isinf(q)) q = 0.0;
        if (isnan(a) || isinf(a)) a = 0.0;
        my_sum += q * a;
    }
    if (isnan(my_sum) || isinf(my_sum)) my_sum = 0.0;
    s_sum[tid] = my_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        double val = s_sum[0];
        if (isnan(val) || isinf(val)) val = 0.0;
        R_hi[i * num_basis + j] = val;
        R_lo[i * num_basis + j] = 0.0;
    }
}

__global__ void recompute_R_gpu_kernel_batch(
    const double* __restrict__ Q_hi, const double* __restrict__ Q_lo,
    const double* __restrict__ A_hi, const double* __restrict__ A_lo,
    int num_basis, int N, double* __restrict__ R_hi, double* __restrict__ R_lo,
    int batch_size) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;
    int i = blockIdx.x, j = blockIdx.y;
    if (i >= num_basis || j >= num_basis) return;
    int frame_offset_Q = batch_idx * num_basis * N;
    int frame_offset_A = batch_idx * num_basis * N;
    int frame_offset_R = batch_idx * num_basis * num_basis;
    const double* Q_ptr = Q_hi + frame_offset_Q;
    const double* A_ptr = A_hi + frame_offset_A;
    double* R_ptr_hi = R_hi + frame_offset_R;
    double* R_ptr_lo = R_lo + frame_offset_R;
    int tid = threadIdx.x;
    extern __shared__ double s_sum[];
    double my_sum = 0.0;
    for (int k = tid; k < N; k += blockDim.x) {
        double q = Q_ptr[i * N + k];
        double a = A_ptr[j * N + k];
        if (isnan(q) || isinf(q)) q = 0.0;
        if (isnan(a) || isinf(a)) a = 0.0;
        my_sum += q * a;
    }
    if (isnan(my_sum) || isinf(my_sum)) my_sum = 0.0;
    s_sum[tid] = my_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        double val = s_sum[0];
        if (isnan(val) || isinf(val)) val = 0.0;
        R_ptr_hi[i * num_basis + j] = val;
        R_ptr_lo[i * num_basis + j] = 0.0;
    }
}

// =============================================================================
// ЯДРО: РЕГУЛЯРИЗАЦИЯ ИСХОДНОГО БАЗИСА
// =============================================================================

__global__ void regularize_basis_kernel(
    double* __restrict__ A_hi, double* __restrict__ A_lo,
    int num_basis, int N, double reg_strength) {
    int col = blockIdx.x, tid = threadIdx.x, num_threads = blockDim.x;
    if (col >= num_basis) return;
    double base_add = reg_strength * (col + 1);
    for (int i = tid; i < N; i += num_threads) {
        double val = A_hi[col * N + i] + base_add * (i + 1) * 1e-6;
        if (isnan(val) || isinf(val)) val = 0.0;
        A_hi[col * N + i] = val;
    }
}

// =============================================================================
// ЯДРО: MGS С ДВУКРАТНОЙ ОРТОГОНАЛИЗАЦИЕЙ (CGS2)
// =============================================================================

__global__ void alw_mgs_fused_kernel(
    double* __restrict__ basis_hi,
    double* __restrict__ basis_lo,
    double* __restrict__ R_hi,
    double* __restrict__ R_lo,
    int num_basis,
    int N)
{
    extern __shared__ double s_red[];
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    for (int k = 0; k < num_basis; ++k) {
        for (int pass = 0; pass < 2; ++pass) {
            for (int j = 0; j < k; ++j) {
                double proj_hi = 0.0, proj_lo = 0.0;
                for (int i = tid; i < N; i += num_threads) {
                    double qj_h = basis_hi[j * N + i];
                    double qj_l = basis_lo[j * N + i];
                    double qk_h = basis_hi[k * N + i];
                    double qk_l = basis_lo[k * N + i];
                    double prod_h, prod_l;
                    alw_mul_dd(qj_h, qj_l, qk_h, qk_l, prod_h, prod_l);
                    alw_add_dd(proj_hi, proj_lo, prod_h, prod_l, proj_hi, proj_lo);
                }
                s_red[tid] = proj_hi;
                s_red[tid + num_threads] = proj_lo;
                __syncthreads();
                for (int s = num_threads / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        alw_add_dd(s_red[tid], s_red[tid + num_threads],
                                   s_red[tid + s], s_red[tid + s + num_threads],
                                   s_red[tid], s_red[tid + num_threads]);
                    }
                    __syncthreads();
                }
                double dot_hi = s_red[0];
                double dot_lo = s_red[num_threads];
                if (pass == 0 && tid == 0) {
                    R_hi[j * num_basis + k] = dot_hi;
                    R_lo[j * num_basis + k] = dot_lo;
                }
                __syncthreads();
                for (int i = tid; i < N; i += num_threads) {
                    double qj_h = basis_hi[j * N + i];
                    double qj_l = basis_lo[j * N + i];
                    double qk_h = basis_hi[k * N + i];
                    double qk_l = basis_lo[k * N + i];
                    double sub_h, sub_l;
                    alw_mul_dd(dot_hi, dot_lo, qj_h, qj_l, sub_h, sub_l);
                    alw_sub_dd(qk_h, qk_l, sub_h, sub_l, qk_h, qk_l);
                    basis_hi[k * N + i] = qk_h;
                    basis_lo[k * N + i] = qk_l;
                }
                __syncthreads();
            }
        }
        double norm_sq_hi = 0.0, norm_sq_lo = 0.0;
        for (int i = tid; i < N; i += num_threads) {
            double qk_h = basis_hi[k * N + i];
            double qk_l = basis_lo[k * N + i];
            double sq_h, sq_l;
            alw_mul_dd(qk_h, qk_l, qk_h, qk_l, sq_h, sq_l);
            alw_add_dd(norm_sq_hi, norm_sq_lo, sq_h, sq_l, norm_sq_hi, norm_sq_lo);
        }
        s_red[tid] = norm_sq_hi;
        s_red[tid + num_threads] = norm_sq_lo;
        __syncthreads();
        for (int s = num_threads / 2; s > 0; s >>= 1) {
            if (tid < s) {
                alw_add_dd(s_red[tid], s_red[tid + num_threads],
                           s_red[tid + s], s_red[tid + s + num_threads],
                           s_red[tid], s_red[tid + num_threads]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            double norm = sqrt(s_red[0]);
            if (norm < 1e-30) norm = 1.0;
            R_hi[k * num_basis + k] = norm;
            R_lo[k * num_basis + k] = 0.0;
            s_red[0] = 1.0 / norm;
        }
        __syncthreads();
        double inv_norm = s_red[0];
        for (int i = tid; i < N; i += num_threads) {
            double qk_h = basis_hi[k * N + i];
            double qk_l = basis_lo[k * N + i];
            alw_mul_dd(qk_h, qk_l, inv_norm, 0.0, basis_hi[k * N + i], basis_lo[k * N + i]);
        }
        __syncthreads();
    }
}

__global__ void alw_mgs_fused_kernel_batch_indexed(
    double* __restrict__ basis_hi,
    double* __restrict__ basis_lo,
    double* __restrict__ R_hi,
    double* __restrict__ R_lo,
    int num_basis,
    int N,
    const int* __restrict__ indices,
    int count)
{
    int batch_idx = blockIdx.x;
    if (batch_idx >= count) return;
    int frame_idx = indices[batch_idx];
    int frame_offset = frame_idx * num_basis * N;
    int R_offset = frame_idx * num_basis * num_basis;
    double* A_hi_frame = basis_hi + frame_offset;
    double* A_lo_frame = basis_lo + frame_offset;
    double* R_hi_frame = R_hi + R_offset;
    double* R_lo_frame = R_lo + R_offset;
    extern __shared__ double s_red[];
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    for (int k = 0; k < num_basis; ++k) {
        for (int pass = 0; pass < 2; ++pass) {
            for (int j = 0; j < k; ++j) {
                double proj_hi = 0.0, proj_lo = 0.0;
                for (int i = tid; i < N; i += num_threads) {
                    double qj_h = A_hi_frame[j * N + i];
                    double qj_l = A_lo_frame[j * N + i];
                    double qk_h = A_hi_frame[k * N + i];
                    double qk_l = A_lo_frame[k * N + i];
                    double prod_h, prod_l;
                    alw_mul_dd(qj_h, qj_l, qk_h, qk_l, prod_h, prod_l);
                    alw_add_dd(proj_hi, proj_lo, prod_h, prod_l, proj_hi, proj_lo);
                }
                s_red[tid] = proj_hi;
                s_red[tid + num_threads] = proj_lo;
                __syncthreads();
                for (int s = num_threads / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        alw_add_dd(s_red[tid], s_red[tid + num_threads],
                                   s_red[tid + s], s_red[tid + s + num_threads],
                                   s_red[tid], s_red[tid + num_threads]);
                    }
                    __syncthreads();
                }
                double dot_hi = s_red[0];
                double dot_lo = s_red[num_threads];
                if (pass == 0 && tid == 0) {
                    R_hi_frame[j * num_basis + k] = dot_hi;
                    R_lo_frame[j * num_basis + k] = dot_lo;
                }
                __syncthreads();
                for (int i = tid; i < N; i += num_threads) {
                    double qj_h = A_hi_frame[j * N + i];
                    double qj_l = A_lo_frame[j * N + i];
                    double qk_h = A_hi_frame[k * N + i];
                    double qk_l = A_lo_frame[k * N + i];
                    double sub_h, sub_l;
                    alw_mul_dd(dot_hi, dot_lo, qj_h, qj_l, sub_h, sub_l);
                    alw_sub_dd(qk_h, qk_l, sub_h, sub_l, qk_h, qk_l);
                    A_hi_frame[k * N + i] = qk_h;
                    A_lo_frame[k * N + i] = qk_l;
                }
                __syncthreads();
            }
        }
        double norm_sq_hi = 0.0, norm_sq_lo = 0.0;
        for (int i = tid; i < N; i += num_threads) {
            double qk_h = A_hi_frame[k * N + i];
            double qk_l = A_lo_frame[k * N + i];
            double sq_h, sq_l;
            alw_mul_dd(qk_h, qk_l, qk_h, qk_l, sq_h, sq_l);
            alw_add_dd(norm_sq_hi, norm_sq_lo, sq_h, sq_l, norm_sq_hi, norm_sq_lo);
        }
        s_red[tid] = norm_sq_hi;
        s_red[tid + num_threads] = norm_sq_lo;
        __syncthreads();
        for (int s = num_threads / 2; s > 0; s >>= 1) {
            if (tid < s) {
                alw_add_dd(s_red[tid], s_red[tid + num_threads],
                           s_red[tid + s], s_red[tid + s + num_threads],
                           s_red[tid], s_red[tid + num_threads]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            double norm = sqrt(s_red[0]);
            if (norm < 1e-30) norm = 1.0;
            R_hi_frame[k * num_basis + k] = norm;
            R_lo_frame[k * num_basis + k] = 0.0;
            s_red[0] = 1.0 / norm;
        }
        __syncthreads();
        double inv_norm = s_red[0];
        for (int i = tid; i < N; i += num_threads) {
            double qk_h = A_hi_frame[k * N + i];
            double qk_l = A_lo_frame[k * N + i];
            alw_mul_dd(qk_h, qk_l, inv_norm, 0.0, A_hi_frame[k * N + i], A_lo_frame[k * N + i]);
        }
        __syncthreads();
    }
}

// =============================================================================
// ЯДРО: БЛОЧНЫЙ ХАУСХОЛДЕР
// =============================================================================

#define HH_CHUNK_SIZE 2048

__global__ void householder_block_kernel_chunked(
    double* A_hi, double* A_lo,
    double* beta_hi, double* beta_lo,
    int num_basis, int N, int block_size) {
    int col_start = blockIdx.x * block_size;
    if (col_start >= num_basis) return;
    int block_end = min(col_start + block_size, num_basis);
    int actual_block_size = block_end - col_start;
    extern __shared__ double sh_red[];
    double* s_hi = sh_red;
    double* s_lo = sh_red + blockDim.x;
    int tid = threadIdx.x, num_threads = blockDim.x;
    for (int idx = 0; idx < actual_block_size; ++idx) {
        int col = col_start + idx;
        double my_norm_hi = 0.0, my_norm_lo = 0.0;
        for (int k = tid; k < N - col; k += num_threads) {
            int row = col + k;
            double a_hi = A_hi[col * N + row];
            double a_lo = A_lo[col * N + row];
            if (isnan(a_hi) || isinf(a_hi)) a_hi = 0.0;
            if (isnan(a_lo) || isinf(a_lo)) a_lo = 0.0;
            double p_hi, p_lo;
            alw_mul_dd(a_hi, a_lo, a_hi, a_lo, p_hi, p_lo);
            alw_add_dd(my_norm_hi, my_norm_lo, p_hi, p_lo, my_norm_hi, my_norm_lo);
        }
        s_hi[tid] = my_norm_hi; s_lo[tid] = my_norm_lo;
        __syncthreads();
        for (int s = num_threads / 2; s > 0; s >>= 1) {
            if (tid < s) { s_hi[tid] += s_hi[tid + s]; s_lo[tid] += s_lo[tid + s]; }
            __syncthreads();
        }
        double norm_hi = s_hi[0], norm_lo = s_lo[0];
        if (isnan(norm_hi) || isinf(norm_hi) || norm_hi < 0.0) norm_hi = 0.0;
        if (tid == 0) {
            double rhi, rlo;
            alw_sqrt_dd(norm_hi, norm_lo, rhi, rlo);
            if (isnan(rhi) || isinf(rhi) || rhi < 1e-30) {
                beta_hi[col] = 0.0;
                beta_lo[col] = 0.0;
            } else {
                double x1_hi = A_hi[col * N + col];
                if (isnan(x1_hi) || isinf(x1_hi)) x1_hi = 0.0;
                double sign = (x1_hi >= 0) ? 1.0 : -1.0;
                double s_hi_val, s_lo_val;
                alw_mul_dd(sign, 0.0, rhi, rlo, s_hi_val, s_lo_val);
                A_hi[col * N + col] = x1_hi + s_hi_val;
                A_lo[col * N + col] = A_lo[col * N + col] + s_lo_val;
                if (isnan(A_hi[col * N + col]) || isinf(A_hi[col * N + col])) {
                    A_hi[col * N + col] = 1.0;
                    A_lo[col * N + col] = 0.0;
                }
                double vv_hi = 0.0, vv_lo = 0.0;
                for (int k = 0; k < N - col; ++k) {
                    int row = col + k;
                    double a_hi_ = A_hi[col * N + row];
                    double a_lo_ = A_lo[col * N + row];
                    if (isnan(a_hi_) || isinf(a_hi_)) a_hi_ = 0.0;
                    if (isnan(a_lo_) || isinf(a_lo_)) a_lo_ = 0.0;
                    double p_hi, p_lo;
                    alw_mul_dd(a_hi_, a_lo_, a_hi_, a_lo_, p_hi, p_lo);
                    alw_add_dd(vv_hi, vv_lo, p_hi, p_lo, vv_hi, vv_lo);
                }
                if (isnan(vv_hi) || isinf(vv_hi) || fabs(vv_hi) < 1e-30) {
                    beta_hi[col] = 0.0;
                    beta_lo[col] = 0.0;
                } else {
                    alw_div_dd(2.0, 0.0, vv_hi, vv_lo, beta_hi[col], beta_lo[col]);
                    if (isnan(beta_hi[col]) || isinf(beta_hi[col])) {
                        beta_hi[col] = 0.0;
                        beta_lo[col] = 0.0;
                    }
                }
            }
            s_hi[0] = beta_hi[col];
            s_lo[0] = beta_lo[col];
        }
        __syncthreads();
        double b_hi = s_hi[0], b_lo = s_lo[0];
        for (int j = idx; j < actual_block_size; ++j) {
            int col_j = col_start + j;
            double alpha_hi = 0.0, alpha_lo = 0.0;
            for (int chunk_start = 0; chunk_start < N - col_start; chunk_start += HH_CHUNK_SIZE) {
                int chunk_end = min(chunk_start + HH_CHUNK_SIZE, N - col_start);
                double my_chunk_hi = 0.0, my_chunk_lo = 0.0;
                for (int k = chunk_start + tid; k < chunk_end; k += num_threads) {
                    int row = col_start + k;
                    double a1_hi = A_hi[col * N + row];
                    double a1_lo = A_lo[col * N + row];
                    double a2_hi = A_hi[col_j * N + row];
                    double a2_lo = A_lo[col_j * N + row];
                    if (isnan(a1_hi) || isinf(a1_hi)) a1_hi = 0.0;
                    if (isnan(a1_lo) || isinf(a1_lo)) a1_lo = 0.0;
                    if (isnan(a2_hi) || isinf(a2_hi)) a2_hi = 0.0;
                    if (isnan(a2_lo) || isinf(a2_lo)) a2_lo = 0.0;
                    double p_hi, p_lo;
                    alw_mul_dd(a1_hi, a1_lo, a2_hi, a2_lo, p_hi, p_lo);
                    alw_add_dd(my_chunk_hi, my_chunk_lo, p_hi, p_lo, my_chunk_hi, my_chunk_lo);
                }
                s_hi[tid] = my_chunk_hi; s_lo[tid] = my_chunk_lo;
                __syncthreads();
                for (int s = num_threads / 2; s > 0; s >>= 1) {
                    if (tid < s) { s_hi[tid] += s_hi[tid + s]; s_lo[tid] += s_lo[tid + s]; }
                    __syncthreads();
                }
                if (tid == 0) {
                    alw_add_dd(alpha_hi, alpha_lo, s_hi[0], s_lo[0], alpha_hi, alpha_lo);
                }
                __syncthreads();
            }
            if (tid == 0) {
                alw_mul_dd(b_hi, b_lo, alpha_hi, alpha_lo, alpha_hi, alpha_lo);
                if (isnan(alpha_hi) || isinf(alpha_hi)) { alpha_hi = 0.0; alpha_lo = 0.0; }
                s_hi[0] = alpha_hi; s_lo[0] = alpha_lo;
            }
            __syncthreads();
            double fa_hi = s_hi[0], fa_lo = s_lo[0];
            for (int chunk_start = 0; chunk_start < N - col_start; chunk_start += HH_CHUNK_SIZE) {
                int chunk_end = min(chunk_start + HH_CHUNK_SIZE, N - col_start);
                for (int k = chunk_start + tid; k < chunk_end; k += num_threads) {
                    int row = col_start + k;
                    double a1_hi = A_hi[col * N + row];
                    double a1_lo = A_lo[col * N + row];
                    if (isnan(a1_hi) || isinf(a1_hi)) a1_hi = 0.0;
                    if (isnan(a1_lo) || isinf(a1_lo)) a1_lo = 0.0;
                    double prod_hi, prod_lo;
                    alw_mul_dd(a1_hi, a1_lo, fa_hi, fa_lo, prod_hi, prod_lo);
                    double new_hi, new_lo;
                    alw_sub_dd(A_hi[col_j * N + row], A_lo[col_j * N + row],
                               prod_hi, prod_lo, new_hi, new_lo);
                    if (isnan(new_hi) || isinf(new_hi)) new_hi = 0.0;
                    if (isnan(new_lo) || isinf(new_lo)) new_lo = 0.0;
                    A_hi[col_j * N + row] = new_hi;
                    A_lo[col_j * N + row] = new_lo;
                }
                __syncthreads();
            }
        }
    }
}

// =============================================================================
// ЯДРО: SVD ЯКОБИ (ОДНОСТОРОННИЕ ВРАЩЕНИЯ)
// =============================================================================

__global__ void svd_jacobi_sweep_kernel(
    double* A_hi, double* A_lo,
    double* V_hi, double* V_lo,
    int m, int n,
    int sweep,
    double tolerance)
{
    extern __shared__ double sh[];
    double* col_i = sh;
    double* col_j = col_i + m;
    double* row_i = col_j + m;
    double* row_j = row_i + n;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            __syncthreads();
            for (int idx = tid; idx < m; idx += num_threads) {
                col_i[idx] = A_hi[i * m + idx];
                col_j[idx] = A_hi[j * m + idx];
            }
            __syncthreads();
            double app_hi = 0.0, app_lo = 0.0;
            double aqq_hi = 0.0, aqq_lo = 0.0;
            double apq_hi = 0.0, apq_lo = 0.0;
            for (int idx = tid; idx < m; idx += num_threads) {
                double ai = col_i[idx];
                double aj = col_j[idx];
                double p_hi, p_lo;
                alw_mul_dd(ai, 0.0, ai, 0.0, p_hi, p_lo);
                alw_add_dd(app_hi, app_lo, p_hi, p_lo, app_hi, app_lo);
                alw_mul_dd(aj, 0.0, aj, 0.0, p_hi, p_lo);
                alw_add_dd(aqq_hi, aqq_lo, p_hi, p_lo, aqq_hi, aqq_lo);
                alw_mul_dd(ai, 0.0, aj, 0.0, p_hi, p_lo);
                alw_add_dd(apq_hi, apq_lo, p_hi, p_lo, apq_hi, apq_lo);
            }
            for (int offset = 16; offset > 0; offset /= 2) {
                double th_hi = __shfl_down_sync(0xffffffff, app_hi, offset);
                double th_lo = __shfl_down_sync(0xffffffff, app_lo, offset);
                alw_add_dd(app_hi, app_lo, th_hi, th_lo, app_hi, app_lo);
                th_hi = __shfl_down_sync(0xffffffff, aqq_hi, offset);
                th_lo = __shfl_down_sync(0xffffffff, aqq_lo, offset);
                alw_add_dd(aqq_hi, aqq_lo, th_hi, th_lo, aqq_hi, aqq_lo);
                th_hi = __shfl_down_sync(0xffffffff, apq_hi, offset);
                th_lo = __shfl_down_sync(0xffffffff, apq_lo, offset);
                alw_add_dd(apq_hi, apq_lo, th_hi, th_lo, apq_hi, apq_lo);
            }
            if (tid == 0) {
                row_i[0] = app_hi; row_i[1] = app_lo;
                row_i[2] = aqq_hi; row_i[3] = aqq_lo;
                row_i[4] = apq_hi; row_i[5] = apq_lo;
            }
            __syncthreads();
            double app_hi_global = row_i[0];
            double app_lo_global = row_i[1];
            double aqq_hi_global = row_i[2];
            double aqq_lo_global = row_i[3];
            double apq_hi_global = row_i[4];
            double apq_lo_global = row_i[5];
            double abs_apq = alw_abs_hi(apq_hi_global, apq_lo_global);
            if (abs_apq < 1e-12) continue;
            double diff_hi, diff_lo;
            alw_sub_dd(aqq_hi_global, aqq_lo_global, app_hi_global, app_lo_global, diff_hi, diff_lo);
            double two_apq_hi, two_apq_lo;
            alw_mul_dd(2.0, 0.0, apq_hi_global, apq_lo_global, two_apq_hi, two_apq_lo);
            double tau_hi, tau_lo;
            alw_div_dd(diff_hi, diff_lo, two_apq_hi, two_apq_lo, tau_hi, tau_lo);
            double sign_tau = (tau_hi >= 0) ? 1.0 : -1.0;
            double tau_abs_hi = fabs(tau_hi);
            double one_plus_tau2_hi, one_plus_tau2_lo;
            alw_mul_dd(tau_hi, tau_lo, tau_hi, tau_lo, one_plus_tau2_hi, one_plus_tau2_lo);
            alw_add_dd(1.0, 0.0, one_plus_tau2_hi, one_plus_tau2_lo, one_plus_tau2_hi, one_plus_tau2_lo);
            double sqrt_val_hi, sqrt_val_lo;
            alw_sqrt_dd(one_plus_tau2_hi, one_plus_tau2_lo, sqrt_val_hi, sqrt_val_lo);
            double denom_hi, denom_lo;
            alw_add_dd(tau_abs_hi, 0.0, sqrt_val_hi, sqrt_val_lo, denom_hi, denom_lo);
            double t_hi, t_lo;
            alw_div_dd(sign_tau, 0.0, denom_hi, denom_lo, t_hi, t_lo);
            double t2_hi, t2_lo;
            alw_mul_dd(t_hi, t_lo, t_hi, t_lo, t2_hi, t2_lo);
            double one_plus_t2_hi, one_plus_t2_lo;
            alw_add_dd(1.0, 0.0, t2_hi, t2_lo, one_plus_t2_hi, one_plus_t2_lo);
            double sqrt_one_plus_t2_hi, sqrt_one_plus_t2_lo;
            alw_sqrt_dd(one_plus_t2_hi, one_plus_t2_lo, sqrt_one_plus_t2_hi, sqrt_one_plus_t2_lo);
            double c_hi, c_lo;
            alw_div_dd(1.0, 0.0, sqrt_one_plus_t2_hi, sqrt_one_plus_t2_lo, c_hi, c_lo);
            double s_hi, s_lo;
            alw_mul_dd(c_hi, c_lo, t_hi, t_lo, s_hi, s_lo);
            for (int idx = tid; idx < m; idx += num_threads) {
                double ai = col_i[idx];
                double aj = col_j[idx];
                double c_ai_hi, c_ai_lo;
                alw_mul_dd(c_hi, c_lo, ai, 0.0, c_ai_hi, c_ai_lo);
                double s_aj_hi, s_aj_lo;
                alw_mul_dd(s_hi, s_lo, aj, 0.0, s_aj_hi, s_aj_lo);
                double new_ai_hi, new_ai_lo;
                alw_sub_dd(c_ai_hi, c_ai_lo, s_aj_hi, s_aj_lo, new_ai_hi, new_ai_lo);
                double s_ai_hi, s_ai_lo;
                alw_mul_dd(s_hi, s_lo, ai, 0.0, s_ai_hi, s_ai_lo);
                double c_aj_hi, c_aj_lo;
                alw_mul_dd(c_hi, c_lo, aj, 0.0, c_aj_hi, c_aj_lo);
                double new_aj_hi, new_aj_lo;
                alw_add_dd(s_ai_hi, s_ai_lo, c_aj_hi, c_aj_lo, new_aj_hi, new_aj_lo);
                col_i[idx] = new_ai_hi;
                col_j[idx] = new_aj_hi;
            }
            __syncthreads();
            for (int idx = tid; idx < m; idx += num_threads) {
                A_hi[i * m + idx] = col_i[idx];
                A_hi[j * m + idx] = col_j[idx];
            }
            if (tid < n) {
                row_i[tid] = V_hi[i * n + tid];
                row_j[tid] = V_hi[j * n + tid];
            }
            __syncthreads();
            for (int idx = tid; idx < n; idx += num_threads) {
                double vi = row_i[idx];
                double vj = row_j[idx];
                double c_vi_hi, c_vi_lo;
                alw_mul_dd(c_hi, c_lo, vi, 0.0, c_vi_hi, c_vi_lo);
                double s_vj_hi, s_vj_lo;
                alw_mul_dd(s_hi, s_lo, vj, 0.0, s_vj_hi, s_vj_lo);
                double new_vi_hi, new_vi_lo;
                alw_sub_dd(c_vi_hi, c_vi_lo, s_vj_hi, s_vj_lo, new_vi_hi, new_vi_lo);
                double s_vi_hi, s_vi_lo;
                alw_mul_dd(s_hi, s_lo, vi, 0.0, s_vi_hi, s_vi_lo);
                double c_vj_hi, c_vj_lo;
                alw_mul_dd(c_hi, c_lo, vj, 0.0, c_vj_hi, c_vj_lo);
                double new_vj_hi, new_vj_lo;
                alw_add_dd(s_vi_hi, s_vi_lo, c_vj_hi, c_vj_lo, new_vj_hi, new_vj_lo);
                row_i[idx] = new_vi_hi;
                row_j[idx] = new_vj_hi;
            }
            __syncthreads();
            if (tid < n) {
                V_hi[i * n + tid] = row_i[tid];
                V_hi[j * n + tid] = row_j[tid];
            }
            __syncthreads();
        }
    }
}

// =============================================================================
// ЯДРО: ОЦЕНКА ОШИБКИ ОРТОГОНАЛЬНОСТИ
// =============================================================================

__global__ void estimate_orthogonality_error_kernel(const double* Q_hi, const double* Q_lo,
                                                    int num_basis, int N, double* max_error) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= num_basis || j >= num_basis) return;
    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
        double a = Q_hi[i * N + k];
        double b = Q_hi[j * N + k];
        if (isnan(a) || isinf(a)) a = 0.0;
        if (isnan(b) || isinf(b)) b = 0.0;
        dot += a * b;
    }
    if (isnan(dot) || isinf(dot) || fabs(dot) > 1e12) dot = 1e10;
    double err = (i == j) ? fabs(dot - 1.0) : fabs(dot);
    atomicMax((unsigned long long*)max_error, __double_as_longlong(err));
}

// =============================================================================
// ГЛОБАЛЬНАЯ ФУНКЦИЯ ОЦЕНКИ ОШИБКИ ОРТОГОНАЛЬНОСТИ (ЭКСПОРТИРУЕМАЯ)
// =============================================================================

double estimate_orthogonality_error(const double* d_Q_hi, const double* d_Q_lo,
                                    int num_basis, int N, bool verbose)
{
    double *d_max_err = nullptr;
    cudaError_t err = cudaMalloc(&d_max_err, sizeof(double));
    if (err != cudaSuccess) {
        ALW_LOG_ERROR("estimate_orthogonality_error cudaMalloc failed: %s", cudaGetErrorString(err));
        return 1e10;
    }
    err = cudaMemset(d_max_err, 0, sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_max_err);
        ALW_LOG_ERROR("estimate_orthogonality_error cudaMemset failed: %s", cudaGetErrorString(err));
        return 1e10;
    }
    dim3 threads(16, 16), blocks((num_basis + 15) / 16, (num_basis + 15) / 16);
    estimate_orthogonality_error_kernel<<<blocks, threads>>>(d_Q_hi, d_Q_lo, num_basis, N, d_max_err);
    err = cudaDeviceSynchronize();
    double max_err = 0.0;
    if (err == cudaSuccess) {
        err = cudaMemcpy(&max_err, d_max_err, sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) max_err = 1e10;
    } else {
        max_err = 1e10;
    }
    cudaFree(d_max_err);
    if (verbose) ALW_LOG_INFO("Orthogonality error = %f", max_err);
    return max_err;
}

// =============================================================================
// ОСНОВНЫЕ ХОСТ-ФУНКЦИИ — ЗАГЛУШКИ (ПОЛНАЯ РЕАЛИЗАЦИЯ УДАЛЕНА)
// =============================================================================

bool orthogonalize_basis_hybrid(double* basis_hi, double* basis_lo,
                                double* R_hi, double* R_lo,
                                double* orig_hi, double* orig_lo,
                                int num_basis, int N,
                                const OrthoParams& params) {
    AEDS_LOGW("orthogonalize_basis_hybrid: демо-версия, возвращает false");
    AEDS_LOGW("  Полная версия содержит гибридный выбор между MGS, Householder и SVD.");
    AEDS_LOGW("  Для получения полной версии обратитесь к автору.");
    (void)basis_hi; (void)basis_lo; (void)R_hi; (void)R_lo;
    (void)orig_hi; (void)orig_lo;
    (void)num_basis; (void)N; (void)params;
    return false;
}

bool orthogonalize_basis_batch(double* basis_hi, double* basis_lo,
                               double* R_hi, double* R_lo,
                               double* orig_hi, double* orig_lo,
                               int batch_size, int num_basis, int N,
                               const OrthoParams& params) {
    AEDS_LOGW("orthogonalize_basis_batch: демо-версия, возвращает false");
    AEDS_LOGW("  Полная версия содержит пакетную ортогонализацию с разделением на MGS/Hybrid.");
    AEDS_LOGW("  Для получения полной версии обратитесь к автору.");
    (void)basis_hi; (void)basis_lo; (void)R_hi; (void)R_lo;
    (void)orig_hi; (void)orig_lo;
    (void)batch_size; (void)num_basis; (void)N; (void)params;
    return false;
}

// =============================================================================
// ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ СБРОСА КЭША cond
// =============================================================================

void reset_cond_cache(OrthoParams& params) {
    params.cached_cond = -1.0;
    params.cond_call_counter = 0;
    if (params.verbose) ALW_LOG_INFO("Condition number cache reset.");
}

// =============================================================================
// СОХРАНЕНИЕ / ЗАГРУЗКА БАЗИСА
// =============================================================================

void save_basis_to_file(const std::string& filename, int num_basis, int frame_size,
                        const double* T_hi, const double* T_lo,
                        const double* R_hi, const double* R_lo) {
    std::fstream f(filename.c_str(), std::ios::out | std::ios::binary);
    if (!f.is_open()) {
        ALW_LOG_ERROR("Cannot open basis file for writing: %s", filename.c_str());
        throw std::runtime_error("Cannot open basis file for writing");
    }
    uint32_t magic = 0x414C5742;
    uint32_t version = 1;
    int32_t nb = num_basis;
    int32_t fs = frame_size;
    f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    f.write(reinterpret_cast<const char*>(&version), sizeof(version));
    f.write(reinterpret_cast<const char*>(&nb), sizeof(nb));
    f.write(reinterpret_cast<const char*>(&fs), sizeof(fs));
    size_t t_size = (size_t)num_basis * frame_size;
    f.write(reinterpret_cast<const char*>(T_hi), t_size * sizeof(double));
    f.write(reinterpret_cast<const char*>(T_lo), t_size * sizeof(double));
    size_t r_size = (size_t)num_basis * num_basis;
    f.write(reinterpret_cast<const char*>(R_hi), r_size * sizeof(double));
    f.write(reinterpret_cast<const char*>(R_lo), r_size * sizeof(double));
    if (!f.good()) {
        ALW_LOG_ERROR("Error writing basis file: %s", filename.c_str());
        throw std::runtime_error("Error writing basis file");
    }
    f.close();
    ALW_LOG_INFO("Basis saved to %s", filename.c_str());
}

bool load_basis_from_file(const std::string& filename, int expected_num_basis, int expected_frame_size,
                          double* T_hi, double* T_lo, double* R_hi, double* R_lo) {
    std::fstream f(filename.c_str(), std::ios::in | std::ios::binary);
    if (!f.is_open()) {
        ALW_LOG_WARN("Cannot open basis file for reading: %s", filename.c_str());
        return false;
    }
    f.seekg(0, std::ios::end);
    std::streampos file_size = f.tellg();
    f.seekg(0, std::ios::beg);
    uint32_t magic, version;
    int32_t nb, fs;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x414C5742) {
        ALW_LOG_WARN("Invalid magic in basis file: %s", filename.c_str());
        return false;
    }
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        ALW_LOG_WARN("Unsupported version in basis file: %d", version);
        return false;
    }
    f.read(reinterpret_cast<char*>(&nb), sizeof(nb));
    f.read(reinterpret_cast<char*>(&fs), sizeof(fs));
    if (nb != expected_num_basis || fs != expected_frame_size) {
        ALW_LOG_WARN("Basis file mismatch: expected num_basis=%d, frame_size=%d but got %d, %d",
                     expected_num_basis, expected_frame_size, nb, fs);
        return false;
    }
    size_t t_size = (size_t)nb * fs;
    size_t r_size = (size_t)nb * nb;
    size_t expected_size = sizeof(magic) + sizeof(version) + sizeof(nb) + sizeof(fs)
                           + 2 * t_size * sizeof(double) + 2 * r_size * sizeof(double);
    if (file_size < (std::streampos)expected_size) {
        ALW_LOG_WARN("Basis file is truncated (size %zu < expected %zu)", (size_t)file_size, expected_size);
        return false;
    }
    f.read(reinterpret_cast<char*>(T_hi), t_size * sizeof(double));
    f.read(reinterpret_cast<char*>(T_lo), t_size * sizeof(double));
    f.read(reinterpret_cast<char*>(R_hi), r_size * sizeof(double));
    f.read(reinterpret_cast<char*>(R_lo), r_size * sizeof(double));
    if (!f.good()) {
        ALW_LOG_ERROR("Error reading basis file: %s", filename.c_str());
        return false;
    }
    for (int i = 0; i < nb * fs; ++i) {
        if (isnan(T_hi[i]) || isinf(T_hi[i]) || isnan(T_lo[i]) || isinf(T_lo[i])) {
            ALW_LOG_WARN("Loaded basis contains NaN/Inf in T matrix");
            return false;
        }
    }
    for (int i = 0; i < nb * nb; ++i) {
        if (isnan(R_hi[i]) || isinf(R_hi[i]) || isnan(R_lo[i]) || isinf(R_lo[i])) {
            ALW_LOG_WARN("Loaded basis contains NaN/Inf in R matrix");
            return false;
        }
    }
    f.close();
    ALW_LOG_INFO("Basis loaded from %s", filename.c_str());
    return true;
}

void save_basis_batch_to_file(const std::string& filename,
                              int batch_size, int num_basis, int frame_size,
                              const double* T_hi, const double* T_lo,
                              const double* R_hi, const double* R_lo) {
    std::fstream f(filename.c_str(), std::ios::out | std::ios::binary);
    if (!f.is_open()) {
        ALW_LOG_ERROR("Cannot open basis batch file for writing: %s", filename.c_str());
        throw std::runtime_error("Cannot open basis batch file for writing");
    }
    uint32_t magic = 0x414C5742;
    uint32_t version = 2;
    int32_t bs = batch_size;
    int32_t nb = num_basis;
    int32_t fs = frame_size;
    f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    f.write(reinterpret_cast<const char*>(&version), sizeof(version));
    f.write(reinterpret_cast<const char*>(&bs), sizeof(bs));
    f.write(reinterpret_cast<const char*>(&nb), sizeof(nb));
    f.write(reinterpret_cast<const char*>(&fs), sizeof(fs));
    size_t t_total = (size_t)batch_size * num_basis * frame_size;
    size_t r_total = (size_t)batch_size * num_basis * num_basis;
    f.write(reinterpret_cast<const char*>(T_hi), t_total * sizeof(double));
    f.write(reinterpret_cast<const char*>(T_lo), t_total * sizeof(double));
    f.write(reinterpret_cast<const char*>(R_hi), r_total * sizeof(double));
    f.write(reinterpret_cast<const char*>(R_lo), r_total * sizeof(double));
    if (!f.good()) {
        ALW_LOG_ERROR("Error writing basis batch file: %s", filename.c_str());
        throw std::runtime_error("Error writing basis batch file");
    }
    f.close();
    ALW_LOG_INFO("Basis batch saved to %s", filename.c_str());
}

bool load_basis_batch_from_file(const std::string& filename,
                                int expected_batch_size, int expected_num_basis, int expected_frame_size,
                                double* T_hi, double* T_lo, double* R_hi, double* R_lo) {
    std::fstream f(filename.c_str(), std::ios::in | std::ios::binary);
    if (!f.is_open()) {
        ALW_LOG_WARN("Cannot open basis batch file: %s", filename.c_str());
        return false;
    }
    f.seekg(0, std::ios::end);
    std::streampos file_size = f.tellg();
    f.seekg(0, std::ios::beg);
    uint32_t magic, version;
    int32_t batch_size, num_basis, frame_size;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x414C5742) {
        ALW_LOG_WARN("Invalid magic in basis batch file");
        return false;
    }
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 2) {
        ALW_LOG_WARN("Unsupported basis batch file version: %d", version);
        return false;
    }
    f.read(reinterpret_cast<char*>(&batch_size), sizeof(batch_size));
    f.read(reinterpret_cast<char*>(&num_basis), sizeof(num_basis));
    f.read(reinterpret_cast<char*>(&frame_size), sizeof(frame_size));
    if (batch_size != expected_batch_size || num_basis != expected_num_basis || frame_size != expected_frame_size) {
        ALW_LOG_WARN("Basis batch file mismatch: expected batch=%d, basis=%d, frame=%d but got %d, %d, %d",
                     expected_batch_size, expected_num_basis, expected_frame_size,
                     batch_size, num_basis, frame_size);
        return false;
    }
    size_t t_total = (size_t)batch_size * num_basis * frame_size;
    size_t r_total = (size_t)batch_size * num_basis * num_basis;
    size_t expected_size = sizeof(magic) + sizeof(version) + sizeof(batch_size) + sizeof(num_basis) + sizeof(frame_size)
                           + 2 * t_total * sizeof(double) + 2 * r_total * sizeof(double);
    if (file_size < (std::streampos)expected_size) {
        ALW_LOG_WARN("Basis batch file is truncated (size %zu < expected %zu)", (size_t)file_size, expected_size);
        return false;
    }
    f.read(reinterpret_cast<char*>(T_hi), t_total * sizeof(double));
    f.read(reinterpret_cast<char*>(T_lo), t_total * sizeof(double));
    f.read(reinterpret_cast<char*>(R_hi), r_total * sizeof(double));
    f.read(reinterpret_cast<char*>(R_lo), r_total * sizeof(double));
    if (!f.good()) {
        ALW_LOG_ERROR("Error reading basis batch file");
        return false;
    }
    for (int i = 0; i < t_total; ++i) {
        if (isnan(T_hi[i]) || isinf(T_hi[i]) || isnan(T_lo[i]) || isinf(T_lo[i])) {
            ALW_LOG_WARN("Loaded basis batch contains NaN/Inf in T matrix");
            return false;
        }
    }
    for (int i = 0; i < r_total; ++i) {
        if (isnan(R_hi[i]) || isinf(R_hi[i]) || isnan(R_lo[i]) || isinf(R_lo[i])) {
            ALW_LOG_WARN("Loaded basis batch contains NaN/Inf in R matrix");
            return false;
        }
    }
    f.close();
    ALW_LOG_INFO("Basis batch loaded from %s", filename.c_str());
    return true;
}