// aeds_solver.hpp — публичный интерфейс адаптивного решателя AEDS (без реализации)
#pragma once

#ifndef AEDS_SOLVER_HPP
#define AEDS_SOLVER_HPP

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <algorithm>
#include "alw_math.h"
#include "aeds_core.hpp"
#include "aeds_profiler.hpp"

// Пороговые значения для выбора CPU/GPU (демонстрационные)
#define AEDS_CPU_THRESHOLD_N 17
#define AEDS_CPU_THRESHOLD_M 64

// ----------------------------------------------------------------------------
// Класс пула для выделения памяти на GPU (внутренний)
// ----------------------------------------------------------------------------
class CudaAllocPool {
    std::vector<void*> allocs;
public:
    ~CudaAllocPool() { for (auto p : allocs) cudaFree(p); }
    void* allocate(size_t n) { void* p = nullptr; cudaMalloc(&p, n); allocs.push_back(p); return p; }
};

// Структуры для хранения данных на хосте и устройстве
struct AEDS_HostData {
    alw_vector<DD> A_cmaj, b, x, L;
    alw_vector<DD> A_orig, b_orig;
    alw_vector<uint16_t> colors;
    uint16_t num_colors;
    alw_vector<DD> row_scales;
};

struct AEDS_DeviceData {
    DD *d_A_cmaj, *d_b, *d_x_old, *d_x_new, *d_L, *d_r;
    double *d_omega;
    DD *d_prev_grad;
    double *d_norm;
    DD *d_norm_dd;
    uint16_t* d_colors;
    uint8_t* d_converged;
};

// Объявления ядер (только сигнатуры, без реализации)
extern __global__ void compute_row_scales_kernel(
    const double* __restrict__, double* __restrict__, const int, const int);
extern __global__ void scale_b_vector_kernel(
    const double* __restrict__, double* __restrict__, const double* __restrict__, const int);
extern __global__ void transpose_and_scale_kernel(
    const double* __restrict__, double* __restrict__, double* __restrict__,
    const double* __restrict__, const int, const int);
extern __global__ void matvec_kernel(
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    double* __restrict__, const int, const int);
extern __global__ void compute_residual_norm_kernel_double(
    const double* __restrict__, double* __restrict__, const int);

extern __global__ void compute_row_scales_kernel_dd(
    const DD* __restrict__, DD* __restrict__, const int, const int);
extern __global__ void scale_b_vector_kernel_dd(
    const DD* __restrict__, DD* __restrict__, const DD* __restrict__, const int);
extern __global__ void transpose_and_scale_kernel_dd(
    const DD* __restrict__, DD* __restrict__, DD* __restrict__,
    const DD* __restrict__, const int, const int);
extern __global__ void matvec_kernel_dd(
    const DD* __restrict__, const DD* __restrict__, const DD* __restrict__,
    DD* __restrict__, const int, const int);
extern __global__ void compute_residual_norm_kernel_dd(
    const DD* __restrict__, DD* __restrict__, const int);

// Экспортируемые функции для решения прямоугольных систем (объявления)
bool aeds_lu_solve_rectangular_gpu(DD* d_A, DD* d_b, DD* d_x, int m, int n);
double check_residual_gpu(DD* d_A, DD* d_b, DD* d_x, int m, int n);
bool aeds_solve_rectangular(DD* d_A, DD* d_b, DD* d_x, int m, int n);

// ============================================================================
// BARE-CORE SOLVER (для малых систем, без рекурсии)
// ============================================================================
static __forceinline__ __device__ bool aeds_solve_dd_device_core(
    double* A_hi, double* A_lo,
    double* b_hi, double* b_lo,
    double* x_hi, double* x_lo,
    int n,
    int max_iter);

// ============================================================================
// КЛАСС AEDSSolver (публичный интерфейс)
// ============================================================================

class AEDSSolver {
    AEDS_Params params_;
    cudaStream_t stream_;
    bool use_dd_;
    bool use_direct_;
    static constexpr uint32_t threads_per_block = 256;
    AEDS_Profiler profiler_;

    // Приватные методы (реализации скрыты)
    void preprocess_host(const alw_vector<DD>& A_in, const alw_vector<DD>& b_in,
                         AEDS_HostData& ho, size_t m, size_t n, bool use_scaling);
    void allocate_and_copy_gpu(const AEDS_HostData& host, AEDS_DeviceData& dev,
                               size_t m, size_t n, CudaAllocPool& pool);
    void aeds_solve_gpu(AEDS_DeviceData& dev, const AEDS_Params& params,
                        size_t m, size_t n, bool& converged);
    void launch_native_preprocess(DD* d_A, DD* d_b, DD* d_x, size_t m, size_t n,
                                  CudaAllocPool& pool, AEDS_DeviceData& dev);
    bool aeds_solve_cpu_fallback(AEDS_HostData& h, size_t m, size_t n);
    bool solve_rectangular_gpu(const alw_vector<DD>& A_row_major, const alw_vector<DD>& b,
                               alw_vector<DD>& x, size_t m, size_t n);
    bool solve_direct_gpu(const alw_vector<DD>& A_row_major, const alw_vector<DD>& b,
                          alw_vector<DD>& x, size_t m, size_t n);

public:
    explicit AEDSSolver(const AEDS_Params& p);
    ~AEDSSolver();

    // Основной метод решения (интерфейс)
    bool solve(const alw_vector<DD>& A_row, const alw_vector<DD>& b,
               alw_vector<DD>& x, size_t m, size_t n);

    // Получение профиля
    const AEDS_Profile& get_profile() const { return profiler_.get_profile(); }
};

// ============================================================================
// Реализации методов — ЗАГЛУШКИ (демонстрационная версия)
// ============================================================================

inline AEDSSolver::AEDSSolver(const AEDS_Params& p)
    : params_(p), stream_(nullptr) {
    use_dd_ = p.use_dd_on_gpu;
    use_direct_ = p.use_direct;
    AEDS_LOGI("AEDSSolver constructed: use_dd=%d, use_direct=%d", use_dd_, use_direct_);
}

inline AEDSSolver::~AEDSSolver() {
    if (stream_) cudaStreamDestroy(stream_);
    cudaDeviceSynchronize();
    AEDS_LOGD("AEDSSolver destroyed");
}

// Основной метод solve — упрощённая версия для демонстрации
inline bool AEDSSolver::solve(const alw_vector<DD>& A_row_major,
                              const alw_vector<DD>& b,
                              alw_vector<DD>& x,
                              size_t m, size_t n) {
    AEDS_TRACE_ENTER();
    AEDS_LOGI("solve: m=%zu n=%zu (public demo version)", m, n);

    // Демонстрационная логика: используем Гаусса как простой fallback
    if (m != n) {
        AEDS_LOGW("Rectangular systems are supported in the full version.");
        AEDS_TRACE_EXIT();
        return false;
    }

    // Преобразуем в стандартный вектор для Гаусса
    std::vector<DD> A(m * n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            A[i * n + j] = A_row_major[i * n + j];

    std::vector<DD> bb(b.data(), b.data() + m);
    std::vector<DD> xx(n);

    bool ok = gauss_solve_dd(A, bb, xx, (int)n);
    if (ok) {
        x.assign(xx.begin(), xx.end());
        AEDS_LOGI("Solved via Gaussian elimination (demo)");
    } else {
        AEDS_LOGE("Gaussian elimination failed");
    }

    AEDS_TRACE_EXIT();
    return ok;
}

// Остальные методы — заглушки (полная реализация доступна в коммерческой версии)
inline void AEDSSolver::preprocess_host(const alw_vector<DD>&, const alw_vector<DD>&,
                                        AEDS_HostData&, size_t, size_t, bool) {
    AEDS_LOGW("preprocess_host: demo stub");
}

inline void AEDSSolver::allocate_and_copy_gpu(const AEDS_HostData&, AEDS_DeviceData&,
                                              size_t, size_t, CudaAllocPool&) {
    AEDS_LOGW("allocate_and_copy_gpu: demo stub");
}

inline void AEDSSolver::aeds_solve_gpu(AEDS_DeviceData&, const AEDS_Params&,
                                       size_t, size_t, bool&) {
    AEDS_LOGW("aeds_solve_gpu: demo stub");
}

inline void AEDSSolver::launch_native_preprocess(DD*, DD*, DD*, size_t, size_t,
                                                 CudaAllocPool&, AEDS_DeviceData&) {
    AEDS_LOGW("launch_native_preprocess: demo stub");
}

inline bool AEDSSolver::aeds_solve_cpu_fallback(AEDS_HostData&, size_t, size_t) {
    AEDS_LOGW("aeds_solve_cpu_fallback: demo stub");
    return false;
}

inline bool AEDSSolver::solve_rectangular_gpu(const alw_vector<DD>&, const alw_vector<DD>&,
                                              alw_vector<DD>&, size_t, size_t) {
    AEDS_LOGW("solve_rectangular_gpu: demo stub");
    return false;
}

inline bool AEDSSolver::solve_direct_gpu(const alw_vector<DD>&, const alw_vector<DD>&,
                                         alw_vector<DD>&, size_t, size_t) {
    AEDS_LOGW("solve_direct_gpu: demo stub");
    return false;
}

// Реализация bare-core solver (без адаптации)
static __forceinline__ __device__ bool aeds_solve_dd_device_core(
    double* A_hi, double* A_lo,
    double* b_hi, double* b_lo,
    double* x_hi, double* x_lo,
    int n,
    int max_iter) {
    // Упрощённая версия для малых систем (без адаптивного omega)
    if (n <= 0 || n > 4) return false;
    // ... (здесь можно оставить код, но для краткости возвращаем false)
    // В полной версии здесь реализован итерационный метод с регуляризацией.
    return false;
}

#endif // AEDS_SOLVER_HPP