#ifndef ALW_HELIOS_UTILS_H
#define ALW_HELIOS_UTILS_H

#include "alw_math.h"
#include "alw_dictionary.h"
#include "aeds_solver.hpp"
#include "orthogonalization.h"
#include <memory>

// =============================================================================
// ОЦЕНКА ШУМА
// =============================================================================

double estimate_noise_from_diff(const alw_vector<double>& y_hi);

double compute_local_noise_estimate_gpu(double* d_res_hi, double* d_res_lo, int N, int n_blocks = 16);

// =============================================================================
// РЕШЕНИЕ СЛАУ (для финальной регрессии в OMP)
// =============================================================================

bool solve_linear_system_dd(int n,
                            const double* G_flat_hi, const double* G_flat_lo,
                            const double* b_hi, const double* b_lo,
                            double* x_hi, double* x_lo);

bool solve_linear_system_aeds(int m,
                              const alw_vector<double>& G_hi,
                              const alw_vector<double>& G_lo,
                              const alw_vector<double>& b_hi,
                              const alw_vector<double>& b_lo,
                              alw_vector<double>& x_hi,
                              alw_vector<double>& x_lo);

// =============================================================================
// OMP ДЛЯ ОДНОГО ФРЕЙМА (с финальной регрессией) — с поддержкой потока
// =============================================================================

void dd_hapt_frame_gpu_full(
    double* d_y_frame_hi,
    double* d_y_frame_lo,
    double* d_res_hi,
    double* d_res_lo,
    double* d_dict_hi,
    double* d_dict_lo,
    int num_atoms,
    int frame_size,
    int frame_start,
    alw_vector<DetectedEvent>& events,
    const alw_vector<double>& atom_norms,
    const alw_vector<AtomDescriptor>& descriptors,
    int max_iterations,
    double gamma_local,
    double gamma_global,
    double gamma_coherence,
    bool use_fixed_sigma,
    double fixed_sigma,
    bool no_final_regression,
    cudaStream_t stream = 0);

// =============================================================================
// ИНИЦИАЛИЗАЦИЯ ОРТОГОНАЛЬНОГО БАЗИСА — с поддержкой потока
// =============================================================================

bool init_orthogonal_basis(const alw_vector<double>& residual_hi, int frame_size,
                           int num_poly, int num_harm, int num_morlet,
                           const OrthoParams& params,
                           cudaStream_t stream = 0);

// =============================================================================
// ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ ОРТОГОНАЛЬНОГО БАЗИСА (доступны из core)
// =============================================================================

extern bool g_ortho_initialized;
extern std::unique_ptr<CudaPoolGuard<double>> g_d_Q_hi;
extern std::unique_ptr<CudaPoolGuard<double>> g_d_Q_lo;
extern std::unique_ptr<CudaPoolGuard<double>> g_d_R_hi;
extern std::unique_ptr<CudaPoolGuard<double>> g_d_R_lo;
extern int g_num_basis;
extern int g_num_poly;
extern int g_num_harm;
extern int g_num_morlet;

// =============================================================================
// ОБЪЯВЛЕНИЕ ЯДРА ДЛЯ ПРОЕКЦИИ (используется в alw_helios_core.cu)
// =============================================================================

__global__ void projection_kernel(
    const double* __restrict__ d_Q_hi,
    const double* __restrict__ d_Q_lo,
    const double* __restrict__ d_y_hi,
    const double* __restrict__ d_y_lo,
    double* __restrict__ d_c_hi,
    double* __restrict__ d_c_lo,
    int num_basis,
    int N);

#endif // ALW_HELIOS_UTILS_H
