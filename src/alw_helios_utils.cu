// =============================================================================
// alw_helios_utils.cu — вспомогательные функции для Helios
// Версия 4.2 — ИСПРАВЛЕНИЯ:
//   - Добавлены глобальные переменные для AFC (g_afc_temp_*)
//   - Удалён дубликат compute_local_noise_estimate_gpu
// =============================================================================

#include "alw_helios_utils.h"
#include "alw_dictionary.h"
#include "amad_x.h"
#include "eop_core.h"
#include "afc.h"
#include "orthogonalization.h"
#include "helios_config.h"
#include "alw_helios_core.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <random>
#include <mutex>

// =============================================================================
// ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ (доступ из alw_helios_core.cu)
// =============================================================================
extern HeliosConfig g_helios_config;
extern bool g_eod_trained;
extern double* g_d_dict_hi_global;
extern double* g_d_dict_lo_global;
extern alw_vector<double> g_atom_norms_global;
extern int g_global_num_atoms;
extern int g_global_frame_size;

// =============================================================================
// ГЛОБАЛЬНЫЙ ФЛАГ ДЛЯ ОДНОКРАТНОЙ ИНИЦИАЛИЗАЦИИ ОРТОГОНАЛЬНОГО БАЗИСА
// =============================================================================
static std::once_flag g_ortho_init_flag;

// =============================================================================
// ГЛОБАЛЬНЫЕ БУФЕРЫ ДЛЯ EOP (ПЕРЕИСПОЛЬЗУЕМЫЕ)
// =============================================================================
static double* g_d_proj_hi = nullptr;
static double* g_d_proj_lo = nullptr;
static double* g_d_coh_hi = nullptr;
static double* g_d_coh_lo = nullptr;
static double* g_d_energy_hi = nullptr;
static double* g_d_energy_lo = nullptr;
static double* g_d_norm_sq = nullptr;
static double* g_d_block_best_energy = nullptr;
static int* g_d_block_best_idx = nullptr;
static int* g_d_selected_indices = nullptr;
static int g_eop_buffers_num_atoms = 0;
static bool g_eop_buffers_allocated = false;

// =============================================================================
// ГЛОБАЛЬНЫЕ БУФЕРЫ ДЛЯ AFC (ПЕРЕГЕНЕРАЦИЯ СЛОВАРЯ)
// =============================================================================
static double* g_afc_temp_dict_hi = nullptr;
static double* g_afc_temp_dict_lo = nullptr;
static int g_afc_temp_num_atoms = 0;
static int g_afc_temp_frame_size = 0;
static bool g_afc_temp_allocated = false;

// =============================================================================
// СТАТИЧЕСКИЕ ЯДРА (используются только в этом файле)
// =============================================================================

// 1. Частотный сканер (для AFC)
__global__ static void frequency_scanner_kernel(
    const double* __restrict__ y_hi,
    const double* __restrict__ y_lo,
    int chunk_offset,
    double w_start,
    double w_step,
    int N,
    double* __restrict__ best_energy_out,
    double* __restrict__ best_w_out)
{
    extern __shared__ double sdata[];
    double* s_dot_s_hi  = &sdata[0 * blockDim.x];
    double* s_dot_s_lo  = &sdata[1 * blockDim.x];
    double* s_norm_s_hi = &sdata[2 * blockDim.x];
    double* s_norm_s_lo = &sdata[3 * blockDim.x];
    double* s_dot_c_hi  = &sdata[4 * blockDim.x];
    double* s_dot_c_lo  = &sdata[5 * blockDim.x];
    double* s_norm_c_hi = &sdata[6 * blockDim.x];
    double* s_norm_c_lo = &sdata[7 * blockDim.x];

    int tid = threadIdx.x;
    int freq_idx = chunk_offset + blockIdx.x;
    double my_w = w_start + (double)freq_idx * w_step;

    double t_dot_s_hi = 0.0, t_dot_s_lo = 0.0;
    double t_norm_s_hi = 0.0, t_norm_s_lo = 0.0;
    double t_dot_c_hi = 0.0, t_dot_c_lo = 0.0;
    double t_norm_c_hi = 0.0, t_norm_c_lo = 0.0;

    for (int i = tid; i < N; i += blockDim.x) {
        double yh = y_hi[i], yl = y_lo[i];
        if (isnan(yh) || isinf(yh)) { yh = 0.0; yl = 0.0; }

        double i_dd = (double)i;
        double n1_dd = (double)(N - 1);
        double t_ratio_hi, t_ratio_lo;
        alw_div_dd(i_dd, 0.0, n1_dd, 0.0, t_ratio_hi, t_ratio_lo);
        double t_hi, t_lo;
        alw_mul_dd(t_ratio_hi, t_ratio_lo, 30.0, 0.0, t_hi, t_lo);

        double wt_hi, wt_lo;
        alw_mul_dd(my_w, 0.0, t_hi, t_lo, wt_hi, wt_lo);
        double s_hi, s_lo, c_hi, c_lo;
        alw_sin_cos_dd(wt_hi, wt_lo, s_hi, s_lo, c_hi, c_lo);

        double p1_hi, p1_lo;
        alw_mul_dd(yh, yl, s_hi, s_lo, p1_hi, p1_lo);
        alw_add_dd(t_dot_s_hi, t_dot_s_lo, p1_hi, p1_lo, t_dot_s_hi, t_dot_s_lo);

        double p2_hi, p2_lo;
        alw_mul_dd(s_hi, s_lo, s_hi, s_lo, p2_hi, p2_lo);
        alw_add_dd(t_norm_s_hi, t_norm_s_lo, p2_hi, p2_lo, t_norm_s_hi, t_norm_s_lo);

        double p3_hi, p3_lo;
        alw_mul_dd(yh, yl, c_hi, c_lo, p3_hi, p3_lo);
        alw_add_dd(t_dot_c_hi, t_dot_c_lo, p3_hi, p3_lo, t_dot_c_hi, t_dot_c_lo);

        double p4_hi, p4_lo;
        alw_mul_dd(c_hi, c_lo, c_hi, c_lo, p4_hi, p4_lo);
        alw_add_dd(t_norm_c_hi, t_norm_c_lo, p4_hi, p4_lo, t_norm_c_hi, t_norm_c_lo);
    }

    s_dot_s_hi[tid] = t_dot_s_hi; s_dot_s_lo[tid] = t_dot_s_lo;
    s_norm_s_hi[tid] = t_norm_s_hi; s_norm_s_lo[tid] = t_norm_s_lo;
    s_dot_c_hi[tid] = t_dot_c_hi; s_dot_c_lo[tid] = t_dot_c_lo;
    s_norm_c_hi[tid] = t_norm_c_hi; s_norm_c_lo[tid] = t_norm_c_lo;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            alw_add_dd(s_dot_s_hi[tid], s_dot_s_lo[tid],
                       s_dot_s_hi[tid+s], s_dot_s_lo[tid+s],
                       s_dot_s_hi[tid], s_dot_s_lo[tid]);
            alw_add_dd(s_norm_s_hi[tid], s_norm_s_lo[tid],
                       s_norm_s_hi[tid+s], s_norm_s_lo[tid+s],
                       s_norm_s_hi[tid], s_norm_s_lo[tid]);
            alw_add_dd(s_dot_c_hi[tid], s_dot_c_lo[tid],
                       s_dot_c_hi[tid+s], s_dot_c_lo[tid+s],
                       s_dot_c_hi[tid], s_dot_c_lo[tid]);
            alw_add_dd(s_norm_c_hi[tid], s_norm_c_lo[tid],
                       s_norm_c_hi[tid+s], s_norm_c_lo[tid+s],
                       s_norm_c_hi[tid], s_norm_c_lo[tid]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        double energy = -1.0;
        double ns_hi = s_norm_s_hi[0];
        double nc_hi = s_norm_c_hi[0];

        if (!isnan(ns_hi) && !isinf(ns_hi) && ns_hi > 1e-12 &&
            !isnan(nc_hi) && !isinf(nc_hi) && nc_hi > 1e-12) {
            energy = (s_dot_s_hi[0] * s_dot_s_hi[0]) / ns_hi +
                     (s_dot_c_hi[0] * s_dot_c_hi[0]) / nc_hi;
        }
        if (isnan(energy) || isinf(energy)) energy = -1.0;
        best_energy_out[freq_idx] = energy;
        best_w_out[freq_idx] = my_w;
    }
}

// 2. Извлечение фрейма с окном Ханна (для AFC)
__global__ static void extract_and_window_frame_kernel(
    const double* __restrict__ in_hi,
    const double* __restrict__ in_lo,
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int start_idx,
    int frame_size,
    int total_len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= frame_size) return;
    if (start_idx + i >= total_len) {
        out_hi[i] = 0.0; out_lo[i] = 0.0;
        return;
    }
    if (frame_size < 2) {
        out_hi[i] = in_hi[start_idx + i];
        out_lo[i] = in_lo[start_idx + i];
        return;
    }
    double val_hi = in_hi[start_idx + i];
    double val_lo = in_lo[start_idx + i];
    if (isnan(val_hi) || isinf(val_hi)) { val_hi = 0.0; val_lo = 0.0; }

    double i_dd = (double)i;
    double n1_dd = (double)(frame_size - 1);
    double t_ratio_hi, t_ratio_lo;
    alw_div_dd(i_dd, 0.0, n1_dd, 0.0, t_ratio_hi, t_ratio_lo);

    double arg_hi, arg_lo;
    alw_mul_dd(ALW_TWO_PI_1, 0.0, t_ratio_hi, t_ratio_lo, arg_hi, arg_lo);
    double s_hi, s_lo, c_hi, c_lo;
    alw_sin_cos_dd(arg_hi, arg_lo, s_hi, s_lo, c_hi, c_lo);

    double w_hi, w_lo;
    alw_add_dd(1.0, 0.0, -c_hi, -c_lo, w_hi, w_lo);
    alw_mul_dd(w_hi, w_lo, 0.5, 0.0, w_hi, w_lo);

    alw_mul_dd(val_hi, val_lo, w_hi, w_lo, out_hi[i], out_lo[i]);
}

// 3. Построение неортогонального базиса (для ортогонализации)
__global__ static void alw_build_basis_kernel(
    double* __restrict__ T_hi,
    double* __restrict__ T_lo,
    double w_hi,
    double w_lo,
    int N_frame,
    int num_poly,
    int num_harm,
    int num_morlet)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;
    if (N_frame < 2) {
        T_hi[i] = 0.0; T_lo[i] = 0.0;
        return;
    }
    if (num_poly < 0) num_poly = 0;
    if (num_harm < 0) num_harm = 0;
    if (num_morlet < 0) num_morlet = 0;

    double i_dd = (double)i;
    double n1_dd = (double)(N_frame - 1);
    double t_ratio_hi, t_ratio_lo;
    alw_div_dd(i_dd, 0.0, n1_dd, 0.0, t_ratio_hi, t_ratio_lo);

    double x_hi, x_lo;
    alw_mul_dd(t_ratio_hi, t_ratio_lo, 2.0, 0.0, x_hi, x_lo);
    alw_add_dd(x_hi, x_lo, -1.0, 0.0, x_hi, x_lo);

    if (num_poly > 0) {
        double t_prev2_hi = 1.0, t_prev2_lo = 0.0;
        T_hi[0 * N_frame + i] = t_prev2_hi;
        T_lo[0 * N_frame + i] = t_prev2_lo;

        if (num_poly > 1) {
            double t_prev1_hi = x_hi, t_prev1_lo = x_lo;
            T_hi[1 * N_frame + i] = t_prev1_hi;
            T_lo[1 * N_frame + i] = t_prev1_lo;

            double two_x_hi, two_x_lo;
            alw_mul_dd(x_hi, x_lo, 2.0, 0.0, two_x_hi, two_x_lo);

            for (int k = 2; k < num_poly; ++k) {
                double tk_hi, tk_lo;
                alw_mul_dd(two_x_hi, two_x_lo, t_prev1_hi, t_prev1_lo, tk_hi, tk_lo);
                alw_add_dd(tk_hi, tk_lo, -t_prev2_hi, -t_prev2_lo, tk_hi, tk_lo);
                T_hi[k * N_frame + i] = tk_hi;
                T_lo[k * N_frame + i] = tk_lo;
                t_prev2_hi = t_prev1_hi;
                t_prev2_lo = t_prev1_lo;
                t_prev1_hi = tk_hi;
                t_prev1_lo = tk_lo;
            }
        }
    }

    double t_real_hi, t_real_lo;
    alw_mul_dd(t_ratio_hi, t_ratio_lo, 30.0, 0.0, t_real_hi, t_real_lo);

    int base = num_poly;
    for (int k = 1; k <= num_harm / 2; ++k) {
        double kw_hi, kw_lo;
        alw_mul_dd(w_hi, w_lo, (double)k, 0.0, kw_hi, kw_lo);

        double wt_hi, wt_lo;
        alw_mul_dd(kw_hi, kw_lo, t_real_hi, t_real_lo, wt_hi, wt_lo);

        double sin_hi, sin_lo, cos_hi, cos_lo;
        alw_sin_cos_dd(wt_hi, wt_lo, sin_hi, sin_lo, cos_hi, cos_lo);

        T_hi[(base + 2 * (k - 1)) * N_frame + i] = sin_hi;
        T_lo[(base + 2 * (k - 1)) * N_frame + i] = sin_lo;
        T_hi[(base + 2 * (k - 1) + 1) * N_frame + i] = cos_hi;
        T_lo[(base + 2 * (k - 1) + 1) * N_frame + i] = cos_lo;
    }

    double sigma = 1.5;
    double w_morlet = 15.0;
    base = num_poly + num_harm;

    for (int j = 0; j < num_morlet; ++j) {
        double mu = (num_morlet > 1) ? ((double)j / (num_morlet - 1)) * 30.0 : 15.0;
        double dt_hi, dt_lo;
        alw_add_dd(t_real_hi, t_real_lo, -mu, 0.0, dt_hi, dt_lo);

        double z_hi, z_lo;
        alw_div_dd(dt_hi, dt_lo, sigma, 0.0, z_hi, z_lo);

        double z2_hi, z2_lo;
        alw_mul_dd(z_hi, z_lo, z_hi, z_lo, z2_hi, z2_lo);

        double env_hi, env_lo;
        alw_mul_dd(z2_hi, z2_lo, -0.5, 0.0, env_hi, env_lo);

        double g_val = exp(env_hi);
        if (isnan(g_val) || isinf(g_val)) g_val = 0.0;

        double arg_m_hi, arg_m_lo;
        alw_mul_dd(dt_hi, dt_lo, w_morlet, 0.0, arg_m_hi, arg_m_lo);

        double ms_hi, ms_lo, mc_hi, mc_lo;
        alw_sin_cos_dd(arg_m_hi, arg_m_lo, ms_hi, ms_lo, mc_hi, mc_lo);

        double wave_hi, wave_lo;
        alw_mul_dd(g_val, 0.0, mc_hi, mc_lo, wave_hi, wave_lo);

        T_hi[(base + j) * N_frame + i] = wave_hi;
        T_lo[(base + j) * N_frame + i] = wave_lo;
    }
}

// 4. Проекция фрейма на ортогональный базис
__global__ void projection_kernel(
    const double* __restrict__ d_Q_hi,
    const double* __restrict__ d_Q_lo,
    const double* __restrict__ d_y_hi,
    const double* __restrict__ d_y_lo,
    double* __restrict__ d_c_hi,
    double* __restrict__ d_c_lo,
    int num_basis,
    int N)
{
    int basis_idx = blockIdx.x;
    if (basis_idx >= num_basis) return;

    int tid = threadIdx.x;
    extern __shared__ double s_proj[];
    double* s_hi = s_proj;
    double* s_lo = &s_proj[blockDim.x];

    double my_sum_hi = 0.0, my_sum_lo = 0.0;

    for (int i = tid; i < N; i += blockDim.x) {
        double qh = __ldg(d_Q_hi + basis_idx * N + i);
        double ql = __ldg(d_Q_lo + basis_idx * N + i);
        double yh = __ldg(d_y_hi + i);
        double yl = __ldg(d_y_lo + i);

        if (isnan(qh) || isinf(qh)) qh = 0.0;
        if (isnan(ql) || isinf(ql)) ql = 0.0;
        if (isnan(yh) || isinf(yh)) { yh = 0.0; yl = 0.0; }

        double p_hi, p_lo;
        alw_mul_dd(qh, ql, yh, yl, p_hi, p_lo);
        alw_add_dd(my_sum_hi, my_sum_lo, p_hi, p_lo, my_sum_hi, my_sum_lo);
    }

    s_hi[tid] = my_sum_hi;
    s_lo[tid] = my_sum_lo;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            alw_add_dd(s_hi[tid], s_lo[tid],
                       s_hi[tid + s], s_lo[tid + s],
                       s_hi[tid], s_lo[tid]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_c_hi[basis_idx] = s_hi[0];
        d_c_lo[basis_idx] = s_lo[0];
    }
}

// =============================================================================
// ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ ОРТОГОНАЛЬНОГО БАЗИСА
// =============================================================================
bool g_ortho_initialized = false;
std::unique_ptr<CudaPoolGuard<double>> g_d_Q_hi;
std::unique_ptr<CudaPoolGuard<double>> g_d_Q_lo;
std::unique_ptr<CudaPoolGuard<double>> g_d_R_hi;
std::unique_ptr<CudaPoolGuard<double>> g_d_R_lo;
int g_num_basis = 0;
int g_num_poly = 10;
int g_num_harm = 10;
int g_num_morlet = 10;

// =============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// =============================================================================

// 1. Оценка шума по разностям (оставлена для обратной совместимости, но не используется)
double estimate_noise_from_diff(const alw_vector<double>& y_hi) {
    int n = (int)y_hi.size();
    if (n < 2) return alw_get_min_epsilon();

    alw_vector<double> diffs;
    diffs.reserve(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        double d = y_hi[i + 1] - y_hi[i];
        if (isnan(d) || isinf(d)) d = 0.0;
        diffs.push_back(fabs(d));
    }
    if (diffs.empty()) return alw_get_min_epsilon();

    double median_diff = alw_median_host(diffs.data(), (int)diffs.size());
    if (median_diff < 1e-12) return alw_get_min_epsilon();

    const double NORM_MEDIAN_TO_SIGMA = 0.67448975;
    const double SQRT2 = 1.4142135623730951;
    double sigma = median_diff / (NORM_MEDIAN_TO_SIGMA * SQRT2);
    if (isnan(sigma) || isinf(sigma) || sigma < alw_get_min_epsilon())
        sigma = alw_get_min_epsilon();
    return sigma;
}

// 2. Оценка шума на GPU через AMAD‑X
double compute_local_noise_estimate_gpu(double* d_res_hi, double* d_res_lo, int N, int n_blocks) {
    return amad_x_estimate_adaptive(d_res_hi, d_res_lo, N, false);
}

// 3. Решение СЛАУ методом Гаусса (CPU, DD) — оставлено для обратной совместимости
bool solve_linear_system_dd(int n,
                            const double* G_flat_hi, const double* G_flat_lo,
                            const double* b_hi, const double* b_lo,
                            double* x_hi, double* x_lo) {
    if (n == 0) return true;

    for (int i = 0; i < n * n; ++i) {
        if (isnan(G_flat_hi[i]) || isinf(G_flat_hi[i]) ||
            isnan(G_flat_lo[i]) || isinf(G_flat_lo[i])) {
            ALW_LOG_ERROR("G matrix contains NaN/Inf");
            return false;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (isnan(b_hi[i]) || isinf(b_hi[i]) ||
            isnan(b_lo[i]) || isinf(b_lo[i])) {
            ALW_LOG_ERROR("b vector contains NaN/Inf");
            return false;
        }
    }

    alw_vector<double> A_hi(n * n), A_lo(n * n);
    alw_vector<double> rhs_hi(n), rhs_lo(n);

    for (int i = 0; i < n * n; ++i) {
        A_hi[i] = G_flat_hi[i];
        A_lo[i] = G_flat_lo[i];
    }
    for (int i = 0; i < n; ++i) {
        rhs_hi[i] = b_hi[i];
        rhs_lo[i] = b_lo[i];
    }

    double max_G = 0.0;
    for (int i = 0; i < n * n; ++i) {
        double val = fabs(A_hi[i]);
        if (isnan(val) || isinf(val)) val = 0.0;
        if (val > max_G) max_G = val;
    }
    if (max_G < 1e-30) {
        ALW_LOG_WARN("G matrix is near zero");
        return false;
    }

    const double tol = 1e-14 * std::max(1.0, max_G);

    for (int col = 0; col < n; ++col) {
        int max_row = col;
        double max_abs = alw_abs_hi(A_hi[col * n + col], A_lo[col * n + col]);
        if (isnan(max_abs) || isinf(max_abs)) max_abs = 0.0;
        for (int row = col + 1; row < n; ++row) {
            double abs_val = alw_abs_hi(A_hi[row * n + col], A_lo[row * n + col]);
            if (isnan(abs_val) || isinf(abs_val)) abs_val = 0.0;
            if (abs_val > max_abs) {
                max_abs = abs_val;
                max_row = row;
            }
        }
        if (max_abs < tol) {
            ALW_LOG_WARN("Matrix singular at column %d", col);
            return false;
        }
        if (max_row != col) {
            for (int k = col; k < n; ++k) {
                std::swap(A_hi[col * n + k], A_hi[max_row * n + k]);
                std::swap(A_lo[col * n + k], A_lo[max_row * n + k]);
            }
            std::swap(rhs_hi[col], rhs_hi[max_row]);
            std::swap(rhs_lo[col], rhs_lo[max_row]);
        }

        double piv_hi = A_hi[col * n + col];
        double piv_lo = A_lo[col * n + col];
        if (isnan(piv_hi) || isinf(piv_hi) || fabs(piv_hi) < tol) return false;
        double inv_piv_hi, inv_piv_lo;
        alw_div_dd(1.0, 0.0, piv_hi, piv_lo, inv_piv_hi, inv_piv_lo);
        for (int k = col; k < n; ++k) {
            double tmp_hi, tmp_lo;
            alw_mul_dd(A_hi[col * n + k], A_lo[col * n + k], inv_piv_hi, inv_piv_lo, tmp_hi, tmp_lo);
            A_hi[col * n + k] = tmp_hi;
            A_lo[col * n + k] = tmp_lo;
        }
        alw_mul_dd(rhs_hi[col], rhs_lo[col], inv_piv_hi, inv_piv_lo, rhs_hi[col], rhs_lo[col]);

        for (int row = col + 1; row < n; ++row) {
            double factor_hi = A_hi[row * n + col];
            double factor_lo = A_lo[row * n + col];
            if (isnan(factor_hi) || isinf(factor_hi) || fabs(factor_hi) < tol) continue;
            for (int k = col; k < n; ++k) {
                double tmp_hi, tmp_lo;
                alw_mul_dd(factor_hi, factor_lo, A_hi[col * n + k], A_lo[col * n + k], tmp_hi, tmp_lo);
                alw_sub_dd(A_hi[row * n + k], A_lo[row * n + k],
                           tmp_hi, tmp_lo, A_hi[row * n + k], A_lo[row * n + k]);
            }
            double tmp_hi, tmp_lo;
            alw_mul_dd(factor_hi, factor_lo, rhs_hi[col], rhs_lo[col], tmp_hi, tmp_lo);
            alw_sub_dd(rhs_hi[row], rhs_lo[row], tmp_hi, tmp_lo, rhs_hi[row], rhs_lo[row]);
        }
    }

    for (int col = n - 1; col >= 0; --col) {
        x_hi[col] = rhs_hi[col];
        x_lo[col] = rhs_lo[col];
        for (int row = col - 1; row >= 0; --row) {
            double factor_hi = A_hi[row * n + col];
            double factor_lo = A_lo[row * n + col];
            if (isnan(factor_hi) || isinf(factor_hi) || fabs(factor_hi) < tol) continue;
            double tmp_hi, tmp_lo;
            alw_mul_dd(factor_hi, factor_lo, x_hi[col], x_lo[col], tmp_hi, tmp_lo);
            alw_sub_dd(rhs_hi[row], rhs_lo[row], tmp_hi, tmp_lo, rhs_hi[row], rhs_lo[row]);
        }
    }

    for (int i = 0; i < n; ++i) {
        if (isnan(x_hi[i]) || isinf(x_hi[i])) {
            ALW_LOG_WARN("Solution contains NaN/Inf");
            return false;
        }
    }
    return true;
}

// 4. Решение СЛАУ через AEDS (CPU) — оставлено для обратной совместимости
bool solve_linear_system_aeds(int m,
                              const alw_vector<double>& G_hi,
                              const alw_vector<double>& G_lo,
                              const alw_vector<double>& b_hi,
                              const alw_vector<double>& b_lo,
                              alw_vector<double>& x_hi,
                              alw_vector<double>& x_lo) {
    if (m == 0) return true;

    alw_vector<DD> A_row_major(m * m);
    for (int i = 0; i < m * m; ++i) {
        A_row_major[i].hi = G_hi[i];
        A_row_major[i].lo = G_lo[i];
        if (isnan(A_row_major[i].hi) || isinf(A_row_major[i].hi)) A_row_major[i].hi = 0.0;
        if (isnan(A_row_major[i].lo) || isinf(A_row_major[i].lo)) A_row_major[i].lo = 0.0;
    }

    alw_vector<DD> b_dd(m);
    for (int i = 0; i < m; ++i) {
        b_dd[i].hi = b_hi[i];
        b_dd[i].lo = b_lo[i];
        if (isnan(b_dd[i].hi) || isinf(b_dd[i].hi)) b_dd[i].hi = 0.0;
        if (isnan(b_dd[i].lo) || isinf(b_dd[i].lo)) b_dd[i].lo = 0.0;
    }

    alw_vector<DD> x_dd(m, DD{0.0, 0.0});

    AEDS_Params aeds_params;
    aeds_params.target_precision = 1e-30;
    aeds_params.max_iterations   = 500;
    aeds_params.omega_init       = 1.0;
    aeds_params.tikhonov_alpha   = 1e-28;
    aeds_params.use_precond      = true;

    AEDSSolver solver(aeds_params);
    bool converged = solver.solve(A_row_major, b_dd, x_dd, (size_t)m, (size_t)m);

    x_hi.resize(m);
    x_lo.resize(m);
    for (int i = 0; i < m; ++i) {
        x_hi[i] = x_dd[i].hi;
        x_lo[i] = x_dd[i].lo;
        if (isnan(x_hi[i]) || isinf(x_hi[i])) { x_hi[i] = 0.0; x_lo[i] = 0.0; }
    }

    return converged;
}

// =============================================================================
// ПОЛНАЯ РЕАЛИЗАЦИЯ AFC С ПЕРЕГЕНЕРАЦИЕЙ СЛОВАРЯ
// =============================================================================

// Вспомогательная функция для поиска доминирующей частоты
static double find_dominant_frequency_gpu(
    const double* d_signal_hi,
    const double* d_signal_lo,
    int N,
    const AFCConfig& afc_cfg)
{
    if (!afc_cfg.enable || N < 4) return 0.0;

    int num_search_points = afc_cfg.coarse_points;
    double f_min = afc_cfg.freq_min;
    double f_max = afc_cfg.freq_max;

    CudaPoolGuard<double> d_best_e(num_search_points);
    CudaPoolGuard<double> d_best_w(num_search_points);
    if (!d_best_e.is_valid() || !d_best_w.is_valid()) {
        ALW_LOG_ERROR("AFC: memory allocation failed");
        return 0.0;
    }

    int scannerThreads = 256;
    double w_step = (f_max - f_min) / num_search_points;

    for (int offset = 0; offset < num_search_points; offset += 4096) {
        int chunk = std::min(4096, num_search_points - offset);
        size_t shared_mem_size = 8 * scannerThreads * sizeof(double);
        frequency_scanner_kernel<<<chunk, scannerThreads, shared_mem_size>>>(
            d_signal_hi, d_signal_lo, offset, f_min, w_step, N,
            d_best_e.get(), d_best_w.get());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    alw_vector<double> h_best_e(num_search_points), h_best_w(num_search_points);
    CUDA_CHECK(cudaMemcpy(h_best_e.data(), d_best_e.get(), num_search_points * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_best_w.data(), d_best_w.get(), num_search_points * sizeof(double), cudaMemcpyDeviceToHost));

    double best_energy = -1.0;
    double best_freq = 0.0;
    for (int i = 0; i < num_search_points; ++i) {
        if (isnan(h_best_e[i]) || isinf(h_best_e[i])) continue;
        if (h_best_e[i] > best_energy) {
            best_energy = h_best_e[i];
            best_freq = h_best_w[i];
        }
    }
    return best_freq;
}

// ПЕРЕГЕНЕРАЦИЯ СЛОВАРЯ ПО ЧАСТОТЕ
static void regenerate_dictionary_with_frequency(
    double* d_dict_hi,
    double* d_dict_lo,
    int num_atoms,
    int frame_size,
    double freq,
    const alw_vector<AtomDescriptor>& descriptors,
    alw_vector<double>& out_norms)
{
    if (freq <= 0.0 || num_atoms == 0 || frame_size == 0) {
        return;
    }

    // Создаём копию дескрипторов с модифицированными параметрами частоты
    // Для этого нам нужно получить оригинальные AtomSpec из глобального менеджера
    // Используем g_dict_manager для доступа к полным спецификациям
    if (g_dict_manager == nullptr) {
        ALW_LOG_WARN("AFC: g_dict_manager is null, cannot regenerate dictionary");
        return;
    }

    const auto& original_specs = g_dict_manager->get_descriptors();
    if (original_specs.size() != (size_t)num_atoms) {
        ALW_LOG_WARN("AFC: descriptor count mismatch, skipping dictionary regeneration");
        return;
    }

    alw_vector<AtomSpec> modified_descriptors = original_specs;
    int modified_count = 0;

    for (int i = 0; i < (int)modified_descriptors.size(); ++i) {
        AtomSpec& spec = modified_descriptors[i];
        bool modified = false;

        // Для CHIRP атомов меняем f0
        if (spec.type == CHIRP) {
            double old_f0 = spec.chirp.f0_hi;
            double new_f0 = freq;
            if (new_f0 < 0.001) new_f0 = 0.001;
            if (new_f0 > 0.49) new_f0 = 0.49;
            if (fabs(new_f0 - old_f0) / (old_f0 + 1e-12) > 0.1) {
                spec.chirp.f0_hi = new_f0;
                modified = true;
                modified_count++;
            }
        }
        // Для MORLET атомов меняем w_morlet (частота несущей)
        else if (spec.type == MORLET) {
            double old_w = spec.morlet.w_morlet;
            double new_w = 2.0 * M_PI * freq;
            if (new_w < 5.0) new_w = 5.0;
            if (new_w > 80.0) new_w = 80.0;
            if (fabs(new_w - old_w) / (old_w + 1e-12) > 0.1) {
                spec.morlet.w_morlet = new_w;
                modified = true;
                modified_count++;
            }
        }
        // Для DAMPED_CHIRP меняем f0
        else if (spec.type == DAMPED_CHIRP) {
            double old_f0 = spec.chirp.f0_hi;
            double new_f0 = freq;
            if (new_f0 < 0.001) new_f0 = 0.001;
            if (new_f0 > 0.49) new_f0 = 0.49;
            if (fabs(new_f0 - old_f0) / (old_f0 + 1e-12) > 0.1) {
                spec.chirp.f0_hi = new_f0;
                modified = true;
                modified_count++;
            }
        }
        // Для SIGMOID_OSC меняем f
        else if (spec.type == SIGMOID_OSC) {
            double old_f0 = spec.chirp.f0_hi;
            double new_f0 = freq;
            if (new_f0 < 0.001) new_f0 = 0.001;
            if (new_f0 > 0.49) new_f0 = 0.49;
            if (fabs(new_f0 - old_f0) / (old_f0 + 1e-12) > 0.1) {
                spec.chirp.f0_hi = new_f0;
                modified = true;
                modified_count++;
            }
        }
    }

    if (modified_count == 0) {
        ALW_LOG_DEBUG("AFC: no atoms needed modification for freq=%.4f", freq);
        return;
    }

    ALW_LOG_INFO("AFC: regenerating dictionary with freq=%.4f, modified %d atoms", freq, modified_count);

    // Выделяем временную память для нового словаря
    if (g_afc_temp_allocated && (g_afc_temp_num_atoms != num_atoms || g_afc_temp_frame_size != frame_size)) {
        cudaFree(g_afc_temp_dict_hi);
        cudaFree(g_afc_temp_dict_lo);
        g_afc_temp_allocated = false;
    }

    if (!g_afc_temp_allocated) {
        cudaError_t err;
        err = cudaMalloc(&g_afc_temp_dict_hi, (size_t)num_atoms * frame_size * sizeof(double));
        if (err != cudaSuccess) {
            ALW_LOG_ERROR("AFC: cudaMalloc for temp dict failed");
            return;
        }
        err = cudaMalloc(&g_afc_temp_dict_lo, (size_t)num_atoms * frame_size * sizeof(double));
        if (err != cudaSuccess) {
            cudaFree(g_afc_temp_dict_hi);
            ALW_LOG_ERROR("AFC: cudaMalloc for temp dict lo failed");
            return;
        }
        g_afc_temp_num_atoms = num_atoms;
        g_afc_temp_frame_size = frame_size;
        g_afc_temp_allocated = true;
    }

    // Перегенерируем словарь с новыми дескрипторами
    alw_vector<double> new_norms;
    build_dictionary_gpu(
        modified_descriptors,
        frame_size,
        g_afc_temp_dict_hi,
        g_afc_temp_dict_lo,
        new_norms
    );

    // Копируем новый словарь на место старого
    CUDA_CHECK(cudaMemcpy(d_dict_hi, g_afc_temp_dict_hi,
                          (size_t)num_atoms * frame_size * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_dict_lo, g_afc_temp_dict_lo,
                          (size_t)num_atoms * frame_size * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    out_norms = new_norms;
}

// =============================================================================
// ИНИЦИАЛИЗАЦИЯ БУФЕРОВ EOP
// =============================================================================
static bool init_eop_buffers(int num_atoms) {
    if (g_eop_buffers_allocated && g_eop_buffers_num_atoms == num_atoms) {
        return true;
    }

    if (g_eop_buffers_allocated) {
        cudaFree(g_d_proj_hi);
        cudaFree(g_d_proj_lo);
        cudaFree(g_d_coh_hi);
        cudaFree(g_d_coh_lo);
        cudaFree(g_d_energy_hi);
        cudaFree(g_d_energy_lo);
        cudaFree(g_d_norm_sq);
        cudaFree(g_d_block_best_energy);
        cudaFree(g_d_block_best_idx);
        cudaFree(g_d_selected_indices);
        g_eop_buffers_allocated = false;
    }

    int num_blocks = (num_atoms + 127) / 128;
    cudaError_t err;
    err = cudaMalloc(&g_d_proj_hi, num_atoms * sizeof(double));
    if (err != cudaSuccess) goto error;
    err = cudaMalloc(&g_d_proj_lo, num_atoms * sizeof(double));
    if (err != cudaSuccess) goto error;
    err = cudaMalloc(&g_d_coh_hi, num_atoms * sizeof(double));
    if (err != cudaSuccess) goto error;
    err = cudaMalloc(&g_d_coh_lo, num_atoms * sizeof(double));
    if (err != cudaSuccess) goto error;
    err = cudaMalloc(&g_d_energy_hi, num_atoms * sizeof(double));
    if (err != cudaSuccess) goto error;
    err = cudaMalloc(&g_d_energy_lo, num_atoms * sizeof(double));
    if (err != cudaSuccess) goto error;
    err = cudaMalloc(&g_d_norm_sq, sizeof(double));
    if (err != cudaSuccess) goto error;
    err = cudaMalloc(&g_d_block_best_energy, num_blocks * sizeof(double));
    if (err != cudaSuccess) goto error;
    err = cudaMalloc(&g_d_block_best_idx, num_blocks * sizeof(int));
    if (err != cudaSuccess) goto error;
    err = cudaMalloc(&g_d_selected_indices, num_atoms * sizeof(int));
    if (err != cudaSuccess) goto error;

    g_eop_buffers_num_atoms = num_atoms;
    g_eop_buffers_allocated = true;
    return true;

error:
    ALW_LOG_ERROR("init_eop_buffers: cudaMalloc failed: %s", cudaGetErrorString(err));
    cudaFree(g_d_proj_hi);
    cudaFree(g_d_proj_lo);
    cudaFree(g_d_coh_hi);
    cudaFree(g_d_coh_lo);
    cudaFree(g_d_energy_hi);
    cudaFree(g_d_energy_lo);
    cudaFree(g_d_norm_sq);
    cudaFree(g_d_block_best_energy);
    cudaFree(g_d_block_best_idx);
    cudaFree(g_d_selected_indices);
    return false;
}

// =============================================================================
// РАЗЛОЖЕНИЕ ОДНОГО ФРЕЙМА (EOP + AMAD‑X + AFC)
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
    cudaStream_t stream)
{
    // 1. Копирование фрейма в остаток
    CUDA_CHECK(cudaMemcpyAsync(d_res_hi, d_y_frame_hi, frame_size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_res_lo, d_y_frame_lo, frame_size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 2. Оценка шума через AMAD‑X
    double sigma_noise;
    if (use_fixed_sigma) {
        sigma_noise = fixed_sigma;
    } else {
        sigma_noise = amad_x_estimate_adaptive(d_res_hi, d_res_lo, frame_size, false);
    }

    // 3. AFC – адаптация словаря под доминирующую частоту
    const HeliosConfig& cfg = g_helios_config;
    if (cfg.afc.enable) {
        double best_freq = find_dominant_frequency_gpu(d_res_hi, d_res_lo, frame_size, cfg.afc);
        if (best_freq > 0.0) {
            alw_vector<double> new_norms = atom_norms;
            regenerate_dictionary_with_frequency(
                d_dict_hi,
                d_dict_lo,
                num_atoms,
                frame_size,
                best_freq,
                descriptors,
                new_norms
            );
        }
    }

    // 4. Подготовка параметров EOP
    EOP_Params eop_params;
    eop_params.alpha = cfg.eop.alpha;
    eop_params.gamma = cfg.eop.gamma;
    eop_params.theta = cfg.eop.theta;
    eop_params.max_iterations = max_iterations;
    eop_params.use_final_regression = !no_final_regression;

    // 5. Вызов EOP с переиспользуемыми буферами
    if (!g_eop_buffers_allocated || g_eop_buffers_num_atoms != num_atoms) {
        if (!init_eop_buffers(num_atoms)) {
            ALW_LOG_ERROR("dd_hapt_frame_gpu_full: EOP buffers not available");
            return;
        }
    }

    eop_pursuit(
        d_y_frame_hi, d_y_frame_lo,
        d_dict_hi, d_dict_lo,
        num_atoms, frame_size, frame_start,
        sigma_noise,
        eop_params,
        events,
        atom_norms,
        descriptors,
        false,
        d_res_hi, d_res_lo,
        g_d_proj_hi, g_d_proj_lo,
        g_d_coh_hi, g_d_coh_lo,
        g_d_energy_hi, g_d_energy_lo,
        g_d_norm_sq,
        g_d_block_best_energy,
        g_d_block_best_idx,
        g_d_selected_indices
    );
}

// =============================================================================
// ИНИЦИАЛИЗАЦИЯ ОРТОГОНАЛЬНОГО БАЗИСА (с AFC)
// =============================================================================
bool init_orthogonal_basis(const alw_vector<double>& residual_hi, int frame_size,
                           int num_poly, int num_harm, int num_morlet,
                           const OrthoParams& params,
                           cudaStream_t stream)
{
    (void)stream;
    std::call_once(g_ortho_init_flag, [&]() {
        int N = (int)residual_hi.size();
        g_num_poly = num_poly;
        g_num_harm = num_harm;
        g_num_morlet = num_morlet;
        g_num_basis = num_poly + num_harm + num_morlet;
        if (g_num_basis > 300) g_num_basis = 300;
        if (g_num_basis < 1) g_num_basis = 1;

        g_d_Q_hi.reset(new CudaPoolGuard<double>(g_num_basis * frame_size));
        g_d_Q_lo.reset(new CudaPoolGuard<double>(g_num_basis * frame_size));
        g_d_R_hi.reset(new CudaPoolGuard<double>(g_num_basis * g_num_basis));
        g_d_R_lo.reset(new CudaPoolGuard<double>(g_num_basis * g_num_basis));

        double* d_Q_hi = g_d_Q_hi->get();
        double* d_Q_lo = g_d_Q_lo->get();
        double* d_R_hi = g_d_R_hi->get();
        double* d_R_lo = g_d_R_lo->get();

        if (!d_Q_hi || !d_Q_lo || !d_R_hi || !d_R_lo) {
            ALW_LOG_ERROR("init_orthogonal_basis: memory allocation failed");
            g_ortho_initialized = false;
            return;
        }

        CudaPoolGuard<double> d_signal_hi(N), d_signal_lo(N);
        CUDA_CHECK(cudaMemcpy(d_signal_hi.get(), residual_hi.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_signal_lo.get(), 0, N * sizeof(double)));

        // Используем AFC для поиска частоты (если включено)
        const HeliosConfig& cfg = g_helios_config;
        double global_best_w = 1.0;
        if (cfg.afc.enable) {
            global_best_w = find_dominant_frequency_gpu(d_signal_hi.get(), d_signal_lo.get(), frame_size, cfg.afc);
            if (global_best_w <= 0.0) global_best_w = 1.0;
        }

        alw_build_basis_kernel<<<(frame_size + 255) / 256, 256>>>(
            d_Q_hi, d_Q_lo, global_best_w, 0.0, frame_size, g_num_poly, g_num_harm, g_num_morlet);
        CUDA_CHECK(cudaDeviceSynchronize());

        alw_vector<double> orig_T_hi(g_num_basis * frame_size);
        alw_vector<double> orig_T_lo(g_num_basis * frame_size);
        CUDA_CHECK(cudaMemcpy(orig_T_hi.data(), d_Q_hi, g_num_basis * frame_size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(orig_T_lo.data(), d_Q_lo, g_num_basis * frame_size * sizeof(double), cudaMemcpyDeviceToHost));

        CudaPoolGuard<double> d_orig_T_hi(g_num_basis * frame_size), d_orig_T_lo(g_num_basis * frame_size);
        CUDA_CHECK(cudaMemcpy(d_orig_T_hi.get(), orig_T_hi.data(), g_num_basis * frame_size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_orig_T_lo.get(), orig_T_lo.data(), g_num_basis * frame_size * sizeof(double), cudaMemcpyHostToDevice));

        if (!orthogonalize_basis_hybrid(d_Q_hi, d_Q_lo, d_R_hi, d_R_lo,
                                        d_orig_T_hi.get(), d_orig_T_lo.get(),
                                        g_num_basis, frame_size, params)) {
            ALW_LOG_ERROR("Orthogonalization failed!");
            g_ortho_initialized = false;
            return;
        }
        g_ortho_initialized = true;
    });
    return g_ortho_initialized;
}

// =============================================================================
// run_alw_helios_ortho — ПОЛНАЯ РЕАЛИЗАЦИЯ С EOD
// =============================================================================
void run_alw_helios_ortho(
    double* d_y_global_hi,
    double* d_y_global_lo,
    int N,
    int frame_size,
    int hop_size,
    double* d_dict_hi,
    double* d_dict_lo,
    int num_atoms,
    alw_vector<DetectedEvent>& all_events,
    alw_vector<double>& atom_norms,
    const alw_vector<AtomDescriptor>& descriptors,
    int max_iterations,
    double gamma_local,
    double gamma_global,
    double gamma_coherence,
    bool use_fixed_sigma,
    double fixed_sigma,
    bool no_final_regression,
    alw_fstream* residual_file,
    bool enable_ortho,
    alw_vector<alw_vector<double>>& out_coeffs_hi,
    alw_vector<alw_vector<double>>& out_coeffs_lo,
    cudaStream_t stream)
{
    if (N <= 0 || frame_size <= 0 || hop_size <= 0 || !d_y_global_hi || !d_y_global_lo ||
        !d_dict_hi || !d_dict_lo || num_atoms <= 0) {
        ALW_LOG_ERROR("run_alw_helios_ortho: invalid parameters");
        return;
    }

    if (frame_size > N) frame_size = N;
    if (hop_size > frame_size) hop_size = frame_size;

    // ===== EOD: обучение словаря на первых данных (если включено и ещё не обучено) =====
    if (g_helios_config.eod.enable && !g_eod_trained) {
        int learn_N = std::min(N, g_helios_config.eod.learn_samples > 0 ? g_helios_config.eod.learn_samples : 10000);
        if (learn_N > frame_size * 2) {
            ALW_LOG_INFO("EOD: Training dictionary inside run_alw_helios_ortho on %d samples", learn_N);
            train_dictionary_with_eod(
                d_y_global_hi,
                d_y_global_lo,
                learn_N,
                frame_size,
                hop_size,
                num_atoms,
                stream,
                d_dict_hi,
                d_dict_lo,
                atom_norms
            );
            // Обновляем глобальные указатели, если они указывают на другие адреса
            if (g_d_dict_hi_global != d_dict_hi && g_d_dict_hi_global != nullptr) {
                CUDA_CHECK(cudaMemcpy(g_d_dict_hi_global, d_dict_hi,
                                      (size_t)num_atoms * frame_size * sizeof(double),
                                      cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(g_d_dict_lo_global, d_dict_lo,
                                      (size_t)num_atoms * frame_size * sizeof(double),
                                      cudaMemcpyDeviceToDevice));
                g_atom_norms_global = atom_norms;
            }
        } else {
            ALW_LOG_WARN("EOD: Not enough samples for training (%d < %d), skipping.", learn_N, frame_size * 2);
        }
    }

    // Инициализация буферов EOP (один раз)
    if (!init_eop_buffers(num_atoms)) {
        ALW_LOG_ERROR("run_alw_helios_ortho: failed to initialize EOP buffers");
        return;
    }

    CudaPoolGuard<double> d_res_hi(frame_size), d_res_lo(frame_size);
    CudaPoolGuard<double> d_frame_hi(frame_size), d_frame_lo(frame_size);
    if (!d_res_hi.is_valid() || !d_res_lo.is_valid() || !d_frame_hi.is_valid() || !d_frame_lo.is_valid()) {
        ALW_LOG_ERROR("run_alw_helios_ortho: memory allocation for frame buffers failed");
        return;
    }

    out_coeffs_hi.clear();
    out_coeffs_lo.clear();
    out_coeffs_hi.reserve((N - frame_size) / hop_size + 1);
    out_coeffs_lo.reserve((N - frame_size) / hop_size + 1);

    for (int frame_start = 0; frame_start <= N - frame_size; frame_start += hop_size) {
        CUDA_CHECK(cudaMemcpyAsync(d_frame_hi.get(), d_y_global_hi + frame_start,
                                   frame_size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_frame_lo.get(), d_y_global_lo + frame_start,
                                   frame_size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (enable_ortho && g_ortho_initialized && g_d_Q_hi && g_d_Q_lo) {
            CudaPoolGuard<double> d_coeff_hi(g_num_basis), d_coeff_lo(g_num_basis);
            if (d_coeff_hi.is_valid() && d_coeff_lo.is_valid()) {
                size_t shmem = 2 * 256 * sizeof(double);
                projection_kernel<<<g_num_basis, 256, shmem, stream>>>(
                    g_d_Q_hi->get(), g_d_Q_lo->get(),
                    d_frame_hi.get(), d_frame_lo.get(),
                    d_coeff_hi.get(), d_coeff_lo.get(), g_num_basis, frame_size);
                CUDA_CHECK(cudaStreamSynchronize(stream));

                alw_vector<double> coeff_hi(g_num_basis), coeff_lo(g_num_basis);
                CUDA_CHECK(cudaMemcpyAsync(coeff_hi.data(), d_coeff_hi.get(),
                                           g_num_basis * sizeof(double), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(coeff_lo.data(), d_coeff_lo.get(),
                                           g_num_basis * sizeof(double), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                out_coeffs_hi.push_back(std::move(coeff_hi));
                out_coeffs_lo.push_back(std::move(coeff_lo));
            }
        }

        alw_vector<DetectedEvent> frame_events;
        dd_hapt_frame_gpu_full(
            d_frame_hi.get(),
            d_frame_lo.get(),
            d_res_hi.get(),
            d_res_lo.get(),
            d_dict_hi,
            d_dict_lo,
            num_atoms,
            frame_size,
            frame_start,
            frame_events,
            atom_norms,
            descriptors,
            max_iterations,
            gamma_local,
            gamma_global,
            gamma_coherence,
            use_fixed_sigma,
            fixed_sigma,
            no_final_regression,
            stream
        );

        all_events.insert(all_events.end(), frame_events.begin(), frame_events.end());

        if (residual_file && residual_file->is_open()) {
            alw_vector<double> res_hi(frame_size), res_lo(frame_size);
            CUDA_CHECK(cudaMemcpyAsync(res_hi.data(), d_res_hi.get(),
                                       frame_size * sizeof(double), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(res_lo.data(), d_res_lo.get(),
                                       frame_size * sizeof(double), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            residual_file->write(reinterpret_cast<const char*>(res_hi.data()), frame_size * sizeof(double));
            residual_file->write(reinterpret_cast<const char*>(res_lo.data()), frame_size * sizeof(double));
        }
    }
}

// =============================================================================
// ОБЁРТКИ ДЛЯ СОВМЕСТИМОСТИ
// =============================================================================

void run_alw_helios(
    double* d_y_global_hi,
    double* d_y_global_lo,
    int N,
    int frame_size,
    int hop_size,
    double* d_dict_hi,
    double* d_dict_lo,
    int num_atoms,
    alw_vector<DetectedEvent>& all_events,
    const alw_vector<double>& atom_norms,
    const alw_vector<AtomDescriptor>& descriptors,
    int max_iterations,
    double gamma_local,
    double gamma_global,
    double gamma_coherence,
    bool use_fixed_sigma,
    double fixed_sigma,
    bool no_final_regression,
    alw_fstream* residual_file,
    cudaStream_t stream)
{
    alw_vector<alw_vector<double>> coeffs_hi, coeffs_lo;
    alw_vector<double> norms = atom_norms;
    run_alw_helios_ortho(d_y_global_hi, d_y_global_lo,
                         N, frame_size, hop_size,
                         d_dict_hi, d_dict_lo, num_atoms,
                         all_events, norms, descriptors,
                         max_iterations,
                         gamma_local, gamma_global, gamma_coherence,
                         use_fixed_sigma, fixed_sigma, no_final_regression,
                         residual_file,
                         false,
                         coeffs_hi, coeffs_lo,
                         stream);
}

void helios_apply_detrend(
    const double* y_hi,
    const double* y_lo,
    int N,
    double* out_hi,
    double* out_lo,
    cudaStream_t stream)
{
    if (stream != 0) {
        CUDA_CHECK(cudaMemcpyAsync(out_hi, y_hi, N * sizeof(double), cudaMemcpyHostToDevice, stream));
        if (y_lo != nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(out_lo, y_lo, N * sizeof(double), cudaMemcpyHostToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemsetAsync(out_lo, 0, N * sizeof(double), stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaMemcpy(out_hi, y_hi, N * sizeof(double), cudaMemcpyHostToDevice));
        if (y_lo != nullptr) {
            CUDA_CHECK(cudaMemcpy(out_lo, y_lo, N * sizeof(double), cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMemset(out_lo, 0, N * sizeof(double)));
        }
    }
    if (stream != 0) {
        CUDA_CHECK(cudaMemcpyAsync(out_hi, y_hi, N * sizeof(double), cudaMemcpyDeviceToHost, stream));
        if (y_lo != nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(out_lo, y_lo, N * sizeof(double), cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaMemcpy(out_hi, y_hi, N * sizeof(double), cudaMemcpyDeviceToHost));
        if (y_lo != nullptr) {
            CUDA_CHECK(cudaMemcpy(out_lo, y_lo, N * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }
}
