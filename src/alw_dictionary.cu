// =============================================================================
// alw_dictionary.cu — атомарный словарь Helios
// Версия 6.2 — все ядра генерации атомов помечены __restrict__, добавлен __ldg()
// =============================================================================

#include "alw_math.h"
#include "alw_dictionary.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <mutex>
#include <list>
#include <unordered_map>
#include <memory>

// =============================================================================
// B-СПЛАЙНОВЫЕ ТАБЛИЦЫ (constant memory)
// =============================================================================

__constant__ double d_bspline_linear[1024];
__constant__ double d_bspline_quadratic[1024];
__constant__ double d_bspline_cubic[1024];

void init_bspline_constant_memory() {
    const int PROTO_LEN = 1024;
    double h_bspline_linear[1024], h_bspline_quadratic[1024], h_bspline_cubic[1024];

    for (int i = 0; i < PROTO_LEN; ++i) {
        double x = (double)i / (PROTO_LEN - 1);
        h_bspline_linear[i] = (x < 0.5) ? (2.0 * x) : (2.0 * (1.0 - x));
    }
    for (int i = 0; i < PROTO_LEN; ++i) {
        double x = (double)i / (PROTO_LEN - 1) * 2.0;
        double val = 0.0;
        if (x < 1.0) val = 0.5 * x * x;
        else if (x < 2.0) val = 0.5 * (2.0 - x) * (2.0 - x);
        h_bspline_quadratic[i] = val;
    }
    for (int i = 0; i < PROTO_LEN; ++i) {
        double x = -2.0 + 4.0 * i / (PROTO_LEN - 1);
        x = fabs(x);
        double val = 0.0;
        if (x < 1.0) val = (2.0 / 3.0) - x * x + 0.5 * x * x * x;
        else if (x < 2.0) val = (1.0 / 6.0) * pow(2.0 - x, 3.0);
        h_bspline_cubic[i] = val;
    }

    cudaMemcpyToSymbol(d_bspline_linear, h_bspline_linear, PROTO_LEN * sizeof(double));
    cudaMemcpyToSymbol(d_bspline_quadratic, h_bspline_quadratic, PROTO_LEN * sizeof(double));
    cudaMemcpyToSymbol(d_bspline_cubic, h_bspline_cubic, PROTO_LEN * sizeof(double));
    cudaDeviceSynchronize();
}

// =============================================================================
// ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ КЛИППИНГА
// =============================================================================

static double clamp_param(double value, double min_val, double max_val, const char* name) {
    if (std::isnan(value) || std::isinf(value) || value < min_val || value > max_val) {
        double clamped = (value < min_val) ? min_val : (value > max_val ? max_val : (min_val + max_val) * 0.5);
        return clamped;
    }
    return value;
}

// =============================================================================
// ШАБЛОННОЕ ЯДРО ДЛЯ ГЕНЕРАЦИИ АТОМОВ (CHIRP, DAMPED, MORLET) — с __restrict__
// =============================================================================

template <AtomType TYPE, bool USE_SIN>
__global__ void generate_atom_kernel(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    double p1_hi, double p1_lo,
    double p2_hi, double p2_lo,
    double p3_hi, double p3_lo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    double t_hi = (double)i / (N_frame - 1);
    double t_lo = 0.0;
    double val_hi = 0.0, val_lo = 0.0;

    if constexpr (TYPE == CHIRP) {
        double f0t_hi, f0t_lo;
        alw_mul_dd(p1_hi, p1_lo, t_hi, t_lo, f0t_hi, f0t_lo);
        double t2_hi, t2_lo;
        alw_mul_dd(t_hi, t_lo, t_hi, t_lo, t2_hi, t2_lo);
        double beta_t2_hi, beta_t2_lo;
        alw_mul_dd(p2_hi, p2_lo, t2_hi, t2_lo, beta_t2_hi, beta_t2_lo);
        double sum_hi, sum_lo;
        alw_add_dd(f0t_hi, f0t_lo, beta_t2_hi, beta_t2_lo, sum_hi, sum_lo);
        double phase_hi, phase_lo;
        alw_mul_dd(sum_hi, sum_lo, 2.0 * ALW_PI_HI, 0.0, phase_hi, phase_lo);
        double s_hi, s_lo, c_hi, c_lo;
        alw_sin_cos_dd(phase_hi, phase_lo, s_hi, s_lo, c_hi, c_lo);
        if constexpr (USE_SIN) {
            val_hi = s_hi; val_lo = s_lo;
        } else {
            val_hi = c_hi; val_lo = c_lo;
        }
    }
    else if constexpr (TYPE == DAMPED) {
        double neg_alpha_t_hi, neg_alpha_t_lo;
        alw_mul_dd(-p1_hi, -p1_lo, t_hi, t_lo, neg_alpha_t_hi, neg_alpha_t_lo);
        double env_hi, env_lo;
        alw_exp_dd(neg_alpha_t_hi, neg_alpha_t_lo, env_hi, env_lo);
        double arg_hi, arg_lo;
        alw_mul_dd(p2_hi, p2_lo, t_hi, t_lo, arg_hi, arg_lo);
        alw_mul_dd(arg_hi, arg_lo, 2.0 * ALW_PI_HI, 0.0, arg_hi, arg_lo);
        double s_hi, s_lo, c_hi, c_lo;
        alw_sin_cos_dd(arg_hi, arg_lo, s_hi, s_lo, c_hi, c_lo);
        double wave_hi, wave_lo;
        if constexpr (USE_SIN) {
            alw_mul_dd(env_hi, env_lo, s_hi, s_lo, wave_hi, wave_lo);
        } else {
            alw_mul_dd(env_hi, env_lo, c_hi, c_lo, wave_hi, wave_lo);
        }
        val_hi = wave_hi; val_lo = wave_lo;
    }
    else if constexpr (TYPE == MORLET) {
        double dt_hi, dt_lo;
        alw_sub_dd(t_hi, t_lo, p2_hi, p2_lo, dt_hi, dt_lo);
        double z_hi, z_lo;
        alw_div_dd(dt_hi, dt_lo, p1_hi, p1_lo, z_hi, z_lo);
        double z2_hi, z2_lo;
        alw_mul_dd(z_hi, z_lo, z_hi, z_lo, z2_hi, z2_lo);
        double neg_half_z2_hi, neg_half_z2_lo;
        alw_mul_dd(z2_hi, z2_lo, -0.5, 0.0, neg_half_z2_hi, neg_half_z2_lo);
        double gauss_hi, gauss_lo;
        alw_exp_dd(neg_half_z2_hi, neg_half_z2_lo, gauss_hi, gauss_lo);
        double arg_hi, arg_lo;
        alw_mul_dd(dt_hi, dt_lo, 2.0 * ALW_PI_HI * p3_hi, 0.0, arg_hi, arg_lo);
        double s_hi, s_lo, c_hi, c_lo;
        alw_sin_cos_dd(arg_hi, arg_lo, s_hi, s_lo, c_hi, c_lo);
        double wave_hi, wave_lo;
        if constexpr (USE_SIN) {
            alw_mul_dd(gauss_hi, gauss_lo, s_hi, s_lo, wave_hi, wave_lo);
        } else {
            alw_mul_dd(gauss_hi, gauss_lo, c_hi, c_lo, wave_hi, wave_lo);
        }
        val_hi = wave_hi; val_lo = wave_lo;
    }

    out_hi[i] = val_hi;
    out_lo[i] = val_lo;
}

// =============================================================================
// ОСТАЛЬНЫЕ ЯДРА ГЕНЕРАЦИИ С __restrict__
// =============================================================================

__global__ void generate_quasi_atom_kernel(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    int k,
    double theta_hi, double theta_lo,
    bool use_sin)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    if (isnan(theta_hi) || isinf(theta_hi) || isnan(theta_lo) || isinf(theta_lo)) {
        out_hi[i] = 0.0; out_lo[i] = 0.0;
        return;
    }

    double n_hi = (double)i, n_lo = 0.0;
    double th_pow_hi = 1.0, th_pow_lo = 0.0;
    for (int p = 0; p < k; ++p) {
        double tmp_hi, tmp_lo;
        alw_mul_dd(th_pow_hi, th_pow_lo, theta_hi, theta_lo, tmp_hi, tmp_lo);
        th_pow_hi = tmp_hi; th_pow_lo = tmp_lo;
    }
    double factor_hi, factor_lo;
    alw_mul_dd(th_pow_hi, th_pow_lo, 2.0 * ALW_PI_HI, 0.0, factor_hi, factor_lo);
    double arg_hi, arg_lo;
    alw_mul_dd(factor_hi, factor_lo, n_hi, n_lo, arg_hi, arg_lo);
    double norm_hi, norm_lo;
    alw_div_dd(arg_hi, arg_lo, (double)N_frame, 0.0, norm_hi, norm_lo);
    double s_hi, s_lo, c_hi, c_lo;
    alw_sin_cos_dd(norm_hi, norm_lo, s_hi, s_lo, c_hi, c_lo);
    if (use_sin) { out_hi[i] = s_hi; out_lo[i] = s_lo; }
    else         { out_hi[i] = c_hi; out_lo[i] = c_lo; }
}

__global__ void generate_step_atom_kernel(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    double position)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    if (isnan(position) || isinf(position)) {
        out_hi[i] = 0.0; out_lo[i] = 0.0;
        return;
    }

    double t = (double)i / (N_frame - 1);
    double val = (t >= position) ? 1.0 : 0.0;
    out_hi[i] = val;
    out_lo[i] = 0.0;
}

__global__ void generate_bspline_atom_kernel(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    double scale, double position, int order)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    if (isnan(scale) || isinf(scale) || isnan(position) || isinf(position) ||
        scale <= 0.0 || position < 0.0 || position > 1.0) {
        out_hi[i] = 0.0; out_lo[i] = 0.0;
        return;
    }

    double t = (double)i / (N_frame - 1);
    double x = (t - position) / scale;
    double idx = x * 1023.0;
    int idx0 = (int)floor(idx);
    int idx1 = idx0 + 1;
    if (idx0 < 0 || idx1 >= 1024) {
        out_hi[i] = 0.0; out_lo[i] = 0.0;
        return;
    }
    double frac = idx - idx0;
    double val = 0.0;
    if (order == 1) {
        val = (1.0 - frac) * d_bspline_linear[idx0] + frac * d_bspline_linear[idx1];
    } else if (order == 2) {
        val = (1.0 - frac) * d_bspline_quadratic[idx0] + frac * d_bspline_quadratic[idx1];
    } else {
        val = (1.0 - frac) * d_bspline_cubic[idx0] + frac * d_bspline_cubic[idx1];
    }
    out_hi[i] = val;
    out_lo[i] = 0.0;
}

__global__ void generate_anti_damped_atom_kernel(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    double alpha, double f0,
    bool use_sin)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    if (isnan(alpha) || isinf(alpha) || isnan(f0) || isinf(f0) || f0 < 0.0) {
        out_hi[i] = 0.0; out_lo[i] = 0.0;
        return;
    }

    double t_hi = (double)i / (N_frame - 1), t_lo = 0.0;
    double alpha_t_hi, alpha_t_lo;
    alw_mul_dd(alpha, 0.0, t_hi, t_lo, alpha_t_hi, alpha_t_lo);
    double env_hi, env_lo;
    alw_exp_dd(alpha_t_hi, alpha_t_lo, env_hi, env_lo);
    double arg_hi, arg_lo;
    alw_mul_dd(t_hi, t_lo, 2.0 * ALW_PI_HI * f0, 0.0, arg_hi, arg_lo);
    double s_hi, s_lo, c_hi, c_lo;
    alw_sin_cos_dd(arg_hi, arg_lo, s_hi, s_lo, c_hi, c_lo);
    double wave_hi, wave_lo;
    if (use_sin) {
        alw_mul_dd(env_hi, env_lo, s_hi, s_lo, wave_hi, wave_lo);
    } else {
        alw_mul_dd(env_hi, env_lo, c_hi, c_lo, wave_hi, wave_lo);
    }
    out_hi[i] = wave_hi;
    out_lo[i] = wave_lo;
}

__global__ void generate_sigmoid_atom_kernel(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    double k, double t0,
    bool direction_up)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    if (isnan(k) || isinf(k) || isnan(t0) || isinf(t0) || k <= 0.0 || t0 < 0.0 || t0 > 1.0) {
        out_hi[i] = 0.0; out_lo[i] = 0.0;
        return;
    }

    double t_hi = (double)i / (N_frame - 1), t_lo = 0.0;
    double diff_hi = t_hi - t0, diff_lo = 0.0;
    double neg_k_diff_hi, neg_k_diff_lo;
    alw_mul_dd(-k, 0.0, diff_hi, diff_lo, neg_k_diff_hi, neg_k_diff_lo);
    double exp_neg_hi, exp_neg_lo;
    alw_exp_dd(neg_k_diff_hi, neg_k_diff_lo, exp_neg_hi, exp_neg_lo);
    double denom_hi, denom_lo;
    alw_add_dd(1.0, 0.0, exp_neg_hi, exp_neg_lo, denom_hi, denom_lo);
    double inv_denom_hi, inv_denom_lo;
    alw_div_dd(1.0, 0.0, denom_hi, denom_lo, inv_denom_hi, inv_denom_lo);
    double val_hi, val_lo;
    if (direction_up) {
        val_hi = inv_denom_hi; val_lo = inv_denom_lo;
    } else {
        alw_sub_dd(1.0, 0.0, inv_denom_hi, inv_denom_lo, val_hi, val_lo);
    }
    out_hi[i] = val_hi;
    out_lo[i] = val_lo;
}

__global__ void generate_chebyshev_atom_kernel_dd(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    int order)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    double t_hi = (double)i / (N_frame - 1);
    double t_lo = 0.0;
    double x_hi = 2.0 * t_hi - 1.0;
    double x_lo = 0.0;

    double val_hi, val_lo;
    if (order == 0) {
        val_hi = 1.0; val_lo = 0.0;
    } else if (order == 1) {
        val_hi = x_hi; val_lo = x_lo;
    } else {
        double T0_hi = 1.0, T0_lo = 0.0;
        double T1_hi = x_hi, T1_lo = x_lo;
        double T2_hi, T2_lo;
        for (int n = 2; n <= order; ++n) {
            double prod_hi, prod_lo;
            alw_mul_dd(2.0, 0.0, T1_hi, T1_lo, prod_hi, prod_lo);
            alw_mul_dd(prod_hi, prod_lo, x_hi, x_lo, prod_hi, prod_lo);
            alw_sub_dd(prod_hi, prod_lo, T0_hi, T0_lo, T2_hi, T2_lo);
            T0_hi = T1_hi; T0_lo = T1_lo;
            T1_hi = T2_hi; T1_lo = T2_lo;
        }
        val_hi = T1_hi; val_lo = T1_lo;
    }
    out_hi[i] = val_hi;
    out_lo[i] = val_lo;
}

__global__ void generate_double_sigmoid_atom_kernel_dd(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    double k_hi, double k_lo,
    double t0_hi, double t0_lo,
    double t1_hi, double t1_lo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    double t_hi = (double)i / (N_frame - 1);
    double t_lo = 0.0;

    double diff0_hi, diff0_lo;
    alw_sub_dd(t_hi, t_lo, t0_hi, t0_lo, diff0_hi, diff0_lo);
    double neg_k_diff0_hi, neg_k_diff0_lo;
    alw_mul_dd(-k_hi, -k_lo, diff0_hi, diff0_lo, neg_k_diff0_hi, neg_k_diff0_lo);
    double exp0_hi, exp0_lo;
    alw_exp_dd(neg_k_diff0_hi, neg_k_diff0_lo, exp0_hi, exp0_lo);
    double denom0_hi, denom0_lo;
    alw_add_dd(1.0, 0.0, exp0_hi, exp0_lo, denom0_hi, denom0_lo);
    double sig1_hi, sig1_lo;
    alw_div_dd(1.0, 0.0, denom0_hi, denom0_lo, sig1_hi, sig1_lo);

    double diff1_hi, diff1_lo;
    alw_sub_dd(t_hi, t_lo, t1_hi, t1_lo, diff1_hi, diff1_lo);
    double neg_k_diff1_hi, neg_k_diff1_lo;
    alw_mul_dd(-k_hi, -k_lo, diff1_hi, diff1_lo, neg_k_diff1_hi, neg_k_diff1_lo);
    double exp1_hi, exp1_lo;
    alw_exp_dd(neg_k_diff1_hi, neg_k_diff1_lo, exp1_hi, exp1_lo);
    double denom1_hi, denom1_lo;
    alw_add_dd(1.0, 0.0, exp1_hi, exp1_lo, denom1_hi, denom1_lo);
    double sig2_hi, sig2_lo;
    alw_div_dd(1.0, 0.0, denom1_hi, denom1_lo, sig2_hi, sig2_lo);

    double val_hi, val_lo;
    alw_sub_dd(sig1_hi, sig1_lo, sig2_hi, sig2_lo, val_hi, val_lo);
    out_hi[i] = val_hi;
    out_lo[i] = val_lo;
}

__global__ void generate_gaussian_atom_kernel_dd(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    double mu_hi, double mu_lo,
    double sigma_hi, double sigma_lo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    double t_hi = (double)i / (N_frame - 1);
    double t_lo = 0.0;

    double diff_hi, diff_lo;
    alw_sub_dd(t_hi, t_lo, mu_hi, mu_lo, diff_hi, diff_lo);
    double z_hi, z_lo;
    alw_div_dd(diff_hi, diff_lo, sigma_hi, sigma_lo, z_hi, z_lo);
    double z2_hi, z2_lo;
    alw_mul_dd(z_hi, z_lo, z_hi, z_lo, z2_hi, z2_lo);
    double neg_half_z2_hi, neg_half_z2_lo;
    alw_mul_dd(z2_hi, z2_lo, -0.5, 0.0, neg_half_z2_hi, neg_half_z2_lo);
    double val_hi, val_lo;
    alw_exp_dd(neg_half_z2_hi, neg_half_z2_lo, val_hi, val_lo);
    out_hi[i] = val_hi;
    out_lo[i] = val_lo;
}

__global__ void generate_erf_atom_kernel_dd(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    double mu_hi, double mu_lo,
    double sigma_hi, double sigma_lo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    double t_hi = (double)i / (N_frame - 1);
    double t_lo = 0.0;

    double diff_hi, diff_lo;
    alw_sub_dd(t_hi, t_lo, mu_hi, mu_lo, diff_hi, diff_lo);
    double sqrt2_hi = 1.4142135623730951, sqrt2_lo = 0.0;
    double denom_hi, denom_lo;
    alw_mul_dd(sigma_hi, sigma_lo, sqrt2_hi, sqrt2_lo, denom_hi, denom_lo);
    double x_hi, x_lo;
    alw_div_dd(diff_hi, diff_lo, denom_hi, denom_lo, x_hi, x_lo);

    double x2_hi, x2_lo;
    alw_mul_dd(x_hi, x_lo, x_hi, x_lo, x2_hi, x2_lo);
    double a_hi = 0.147, a_lo = 0.0;
    double ax2_hi, ax2_lo;
    alw_mul_dd(a_hi, a_lo, x2_hi, x2_lo, ax2_hi, ax2_lo);
    double four_pi_hi = 1.2732395447351627, four_pi_lo = 0.0;
    double num_hi, num_lo;
    alw_add_dd(four_pi_hi, four_pi_lo, ax2_hi, ax2_lo, num_hi, num_lo);
    alw_mul_dd(num_hi, num_lo, x2_hi, x2_lo, num_hi, num_lo);
    double denom_erf_hi, denom_erf_lo;
    alw_add_dd(1.0, 0.0, ax2_hi, ax2_lo, denom_erf_hi, denom_erf_lo);
    double ratio_hi, ratio_lo;
    alw_div_dd(num_hi, num_lo, denom_erf_hi, denom_erf_lo, ratio_hi, ratio_lo);
    double neg_ratio_hi, neg_ratio_lo;
    alw_mul_dd(-1.0, 0.0, ratio_hi, ratio_lo, neg_ratio_hi, neg_ratio_lo);
    double exp_neg_hi, exp_neg_lo;
    alw_exp_dd(neg_ratio_hi, neg_ratio_lo, exp_neg_hi, exp_neg_lo);
    double one_minus_exp_hi, one_minus_exp_lo;
    alw_sub_dd(1.0, 0.0, exp_neg_hi, exp_neg_lo, one_minus_exp_hi, one_minus_exp_lo);
    double sqrt_val_hi, sqrt_val_lo;
    alw_sqrt_dd(one_minus_exp_hi, one_minus_exp_lo, sqrt_val_hi, sqrt_val_lo);
    double sign = (x_hi >= 0.0) ? 1.0 : -1.0;
    double erf_hi = sign * sqrt_val_hi;
    double erf_lo = 0.0;

    double val_hi, val_lo;
    alw_mul_dd(0.5, 0.0, 1.0 + erf_hi, 0.0, val_hi, val_lo);
    out_hi[i] = val_hi;
    out_lo[i] = val_lo;
}

__global__ void generate_damped_chirp_atom_kernel_dd(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    double alpha_hi, double alpha_lo,
    double f0_hi, double f0_lo,
    double beta_hi, double beta_lo,
    bool use_sin)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    double t_hi = (double)i / (N_frame - 1);
    double t_lo = 0.0;

    double neg_alpha_t_hi, neg_alpha_t_lo;
    alw_mul_dd(-alpha_hi, -alpha_lo, t_hi, t_lo, neg_alpha_t_hi, neg_alpha_t_lo);
    double env_hi, env_lo;
    alw_exp_dd(neg_alpha_t_hi, neg_alpha_t_lo, env_hi, env_lo);

    double f0t_hi, f0t_lo;
    alw_mul_dd(f0_hi, f0_lo, t_hi, t_lo, f0t_hi, f0t_lo);
    double t2_hi, t2_lo;
    alw_mul_dd(t_hi, t_lo, t_hi, t_lo, t2_hi, t2_lo);
    double half_beta_t2_hi, half_beta_t2_lo;
    alw_mul_dd(beta_hi, beta_lo, t2_hi, t2_lo, half_beta_t2_hi, half_beta_t2_lo);
    alw_mul_dd(half_beta_t2_hi, half_beta_t2_lo, 0.5, 0.0, half_beta_t2_hi, half_beta_t2_lo);
    double sum_hi, sum_lo;
    alw_add_dd(f0t_hi, f0t_lo, half_beta_t2_hi, half_beta_t2_lo, sum_hi, sum_lo);
    double phase_hi, phase_lo;
    alw_mul_dd(sum_hi, sum_lo, 2.0 * ALW_PI_HI, 0.0, phase_hi, phase_lo);

    double s_hi, s_lo, c_hi, c_lo;
    alw_sin_cos_dd(phase_hi, phase_lo, s_hi, s_lo, c_hi, c_lo);
    double wave_hi, wave_lo;
    if (use_sin) {
        alw_mul_dd(env_hi, env_lo, s_hi, s_lo, wave_hi, wave_lo);
    } else {
        alw_mul_dd(env_hi, env_lo, c_hi, c_lo, wave_hi, wave_lo);
    }
    out_hi[i] = wave_hi;
    out_lo[i] = wave_lo;
}

__global__ void generate_sigmoid_osc_atom_kernel_dd(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    double k_hi, double k_lo,
    double t0_hi, double t0_lo,
    double f_hi, double f_lo,
    bool use_sin)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    double t_hi = (double)i / (N_frame - 1);
    double t_lo = 0.0;

    double diff_hi, diff_lo;
    alw_sub_dd(t_hi, t_lo, t0_hi, t0_lo, diff_hi, diff_lo);
    double neg_k_diff_hi, neg_k_diff_lo;
    alw_mul_dd(-k_hi, -k_lo, diff_hi, diff_lo, neg_k_diff_hi, neg_k_diff_lo);
    double exp_hi, exp_lo;
    alw_exp_dd(neg_k_diff_hi, neg_k_diff_lo, exp_hi, exp_lo);
    double denom_hi, denom_lo;
    alw_add_dd(1.0, 0.0, exp_hi, exp_lo, denom_hi, denom_lo);
    double sig_hi, sig_lo;
    alw_div_dd(1.0, 0.0, denom_hi, denom_lo, sig_hi, sig_lo);

    double phase_hi, phase_lo;
    alw_mul_dd(f_hi, f_lo, t_hi, t_lo, phase_hi, phase_lo);
    alw_mul_dd(phase_hi, phase_lo, 2.0 * ALW_PI_HI, 0.0, phase_hi, phase_lo);
    double s_hi, s_lo, c_hi, c_lo;
    alw_sin_cos_dd(phase_hi, phase_lo, s_hi, s_lo, c_hi, c_lo);
    double wave_hi, wave_lo;
    if (use_sin) {
        alw_mul_dd(sig_hi, sig_lo, s_hi, s_lo, wave_hi, wave_lo);
    } else {
        alw_mul_dd(sig_hi, sig_lo, c_hi, c_lo, wave_hi, wave_lo);
    }
    out_hi[i] = wave_hi;
    out_lo[i] = wave_lo;
}

__global__ void generate_exp_growth_atom_kernel_dd(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    double alpha_hi, double alpha_lo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    double t_hi = (double)i / (N_frame - 1);
    double t_lo = 0.0;
    double alpha_t_hi, alpha_t_lo;
    alw_mul_dd(alpha_hi, alpha_lo, t_hi, t_lo, alpha_t_hi, alpha_t_lo);
    double val_hi, val_lo;
    alw_exp_dd(alpha_t_hi, alpha_t_lo, val_hi, val_lo);
    out_hi[i] = val_hi;
    out_lo[i] = val_lo;
}

__global__ void generate_exp_decay_atom_kernel_dd(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    double alpha_hi, double alpha_lo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    double t_hi = (double)i / (N_frame - 1);
    double t_lo = 0.0;
    double neg_alpha_t_hi, neg_alpha_t_lo;
    alw_mul_dd(-alpha_hi, -alpha_lo, t_hi, t_lo, neg_alpha_t_hi, neg_alpha_t_lo);
    double val_hi, val_lo;
    alw_exp_dd(neg_alpha_t_hi, neg_alpha_t_lo, val_hi, val_lo);
    out_hi[i] = val_hi;
    out_lo[i] = val_lo;
}

__global__ void generate_tanh_atom_kernel_dd(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    double k_hi, double k_lo,
    double t0_hi, double t0_lo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    double t_hi = (double)i / (N_frame - 1);
    double t_lo = 0.0;
    double diff_hi, diff_lo;
    alw_sub_dd(t_hi, t_lo, t0_hi, t0_lo, diff_hi, diff_lo);
    double k_diff_hi, k_diff_lo;
    alw_mul_dd(k_hi, k_lo, diff_hi, diff_lo, k_diff_hi, k_diff_lo);
    double two_x_hi, two_x_lo;
    alw_mul_dd(k_diff_hi, k_diff_lo, 2.0, 0.0, two_x_hi, two_x_lo);
    double exp_hi, exp_lo;
    alw_exp_dd(two_x_hi, two_x_lo, exp_hi, exp_lo);
    double num_hi, num_lo;
    alw_sub_dd(exp_hi, exp_lo, 1.0, 0.0, num_hi, num_lo);
    double denom_hi, denom_lo;
    alw_add_dd(exp_hi, exp_lo, 1.0, 0.0, denom_hi, denom_lo);
    double val_hi, val_lo;
    alw_div_dd(num_hi, num_lo, denom_hi, denom_lo, val_hi, val_lo);
    out_hi[i] = val_hi;
    out_lo[i] = val_lo;
}

__global__ void generate_lorentzian_atom_kernel_dd(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    double mu_hi, double mu_lo,
    double gamma_hi, double gamma_lo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    double t_hi = (double)i / (N_frame - 1);
    double t_lo = 0.0;
    double diff_hi, diff_lo;
    alw_sub_dd(t_hi, t_lo, mu_hi, mu_lo, diff_hi, diff_lo);
    double diff2_hi, diff2_lo;
    alw_mul_dd(diff_hi, diff_lo, diff_hi, diff_lo, diff2_hi, diff2_lo);
    double gamma2_hi, gamma2_lo;
    alw_mul_dd(gamma_hi, gamma_lo, gamma_hi, gamma_lo, gamma2_hi, gamma2_lo);
    double denom_hi, denom_lo;
    alw_add_dd(diff2_hi, diff2_lo, gamma2_hi, gamma2_lo, denom_hi, denom_lo);
    double val_hi, val_lo;
    alw_div_dd(gamma_hi, gamma_lo, denom_hi, denom_lo, val_hi, val_lo);
    out_hi[i] = val_hi;
    out_lo[i] = val_lo;
}

__global__ void generate_power_atom_kernel_dd(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    double alpha_hi, double alpha_lo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    double t_hi = (double)i / (N_frame - 1);
    double t_lo = 0.0;
    if (t_hi < 1e-12) t_hi = 1e-12;

    double log_t_hi, log_t_lo;
    alw_log_dd(t_hi, t_lo, log_t_hi, log_t_lo);
    double alpha_log_t_hi, alpha_log_t_lo;
    alw_mul_dd(alpha_hi, alpha_lo, log_t_hi, log_t_lo, alpha_log_t_hi, alpha_log_t_lo);
    double val_hi, val_lo;
    alw_exp_dd(alpha_log_t_hi, alpha_log_t_lo, val_hi, val_lo);
    out_hi[i] = val_hi;
    out_lo[i] = val_lo;
}

__global__ void generate_haar_atom_kernel_dd(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;
    double t = (double)i / (N_frame - 1);
    out_hi[i] = (t < 0.5) ? 1.0 : -1.0;
    out_lo[i] = 0.0;
}

__global__ void generate_bessel_atom_kernel_dd(
    double* __restrict__ out_hi,
    double* __restrict__ out_lo,
    int N_frame,
    int order)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    double x_hi = (double)i / (N_frame - 1) * 20.0;
    double x_lo = 0.0;
    double x2_hi, x2_lo;
    alw_mul_dd(x_hi, x_lo, x_hi, x_lo, x2_hi, x2_lo);

    double term_hi, term_lo;
    double sum_hi = 0.0, sum_lo = 0.0;
    double x_half_hi, x_half_lo;
    alw_mul_dd(x_hi, x_lo, 0.5, 0.0, x_half_hi, x_half_lo);
    double pow_hi = 1.0, pow_lo = 0.0;
    for (int p = 0; p < order; ++p) {
        alw_mul_dd(pow_hi, pow_lo, x_half_hi, x_half_lo, pow_hi, pow_lo);
    }
    double fact_hi = 1.0, fact_lo = 0.0;
    for (int f = 1; f <= order; ++f) {
        alw_mul_dd(fact_hi, fact_lo, (double)f, 0.0, fact_hi, fact_lo);
    }
    alw_div_dd(pow_hi, pow_lo, fact_hi, fact_lo, term_hi, term_lo);
    sum_hi = term_hi; sum_lo = term_lo;

    for (int k = 1; k <= 20; ++k) {
        double neg_x2_hi, neg_x2_lo;
        alw_mul_dd(-1.0, 0.0, x2_hi, x2_lo, neg_x2_hi, neg_x2_lo);
        alw_mul_dd(neg_x2_hi, neg_x2_lo, 0.25, 0.0, neg_x2_hi, neg_x2_lo);
        double denom_hi, denom_lo;
        alw_mul_dd((double)k, 0.0, (double)(k + order), 0.0, denom_hi, denom_lo);
        double factor_hi, factor_lo;
        alw_div_dd(neg_x2_hi, neg_x2_lo, denom_hi, denom_lo, factor_hi, factor_lo);
        alw_mul_dd(term_hi, term_lo, factor_hi, factor_lo, term_hi, term_lo);
        alw_add_dd(sum_hi, sum_lo, term_hi, term_lo, sum_hi, sum_lo);
        if (term_hi < 1e-30) break;
    }
    out_hi[i] = sum_hi;
    out_lo[i] = sum_lo;
}

// =============================================================================
// НОРМАЛИЗАЦИЯ АТОМА (DD)
// =============================================================================

__global__ void compute_norm_sq_kernel(
    const double* __restrict__ atom_hi,
    const double* __restrict__ atom_lo,
    int N_frame, double* __restrict__ norm_sq_out)
{
    int tid = threadIdx.x;
    double sum_hi = 0.0, sum_lo = 0.0;
    for (int i = tid; i < N_frame; i += blockDim.x) {
        double a_hi = atom_hi[i], a_lo = atom_lo[i];
        if (isnan(a_hi) || isinf(a_hi)) a_hi = 0.0;
        if (isnan(a_lo) || isinf(a_lo)) a_lo = 0.0;
        double p_hi, p_lo;
        alw_mul_dd(a_hi, a_lo, a_hi, a_lo, p_hi, p_lo);
        alw_add_dd(sum_hi, sum_lo, p_hi, p_lo, sum_hi, sum_lo);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        double th_hi = __shfl_down_sync(0xffffffff, sum_hi, offset);
        double th_lo = __shfl_down_sync(0xffffffff, sum_lo, offset);
        alw_add_dd(sum_hi, sum_lo, th_hi, th_lo, sum_hi, sum_lo);
    }
    __shared__ double warp_sums_hi[32], warp_sums_lo[32];
    int lane = tid & 31;
    int wid = tid / 32;
    if (lane == 0) {
        warp_sums_hi[wid] = sum_hi;
        warp_sums_lo[wid] = sum_lo;
    }
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    if (tid < num_warps) {
        sum_hi = warp_sums_hi[tid];
        sum_lo = warp_sums_lo[tid];
    } else {
        sum_hi = 0.0; sum_lo = 0.0;
    }
    if (tid < 32) {
        for (int offset = 16; offset > 0; offset /= 2) {
            double th_hi = __shfl_down_sync(0xffffffff, sum_hi, offset);
            double th_lo = __shfl_down_sync(0xffffffff, sum_lo, offset);
            alw_add_dd(sum_hi, sum_lo, th_hi, th_lo, sum_hi, sum_lo);
        }
    }
    if (tid == 0) {
        norm_sq_out[0] = sum_hi;
        norm_sq_out[1] = sum_lo;
    }
}

__global__ void normalize_atom_kernel(
    double* __restrict__ atom_hi,
    double* __restrict__ atom_lo,
    int N_frame,
    double norm_hi, double norm_lo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_frame) return;

    if (isnan(norm_hi) || isinf(norm_hi) || norm_hi < 1e-12) {
        atom_hi[i] = (i == 0) ? 1.0 : 0.0;
        atom_lo[i] = 0.0;
        return;
    }
    double new_hi, new_lo;
    alw_div_dd(atom_hi[i], atom_lo[i], norm_hi, norm_lo, new_hi, new_lo);
    if (isnan(new_hi) || isinf(new_hi)) {
        atom_hi[i] = (i == 0) ? 1.0 : 0.0;
        atom_lo[i] = 0.0;
    } else {
        atom_hi[i] = new_hi;
        atom_lo[i] = new_lo;
    }
}

double normalize_atom_gpu(double* d_atom_hi, double* d_atom_lo, int frame_size) {
    double* d_norm_sq;
    cudaMalloc(&d_norm_sq, 2 * sizeof(double));
    int threads = 256;

    compute_norm_sq_kernel<<<1, threads>>>(d_atom_hi, d_atom_lo, frame_size, d_norm_sq);
    cudaDeviceSynchronize();

    double norm_sq_hi, norm_sq_lo;
    cudaMemcpy(&norm_sq_hi, d_norm_sq, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&norm_sq_lo, d_norm_sq + 1, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_norm_sq);

    if (isnan(norm_sq_hi) || isinf(norm_sq_hi) || norm_sq_hi < 1e-12) {
        double h_atom_hi[256] = {0.0};
        h_atom_hi[0] = 1.0;
        cudaMemcpy(d_atom_hi, h_atom_hi, frame_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_atom_lo, 0, frame_size * sizeof(double));
        return 1.0;
    }

    double norm_hi, norm_lo;
    alw_sqrt_dd(norm_sq_hi, norm_sq_lo, norm_hi, norm_lo);

    int blocks = (frame_size + threads - 1) / threads;
    normalize_atom_kernel<<<blocks, threads>>>(d_atom_hi, d_atom_lo, frame_size, norm_hi, norm_lo);
    cudaDeviceSynchronize();

    return norm_hi;
}

// =============================================================================
// ВАЛИДАЦИЯ СЛОВАРЯ
// =============================================================================

__global__ void validate_dictionary_kernel(
    const double* __restrict__ dict_hi,
    const double* __restrict__ dict_lo,
    int num_atoms, int frame_size, int* __restrict__ bad_flags)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_atoms * frame_size;
    if (idx >= total) return;
    double val = dict_hi[idx];
    if (isnan(val) || isinf(val)) atomicAdd(bad_flags, 1);
}

// =============================================================================
// ПОСТРОЕНИЕ СЛОВАРЯ НА GPU (использует AtomSpec)
// =============================================================================

void build_dictionary_gpu(
    const alw_vector<AtomSpec>& descriptors,
    int frame_size,
    double* d_dict_hi, double* d_dict_lo,
    alw_vector<double>& atom_norms)
{
    if (frame_size < 2) {
        throw std::runtime_error("build_dictionary_gpu: frame_size must be at least 2");
    }

    int threads = 256;
    int blocks = (frame_size + threads - 1) / threads;
    int num_atoms = (int)descriptors.size();

    atom_norms.clear();
    atom_norms.resize(num_atoms, 0.0);

    for (int a = 0; a < num_atoms; ++a) {
        AtomSpec desc = descriptors[a];
        double* atom_hi = d_dict_hi + a * frame_size;
        double* atom_lo = d_dict_lo + a * frame_size;
        bool use_sin = desc.is_pair ? (a % 2 == 0) : true;

        switch (desc.type) {
            case CHIRP: {
                desc.chirp.f0_hi = clamp_param(desc.chirp.f0_hi, 0.001, 0.49, "chirp.f0");
                desc.chirp.beta_hi = clamp_param(desc.chirp.beta_hi, -0.01, 0.01, "chirp.beta");
                desc.chirp.f0_lo = 0.0;
                desc.chirp.beta_lo = 0.0;
                if (use_sin) {
                    generate_atom_kernel<CHIRP, true><<<blocks, threads>>>(
                        atom_hi, atom_lo, frame_size,
                        desc.chirp.f0_hi, desc.chirp.f0_lo,
                        desc.chirp.beta_hi, desc.chirp.beta_lo,
                        0.0, 0.0);
                } else {
                    generate_atom_kernel<CHIRP, false><<<blocks, threads>>>(
                        atom_hi, atom_lo, frame_size,
                        desc.chirp.f0_hi, desc.chirp.f0_lo,
                        desc.chirp.beta_hi, desc.chirp.beta_lo,
                        0.0, 0.0);
                }
                break;
            }
            case DAMPED: {
                desc.damped.alpha = clamp_param(desc.damped.alpha, 0.0, 5.0, "damped.alpha");
                desc.damped.f0 = clamp_param(desc.damped.f0, 0.001, 0.49, "damped.f0");
                if (use_sin) {
                    generate_atom_kernel<DAMPED, true><<<blocks, threads>>>(
                        atom_hi, atom_lo, frame_size,
                        desc.damped.alpha, 0.0,
                        desc.damped.f0, 0.0,
                        0.0, 0.0);
                } else {
                    generate_atom_kernel<DAMPED, false><<<blocks, threads>>>(
                        atom_hi, atom_lo, frame_size,
                        desc.damped.alpha, 0.0,
                        desc.damped.f0, 0.0,
                        0.0, 0.0);
                }
                break;
            }
            case MORLET: {
                desc.morlet.sigma = clamp_param(desc.morlet.sigma, 0.001, 0.5, "morlet.sigma");
                desc.morlet.mu = clamp_param(desc.morlet.mu, 0.0, 1.0, "morlet.mu");
                desc.morlet.w_morlet = clamp_param(desc.morlet.w_morlet, 1.0, 50.0, "morlet.w");
                if (use_sin) {
                    generate_atom_kernel<MORLET, true><<<blocks, threads>>>(
                        atom_hi, atom_lo, frame_size,
                        desc.morlet.sigma, 0.0,
                        desc.morlet.mu, 0.0,
                        desc.morlet.w_morlet, 0.0);
                } else {
                    generate_atom_kernel<MORLET, false><<<blocks, threads>>>(
                        atom_hi, atom_lo, frame_size,
                        desc.morlet.sigma, 0.0,
                        desc.morlet.mu, 0.0,
                        desc.morlet.w_morlet, 0.0);
                }
                break;
            }
            case QUASI: {
                desc.quasi.theta_hi = clamp_param(desc.quasi.theta_hi, -10.0, 10.0, "quasi.theta");
                desc.quasi.theta_lo = 0.0;
                generate_quasi_atom_kernel<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size,
                    desc.quasi.k, desc.quasi.theta_hi, desc.quasi.theta_lo, use_sin);
                break;
            }
            case STEP: {
                desc.step.position = clamp_param(desc.step.position, 0.0, 1.0, "step.position");
                generate_step_atom_kernel<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size, desc.step.position);
                break;
            }
            case BSPLINE: {
                desc.bspline.scale = clamp_param(desc.bspline.scale, 0.01, 2.0, "bspline.scale");
                desc.bspline.position = clamp_param(desc.bspline.position, 0.0, 1.0, "bspline.position");
                if (desc.bspline.order < 1 || desc.bspline.order > 3) {
                    desc.bspline.order = 3;
                }
                generate_bspline_atom_kernel<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size,
                    desc.bspline.scale, desc.bspline.position, desc.bspline.order);
                break;
            }
            case ANTI_DAMPED: {
                desc.anti_damped.alpha = clamp_param(desc.anti_damped.alpha, 0.0, 5.0, "anti_damped.alpha");
                desc.anti_damped.f0 = clamp_param(desc.anti_damped.f0, 0.001, 0.49, "anti_damped.f0");
                generate_anti_damped_atom_kernel<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size,
                    desc.anti_damped.alpha, desc.anti_damped.f0, use_sin);
                break;
            }
            case SIGMOID: {
                desc.sigmoid.k = clamp_param(desc.sigmoid.k, 0.1, 50.0, "sigmoid.k");
                desc.sigmoid.t0 = clamp_param(desc.sigmoid.t0, 0.0, 1.0, "sigmoid.t0");
                generate_sigmoid_atom_kernel<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size,
                    desc.sigmoid.k, desc.sigmoid.t0, desc.sigmoid.direction);
                break;
            }
            case CHEBYSHEV: {
                int order = (int)desc.quasi.k;
                if (order < 1) order = 1;
                if (order > 5) order = 5;
                generate_chebyshev_atom_kernel_dd<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size, order);
                break;
            }
            case DOUBLE_SIGMOID: {
                generate_double_sigmoid_atom_kernel_dd<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size,
                    desc.sigmoid.k, 0.0,
                    desc.sigmoid.t0, 0.0,
                    desc.sigmoid.t1, 0.0);
                break;
            }
            case GAUSSIAN: {
                generate_gaussian_atom_kernel_dd<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size,
                    desc.morlet.mu, 0.0,
                    desc.morlet.sigma, 0.0);
                break;
            }
            case ERF: {
                generate_erf_atom_kernel_dd<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size,
                    desc.morlet.mu, 0.0,
                    desc.morlet.sigma, 0.0);
                break;
            }
            case DAMPED_CHIRP: {
                generate_damped_chirp_atom_kernel_dd<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size,
                    desc.damped.alpha, 0.0,
                    desc.chirp.f0_hi, 0.0,
                    desc.chirp.beta_hi, 0.0,
                    use_sin);
                break;
            }
            case SIGMOID_OSC: {
                generate_sigmoid_osc_atom_kernel_dd<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size,
                    desc.sigmoid.k, 0.0,
                    desc.sigmoid.t0, 0.0,
                    desc.chirp.f0_hi, 0.0,
                    use_sin);
                break;
            }
            case EXP_GROWTH: {
                generate_exp_growth_atom_kernel_dd<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size,
                    desc.damped.alpha, 0.0);
                break;
            }
            case EXP_DECAY: {
                generate_exp_decay_atom_kernel_dd<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size,
                    desc.damped.alpha, 0.0);
                break;
            }
            case TANH: {
                generate_tanh_atom_kernel_dd<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size,
                    desc.sigmoid.k, 0.0,
                    desc.sigmoid.t0, 0.0);
                break;
            }
            case LORENTZIAN: {
                generate_lorentzian_atom_kernel_dd<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size,
                    desc.morlet.mu, 0.0,
                    desc.morlet.sigma, 0.0);
                break;
            }
            case POWER: {
                generate_power_atom_kernel_dd<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size,
                    desc.quasi.theta_hi, 0.0);
                break;
            }
            case HAAR: {
                generate_haar_atom_kernel_dd<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size);
                break;
            }
            case BESSEL: {
                int order = (int)desc.quasi.k;
                if (order < 0) order = 0;
                if (order > 5) order = 5;
                generate_bessel_atom_kernel_dd<<<blocks, threads>>>(
                    atom_hi, atom_lo, frame_size, order);
                break;
            }
            default: {
                cudaMemset(atom_hi, 0, frame_size * sizeof(double));
                cudaMemset(atom_lo, 0, frame_size * sizeof(double));
                break;
            }
        }
        cudaDeviceSynchronize();

        double norm_before = normalize_atom_gpu(atom_hi, atom_lo, frame_size);
        atom_norms[a] = norm_before;
    }

    // Валидация всего словаря
    int* d_bad;
    cudaMalloc(&d_bad, sizeof(int));
    cudaMemset(d_bad, 0, sizeof(int));
    int total_elements = num_atoms * frame_size;
    int val_threads = 256;
    int val_blocks = (total_elements + val_threads - 1) / val_threads;
    validate_dictionary_kernel<<<val_blocks, val_threads>>>(
        d_dict_hi, d_dict_lo, num_atoms, frame_size, d_bad);
    cudaDeviceSynchronize();
    int bad_count_host;
    cudaMemcpy(&bad_count_host, d_bad, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_bad);
}

// =============================================================================
// ЗАГРУЗКА ОБУЧЕННОГО СЛОВАРЯ ИЗ БУФЕРА
// =============================================================================

void load_learned_dictionary_from_buffer(
    const void* buffer,
    size_t size,
    int expected_num_atoms,
    int expected_frame_size,
    double* d_dict_hi,
    double* d_dict_lo,
    alw_vector<double>& atom_norms)
{
    if (!buffer || size < 16) return;

    const uint8_t* data = static_cast<const uint8_t*>(buffer);
    uint32_t magic = *reinterpret_cast<const uint32_t*>(data); data += 4;
    if (magic != 0x4C454152) return;
    uint32_t version = *reinterpret_cast<const uint32_t*>(data); data += 4;
    if (version != 1) return;
    uint32_t num_atoms = *reinterpret_cast<const uint32_t*>(data); data += 4;
    uint32_t frame_size = *reinterpret_cast<const uint32_t*>(data); data += 4;

    if (num_atoms != (uint32_t)expected_num_atoms || frame_size != (uint32_t)expected_frame_size) return;

    size_t expected_size = 4 * sizeof(uint32_t) + num_atoms * frame_size * sizeof(double) + num_atoms * sizeof(double);
    if (size < expected_size) return;

    const double* dict_data = reinterpret_cast<const double*>(data);
    cudaMemcpy(d_dict_hi, dict_data, num_atoms * frame_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_dict_lo, 0, num_atoms * frame_size * sizeof(double));
    data += num_atoms * frame_size * sizeof(double);

    const double* norms_data = reinterpret_cast<const double*>(data);
    atom_norms.resize(num_atoms);
    std::memcpy(atom_norms.data(), norms_data, num_atoms * sizeof(double));
}

// =============================================================================
// СОЗДАНИЕ ПОЛНОГО СЛОВАРЯ
// =============================================================================

alw_vector<AtomSpec> create_full_dictionary() {
    alw_vector<AtomSpec> descriptors;

    double chirp_freqs[] = {0.01, 0.03, 0.06, 0.1, 0.2, 0.4, 0.7};
    double chirp_betas[] = {-0.005, -0.001, 0.0, 0.001, 0.005};
    for (int f = 0; f < 7; ++f) {
        for (int b = 0; b < 5; ++b) {
            AtomSpec d;
            d.type = CHIRP;
            d.is_pair = true;
            d.chirp.f0_hi = chirp_freqs[f];
            d.chirp.f0_lo = 0.0;
            d.chirp.beta_hi = chirp_betas[b];
            d.chirp.beta_lo = 0.0;
            descriptors.push_back(d);
            descriptors.push_back(d);
        }
    }

    double morlet_sigmas[] = {0.03, 0.07, 0.15};
    double morlet_mus[] = {0.2, 0.4, 0.6, 0.8};
    double morlet_ws[] = {10.0, 18.0, 26.0};
    for (int s = 0; s < 3; ++s) {
        for (int m = 0; m < 4; ++m) {
            for (int w = 0; w < 3; ++w) {
                AtomSpec d;
                d.type = MORLET;
                d.is_pair = true;
                d.morlet.sigma = morlet_sigmas[s];
                d.morlet.mu = morlet_mus[m];
                d.morlet.w_morlet = morlet_ws[w];
                descriptors.push_back(d);
                descriptors.push_back(d);
            }
        }
    }

    double damped_alphas[] = {0.05, 0.15, 0.3, 0.6, 1.2};
    double damped_freqs[] = {0.03, 0.06, 0.12, 0.25};
    for (int a = 0; a < 5; ++a) {
        for (int f = 0; f < 4; ++f) {
            AtomSpec d;
            d.type = DAMPED;
            d.is_pair = true;
            d.damped.alpha = damped_alphas[a];
            d.damped.f0 = damped_freqs[f];
            descriptors.push_back(d);
            descriptors.push_back(d);
        }
    }

    double bspline_scales[] = {0.05, 0.1, 0.2, 0.4};
    double bspline_positions[] = {0.1, 0.25, 0.4, 0.55, 0.7, 0.85};
    for (int s = 0; s < 4; ++s) {
        for (int p = 0; p < 6; ++p) {
            AtomSpec d;
            d.type = BSPLINE;
            d.is_pair = false;
            d.bspline.scale = bspline_scales[s];
            d.bspline.position = bspline_positions[p];
            d.bspline.order = 3;
            descriptors.push_back(d);
        }
    }

    double anti_alphas[] = {0.05, 0.15, 0.3};
    double anti_freqs[] = {0.05, 0.1, 0.2};
    for (int a = 0; a < 3; ++a) {
        for (int f = 0; f < 3; ++f) {
            AtomSpec d;
            d.type = ANTI_DAMPED;
            d.is_pair = true;
            d.anti_damped.alpha = anti_alphas[a];
            d.anti_damped.f0 = anti_freqs[f];
            descriptors.push_back(d);
            descriptors.push_back(d);
        }
    }

    double sigmoid_ks[] = {5.0, 10.0, 20.0};
    double sigmoid_t0s[] = {0.2, 0.5, 0.8};
    for (int k = 0; k < 3; ++k) {
        for (int t = 0; t < 3; ++t) {
            AtomSpec d_up, d_down;
            d_up.type = SIGMOID;
            d_up.is_pair = false;
            d_up.sigmoid.k = sigmoid_ks[k];
            d_up.sigmoid.t0 = sigmoid_t0s[t];
            d_up.sigmoid.direction = true;
            d_down.type = SIGMOID;
            d_down.is_pair = false;
            d_down.sigmoid.k = sigmoid_ks[k];
            d_down.sigmoid.t0 = sigmoid_t0s[t];
            d_down.sigmoid.direction = false;
            descriptors.push_back(d_up);
            descriptors.push_back(d_down);
        }
    }

    for (int order = 1; order <= 5; ++order) {
        AtomSpec d;
        d.type = CHEBYSHEV;
        d.is_pair = false;
        d.quasi.k = order;
        descriptors.push_back(d);
    }

    double ds_ks[] = {5.0, 10.0, 20.0};
    double ds_t0s[] = {0.2, 0.4, 0.6, 0.8};
    double ds_t1s[] = {0.3, 0.5, 0.7, 0.9};
    for (int k = 0; k < 3; ++k) {
        for (int t0 = 0; t0 < 4; ++t0) {
            for (int t1 = 0; t1 < 4; ++t1) {
                if (ds_t1s[t1] <= ds_t0s[t0]) continue;
                AtomSpec d;
                d.type = DOUBLE_SIGMOID;
                d.is_pair = false;
                d.sigmoid.k = ds_ks[k];
                d.sigmoid.t0 = ds_t0s[t0];
                d.sigmoid.t1 = ds_t1s[t1];
                descriptors.push_back(d);
            }
        }
    }

    double g_mus[] = {0.1, 0.3, 0.5, 0.7, 0.9};
    double g_sigmas[] = {0.03, 0.06, 0.1, 0.15, 0.2};
    for (int m = 0; m < 5; ++m) {
        for (int s = 0; s < 5; ++s) {
            AtomSpec d;
            d.type = GAUSSIAN;
            d.is_pair = false;
            d.morlet.mu = g_mus[m];
            d.morlet.sigma = g_sigmas[s];
            descriptors.push_back(d);
        }
    }

    double e_mus[] = {0.2, 0.4, 0.6, 0.8};
    double e_sigmas[] = {0.05, 0.1, 0.2, 0.3};
    for (int m = 0; m < 4; ++m) {
        for (int s = 0; s < 4; ++s) {
            AtomSpec d;
            d.type = ERF;
            d.is_pair = false;
            d.morlet.mu = e_mus[m];
            d.morlet.sigma = e_sigmas[s];
            descriptors.push_back(d);
        }
    }

    double dc_alphas[] = {0.1, 0.5, 1.0, 2.0};
    double dc_f0s[] = {0.01, 0.05, 0.1, 0.2, 0.4};
    double dc_betas[] = {-0.01, -0.005, 0.0, 0.005, 0.01};
    for (int a = 0; a < 4; ++a) {
        for (int f = 0; f < 5; ++f) {
            for (int b = 0; b < 5; ++b) {
                AtomSpec d;
                d.type = DAMPED_CHIRP;
                d.is_pair = true;
                d.damped.alpha = dc_alphas[a];
                d.chirp.f0_hi = dc_f0s[f];
                d.chirp.beta_hi = dc_betas[b];
                descriptors.push_back(d);
                descriptors.push_back(d);
            }
        }
    }

    double so_ks[] = {5.0, 10.0, 20.0};
    double so_t0s[] = {0.2, 0.5, 0.8};
    double so_fs[] = {0.05, 0.1, 0.2, 0.3};
    for (int k = 0; k < 3; ++k) {
        for (int t = 0; t < 3; ++t) {
            for (int f = 0; f < 4; ++f) {
                AtomSpec d;
                d.type = SIGMOID_OSC;
                d.is_pair = true;
                d.sigmoid.k = so_ks[k];
                d.sigmoid.t0 = so_t0s[t];
                d.chirp.f0_hi = so_fs[f];
                descriptors.push_back(d);
                descriptors.push_back(d);
            }
        }
    }

    double eg_alphas[] = {0.1, 0.5, 1.0, 2.0};
    for (int a = 0; a < 4; ++a) {
        AtomSpec d;
        d.type = EXP_GROWTH;
        d.is_pair = false;
        d.damped.alpha = eg_alphas[a];
        descriptors.push_back(d);
    }

    double ed_alphas[] = {0.1, 0.5, 1.0, 2.0, 5.0};
    for (int a = 0; a < 5; ++a) {
        AtomSpec d;
        d.type = EXP_DECAY;
        d.is_pair = false;
        d.damped.alpha = ed_alphas[a];
        descriptors.push_back(d);
    }

    double tanh_ks[] = {5.0, 10.0, 20.0, 50.0};
    double tanh_t0s[] = {0.2, 0.5, 0.8};
    for (int k = 0; k < 4; ++k) {
        for (int t = 0; t < 3; ++t) {
            AtomSpec d;
            d.type = TANH;
            d.is_pair = false;
            d.sigmoid.k = tanh_ks[k];
            d.sigmoid.t0 = tanh_t0s[t];
            descriptors.push_back(d);
        }
    }

    double l_mus[] = {0.2, 0.4, 0.6, 0.8};
    double l_gammas[] = {0.02, 0.05, 0.1, 0.15};
    for (int m = 0; m < 4; ++m) {
        for (int g = 0; g < 4; ++g) {
            AtomSpec d;
            d.type = LORENTZIAN;
            d.is_pair = false;
            d.morlet.mu = l_mus[m];
            d.morlet.sigma = l_gammas[g];
            descriptors.push_back(d);
        }
    }

    double p_alphas[] = {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0};
    for (int a = 0; a < 7; ++a) {
        AtomSpec d;
        d.type = POWER;
        d.is_pair = false;
        d.quasi.theta_hi = p_alphas[a];
        descriptors.push_back(d);
    }

    {
        AtomSpec d;
        d.type = HAAR;
        d.is_pair = false;
        descriptors.push_back(d);
    }

    for (int order = 0; order <= 5; ++order) {
        AtomSpec d;
        d.type = BESSEL;
        d.is_pair = false;
        d.quasi.k = order;
        descriptors.push_back(d);
    }

    return descriptors;
}

// =============================================================================
// РЕАЛИЗАЦИЯ DictionaryManager (LRU-кэш словарей)
// =============================================================================

std::unique_ptr<DictionaryManager> g_dict_manager;

DictionaryManager::DictionaryManager(size_t max_cache_size)
    : max_cache_size_(max_cache_size)
{
    ensure_descriptors_initialized();
}

void DictionaryManager::ensure_descriptors_initialized() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (descriptors_initialized_) return;
    descriptors_ = create_full_dictionary();
    num_atoms_ = (int)descriptors_.size();
    descriptors_initialized_ = true;
}

void DictionaryManager::build_and_upload_dictionary(int frame_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cache_.find(frame_size) != cache_.end()) return;
    if (cache_.size() >= max_cache_size_) evict_lru();

    auto entry = DictionaryEntry(frame_size, ++access_counter_);
    entry.dict_hi.reset(new CudaPoolGuard<double>(num_atoms_ * frame_size));
    entry.dict_lo.reset(new CudaPoolGuard<double>(num_atoms_ * frame_size));
    if (!entry.dict_hi->is_valid() || !entry.dict_lo->is_valid()) {
        throw std::runtime_error("DictionaryManager: failed to allocate GPU memory for dictionary");
    }
    double* d_hi = entry.dict_hi->get();
    double* d_lo = entry.dict_lo->get();
    build_dictionary_gpu(descriptors_, frame_size, d_hi, d_lo, entry.atom_norms);
    cache_[frame_size] = std::move(entry);
    lru_list_.push_front(frame_size);
}

void DictionaryManager::evict_lru() {
    if (cache_.empty()) return;
    int lru_frame = -1;
    size_t min_access = SIZE_MAX;
    for (const auto& pair : cache_) {
        if (pair.second.last_access < min_access) {
            min_access = pair.second.last_access;
            lru_frame = pair.first;
        }
    }
    if (lru_frame >= 0) {
        lru_list_.remove(lru_frame);
        cache_.erase(lru_frame);
    }
}

void DictionaryManager::touch_entry(int frame_size) {
    auto it = cache_.find(frame_size);
    if (it != cache_.end()) {
        it->second.last_access = ++access_counter_;
        lru_list_.remove(frame_size);
        lru_list_.push_front(frame_size);
    }
}

double* DictionaryManager::get_dictionary_hi(int frame_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(frame_size);
    if (it == cache_.end()) {
        if (cache_.size() >= max_cache_size_) evict_lru();
        auto entry = DictionaryEntry(frame_size, ++access_counter_);
        entry.dict_hi.reset(new CudaPoolGuard<double>(num_atoms_ * frame_size));
        entry.dict_lo.reset(new CudaPoolGuard<double>(num_atoms_ * frame_size));
        if (!entry.dict_hi->is_valid() || !entry.dict_lo->is_valid()) {
            throw std::runtime_error("DictionaryManager: failed to allocate GPU memory for dictionary");
        }
        double* d_hi = entry.dict_hi->get();
        double* d_lo = entry.dict_lo->get();
        build_dictionary_gpu(descriptors_, frame_size, d_hi, d_lo, entry.atom_norms);
        cache_[frame_size] = std::move(entry);
        lru_list_.push_front(frame_size);
        it = cache_.find(frame_size);
    } else {
        it->second.last_access = ++access_counter_;
        lru_list_.remove(frame_size);
        lru_list_.push_front(frame_size);
    }
    return it->second.dict_hi->get();
}

double* DictionaryManager::get_dictionary_lo(int frame_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(frame_size);
    if (it == cache_.end()) {
        if (cache_.size() >= max_cache_size_) evict_lru();
        auto entry = DictionaryEntry(frame_size, ++access_counter_);
        entry.dict_hi.reset(new CudaPoolGuard<double>(num_atoms_ * frame_size));
        entry.dict_lo.reset(new CudaPoolGuard<double>(num_atoms_ * frame_size));
        if (!entry.dict_hi->is_valid() || !entry.dict_lo->is_valid()) {
            throw std::runtime_error("DictionaryManager: failed to allocate GPU memory for dictionary");
        }
        double* d_hi = entry.dict_hi->get();
        double* d_lo = entry.dict_lo->get();
        build_dictionary_gpu(descriptors_, frame_size, d_hi, d_lo, entry.atom_norms);
        cache_[frame_size] = std::move(entry);
        lru_list_.push_front(frame_size);
        it = cache_.find(frame_size);
    } else {
        it->second.last_access = ++access_counter_;
        lru_list_.remove(frame_size);
        lru_list_.push_front(frame_size);
    }
    return it->second.dict_lo->get();
}

const alw_vector<double>& DictionaryManager::get_atom_norms(int frame_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(frame_size);
    if (it == cache_.end()) {
        if (cache_.size() >= max_cache_size_) evict_lru();
        auto entry = DictionaryEntry(frame_size, ++access_counter_);
        entry.dict_hi.reset(new CudaPoolGuard<double>(num_atoms_ * frame_size));
        entry.dict_lo.reset(new CudaPoolGuard<double>(num_atoms_ * frame_size));
        if (!entry.dict_hi->is_valid() || !entry.dict_lo->is_valid()) {
            throw std::runtime_error("DictionaryManager: failed to allocate GPU memory for dictionary");
        }
        double* d_hi = entry.dict_hi->get();
        double* d_lo = entry.dict_lo->get();
        build_dictionary_gpu(descriptors_, frame_size, d_hi, d_lo, entry.atom_norms);
        cache_[frame_size] = std::move(entry);
        lru_list_.push_front(frame_size);
        it = cache_.find(frame_size);
    } else {
        it->second.last_access = ++access_counter_;
        lru_list_.remove(frame_size);
        lru_list_.push_front(frame_size);
    }
    return it->second.atom_norms;
}

void DictionaryManager::clear_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    lru_list_.clear();
    access_counter_ = 0;
}

// =============================================================================
// УСТАРЕВШИЕ ЭКСПОРТЫ ДЛЯ ОБРАТНОЙ СОВМЕСТИМОСТИ
// =============================================================================

void set_dict_config(const void* buffer, size_t size) { (void)buffer; (void)size; }
bool is_custom_dict_enabled() { return false; }
void reset_dict_config() {}
bool is_learned_dict_enabled() { return false; }
const alw_vector<double>& get_learned_dict() { static alw_vector<double> empty; return empty; }
const alw_vector<double>& get_learned_norms() { static alw_vector<double> empty; return empty; }
void load_learned_dict_to_gpu(int frame_size, double* d_dict_hi, double* d_dict_lo, alw_vector<double>& atom_norms) {
    (void)frame_size; (void)d_dict_hi; (void)d_dict_lo; (void)atom_norms;
}

void init_global_dictionary_manager(size_t max_cache_size) {
    if (!g_dict_manager) {
        g_dict_manager.reset(new DictionaryManager(max_cache_size));
    }
}
