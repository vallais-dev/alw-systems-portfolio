// aeds_core.hpp (public interface, stripped of adaptive logic)
#ifndef AEDS_CORE_HPP
#define AEDS_CORE_HPP

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cstdio>

#include "alw_math.h"
#include "aeds_profiler.hpp"

struct AEDS_Params {
    double target_precision;
    uint32_t max_iterations;
    double omega_init;
    double tikhonov_alpha;
    bool use_precond;
    bool use_normal_equations;
    bool adaptive_omega;
    bool use_dd_on_gpu;
    bool use_direct;
    bool use_row_scaling;
};

// ---- базовые DD-операции (CPU) ----
inline DD alw_add_dd_cpu(DD a, DD b) { ... }   // полная реализация
inline DD alw_sub_dd_cpu(DD a, DD b) { ... }
inline DD alw_mul_dd_cpu(DD a, DD b) { ... }
inline DD alw_div_dd_cpu(DD a, DD b) { ... }
inline DD alw_abs_dd_cpu(DD a) { ... }
inline bool alw_less_than_cpu(DD a, double t) { ... }
inline bool alw_is_critically_small_cpu(DD a, double e) { ... }

// ---- Gaussian elimination (public) ----
inline bool gauss_solve_dd(const std::vector<DD>& A, const std::vector<DD>& b,
                           std::vector<DD>& x, int n);

// ---- Оценка числа обусловленности ----
inline double estimate_condition_number(const std::vector<DD>& A_row, int m, int n);

#endif