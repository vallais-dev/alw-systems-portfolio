// =============================================================================
// detrend_ca.cu — ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ОНЛАЙН-ДЕТРЕНДИНГА (ДЕМО-ВЕРСИЯ)
// =============================================================================
// ВНИМАНИЕ: Полная версия содержит адаптивный RLS + CUSUM + Huber + AUTO-ORDER.
//           Данная демонстрация включает только базовые вспомогательные функции.
//           Для получения полной версии обратитесь к автору.
// =============================================================================

#include "alw_math.h"
#include "aeds_solver.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <limits>

// =============================================================================
// КОНСТАНТЫ ДЛЯ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ
// =============================================================================
#define MAX_VOL_WINDOW 32
#define MAX_P 4
#define MAX_ORDERS 3

// =============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (device)
// =============================================================================

__forceinline__ __device__ double huber_weight_dd(double err_hi, double err_lo, double c) {
    if (isnan(err_hi) || isinf(err_hi)) return 0.0;
    double abs_r = fabs(err_hi + err_lo);
    if (isnan(abs_r) || isinf(abs_r)) return 0.0;
    return (abs_r <= c) ? 1.0 : c / (abs_r + 1e-30);
}

__forceinline__ __device__ bool init_rls_ls(
    const double* y_hi,
    const double* y_lo,
    const double* buf_hi,
    const double* buf_lo,
    int buf_start,
    int buf_count,
    int p,
    double* theta_hi,
    double* theta_lo,
    double* P_hi,
    double* P_lo)
{
    // Инициализация RLS методом наименьших квадратов (демо-версия)
    if (buf_count < p) return false;

    for (int i = 0; i < buf_count && i < MAX_VOL_WINDOW; ++i) {
        int idx = (buf_start + i) % MAX_VOL_WINDOW;
        if (isnan(buf_hi[idx]) || isinf(buf_hi[idx])) return false;
    }

    // Упрощённая инициализация: здесь могла бы быть матрица Грама и решение СЛАУ,
    // но в демо-версии мы просто заполняем единичную матрицу ковариации.
    for (int i = 0; i < MAX_P; ++i) {
        if (i >= p) break;
        theta_hi[i] = 0.0; theta_lo[i] = 0.0;
        for (int j = 0; j < MAX_P; ++j) {
            if (j >= p) break;
            P_hi[i*p+j] = (i == j) ? 1e6 : 0.0;
            P_lo[i*p+j] = 0.0;
        }
    }
    return true;
}

__forceinline__ __device__ void update_rls_step(
    double* theta_hi,
    double* theta_lo,
    double* P_hi,
    double* P_lo,
    const double* x_pow_hi,
    const double* x_pow_lo,
    double err_hi,
    double err_lo,
    double lambda,
    int p)
{
    if (isnan(err_hi) || isinf(err_hi)) return;

    // Один шаг RLS-обновления (упрощённо)
    // В полной версии здесь вычисляется K, обновляются theta и P.
    // Оставляем только структуру.
    double S_hi[4] = {0.0}, S_lo[4] = {0.0};
    for (int i = 0; i < MAX_P; ++i) {
        if (i >= p) break;
        for (int j = 0; j < MAX_P; ++j) {
            if (j >= p) break;
            double term_hi, term_lo;
            alw_mul_dd(P_hi[i*p+j], P_lo[i*p+j], x_pow_hi[j], x_pow_lo[j], term_hi, term_lo);
            alw_add_dd(S_hi[i], S_lo[i], term_hi, term_lo, S_hi[i], S_lo[i]);
        }
    }

    double denom_hi = lambda, denom_lo = 0.0;
    for (int i = 0; i < MAX_P; ++i) {
        if (i >= p) break;
        double term_hi, term_lo;
        alw_mul_dd(S_hi[i], S_lo[i], x_pow_hi[i], x_pow_lo[i], term_hi, term_lo);
        alw_add_dd(denom_hi, denom_lo, term_hi, term_lo, denom_hi, denom_lo);
    }
    if (isnan(denom_hi) || isinf(denom_hi) || fabs(denom_hi) < 1e-30) {
        denom_hi = 1e-6;
        denom_lo = 0.0;
    }

    double K_hi[4] = {0.0}, K_lo[4] = {0.0};
    for (int i = 0; i < MAX_P; ++i) {
        if (i >= p) break;
        alw_div_dd(S_hi[i], S_lo[i], denom_hi, denom_lo, K_hi[i], K_lo[i]);
        if (isnan(K_hi[i]) || isinf(K_hi[i])) { K_hi[i] = 0.0; K_lo[i] = 0.0; }
    }

    // Обновление theta
    for (int i = 0; i < MAX_P; ++i) {
        if (i >= p) break;
        double term_hi, term_lo;
        alw_mul_dd(K_hi[i], K_lo[i], err_hi, err_lo, term_hi, term_lo);
        alw_add_dd(theta_hi[i], theta_lo[i], term_hi, term_lo, theta_hi[i], theta_lo[i]);
        if (isnan(theta_hi[i]) || isinf(theta_hi[i])) { theta_hi[i] = 0.0; theta_lo[i] = 0.0; }
    }

    // Обновление P (упрощённо)
    for (int i = 0; i < MAX_P; ++i) {
        if (i >= p) break;
        for (int j = 0; j < MAX_P; ++j) {
            if (j >= p) break;
            double term_hi, term_lo;
            alw_mul_dd(K_hi[i], K_lo[i], S_hi[j], S_lo[j], term_hi, term_lo);
            alw_sub_dd(P_hi[i*p+j], P_lo[i*p+j], term_hi, term_lo, P_hi[i*p+j], P_lo[i*p+j]);
            alw_div_dd(P_hi[i*p+j], P_lo[i*p+j], lambda, 0.0, P_hi[i*p+j], P_lo[i*p+j]);
            if (P_hi[i*p+j] > 1e9) { P_hi[i*p+j] = 1e9; P_lo[i*p+j] = 0.0; }
            if (isnan(P_hi[i*p+j]) || isinf(P_hi[i*p+j])) { P_hi[i*p+j] = 0.0; P_lo[i*p+j] = 0.0; }
        }
    }
}

__forceinline__ __device__ double predict_poly_extrap(
    double* theta_hi,
    double* theta_lo,
    double t,
    double d,
    int p,
    double &pred_hi,
    double &pred_lo)
{
    // Экстраполяция полиномом (демо)
    pred_hi = 0.0; pred_lo = 0.0;
    return 0.0;
}

__forceinline__ __device__ double median_of_buffer(double* buf, int n) {
    // Медиана небольшого буфера (упрощённо)
    if (n <= 0) return 0.0;
    if (n > 9) n = 9;
    double arr[9];
    for (int i = 0; i < n; ++i) arr[i] = buf[i];
    for (int i = 1; i < n; ++i) {
        double key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j+1] = arr[j];
            j--;
        }
        arr[j+1] = key;
    }
    return arr[n/2];
}

// =============================================================================
// ПРИМЕЧАНИЕ: Все адаптивные ядра и хост-функции удалены.
// Полная версия включает детектор разрывов CUSUM, автоматический выбор порядка,
// адаптивный λ, Huber-итерации и многое другое.
// =============================================================================