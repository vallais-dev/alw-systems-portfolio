// =============================================================================
// detrend_integration.cu — ИНТЕГРАЦИЯ ДЕТРЕНДИНГА С ОЦЕНКОЙ ШУМА (ДЕМО-ВЕРСИЯ)
// =============================================================================
//
// ВНИМАНИЕ: Полная версия содержит адаптивные детрендеры (offline/causal)
//           и выбеливание сигнала. В демонстрационной версии оставлены только
//           вспомогательные функции и оценка шума через AMAD-X.
//           Детрендинг и выбеливание отключены.
//           Для получения полной версии обратитесь к автору.
// =============================================================================

#include "detrend_integration.h"
#include "alw_math.h"
#include "detrend_types.h"
#include "whitening.h"
#include "amad_x.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>

// =============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ОСТАВЛЕНЫ ДЛЯ ДЕМОНСТРАЦИИ)
// =============================================================================

// Оценка уровня шума по сигналу (MAD-оценка)
static double estimate_sigma_from_signal(const double* d_signal, int N) {
    if (N < 2) return 1e-9;
    std::vector<double> signal(N);
    cudaMemcpy(signal.data(), d_signal, N * sizeof(double), cudaMemcpyDeviceToHost);
    std::vector<double> sorted = signal;
    std::sort(sorted.begin(), sorted.end());
    double median = (N % 2 == 0) ? (sorted[N/2 - 1] + sorted[N/2]) / 2.0 : sorted[N/2];
    double mad = 0.0;
    for (int i = 0; i < N; ++i) mad += fabs(signal[i] - median);
    mad /= N;
    return mad / 0.6745;
}

// Детектор цветного шума (упрощённая версия — всегда возвращает false)
static bool detect_colored_noise(const double* d_signal, int N, bool verbose, double& confidence) {
    if (verbose) {
        AEDS_LOGD("detect_colored_noise: демо-версия всегда возвращает 'white'");
    }
    (void)d_signal; (void)N;
    confidence = 0.0;
    return false;
}

// =============================================================================
// ОСНОВНЫЕ ФУНКЦИИ ИНТЕГРАЦИИ — ЗАГЛУШКИ
// =============================================================================

void apply_offline_detrend(
    const alw_vector<double>& signal_hi,
    const alw_vector<double>& signal_lo,
    alw_vector<double>& residual_hi,
    alw_vector<double>& residual_lo,
    const DetrendConfig& config)
{
    AEDS_LOGW("apply_offline_detrend: демо-версия, копирует сигнал в остаток");
    AEDS_LOGW("  Полная версия содержит адаптивную сегментацию на основе BIC.");
    int N = (int)signal_hi.size();
    residual_hi = signal_hi;
    residual_lo = signal_lo;
    (void)config;
}

void apply_causal_detrend(
    const alw_vector<double>& signal_hi,
    const alw_vector<double>& signal_lo,
    alw_vector<double>& residual_hi,
    alw_vector<double>& residual_lo,
    const DetrendConfig& config,
    cudaStream_t stream)
{
    AEDS_LOGW("apply_causal_detrend: демо-версия, копирует сигнал в остаток");
    AEDS_LOGW("  Полная версия содержит RLS + CUSUM с адаптивным λ.");
    int N = (int)signal_hi.size();
    residual_hi = signal_hi;
    residual_lo = signal_lo;
    (void)config;
    (void)stream;
}

// =============================================================================
// КОМБИНИРОВАННАЯ ОЦЕНКА ШУМА (С ИСПОЛЬЗОВАНИЕМ AMAD-X)
// =============================================================================
double estimate_noise_with_detrend(
    const double* d_signal_hi,
    const double* d_signal_lo,
    int N,
    const DetrendConfig& config,
    bool verbose,
    cudaStream_t stream)
{
    if (N < 4) {
        if (verbose) AEDS_LOGW("estimate_noise_with_detrend: N < 4, returning MIN_EPSILON");
        return MIN_EPSILON;
    }

    // В демо-версии детрендинг пропускается, используется AMAD-X
    if (verbose) {
        AEDS_LOGD("estimate_noise_with_detrend: демо-версия, детрендинг пропущен");
    }

    // Оценка шума через AMAD-X
    double sigma = amad_x_estimate_adaptive(d_signal_hi, d_signal_lo, N, verbose);

    // Дополнительная защита от слишком маленькой сигмы
    double global_std = 0.0;
    {
        std::vector<double> signal(N);
        cudaMemcpy(signal.data(), d_signal_hi, N * sizeof(double), cudaMemcpyDeviceToHost);
        double mean = 0.0;
        for (int i = 0; i < N; ++i) mean += signal[i];
        mean /= N;
        double var = 0.0;
        for (int i = 0; i < N; ++i) { double diff = signal[i] - mean; var += diff * diff; }
        var /= N;
        global_std = sqrt(var);
    }
    double min_sigma = 0.01 * global_std;
    if (sigma < min_sigma && sigma > 0.0) {
        if (verbose) AEDS_LOGD("AMAD-X sigma %.6f < min_sigma %.6f, clamping", sigma, min_sigma);
        sigma = min_sigma;
    }
    if (isnan(sigma) || isinf(sigma) || sigma < MIN_EPSILON) {
        sigma = MIN_EPSILON;
    }

    if (verbose) {
        AEDS_LOGI("estimate_noise_with_detrend: финальная сигма = %.6f (демо-версия)", sigma);
    }

    (void)config;
    (void)stream;
    return sigma;
}
