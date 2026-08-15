#ifndef HELIOS_CONFIG_H
#define HELIOS_CONFIG_H

#include "alw_math.h"
#include "orthogonalization.h"
#include "volume_bars.h"

// =============================================================================
// НОВЫЕ СТРУКТУРЫ ДЛЯ НОВОГО СТЕКА
// =============================================================================

struct EOPConfig {
    double alpha = 0.1;          // регуляризация
    double gamma = 0.05;         // порог энергии
    double theta = 0.001;        // порог относительного уменьшения
    bool enable = true;
};

struct AMADConfig {
    bool enable = true;
    int min_block_size = 8;
    int max_block_size = 128;
    int num_scales = 3;
    int refine_iter = 2;
};

struct AFCConfig {
    bool enable = false;
    int coarse_points = 50;
    int refine_iter = 5;
    double freq_min = 0.01;
    double freq_max = 0.49;
    double inertia_alpha = 0.5;
};

struct EODConfig {
    bool enable = false;
    int max_iter = 100;
    double learning_rate = 0.1;
    double regularization_lambda = 0.001;
    int hop_size = 64;
    int max_atoms_per_frame = 10;
    double sigma_noise = 1e-5;
    double tolerance = 1e-6;
    bool verbose = false;
    int learn_samples = 10000;   // количество выборок для обучения EOD
};

// =============================================================================
// СТРУКТУРА КОНФИГУРАЦИИ HELIOS
// Объединяет все параметры для настройки процесса разложения
// =============================================================================
struct HeliosConfig {
    // =========================================================================
    // 1. ОСНОВНЫЕ ПАРАМЕТРЫ РАЗЛОЖЕНИЯ
    // =========================================================================
    
    // Размер фрейма (количество отсчётов в одном окне)
    int frame_size = 1024;
    
    // Шаг между фреймами (перекрытие = frame_size - hop_size)
    int hop_size = 512;
    
    // Максимальное число итераций OMP на фрейм
    int max_iterations = 10;
    
    // Параметр локальной регуляризации (для OMP)
    double gamma_local = 0.05;
    
    // Параметр глобальной регуляризации (для финальной регрессии)
    double gamma_global = 1e-3;
    
    // Коэффициент порога когерентности (умножается на оценку шума)
    double gamma_coherence = 0.90;
    
    // Использовать фиксированную оценку шума (вместо адаптивной)
    bool use_fixed_sigma = false;
    
    // Значение фиксированной сигмы (если use_fixed_sigma = true)
    double fixed_sigma = 1e-9;
    
    // Отключить финальную регрессию (использовать только OMP-коэффициенты)
    bool no_final_regression = false;
    
    // =========================================================================
    // 2. ОРТОГОНАЛЬНЫЙ БАЗИС
    // =========================================================================
    
    // Включить вычисление ортогональных коэффициентов
    bool enable_ortho = true;
    
    // Число полиномиальных базисных функций (полиномы Лежандра)
    int num_poly = 10;
    
    // Число гармонических функций (синусы и косинусы)
    int num_harm = 10;
    
    // Число морлет-функций (вейвлеты)
    int num_morlet = 10;
    
    // Параметры ортогонализации (режим, точность, итерации и т.д.)
    OrthoParams ortho_params;
    
    // =========================================================================
    // 3. ДЕТРЕНДИНГ (адаптивное выделение тренда)
    // =========================================================================
    
    // Включить детрендинг перед разложением
    bool enable_detrend = false;
    
    // Максимальное число сегментов для адаптивного детрендинга
    int detrend_max_segments = 10;
    
    // Минимальная длина сегмента (в отсчётах)
    int detrend_min_segment_len = 32;
    
    // Порог улучшения BIC для принятия разрыва сегмента
    double detrend_bic_threshold = 2.0;
    
    // Параметр сглаживания (для EWMA)
    double detrend_lambda = 0.99;
    
    // Максимальный порядок полинома для детрендинга (1-3)
    int detrend_max_order = 2;
    
    // Автоматический выбор порядка полинома
    bool detrend_auto_order = true;
    
    // Число итераций Huber (для робастной регрессии)
    int huber_iter = 5;
    
    // Параметр Huber (константа для взвешивания)
    double huber_c = 1.345;
    
    // Порог F-статистики для сшивки сегментов
    double stitch_threshold = 2.0;
    
    // Выводить отладочную информацию по детрендингу
    bool detrend_verbose = false;
    
    // =========================================================================
    // 4. VOLUME BARS (построение баров по объёму)
    // =========================================================================
    
    // Включить построение Volume Bars (вместо Time Bars)
    bool use_volume_bars = false;
    
    // Параметры построения Volume Bars (порог объёма, долларовый объём и т.д.)
    VolumeBarParams volume_params;
    
    // =========================================================================
    // 5. НОВЫЕ КОМПОНЕНТЫ СТЕКА
    // =========================================================================
    
    // EOP (Energy-Optimized Pursuit) — алгоритм разложения
    EOPConfig eop;
    
    // AMAD‑X — адаптивная оценка шума
    AMADConfig amad;
    
    // AFC — адаптивный подбор частоты
    AFCConfig afc;
    
    // EOD — обучение словаря
    EODConfig eod;
    
    // =========================================================================
    // 6. МЕТОДЫ ДЛЯ РАБОТЫ С КОНФИГУРАЦИЕЙ
    // =========================================================================
    
    // Сбросить все параметры на значения по умолчанию
    void reset() {
        *this = HeliosConfig();
    }
    
    // Проверить корректность параметров (вернёт false, если что-то не так)
    bool validate() const {
        if (frame_size < 4) return false;
        if (hop_size < 1) return false;
        if (hop_size > frame_size) return false;
        if (max_iterations < 1) return false;
        if (gamma_local < 0.0 || gamma_local > 1.0) return false;
        if (gamma_global < 0.0 || gamma_global > 1.0) return false;
        if (gamma_coherence < 0.0 || gamma_coherence > 2.0) return false;
        if (fixed_sigma < 0.0) return false;
        if (num_poly < 0) return false;
        if (num_harm < 0) return false;
        if (num_morlet < 0) return false;
        if (detrend_max_segments < 1) return false;
        if (detrend_min_segment_len < 4) return false;
        if (detrend_max_order < 1 || detrend_max_order > 3) return false;
        if (huber_iter < 0) return false;
        if (huber_c <= 0.0) return false;
        if (stitch_threshold <= 0.0) return false;
        if (volume_params.threshold_volume < 1) return false;
        if (volume_params.max_bars < 0) return false;
        if (volume_params.min_tick_size < 0.0) return false;
        
        // Валидация новых компонентов
        if (eop.alpha < 0.0 || eop.alpha > 1.0) return false;
        if (eop.gamma < 0.0 || eop.gamma > 1.0) return false;
        if (eop.theta < 0.0 || eop.theta > 1.0) return false;
        if (amad.min_block_size < 1) return false;
        if (amad.max_block_size < amad.min_block_size) return false;
        if (amad.num_scales < 1) return false;
        if (amad.refine_iter < 0) return false;
        if (afc.coarse_points < 2) return false;
        if (afc.refine_iter < 0) return false;
        if (afc.freq_min < 0.0 || afc.freq_min >= afc.freq_max) return false;
        if (afc.freq_max > 0.5) return false;
        if (afc.inertia_alpha < 0.0 || afc.inertia_alpha > 1.0) return false;
        if (eod.max_iter < 1) return false;
        if (eod.learning_rate <= 0.0 || eod.learning_rate > 1.0) return false;
        if (eod.regularization_lambda < 0.0) return false;
        if (eod.hop_size < 1) return false;
        if (eod.max_atoms_per_frame < 1) return false;
        if (eod.sigma_noise < 0.0) return false;
        if (eod.tolerance < 0.0) return false;
        if (eod.learn_samples < 0) return false;
        
        return true;
    }
};

#endif // HELIOS_CONFIG_H
