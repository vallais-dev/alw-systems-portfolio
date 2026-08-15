// =============================================================================
// volume_bars.h — ПОСТРОЕНИЕ VOLUME BARS С АДАПТИВНЫМ ПОРОГОМ И ФИЛЬТРАЦИЕЙ
// Версия 3.2 — исправлено: убраны inline у методов, которые определены в .cu
// =============================================================================

#ifndef VOLUME_BARS_H
#define VOLUME_BARS_H

#include "alw_math.h"
#include <cstdint>
#include <algorithm>

// Включаем экспорт для GCC/Clang (если не Windows)
#pragma GCC visibility push(default)

// =============================================================================
// СТРУКТУРА ДЛЯ ОДНОГО VOLUME BAR
// =============================================================================
struct VolumeBar {
    double open;        // цена открытия
    double high;        // максимальная цена
    double low;         // минимальная цена
    double close;       // цена закрытия
    double volume;      // накопленный объём
    double vwap;        // средневзвешенная по объёму цена
    uint64_t timestamp; // метка времени
};

// =============================================================================
// ПАРАМЕТРЫ ПОСТРОЕНИЯ VOLUME BARS
// =============================================================================
struct VolumeBarParams {
    double threshold_volume = 5000.0;
    bool use_dollar_volume = false;
    int max_bars = 0;
    double min_tick_size = 0.0;
    bool enable_sanitizer = true;
    double max_price_spike_pct = 0.05;
    bool use_adaptive_threshold = true;
    double target_bars_per_day = 1000.0;
    double ewma_alpha = 0.005;
    double min_threshold = 100.0;
    double max_threshold = 100000.0;
};

// =============================================================================
// КЛАСС ДЛЯ ПОСТРОЕНИЯ VOLUME BARS ИЗ ПОТОКА ТИКОВ
// =============================================================================
class VolumeBarBuilder {
public:
    explicit VolumeBarBuilder(const VolumeBarParams& params);

    void add_tick(double price, double volume, uint64_t timestamp = 0);
    bool is_bar_ready() const;
    VolumeBar pop_bar();
    const alw_vector<VolumeBar>& get_bars() const;
    void reset();
    double get_current_threshold() const { return params_.threshold_volume; }

private:
    VolumeBarParams params_;
    double accumulated_volume_;
    double open_, high_, low_, close_;
    double sum_price_volume_;
    double sum_volume_;
    alw_vector<VolumeBar> bars_;
    bool bar_started_;
    double rolling_volume_accumulator_;
    uint64_t last_tick_timestamp_;

    void close_bar(uint64_t timestamp);
    void reset_bar(double price, double volume);

    // Объявления без inline (реализация в .cu)
    bool sanitize_tick(double price, double volume, uint64_t timestamp) const;
    void update_adaptive_threshold(double bar_volume);
};

// =============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (не требуют экспорта)
// =============================================================================
alw_vector<VolumeBar> build_volume_bars_from_ohlcv(
    const alw_vector<double>& opens,
    const alw_vector<double>& highs,
    const alw_vector<double>& lows,
    const alw_vector<double>& closes,
    const alw_vector<double>& volumes,
    const VolumeBarParams& params
);

alw_vector<double> volume_bars_to_prices(const alw_vector<VolumeBar>& bars);
alw_vector<double> volume_bars_to_log_prices(const alw_vector<VolumeBar>& bars);
alw_vector<double> volume_bars_to_vwap(const alw_vector<VolumeBar>& bars);

#pragma GCC visibility pop

#endif // VOLUME_BARS_H
