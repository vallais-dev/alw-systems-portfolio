// =============================================================================
// volume_bars.cu — РЕАЛИЗАЦИЯ ПОСТРОЕНИЯ VOLUME BARS
// Версия 3.1 — добавлены реализации sanitize_tick и update_adaptive_threshold
// =============================================================================

#include "volume_bars.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

// =============================================================================
// РЕАЛИЗАЦИЯ КЛАССА VolumeBarBuilder
// =============================================================================

VolumeBarBuilder::VolumeBarBuilder(const VolumeBarParams& params)
    : params_(params),
      accumulated_volume_(0.0),
      open_(0.0),
      high_(0.0),
      low_(0.0),
      close_(0.0),
      sum_price_volume_(0.0),
      sum_volume_(0.0),
      bar_started_(false),
      rolling_volume_accumulator_(0.0),
      last_tick_timestamp_(0)
{
    // Инициализация завершена
}

void VolumeBarBuilder::add_tick(double price, double volume, uint64_t timestamp) {
    if (params_.enable_sanitizer && !sanitize_tick(price, volume, timestamp)) {
        return;
    }

    last_tick_timestamp_ = timestamp;

    if (params_.min_tick_size > 0.0 && bar_started_) {
        if (std::fabs(price - close_) < params_.min_tick_size) {
            // Игнорируем малый шум цены, но объём всё равно накапливаем
        }
    }

    double vol_effective = params_.use_dollar_volume ? price * volume : volume;

    if (!bar_started_) {
        reset_bar(price, vol_effective);
        bar_started_ = true;
    } else {
        if (price > high_) high_ = price;
        if (price < low_) low_ = price;
        close_ = price;

        sum_price_volume_ += price * vol_effective;
        sum_volume_ += vol_effective;
        accumulated_volume_ += vol_effective;
    }

    if (accumulated_volume_ >= params_.threshold_volume) {
        close_bar(timestamp);
        reset_bar(price, 0.0);
        bar_started_ = true;
    }
}

bool VolumeBarBuilder::is_bar_ready() const {
    return accumulated_volume_ >= params_.threshold_volume;
}

VolumeBar VolumeBarBuilder::pop_bar() {
    if (bars_.empty()) {
        return VolumeBar{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0};
    }
    VolumeBar bar = bars_.back();
    bars_.pop_back();
    return bar;
}

const alw_vector<VolumeBar>& VolumeBarBuilder::get_bars() const {
    return bars_;
}

void VolumeBarBuilder::reset() {
    accumulated_volume_ = 0.0;
    open_ = high_ = low_ = close_ = 0.0;
    sum_price_volume_ = 0.0;
    sum_volume_ = 0.0;
    bars_.clear();
    bar_started_ = false;
    rolling_volume_accumulator_ = 0.0;
    last_tick_timestamp_ = 0;
}

void VolumeBarBuilder::close_bar(uint64_t timestamp) {
    VolumeBar bar;
    bar.open = open_;
    bar.high = high_;
    bar.low = low_;
    bar.close = close_;
    bar.volume = accumulated_volume_;
    bar.vwap = (sum_volume_ > 0.0) ? (sum_price_volume_ / sum_volume_) : close_;
    bar.timestamp = timestamp;
    bars_.push_back(bar);

    update_adaptive_threshold(accumulated_volume_);

    if (params_.max_bars > 0 && (int)bars_.size() > params_.max_bars) {
        bars_.erase(bars_.begin());
    }
}

void VolumeBarBuilder::reset_bar(double price, double volume) {
    open_ = price;
    high_ = price;
    low_ = price;
    close_ = price;
    accumulated_volume_ = volume;
    sum_price_volume_ = price * volume;
    sum_volume_ = volume;
    bar_started_ = true;
}

// =============================================================================
// РЕАЛИЗАЦИИ ВСПОМОГАТЕЛЬНЫХ МЕТОДОВ
// =============================================================================

bool VolumeBarBuilder::sanitize_tick(double price, double volume, uint64_t timestamp) const {
    (void)timestamp; // подавляем предупреждение
    if (volume <= 0.0 || price <= 0.0) return false;
    if (std::isnan(price) || std::isinf(price) ||
        std::isnan(volume) || std::isinf(volume)) return false;
    if (bar_started_ && params_.max_price_spike_pct > 0.0) {
        double price_dev = std::fabs(price - close_) / (close_ + 1e-12);
        if (price_dev > params_.max_price_spike_pct) return false;
    }
    return true;
}

void VolumeBarBuilder::update_adaptive_threshold(double bar_volume) {
    if (!params_.use_adaptive_threshold) return;
    double estimated_bar_target = bar_volume;
    double new_threshold = params_.ewma_alpha * estimated_bar_target +
                           (1.0 - params_.ewma_alpha) * params_.threshold_volume;
    params_.threshold_volume = std::clamp(new_threshold, params_.min_threshold, params_.max_threshold);
}

// =============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (без изменений)
// =============================================================================

alw_vector<VolumeBar> build_volume_bars_from_ohlcv(
    const alw_vector<double>& opens,
    const alw_vector<double>& highs,
    const alw_vector<double>& lows,
    const alw_vector<double>& closes,
    const alw_vector<double>& volumes,
    const VolumeBarParams& params)
{
    alw_vector<VolumeBar> result;
    if (opens.empty() || closes.empty() || volumes.empty()) {
        return result;
    }

    VolumeBarBuilder builder(params);
    double accumulated = 0.0;
    VolumeBar current;
    bool started = false;

    for (size_t i = 0; i < closes.size(); ++i) {
        double price = closes[i];
        double volume = params.use_dollar_volume ? price * volumes[i] : volumes[i];

        if (params.enable_sanitizer) {
            if (volume <= 0.0 || price <= 0.0) continue;
            if (std::isnan(price) || std::isinf(price) ||
                std::isnan(volume) || std::isinf(volume)) continue;
        }

        if (!started) {
            current.open = opens[i];
            current.high = highs[i];
            current.low = lows[i];
            current.close = closes[i];
            current.volume = 0.0;
            current.vwap = 0.0;
            current.timestamp = 0;
            accumulated = 0.0;
            started = true;
        }

        if (highs[i] > current.high) current.high = highs[i];
        if (lows[i] < current.low) current.low = lows[i];
        current.close = closes[i];
        accumulated += volume;
        current.volume += volume;

        if (accumulated >= params.threshold_volume) {
            current.vwap = (current.close + current.open + current.high + current.low) / 4.0;
            result.push_back(current);
            started = false;
        }
    }

    if (started && accumulated > 0) {
        current.vwap = (current.close + current.open + current.high + current.low) / 4.0;
        result.push_back(current);
    }

    return result;
}

alw_vector<double> volume_bars_to_prices(const alw_vector<VolumeBar>& bars) {
    alw_vector<double> prices;
    prices.reserve(bars.size());
    for (const auto& bar : bars) {
        prices.push_back(bar.close);
    }
    return prices;
}

alw_vector<double> volume_bars_to_log_prices(const alw_vector<VolumeBar>& bars) {
    alw_vector<double> prices;
    prices.reserve(bars.size());
    for (const auto& bar : bars) {
        double p = bar.close;
        if (p > 0.0) {
            prices.push_back(std::log(p));
        } else {
            prices.push_back(0.0);
        }
    }
    return prices;
}

alw_vector<double> volume_bars_to_vwap(const alw_vector<VolumeBar>& bars) {
    alw_vector<double> vwaps;
    vwaps.reserve(bars.size());
    for (const auto& bar : bars) {
        vwaps.push_back(bar.vwap);
    }
    return vwaps;
}
