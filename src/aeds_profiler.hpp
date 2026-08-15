// aeds_profiler.hpp
#ifndef AEDS_PROFILER_HPP
#define AEDS_PROFILER_HPP

#include <chrono>
#include <vector>
#include <cmath>
#include <cstdio>
#include "alw_math.h"

// Структура для хранения профиля одной итерации или блока итераций
struct AEDS_IterationProfile {
    int iteration;
    double residual_norm;
    double time_elapsed_ms;      // время выполнения итерации (мс)
    double omega;                // значение omega на этой итерации
    double grad_norm;            // норма градиента (опционально)
};

// Структура для хранения статистики по решателю
struct AEDS_Profile {
    bool is_active;              // включено ли профилирование
    int max_profile_iterations;  // сколько итераций профилируем
    std::vector<AEDS_IterationProfile> iter_profiles;
    double initial_residual;
    double final_residual;
    double total_time_ms;
    double avg_iter_time_ms;
    double convergence_rate;     // отношение residual к начальному
    double cond_estimate;        // оценка числа обусловленности
    bool is_well_conditioned;    // cond < 1e6
    bool is_ill_conditioned;     // cond > 1e8
    bool is_extreme;             // cond > 1e12

    // Рекомендации по настройке
    double recommended_omega;
    bool recommended_use_gauss;
    bool recommended_use_regularization;
    double recommended_lambda;
    bool recommended_switch_to_gpu;   // если текущий CPU, а стоит переключить на GPU
    bool recommended_switch_to_cpu;   // если текущий GPU, а стоит переключить на CPU
    bool recommended_iterative_refinement; // стоит ли применить уточнение

    // Сброс
    void reset() {
        is_active = false;
        max_profile_iterations = 10;
        iter_profiles.clear();
        initial_residual = 0.0;
        final_residual = 0.0;
        total_time_ms = 0.0;
        avg_iter_time_ms = 0.0;
        convergence_rate = 0.0;
        cond_estimate = 1.0;
        is_well_conditioned = true;
        is_ill_conditioned = false;
        is_extreme = false;
        recommended_omega = 1.0;
        recommended_use_gauss = false;
        recommended_use_regularization = false;
        recommended_lambda = 1e-10;
        recommended_switch_to_gpu = false;
        recommended_switch_to_cpu = false;
        recommended_iterative_refinement = false;
    }

    AEDS_Profile() { reset(); }
};

// Класс для сбора профиля
class AEDS_Profiler {
public:
    AEDS_Profiler() : active_(false), max_iter_(10) {}

    void start(int max_iter = 10) {
        active_ = true;
        max_iter_ = max_iter;
        profile_.reset();
        profile_.is_active = true;
        profile_.max_profile_iterations = max_iter;
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    // Запись итерации
    void record_iteration(int iter, double residual, double omega, double grad_norm = 0.0) {
        if (!active_ || iter > max_iter_) return;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time_).count();
        AEDS_IterationProfile prof;
        prof.iteration = iter;
        prof.residual_norm = residual;
        prof.time_elapsed_ms = elapsed_ms;
        prof.omega = omega;
        prof.grad_norm = grad_norm;
        profile_.iter_profiles.push_back(prof);
        if (iter == 0) profile_.initial_residual = residual;
        profile_.final_residual = residual;
        if (iter == max_iter_) {
            // вычисляем среднее время
            if (!profile_.iter_profiles.empty()) {
                double total = 0.0;
                for (auto& p : profile_.iter_profiles) total += p.time_elapsed_ms;
                profile_.avg_iter_time_ms = total / profile_.iter_profiles.size();
                profile_.total_time_ms = total;
            }
            if (profile_.initial_residual > 0) {
                profile_.convergence_rate = profile_.final_residual / profile_.initial_residual;
            }
            // Останавливаем профилирование
            active_ = false;
        }
    }

    // Установка оценки числа обусловленности
    void set_cond_estimate(double cond) {
        profile_.cond_estimate = cond;
        profile_.is_well_conditioned = (cond < 1e6);
        profile_.is_ill_conditioned = (cond > 1e8);
        profile_.is_extreme = (cond > 1e12);
    }

    // Анализ профиля и генерация рекомендаций
    void analyze_and_recommend(AEDS_Profile& out_profile) {
        out_profile = profile_; // копируем
        // Базовая логика
        double rate = out_profile.convergence_rate;
        if (rate > 0.9) {
            // сходимость очень медленная – увеличиваем omega
            out_profile.recommended_omega = std::min(1.9, out_profile.iter_profiles.back().omega * 1.5);
            out_profile.recommended_use_gauss = true; // предложить Гаусса
        } else if (rate > 0.5) {
            // средняя сходимость – немного увеличим omega
            out_profile.recommended_omega = std::min(1.9, out_profile.iter_profiles.back().omega * 1.2);
        } else if (rate < 0.1) {
            // быстрая сходимость – можно уменьшить omega для стабильности
            out_profile.recommended_omega = std::max(0.5, out_profile.iter_profiles.back().omega * 0.8);
        } else {
            out_profile.recommended_omega = out_profile.iter_profiles.back().omega;
        }

        // Если итерации долгие (> 100 мс) и residual не падает – переключиться на Гаусса
        if (out_profile.avg_iter_time_ms > 100.0 && rate > 0.5) {
            out_profile.recommended_use_gauss = true;
        }

        // Если плохая обусловленность – регуляризация
        if (out_profile.is_ill_conditioned) {
            out_profile.recommended_use_regularization = true;
            out_profile.recommended_lambda = 1e-8 * (out_profile.cond_estimate / 1e8);
            if (out_profile.recommended_lambda < 1e-12) out_profile.recommended_lambda = 1e-12;
        }

        // Если extreme – обязательно регуляризация и уточнение
        if (out_profile.is_extreme) {
            out_profile.recommended_use_regularization = true;
            out_profile.recommended_lambda = 1e-6;
            out_profile.recommended_iterative_refinement = true;
            out_profile.recommended_use_gauss = true;
        }

        // Если среднее время итерации > 1 с – переключиться на GPU (если сейчас CPU)
        if (out_profile.avg_iter_time_ms > 1000.0) {
            out_profile.recommended_switch_to_gpu = true;
        } else if (out_profile.avg_iter_time_ms < 10.0 && out_profile.iter_profiles.size() > 0) {
            // очень быстро – возможно, GPU не нужен
            out_profile.recommended_switch_to_cpu = true;
        }

        // Если residual уже мал – уточнение не нужно
        if (out_profile.final_residual < 1e-12) {
            out_profile.recommended_iterative_refinement = false;
        }

        AEDS_LOGI("Profiler analysis: rate=%.3e, avg_time=%.2f ms, cond=%.2e", rate, out_profile.avg_iter_time_ms, out_profile.cond_estimate);
        AEDS_LOGI("Recommendations: omega=%.4f, use_gauss=%d, use_reg=%d, lambda=%.2e, switch_gpu=%d, switch_cpu=%d, refine=%d",
                  out_profile.recommended_omega, out_profile.recommended_use_gauss, out_profile.recommended_use_regularization,
                  out_profile.recommended_lambda, out_profile.recommended_switch_to_gpu, out_profile.recommended_switch_to_cpu,
                  out_profile.recommended_iterative_refinement);
        // Сохраняем профиль во внутреннюю переменную
        profile_ = out_profile;
    }

    const AEDS_Profile& get_profile() const { return profile_; }

private:
    bool active_;
    int max_iter_;
    AEDS_Profile profile_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

#endif // AEDS_PROFILER_HPP
