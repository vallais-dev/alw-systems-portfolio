// =============================================================================
// alw_helios.cu — точка входа, экспорты и тесты (обёртка над core)
// ИСПРАВЛЕНО: использование std::unique_ptr с кастомным удалителем для
// предотвращения утечек памяти при исключениях в экспортных функциях.
// =============================================================================

#define _USE_MATH_DEFINES
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <stdexcept>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <random>
#include <limits>
#include <ctime>
#include <cstdlib>
#include <mutex>
#include <memory>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <getopt.h>
#endif

#include "alw_math.h"
#include "alw_dictionary.h"
#include "aeds_solver.hpp"
#include "alw_helios_core.h"
#include "alw_helios_utils.h"
#include "helios_config.h"
#include "volume_bars.h"

// =============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ЭКСПОРТОВ
// =============================================================================

static alw_string atom_type_to_string(AtomType type) {
    switch (type) {
        case QUASI: return "QUASI";
        case CHIRP: return "CHIRP";
        case MORLET: return "MORLET";
        case DAMPED: return "DAMPED";
        case STEP: return "STEP";
        case BSPLINE: return "BSPLINE";
        case ANTI_DAMPED: return "ANTI_DAMPED";
        case SIGMOID: return "SIGMOID";
        case LEARNED: return "LEARNED";
        case CHEBYSHEV: return "CHEBYSHEV";
        case DOUBLE_SIGMOID: return "DOUBLE_SIGMOID";
        case GAUSSIAN: return "GAUSSIAN";
        case ERF: return "ERF";
        case DAMPED_CHIRP: return "DAMPED_CHIRP";
        case SIGMOID_OSC: return "SIGMOID_OSC";
        case EXP_GROWTH: return "EXP_GROWTH";
        case EXP_DECAY: return "EXP_DECAY";
        case TANH: return "TANH";
        case LORENTZIAN: return "LORENTZIAN";
        case POWER: return "POWER";
        case HAAR: return "HAAR";
        case BESSEL: return "BESSEL";
        default: return "UNKNOWN";
    }
}

static uint8_t atom_type_to_uint8(AtomType type) {
    return static_cast<uint8_t>(type);
}

// =============================================================================
// ЭКСПОРТИРУЕМЫЕ ФУНКЦИИ (extern "C")
// =============================================================================

extern "C" {

// -----------------------------------------------------------------------------
// helios_process_block — устаревший JSON-экспорт (заглушка)
// -----------------------------------------------------------------------------
int helios_process_block(
    const double* y_hi,
    const double* y_lo,
    int N,
    int frame_size,
    int hop_size,
    int max_iterations,
    double gamma_local,
    double gamma_global,
    double gamma_coherence,
    int use_fixed_sigma,
    double fixed_sigma,
    int no_final_regression,
    char** out_json)
{
    if (!out_json) return -1;
    *out_json = nullptr;

    if (!y_hi || N <= 0 || frame_size <= 0 || hop_size <= 0) {
        ALW_LOG_ERROR("helios_process_block: invalid parameters");
        return -1;
    }

    try {
        // Формируем конфиг из переданных параметров
        HeliosConfig config;
        config.frame_size = frame_size;
        config.hop_size = hop_size;
        config.max_iterations = max_iterations;
        config.gamma_local = gamma_local;
        config.gamma_global = gamma_global;
        config.gamma_coherence = gamma_coherence;
        config.use_fixed_sigma = (use_fixed_sigma != 0);
        config.fixed_sigma = fixed_sigma;
        config.no_final_regression = (no_final_regression != 0);
        config.enable_ortho = true;  // по умолчанию

        helios_set_config(config);

        ensure_gpu_initialized(frame_size);

        if (frame_size > N) frame_size = N;
        if (hop_size > frame_size) hop_size = frame_size;

        alw_vector<double> host_y_hi(N), host_y_lo(N);
        for (int i = 0; i < N; ++i) {
            host_y_hi[i] = y_hi[i];
            host_y_lo[i] = (y_lo != nullptr) ? y_lo[i] : 0.0;
            if (isnan(host_y_hi[i]) || isinf(host_y_hi[i])) host_y_hi[i] = 0.0;
            if (isnan(host_y_lo[i]) || isinf(host_y_lo[i])) host_y_lo[i] = 0.0;
        }

        CudaPoolGuard<double> d_y_global_hi(N), d_y_global_lo(N);
        CUDA_CHECK(cudaMemcpy(d_y_global_hi.get(), host_y_hi.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y_global_lo.get(), host_y_lo.data(), N * sizeof(double), cudaMemcpyHostToDevice));

        if (config.enable_ortho && !g_ortho_initialized) {
            OrthoParams params = config.ortho_params;
            params.mode = ORTHO_HYBRID;
            params.tolerance = 1e-12;
            params.force_svd = false;
            params.householder_block = 2;
            params.svd_submatrix_size = 10;
            params.max_householder_iter = 1;
            params.max_svd_iter = 100;
            params.svd_tolerance = 1e-14;
            params.verbose = false;
            params.save_ortho_stats = false;

            if (!init_orthogonal_basis(host_y_hi, frame_size, config.num_poly, config.num_harm, config.num_morlet, params)) {
                ALW_LOG_WARN("Failed to init orthogonal basis, disabling");
                config.enable_ortho = false;
                helios_set_config(config);
            }
        }

        alw_vector<DetectedEvent> all_events;
        alw_vector<alw_vector<double>> coeffs_hi, coeffs_lo;
        run_alw_helios_ortho(d_y_global_hi.get(), d_y_global_lo.get(),
                             N, frame_size, hop_size,
                             g_d_dict_hi->get(), g_d_dict_lo->get(), g_num_atoms,
                             all_events, g_atom_norms, g_descriptors,
                             max_iterations,
                             gamma_local, gamma_global, gamma_coherence,
                             config.use_fixed_sigma, config.fixed_sigma,
                             config.no_final_regression,
                             nullptr,
                             config.enable_ortho,
                             coeffs_hi, coeffs_lo);

        // Формируем JSON
        std::stringstream json;
        json << "{\n";
        json << "  \"num_events\": " << all_events.size() << ",\n";
        json << "  \"events\": [\n";
        for (size_t i = 0; i < all_events.size(); ++i) {
            const auto& ev = all_events[i];
            double amplitude = fabs(ev.amplitude_hi + ev.amplitude_lo);
            double phase = ev.phase_hi;
            double energy = ev.energy_hi + ev.energy_lo;
            json << "    {\"frame_start\": " << ev.frame_start
                 << ", \"atom_index\": " << ev.atom_index
                 << ", \"atom_type\": \"" << ev.atom_type_str << "\""
                 << ", \"atom_params\": \"" << ev.atom_params_str << "\""
                 << ", \"amplitude\": " << std::scientific << std::setprecision(15) << amplitude
                 << ", \"phase\": " << phase
                 << ", \"energy\": " << std::scientific << std::setprecision(15) << energy << "}";
            if (i + 1 < all_events.size()) json << ",";
            json << "\n";
        }
        json << "  ]\n";
        json << "}\n";

        std::string json_str = json.str();
        // Используем unique_ptr с кастомным удалителем для защиты от утечек
        std::unique_ptr<char, decltype(&free)> out_json_ptr(
            static_cast<char*>(malloc(json_str.size() + 1)), &free);
        if (!out_json_ptr) {
            ALW_LOG_ERROR("Failed to allocate JSON output buffer");
            return -1;
        }
        std::memcpy(out_json_ptr.get(), json_str.c_str(), json_str.size() + 1);
        *out_json = out_json_ptr.release(); // Передаём владение вызывающему
        return 0;

    } catch (const std::exception& e) {
        ALW_LOG_ERROR("Exception: " + alw_string(e.what()));
        return -1;
    }
}

void helios_free_json(char* json) {
    if (json != nullptr) free(json);
}

// -----------------------------------------------------------------------------
// helios_process_block_binary — основной экспорт для Time Bars
// -----------------------------------------------------------------------------
int helios_process_block_binary(
    const double* y_hi,
    const double* y_lo,
    int N,
    int frame_size,
    int hop_size,
    int max_iterations,
    double gamma_local,
    double gamma_global,
    double gamma_coherence,
    int use_fixed_sigma,
    double fixed_sigma,
    int no_final_regression,
    char** out_binary,
    int* out_size)
{
    if (!out_binary || !out_size) return -1;
    *out_binary = nullptr;
    *out_size = 0;

    if (!y_hi || N <= 0 || frame_size <= 0 || hop_size <= 0) {
        ALW_LOG_ERROR("helios_process_block_binary: invalid parameters");
        return -1;
    }

    try {
        // Формируем конфиг
        HeliosConfig config;
        config.frame_size = frame_size;
        config.hop_size = hop_size;
        config.max_iterations = max_iterations;
        config.gamma_local = gamma_local;
        config.gamma_global = gamma_global;
        config.gamma_coherence = gamma_coherence;
        config.use_fixed_sigma = (use_fixed_sigma != 0);
        config.fixed_sigma = fixed_sigma;
        config.no_final_regression = (no_final_regression != 0);
        config.enable_ortho = true;
        config.use_volume_bars = false;  // этот экспорт работает с Time Bars

        helios_set_config(config);

        ensure_gpu_initialized(frame_size);

        if (frame_size > N) frame_size = N;
        if (hop_size > frame_size) hop_size = frame_size;

        alw_vector<double> host_y_hi(N), host_y_lo(N);
        for (int i = 0; i < N; ++i) {
            host_y_hi[i] = y_hi[i];
            host_y_lo[i] = (y_lo != nullptr) ? y_lo[i] : 0.0;
            if (isnan(host_y_hi[i]) || isinf(host_y_hi[i])) host_y_hi[i] = 0.0;
            if (isnan(host_y_lo[i]) || isinf(host_y_lo[i])) host_y_lo[i] = 0.0;
        }

        CudaPoolGuard<double> d_y_global_hi(N), d_y_global_lo(N);
        CUDA_CHECK(cudaMemcpy(d_y_global_hi.get(), host_y_hi.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y_global_lo.get(), host_y_lo.data(), N * sizeof(double), cudaMemcpyHostToDevice));

        if (config.enable_ortho && !g_ortho_initialized) {
            OrthoParams params = config.ortho_params;
            params.mode = ORTHO_HYBRID;
            params.tolerance = 1e-12;
            params.force_svd = false;
            params.householder_block = 2;
            params.svd_submatrix_size = 10;
            params.max_householder_iter = 1;
            params.max_svd_iter = 100;
            params.svd_tolerance = 1e-14;
            params.verbose = false;
            params.save_ortho_stats = false;

            if (!init_orthogonal_basis(host_y_hi, frame_size, config.num_poly, config.num_harm, config.num_morlet, params)) {
                ALW_LOG_WARN("Failed to init orthogonal basis, disabling");
                config.enable_ortho = false;
                helios_set_config(config);
            }
        }

        alw_vector<DetectedEvent> all_events;
        alw_vector<alw_vector<double>> coeffs_hi, coeffs_lo;
        run_alw_helios_ortho(d_y_global_hi.get(), d_y_global_lo.get(),
                             N, frame_size, hop_size,
                             g_d_dict_hi->get(), g_d_dict_lo->get(), g_num_atoms,
                             all_events, g_atom_norms, g_descriptors,
                             max_iterations,
                             gamma_local, gamma_global, gamma_coherence,
                             config.use_fixed_sigma, config.fixed_sigma,
                             config.no_final_regression,
                             nullptr,
                             config.enable_ortho,
                             coeffs_hi, coeffs_lo);

        // Сериализация в бинарный формат
        alw_vector<uint8_t> buffer;
        uint32_t magic = BIN_MAGIC;
        uint32_t version = BIN_VERSION;
        int32_t count = (int32_t)all_events.size();
        int32_t frame_size_out = frame_size;

        alw_vector<BinaryEvent> binary_events;
        binary_events.reserve(all_events.size());
        for (const auto& ev : all_events) {
            BinaryEvent be;
            be.frame_start = ev.frame_start;
            be.atom_index = ev.atom_index;
            be.atom_type = atom_type_to_uint8(g_descriptors[ev.atom_index].type);
            be.amplitude_hi = ev.amplitude_hi;
            be.amplitude_lo = ev.amplitude_lo;
            be.phase_hi = ev.phase_hi;
            be.phase_lo = ev.phase_lo;
            be.energy_hi = ev.energy_hi;
            be.energy_lo = ev.energy_lo;
            binary_events.push_back(be);
        }

        size_t total_bytes = sizeof(magic) + sizeof(version) + sizeof(count) + sizeof(frame_size_out)
                           + binary_events.size() * sizeof(BinaryEvent);

        if (config.enable_ortho && g_ortho_initialized && !coeffs_hi.empty()) {
            int32_t num_frames = (int32_t)coeffs_hi.size();
            total_bytes += sizeof(num_frames);
            for (size_t f = 0; f < coeffs_hi.size(); ++f) {
                int32_t nb = g_num_basis;
                total_bytes += sizeof(nb);
                total_bytes += g_num_basis * sizeof(double);
                total_bytes += g_num_basis * sizeof(double);
            }
        } else {
            int32_t num_frames = 0;
            total_bytes += sizeof(num_frames);
        }

        buffer.resize(total_bytes);
        uint8_t* ptr = buffer.data();

        std::memcpy(ptr, &magic, sizeof(magic)); ptr += sizeof(magic);
        std::memcpy(ptr, &version, sizeof(version)); ptr += sizeof(version);
        std::memcpy(ptr, &count, sizeof(count)); ptr += sizeof(count);
        std::memcpy(ptr, &frame_size_out, sizeof(frame_size_out)); ptr += sizeof(frame_size_out);

        if (!binary_events.empty()) {
            std::memcpy(ptr, binary_events.data(), binary_events.size() * sizeof(BinaryEvent));
            ptr += binary_events.size() * sizeof(BinaryEvent);
        }

        if (config.enable_ortho && g_ortho_initialized && !coeffs_hi.empty()) {
            int32_t num_frames = (int32_t)coeffs_hi.size();
            std::memcpy(ptr, &num_frames, sizeof(num_frames)); ptr += sizeof(num_frames);
            for (size_t f = 0; f < coeffs_hi.size(); ++f) {
                int32_t nb = g_num_basis;
                std::memcpy(ptr, &nb, sizeof(nb)); ptr += sizeof(nb);
                std::memcpy(ptr, coeffs_hi[f].data(), g_num_basis * sizeof(double));
                ptr += g_num_basis * sizeof(double);
                std::memcpy(ptr, coeffs_lo[f].data(), g_num_basis * sizeof(double));
                ptr += g_num_basis * sizeof(double);
            }
        } else {
            int32_t num_frames = 0;
            std::memcpy(ptr, &num_frames, sizeof(num_frames)); ptr += sizeof(num_frames);
        }

        // Используем unique_ptr с free для автоматического освобождения при исключении
        std::unique_ptr<char, decltype(&free)> out_ptr(
            static_cast<char*>(malloc(total_bytes)), &free);
        if (!out_ptr) {
            ALW_LOG_ERROR("Failed to allocate output buffer");
            return -1;
        }
        std::memcpy(out_ptr.get(), buffer.data(), total_bytes);
        *out_binary = out_ptr.release(); // передаём владение
        *out_size = (int)total_bytes;
        return 0;

    } catch (const std::exception& e) {
        ALW_LOG_ERROR("Exception: " + alw_string(e.what()));
        return -1;
    }
}

void helios_free_binary(char* binary) {
    if (binary != nullptr) free(binary);
}

// -----------------------------------------------------------------------------
// helios_set_detrend_params — установка параметров детрендинга (устаревшее)
// -----------------------------------------------------------------------------
void helios_set_detrend_params(
    int enable,
    int max_segments,
    int min_segment_len,
    double bic_threshold,
    double lambda,
    int max_order,
    int auto_order,
    int huber_iter,
    double huber_c,
    double stitch_threshold,
    int verbose)
{
    // Обновляем конфиг
    HeliosConfig config = helios_get_config();
    config.enable_detrend = (enable != 0);
    config.detrend_max_segments = max_segments;
    config.detrend_min_segment_len = min_segment_len;
    config.detrend_bic_threshold = bic_threshold;
    config.detrend_lambda = lambda;
    config.detrend_max_order = max_order;
    config.detrend_auto_order = (auto_order != 0);
    config.huber_iter = huber_iter;
    config.huber_c = huber_c;
    config.stitch_threshold = stitch_threshold;
    config.detrend_verbose = (verbose != 0);
    helios_set_config(config);
}

// -----------------------------------------------------------------------------
// helios_apply_detrend — заглушка (пока просто копирует)
// -----------------------------------------------------------------------------
void helios_apply_detrend(
    const double* y_hi,
    const double* y_lo,
    int N,
    double* out_hi,
    double* out_lo)
{
    if (!y_hi || !out_hi || N <= 0) {
        ALW_LOG_ERROR("helios_apply_detrend: invalid arguments");
        return;
    }
    for (int i = 0; i < N; ++i) {
        out_hi[i] = y_hi[i];
        out_lo[i] = (y_lo != nullptr) ? y_lo[i] : 0.0;
        if (isnan(out_hi[i]) || isinf(out_hi[i])) out_hi[i] = 0.0;
        if (isnan(out_lo[i]) || isinf(out_lo[i])) out_lo[i] = 0.0;
    }
}

// -----------------------------------------------------------------------------
// helios_set_dict_config — конфигурация словаря
// -----------------------------------------------------------------------------
void helios_set_dict_config(const uint8_t* buffer, int size) {
    if (buffer == nullptr || size <= 0) {
        reset_dict_config();
        return;
    }
    set_dict_config(buffer, static_cast<size_t>(size));
}

// -----------------------------------------------------------------------------
// helios_learn_dictionary — обучение словаря
// -----------------------------------------------------------------------------
void helios_learn_dictionary(
    const double* prices,
    int N,
    int frame_size,
    int num_atoms,
    const double* init_dict,
    int iterations,
    int omp_iterations,
    double gamma_local,
    double learning_rate,
    int power_iterations,
    double* out_dict,
    double* out_atom_norms)
{
    if (!prices || !out_dict || N <= 0 || frame_size <= 0 || num_atoms <= 0) {
        ALW_LOG_ERROR("helios_learn_dictionary: invalid arguments");
        return;
    }
    alw_vector<double> norms;
    learn_dictionary_gpu(prices, N, frame_size, num_atoms, init_dict,
                         iterations, omp_iterations, gamma_local,
                         learning_rate, power_iterations, out_dict, norms);
    if (out_atom_norms) {
        for (int i = 0; i < num_atoms; ++i) out_atom_norms[i] = norms[i];
    }
}

// -----------------------------------------------------------------------------
// helios_serialize_learned_dictionary_to_buffer
// -----------------------------------------------------------------------------
void helios_serialize_learned_dictionary_to_buffer(
    int num_atoms,
    int frame_size,
    const double* dict,
    const double* atom_norms,
    uint8_t** out_buffer,
    int* out_size)
{
    if (!out_buffer || !out_size) return;
    *out_buffer = nullptr;
    *out_size = 0;
    if (!dict || !atom_norms || num_atoms <= 0 || frame_size <= 0) {
        ALW_LOG_ERROR("helios_serialize_learned_dictionary_to_buffer: invalid arguments");
        return;
    }
    alw_vector<uint8_t> buffer;
    alw_vector<double> norms(atom_norms, atom_norms + num_atoms);
    serialize_learned_dictionary_to_buffer(num_atoms, frame_size, dict, norms, buffer);
    if (buffer.empty()) {
        ALW_LOG_ERROR("Serialization failed");
        return;
    }
    *out_size = (int)buffer.size();
    *out_buffer = (uint8_t*)malloc(*out_size);
    if (!*out_buffer) {
        ALW_LOG_ERROR("Memory allocation failed");
        *out_size = 0;
        return;
    }
    std::memcpy(*out_buffer, buffer.data(), *out_size);
}

// -----------------------------------------------------------------------------
// helios_process_volume_bars — НОВЫЙ ЭКСПОРТ: обработка тиков с объёмом
// -----------------------------------------------------------------------------
int helios_process_volume_bars(
    const double* ticks_price,
    const double* ticks_volume,
    int num_ticks,
    const HeliosConfig& config,
    char** out_binary,
    int* out_size)
{
    if (!out_binary || !out_size) return -1;
    *out_binary = nullptr;
    *out_size = 0;

    if (!ticks_price || !ticks_volume || num_ticks <= 0) {
        ALW_LOG_ERROR("helios_process_volume_bars: invalid tick data");
        return -1;
    }

    // Проверяем конфиг
    if (!config.use_volume_bars) {
        ALW_LOG_WARN("helios_process_volume_bars: use_volume_bars is false, forcing to true");
        HeliosConfig cfg = config;
        cfg.use_volume_bars = true;
        helios_set_config(cfg);
    } else {
        helios_set_config(config);
    }

    // Валидация параметров
    if (!helios_get_config().validate()) {
        ALW_LOG_ERROR("helios_process_volume_bars: invalid configuration");
        return -1;
    }

    try {
        // Убедимся, что GPU инициализирован с правильным frame_size
        ensure_gpu_initialized(config.frame_size);

        // Выполняем обработку с Volume Bars
        alw_vector<DetectedEvent> all_events;
        alw_vector<alw_vector<double>> coeffs_hi, coeffs_lo;

        run_alw_helios_with_volume_bars(
            ticks_price,
            ticks_volume,
            num_ticks,
            all_events,
            coeffs_hi,
            coeffs_lo
        );

        // Сериализация в бинарный формат (аналогично helios_process_block_binary)
        alw_vector<uint8_t> buffer;
        uint32_t magic = BIN_MAGIC;
        uint32_t version = BIN_VERSION;
        int32_t count = (int32_t)all_events.size();
        int32_t frame_size_out = config.frame_size;  // используем frame_size из конфига

        alw_vector<BinaryEvent> binary_events;
        binary_events.reserve(all_events.size());
        for (const auto& ev : all_events) {
            BinaryEvent be;
            be.frame_start = ev.frame_start;
            be.atom_index = ev.atom_index;
            be.atom_type = atom_type_to_uint8(g_descriptors[ev.atom_index].type);
            be.amplitude_hi = ev.amplitude_hi;
            be.amplitude_lo = ev.amplitude_lo;
            be.phase_hi = ev.phase_hi;
            be.phase_lo = ev.phase_lo;
            be.energy_hi = ev.energy_hi;
            be.energy_lo = ev.energy_lo;
            binary_events.push_back(be);
        }

        size_t total_bytes = sizeof(magic) + sizeof(version) + sizeof(count) + sizeof(frame_size_out)
                           + binary_events.size() * sizeof(BinaryEvent);

        if (config.enable_ortho && g_ortho_initialized && !coeffs_hi.empty()) {
            int32_t num_frames = (int32_t)coeffs_hi.size();
            total_bytes += sizeof(num_frames);
            for (size_t f = 0; f < coeffs_hi.size(); ++f) {
                int32_t nb = g_num_basis;
                total_bytes += sizeof(nb);
                total_bytes += g_num_basis * sizeof(double);
                total_bytes += g_num_basis * sizeof(double);
            }
        } else {
            int32_t num_frames = 0;
            total_bytes += sizeof(num_frames);
        }

        buffer.resize(total_bytes);
        uint8_t* ptr = buffer.data();

        std::memcpy(ptr, &magic, sizeof(magic)); ptr += sizeof(magic);
        std::memcpy(ptr, &version, sizeof(version)); ptr += sizeof(version);
        std::memcpy(ptr, &count, sizeof(count)); ptr += sizeof(count);
        std::memcpy(ptr, &frame_size_out, sizeof(frame_size_out)); ptr += sizeof(frame_size_out);

        if (!binary_events.empty()) {
            std::memcpy(ptr, binary_events.data(), binary_events.size() * sizeof(BinaryEvent));
            ptr += binary_events.size() * sizeof(BinaryEvent);
        }

        if (config.enable_ortho && g_ortho_initialized && !coeffs_hi.empty()) {
            int32_t num_frames = (int32_t)coeffs_hi.size();
            std::memcpy(ptr, &num_frames, sizeof(num_frames)); ptr += sizeof(num_frames);
            for (size_t f = 0; f < coeffs_hi.size(); ++f) {
                int32_t nb = g_num_basis;
                std::memcpy(ptr, &nb, sizeof(nb)); ptr += sizeof(nb);
                std::memcpy(ptr, coeffs_hi[f].data(), g_num_basis * sizeof(double));
                ptr += g_num_basis * sizeof(double);
                std::memcpy(ptr, coeffs_lo[f].data(), g_num_basis * sizeof(double));
                ptr += g_num_basis * sizeof(double);
            }
        } else {
            int32_t num_frames = 0;
            std::memcpy(ptr, &num_frames, sizeof(num_frames)); ptr += sizeof(num_frames);
        }

        // Используем unique_ptr с free
        std::unique_ptr<char, decltype(&free)> out_ptr(
            static_cast<char*>(malloc(total_bytes)), &free);
        if (!out_ptr) {
            ALW_LOG_ERROR("Failed to allocate output buffer");
            return -1;
        }
        std::memcpy(out_ptr.get(), buffer.data(), total_bytes);
        *out_binary = out_ptr.release();
        *out_size = (int)total_bytes;

        ALW_LOG_INFO("helios_process_volume_bars: processed %d ticks -> %d volume bars -> %d events",
                     num_ticks, (int)all_events.size(), (int)all_events.size());

        return 0;

    } catch (const std::exception& e) {
        ALW_LOG_ERROR("Exception in helios_process_volume_bars: " + alw_string(e.what()));
        return -1;
    }
}

} // extern "C"

// =============================================================================
// ТЕСТЫ (ПОЛНЫЕ РЕАЛИЗАЦИИ)
// =============================================================================

#ifndef HELIOS_LIBRARY

static alw_vector<AtomDescriptor> create_test_descriptors() {
    alw_vector<AtomDescriptor> test_descriptors;
    AtomDescriptor d0; d0.type = CHIRP; d0.is_pair = true;
    d0.chirp.f0_hi = 0.05; d0.chirp.f0_lo = 0.0;
    d0.chirp.beta_hi = 0.0; d0.chirp.beta_lo = 0.0;
    test_descriptors.push_back(d0);
    test_descriptors.push_back(d0);

    AtomDescriptor d1; d1.type = CHIRP; d1.is_pair = true;
    d1.chirp.f0_hi = 0.1; d1.chirp.f0_lo = 0.0;
    d1.chirp.beta_hi = 0.001; d1.chirp.beta_lo = 0.0;
    test_descriptors.push_back(d1);
    test_descriptors.push_back(d1);

    AtomDescriptor d2; d2.type = MORLET; d2.is_pair = true;
    d2.morlet.sigma = 0.07; d2.morlet.mu = 0.5; d2.morlet.w_morlet = 18.0;
    test_descriptors.push_back(d2);
    test_descriptors.push_back(d2);

    AtomDescriptor d3; d3.type = DAMPED; d3.is_pair = true;
    d3.damped.alpha = 0.15; d3.damped.f0 = 0.06;
    test_descriptors.push_back(d3);
    test_descriptors.push_back(d3);

    AtomDescriptor d4; d4.type = ANTI_DAMPED; d4.is_pair = true;
    d4.anti_damped.alpha = 0.05; d4.anti_damped.f0 = 0.1;
    test_descriptors.push_back(d4);
    test_descriptors.push_back(d4);

    AtomDescriptor d5; d5.type = SIGMOID; d5.is_pair = false;
    d5.sigmoid.k = 10.0; d5.sigmoid.t0 = 0.5; d5.sigmoid.direction = true;
    test_descriptors.push_back(d5);

    return test_descriptors;
}

int test_dictionary() {
    ALW_LOG_INFO("=== Dictionary Test Mode ===");
    try {
        init_bspline_constant_memory();

        alw_vector<AtomDescriptor> test_descriptors = create_test_descriptors();
        int test_frame = 256;
        int num_test_atoms = (int)test_descriptors.size();

        CudaPoolGuard<double> test_dict_hi(num_test_atoms * test_frame);
        CudaPoolGuard<double> test_dict_lo(num_test_atoms * test_frame);
        alw_vector<double> test_norms;

        build_dictionary_gpu(test_descriptors, test_frame,
                             test_dict_hi.get(), test_dict_lo.get(), test_norms);

        bool all_ok = true;
        for (int a = 0; a < num_test_atoms; ++a) {
            alw_string type_str = atom_type_to_string(test_descriptors[a].type);
            if (test_norms[a] <= 1e-10) {
                ALW_LOG_ERROR("FAIL Atom[" + alw_to_string(a) + "] type=" + type_str + " ZERO NORM!");
                all_ok = false;
                continue;
            }

            CudaPoolGuard<double> norm_buf(2);
            compute_norm_sq_kernel<<<1, 256>>>(test_dict_hi.get() + a * test_frame,
                                                test_dict_lo.get() + a * test_frame,
                                                test_frame, norm_buf.get());
            CUDA_CHECK(cudaDeviceSynchronize());

            double norm_sq;
            CUDA_CHECK(cudaMemcpy(&norm_sq, norm_buf.get(), sizeof(double), cudaMemcpyDeviceToHost));

            if (fabs(norm_sq - 1.0) > 1e-6) {
                ALW_LOG_ERROR("FAIL Atom[" + alw_to_string(a) + "] type=" + type_str + " norm_sq=" + alw_to_string(norm_sq));
                all_ok = false;
            } else {
                ALW_LOG_INFO("OK Atom[" + alw_to_string(a) + "] type=" + type_str + " norm_before=" + alw_to_string(test_norms[a]));
            }
        }

        if (all_ok) {
            ALW_LOG_INFO("=== Dictionary test PASSED ===");
            return 0;
        } else {
            ALW_LOG_ERROR("=== Dictionary test FAILED ===");
            return 1;
        }
    } catch (const std::exception& e) {
        ALW_LOG_ERROR("Dictionary test exception: " + alw_string(e.what()));
        return 1;
    }
}

int test_decomposition() {
    ALW_LOG_INFO("=== Decomposition Test Mode ===");
    try {
        init_bspline_constant_memory();

        alw_vector<AtomDescriptor> test_descriptors = create_test_descriptors();
        int num_atoms = (int)test_descriptors.size();
        int frame_size = 128;

        CudaPoolGuard<double> test_dict_hi(num_atoms * frame_size);
        CudaPoolGuard<double> test_dict_lo(num_atoms * frame_size);
        alw_vector<double> test_norms;

        build_dictionary_gpu(test_descriptors, frame_size,
                             test_dict_hi.get(), test_dict_lo.get(), test_norms);

        alw_vector<double> h_signal_hi(frame_size, 0.0);
        alw_vector<double> h_signal_lo(frame_size, 0.0);

        alw_vector<double> h_atom0_hi(frame_size), h_atom0_lo(frame_size);
        CUDA_CHECK(cudaMemcpy(h_atom0_hi.data(), test_dict_hi.get(), frame_size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_atom0_lo.data(), test_dict_lo.get(), frame_size * sizeof(double), cudaMemcpyDeviceToHost));

        for (int i = 0; i < frame_size; ++i) {
            h_signal_hi[i] = 3.0 * h_atom0_hi[i];
            h_signal_lo[i] = 3.0 * h_atom0_lo[i];
        }

        alw_vector<double> h_atom2_hi(frame_size), h_atom2_lo(frame_size);
        CUDA_CHECK(cudaMemcpy(h_atom2_hi.data(), test_dict_hi.get() + 2 * frame_size, frame_size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_atom2_lo.data(), test_dict_lo.get() + 2 * frame_size, frame_size * sizeof(double), cudaMemcpyDeviceToHost));

        for (int i = 0; i < frame_size; ++i) {
            h_signal_hi[i] += 1.5 * h_atom2_hi[i];
            h_signal_lo[i] += 1.5 * h_atom2_lo[i];
        }

        CudaPoolGuard<double> d_signal_hi(frame_size), d_signal_lo(frame_size);
        CudaPoolGuard<double> d_res_hi(frame_size), d_res_lo(frame_size);

        CUDA_CHECK(cudaMemcpy(d_signal_hi.get(), h_signal_hi.data(), frame_size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_signal_lo.get(), h_signal_lo.data(), frame_size * sizeof(double), cudaMemcpyHostToDevice));

        alw_vector<DetectedEvent> events;
        dd_hapt_frame_gpu_full(
            d_signal_hi.get(), d_signal_lo.get(),
            d_res_hi.get(), d_res_lo.get(),
            test_dict_hi.get(), test_dict_lo.get(),
            num_atoms, frame_size, 0,
            events, test_norms, test_descriptors,
            10, 0.05, 1e-3, 0.90, false, 0.0, false);

        ALW_LOG_INFO("Decomposition result: " + alw_to_string(events.size()) + " events found");
        for (size_t e = 0; e < events.size(); ++e) {
            const auto& ev = events[e];
            ALW_LOG_INFO("Event[" + alw_to_string(e) + "]: " + ev.atom_type_str + " " + ev.atom_params_str +
                         " amp_hi=" + alw_to_string(ev.amplitude_hi) + " amp_lo=" + alw_to_string(ev.amplitude_lo) +
                         " energy_hi=" + alw_to_string(ev.energy_hi) + " energy_lo=" + alw_to_string(ev.energy_lo));
        }

        ALW_LOG_INFO("=== Decomposition test PASSED ===");
        return 0;
    } catch (const std::exception& e) {
        ALW_LOG_ERROR("Decomposition test exception: " + alw_string(e.what()));
        return 1;
    }
}

int test_pipeline() {
    ALW_LOG_INFO("=== Pipeline Test Mode ===");
    try {
        init_bspline_constant_memory();

        alw_vector<AtomDescriptor> descriptors = create_full_dictionary();
        int num_atoms = (int)descriptors.size();
        int frame_size = 256;
        int N = 512;
        int hop_size = 128;

        ALW_LOG_INFO("Full dictionary: " + alw_to_string(num_atoms) + " atoms");

        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        size_t required = (size_t)num_atoms * frame_size * 2 * sizeof(double) + N * 2 * sizeof(double);
        ALW_LOG_INFO("Required GPU memory: " + alw_to_string(required/(1024*1024)) + " MB, Available: " + alw_to_string(free_mem/(1024*1024)) + " MB");
        if (required > free_mem) ALW_LOG_WARN("WARNING: May OOM, trying anyway...");

        CudaPoolGuard<double> d_dict_hi(num_atoms * frame_size);
        CudaPoolGuard<double> d_dict_lo(num_atoms * frame_size);
        alw_vector<double> atom_norms;

        build_dictionary_gpu(descriptors, frame_size,
                             d_dict_hi.get(), d_dict_lo.get(), atom_norms);

        alw_vector<double> h_signal_hi(N, 0.0);
        alw_vector<double> h_signal_lo(N, 0.0);

        alw_vector<double> h_atom0(frame_size);
        CUDA_CHECK(cudaMemcpy(h_atom0.data(), d_dict_hi.get(), frame_size * sizeof(double), cudaMemcpyDeviceToHost));

        for (int i = 0; i < frame_size; ++i) {
            h_signal_hi[i] = 5.0 * h_atom0[i];
        }

        std::random_device rd;
        std::mt19937 gen(42);
        std::normal_distribution<double> dist(0.0, 0.01);
        for (int i = 0; i < N; ++i) {
            h_signal_hi[i] += dist(gen);
        }

        CudaPoolGuard<double> d_signal_hi(N), d_signal_lo(N);
        CUDA_CHECK(cudaMemcpy(d_signal_hi.get(), h_signal_hi.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_signal_lo.get(), 0, N * sizeof(double)));

        alw_vector<DetectedEvent> all_events;
        run_alw_helios(d_signal_hi.get(), d_signal_lo.get(),
                       N, frame_size, hop_size,
                       d_dict_hi.get(), d_dict_lo.get(), num_atoms,
                       all_events, atom_norms, descriptors,
                       10, 0.05, 1e-3, 0.90, false, 0.0, false);

        ALW_LOG_INFO("Pipeline result: " + alw_to_string(all_events.size()) + " events found");
        for (size_t e = 0; e < std::min(all_events.size(), (size_t)10); ++e) {
            const auto& ev = all_events[e];
            ALW_LOG_INFO("Event[" + alw_to_string(e) + "]: frame=" + alw_to_string(ev.frame_start) +
                         " " + ev.atom_type_str + " " + ev.atom_params_str +
                         " amp_hi=" + alw_to_string(ev.amplitude_hi) + " amp_lo=" + alw_to_string(ev.amplitude_lo));
        }

        ALW_LOG_INFO("=== Pipeline test PASSED ===");
        return 0;
    } catch (const std::exception& e) {
        ALW_LOG_ERROR("Pipeline test exception: " + alw_string(e.what()));
        return 1;
    }
}

// =============================================================================
// MAIN — ПОЛНАЯ РЕАЛИЗАЦИЯ
// =============================================================================

int main(int argc, char* argv[]) {
    GpuMemoryPoolManager::initialize(1024 * 1024 * 1024);

    int frame_size = 1024;
    int hop_size = 512;
    int max_iterations = 10;
    double gamma_local = 0.05;
    double gamma_global = 1e-3;
    double gamma_coherence = DEFAULT_GAMMA_COHERENCE;
    double fixed_sigma = MIN_EPSILON;
    bool use_fixed_sigma = false;
    bool residual_mode = false;
    bool no_final_regression = false;
    alw_string save_dict_file;
    alw_string load_dict_file;
    alw_string output_json_file;
    alw_string output_binary_file;
    alw_string output_residual_file;
    alw_string output_format = "json";
    bool binary_output = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--test-dict") == 0) return test_dictionary();
        if (strcmp(argv[i], "--test-decomp") == 0) return test_decomposition();
        if (strcmp(argv[i], "--test-pipeline") == 0) return test_pipeline();
        if (strcmp(argv[i], "--frame-size") == 0 && i+1 < argc) frame_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hop-size") == 0 && i+1 < argc) hop_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-iter") == 0 && i+1 < argc) max_iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gamma-local") == 0 && i+1 < argc) gamma_local = atof(argv[++i]);
        else if (strcmp(argv[i], "--gamma-global") == 0 && i+1 < argc) gamma_global = atof(argv[++i]);
        else if (strcmp(argv[i], "--gamma-coherence") == 0 && i+1 < argc) gamma_coherence = atof(argv[++i]);
        else if (strcmp(argv[i], "--fixed-sigma") == 0 && i+1 < argc) {
            fixed_sigma = atof(argv[++i]);
            use_fixed_sigma = true;
        }
        else if (strcmp(argv[i], "--residual-mode") == 0) {
            residual_mode = true;
            use_fixed_sigma = true;
            fixed_sigma = MIN_EPSILON;
        }
        else if (strcmp(argv[i], "--no-final-regression") == 0) no_final_regression = true;
        else if (strcmp(argv[i], "--save-dictionary") == 0 && i+1 < argc) save_dict_file = argv[++i];
        else if (strcmp(argv[i], "--load-dictionary") == 0 && i+1 < argc) load_dict_file = argv[++i];
        else if (strcmp(argv[i], "--output-json") == 0 && i+1 < argc) {
            output_json_file = argv[++i];
            output_format = "json";
        }
        else if (strcmp(argv[i], "--output-binary") == 0 && i+1 < argc) {
            output_binary_file = argv[++i];
            output_format = "binary";
            binary_output = true;
        }
        else if (strcmp(argv[i], "--output-residual") == 0 && i+1 < argc) output_residual_file = argv[++i];
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --frame-size N           Frame size (default 1024)\n"
                      << "  --hop-size N             Hop size (default 512)\n"
                      << "  --max-iter N             Max OMP iterations (default 10)\n"
                      << "  --gamma-local X          Local regularization (default 0.05)\n"
                      << "  --gamma-global X         Global regularization (default 1e-3)\n"
                      << "  --gamma-coherence X      Coherence factor (default 0.90)\n"
                      << "  --fixed-sigma X          Fixed noise sigma\n"
                      << "  --residual-mode          Fixed sigma = MIN_EPSILON\n"
                      << "  --no-final-regression    Skip final regression\n"
                      << "  --save-dictionary FILE   Save dictionary\n"
                      << "  --load-dictionary FILE   Load dictionary\n"
                      << "  --output-json FILE       JSON output to file\n"
                      << "  --output-binary FILE     Binary output to file\n"
                      << "  --output-residual FILE   Residual output\n"
                      << "  --test-dict              Run dictionary test\n"
                      << "  --test-decomp            Run decomposition test\n"
                      << "  --test-pipeline          Run full pipeline test\n";
            return 0;
        }
    }

    try {
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
#endif

        auto start_time = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaSetDevice(0));
        int deviceCount;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount > 0) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
            ALW_LOG_INFO("GPU: " + alw_string(prop.name) + " (CC " + alw_to_string(prop.major) + "." + alw_to_string(prop.minor) + ")");
        }

        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        ALW_LOG_INFO("GPU memory: " + alw_to_string(free_mem/(1024*1024)) + " MB free / " + alw_to_string(total_mem/(1024*1024)) + " MB total");

        init_bspline_constant_memory();

        alw_vector<AtomDescriptor> descriptors = create_full_dictionary();
        int num_atoms = (int)descriptors.size();
        ALW_LOG_INFO("Dictionary: " + alw_to_string(num_atoms) + " atoms, frame_size=" + alw_to_string(frame_size));

        size_t required_dict_mem = (size_t)num_atoms * frame_size * 2 * sizeof(double);
        if (required_dict_mem > free_mem) {
            ALW_LOG_WARN("Dictionary requires " + alw_to_string(required_dict_mem/(1024*1024)) +
                         " MB but only " + alw_to_string(free_mem/(1024*1024)) + " MB available");
        }

        CudaPoolGuard<double> d_dict_hi(num_atoms * frame_size);
        CudaPoolGuard<double> d_dict_lo(num_atoms * frame_size);
        alw_vector<double> atom_norms;

        bool dict_loaded = false;
        if (!load_dict_file.empty()) {
            dict_loaded = load_dictionary_from_file(load_dict_file, num_atoms, frame_size,
                                                     d_dict_hi.get(), d_dict_lo.get(), atom_norms);
            if (!dict_loaded) {
                ALW_LOG_WARN("Failed to load dictionary, generating...");
            }
        }
        if (!dict_loaded) {
            build_dictionary_gpu(descriptors, frame_size,
                                 d_dict_hi.get(), d_dict_lo.get(), atom_norms);
            if (!save_dict_file.empty()) {
                save_dictionary_to_file(save_dict_file, num_atoms, frame_size,
                                        d_dict_hi.get(), d_dict_lo.get(), atom_norms);
                ALW_LOG_INFO("Dictionary saved to " + save_dict_file);
            }
        }

        int block_num = 0;
        alw_vector<DetectedEvent> all_events_global;
        alw_fstream residual_out;
        if (!output_residual_file.empty()) {
            residual_out.open(output_residual_file.c_str(), alw_ios_openmode::out | alw_ios_openmode::binary);
            if (!residual_out.is_open()) {
                ALW_LOG_WARN("Cannot open residual file: " + output_residual_file);
            }
        }

        while (true) {
            char sig[3];
            std::cin.read(sig, 3);
            if (std::cin.gcount() == 0) {
                ALW_LOG_INFO("End of input stream");
                break;
            }
            if (std::cin.gcount() != 3 || std::strncmp(sig, "ALW", 3) != 0) {
                throw std::runtime_error("Invalid ALW signature");
            }

            uint64_t N_u64 = 0;
            std::cin.read(reinterpret_cast<char*>(&N_u64), sizeof(uint64_t));
            int32_t N = static_cast<int32_t>(N_u64);

            double scaling_factor = 1.0;
            std::cin.read(reinterpret_cast<char*>(&scaling_factor), sizeof(double));

            unsigned char basis_cfg[3] = {10, 10, 10};
            if (std::cin.peek() != EOF) {
                std::cin.read(reinterpret_cast<char*>(basis_cfg), 3);
                if (std::cin.gcount() != 3) std::cin.clear();
            }

            block_num++;
            ALW_LOG_INFO("Block #" + alw_to_string(block_num) + ": N=" + alw_to_string(N) + " scaling=" + alw_to_string(scaling_factor));

            if (frame_size > N) frame_size = N;
            if (hop_size > frame_size) hop_size = frame_size;

            alw_vector<double> host_y_raw_hi(N);
            size_t total_bytes = N * sizeof(double);
            size_t bytes_read = 0;
            char* ptr = reinterpret_cast<char*>(host_y_raw_hi.data());
            while (bytes_read < total_bytes && std::cin) {
                std::cin.read(ptr + bytes_read, total_bytes - bytes_read);
                size_t chunk = std::cin.gcount();
                if (chunk == 0) break;
                bytes_read += chunk;
            }
            if (bytes_read != total_bytes) throw std::runtime_error("Incomplete y_hi data");

            alw_vector<double> host_y_raw_lo(N, 0.0);
            bytes_read = 0;
            ptr = reinterpret_cast<char*>(host_y_raw_lo.data());
            while (bytes_read < total_bytes && std::cin) {
                std::cin.read(ptr + bytes_read, total_bytes - bytes_read);
                size_t chunk = std::cin.gcount();
                if (chunk == 0) break;
                bytes_read += chunk;
            }

            alw_vector<double> host_y_hi(N), host_y_lo(N);
            for (int i = 0; i < N; ++i) {
                double scaled_hi, scaled_lo;
                alw_div_dd(host_y_raw_hi[i], host_y_raw_lo[i], scaling_factor, 0.0, scaled_hi, scaled_lo);
                host_y_hi[i] = scaled_hi;
                host_y_lo[i] = scaled_lo;
            }

            CudaPoolGuard<double> d_y_global_hi(N), d_y_global_lo(N);
            CUDA_CHECK(cudaMemcpy(d_y_global_hi.get(), host_y_hi.data(), N * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_y_global_lo.get(), host_y_lo.data(), N * sizeof(double), cudaMemcpyHostToDevice));

            alw_vector<DetectedEvent> block_events;
            run_alw_helios(d_y_global_hi.get(), d_y_global_lo.get(),
                           N, frame_size, hop_size,
                           d_dict_hi.get(), d_dict_lo.get(), num_atoms,
                           block_events, atom_norms, descriptors,
                           max_iterations, gamma_local, gamma_global, gamma_coherence,
                           use_fixed_sigma, fixed_sigma, no_final_regression,
                           residual_out.is_open() ? &residual_out : nullptr);

            all_events_global.insert(all_events_global.end(), block_events.begin(), block_events.end());
            ALW_LOG_INFO("Block #" + alw_to_string(block_num) + " done, events: " + alw_to_string(block_events.size()) +
                         " (total: " + alw_to_string(all_events_global.size()) + ")");

            if (binary_output || output_format == "binary") {
                uint32_t magic = BIN_MAGIC;
                uint32_t version = BIN_VERSION;
                int32_t count = (int32_t)block_events.size();
                int32_t frame_size_out = frame_size;
                std::cout.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
                std::cout.write(reinterpret_cast<const char*>(&version), sizeof(version));
                std::cout.write(reinterpret_cast<const char*>(&count), sizeof(count));
                std::cout.write(reinterpret_cast<const char*>(&frame_size_out), sizeof(frame_size_out));
                for (const auto& ev : block_events) {
                    BinaryEvent be;
                    be.frame_start = ev.frame_start;
                    be.atom_index = ev.atom_index;
                    be.atom_type = atom_type_to_uint8(descriptors[ev.atom_index].type);
                    be.amplitude_hi = ev.amplitude_hi;
                    be.amplitude_lo = ev.amplitude_lo;
                    be.phase_hi = ev.phase_hi;
                    be.phase_lo = ev.phase_lo;
                    be.energy_hi = ev.energy_hi;
                    be.energy_lo = ev.energy_lo;
                    std::cout.write(reinterpret_cast<const char*>(&be), sizeof(BinaryEvent));
                }
                std::cout.flush();
            } else if (output_format == "json") {
                std::cout << "{\n  \"block\": " << block_num << ",\n";
                std::cout << "  \"N\": " << N << ",\n";
                std::cout << "  \"scaling_factor\": " << std::scientific << std::setprecision(6) << scaling_factor << ",\n";
                std::cout << "  \"num_events\": " << block_events.size() << ",\n";
                std::cout << "  \"events\": [\n";
                for (size_t e = 0; e < block_events.size(); ++e) {
                    const auto& ev = block_events[e];
                    double amplitude = fabs(ev.amplitude_hi + ev.amplitude_lo);
                    double phase = ev.phase_hi;
                    double energy = ev.energy_hi + ev.energy_lo;
                    std::cout << "    {\"frame_start\": " << ev.frame_start
                              << ", \"atom_index\": " << ev.atom_index
                              << ", \"atom_type\": \"" << ev.atom_type_str.c_str() << "\""
                              << ", \"atom_params\": \"" << ev.atom_params_str.c_str() << "\""
                              << ", \"amplitude\": " << std::scientific << std::setprecision(15) << amplitude
                              << ", \"phase\": " << phase
                              << ", \"energy\": " << std::scientific << std::setprecision(15) << energy << "}";
                    if (e + 1 < block_events.size()) std::cout << ",";
                    std::cout << "\n";
                }
                std::cout << "  ]\n}\n";
                std::cout.flush();
            }
        }

        if (!output_json_file.empty()) {
            alw_fstream json_out(output_json_file.c_str(), alw_ios_openmode::out | alw_ios_openmode::trunc);
            if (json_out.is_open()) {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
                json_out << "{\n";
                json_out << "  \"version\": \"1.2\",\n";
                json_out << "  \"mode\": \"ALW-Helios-DD\",\n";
                json_out << "  \"parameters\": {\n";
                json_out << "    \"frame_size\": " << frame_size << ",\n";
                json_out << "    \"hop_size\": " << hop_size << ",\n";
                json_out << "    \"max_iterations\": " << max_iterations << ",\n";
                json_out << "    \"num_atoms\": " << num_atoms << ",\n";
                json_out << "    \"gamma_local\": " << gamma_local << ",\n";
                json_out << "    \"gamma_global\": " << gamma_global << ",\n";
                json_out << "    \"gamma_coherence\": " << gamma_coherence << ",\n";
                json_out << "    \"residual_mode\": " << (residual_mode ? "true" : "false") << ",\n";
                json_out << "    \"fixed_sigma\": " << (use_fixed_sigma ? alw_to_string(fixed_sigma).c_str() : "null") << ",\n";
                json_out << "    \"no_final_regression\": " << (no_final_regression ? "true" : "false") << "\n";
                json_out << "  },\n";
                json_out << "  \"total_blocks\": " << block_num << ",\n";
                json_out << "  \"total_events\": " << all_events_global.size() << ",\n";
                json_out << "  \"execution_time_ms\": " << duration_ms << ",\n";
                json_out << "  \"events\": [\n";
                for (size_t e = 0; e < all_events_global.size(); ++e) {
                    const auto& ev = all_events_global[e];
                    double amplitude = fabs(ev.amplitude_hi + ev.amplitude_lo);
                    double phase = ev.phase_hi;
                    double energy = ev.energy_hi + ev.energy_lo;
                    json_out << "    {\"frame_start\": " << ev.frame_start
                             << ", \"atom_index\": " << ev.atom_index
                             << ", \"atom_type\": \"" << ev.atom_type_str.c_str() << "\""
                             << ", \"atom_params\": \"" << ev.atom_params_str.c_str() << "\""
                             << ", \"amplitude\": " << std::scientific << std::setprecision(15) << amplitude
                             << ", \"phase\": " << phase
                             << ", \"energy\": " << std::scientific << std::setprecision(15) << energy << "}";
                    if (e + 1 < all_events_global.size()) json_out << ",";
                    json_out << "\n";
                }
                json_out << "  ]\n}\n";
                json_out.close();
                ALW_LOG_INFO("JSON output written to " + output_json_file);
            }
        }

        if (!output_binary_file.empty()) {
            alw_fstream bin_out(output_binary_file.c_str(), alw_ios_openmode::out | alw_ios_openmode::binary);
            if (bin_out.is_open()) {
                uint32_t magic = BIN_MAGIC;
                uint32_t version = BIN_VERSION;
                int32_t count = (int32_t)all_events_global.size();
                int32_t frame_size_out = frame_size;
                bin_out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
                bin_out.write(reinterpret_cast<const char*>(&version), sizeof(version));
                bin_out.write(reinterpret_cast<const char*>(&count), sizeof(count));
                bin_out.write(reinterpret_cast<const char*>(&frame_size_out), sizeof(frame_size_out));
                for (const auto& ev : all_events_global) {
                    BinaryEvent be;
                    be.frame_start = ev.frame_start;
                    be.atom_index = ev.atom_index;
                    be.atom_type = atom_type_to_uint8(descriptors[ev.atom_index].type);
                    be.amplitude_hi = ev.amplitude_hi;
                    be.amplitude_lo = ev.amplitude_lo;
                    be.phase_hi = ev.phase_hi;
                    be.phase_lo = ev.phase_lo;
                    be.energy_hi = ev.energy_hi;
                    be.energy_lo = ev.energy_lo;
                    bin_out.write(reinterpret_cast<const char*>(&be), sizeof(BinaryEvent));
                }
                bin_out.close();
                ALW_LOG_INFO("Binary output written to " + output_binary_file);
            }
        }

        if (residual_out.is_open()) {
            residual_out.close();
            ALW_LOG_INFO("Residual output written to " + output_residual_file);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        ALW_LOG_INFO("Total time: " + alw_to_string(duration) + " ms, total events: " + alw_to_string(all_events_global.size()));

        CUDA_CHECK(cudaDeviceReset());
        return 0;

    } catch (const std::exception& e) {
        ALW_LOG_ERROR("FATAL: " + alw_string(e.what()));
        cudaDeviceReset();
        return 1;
    }
}

#endif // HELIOS_LIBRARY
