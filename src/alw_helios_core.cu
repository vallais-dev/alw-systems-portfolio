// =============================================================================
// alw_helios_core.cu — основные функции разложения, управление словарём и GPU
// Версия 4.2 — добавлены атрибуты видимости для глобальных переменных,
//              используемых в helios_exports.cu.
// =============================================================================

#include "alw_helios_core.h"
#include "alw_helios_utils.h"
#include "detrend_integration.h"
#include "helios_config.h"
#include "volume_bars.h"
#include "alw_dictionary.h"
#include "eod_core.h"
#include "afc.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <random>
#include <mutex>
#include <memory>
#include <sstream>

// =============================================================================
// ГЛОБАЛЬНОЕ СОСТОЯНИЕ
// =============================================================================

// ---- ГЛОБАЛЬНЫЙ КОНФИГ ----
HeliosConfig g_helios_config;

// ---- ГЛОБАЛЬНЫЕ УКАЗАТЕЛИ НА СЛОВАРЬ (для EOD) ----
__attribute__((visibility("default"))) double* g_d_dict_hi_global = nullptr;
__attribute__((visibility("default"))) double* g_d_dict_lo_global = nullptr;
__attribute__((visibility("default"))) alw_vector<double> g_atom_norms_global;
__attribute__((visibility("default"))) int g_global_num_atoms = 0;
__attribute__((visibility("default"))) int g_global_frame_size = 0;

// ---- Флаг, что словарь уже обучен EOD ----
__attribute__((visibility("default"))) bool g_eod_trained = false;

// ---- Дескрипторы атомов (используются в helios_exports.cu) ----
__attribute__((visibility("default"))) alw_vector<AtomDescriptor> g_descriptors;

// ---- Внутренние переменные (не экспортируются) ----
static std::once_flag g_gpu_init_flag;
static bool g_gpu_initialized = false;
static int g_num_atoms = 0;
static int g_frame_size = 0;

// ---- Функции доступа к конфигу ----
void helios_set_config(const HeliosConfig& config) {
    g_helios_config = config;
    // Обновляем устаревшие глобальные переменные для обратной совместимости
    g_enable_detrend = config.enable_detrend;
    g_enable_ortho = config.enable_ortho;
    g_detrend_max_segments = config.detrend_max_segments;
    g_detrend_min_segment_len = config.detrend_min_segment_len;
    g_detrend_bic_threshold = config.detrend_bic_threshold;
    g_detrend_lambda = config.detrend_lambda;
    g_detrend_max_order = config.detrend_max_order;
    g_detrend_auto_order = config.detrend_auto_order;
    g_huber_iter = config.huber_iter;
    g_huber_c = config.huber_c;
    g_stitch_threshold = config.stitch_threshold;
    g_detrend_verbose = config.detrend_verbose;
}

const HeliosConfig& helios_get_config() {
    return g_helios_config;
}

// ---- Остальные глобальные переменные для обратной совместимости ----
bool g_enable_detrend = false;
bool g_enable_ortho = true;
int g_detrend_max_segments = 10;
int g_detrend_min_segment_len = 32;
double g_detrend_bic_threshold = 2.0;
double g_detrend_lambda = 0.99;
int g_detrend_max_order = 2;
bool g_detrend_auto_order = true;
int g_huber_iter = 5;
double g_huber_c = 1.345;
double g_stitch_threshold = 2.0;
bool g_detrend_verbose = false;

// Вспомогательные функции преобразования типов
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

static alw_string atom_params_to_string(const AtomSpec& spec) {
    std::stringstream ss;
    switch (spec.type) {
        case CHIRP:
            ss << "f0=" << spec.chirp.f0_hi << " beta=" << spec.chirp.beta_hi;
            break;
        case MORLET:
            ss << "sigma=" << spec.morlet.sigma << " mu=" << spec.morlet.mu << " w=" << spec.morlet.w_morlet;
            break;
        case DAMPED:
            ss << "alpha=" << spec.damped.alpha << " f0=" << spec.damped.f0;
            break;
        case QUASI:
            ss << "k=" << spec.quasi.k << " theta=" << spec.quasi.theta_hi;
            break;
        case STEP:
            ss << "pos=" << spec.step.position;
            break;
        case BSPLINE:
            ss << "scale=" << spec.bspline.scale << " pos=" << spec.bspline.position << " order=" << spec.bspline.order;
            break;
        case ANTI_DAMPED:
            ss << "alpha=" << spec.anti_damped.alpha << " f0=" << spec.anti_damped.f0;
            break;
        case SIGMOID:
            ss << "k=" << spec.sigmoid.k << " t0=" << spec.sigmoid.t0 << " dir=" << (spec.sigmoid.direction ? "up" : "down");
            break;
        case CHEBYSHEV:
            ss << "order=" << spec.quasi.k;
            break;
        case DOUBLE_SIGMOID:
            ss << "k=" << spec.sigmoid.k << " t0=" << spec.sigmoid.t0 << " t1=" << spec.sigmoid.t1;
            break;
        case GAUSSIAN:
            ss << "mu=" << spec.morlet.mu << " sigma=" << spec.morlet.sigma;
            break;
        case ERF:
            ss << "mu=" << spec.morlet.mu << " sigma=" << spec.morlet.sigma;
            break;
        case DAMPED_CHIRP:
            ss << "alpha=" << spec.damped.alpha << " f0=" << spec.chirp.f0_hi << " beta=" << spec.chirp.beta_hi;
            break;
        case SIGMOID_OSC:
            ss << "k=" << spec.sigmoid.k << " t0=" << spec.sigmoid.t0 << " f=" << spec.chirp.f0_hi;
            break;
        case EXP_GROWTH:
            ss << "alpha=" << spec.damped.alpha;
            break;
        case EXP_DECAY:
            ss << "alpha=" << spec.damped.alpha;
            break;
        case TANH:
            ss << "k=" << spec.sigmoid.k << " t0=" << spec.sigmoid.t0;
            break;
        case LORENTZIAN:
            ss << "mu=" << spec.morlet.mu << " gamma=" << spec.morlet.sigma;
            break;
        case POWER:
            ss << "alpha=" << spec.quasi.theta_hi;
            break;
        case HAAR:
            ss << "";
            break;
        case BESSEL:
            ss << "order=" << spec.quasi.k;
            break;
        default:
            ss << "";
            break;
    }
    return ss.str();
}

// =============================================================================
// ИНИЦИАЛИЗАЦИЯ GPU (с поддержкой потока)
// =============================================================================

void ensure_gpu_initialized(int frame_size, cudaStream_t stream) {
    std::call_once(g_gpu_init_flag, [&]() {
        CUDA_CHECK(cudaSetDevice(0));
        int deviceCount;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            throw std::runtime_error("No CUDA device found");
        }
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        init_bspline_constant_memory();

        // Инициализируем менеджер словарей, если ещё не создан
        if (!g_dict_manager) {
            g_dict_manager.reset(new DictionaryManager(4));
        }

        g_frame_size = frame_size;

        // Получаем дескрипторы и атомы через менеджер
        const auto& specs = g_dict_manager->get_descriptors();
        g_num_atoms = g_dict_manager->get_num_atoms();
        g_descriptors.resize(g_num_atoms);
        for (int i = 0; i < g_num_atoms; ++i) {
            g_descriptors[i].type_str = atom_type_to_string(specs[i].type);
            g_descriptors[i].params_str = atom_params_to_string(specs[i]);
        }

        // Загружаем словарь в глобальные указатели (по умолчанию)
        g_d_dict_hi_global = g_dict_manager->get_dictionary_hi(frame_size);
        g_d_dict_lo_global = g_dict_manager->get_dictionary_lo(frame_size);
        g_atom_norms_global = g_dict_manager->get_atom_norms(frame_size);
        g_global_num_atoms = g_num_atoms;
        g_global_frame_size = frame_size;

        g_gpu_initialized = true;
    });
}

// =============================================================================
// ФУНКЦИЯ ОБУЧЕНИЯ СЛОВАРЯ (EOD) — ПОЛНАЯ РЕАЛИЗАЦИЯ, ЭКСПОРТИРУЕТСЯ
// =============================================================================

void train_dictionary_with_eod(
    const double* d_signal_hi,
    const double* d_signal_lo,
    int N,
    int frame_size,
    int hop_size,
    int num_atoms,
    cudaStream_t stream,
    double*& out_dict_hi,
    double*& out_dict_lo,
    alw_vector<double>& out_atom_norms)
{
    const EODConfig& eod_cfg = g_helios_config.eod;
    if (!eod_cfg.enable) return;

    // Если уже обучено, пропускаем (если не принудительно)
    if (g_eod_trained) {
        ALW_LOG_INFO("EOD: Dictionary already trained, skipping.");
        return;
    }

    // Проверяем, что данных достаточно для обучения
    int min_samples = frame_size * 2;
    if (N < min_samples) {
        ALW_LOG_WARN("EOD: Not enough samples (%d < %d) for dictionary training, skipping.", N, min_samples);
        return;
    }

    ALW_LOG_INFO("EOD: Starting dictionary training on %d samples, frame_size=%d, num_atoms=%d",
                 N, frame_size, num_atoms);

    // Копируем сигнал на хост (если он на GPU)
    alw_vector<double> host_signal(N);
    CUDA_CHECK(cudaMemcpyAsync(host_signal.data(), d_signal_hi, N * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Подготавливаем параметры EOP
    EOP_Params eop_params;
    eop_params.alpha = g_helios_config.eop.alpha;
    eop_params.gamma = g_helios_config.eop.gamma;
    eop_params.theta = g_helios_config.eop.theta;
    eop_params.max_iterations = g_helios_config.max_iterations;
    eop_params.use_final_regression = !g_helios_config.no_final_regression;

    // Подготавливаем параметры EOD
    EOD_Params eod_params;
    eod_params.max_iter = eod_cfg.max_iter;
    eod_params.learning_rate = eod_cfg.learning_rate;
    eod_params.epsilon = 1e-6;
    eod_params.tolerance = eod_cfg.tolerance;
    eod_params.hop_size = hop_size;
    eod_params.max_atoms_per_frame = eod_cfg.max_atoms_per_frame;
    eod_params.use_aeds = true;
    eod_params.verbose = eod_cfg.verbose;
    eod_params.regularization_lambda = eod_cfg.regularization_lambda;
    eod_params.sigma_noise = eod_cfg.sigma_noise;

    // Инициализация словаря (используем текущий словарь как начальный, если он есть)
    alw_vector<double> init_dict;
    if (g_d_dict_hi_global != nullptr) {
        init_dict.resize((size_t)num_atoms * frame_size);
        CUDA_CHECK(cudaMemcpyAsync(init_dict.data(), g_d_dict_hi_global,
                                   (size_t)num_atoms * frame_size * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ALW_LOG_INFO("EOD: Using existing dictionary as initialization.");
    } else {
        ALW_LOG_INFO("EOD: No existing dictionary, will use random initialization.");
    }

    // Выделяем память для обученного словаря
    alw_vector<double> learned_dict((size_t)num_atoms * frame_size);
    alw_vector<double> learned_norms;

    // Запускаем EOD обучение
    eod_learn_dictionary_gpu(
        host_signal.data(),
        N,
        frame_size,
        num_atoms,
        init_dict.empty() ? nullptr : init_dict.data(),
        eop_params,
        eod_params,
        learned_dict.data(),
        learned_norms
    );

    // Проверяем результат
    bool valid = true;
    for (size_t i = 0; i < learned_dict.size(); ++i) {
        if (isnan(learned_dict[i]) || isinf(learned_dict[i])) {
            valid = false;
            break;
        }
    }
    if (!valid) {
        ALW_LOG_ERROR("EOD: Learned dictionary contains NaN/Inf, keeping old dictionary.");
        return;
    }

    // Копируем обученный словарь на GPU в переданные буферы
    CUDA_CHECK(cudaMemcpyAsync(out_dict_hi, learned_dict.data(),
                               (size_t)num_atoms * frame_size * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(out_dict_lo, learned_dict.data(),
                               (size_t)num_atoms * frame_size * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    out_atom_norms = learned_norms;
    g_eod_trained = true;

    ALW_LOG_INFO("EOD: Dictionary training completed successfully, learned %d atoms.", num_atoms);
    ALW_LOG_INFO("EOD: Norms range: min=%.6e, max=%.6e",
                 *std::min_element(learned_norms.begin(), learned_norms.end()),
                 *std::max_element(learned_norms.begin(), learned_norms.end()));
}

// =============================================================================
// run_alw_helios_with_volume_bars — ПОЛНАЯ РЕАЛИЗАЦИЯ С EOD
// =============================================================================

void run_alw_helios_with_volume_bars(
    const double* ticks_price,
    const double* ticks_volume,
    int num_ticks,
    alw_vector<DetectedEvent>& all_events,
    alw_vector<alw_vector<double>>& out_coeffs_hi,
    alw_vector<alw_vector<double>>& out_coeffs_lo,
    cudaStream_t stream)
{
    const HeliosConfig& cfg = g_helios_config;
    if (!cfg.use_volume_bars) {
        ALW_LOG_ERROR("run_alw_helios_with_volume_bars: volume bars not enabled in config");
        return;
    }

    if (ticks_price == nullptr || ticks_volume == nullptr || num_ticks <= 0) {
        ALW_LOG_ERROR("run_alw_helios_with_volume_bars: invalid tick data");
        return;
    }

    try {
        // 1. Построение Volume Bars
        VolumeBarBuilder builder(cfg.volume_params);
        for (int i = 0; i < num_ticks; ++i) {
            double price = ticks_price[i];
            double volume = ticks_volume[i];
            if (std::isnan(price) || std::isinf(price)) {
                ALW_LOG_WARN("run_alw_helios_with_volume_bars: skipping NaN/Inf price at tick %d", i);
                continue;
            }
            if (volume <= 0.0) {
                continue;
            }
            builder.add_tick(price, volume, 0);
        }

        const alw_vector<VolumeBar>& bars = builder.get_bars();
        if (bars.empty()) {
            ALW_LOG_WARN("run_alw_helios_with_volume_bars: no volume bars generated");
            return;
        }

        alw_vector<double> prices = volume_bars_to_prices(bars);
        int N = (int)prices.size();

        if (N < cfg.frame_size) {
            ALW_LOG_WARN("run_alw_helios_with_volume_bars: too few volume bars (%d) for frame_size %d", N, cfg.frame_size);
            return;
        }

        // 2. Копируем данные на GPU
        CudaPoolGuard<double> d_prices(N);
        CUDA_CHECK(cudaMemcpyAsync(d_prices.get(), prices.data(), N * sizeof(double), cudaMemcpyHostToDevice, stream));
        CudaPoolGuard<double> d_prices_lo(N);
        CUDA_CHECK(cudaMemsetAsync(d_prices_lo.get(), 0, N * sizeof(double), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 3. Инициализация GPU и словаря
        ensure_gpu_initialized(cfg.frame_size, stream);

        // 4. EOD: обучение словаря на первых данных (если включено и ещё не обучено)
        if (cfg.eod.enable && !g_eod_trained) {
            // Используем обучающие данные (первые learn_samples)
            int learn_N = std::min(N, cfg.eod.learn_samples > 0 ? cfg.eod.learn_samples : 10000);
            if (learn_N > cfg.frame_size * 2) {
                train_dictionary_with_eod(
                    d_prices.get(),
                    d_prices_lo.get(),
                    learn_N,
                    cfg.frame_size,
                    cfg.hop_size,
                    g_global_num_atoms,
                    stream,
                    g_d_dict_hi_global,
                    g_d_dict_lo_global,
                    g_atom_norms_global
                );
            } else {
                ALW_LOG_WARN("EOD: Not enough samples for training (%d < %d), skipping.", learn_N, cfg.frame_size * 2);
            }
        }

        // 5. Запуск разложения (использует глобальные словари)
        run_alw_helios_ortho(
            d_prices.get(),
            d_prices_lo.get(),
            N,
            cfg.frame_size,
            cfg.hop_size,
            g_d_dict_hi_global,
            g_d_dict_lo_global,
            g_global_num_atoms,
            all_events,
            g_atom_norms_global,
            g_descriptors,
            cfg.max_iterations,
            cfg.gamma_local,
            cfg.gamma_global,
            cfg.gamma_coherence,
            cfg.use_fixed_sigma,
            cfg.fixed_sigma,
            cfg.no_final_regression,
            nullptr,
            cfg.enable_ortho,
            out_coeffs_hi,
            out_coeffs_lo,
            stream
        );

        ALW_LOG_INFO("run_alw_helios_with_volume_bars: processed %d ticks -> %d volume bars -> %d events",
                     num_ticks, N, (int)all_events.size());

    } catch (const std::exception& e) {
        ALW_LOG_ERROR("run_alw_helios_with_volume_bars: exception: %s", e.what());
    }
}

// =============================================================================
// СОХРАНЕНИЕ СЛОВАРЯ В ФАЙЛ
// =============================================================================

void save_dictionary_to_file(const alw_string& filename,
                             int num_atoms, int frame_size,
                             const double* dict_hi, const double* dict_lo,
                             const alw_vector<double>& atom_norms) {
    if (dict_hi == nullptr || dict_lo == nullptr) {
        ALW_LOG_ERROR("save_dictionary_to_file: null dictionary pointers");
        throw std::runtime_error("Null dictionary pointers");
    }
    if (num_atoms <= 0 || frame_size <= 0) {
        ALW_LOG_ERROR("save_dictionary_to_file: invalid dimensions");
        throw std::runtime_error("Invalid dimensions");
    }

    std::fstream f(filename.c_str(), std::ios::out | std::ios::binary);
    if (!f.is_open()) {
        ALW_LOG_ERROR("Cannot open dictionary file for writing: %s", filename.c_str());
        throw std::runtime_error("Cannot open dictionary file for writing");
    }

    uint32_t magic = DICT_MAGIC;
    uint32_t version = DICT_VERSION;
    int32_t n_atoms = num_atoms;
    int32_t f_size = frame_size;

    f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    f.write(reinterpret_cast<const char*>(&version), sizeof(version));
    f.write(reinterpret_cast<const char*>(&n_atoms), sizeof(n_atoms));
    f.write(reinterpret_cast<const char*>(&f_size), sizeof(f_size));

    size_t data_size = (size_t)num_atoms * frame_size;
    f.write(reinterpret_cast<const char*>(dict_hi), data_size * sizeof(double));
    f.write(reinterpret_cast<const char*>(dict_lo), data_size * sizeof(double));

    // Сохраняем нормы
    if (!atom_norms.empty()) {
        f.write(reinterpret_cast<const char*>(atom_norms.data()), num_atoms * sizeof(double));
    } else {
        ALW_LOG_WARN("save_dictionary_to_file: atom_norms is empty, saving zeros.");
        alw_vector<double> zeros(num_atoms, 0.0);
        f.write(reinterpret_cast<const char*>(zeros.data()), num_atoms * sizeof(double));
    }

    if (!f.good()) {
        ALW_LOG_ERROR("Error writing dictionary file");
        throw std::runtime_error("Error writing dictionary file");
    }
    f.close();
    ALW_LOG_INFO("Dictionary saved to %s (%d atoms, %d frame_size)", filename.c_str(), num_atoms, frame_size);
}

// =============================================================================
// ЗАГРУЗКА СЛОВАРЯ ИЗ ФАЙЛА
// =============================================================================

bool load_dictionary_from_file(const alw_string& filename,
                               int expected_num_atoms, int expected_frame_size,
                               double* dict_hi, double* dict_lo,
                               alw_vector<double>& atom_norms) {
    if (dict_hi == nullptr || dict_lo == nullptr) {
        ALW_LOG_ERROR("load_dictionary_from_file: null dictionary pointers");
        return false;
    }

    std::fstream f(filename.c_str(), std::ios::in | std::ios::binary);
    if (!f.is_open()) {
        ALW_LOG_WARN("Cannot open dictionary file for reading: %s", filename.c_str());
        return false;
    }

    // Проверяем размер файла
    f.seekg(0, std::ios::end);
    std::streampos file_size = f.tellg();
    f.seekg(0, std::ios::beg);

    uint32_t magic, version;
    int32_t n_atoms, f_size;

    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != DICT_MAGIC) {
        ALW_LOG_WARN("Invalid magic in dictionary file: expected 0x%08X, got 0x%08X", DICT_MAGIC, magic);
        return false;
    }

    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != DICT_VERSION) {
        ALW_LOG_WARN("Unsupported dictionary version: %d (expected %d)", version, DICT_VERSION);
        return false;
    }

    f.read(reinterpret_cast<char*>(&n_atoms), sizeof(n_atoms));
    f.read(reinterpret_cast<char*>(&f_size), sizeof(f_size));

    if (n_atoms != expected_num_atoms || f_size != expected_frame_size) {
        ALW_LOG_WARN("Dictionary size mismatch: expected %dx%d, got %dx%d",
                     expected_num_atoms, expected_frame_size, n_atoms, f_size);
        return false;
    }

    // Проверяем, что файл достаточно велик
    size_t expected_size = sizeof(magic) + sizeof(version) + sizeof(n_atoms) + sizeof(f_size)
                           + 2 * (size_t)n_atoms * f_size * sizeof(double)
                           + n_atoms * sizeof(double);
    if (file_size < (std::streampos)expected_size) {
        ALW_LOG_WARN("Dictionary file truncated: size %zu < expected %zu", (size_t)file_size, expected_size);
        return false;
    }

    size_t data_size = (size_t)n_atoms * f_size;
    f.read(reinterpret_cast<char*>(dict_hi), data_size * sizeof(double));
    f.read(reinterpret_cast<char*>(dict_lo), data_size * sizeof(double));

    atom_norms.resize(n_atoms);
    f.read(reinterpret_cast<char*>(atom_norms.data()), n_atoms * sizeof(double));

    if (!f.good()) {
        ALW_LOG_ERROR("Error reading dictionary file");
        return false;
    }

    // Проверяем данные на NaN/Inf
    bool valid = true;
    for (size_t i = 0; i < data_size; ++i) {
        if (isnan(dict_hi[i]) || isinf(dict_hi[i]) || isnan(dict_lo[i]) || isinf(dict_lo[i])) {
            ALW_LOG_WARN("Loaded dictionary contains NaN/Inf at index %zu", i);
            valid = false;
            break;
        }
    }
    for (int i = 0; i < n_atoms; ++i) {
        if (isnan(atom_norms[i]) || isinf(atom_norms[i]) || atom_norms[i] <= 0.0) {
            ALW_LOG_WARN("Loaded atom_norms contains invalid value at index %d: %.6e", i, atom_norms[i]);
            valid = false;
            break;
        }
    }

    if (!valid) {
        ALW_LOG_ERROR("Loaded dictionary contains invalid data.");
        return false;
    }

    f.close();
    ALW_LOG_INFO("Dictionary loaded from %s (%d atoms, %d frame_size)", filename.c_str(), n_atoms, f_size);
    return true;
}
