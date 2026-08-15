#ifndef ALW_HELIOS_CORE_H
#define ALW_HELIOS_CORE_H

#include "alw_math.h"
#include "alw_dictionary.h"
#include "alw_helios_utils.h"
#include "helios_config.h"

// =============================================================================
// ГЛОБАЛЬНЫЙ КОНФИГ (доступен из других модулей через функции)
// =============================================================================

// Установить конфигурацию (сохраняется в глобальной переменной)
void helios_set_config(const HeliosConfig& config);

// Получить текущую конфигурацию (константная ссылка)
const HeliosConfig& helios_get_config();

// =============================================================================
// ГЛОБАЛЬНЫЕ УКАЗАТЕЛИ НА СЛОВАРЬ (используются в EOD и run_alw_helios_with_volume_bars)
// =============================================================================

extern double* g_d_dict_hi_global;
extern double* g_d_dict_lo_global;
extern alw_vector<double> g_atom_norms_global;
extern int g_global_num_atoms;
extern int g_global_frame_size;
extern bool g_eod_trained;   // флаг, что словарь уже обучен

// =============================================================================
// EOD (обучение словаря) – вызывается из run_alw_helios_ortho
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
    alw_vector<double>& out_atom_norms);

// =============================================================================
// ОСНОВНЫЕ ФУНКЦИИ РАЗЛОЖЕНИЯ С ПОДДЕРЖКОЙ ПОТОКА
// =============================================================================

void run_alw_helios_ortho(
    double* d_y_global_hi,
    double* d_y_global_lo,
    int N,
    int frame_size,
    int hop_size,
    double* d_dict_hi,
    double* d_dict_lo,
    int num_atoms,
    alw_vector<DetectedEvent>& all_events,
    alw_vector<double>& atom_norms,   // <-- УБРАН const
    const alw_vector<AtomDescriptor>& descriptors,
    int max_iterations,
    double gamma_local,
    double gamma_global,
    double gamma_coherence,
    bool use_fixed_sigma,
    double fixed_sigma,
    bool no_final_regression,
    alw_fstream* residual_file,
    bool enable_ortho,
    alw_vector<alw_vector<double>>& out_coeffs_hi,
    alw_vector<alw_vector<double>>& out_coeffs_lo,
    cudaStream_t stream = 0);

void run_alw_helios(
    double* d_y_global_hi,
    double* d_y_global_lo,
    int N,
    int frame_size,
    int hop_size,
    double* d_dict_hi,
    double* d_dict_lo,
    int num_atoms,
    alw_vector<DetectedEvent>& all_events,
    const alw_vector<double>& atom_norms,
    const alw_vector<AtomDescriptor>& descriptors,
    int max_iterations,
    double gamma_local,
    double gamma_global,
    double gamma_coherence,
    bool use_fixed_sigma,
    double fixed_sigma,
    bool no_final_regression,
    alw_fstream* residual_file,
    cudaStream_t stream = 0);

// =============================================================================
// СОХРАНЕНИЕ / ЗАГРУЗКА СЛОВАРЯ
// =============================================================================

void save_dictionary_to_file(const alw_string& filename,
                             int num_atoms, int frame_size,
                             const double* dict_hi, const double* dict_lo,
                             const alw_vector<double>& atom_norms);

bool load_dictionary_from_file(const alw_string& filename,
                               int expected_num_atoms, int expected_frame_size,
                               double* dict_hi, double* dict_lo,
                               alw_vector<double>& atom_norms);

// =============================================================================
// ИНИЦИАЛИЗАЦИЯ GPU (однократная) — с поддержкой потока
// =============================================================================

void ensure_gpu_initialized(int frame_size, cudaStream_t stream = 0);

// =============================================================================
// ОБРАБОТКА С VOLUME BARS (с поддержкой потока)
// =============================================================================

void run_alw_helios_with_volume_bars(
    const double* ticks_price,
    const double* ticks_volume,
    int num_ticks,
    alw_vector<DetectedEvent>& all_events,
    alw_vector<alw_vector<double>>& out_coeffs_hi,
    alw_vector<alw_vector<double>>& out_coeffs_lo,
    cudaStream_t stream = 0);

// =============================================================================
// ДЕТРЕНДИНГ С ПОДДЕРЖКОЙ ПОТОКА
// =============================================================================

void helios_apply_detrend(
    const double* y_hi,
    const double* y_lo,
    int N,
    double* out_hi,
    double* out_lo,
    cudaStream_t stream = 0);

// =============================================================================
// ГЛОБАЛЬНЫЕ НАСТРОЙКИ (управляются через экспорты, устаревают)
// =============================================================================

extern bool g_enable_detrend;
extern bool g_enable_ortho;
extern int g_detrend_max_segments;
extern int g_detrend_min_segment_len;
extern double g_detrend_bic_threshold;
extern double g_detrend_lambda;
extern int g_detrend_max_order;
extern bool g_detrend_auto_order;
extern int g_huber_iter;
extern double g_huber_c;
extern double g_stitch_threshold;
extern bool g_detrend_verbose;

#endif // ALW_HELIOS_CORE_H
