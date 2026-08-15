// =============================================================================
// alw_dictionary.h — атомарный словарь Helios (ОБНОВЛЁН)
// ДОБАВЛЕНО: DictionaryManager для LRU-кэширования словарей под разные frame_size
// =============================================================================

#ifndef ALW_DICTIONARY_H
#define ALW_DICTIONARY_H

#include "alw_math.h"
#include <cstddef>
#include <unordered_map>
#include <list>
#include <mutex>
#include <memory>

// =============================================================================
// КОНСТАНТЫ ДЛЯ СЕРИАЛИЗАЦИИ СЛОВАРЯ
// =============================================================================

#define DICT_MAGIC  0x4C454152  // "LEAR"
#define DICT_VERSION 1

// =============================================================================
// ТИПЫ АТОМОВ
// =============================================================================
enum AtomType {
    QUASI,
    CHIRP,
    MORLET,
    DAMPED,
    STEP,
    BSPLINE,
    ANTI_DAMPED,
    SIGMOID,
    LEARNED,
    CHEBYSHEV,
    DOUBLE_SIGMOID,
    GAUSSIAN,
    ERF,
    DAMPED_CHIRP,
    SIGMOID_OSC,
    EXP_GROWTH,
    EXP_DECAY,
    TANH,
    LORENTZIAN,
    POWER,
    HAAR,
    BESSEL
};

// =============================================================================
// ПАРАМЕТРЫ АТОМОВ
// =============================================================================
struct QuasiParams {
    int k;
    double theta_hi;
    double theta_lo;
};

struct ChirpParams {
    double f0_hi, f0_lo;
    double beta_hi, beta_lo;
};

struct MorletParams {
    double sigma;
    double mu;
    double w_morlet;
};

struct DampedParams {
    double alpha;
    double f0;
};

struct StepParams {
    double position;
};

struct BSplineParams {
    double scale;
    double position;
    int order;
};

struct AntiDampedParams {
    double alpha;
    double f0;
};

struct SigmoidParams {
    double k;
    double t0;
    double t1;
    bool direction;
};

// СПЕЦИФИКАЦИЯ АТОМА
struct AtomSpec {
    AtomType type;
    bool is_pair;
    union {
        QuasiParams quasi;
        ChirpParams chirp;
        MorletParams morlet;
        DampedParams damped;
        StepParams step;
        BSplineParams bspline;
        AntiDampedParams anti_damped;
        SigmoidParams sigmoid;
    };
};

// =============================================================================
// КЛАСС МЕНЕДЖЕРА СЛОВАРЕЙ С LRU-КЭШЕМ
// =============================================================================

class DictionaryManager {
public:
    // Конструктор с параметром максимального размера кэша (по умолчанию 4 словаря)
    explicit DictionaryManager(size_t max_cache_size = 4);

    // Получить указатель на словарь (hi-компонента) для заданного frame_size
    // Если словарь отсутствует в кэше, он генерируется и загружается в GPU
    double* get_dictionary_hi(int frame_size);

    // Получить указатель на словарь (lo-компонента) для заданного frame_size
    double* get_dictionary_lo(int frame_size);

    // Получить число атомов (одинаково для всех словарей)
    int get_num_atoms() const { return num_atoms_; }

    // Получить нормы атомов для текущего словаря
    const alw_vector<double>& get_atom_norms(int frame_size);

    // Получить дескрипторы атомов
    const alw_vector<AtomSpec>& get_descriptors() const { return descriptors_; }

    // Принудительно очистить кэш (освободить всю GPU-память)
    void clear_cache();

    // Установить максимальный размер кэша
    void set_max_cache_size(size_t max_size) { max_cache_size_ = max_size; }

private:
    // Структура для хранения одного словаря в кэше
    struct DictionaryEntry {
        std::unique_ptr<CudaPoolGuard<double>> dict_hi;   // hi-компонента
        std::unique_ptr<CudaPoolGuard<double>> dict_lo;   // lo-компонента
        alw_vector<double> atom_norms;                    // нормы атомов
        int frame_size;                                   // размер фрейма
        size_t last_access;                               // счётчик для LRU

        DictionaryEntry() : frame_size(0), last_access(0) {}
        DictionaryEntry(int fs, size_t access)
            : frame_size(fs), last_access(access) {}
    };

    // Построить и загрузить словарь для заданного frame_size
    void build_and_upload_dictionary(int frame_size);

    // Обновить счётчик доступа для элемента
    void touch_entry(int frame_size);

    // Применить LRU-политику (удалить наименее используемый элемент)
    void evict_lru();

    // Генерация базового словаря (один раз при первом вызове)
    void ensure_descriptors_initialized();

    // Параметры
    size_t max_cache_size_;
    mutable std::mutex mutex_;

    // Кэш словарей
    std::unordered_map<int, DictionaryEntry> cache_;
    // Список для LRU (хранит frame_size в порядке доступа)
    std::list<int> lru_list_;
    // Счётчик доступа (монотонно возрастает)
    size_t access_counter_ = 0;

    // Глобальные данные словаря (одинаковы для всех frame_size)
    alw_vector<AtomSpec> descriptors_;
    int num_atoms_ = 0;
    bool descriptors_initialized_ = false;
};

// =============================================================================
// ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР МЕНЕДЖЕРА (объявлен extern, определён в .cu)
// =============================================================================

extern std::unique_ptr<DictionaryManager> g_dict_manager;

// =============================================================================
// ЭКСПОРТИРУЕМЫЕ ФУНКЦИИ (устаревшие, сохранены для обратной совместимости)
// =============================================================================

void init_bspline_constant_memory();
alw_vector<AtomSpec> create_full_dictionary();
void build_dictionary_gpu(const alw_vector<AtomSpec>& descriptors,
                          int frame_size,
                          double* d_dict_hi,
                          double* d_dict_lo,
                          alw_vector<double>& atom_norms);
double normalize_atom_gpu(double* d_atom_hi, double* d_atom_lo, int frame_size);

void set_dict_config(const void* buffer, size_t size);
bool is_custom_dict_enabled();
void reset_dict_config();
bool is_learned_dict_enabled();
const alw_vector<double>& get_learned_dict();
const alw_vector<double>& get_learned_norms();
void load_learned_dict_to_gpu(int frame_size,
                              double* d_dict_hi,
                              double* d_dict_lo,
                              alw_vector<double>& atom_norms);

#endif // ALW_DICTIONARY_H
