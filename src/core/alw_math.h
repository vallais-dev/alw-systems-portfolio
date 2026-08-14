// =============================================================================
// alw_math.h — единый заголовочный файл для всего проекта (ОБНОВЛЁН)
// Добавлены поля для кэширования числа обусловленности в OrthoParams
// =============================================================================

#ifndef ALW_MATH_H
#define ALW_MATH_H

// =============================================================================
// 1. БАЗОВЫЕ СИСТЕМНЫЕ ЗАГОЛОВКИ
// =============================================================================

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstdarg>
#include <string>
#include <functional>
#include <vector>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <fstream>

// =============================================================================
// 2. ТИП DOUBLE-DOUBLE (DD) И БАЗОВЫЕ ОПЕРАЦИИ
// =============================================================================

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

struct DD {
    double hi;
    double lo;
};

static inline HOST_DEVICE DD make_dd(double hi, double lo = 0.0) {
    return {hi, lo};
}

static inline HOST_DEVICE DD dd_add(DD a, DD b) {
    double t1 = a.hi + b.hi;
    double e  = t1 - a.hi;
    double t2 = ((b.hi - e) + (a.hi - (t1 - e))) + a.lo + b.lo;
    DD r;
    r.hi = t1 + t2;
    r.lo = t2 - (r.hi - t1);
    return r;
}

static inline HOST_DEVICE DD dd_sub(DD a, DD b) {
    double t1 = a.hi - b.hi;
    double e  = t1 - a.hi;
    double t2 = ((-b.hi - e) + (a.hi - (t1 - e))) + a.lo - b.lo;
    DD r;
    r.hi = t1 + t2;
    r.lo = t2 - (r.hi - t1);
    return r;
}

static inline HOST_DEVICE DD dd_mul(DD a, DD b) {
    DD r;
    double p = a.hi * b.hi;
    r.hi = p;
    r.lo = std::fma(a.hi, b.hi, -p) + a.hi * b.lo + a.lo * b.hi + a.lo * b.lo;
    return r;
}

static inline HOST_DEVICE DD dd_div(DD a, DD b) {
    double q = a.hi / b.hi;
    if (std::fabs(b.hi) < 1e-300) return {0.0, 0.0};
    DD p = dd_mul(b, make_dd(q, 0.0));
    DD r = make_dd(a.hi - p.hi, a.lo - p.lo);
    double q2 = r.hi / b.hi;
    return make_dd(q + q2, 0.0);
}

static inline HOST_DEVICE DD dd_abs(DD a) {
    return (a.hi < 0.0) ? make_dd(-a.hi, -a.lo) : a;
}

static inline HOST_DEVICE bool dd_less_than(DD a, double threshold) {
    return std::fabs(a.hi) < threshold;
}

static inline HOST_DEVICE bool dd_is_critically_small(DD a, double eps) {
    return std::fabs(a.hi) <= eps;
}

// =============================================================================
// 3. ОБЁРТКИ ДЛЯ УДОБСТВА (alw_*)
// =============================================================================

inline HOST_DEVICE DD alw_add_dd(DD a, DD b) { return dd_add(a, b); }
inline HOST_DEVICE DD alw_sub_dd(DD a, DD b) { return dd_sub(a, b); }
inline HOST_DEVICE DD alw_mul_dd(DD a, DD b) { return dd_mul(a, b); }
inline HOST_DEVICE DD alw_div_dd(DD a, DD b) { return dd_div(a, b); }
inline HOST_DEVICE DD alw_abs_dd(DD a) { return dd_abs(a); }
inline HOST_DEVICE bool alw_less_than(DD a, double threshold) { return dd_less_than(a, threshold); }
inline HOST_DEVICE bool alw_is_critically_small(DD a, double eps) { return dd_is_critically_small(a, eps); }

inline HOST_DEVICE void alw_mul_dd(double a_hi, double a_lo, double b_hi, double b_lo,
                                   double& out_hi, double& out_lo) {
    DD r = dd_mul(make_dd(a_hi, a_lo), make_dd(b_hi, b_lo));
    out_hi = r.hi; out_lo = r.lo;
}
inline HOST_DEVICE void alw_add_dd(double a_hi, double a_lo, double b_hi, double b_lo,
                                   double& out_hi, double& out_lo) {
    DD r = dd_add(make_dd(a_hi, a_lo), make_dd(b_hi, b_lo));
    out_hi = r.hi; out_lo = r.lo;
}
inline HOST_DEVICE void alw_sub_dd(double a_hi, double a_lo, double b_hi, double b_lo,
                                   double& out_hi, double& out_lo) {
    DD r = dd_sub(make_dd(a_hi, a_lo), make_dd(b_hi, b_lo));
    out_hi = r.hi; out_lo = r.lo;
}
inline HOST_DEVICE void alw_div_dd(double a_hi, double a_lo, double b_hi, double b_lo,
                                   double& out_hi, double& out_lo) {
    DD r = dd_div(make_dd(a_hi, a_lo), make_dd(b_hi, b_lo));
    out_hi = r.hi; out_lo = r.lo;
}
inline HOST_DEVICE double alw_abs_hi(double hi, double /*lo*/) {
    return std::fabs(hi);
}

// =============================================================================
// 4. СПЕЦИАЛЬНЫЕ DD-ФУНКЦИИ (sqrt, exp, log, sin_cos, abs, is_critically_small)
// =============================================================================

inline HOST_DEVICE void alw_sqrt_dd(double hi, double lo, double& out_hi, double& out_lo) {
    if (hi < 0.0) { out_hi = 0.0; out_lo = 0.0; return; }
    double s = ::sqrt(hi);
    out_hi = s;
    double diff = hi - s * s;
    if (diff < 0.0) diff = 0.0;
    out_lo = diff / (2.0 * s);
    if (isnan(out_lo) || isinf(out_lo)) out_lo = 0.0;
}

inline HOST_DEVICE void alw_exp_dd(double hi, double lo, double& out_hi, double& out_lo) {
    double e = ::exp(hi);
    out_hi = e;
    out_lo = e * lo;
    if (isnan(out_lo) || isinf(out_lo)) out_lo = 0.0;
}

inline HOST_DEVICE void alw_log_dd(double hi, double lo, double& out_hi, double& out_lo) {
    if (hi <= 0.0) { out_hi = 0.0; out_lo = 0.0; return; }
    double l = ::log(hi);
    out_hi = l;
    out_lo = lo / hi;
    if (isnan(out_lo) || isinf(out_lo)) out_lo = 0.0;
}

inline HOST_DEVICE void alw_sin_cos_dd(double hi, double lo,
                                       double& out_sin_hi, double& out_sin_lo,
                                       double& out_cos_hi, double& out_cos_lo) {
    double s = ::sin(hi);
    double c = ::cos(hi);
    out_sin_hi = s;
    out_sin_lo = c * lo;
    out_cos_hi = c;
    out_cos_lo = -s * lo;
    if (isnan(out_sin_lo) || isinf(out_sin_lo)) out_sin_lo = 0.0;
    if (isnan(out_cos_lo) || isinf(out_cos_lo)) out_cos_lo = 0.0;
}

inline HOST_DEVICE void alw_abs_dd(double hi, double lo, double& out_hi, double& out_lo) {
    if (hi < 0.0) { out_hi = -hi; out_lo = -lo; }
    else { out_hi = hi; out_lo = lo; }
}

inline HOST_DEVICE bool alw_is_critically_small(double hi, double /*lo*/, double eps) {
    return std::fabs(hi) <= eps;
}

// =============================================================================
// 5. МАТЕМАТИЧЕСКИЕ КОНСТАНТЫ
// =============================================================================

#define ALW_PI_HI 3.141592653589793
#define ALW_PI_LO 0.0
#define ALW_TWO_PI_1 6.283185307179586

#define MEDIAN_TO_SIGMA 0.67448975
#define DEFAULT_GAMMA_COHERENCE 0.90
#define MIN_EPSILON 1e-9

// =============================================================================
// 6. ЛОГИРОВАНИЕ И ТРАССИРОВКА (включая TelemetryLevel)
// =============================================================================

enum class TelemetryLevel { SILENT, ERROR, WARNING, INFO, DEBUG };

#ifndef AEDS_LOG_LEVEL
#define AEDS_LOG_LEVEL 5
#endif

#ifndef AEDS_ENABLE_TELEMETRY
#define AEDS_ENABLE_TELEMETRY 1
#endif

static std::mutex log_mutex;

inline uint64_t get_timestamp_us() {
    auto now = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return static_cast<uint64_t>(us);
}

inline void aeds_log_internal(int level, const char* file, int line, const char* func, const char* fmt, ...) {
#if AEDS_ENABLE_TELEMETRY
    if (level > AEDS_LOG_LEVEL) return;
    char buffer[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    std::lock_guard<std::mutex> lock(log_mutex);
    uint64_t ts = get_timestamp_us();
    const char* level_str[] = {"ERROR", "WARN ", "INFO ", "DEBUG", "TRACE"};
    const char* lvl = (level >= 0 && level <= 4) ? level_str[level] : "????";
    fprintf(stderr, "[%s][%s:%d %s] %s\n", lvl, file, line, func, buffer);
    fflush(stderr);
#endif
}

#define AEDS_LOGE(...) aeds_log_internal(0, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)
#define AEDS_LOGW(...) aeds_log_internal(1, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)
#define AEDS_LOGI(...) aeds_log_internal(2, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)
#define AEDS_LOGD(...) aeds_log_internal(3, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)
#define AEDS_LOGT(...) aeds_log_internal(4, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)

#define AEDS_TRACE_ENTER() AEDS_LOGT("--> enter")
#define AEDS_TRACE_EXIT()  AEDS_LOGT("<-- exit")

#define AEDS_TIME_BLOCK(name) \
    auto _start_##name = std::chrono::high_resolution_clock::now(); \
    AEDS_LOGD("Timer start: %s", #name); \
    auto _end_##name = std::chrono::high_resolution_clock::now(); \
    auto _dur_##name = std::chrono::duration_cast<std::chrono::microseconds>(_end_##name - _start_##name).count(); \
    AEDS_LOGD("Timer end: %s took %.3f ms", #name, _dur_##name / 1000.0);

// =============================================================================
// 7. АЛИАСЫ ДЛЯ ЛОГИРОВАНИЯ (ALW_LOG_* для совместимости)
// =============================================================================

#define ALW_LOG_ERROR  AEDS_LOGE
#define ALW_LOG_WARN   AEDS_LOGW
#define ALW_LOG_INFO   AEDS_LOGI
#define ALW_LOG_DEBUG  AEDS_LOGD

// =============================================================================
// 8. ОРТОГОНАЛИЗАЦИЯ (с добавленными полями для кэширования cond)
// =============================================================================

enum OrthoMode {
    ORTHO_AUTO,
    ORTHO_MGS,
    ORTHO_HOUSEHOLDER,
    ORTHO_HYBRID
};

struct OrthoParams {
    OrthoMode mode = ORTHO_AUTO;
    bool verbose = false;
    int householder_block = 32;
    int max_householder_iter = 5;
    double tolerance = 1e-12;
    int svd_submatrix_size = 32;
    double svd_tolerance = 1e-14;
    int max_svd_iter = 20;
    bool force_svd = false;
    bool save_ortho_stats = false;

    // --- НОВЫЕ ПОЛЯ ДЛЯ КЭШИРОВАНИЯ cond ---
    bool use_cond_cache = true;          // включить кэширование
    int cond_cache_interval = 32;        // пересчитывать каждые N вызовов
    double cached_cond = -1.0;           // сохранённое значение ( -1 означает отсутствие)
    int cond_call_counter = 0;           // счётчик вызовов для обновления
};

// =============================================================================
// 9. СТРУКТУРЫ ДАННЫХ
// =============================================================================

struct AtomDescriptor {
    std::string type_str;
    std::string params_str;
};

struct DetectedEvent {
    int atom_index;
    double amplitude_hi;
    double amplitude_lo = 0.0;
    double phase_hi = 0.0;
    double phase_lo = 0.0;
    double energy_hi = 0.0;
    double energy_lo = 0.0;
    int frame_start = 0;
    std::string atom_type_str;
    std::string atom_params_str;
};

struct BinaryEvent {};
struct Segment {};
struct GpuSegment {};
struct SegmentStats {};
struct DetrendStats {};

// =============================================================================
// 10. КОНТЕЙНЕРЫ И АЛИАСЫ
// =============================================================================

template <typename T>
using alw_vector = std::vector<T>;

using alw_string = std::string;

// =============================================================================
// 11. ПОТОКОВЫЕ АЛИАСЫ
// =============================================================================

using alw_fstream = std::fstream;

namespace alw_ios_openmode {
    static constexpr auto out    = std::ios::out;
    static constexpr auto binary = std::ios::binary;
    static constexpr auto in     = std::ios::in;
    static constexpr auto trunc  = std::ios::trunc;
    static constexpr auto app    = std::ios::app;
}

// =============================================================================
// 12. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ РАБОТЫ С ВЕКТОРАМИ (хост)
// =============================================================================

inline double alw_median_host(const double* data, int n) {
    if (n <= 0) return 0.0;
    alw_vector<double> sorted(data, data + n);
    std::sort(sorted.begin(), sorted.end());
    if (n % 2 == 0) {
        return (sorted[n/2 - 1] + sorted[n/2]) * 0.5;
    } else {
        return sorted[n/2];
    }
}

inline std::string alw_to_string(int value) {
    return std::to_string(value);
}

inline std::string alw_to_string(double value) {
    return std::to_string(value);
}

inline std::string alw_to_string(size_t value) {
    return std::to_string(value);
}

// =============================================================================
// 13. МАКРОСЫ И УТИЛИТЫ ДЛЯ CUDA
// =============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

template <typename T>
class CudaPoolGuard {
public:
    CudaPoolGuard() : ptr_(nullptr), count_(0) {}

    explicit CudaPoolGuard(size_t count) : ptr_(nullptr), count_(count) {
        if (count > 0) {
            cudaError_t err = cudaMalloc(&ptr_, count * sizeof(T));
            if (err != cudaSuccess) {
                ptr_ = nullptr;
                count_ = 0;
                fprintf(stderr, "CudaPoolGuard: cudaMalloc failed: %s\n", cudaGetErrorString(err));
            }
        }
    }

    CudaPoolGuard(const CudaPoolGuard&) = delete;
    CudaPoolGuard& operator=(const CudaPoolGuard&) = delete;

    CudaPoolGuard(CudaPoolGuard&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    CudaPoolGuard& operator=(CudaPoolGuard&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    ~CudaPoolGuard() { reset(); }

    T* get() const { return ptr_; }
    bool is_valid() const { return ptr_ != nullptr; }
    void reset() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }
    size_t size() const { return count_; }

private:
    T* ptr_;
    size_t count_;
};

class GpuMemoryPoolManager {
public:
    static void initialize(size_t /*size*/) {}
};

// =============================================================================
// 14. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// =============================================================================

inline double& alw_get_min_epsilon() {
    static double eps = 1e-9;
    return eps;
}
inline void alw_set_min_epsilon(double e) {
    alw_get_min_epsilon() = e;
}

// =============================================================================
// 15. УСТРОЙСТВЕННЫЙ РЕШАТЕЛЬ AEDS (in‑kernel, до 4×4)
// =============================================================================

static inline __device__ bool aeds_solve_dd_device_advanced(
    double* A_hi, double* A_lo,
    double* b_hi, double* b_lo,
    double* x_hi, double* x_lo,
    int n,
    int max_iter = 100,
    int refine_iter = 2)
{
    if (n <= 0) return true;
    if (n > 4) return false;

    for (int i = 0; i < n * n; ++i) {
        if (isnan(A_hi[i]) || isinf(A_hi[i]) || isnan(A_lo[i]) || isinf(A_lo[i]))
            return false;
    }
    for (int i = 0; i < n; ++i) {
        if (isnan(b_hi[i]) || isinf(b_hi[i]) || isnan(b_lo[i]) || isinf(b_lo[i]))
            return false;
    }

    DD A_cmaj[16];
    DD b_dd[4];
    DD L[4];
    for (int j = 0; j < n; ++j) {
        DD accum_L = {0.0, 0.0};
        for (int i = 0; i < n; ++i) {
            DD a_elem = {A_hi[i * n + j], A_lo[i * n + j]};
            A_cmaj[j * n + i] = a_elem;
            DD sqr = alw_mul_dd(a_elem, a_elem);
            accum_L = alw_add_dd(accum_L, sqr);
        }
        L[j] = accum_L;
        b_dd[j] = {b_hi[j], b_lo[j]};
        x_hi[j] = 0.0; x_lo[j] = 0.0;
    }

    double max_L = 0.0, min_L = 1e300;
    for (int i = 0; i < n; ++i) {
        double val = L[i].hi;
        if (isnan(val) || isinf(val)) val = 0.0;
        if (val > max_L) max_L = val;
        if (val < min_L && val > 1e-30) min_L = val;
    }
    double cond_est = (min_L > 1e-30) ? (max_L / min_L) : 1e15;
    if (isnan(cond_est) || isinf(cond_est)) cond_est = 1e15;

    const double REG_THRESHOLD = 1e6;
    double lambda = 0.0;
    if (cond_est > REG_THRESHOLD) {
        lambda = 1e-6 * max_L;
        if (lambda < 1e-12) lambda = 1e-12;
        for (int i = 0; i < n; ++i) {
            L[i] = alw_add_dd(L[i], {lambda, 0.0});
        }
        #ifdef AEDS_DEVICE_DEBUG
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("[aeds_solve_dd_device] cond=%.2e, regularization lambda=%.2e\n", cond_est, lambda);
        }
        #endif
    }

    DD x_old[4], x_new[4];
    for (int i = 0; i < n; ++i) {
        x_old[i] = {x_hi[i], x_lo[i]};
        x_new[i] = {x_hi[i], x_lo[i]};
    }

    double omega = 1.0;
    DD prev_grad[4] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

    for (int iter = 0; iter < max_iter; ++iter) {
        DD r[4];
        for (int i = 0; i < n; ++i) {
            DD ax = {0.0, 0.0};
            for (int j = 0; j < n; ++j) {
                DD a_elem = A_cmaj[j * n + i];
                DD prod = alw_mul_dd(a_elem, x_old[j]);
                ax = alw_add_dd(ax, prod);
            }
            r[i] = alw_sub_dd(ax, b_dd[i]);
        }

        double r_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            double val = r[i].hi;
            r_norm += val * val;
        }
        if (r_norm < 1e-30) break;

        for (int idx = 0; idx < n; ++idx) {
            DD grad = {0.0, 0.0};
            const int col_offset = idx * n;
            for (int j = 0; j < n; ++j) {
                DD a_elem = A_cmaj[col_offset + j];
                grad = alw_add_dd(grad, alw_mul_dd(a_elem, r[j]));
            }

            DD L_safe = L[idx];
            if (alw_is_critically_small(L_safe, 1e-28)) {
                DD alpha = alw_mul_dd(alw_abs_dd(grad), {1e-6, 0.0});
                L_safe = alw_add_dd(L_safe, alpha);
            }

            if (iter > 0) {
                double dot = grad.hi * prev_grad[idx].hi;
                if (dot < 0.0) {
                    omega = fmax(omega * 0.5, 0.01);
                } else {
                    omega = fmin(omega * 1.1, 1.9);
                }
                if (isnan(omega) || isinf(omega)) omega = 0.01;
            }
            prev_grad[idx] = grad;

            DD step = alw_mul_dd(grad, {omega, 0.0});
            step = alw_div_dd(step, L_safe);
            x_new[idx] = alw_sub_dd(x_old[idx], step);
        }

        for (int i = 0; i < n; ++i) x_old[i] = x_new[i];
    }

    for (int i = 0; i < n; ++i) {
        x_hi[i] = x_new[i].hi;
        x_lo[i] = x_new[i].lo;
        if (isnan(x_hi[i]) || isinf(x_hi[i])) return false;
    }

    for (int it = 0; it < refine_iter; ++it) {
        DD r[4];
        for (int i = 0; i < n; ++i) {
            DD ax = {0.0, 0.0};
            for (int j = 0; j < n; ++j) {
                DD a_elem = A_cmaj[j * n + i];
                DD prod = alw_mul_dd(a_elem, {x_hi[j], x_lo[j]});
                ax = alw_add_dd(ax, prod);
            }
            r[i] = alw_sub_dd(b_dd[i], ax);
        }

        double r_norm = 0.0;
        for (int i = 0; i < n; ++i) r_norm += r[i].hi * r[i].hi;
        if (r_norm < 1e-30) break;

        double dx_hi[4] = {0.0}, dx_lo[4] = {0.0};
        double r_hi[4], r_lo[4];
        for (int i = 0; i < n; ++i) { r_hi[i] = r[i].hi; r_lo[i] = r[i].lo; }

        bool ok = aeds_solve_dd_device_advanced(
            A_hi, A_lo, r_hi, r_lo, dx_hi, dx_lo, n, 20, 0
        );
        if (!ok) break;

        for (int i = 0; i < n; ++i) {
            DD new_x = alw_add_dd({x_hi[i], x_lo[i]}, {dx_hi[i], dx_lo[i]});
            x_hi[i] = new_x.hi;
            x_lo[i] = new_x.lo;
            if (isnan(x_hi[i]) || isinf(x_hi[i])) return false;
        }
    }

    return true;
}

static inline __device__ bool aeds_solve_dd_device(
    double* A_hi, double* A_lo,
    double* b_hi, double* b_lo,
    double* x_hi, double* x_lo,
    int n)
{
    return aeds_solve_dd_device_advanced(A_hi, A_lo, b_hi, b_lo, x_hi, x_lo, n, 100, 2);
}

#endif // ALW_MATH_H
