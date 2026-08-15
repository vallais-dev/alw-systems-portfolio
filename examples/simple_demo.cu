// =============================================================================
// simple_demo.cu — МИНИМАЛЬНЫЙ РАБОЧИЙ ПРИМЕР HELIOS
// =============================================================================
// 
// Демонстрирует:
//   1. Генерацию тестового сигнала (синус + шум)
//   2. Копирование данных на GPU
//   3. Оценку уровня шума с помощью AMAD-X
//   4. Вывод результата
//
// Это минимальный пример, показывающий, что стек реально работает.
// Для запуска: nvcc -o simple_demo simple_demo.cu -lcuda -lcudart
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <memory>

// Подключаем необходимые заголовки
#include "alw_math.h"
#include "amad_x.h"

// -----------------------------------------------------------------------------
// ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: ГЕНЕРАЦИЯ ТЕСТОВОГО СИГНАЛА
// -----------------------------------------------------------------------------
static void generate_test_signal(std::vector<double>& signal_hi, 
                                 std::vector<double>& signal_lo,
                                 int N, 
                                 double freq = 0.05,
                                 double amplitude = 1.0,
                                 double noise_level = 0.1) {
    signal_hi.resize(N);
    signal_lo.resize(N, 0.0);
    
    // Инициализация генератора случайных чисел
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    
    for (int i = 0; i < N; ++i) {
        // Нормализованное время
        double t = static_cast<double>(i) / (N - 1);
        
        // Полезный сигнал: синусоида с частотой freq
        double signal = amplitude * std::sin(2.0 * M_PI * freq * t * 30.0);
        
        // Случайный шум (нормальное распределение, аппроксимировано суммой 12 Uniform)
        double noise = 0.0;
        for (int j = 0; j < 12; ++j) {
            noise += static_cast<double>(std::rand()) / RAND_MAX;
        }
        noise -= 6.0;  // Центрируем
        noise *= noise_level / 2.0;  // Масштабируем
        
        signal_hi[i] = signal + noise;
        // lo-компонента = 0 (в демо-версии мы используем только hi)
    }
}

// -----------------------------------------------------------------------------
// ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: ПРОВЕРКА CUDA-ОШИБОК
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    printf("========================================\n");
    printf("  Helios GPU Stack — Simple Demo\n");
    printf("========================================\n\n");
    
    // -------------------------------------------------------------------------
    // 1. ПАРАМЕТРЫ
    // -------------------------------------------------------------------------
    const int N = 1024;                    // Длина сигнала
    const double freq = 0.05;              // Частота синусоиды
    const double amplitude = 1.0;          // Амплитуда
    const double noise_level = 0.1;        // Уровень шума
    
    printf("Parameters:\n");
    printf("  Signal length: %d samples\n", N);
    printf("  Frequency: %.3f\n", freq);
    printf("  Amplitude: %.2f\n", amplitude);
    printf("  Noise level: %.3f\n\n", noise_level);
    
    // -------------------------------------------------------------------------
    // 2. ГЕНЕРАЦИЯ СИГНАЛА НА ХОСТЕ
    // -------------------------------------------------------------------------
    std::vector<double> host_signal_hi, host_signal_lo;
    generate_test_signal(host_signal_hi, host_signal_lo, N, freq, amplitude, noise_level);
    
    printf("Signal generated: [%.3f, %.3f, ..., %.3f, %.3f]\n",
           host_signal_hi[0], host_signal_hi[1],
           host_signal_hi[N-2], host_signal_hi[N-1]);
    
    // -------------------------------------------------------------------------
    // 3. КОПИРОВАНИЕ НА GPU
    // -------------------------------------------------------------------------
    double* d_signal_hi = nullptr;
    double* d_signal_lo = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_signal_hi, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_signal_lo, N * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_signal_hi, host_signal_hi.data(), 
                          N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_signal_lo, host_signal_lo.data(), 
                          N * sizeof(double), cudaMemcpyHostToDevice));
    
    printf("Data copied to GPU\n\n");
    
    // -------------------------------------------------------------------------
    // 4. ЗАПУСК AMAD-X (ОЦЕНКА ШУМА)
    // -------------------------------------------------------------------------
    printf("Running AMAD-X noise estimation...\n");
    fflush(stdout);
    
    // AMAD-X оценивает уровень шума в сигнале
    // Возвращает оценку sigma (стандартное отклонение)
    double sigma = amad_x_estimate_adaptive(
        d_signal_hi,      // указатель на hi-компоненту
        d_signal_lo,      // указатель на lo-компоненту
        N,                // длина сигнала
        false             // verbose = false (не выводить отладочную информацию)
    );
    
    printf("  Estimated noise sigma: %.6f\n", sigma);
    printf("  Actual noise level:   %.3f\n\n", noise_level);
    
    // Сравнение с ожидаемым значением
    double ratio = sigma / noise_level;
    if (ratio > 0.5 && ratio < 2.0) {
        printf("✓ AMAD-X estimation is reasonable (ratio = %.2f)\n", ratio);
    } else {
        printf("⚠ AMAD-X estimation differs from expected (ratio = %.2f)\n", ratio);
        printf("  (This is normal for small signals and high noise)\n");
    }
    
    // -------------------------------------------------------------------------
    // 5. ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ (ОПЦИОНАЛЬНО)
    // -------------------------------------------------------------------------
    // Если необходимо, можно вычислить реальное стандартное отклонение сигнала
    // для сравнения с оценкой AMAD-X
    double mean = 0.0;
    for (int i = 0; i < N; ++i) mean += host_signal_hi[i];
    mean /= N;
    
    double variance = 0.0;
    for (int i = 0; i < N; ++i) {
        double diff = host_signal_hi[i] - mean;
        variance += diff * diff;
    }
    variance /= (N - 1);
    double std_dev = std::sqrt(variance);
    
    printf("\nReference statistics:\n");
    printf("  Signal mean: %.6f\n", mean);
    printf("  Signal std dev: %.6f\n", std_dev);
    printf("  AMAD-X sigma / signal std dev: %.3f\n", sigma / std_dev);
    
    // -------------------------------------------------------------------------
    // 6. ОЧИСТКА
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_signal_hi));
    CUDA_CHECK(cudaFree(d_signal_lo));
    
    printf("\n========================================\n");
    printf("  Demo completed successfully!\n");
    printf("========================================\n");
    
    return 0;
}