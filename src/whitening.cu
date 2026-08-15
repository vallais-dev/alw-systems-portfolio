// whitening.cu
#include "whitening.h"
#include "alw_math.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// ----------------------------------------------------------------------------
// ВСПОМОГАТЕЛЬНЫЕ ЯДРА
// ----------------------------------------------------------------------------

// Умножение на окно Хэмминга (in-place)
__global__ void apply_hamming_window_kernel(double* d_data, int segment_len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= segment_len) return;
    double w = 0.54 - 0.46 * cos(2.0 * M_PI * idx / (segment_len - 1));
    d_data[idx] *= w;
}

// Копирование сегмента с перекрытием и применение окна
__global__ void copy_and_window_segment_kernel(
    const double* d_signal,
    double* d_segment,
    int segment_len,
    int offset,
    int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= segment_len) return;
    if (offset + idx >= N) {
        d_segment[idx] = 0.0;
        return;
    }
    double val = d_signal[offset + idx];
    double w = 0.54 - 0.46 * cos(2.0 * M_PI * idx / (segment_len - 1));
    d_segment[idx] = val * w;
}

// Суммирование квадратов модулей для усреднения PSD
__global__ void accumulate_psd_kernel(
    const cufftDoubleComplex* d_fft,
    double* d_psd_acc,
    int fft_size,
    int num_accumulated) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= fft_size / 2 + 1) return;
    double re = d_fft[idx].x;
    double im = d_fft[idx].y;
    double mag = re * re + im * im;
    atomicAdd(&d_psd_acc[idx], mag);
}

// Нормализация PSD (усреднение и формирование одностороннего спектра)
__global__ void normalize_psd_kernel(
    double* d_psd_acc,
    double* d_psd,
    int fft_size,
    int num_segments) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= fft_size / 2 + 1) return;
    double val = d_psd_acc[idx] / (num_segments * fft_size);
    if (idx > 0 && idx < fft_size / 2) val *= 2.0;
    d_psd[idx] = val;
}

// Построение H(f) = 1 / sqrt(PSD + epsilon)
__global__ void compute_whitening_response_kernel(
    const double* d_psd,
    double* d_H,
    int fft_size,
    double epsilon) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= fft_size / 2 + 1) return;
    double psd = d_psd[idx] + epsilon;
    d_H[idx] = 1.0 / sqrt(psd);
}

// Заполнение зеркальной части для обратного БПФ (симметричный спектр)
__global__ void mirror_spectrum_kernel(
    const double* d_H_half,
    cufftDoubleComplex* d_H_full,
    int fft_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= fft_size) return;
    int half = fft_size / 2;
    if (idx <= half) {
        d_H_full[idx].x = d_H_half[idx];
        d_H_full[idx].y = 0.0;
    } else {
        int mirror = fft_size - idx;
        if (mirror <= half) {
            d_H_full[idx].x = d_H_half[mirror];
            d_H_full[idx].y = 0.0;
        } else {
            d_H_full[idx].x = 0.0;
            d_H_full[idx].y = 0.0;
        }
    }
}

// Обрезка импульсной характеристики и применение окна
__global__ void truncate_and_window_kernel(
    double* d_ir,
    int fft_size,
    int filter_len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= filter_len) return;
    double val = d_ir[idx];
    // Окно Хэмминга для сглаживания
    double w = 0.54 - 0.46 * cos(2.0 * M_PI * idx / (filter_len - 1));
    d_ir[idx] = val * w;
}

// Нормализация фильтра (сумма коэффициентов = 1)
__global__ void normalize_filter_kernel(
    double* d_filter,
    int filter_len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= filter_len) return;
    // Здесь используется редукция на блоке, для простоты оставляем пустым
    // Нормализация будет выполнена на CPU
}

// Прямая свертка: каждый поток вычисляет одну точку выхода
__global__ void convolution_kernel(
    const double* d_signal,
    int N,
    const double* d_filter,
    int filter_len,
    double* d_output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    double sum = 0.0;
    for (int k = 0; k < filter_len; ++k) {
        int src = idx - k;
        double sample;
        if (src < 0) {
            src = -src - 1;
            if (src >= N) src = N - 1;
        }
        if (src >= N) src = N - 1;
        sample = d_signal[src];
        sum += sample * d_filter[k];
    }
    d_output[idx] = sum;
}

// ----------------------------------------------------------------------------
// РЕАЛИЗАЦИЯ ФУНКЦИЙ
// ----------------------------------------------------------------------------

bool estimate_psd(const double* d_signal, int N, int segment_len,
                  int overlap, int fft_size, double* d_psd,
                  cudaStream_t stream) {
    // Проверка параметров
    if (segment_len > fft_size) return false;
    if (N < segment_len) return false;
    if (overlap >= segment_len) return false;

    int step = segment_len - overlap;
    if (step <= 0) return false;

    int num_segments = (N - segment_len) / step + 1;
    if (num_segments < 1) return false;

    int threads = 256;
    int blocks_seg = (segment_len + threads - 1) / threads;
    int blocks_fft = (fft_size + threads - 1) / threads;
    int blocks_psd = (fft_size / 2 + 1 + threads - 1) / threads;

    // Выделяем память
    double* d_segment = nullptr;
    cufftDoubleComplex* d_fft = nullptr;
    double* d_psd_acc = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_segment, segment_len * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "estimate_psd: cudaMalloc d_segment failed\n");
        return false;
    }

    err = cudaMalloc(&d_fft, fft_size * sizeof(cufftDoubleComplex));
    if (err != cudaSuccess) {
        fprintf(stderr, "estimate_psd: cudaMalloc d_fft failed\n");
        cudaFree(d_segment);
        return false;
    }

    err = cudaMalloc(&d_psd_acc, (fft_size / 2 + 1) * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "estimate_psd: cudaMalloc d_psd_acc failed\n");
        cudaFree(d_segment);
        cudaFree(d_fft);
        return false;
    }

    cudaMemsetAsync(d_psd_acc, 0, (fft_size / 2 + 1) * sizeof(double), stream);

    // Создаём план БПФ
    cufftHandle plan;
    if (cufftPlan1d(&plan, fft_size, CUFFT_D2Z, 1) != CUFFT_SUCCESS) {
        fprintf(stderr, "estimate_psd: cufftPlan1d failed\n");
        cudaFree(d_segment);
        cudaFree(d_fft);
        cudaFree(d_psd_acc);
        return false;
    }

    // Обработка сегментов
    for (int s = 0; s < num_segments; ++s) {
        int offset = s * step;
        if (offset + segment_len > N) break;

        // Копируем сегмент с окном
        copy_and_window_segment_kernel<<<blocks_seg, threads, 0, stream>>>(
            d_signal, d_segment, segment_len, offset, N);

        // Заполняем комплексный буфер (действительная часть = d_segment)
        cudaMemsetAsync(d_fft, 0, fft_size * sizeof(cufftDoubleComplex), stream);

        // Копируем действительную часть в комплексный буфер
        // Используем reinterpret_cast для совместимости с cufftExecD2Z
        err = cudaMemcpyAsync(
            reinterpret_cast<double*>(d_fft),
            d_segment,
            segment_len * sizeof(double),
            cudaMemcpyDeviceToDevice,
            stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "estimate_psd: cudaMemcpyAsync failed\n");
            break;
        }

        // БПФ (D2Z: double -> cufftDoubleComplex)
        if (cufftExecD2Z(plan,
                         reinterpret_cast<cufftDoubleReal*>(d_fft),
                         d_fft) != CUFFT_SUCCESS) {
            fprintf(stderr, "estimate_psd: cufftExecD2Z failed\n");
            break;
        }

        // Накопление квадратов модулей
        accumulate_psd_kernel<<<blocks_psd, threads, 0, stream>>>(
            d_fft, d_psd_acc, fft_size, s);
    }

    // Нормализация PSD
    normalize_psd_kernel<<<blocks_psd, threads, 0, stream>>>(
        d_psd_acc, d_psd, fft_size, num_segments);

    cudaStreamSynchronize(stream);

    // Освобождение ресурсов
    cufftDestroy(plan);
    cudaFree(d_segment);
    cudaFree(d_fft);
    cudaFree(d_psd_acc);

    return true;
}

void design_whitening_filter(const double* d_psd, int fft_size, int filter_len,
                             double epsilon, double* d_filter,
                             cudaStream_t stream) {
    int threads = 256;
    int blocks_half = (fft_size / 2 + 1 + threads - 1) / threads;
    int blocks_full = (fft_size + threads - 1) / threads;
    int blocks_filter = (filter_len + threads - 1) / threads;

    // H(f) = 1 / sqrt(PSD + epsilon)
    double* d_H = nullptr;
    cudaMalloc(&d_H, (fft_size / 2 + 1) * sizeof(double));
    compute_whitening_response_kernel<<<blocks_half, threads, 0, stream>>>(
        d_psd, d_H, fft_size, epsilon);

    // Расширение H до полного спектра (зеркально)
    cufftDoubleComplex* d_H_full = nullptr;
    cudaMalloc(&d_H_full, fft_size * sizeof(cufftDoubleComplex));
    mirror_spectrum_kernel<<<blocks_full, threads, 0, stream>>>(
        d_H, d_H_full, fft_size);

    // Обратное БПФ (Z2D: cufftDoubleComplex -> double)
    cufftHandle plan;
    cufftPlan1d(&plan, fft_size, CUFFT_Z2D, 1);

    double* d_ir = nullptr;
    cudaMalloc(&d_ir, fft_size * sizeof(double));

    // Обратное БПФ (из комплексного в действительное)
    if (cufftExecZ2D(plan, d_H_full,
                     reinterpret_cast<cufftDoubleReal*>(d_ir)) != CUFFT_SUCCESS) {
        fprintf(stderr, "design_whitening_filter: cufftExecZ2D failed\n");
        goto cleanup;
    }

    // Обрезаем до filter_len и применяем окно
    truncate_and_window_kernel<<<blocks_filter, threads, 0, stream>>>(
        d_ir, fft_size, filter_len);

    // Копируем результат в d_filter
    cudaMemcpyAsync(d_filter, d_ir, filter_len * sizeof(double),
                    cudaMemcpyDeviceToDevice, stream);

    // Нормализация фильтра (сумма коэффициентов должна быть 1)
    // Вычисляем сумму на CPU, затем нормализуем
    {
        std::vector<double> h(filter_len);
        cudaMemcpyAsync(h.data(), d_filter, filter_len * sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        double sum = 0.0;
        for (int i = 0; i < filter_len; ++i) sum += h[i];
        if (fabs(sum) > 1e-30) {
            for (int i = 0; i < filter_len; ++i) h[i] /= sum;
            cudaMemcpyAsync(d_filter, h.data(), filter_len * sizeof(double),
                            cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        }
    }

cleanup:
    cudaFree(d_H);
    cudaFree(d_H_full);
    cudaFree(d_ir);
    cufftDestroy(plan);
}

void apply_whitening_filter(const double* d_signal, int N,
                            const double* d_filter, int filter_len,
                            double* d_output, cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    convolution_kernel<<<blocks, threads, 0, stream>>>(
        d_signal, N, d_filter, filter_len, d_output);
}

void whiten_signal(double* d_signal, int N,
                   int segment_len,
                   int overlap,
                   int fft_size,
                   int filter_len,
                   double epsilon,
                   cudaStream_t stream) {
    // Выделяем память для PSD
    double* d_psd = nullptr;
    cudaMalloc(&d_psd, (fft_size / 2 + 1) * sizeof(double));

    // Оценка PSD
    if (!estimate_psd(d_signal, N, segment_len, overlap, fft_size, d_psd, stream)) {
        fprintf(stderr, "whiten_signal: estimate_psd failed\n");
        cudaFree(d_psd);
        return;
    }

    // Синтез фильтра
    double* d_filter = nullptr;
    cudaMalloc(&d_filter, filter_len * sizeof(double));
    design_whitening_filter(d_psd, fft_size, filter_len, epsilon, d_filter, stream);

    // Применение фильтра (результат записываем во временный буфер)
    double* d_temp = nullptr;
    cudaMalloc(&d_temp, N * sizeof(double));
    apply_whitening_filter(d_signal, N, d_filter, filter_len, d_temp, stream);

    // Копируем результат обратно в d_signal
    cudaMemcpyAsync(d_signal, d_temp, N * sizeof(double),
                    cudaMemcpyDeviceToDevice, stream);

    // Освобождение
    cudaFree(d_psd);
    cudaFree(d_filter);
    cudaFree(d_temp);
}
