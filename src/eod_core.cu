// =============================================================================
// eod_core.cu — Energy-Optimized Dictionary Learning (ИСПРАВЛЕН)
// =============================================================================

#include "eod_core.h"
#include "eop_core.h"
#include "alw_math.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <string>

#ifdef CUDA_CHECK
#undef CUDA_CHECK
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + \
                __FILE__ + ":" + std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
        } \
    } while(0)

#ifdef CUDA_CHECK_KERNEL
#undef CUDA_CHECK_KERNEL
#endif

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Kernel error at ") + \
                __FILE__ + ":" + std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
        } \
    } while(0)

__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

static void build_data_matrix(const double* prices, int N, int frame_size, int hop_size,
                              double* Y, int M) {
    for (int j = 0; j < M; ++j) {
        int start = j * hop_size;
        for (int i = 0; i < frame_size; ++i) {
            Y[i * M + j] = (start + i < N) ? prices[start + i] : 0.0;
        }
    }
}

__global__ void eod_compute_residuals_kernel(
    const double* __restrict__ Y,
    const double* __restrict__ D,
    const double* __restrict__ X,
    double* __restrict__ R,
    int frame_size,
    int M,
    int K)
{
    int j = blockIdx.x;
    int i = threadIdx.x;
    if (j >= M || i >= frame_size) return;
    double sum = 0.0;
    for (int k = 0; k < K; ++k) {
        sum += D[i * K + k] * X[k * M + j];
    }
    R[i * M + j] = Y[i * M + j] - sum;
}

__global__ void eod_compute_gradients_kernel(
    const double* __restrict__ R,
    const double* __restrict__ X,
    const double* __restrict__ D,
    double* __restrict__ Grad,
    int frame_size,
    int M,
    int K)
{
    int k = blockIdx.x;
    int i = threadIdx.x;
    if (k >= K || i >= frame_size) return;
    double grad = 0.0;
    double sum_x2 = 0.0;
    for (int j = 0; j < M; ++j) {
        double x = X[k * M + j];
        if (fabs(x) > 1e-12) {
            grad += x * R[i * M + j];
        }
        sum_x2 += x * x;
    }
    grad = 2.0 * (grad + sum_x2 * D[i * K + k]);
    Grad[i * K + k] = grad;
}

__global__ void eod_update_atoms_kernel(
    double* __restrict__ D,
    const double* __restrict__ Grad,
    double learning_rate,
    double lambda,
    int frame_size,
    int K)
{
    int k = blockIdx.x;
    int i = threadIdx.x;
    if (k >= K || i >= frame_size) return;
    D[i * K + k] += learning_rate * Grad[i * K + k] - lambda * D[i * K + k];
}

__global__ void eod_normalize_atoms_kernel(
    double* __restrict__ D,
    int frame_size,
    int K)
{
    int k = blockIdx.x;
    if (k >= K) return;
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    double sum = 0.0;
    int i = tid;
    while (i < frame_size) {
        double val = D[i * K + k];
        sum += val * val;
        i += blockDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    double norm = sqrt(sdata[0]);
    if (norm > 1e-12) {
        i = tid;
        while (i < frame_size) {
            D[i * K + k] /= norm;
            i += blockDim.x;
        }
    }
}

__global__ void eod_transpose_D_kernel(
    const double* __restrict__ D,
    double* __restrict__ D_T,
    int frame_size,
    int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = frame_size * K;
    if (idx >= total) return;
    int i = idx / K;
    int k = idx % K;
    D_T[k * frame_size + i] = D[i * K + k];
}

__global__ void eod_compute_error_kernel(
    const double* __restrict__ R,
    double* __restrict__ error,
    int total_elements)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    double sum = 0.0;
    while (idx < total_elements) {
        double val = R[idx];
        sum += val * val;
        idx += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd_double(error, sdata[0]);
}

void eod_learn_dictionary_gpu(
    const double* prices,
    int N,
    int frame_size,
    int num_atoms,
    const double* init_dict,
    const EOP_Params& eop_params,
    const EOD_Params& eod_params,
    double* out_dict,
    alw_vector<double>& atom_norms)
{
    if (frame_size < 2 || num_atoms < 1 || N < frame_size) {
        fprintf(stderr, "eod_learn_dictionary_gpu: invalid parameters\n");
        return;
    }

    int hop = (eod_params.hop_size > 0) ? eod_params.hop_size : frame_size / 2;
    int M = (N - frame_size) / hop + 1;
    if (M < 1) {
        fprintf(stderr, "eod_learn_dictionary_gpu: not enough frames (M=%d)\n", M);
        return;
    }

    if (eod_params.verbose) {
        printf("EOD: N=%d, frame_size=%d, K=%d, M=%d, hop=%d\n",
               N, frame_size, num_atoms, M, hop);
    }

    alw_vector<double> Y_host(frame_size * M, 0.0);
    build_data_matrix(prices, N, frame_size, hop, Y_host.data(), M);

    double *d_Y = nullptr, *d_D = nullptr, *d_D_T = nullptr, *d_X = nullptr, *d_R = nullptr;
    double *d_Grad = nullptr, *d_error = nullptr;
    double *d_D_lo = nullptr, *d_frame_hi = nullptr, *d_frame_lo = nullptr, *d_D_best = nullptr;
    double *d_res_hi = nullptr, *d_res_lo = nullptr;
    double *d_proj_hi = nullptr, *d_proj_lo = nullptr;
    double *d_coh_hi = nullptr, *d_coh_lo = nullptr;
    double *d_energy_hi = nullptr, *d_energy_lo = nullptr;
    double *d_norm_sq = nullptr;
    double *d_block_best_energy = nullptr;
    int *d_block_best_idx = nullptr;
    int *d_selected_indices = nullptr;

    try {
        CUDA_CHECK(cudaMalloc(&d_Y, frame_size * M * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_D, frame_size * num_atoms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_D_T, num_atoms * frame_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_X, num_atoms * M * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_R, frame_size * M * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Grad, frame_size * num_atoms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_error, sizeof(double)));

        CUDA_CHECK(cudaMalloc(&d_frame_hi, frame_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_frame_lo, frame_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_D_lo, num_atoms * frame_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_D_best, frame_size * num_atoms * sizeof(double)));

        CUDA_CHECK(cudaMemset(d_D_lo, 0, num_atoms * frame_size * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_frame_lo, 0, frame_size * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_Y, Y_host.data(), frame_size * M * sizeof(double), cudaMemcpyHostToDevice));

        if (init_dict) {
            CUDA_CHECK(cudaMemcpy(d_D, init_dict, frame_size * num_atoms * sizeof(double), cudaMemcpyHostToDevice));
        } else {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> dist(0.0, 1.0);
            alw_vector<double> h_D(frame_size * num_atoms);
            for (int i = 0; i < frame_size * num_atoms; ++i) h_D[i] = dist(gen);
            CUDA_CHECK(cudaMemcpy(d_D, h_D.data(), frame_size * num_atoms * sizeof(double), cudaMemcpyHostToDevice));
            eod_normalize_atoms_kernel<<<num_atoms, 256>>>(d_D, frame_size, num_atoms);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        int threads_eop = 128;
        int num_blocks_eop = (num_atoms + threads_eop - 1) / threads_eop;

        CUDA_CHECK(cudaMalloc(&d_res_hi, frame_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_res_lo, frame_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_proj_hi, num_atoms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_proj_lo, num_atoms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_coh_hi, num_atoms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_coh_lo, num_atoms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_energy_hi, num_atoms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_energy_lo, num_atoms * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_norm_sq, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_block_best_energy, num_blocks_eop * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_block_best_idx, num_blocks_eop * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_selected_indices, num_atoms * sizeof(int)));

        double prev_error = 1e30;
        double best_error = 1e30;
        int best_iter = 0;
        bool best_initialized = false;

        alw_vector<double> X_host(num_atoms * M, 0.0);
        EOP_Params eop_local = eop_params;
        eop_local.max_iterations = eod_params.max_atoms_per_frame;

        alw_vector<DetectedEvent> events;
        alw_vector<double> atom_norms_dummy(num_atoms, 1.0);
        alw_vector<AtomDescriptor> descriptors_dummy;

        double lambda = eod_params.regularization_lambda;

        for (int iter = 0; iter < eod_params.max_iter; ++iter) {
            double lr = eod_params.learning_rate / (1.0 + 0.01 * iter);

            int total_elem = frame_size * num_atoms;
            int blocks_trans = (total_elem + threads_eop - 1) / threads_eop;
            eod_transpose_D_kernel<<<blocks_trans, threads_eop>>>(d_D, d_D_T, frame_size, num_atoms);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());

            std::fill(X_host.begin(), X_host.end(), 0.0);

            for (int j = 0; j < M; ++j) {
                CUDA_CHECK(cudaMemcpy2D(d_frame_hi, sizeof(double),
                                        d_Y + j, M * sizeof(double),
                                        sizeof(double), frame_size,
                                        cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemset(d_frame_lo, 0, frame_size * sizeof(double)));

                events.clear();
                eop_pursuit(d_frame_hi, d_frame_lo,
                            d_D_T, d_D_lo,
                            num_atoms, frame_size, 0,
                            eod_params.sigma_noise,
                            eop_local,
                            events,
                            atom_norms_dummy,
                            descriptors_dummy,
                            false,
                            d_res_hi, d_res_lo,
                            d_proj_hi, d_proj_lo,
                            d_coh_hi, d_coh_lo,
                            d_energy_hi, d_energy_lo,
                            d_norm_sq,
                            d_block_best_energy,
                            d_block_best_idx,
                            d_selected_indices);

                for (size_t e = 0; e < events.size(); ++e) {
                    int idx = events[e].atom_index;
                    double amp = events[e].amplitude_hi;
                    X_host[idx * M + j] = amp;
                }
            }

            CUDA_CHECK(cudaMemcpy(d_X, X_host.data(), num_atoms * M * sizeof(double), cudaMemcpyHostToDevice));

            eod_compute_residuals_kernel<<<M, frame_size>>>(d_Y, d_D, d_X, d_R, frame_size, M, num_atoms);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());

            eod_compute_gradients_kernel<<<num_atoms, frame_size>>>(d_R, d_X, d_D, d_Grad, frame_size, M, num_atoms);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());

            eod_update_atoms_kernel<<<num_atoms, frame_size>>>(d_D, d_Grad, lr, lambda, frame_size, num_atoms);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());

            eod_normalize_atoms_kernel<<<num_atoms, 256>>>(d_D, frame_size, num_atoms);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemset(d_error, 0, sizeof(double)));
            int total_R = frame_size * M;
            int blocks_error = (total_R + threads_eop - 1) / threads_eop;
            size_t shmem = threads_eop * sizeof(double);
            eod_compute_error_kernel<<<blocks_error, threads_eop, shmem>>>(d_R, d_error, total_R);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());

            double error;
            CUDA_CHECK(cudaMemcpy(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost));
            error = sqrt(error / total_R);

            if (eod_params.verbose) {
                printf("iter %3d: RMSE = %.6f (lr=%.5f)\n", iter, error, lr);
            }

            if (!best_initialized || error < best_error) {
                best_error = error;
                best_iter = iter;
                CUDA_CHECK(cudaMemcpy(d_D_best, d_D, frame_size * num_atoms * sizeof(double), cudaMemcpyDeviceToDevice));
                best_initialized = true;
                if (eod_params.verbose) {
                    printf("  -> New best RMSE = %.6f at iter %d\n", best_error, best_iter);
                }
            }

            if (eod_params.tolerance > 0 && iter > 0) {
                double rel_change = fabs(prev_error - error) / (prev_error + 1e-12);
                if (rel_change < eod_params.tolerance) {
                    if (eod_params.verbose) {
                        printf("EOD converged at iter %d (rel_change=%.2e)\n", iter, rel_change);
                    }
                    break;
                }
            }
            prev_error = error;
        }

        CUDA_CHECK(cudaMemcpy(out_dict, d_D_best, frame_size * num_atoms * sizeof(double), cudaMemcpyDeviceToHost));

        atom_norms.resize(num_atoms);
        for (int k = 0; k < num_atoms; ++k) {
            double norm = 0.0;
            for (int i = 0; i < frame_size; ++i) {
                double val = out_dict[i * num_atoms + k];
                norm += val * val;
            }
            atom_norms[k] = sqrt(norm);
        }

        cudaFree(d_Y); cudaFree(d_D); cudaFree(d_D_T); cudaFree(d_X);
        cudaFree(d_R); cudaFree(d_Grad); cudaFree(d_error);
        cudaFree(d_frame_hi); cudaFree(d_frame_lo); cudaFree(d_D_lo); cudaFree(d_D_best);
        cudaFree(d_res_hi); cudaFree(d_res_lo);
        cudaFree(d_proj_hi); cudaFree(d_proj_lo);
        cudaFree(d_coh_hi); cudaFree(d_coh_lo);
        cudaFree(d_energy_hi); cudaFree(d_energy_lo);
        cudaFree(d_norm_sq);
        cudaFree(d_block_best_energy); cudaFree(d_block_best_idx);
        cudaFree(d_selected_indices);

        if (eod_params.verbose) {
            printf("EOD finished. Best RMSE = %.6f at iter %d\n", best_error, best_iter);
        }
    }
    catch (const std::exception& e) {
        cudaFree(d_Y); cudaFree(d_D); cudaFree(d_D_T); cudaFree(d_X);
        cudaFree(d_R); cudaFree(d_Grad); cudaFree(d_error);
        cudaFree(d_frame_hi); cudaFree(d_frame_lo); cudaFree(d_D_lo); cudaFree(d_D_best);
        cudaFree(d_res_hi); cudaFree(d_res_lo);
        cudaFree(d_proj_hi); cudaFree(d_proj_lo);
        cudaFree(d_coh_hi); cudaFree(d_coh_lo);
        cudaFree(d_energy_hi); cudaFree(d_energy_lo);
        cudaFree(d_norm_sq);
        cudaFree(d_block_best_energy); cudaFree(d_block_best_idx);
        cudaFree(d_selected_indices);
        throw;
    }
}
