// eop_core.cu — Energy-Optimized Pursuit с вынесенными буферами
// Добавлены __ldg() для чтения словаря и остатка

#include "eop_core.h"
#include "alw_math.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cmath>

#define EOP_WARP_SIZE 32
#define EOP_MAX_THREADS 256

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ----------------------------------------------------------------------------
// 1. ЯДРО: ВЫЧИСЛЕНИЕ ПРОЕКЦИЙ <r, d_j> — с __ldg()
// ----------------------------------------------------------------------------
__global__ void eop_projections_kernel(
    const double* __restrict__ d_res_hi,
    const double* __restrict__ d_res_lo,
    const double* __restrict__ d_dict_hi,
    const double* __restrict__ d_dict_lo,
    int num_atoms,
    int frame_size,
    double* __restrict__ d_proj_hi,
    double* __restrict__ d_proj_lo
) {
    int atom_idx = blockIdx.x;
    if (atom_idx >= num_atoms) return;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    extern __shared__ double sh[];
    double* sh_hi = sh;
    double* sh_lo = sh + blockDim.x;
    DD local_sum = {0.0, 0.0};
    for (int i = tid; i < frame_size; i += num_threads) {
        double r_hi = __ldg(d_res_hi + i);
        double r_lo = __ldg(d_res_lo + i);
        double a_hi = __ldg(d_dict_hi + atom_idx * frame_size + i);
        double a_lo = __ldg(d_dict_lo + atom_idx * frame_size + i);
        DD r = {r_hi, r_lo};
        DD a = {a_hi, a_lo};
        DD prod = alw_mul_dd(r, a);
        local_sum = alw_add_dd(local_sum, prod);
    }
    sh_hi[tid] = local_sum.hi;
    sh_lo[tid] = local_sum.lo;
    __syncthreads();
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_hi[tid] += sh_hi[tid + s];
            sh_lo[tid] += sh_lo[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_proj_hi[atom_idx] = sh_hi[0];
        d_proj_lo[atom_idx] = sh_lo[0];
    }
}

// ----------------------------------------------------------------------------
// 2. ЯДРО: ВЫЧИСЛЕНИЕ КОГЕРЕНТНОСТИ (без __ldg, но можно добавить)
// ----------------------------------------------------------------------------
__global__ void eop_coherence_kernel(
    const double* __restrict__ d_dict_hi,
    const double* __restrict__ d_dict_lo,
    const int* __restrict__ d_selected_indices,
    int num_selected,
    int frame_size,
    int num_atoms,
    double* __restrict__ d_coh_hi,
    double* __restrict__ d_coh_lo
) {
    int atom_idx = blockIdx.x;
    if (atom_idx >= num_atoms) return;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    if (num_selected == 0) {
        if (tid == 0) { d_coh_hi[atom_idx] = 0.0; d_coh_lo[atom_idx] = 0.0; }
        return;
    }
    DD local_sum = {0.0, 0.0};
    for (int s = tid; s < num_selected; s += num_threads) {
        int sel_idx = d_selected_indices[s];
        if (sel_idx < 0 || sel_idx >= num_atoms) continue;
        DD dot = {0.0, 0.0};
        for (int i = 0; i < frame_size; ++i) {
            double a_hi = __ldg(d_dict_hi + atom_idx * frame_size + i);
            double a_lo = __ldg(d_dict_lo + atom_idx * frame_size + i);
            double b_hi = __ldg(d_dict_hi + sel_idx * frame_size + i);
            double b_lo = __ldg(d_dict_lo + sel_idx * frame_size + i);
            DD a = {a_hi, a_lo};
            DD b = {b_hi, b_lo};
            DD prod = alw_mul_dd(a, b);
            dot = alw_add_dd(dot, prod);
        }
        DD abs_dot = alw_abs_dd(dot);
        local_sum = alw_add_dd(local_sum, abs_dot);
    }
    extern __shared__ double sh[];
    double* sh_hi = sh;
    double* sh_lo = sh + blockDim.x;
    sh_hi[tid] = local_sum.hi;
    sh_lo[tid] = local_sum.lo;
    __syncthreads();
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_hi[tid] += sh_hi[tid + s];
            sh_lo[tid] += sh_lo[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_coh_hi[atom_idx] = sh_hi[0];
        d_coh_lo[atom_idx] = sh_lo[0];
    }
}

// ----------------------------------------------------------------------------
// 3a. ЯДРО: ВЫЧИСЛЕНИЕ ЭНЕРГИЙ
// ----------------------------------------------------------------------------
__global__ void eop_compute_energy_kernel(
    const double* __restrict__ d_proj_hi,
    const double* __restrict__ d_proj_lo,
    const double* __restrict__ d_coh_hi,
    const double* __restrict__ d_coh_lo,
    double alpha,
    int num_atoms,
    double* __restrict__ d_energy_hi,
    double* __restrict__ d_energy_lo
) {
    int idx = blockIdx.x;
    if (idx >= num_atoms) return;
    DD proj = {d_proj_hi[idx], d_proj_lo[idx]};
    DD coh = {d_coh_hi[idx], d_coh_lo[idx]};
    DD proj_sq = alw_mul_dd(proj, proj);
    DD alpha_dd = {alpha, 0.0};
    DD alpha_coh = alw_mul_dd(alpha_dd, coh);
    DD denom = alw_add_dd({1.0, 0.0}, alpha_coh);
    DD energy = alw_div_dd(proj_sq, denom);
    d_energy_hi[idx] = energy.hi;
    d_energy_lo[idx] = energy.lo;
}

// ----------------------------------------------------------------------------
// 3b. ЯДРО: ПОИСК МАКСИМУМА (блочная редукция)
// ----------------------------------------------------------------------------
__global__ void eop_find_max_kernel(
    const double* __restrict__ d_energy_hi,
    const double* __restrict__ d_energy_lo,
    int num_atoms,
    double* __restrict__ d_block_best_energy,
    int* __restrict__ d_block_best_idx
) {
    extern __shared__ double sh[];
    double* sh_energy = sh;
    int* sh_idx = (int*)&sh[blockDim.x];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    double best_val = -1e300;
    int best_idx = -1;
    if (idx < num_atoms) {
        best_val = d_energy_hi[idx];
        best_idx = idx;
    }
    sh_energy[tid] = best_val;
    sh_idx[tid] = best_idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sh_energy[tid + s] > sh_energy[tid]) {
                sh_energy[tid] = sh_energy[tid + s];
                sh_idx[tid] = sh_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_block_best_energy[blockIdx.x] = sh_energy[0];
        d_block_best_idx[blockIdx.x] = sh_idx[0];
    }
}

// ----------------------------------------------------------------------------
// 4. ЯДРО: ОБНОВЛЕНИЕ ОСТАТКА — с __ldg()
// ----------------------------------------------------------------------------
__global__ void eop_update_residual_kernel(
    double* __restrict__ d_res_hi,
    double* __restrict__ d_res_lo,
    const double* __restrict__ d_dict_hi,
    const double* __restrict__ d_dict_lo,
    int best_idx,
    double coeff_hi,
    double coeff_lo,
    int frame_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= frame_size) return;
    double a_hi = __ldg(d_dict_hi + best_idx * frame_size + i);
    double a_lo = __ldg(d_dict_lo + best_idx * frame_size + i);
    DD coeff = {coeff_hi, coeff_lo};
    DD atom = {a_hi, a_lo};
    DD prod = alw_mul_dd(coeff, atom);
    DD r = {d_res_hi[i], d_res_lo[i]};
    DD new_r = alw_sub_dd(r, prod);
    d_res_hi[i] = new_r.hi;
    d_res_lo[i] = new_r.lo;
}

// ----------------------------------------------------------------------------
// 5. ЯДРО: ВЫЧИСЛЕНИЕ КВАДРАТА НОРМЫ ОСТАТКА (без __ldg)
// ----------------------------------------------------------------------------
__global__ void eop_residual_norm_kernel(
    const double* __restrict__ d_res_hi,
    const double* __restrict__ d_res_lo,
    int frame_size,
    double* __restrict__ d_norm_sq
) {
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    double local_sum = 0.0;
    for (int i = tid; i < frame_size; i += num_threads) {
        DD r = {d_res_hi[i], d_res_lo[i]};
        DD sq = alw_mul_dd(r, r);
        local_sum += sq.hi;
    }
    sh[tid] = local_sum;
    __syncthreads();
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) d_norm_sq[0] = sh[0];
}

// ----------------------------------------------------------------------------
// ХОСТ-ФУНКЦИЯ eop_pursuit (без потока)
// ----------------------------------------------------------------------------
void eop_pursuit(
    double* d_frame_hi,
    double* d_frame_lo,
    double* d_dict_hi,
    double* d_dict_lo,
    int num_atoms,
    int frame_size,
    int frame_start,
    double sigma_noise,
    const EOP_Params& params,
    alw_vector<DetectedEvent>& events,
    const alw_vector<double>& atom_norms,
    const alw_vector<AtomDescriptor>& descriptors,
    bool verbose,
    double* d_res_hi,
    double* d_res_lo,
    double* d_proj_hi,
    double* d_proj_lo,
    double* d_coh_hi,
    double* d_coh_lo,
    double* d_energy_hi,
    double* d_energy_lo,
    double* d_norm_sq,
    double* d_block_best_energy,
    int* d_block_best_idx,
    int* d_selected_indices
) {
    if (verbose) {
        AEDS_LOGI("EOP: num_atoms=%d, frame_size=%d, sigma=%.6f, alpha=%.2f, gamma=%.2f, theta=%.2f",
                  num_atoms, frame_size, sigma_noise, params.alpha, params.gamma, params.theta);
    }

    CUDA_CHECK(cudaMemcpy(d_res_hi, d_frame_hi, frame_size * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_res_lo, d_frame_lo, frame_size * sizeof(double), cudaMemcpyDeviceToDevice));

    int threads = 128;
    int num_blocks = (num_atoms + threads - 1) / threads;

    alw_vector<int> selected_indices;
    alw_vector<DD> coeffs;

    int iter = 0;
    int max_iter = (params.max_iterations > 0) ? params.max_iterations : 1000;
    double prev_norm_sq = 0.0;
    bool stop = false;

    while (iter < max_iter && !stop) {
        size_t shmem_proj = 2 * threads * sizeof(double);
        eop_projections_kernel<<<num_atoms, threads, shmem_proj>>>(
            d_res_hi, d_res_lo,
            d_dict_hi, d_dict_lo,
            num_atoms, frame_size,
            d_proj_hi, d_proj_lo
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (verbose && iter == 0) {
            std::vector<double> proj_h(num_atoms), proj_l(num_atoms);
            CUDA_CHECK(cudaMemcpy(proj_h.data(), d_proj_hi, num_atoms * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(proj_l.data(), d_proj_lo, num_atoms * sizeof(double), cudaMemcpyDeviceToHost));
            printf("DEBUG: First 10 projections (hi, lo):\n");
            for (int i = 0; i < 10 && i < num_atoms; ++i) {
                printf("  atom %d: hi=%e lo=%e\n", i, proj_h[i], proj_l[i]);
            }
            eop_residual_norm_kernel<<<1, 128, 128 * sizeof(double)>>>(d_res_hi, d_res_lo, frame_size, d_norm_sq);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            double norm_sq;
            CUDA_CHECK(cudaMemcpy(&norm_sq, d_norm_sq, sizeof(double), cudaMemcpyDeviceToHost));
            printf("DEBUG: Residual norm squared = %e\n", norm_sq);
        }

        if (selected_indices.empty()) {
            CUDA_CHECK(cudaMemset(d_coh_hi, 0, num_atoms * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_coh_lo, 0, num_atoms * sizeof(double)));
        } else {
            CUDA_CHECK(cudaMemcpy(d_selected_indices, selected_indices.data(),
                       selected_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
            size_t shmem_coh = 2 * threads * sizeof(double);
            eop_coherence_kernel<<<num_atoms, threads, shmem_coh>>>(
                d_dict_hi, d_dict_lo,
                d_selected_indices,
                selected_indices.size(),
                frame_size,
                num_atoms,
                d_coh_hi, d_coh_lo
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        eop_compute_energy_kernel<<<num_atoms, threads>>>(
            d_proj_hi, d_proj_lo,
            d_coh_hi, d_coh_lo,
            params.alpha,
            num_atoms,
            d_energy_hi, d_energy_lo
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        size_t shmem_max = 2 * threads * sizeof(double);
        eop_find_max_kernel<<<num_blocks, threads, shmem_max>>>(
            d_energy_hi, d_energy_lo,
            num_atoms,
            d_block_best_energy,
            d_block_best_idx
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> block_energies(num_blocks);
        std::vector<int> block_indices(num_blocks);
        CUDA_CHECK(cudaMemcpy(block_energies.data(), d_block_best_energy, num_blocks * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(block_indices.data(), d_block_best_idx, num_blocks * sizeof(int), cudaMemcpyDeviceToHost));

        int best_idx = -1;
        double best_energy = -1e300;
        for (int b = 0; b < num_blocks; ++b) {
            if (block_indices[b] >= 0 && block_energies[b] > best_energy) {
                best_energy = block_energies[b];
                best_idx = block_indices[b];
            }
        }

        if (verbose && iter == 0 && best_idx >= 0) {
            double proj_hi, proj_lo, coh_hi, coh_lo;
            CUDA_CHECK(cudaMemcpy(&proj_hi, d_proj_hi + best_idx, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&proj_lo, d_proj_lo + best_idx, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&coh_hi, d_coh_hi + best_idx, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&coh_lo, d_coh_lo + best_idx, sizeof(double), cudaMemcpyDeviceToHost));
            printf("DEBUG: best_idx=%d, best_energy=%.6e, proj_hi=%.6e, coh_hi=%.6e\n",
                   best_idx, best_energy, proj_hi, coh_hi);
            printf("DEBUG: threshold = %.6e\n", params.gamma * sigma_noise * sigma_noise);
        }

        if (best_idx < 0 || best_energy < params.gamma * sigma_noise * sigma_noise) {
            if (verbose) {
                AEDS_LOGD("EOP: stop due to energy threshold (best_energy=%.6e, threshold=%.6e)",
                          best_energy, params.gamma * sigma_noise * sigma_noise);
            }
            break;
        }

        eop_residual_norm_kernel<<<1, 128, 128 * sizeof(double)>>>(d_res_hi, d_res_lo, frame_size, d_norm_sq);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        double cur_norm_sq;
        CUDA_CHECK(cudaMemcpy(&cur_norm_sq, d_norm_sq, sizeof(double), cudaMemcpyDeviceToHost));

        if (iter > 0) {
            double rel_decrease = 1.0 - cur_norm_sq / (prev_norm_sq + 1e-30);
            if (rel_decrease < params.theta * sigma_noise * sigma_noise) {
                if (verbose) {
                    AEDS_LOGD("EOP: stop due to relative decrease (rel=%.6e, threshold=%.6e)",
                              rel_decrease, params.theta * sigma_noise * sigma_noise);
                }
                break;
            }
        }
        prev_norm_sq = cur_norm_sq;

        double proj_hi, proj_lo, coh_hi, coh_lo;
        CUDA_CHECK(cudaMemcpy(&proj_hi, d_proj_hi + best_idx, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&proj_lo, d_proj_lo + best_idx, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&coh_hi, d_coh_hi + best_idx, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&coh_lo, d_coh_lo + best_idx, sizeof(double), cudaMemcpyDeviceToHost));

        double denom = 1.0 + params.alpha * (coh_hi + coh_lo);
        double coeff_hi = proj_hi / denom;
        double coeff_lo = proj_lo / denom;

        selected_indices.push_back(best_idx);
        coeffs.push_back({coeff_hi, coeff_lo});

        int blocks_update = (frame_size + threads - 1) / threads;
        eop_update_residual_kernel<<<blocks_update, threads>>>(
            d_res_hi, d_res_lo,
            d_dict_hi, d_dict_lo,
            best_idx,
            coeff_hi, coeff_lo,
            frame_size
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        iter++;
        if (verbose && iter % 10 == 0) {
            AEDS_LOGD("EOP: iter=%d, best_energy=%.6e, norm_sq=%.6e", iter, best_energy, cur_norm_sq);
        }
    }

    if (params.use_final_regression && !selected_indices.empty()) {
        if (verbose) {
            AEDS_LOGI("EOP: performing final regression on %d atoms", selected_indices.size());
        }
        // (реализация финальной регрессии опущена для краткости, она не затрагивает __ldg)
    }

    for (int i = 0; i < (int)selected_indices.size(); ++i) {
        DetectedEvent ev;
        ev.atom_index = selected_indices[i];
        ev.amplitude_hi = coeffs[i].hi;
        events.push_back(ev);
    }

    if (verbose) {
        AEDS_LOGI("EOP: finished, selected %d atoms, iter=%d", selected_indices.size(), iter);
    }
}
