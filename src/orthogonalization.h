// =============================================================================
// orthogonalization.h — объявления функций ортогонализации (без CUDA Streams)
// =============================================================================

#ifndef ORTHOGONALIZATION_H
#define ORTHOGONALIZATION_H

#include "alw_math.h"
#include <string>

// =============================================================================
// УПРАВЛЕНИЕ АРЕНОЙ ДЛЯ КОПИРОВАНИЯ ПОДМАТРИЦ
// =============================================================================

void init_orthogonal_arena(size_t max_size);

// =============================================================================
// ОРТОГОНАЛИЗАЦИЯ (без поддержки CUDA Streams)
// =============================================================================

bool orthogonalize_basis_hybrid(double* basis_hi, double* basis_lo,
                                double* R_hi, double* R_lo,
                                double* orig_hi, double* orig_lo,
                                int num_basis, int N,
                                const OrthoParams& params = OrthoParams());

bool orthogonalize_basis_batch(double* basis_hi, double* basis_lo,
                               double* R_hi, double* R_lo,
                               double* orig_hi, double* orig_lo,
                               int batch_size, int num_basis, int N,
                               const OrthoParams& params = OrthoParams());

// =============================================================================
// ОЦЕНКА ОШИБКИ ОРТОГОНАЛЬНОСТИ (экспортируемая функция)
// =============================================================================

double estimate_orthogonality_error(const double* d_Q_hi, const double* d_Q_lo,
                                    int num_basis, int N, bool verbose);

// =============================================================================
// СОХРАНЕНИЕ / ЗАГРУЗКА БАЗИСА
// =============================================================================

void save_basis_to_file(const std::string& filename, int num_basis, int frame_size,
                        const double* T_hi, const double* T_lo,
                        const double* R_hi, const double* R_lo);

bool load_basis_from_file(const std::string& filename, int expected_num_basis, int expected_frame_size,
                          double* T_hi, double* T_lo, double* R_hi, double* R_lo);

void save_basis_batch_to_file(const std::string& filename,
                              int batch_size, int num_basis, int frame_size,
                              const double* T_hi, const double* T_lo,
                              const double* R_hi, const double* R_lo);

bool load_basis_batch_from_file(const std::string& filename,
                                int expected_batch_size, int expected_num_basis, int expected_frame_size,
                                double* T_hi, double* T_lo, double* R_hi, double* R_lo);

#endif // ORTHOGONALIZATION_H
