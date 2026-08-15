#ifndef AMAD_X_H
#define AMAD_X_H

#include "alw_math.h"

#ifdef __cplusplus
extern "C" {
#endif

double amad_x_estimate(
    const double* d_res_hi,
    const double* d_res_lo,
    int N,
    int min_block_size,
    int max_block_size,
    int num_scales,
    int refine_iter,
    bool verbose
);

double amad_x_estimate_adaptive(
    const double* d_res_hi,
    const double* d_res_lo,
    int N,
    bool verbose
);

#ifdef __cplusplus
}
#endif

#endif // AMAD_X_H
