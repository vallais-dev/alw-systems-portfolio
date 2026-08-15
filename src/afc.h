#ifndef AFC_H
#define AFC_H

#include "alw_math.h"

struct AFC_Params {
    int coarse_points;
    int refine_iter;
    double inertia_alpha;
    double freq_min;
    double freq_max;
    bool use_dd;
};

double afc_find_frequency(
    const double* d_y_hi,
    const double* d_y_lo,
    int N,
    double f_prev,
    const AFC_Params& params,
    bool verbose = false,
    double* d_energy_out = nullptr
);

#endif // AFC_H
