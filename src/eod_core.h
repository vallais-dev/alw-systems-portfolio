#ifndef EOD_CORE_H
#define EOD_CORE_H

#include "alw_math.h"
#include "eop_core.h"

struct EOD_Params {
    int max_iter;
    double learning_rate;
    double epsilon;
    double tolerance;
    int hop_size;
    int max_atoms_per_frame;
    bool use_aeds;
    bool verbose;
    double regularization_lambda;
    double sigma_noise;
};

void eod_learn_dictionary_gpu(
    const double* prices,
    int N,
    int frame_size,
    int num_atoms,
    const double* init_dict,
    const EOP_Params& eop_params,
    const EOD_Params& eod_params,
    double* out_dict,
    alw_vector<double>& atom_norms
);

#endif // EOD_CORE_H
