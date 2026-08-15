#ifndef EOP_CORE_H
#define EOP_CORE_H

#include "alw_math.h"

struct EOP_Params {
    double alpha;
    double gamma;
    double theta;
    int max_iterations;
    bool use_final_regression;
};

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
);

#endif // EOP_CORE_H
