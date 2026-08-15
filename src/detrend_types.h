#ifndef DETREND_TYPES_H
#define DETREND_TYPES_H

#include "alw_math.h"

struct DetrendSegment {
    int start = 0, end = 0, order = 0, n = 0;
    double sse = 0.0, p_value = 1.0;
    bool is_significant = false, significant = false;
    alw_vector<double> coeff_hi;
    alw_vector<double> coeff_lo;
};

struct DetrendGpuSegment { int start, end, order; };

struct DetrendSegmentStats {
    int start, end, order;
    double sse, p_value, local_r_squared;
};

struct DetrendStatsFull {
    double time_total = 0.0, time_reg = 0.0, time_initial_sse = 0.0;
    double time_iterations = 0.0, time_final_fit = 0.0, time_stitching = 0.0;
    int num_iterations = 0, num_segments = 0, num_chunks = 0;
    double global_r_squared = 0.0, global_snr_db = 0.0, total_sse = 0.0;
    size_t peak_memory_bytes = 0;
    int num_allocations = 0;
    alw_vector<DetrendSegmentStats> segments;
};

struct DetrendChunkConfig {
    bool chunking_enabled = false;
    int num_chunks = 1;
    int chunk_size = 0;
    int overlap = 0;
};

#endif // DETREND_TYPES_H
