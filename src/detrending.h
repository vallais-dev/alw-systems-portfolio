#ifndef DETRENDING_H
#define DETRENDING_H

#include "alw_math.h"

void interpolate_nan(alw_vector<double>& data);
bool check_and_handle_nan(alw_vector<double>& data_hi,
                          alw_vector<double>& data_lo,
                          int N,
                          bool verbose);

bool can_merge_segments(const Segment& left,
                        const Segment& right,
                        const alw_vector<double>& y_hi,
                        const alw_vector<double>& y_lo,
                        double significance_threshold,
                        double stitch_threshold);

void stitch_chunk_boundaries(alw_vector<Segment>& segments,
                             const alw_vector<double>& y_hi,
                             const alw_vector<double>& y_lo,
                             int overlap_start,
                             int overlap_end,
                             double significance_threshold,
                             double stitch_threshold,
                             int huber_iter,
                             double huber_c,
                             bool verbose);

void save_detrend_stats_to_json(const DetrendStats& stats,
                                const alw_string& filename);

void adaptive_detrend(const alw_vector<double>& y_hi,
                      const alw_vector<double>& y_lo,
                      int N,
                      alw_vector<double>& trend_hi,
                      alw_vector<double>& trend_lo,
                      alw_vector<Segment>& out_segments,
                      int max_segments,
                      int min_segment_len,
                      double bic_threshold,
                      double lambda,
                      int max_order,
                      bool auto_order,
                      bool compute_significance,
                      double significance_threshold,
                      int huber_iter,
                      double huber_c,
                      bool verbose = false,
                      DetrendStats* out_stats = nullptr,
                      double stitch_threshold = 2.0);

#endif // DETRENDING_H
