#include "quantizer.h"

namespace bslib{
    struct PQ_quantizer : Base_quantizer
    {
        size_t M;
        size_t nbits;
        size_t ksub;
        size_t dsub;
        std::vector<faiss::ProductQuantizer> PQs;
        explicit PQ_quantizer(size_t dimension, size_t nc_upper, size_t M, size_t nbits);

        void build_centroids(const float * train_data, size_t train_set_size, idx_t * train_set_idxs, bool update_idxs);
        void compute_final_centroid(idx_t group_idx, idx_t group_label, float * final_centroid);
        void compute_residual_group_id(size_t n, const idx_t * group_idxs, const idx_t * group_labels, const float * x, float * residuals);
        void recover_residual_group_id(size_t n, const idx_t * group_idxs, const idx_t * group_labels, const float * residuals, float * x);
        void search_in_group(size_t n, const float * queries, const idx_t * group_idxs, float * result_dists,uint8_t * result_labels, size_t keep_space);
        void multi_sequence_sort(const float * dist_sequence, size_t keep_space, float * result_dists, idx_t * result_labels);
    };
}
