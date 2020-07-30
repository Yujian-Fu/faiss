#include "quantizer.h"

namespace bslib{
    struct PQ_quantizer : Base_quantizer
    {
        size_t M;
        size_t nbits;
        size_t ksub;
        size_t dsub;
        size_t hash_size;
        std::vector<faiss::ProductQuantizer *> PQs;
        std::vector<std::vector<float>> centroid_norms;

        explicit PQ_quantizer(size_t dimension, size_t nc_upper, size_t M, size_t nbits);

        idx_t new_pow(size_t k_sub, size_t pow);
        idx_t index_2_id(const idx_t * index, const idx_t group_id);
        idx_t id_2_index(idx_t id, idx_t * index);
        bool traversed(const idx_t * visited_index, const idx_t * index_check, const size_t index_num);
        void build_centroids(const float * train_data, size_t train_set_size, idx_t * train_set_idxs);
        void compute_final_centroid(const idx_t group_idx, float * final_centroid);
        void compute_residual_group_id(size_t n, const idx_t * group_idxs, const float * x, float * residuals);
        void recover_residual_group_id(size_t n, const idx_t * group_idxs, const float * residuals, float * x);
        void search_in_group(size_t n, const float * queries, const idx_t * group_idxs, float * result_dists,idx_t * result_labels, size_t keep_space);
        void multi_sequence_sort(const idx_t group_id, const float * dist_sequence, size_t keep_space, float * result_dists, idx_t * result_labels);
        float get_centroid_norms(const idx_t group_id);
    };
}
