#include "quantizer.h"

namespace bslib{
    struct LQ_quantizer : Base_quantizer
    {
        faiss::IndexFlatL2 all_quantizer; 
        std::vector<float> alphas; // Initialized in read_quantizer
        std::vector<float> upper_centroids; // Initialized in constructor
        std::vector<std::vector<idx_t>> nn_centroid_idxs; // Initialized in constructor
        std::vector<std::vector<float>> nn_centroid_dists; // Initialized in constructor
        explicit LQ_quantizer(size_t dimension, size_t nc_upper, size_t nc_per_group, const float * upper_centroids,
                              const idx_t * upper_centroid_idxs, const float * upper_centroid_dists);
        void build_centroids(const float * train_data, size_t train_set_size, idx_t * train_set_idxs);
        float compute_alpha(const float * centroid_vectors, const float * points, const float * centroid,
                                      const float * centroid_vector_norms_L2sqr, size_t group_size);
        void compute_final_centroid(idx_t label, float * sub_centroid);
        void compute_residual_group_id(size_t n, const idx_t * labels, const float * x, float * residuals);
        void recover_residual_group_id(size_t n, const idx_t * labels, const float * residuals, float * x);
        void search_in_group(size_t n, size_t k, float * dists, idx_t * labels, const idx_t * group_id, const float * query_nn_dists);
        void search_all(size_t n, const float * instance, size_t k, float * dists, idx_t * labels);
        void compute_nn_centroids(size_t k, float * nn_centroids, float * nn_centroid_dists, idx_t * labels);
    };
}