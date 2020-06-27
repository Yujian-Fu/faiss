#include "quantizer.h"

namespace bslib{
    struct VQ_quantizer : Base_quantizer
    {
        /*
        dimension:initilized in constructor
        nc_upper:initilized in constructor
        nc_per_group:initilized in constructor
        XXXmap: initilized in constructor
        */
        std::vector<faiss::IndexFlatL2 > quantizers; //Resized with nc_upper in read quantizer
         // The size should be train_set_size, the max size is nc_upper
        explicit VQ_quantizer(size_t dimension, size_t nc_upper, size_t nc_per_group);
        void build_centroids(const float * train_data, size_t train_set_size, idx_t * train_data_idx, bool update_idxs);
        void compute_final_centroid(idx_t label, float * sub_centroid);
        void compute_residual_group_id(size_t n, const idx_t * labels, const float * x, float * residuals); 
        void recover_residual_group_id(size_t n, const idx_t * labels, const float * residuals, float * x);
        void search_in_group(size_t n, const float * queries, const idx_t * group_idxs, float * result_dists);
        void compute_nn_centroids(size_t k, float * nn_centroids, float * nn_dists, idx_t * labels);
    };
}

