#include "quantizer.h"
#include "HNSWlib/hnswalg.h"
#include "faiss/IndexHNSW.h"

namespace bslib{
    struct VQ_quantizer : Base_quantizer
    {
        /*
        dimension:initilized in constructor
        nc_upper:initilized in constructor
        nc_per_group:initilized in constructor
        XXXmap: initilized in constructor
        */
        size_t M;
        size_t efConstruction;
        size_t efSearch;
        bool use_HNSW;
        std::vector<hnswlib::HierarchicalNSW *> HNSW_quantizers;
        std::vector<faiss::IndexFlatL2 *> L2_quantizers; //Resized with nc_upper in read quantizer

         // The size should be train_set_size, the max size is nc_upper
        explicit VQ_quantizer(size_t dimension, size_t nc_upper, size_t nc_per_group, size_t M = 16, size_t efConstruction = 100, size_t efSearch = 50, bool use_HNSW = false);
        void build_centroids(const float * train_data, size_t train_set_size, idx_t * train_data_ids);
        void compute_final_centroid(const idx_t group_id, const idx_t inner_group_id, float * final_centroid);
        void compute_residual_group_id(size_t n, const idx_t * labels, const float * x, float * residuals); 
        void recover_residual_group_id(size_t n, const idx_t * labels, const float * residuals, float * x);
        void search_in_group(const float * query, const idx_t group_id, float * result_dists, idx_t * result_labels, size_t k);
        void search_all(const size_t n, const size_t k, const float * query_data, idx_t * query_data_ids);
        void compute_nn_centroids(size_t k, float * nn_centroids, float * nn_dists, idx_t * labels);
        void write_HNSW(std::ofstream & output);
        void read_HNSW(std::ifstream & input);
    };
}

