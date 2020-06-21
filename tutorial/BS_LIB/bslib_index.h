#include <string>
#include "VQ_quantizer.h"
#include "LQ_quantizer.h"
#include "utils.h"
#include <unordered_set>
#include <algorithm>

#define MAX_DIST 1e9
#define INVALID_ID -1

namespace bslib{

typedef uint32_t idx_t;
typedef uint8_t learn_data_type;

struct Bslib_Index{
    size_t dimension; // Initialized in constructer
    size_t layers; // Initialized in constructer
    std::vector<std::string> index_type; // Initialized in constructer
    size_t nt; //Initialized in constructer by 0, assigned in main
    size_t subnt; //Initialized in constructer by 0, assigned in main 
    bool use_subset;
    bool pq_use_subset;
    bool use_reranking;
    size_t reraking_space;

    size_t M; // Initialized by training pq
    size_t norm_M;
    size_t nbits; // Initialized by training pq
    size_t code_size; // Initialized by reading PQ
    size_t norm_code_size; // Initialized by reading PQ
    size_t final_nc; // Initialized by compute final nc
    size_t max_visited_vectors; //
    
    

    std::vector<VQ_quantizer > vq_quantizer_index; // Initialized in read_quantizer
    std::vector<LQ_quantizer > lq_quantizer_index; // Initialized in read_quantizer
    faiss::ProductQuantizer pq; // Initialized in train_pq
    faiss::ProductQuantizer norm_pq; // Initialized in train_pq
    
    std::vector<float> centroid_norms;
    std::vector<uint8_t> centroid_norm_codes;

    std::vector<std::vector<uint8_t>> base_norm_codes;
    std::vector<std::vector<uint8_t>> base_codes;
    std::vector<std::vector<idx_t>> origin_ids;

    std::vector<float> precomputed_table; 

    std::vector<float> train_data; // Initialized in build_quantizers (without reading)
    std::vector<idx_t> train_data_idxs; // Initialized in build_quantizers (without reading)



    explicit Bslib_Index(const size_t dimension, const size_t layers, const std::string * index_type, const bool use_subset, const bool pq_use_subset);

    void build_quantizers(const uint32_t * ncentroids, const char * path_quantizers, const char * path_learn);
    
    void add_vq_quantizer(size_t nc_upper, size_t nc_per_group, bool update_idxs);
    void add_lq_quantizer(size_t nc_upper, size_t nc_per_group, const float * upper_centroids, const idx_t * upper_nn_centroid_idxs, const float * upper_nn_centroid_dists, bool update_idxs);
    
    void train_pq(const char * path_pq, const char * path_norm_pq, const char * path_learn);

    void encode(size_t n, const float * data, const idx_t * encoded_ids, float * encoded_data);
    void decode(size_t n, const float * encoded_data, const idx_t * encoded_ids, float * decoded_data);
    void assign(const size_t n, const float * assign_data, idx_t * assigned_ids);
    void add_batch(size_t n, const float * data, const idx_t * origin_ids, idx_t * encoded_ids);
    void get_final_nc();
    void compute_centroid_norm();
    void search(size_t n, size_t result_k, float * queries, float * query_dists, faiss::Index::idx_t * query_ids, size_t * keep_space, uint32_t * groundtruth);
    void get_next_group_idx(size_t keep_result_space, idx_t * group_ids, float * query_group_dists, float * result_idx_dist);
    void keep_k_min(const size_t m, const size_t k, const float * all_dists, const idx_t * all_ids, float * sub_dists, idx_t * sub_ids);
    float pq_L2sqr(const uint8_t *code);

    void read_quantizers(const char * path_quantizers);
    void write_quantizers(const char * path_quantizers);
    void read_index(const char * path_index);
    void write_index(const char * path_index);
};
}