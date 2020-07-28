#include <string>
#include "VQ_quantizer.h"
#include "LQ_quantizer.h"
#include "PQ_quantizer.h"
#include "utils.h"
#include <unordered_set>
#include <algorithm>
#include <faiss/VectorTransform.h>
#define EPSILON 5

namespace bslib{


// Change this type for different datasets
typedef float learn_data_type;
typedef float base_data_type;

typedef faiss::Index::idx_t idx_t;
typedef std::pair<std::pair<size_t, size_t>, size_t> HNSW_para;
typedef std::pair<size_t, size_t> PQ_para;

struct Bslib_Index{
    size_t dimension; // Initialized in constructer
    size_t layers; // Initialized in constructer
    std::vector<std::string> index_type; // Initialized in constructer
    size_t train_size; //Initialized in constructer by 0, assigned in main
    size_t max_group_size;
    bool use_reranking;
    bool use_HNSW_VQ;
    bool use_HNSW_group;
    bool use_norm_quantization;
    bool use_OPQ;
    size_t reranking_space;

    size_t M; // Initialized by training pq
    size_t norm_M;
    size_t nbits; // Initialized by training pq
    size_t code_size; // Initialized by reading PQ
    size_t norm_code_size; // Initialized by reading PQ
    size_t final_nc; // Initialized by compute final nc
    size_t max_visited_vectors; //
    
    
    std::vector<size_t> ncentroids;
    std::vector<VQ_quantizer > vq_quantizer_index; // Initialized in read_quantizer
    std::vector<LQ_quantizer > lq_quantizer_index; // Initialized in read_quantizer
    std::vector<PQ_quantizer > pq_quantizer_index;

    faiss::ProductQuantizer pq; // Initialized in train_pq
    faiss::ProductQuantizer norm_pq; // Initialized in train_pq
    faiss::LinearTransform opq_matrix;

    std::vector<float> centroid_norms;
    //std::vector<uint8_t> centroid_norm_codes;

    std::vector<std::vector<uint8_t>> base_norm_codes;
    std::vector<std::vector<uint8_t>> base_codes;
    std::vector<std::vector<float>> base_norm;
    std::vector<std::vector<idx_t>> origin_ids;

    std::vector<float> precomputed_table; 

    std::vector<std::vector<idx_t>> train_set_ids;
    std::vector<float> train_data; // Initialized in build_quantizers (without reading)
    std::vector<idx_t> train_data_ids; // Initialized in build_quantizers (without reading)

    explicit Bslib_Index(const size_t dimension, const size_t layers, const std::string * index_type, const bool use_HNSW_VQ, const bool use_norm_quantization);

    void build_quantizers(const uint32_t * ncentroids, const std::string path_quantizer, const std::string path_learn, const size_t * num_train, const std::vector<HNSW_para> HNSW_paras, const std::vector<PQ_para> PQ_paras);
    
    void add_vq_quantizer(size_t nc_upper, size_t nc_per_group, size_t M, size_t efConstruction, size_t efSearch);
    void add_lq_quantizer(size_t nc_upper, size_t nc_per_group, const float * upper_centroids, const idx_t * upper_nn_centroid_idxs, const float * upper_nn_centroid_dists);
    void add_pq_quantizer(size_t nc_upper, size_t M, size_t nbits);

    void build_train_selector(const std::string path_learn, const std::string path_groups, const std::string path_labels, size_t total_train_size, size_t sub_train_size, size_t group_size);
    void read_train_set(const std::string path_learn, size_t total_size, size_t train_set_size);

    void train_pq(const std::string path_pq, const std::string path_norm_pq, const std::string path_learn, const size_t train_set_size);

    void encode(size_t n, const float * data, const idx_t * encoded_ids, float * encoded_data);
    void decode(size_t n, const float * encoded_data, const idx_t * encoded_ids, float * decoded_data);
    void assign(const size_t n, const float * assign_data, idx_t * assigned_ids);
    void add_batch(size_t n, const float * data, const idx_t * origin_ids, idx_t * encoded_ids);
    void get_final_nc();
    void compute_centroid_norm();
    void search(size_t n, size_t result_k, float * queries, float * query_dists, idx_t * query_ids, size_t * keep_space, uint32_t * groundtruth, std::string path_base);
    void get_next_group_idx(size_t keep_result_space, idx_t * group_ids, float * query_group_dists, std::pair<idx_t, float> & result_idx_dist);
    void keep_k_min(const size_t m, const size_t k, const float * all_dists, const idx_t * all_ids, float * sub_dists, idx_t * sub_ids);
    float pq_L2sqr(const uint8_t *code);

    void read_quantizers(const std::string path_quantizers);
    void write_quantizers(const std::string path_quantizers);
    void read_index(const std::string path_index);
    void write_index(const std::string path_index);

    void get_final_centroid(const size_t group_id, float * final_centroid);


    /**
     ** This is the function for dynamically get the reranking space for 
     **/
    size_t get_reranking_space(size_t k, size_t group_label, size_t group_id);
};
}