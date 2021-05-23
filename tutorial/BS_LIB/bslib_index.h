#include <string>
#include "VQ_quantizer.h"
#include "LQ_quantizer.h"
#include "PQ_quantizer.h"
#include "utils/utils.h"
#include <unordered_set>
#include <algorithm>
#include "faiss/VectorTransform.h"
#define VALIDATION_EPSILON 5

namespace bslib{

// Change this type for different datasets
typedef uint8_t learn_data_type;
typedef uint8_t base_data_type;

//typedef float learn_data_type;
//typedef float base_data_type;

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
    bool use_OPQ;
    bool use_norm_quantization;
    bool use_train_selector;
    bool saving_index;
    bool is_recording;

    memory_recorder Mrecorder = memory_recorder();
    recall_recorder Rrecorder = recall_recorder();
    time_recorder Trecorder = time_recorder();

    

    size_t M; // Initialized by training pq
    size_t norm_M;
    size_t nbits; // Initialized by training pq
    size_t code_size; // Initialized by reading PQ
    size_t norm_code_size; // Initialized by reading PQ
    size_t final_group_num; // Initialized by compute final nc
    size_t max_visited_vectors; //
    size_t reranking_space;
    
    std::vector<size_t> ncentroids;
    std::vector<VQ_quantizer > vq_quantizer_index; // Initialized in read_quantizer
    std::vector<LQ_quantizer > lq_quantizer_index; // Initialized in read_quantizer
    std::vector<PQ_quantizer > pq_quantizer_index;

    faiss::ProductQuantizer pq; // Initialized in train_pq
    faiss::ProductQuantizer norm_pq; // Initialized in train_pq
    faiss::OPQMatrix * opq_matrix;

    std::vector<float> base_norms;
    std::vector<uint8_t> base_norm_codes;
    std::vector<float> centroid_norms;
    std::vector<uint8_t> centroid_norm_codes;

    std::vector<std::vector<uint8_t>> base_codes;
    std::vector<std::vector<idx_t>> base_sequence_ids;
    //std::vector<std::vector<idx_t>> base_pre_hash_ids;

    std::vector<std::vector<idx_t>> train_set_ids; // This is for train data selection 
    std::vector<float> train_data; // Initialized in build_quantizers (without reading)
    std::vector<idx_t> train_data_ids; // Initialized in build_quantizers (without reading)

    explicit Bslib_Index(const size_t dimension, const size_t layers, const std::string * index_type, 
    const bool use_reranking, const bool save_index, const bool use_norm_quantization, const bool is_recording,
    const bool use_HNSW_VQ, const bool use_HNSW_group, const bool use_OPQ, const bool use_train_selector,
    const size_t train_size, const size_t M_PQ, const size_t nbits);

    void do_OPQ(idx_t n, float * dataset);
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
    void add_batch(size_t n, const float * data, const idx_t * sequence_ids, const idx_t * group_ids, const size_t * group_positions);
    void get_final_group_num();
    void compute_centroid_norm();
    void keep_k_min(const size_t m, const size_t k, const float * all_dists, const idx_t * all_labels, float * sub_dists, idx_t * sub_labels);
    void search(size_t n, size_t result_k, float * queries, float * query_dists, idx_t * query_ids, const size_t * keep_space, uint32_t * groundtruth, std::string path_base);
    size_t get_next_group_idx(size_t keep_result_space, idx_t * group_ids, float * query_group_dists, std::pair<idx_t, float> & result_idx_dist);
    
    float pq_L2sqr(const uint8_t *code, const float * precomputed_table);

    void read_quantizers(const std::string path_quantizers);
    void write_quantizers(const std::string path_quantizers);
    void read_index(const std::string path_index);
    void write_index(const std::string path_index);

    void get_final_centroid(const size_t group_id, float * final_centroid);

    void build_index(const size_t M_PQ, std::string path_learn, std::string path_groups, std::string path_labels,
    std::string path_quantizers, uint32_t VQ_layers, uint32_t PQ_layers, std::string path_OPQ, 
    const uint32_t * ncentroids, const size_t * M_HNSW, const size_t * efConstruction, 
    const size_t * efSearch, const size_t * M_PQ_layer, const size_t * nbits_PQ_layer, const size_t * num_train,
    size_t OPQ_train_size, size_t selector_train_size, size_t selector_group_size, std::ofstream & record_file);

    void assign_vectors(std::string path_ids, std::string path_base, uint32_t batch_size, size_t nbatches, std::ofstream & record_file);

    void train_pq_quantizer(std::string path_pq, std::string path_pq_norm,
        size_t M_norm_PQ, std::string path_learn, size_t PQ_train_size, std::ofstream & record_file);

    void load_index(std::string path_index, std::string path_ids, std::string path_base,
        size_t batch_size, size_t nbatches, size_t nb, std::ofstream & record_file);
    
    void index_statistic();

    void query_test(size_t num_search_paras, size_t num_recall, size_t nq, size_t ngt,
        const size_t * max_vectors, const size_t * result_k, const size_t * keep_space, const size_t * reranking_space,
        std::ofstream & record_file, std::ofstream & qps_record_file, 
        std::string search_mode, std::string path_base, std::string path_gt, std::string path_query);

    /**
     ** This is the function for dynamically get the reranking space for 
     ** For future work
     **/
    size_t get_reranking_space(size_t k, size_t group_label, size_t group_id);


};
}