#include <cstdio>
#include <iostream>

typedef float origin_data_type;

/* Parameter setting: */
//Exp parameters
//For index initialization
const size_t layers = 1;
const size_t VQ_layers = 1;
const size_t PQ_layers = 0;
const std::string index_type[layers] = {"VQ"};
const uint32_t ncentroids[layers] = {10000};

const bool pq_use_subset = false;
const bool use_reranking = false;
const bool use_HNSW_VQ = false;
const bool use_HNSW_group = false;
const bool use_norm_quantization = false;
const bool use_train_selection = false;
const bool use_dynamic_reranking = false;
const bool use_OPQ = false;
const size_t reranking_space = 20;

//For train PQ
const size_t M_PQ = 16;
const size_t M_norm_PQ = 1;
const size_t nbits = 8; //Or 16
const size_t train_size = 1000000; //This is the size of train set
const size_t dimension = 128;

//For assigning ID

//For building index
const size_t M_HNSW[VQ_layers] = {16};
const size_t efConstruction [VQ_layers] = {300};
const size_t efSearch[VQ_layers] = {100};

const size_t M_PQ_layer[PQ_layers] = {};
const size_t nbits_PQ_layer[PQ_layers] = {};

const size_t selector_train_size = 100000;
const size_t selector_group_size = 1000;

const size_t PQ_train_size = 10000;

const size_t num_train[layers] = {100000};
const size_t nb = 1000000;
const uint32_t batch_size = 10000;
const size_t nbatches = nb / batch_size; //100

//For searching
const size_t ngt = 100;
const size_t nq = 1000;
const size_t result_k = 1;
const size_t max_vectors = 3000;
size_t keep_space[layers] = {100};

bool is_recording = true;

// Folder path
const char * folder_model = "/home/y/yujianfu/ivf-hnsw/models_VQ";

//File paths
const char * path_learn =     "/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_base.fvecs";
const char * path_base =      "/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_base.fvecs";
const char * path_gt =        "/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_groundtruth.ivecs";
const char * path_query =     "/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_query.fvecs";

const char * path_record =    "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1M/recording_10000.txt";
const char * path_pq =        "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1M/PQ16_10000.pq";
const char * path_pq_norm =   "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1M/PQ_NORM16_10000.pq";
const char * path_idxs =      "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1M/base_idxs_10000.ivecs";
const char * path_index =     "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1M/PQ16_10000.index";
const char * path_quantizers = "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1M/quantizer_10000.qt";
//const char * path_quantizers = "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1M/centroids_sift1M.fvecs";

/**
 **This is the centroids for assigining origin train vectors  size: n_group * dimension
 **/
const char * path_groups = "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1M/selector_centroids_1000.fvecs";
//This is for recording the labels based on the generated centroids

/**
 ** This is the labels for all assigned vectors, n_group * group_size 
 **/
const char * path_labels = "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1M/selector_ids";
