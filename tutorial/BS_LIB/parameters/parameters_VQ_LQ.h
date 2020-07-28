#include <cstdio>
#include <iostream>

typedef uint8_t origin_data_type;

/* Parameter setting: */
//Exp parameters
//For index initialization
const size_t layers = 2;
const std::string index_type[layers] = {"VQ", "LQ"};
const uint32_t ncentroids[layers] = {10000, 100};
const char * path_quantizers = "/home/y/yujianfu/ivf-hnsw/models_VQ_LQ/SIFT1B/quantizer_10000_100.qt";;
const bool use_subset = true;
const bool pq_use_subset = true;
const bool use_reranking = true;
const size_t reranking_space = 200;

//For train PQ
const size_t bytes_per_code = 16;
const size_t bytes_per_norm_code = 1;
const size_t nbits = 8; //Or 16
const size_t nt = 100000000;
const size_t subnt = 1000000;
const size_t dimension = 128;

//For assigning ID

//For building index
const size_t nb = 1000000000;
const uint32_t batch_size = 1000000;
const size_t nbatches = nb / batch_size; //1000

//For searching
const size_t ngt = 1000;
const size_t nq = 10000;
const size_t result_k = 10;
const size_t max_vectors = 10000;
size_t keep_space[layers] = {50, 10};


bool is_recording = true;

// Folder path
const char * folder_model = "/home/y/yujianfu/ivf-hnsw/models_VQ_LQ";
const char * folder_data = "/home/y/yujianfu/ivf-hnsw/data";

//File paths
const char * path_learn = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";
const char * path_base = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_base.bvecs";
const char * path_gt = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/gnd/idx_1000M.ivecs";
const char * path_query = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";

const char * path_record =    "/home/y/yujianfu/ivf-hnsw/models_VQ_LQ/SIFT1B/recording_10000_100.txt";
const char * path_pq =        "/home/y/yujianfu/ivf-hnsw/models_VQ_LQ/SIFT1B/PQ16_10000_100.pq";
const char * path_pq_norm =   "/home/y/yujianfu/ivf-hnsw/models_VQ_LQ/SIFT1B/PQ_NORM16_10000_100.pq";
const char * path_idxs =      "/home/y/yujianfu/ivf-hnsw/models_VQ_LQ/SIFT1B/base_idxs_10000_100.ivecs";
const char * path_index =     "/home/y/yujianfu/ivf-hnsw/models_VQ_LQ/SIFT1B/PQ16_10000_100.index";
