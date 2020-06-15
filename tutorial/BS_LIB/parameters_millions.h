#include <cstdio>
#include <iostream>

typedef float origin_data_type;

/* Parameter setting: */
//Exp parameters
//For index initialization
const size_t layers = 2;
const std::string index_type[layers] = {"VQ", "VQ"};
const uint32_t ncentroids[layers] = {100, 50};
const char * path_quantizers = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1M/quantizer_100_50.qt";
const bool use_subset = false;
const bool pq_use_subset = true;

//For train PQ
const size_t bytes_per_code = 16;
const size_t bytes_per_norm_code = 1;
const size_t nbits = 8; //Or 16
const size_t nt = 100000;
const size_t subnt = 10000;
const size_t dimension = 128;

//For assigning ID

//For building index
const size_t nb = 1000000;
const uint32_t batch_size = 100000;
const size_t nbatches = nb / batch_size; //1000

//For searching
const size_t ngt = 100;
const size_t nq = 10000;
const size_t result_k = 10;
const size_t max_vectors = 2000;
size_t keep_space[layers] = {10, 2};

bool is_recording = true;

// Folder path
const char * folder_model = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ";

//File paths
const char * path_learn =     "/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_learn.fvecs";
const char * path_base =      "/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_base.fvecs";
const char * path_gt =        "/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_groundtruth.ivecs";
const char * path_query =     "/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_query.fvecs";

const char * path_record =    "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1M/recording_100_50.txt";
const char * path_pq =        "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1M/PQ16_100_50.pq";
const char * path_pq_norm =   "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1M/PQ_NORM16_100_50.pq";
const char * path_idxs =      "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1M/base_idxs_100_50.ivecs";
const char * path_index =     "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1M/PQ16_100_50.index";

