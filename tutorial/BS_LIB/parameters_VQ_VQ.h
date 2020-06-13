#include <cstdio>
#include <iostream>

/* Parameter setting: */
//Exp parameters
//For index initialization
const size_t layers = 2;
const std::string index_type[layers] = {"VQ", "VQ"};
const uint32_t ncentroids[layers] = {1500, 500};
const char * path_quantizers = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1B/quantizer_1500_500.qt";;

//For train PQ
const size_t bytes_per_code = 16;
const size_t bytes_per_norm_code = 1;
const size_t nbits = 8; //Or 16
const size_t nt = 100000000;
const size_t subnt = 10000000;
const size_t dimension = 128;

//For assigning ID

//For building index
const size_t nb = 1000000000;
const uint32_t batch_size = 1000000;
const size_t nbatches = nb / batch_size; //1000

//For searching
const size_t ngt = 1000;
const size_t nq = 10000;
const size_t result_k = 1;
const size_t max_vectors = 50000;
size_t search_space[layers] = {100, 20};
size_t keep_space[layers] = {50, 10};


bool is_recording = true;

// Folder path
const char * folder_model = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ";
const char * folder_data = "/home/y/yujianfu/ivf-hnsw/data";

//File paths
const char * path_learn = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";
const char * path_pq = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1B/PQ16_1500_500.pq";
const char * path_pq_norm = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1B/PQ_NORM16_1500_500.pq";

const char * path_base = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_base.bvecs";
const char * path_idxs = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1B/base_idxs_1500_500.ivecs";

const char * path_index = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1B/PQ16_1500_500.index";

const char * path_gt = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/gnd/idx_1000M.ivecs";
const char * path_query = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";

const char * path_record = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1B/recording_1500_500.txt";
