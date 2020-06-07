#include <cstdio>
#include <iostream>

/* Parameter setting: */
//Exp parameters
//For index initialization
const size_t layers = 4;
const std::string index_type[layers] = {"VQ", "VQ", "VQ", "LQ"};
const uint32_t ncentroids[layers] = {100, 100, 100, 10};
const char * path_quantizers = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ_VQ_LQ/SIFT1B/quantizer_100_100_100_10.qt";;

//For train PQ
const size_t bytes_per_code = 16;
const size_t bytes_per_norm_code = 1;
const size_t nbits = 8; //Or 16
const size_t nt = 100000000;
const size_t subnt = 10000000;
const size_t dimension = 128;

//For assigning ID

//For building index
const size_t nb = 100000000;
const uint32_t batch_size = 1000000;
const size_t nbatches = nb / batch_size; //1000

//For searching
const size_t ngt = 1000;
const size_t nq = 10000;
const size_t result_k = 1;
const size_t max_vectors = 10000;
size_t search_space[4] = {10, 10, 10, 10};
size_t keep_space[4] = {5, 5, 5, 5};


bool is_recording = true;
// Folder path
const char * folder_model = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ_VQ_LQ";
const char * folder_data = "/home/y/yujianfu/ivf-hnsw/data";

//File paths
const char * path_learn = (char *) (std::string(folder_data) + "/SIFT1B/bigann_base.bvecs").c_str();
const char * path_pq = (char *) ((std::string(folder_model) + "/SIFT1B/PQ16.pq").c_str());
const char * path_pq_norm = (char *) ((std::string (folder_model) + "SIFT1B/PQ_NORM.pq").c_str());

const char * path_base = (char *) (std::string(folder_data) + "/SIFT1B/bigann_base.bvecs").c_str();
const char * path_idxs = (char *) (std::string(folder_model) + "/SIFT1B/base_idxs.ivecs").c_str();

const char * path_index = (char *) (std::string(folder_model) + "/SIFT1B/PQ16.index").c_str();

const char * path_gt = (char *) (std::string(folder_data) + "/SIFT1B/gnd/idx_1000M.ivecs").c_str();
const char * path_query = (char *) (std::string(folder_data) + "/SIFT1B/bigann_learn.bvecs").c_str();

const char * path_record = (char *) (std::string(folder_model) + "/SIFT1B/recording.txt").c_str();
