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
