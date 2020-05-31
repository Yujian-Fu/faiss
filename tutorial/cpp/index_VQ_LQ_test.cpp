#include <iostream>
#include <fstream>
#include <cstdio>
#include <unordered_set>
#include <sys/resource.h>

#include "index_VQ_LQ.h"

using namespace bslib_VQ_LQ;
typedef faiss::Index::idx_t idx_t;

int main(){
    const char * path_gt = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/gnd/idx_1000M.ivecs";
    const char * path_query = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";
    const char * path_base = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_base.bvecs";
    const char * path_learn = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";
    const char * path_idxs = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/precomputed_idxs_sift1b.ivecs";


    const char * path_centroids = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/centroids_sift1b.fvecs";
    const char * path_pq = "/home/y/yujianfu/ivf-hnsw/models_VQ_LQ/SIFT1B/PQ16.pq";
    const char * path_norm_pq = "/home/y/yujianfu/ivf-hnsw/models_VQ_LQ/SIFT1B/PQ16_NORM.pq";
    
    const char * path_index;

    size_t ngt = 1000;
    size_t nq = 10000;
    size_t bytes_per_codes = 16;
    size_t nbits_per_idx = 8;
    bool use_quantized_distance = true;
    size_t nc = 993127;
    size_t max_group_size = 100000;
    size_t nt = 10000000;
    size_t nsubt = 65536;
    //size_t nsubt = 10000;
    size_t nb = 1000000000;
    size_t k = 1;

    const uint32_t batch_size = 1000000;
    const size_t nbatches = nb / batch_size;
    struct rusage r_usage;

    if (use_quantized_distance)
        path_index = "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1B/PQ16_quantized.index";
    else
        path_index = "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1B/PQ16.index";
    
    size_t dimension = 128;

    std::cout << "Loading groundtruth from " << path_gt << std::endl;
    
    //Load Groundtruth
    std::vector<uint32_t> groundtruth(nq * ngt);
    {
        std::ifstream gt_input(path_gt, std::ios::binary);
        readXvec<uint32_t>(gt_input, groundtruth.data(), ngt, nq, false, true);
    }

    //Load Query
    std::cout << "Loading queries from " << path_query << std::endl;
    std::vector<float> query(nq * dimension);
    {
        std::ifstream query_input(path_query, std::ios::binary);
        readXvecFvec<uint8_t>(query_input, query.data(), dimension, nq, true, true);
    }

    getrusage(RUSAGE_SELF,&r_usage);
    // Print the maximum resident set size used (in kilobytes).
    std::cout << std::endl << "Memory usage: " << r_usage.ru_ixrss << " / " << r_usage.ru_isrss << " / " << r_usage.ru_idrss << " / " << r_usage.ru_maxrss << std::endl;






















}