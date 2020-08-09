#include "utils/utils.h"
#include <unordered_set>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/Index.h>
#include <algorithm>
#include <cassert>
//Parameters
    
    const std::string dataset = "SIFT1M";
    const size_t dimension = 128;
    size_t train_set_size = 100000;
    const size_t base_set_size = 1000000;
    const size_t query_set_size = 1000;
    const size_t ngt = 100;
    const size_t recall_test_size = 3;
    const float MIN_DISTANCE = 1e8;
    
    
    /*
    const std::string dataset = "DEEP1M";
    const size_t dimension = 256;
    size_t train_set_size =  100000;
    const size_t base_set_size = 1000000;
    const size_t query_set_size = 1000;
    const size_t ngt = 100;
    const size_t recall_test_size = 3;
    const float MIN_DISTANCE = 100;
    */
    

    const std::string model = "models_VQ_VQ";
    const bool use_fast_assign = false;

    const std::string path_learn = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_learn.fvecs";
    const std::string path_base = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_base.fvecs";
    const std::string path_gt = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_groundtruth.ivecs";
    const std::string path_query = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_query.fvecs";
    std::string path_record = "/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset + "/reasoning_speed.txt";
    

    std::vector<float> train_set(dimension * train_set_size);
    std::vector<float> base_set(dimension * base_set_size);
    std::vector<float> query_set(dimension * query_set_size);
    std::vector<uint32_t> gt_set(ngt * base_set_size);

    std::ifstream train_input(path_learn, std::ios::binary);
    std::ifstream base_input(path_base, std::ios::binary);
    std::ifstream gt_input(path_gt, std::ios::binary);
    std::ifstream query_input(path_query, std::ios::binary);
    std::ofstream record_output;

typedef faiss::Index::idx_t idx_t;
using namespace bslib;
int main(){
    PrepareFolder((char *) ("/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset).c_str());
    
    readXvec<float>(train_input, train_set.data(), dimension, train_set_size, false, false);
    readXvec<float>(base_input, base_set.data(), dimension, base_set_size, false, false);
    readXvec<uint32_t>(gt_input, gt_set.data(), ngt, query_set_size, false, false);
    readXvec<float> (query_input, query_set.data(), dimension, query_set_size, false, false);

    const size_t centroid_num[2] = {1000, 2000};
    const size_t centroid_keep_space[2] = {100, 200};
    const size_t centroid_num1[4] = {10, 100, 20, 100};
    const size_t centroid_num2[4] = {100, 10, 100, 20};
    const size_t centroid_keep_space1[4] = {2, 50, 4, 50};
    const size_t centroid_keep_space2[4] = {50, 2, 50, 4};


    time_recorder trecorder;
    record_output << "The searching time for " << base_set_size << " vectors" << std::endl;

    // The 1 layer structure
    for (size_t i = 0; i < 2; i++){
        faiss::Clustering clus (dimension, centroid_num[i]);
        clus.verbose = true;
        faiss::IndexFlat index(dimension);
        clus.train(train_set_size, train_set.data(), index);

        trecorder.reset();
        std::vector<idx_t> base_assigned_ids(base_set_size); std::vector<float> base_assigned_dists(base_set_size);
        index.search(base_set_size, base_set.data(), centroid_keep_space[i], base_assigned_dists.data(), base_assigned_ids.data());
        std::string message = "Finished searching:  ncentroids: " + std::to_string(centroid_num[i]) + " search_space: " + std::to_string(centroid_keep_space[i]);
        trecorder.record_time_usage(record_output, message);
    }

    for (size_t i = 0; i < 4; i++){
        faiss::Clustering clus1(dimension, centroid_num1[i]);
        clus1.verbose = true;
        faiss::Index index1(dimension);
        clus1.train(train_set_size, )
    }


    

