#include <cstdio>
#include <iostream>
#include <string>

/* Parameter setting: */
//Exp parameters
//For index initialization
const size_t layers = 2;
const size_t VQ_layers = 2;
const size_t PQ_layers = 0;
const std::string index_type[layers] = {"VQ", "VQ"};
const uint32_t ncentroids[layers] = {200, 10};

const bool use_reranking = false;
const bool use_HNSW_VQ = false;
const bool use_norm_quantization = false;
const bool use_dynamic_reranking = false;
const bool use_OPQ = false;
const bool use_parallel_indexing = false;
const bool use_hash = PQ_layers > 0 ? true: false;
const size_t reranking_space = 20;

//For train PQ
const size_t M_PQ = 16;
const size_t M_norm_PQ = 1;
const size_t nbits = 8; //Or 16
const size_t dimension = 128;
//For assigning ID

//For building index
const size_t train_size = 100000; //This is the size of train set
const size_t M_HNSW[VQ_layers] = {};
const size_t efConstruction [VQ_layers] = {};
const size_t efSearch[VQ_layers] = {};

const size_t M_PQ_layer[PQ_layers] = {};
const size_t nbits_PQ_layer[PQ_layers] = {};

const size_t selector_train_size = 100000;
const size_t selector_group_size = 1000;

const size_t PQ_train_size = 10000;

const size_t num_train[layers] = {10000, 100000};
const size_t nb = 1000000;
const uint32_t batch_size = 10000;
const size_t nbatches = nb / batch_size; //100

//For searching
const size_t ngt = 100;
const size_t nq = 1000;
const size_t result_k = 10;
const size_t max_vectors = 3000;
size_t keep_space[layers] = {100};

bool is_recording = true;

std::string conf_combination(){
    std::string result = "";
    for (size_t i = 0; i < layers; i++){result += "_"; result += index_type[i] == "PQ"? std::to_string(M_PQ_layer[i]) + "_" + std::to_string(nbits_PQ_layer[i]) : std::to_string(ncentroids[i]);}
    return result;
}

std::string index_combination(){
    std::string result = "";
    for (size_t i = 0; i < layers; i++){result += "_"; result += index_type[i]; if (index_type[i] == "VQ" && use_HNSW_VQ) result += "_HNSW";}
    return result;
}

// Folder path
std::string ncentroid_conf = conf_combination();
std::string model = "models" + index_combination();
const std::string dataset = "SIFT1M";

const std::string folder_model = "/home/y/yujianfu/ivf-hnsw/" + model;

//File paths
const std::string path_learn =     "/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_learn.fvecs";
const std::string path_base =      "/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_base.fvecs";
const std::string path_gt =        "/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_groundtruth.ivecs";
const std::string path_query =     "/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_query.fvecs";

const std::string path_record =    "/home/y/yujianfu/ivf-hnsw/" + model + "/SIFT1M/recording" + ncentroid_conf + ".txt";
const std::string path_pq =        "/home/y/yujianfu/ivf-hnsw/" + model + "/SIFT1M/PQ" + std::to_string(M_PQ) + ncentroid_conf + ".pq";
const std::string path_pq_norm =   "/home/y/yujianfu/ivf-hnsw/" + model + "/SIFT1M/PQ_NORM" + std::to_string(M_PQ) + ncentroid_conf + ".pq";
const std::string path_ids =      "/home/y/yujianfu/ivf-hnsw/" + model + "/SIFT1M/base_idxs" + ncentroid_conf + ".ivecs";
const std::string path_index =     "/home/y/yujianfu/ivf-hnsw/" + model + "/SIFT1M/PQ" + std::to_string(M_PQ) + ncentroid_conf + ".index";
const std::string path_quantizers = "/home/y/yujianfu/ivf-hnsw/" + model + "/SIFT1M/quantizer" + ncentroid_conf + ".qt";

/**
 **This is the centroids for assigining origin train vectors  size: n_group * dimension
 **/
const std::string path_groups = "/home/y/yujianfu/ivf-hnsw/" + model + "/SIFT1M/selector_centroids_" + std::to_string(selector_group_size) + ".fvecs";
//This is for recording the labels based on the generated centroids

/**
 ** This is the labels for all assigned vectors, n_group * group_size 
 **/
const std::string path_labels = "/home/y/yujianfu/ivf-hnsw/" + model + "/SIFT1M/selector_ids_" + std::to_string(train_size);
