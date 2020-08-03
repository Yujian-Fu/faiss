#include <cstdio>
#include <iostream>
#include <string>

/* Parameter setting: */
//Exp parameters
//For index initialization
const size_t layers = 2;
const size_t VQ_layers = 1;
const size_t PQ_layers = 0;
const std::string index_type[layers] = {"VQ", "LQ"};
const uint32_t ncentroids[layers] = {200, 20};

const bool use_reranking = false;
const bool use_HNSW_VQ = false;
const bool use_norm_quantization = false;
const bool use_dynamic_reranking = false;
const bool use_OPQ = false;
const bool use_parallel_indexing = false;
const bool use_hash = PQ_layers > 0 ? true: false;

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
const size_t selector_group_size = 2000;

const size_t PQ_train_size = 10000;

const size_t num_train[layers] = {100000, 100000};
const size_t nb = 1000000;
const uint32_t batch_size = 10000;
const size_t nbatches = nb / batch_size; //100

//For searching
const size_t ngt = 100;
const size_t nq = 1000;
const size_t num_search_paras = 10;
const size_t num_recall = 3;

const size_t result_k[num_recall] = {1, 10, 100};
const size_t max_vectors[num_search_paras] = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000};
const size_t keep_space[layers * num_search_paras] = {10, 5, 15, 5, 20, 5, 25, 5, 30, 5, 35, 5, 40, 5, 45, 5, 50, 5, 50, 5};
const size_t reranking_space[num_recall] = {10, 20, 150};
const std::string search_mode = "parallel";

bool is_recording = true;

std::string conf_combination(){
    std::string result = "";
    for (size_t i = 0; i < layers; i++){
        result += "_"; result += index_type[i] == "PQ"? std::to_string(M_PQ_layer[i]) + "_" + std::to_string(nbits_PQ_layer[i]) : std::to_string(ncentroids[i]);}
    return result;
}

std::string index_combination(){
    std::string result = "";
    for (size_t i = 0; i < layers; i++){
        result += "_"; result += index_type[i]; if (index_type[i] == "VQ" && use_HNSW_VQ) result += "_HNSW";}
    return result;
}

// Folder path
std::string ncentroid_conf = conf_combination();
std::string model = "models" + index_combination();
const std::string dataset = "SIFT1M";

const std::string folder_model = "/home/y/yujianfu/ivf-hnsw/" + model;

//File paths
const std::string path_learn =     "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_learn.fvecs";
const std::string path_base =      "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_base.fvecs";
const std::string path_gt =        "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_groundtruth.ivecs";
const std::string path_query =     "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_query.fvecs";

const std::string path_record =    "/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset + "/recording" + ncentroid_conf + ".txt";
const std::string path_quantizers = "/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset + "/quantizer" + ncentroid_conf + ".qt";
const std::string path_ids =      "/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset + "/base_idxs" + ncentroid_conf + ".ivecs";
const std::string path_pq =        "/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset + "/PQ" + std::to_string(M_PQ) + ncentroid_conf + "_" + std::to_string(M_PQ) + "_" + std::to_string(nbits) + ".pq";
const std::string path_pq_norm =   "/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset + "/PQ_NORM" + std::to_string(M_PQ) + ncentroid_conf + ".pq";
const std::string path_index =     "/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset + "/PQ" + std::to_string(M_PQ) + ncentroid_conf + "_" + std::to_string(M_PQ) + "_" + std::to_string(nbits) + ".index";


/**
 **This is the centroids for assigining origin train vectors  size: n_group * dimension
 **/
const std::string path_groups = "/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset + "/selector_centroids_" + std::to_string(selector_group_size) + ".fvecs";
//This is for recording the labels based on the generated centroids

/**
 ** This is the labels for all assigned vectors, n_group * group_size 
 **/
const std::string path_labels = "/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset + "/selector_ids_" + std::to_string(selector_group_size);
