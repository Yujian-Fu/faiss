#include <cstdio>
#include <iostream>
#include <string>


const bool use_reranking = false;
const bool use_norm_quantization = false;
const bool use_dynamic_reranking = false;
const bool use_OPQ = false;
const bool use_parallel_indexing = false;
const bool use_train_selector = false;

const bool use_HNSW_group = false;
const bool use_HNSW_VQ = false;
const bool use_all_HNSW = false;

const bool is_recording = true;
const bool saving_index = true;

const std::string dataset = "SIFT1M";
//const std::string dataset = "GIST1M";
const std::string path_folder = "/home/y/yujianfu/ivf-hnsw/";

//For train PQ
const size_t M_PQ = 8;
const size_t M_norm_PQ = 1;
const size_t nbits = 8; //Or 16
const size_t dimension = 128;

const size_t train_size = 100000; //This is the size of train set
const size_t OPQ_train_size = 1000000;
const size_t group_HNSW_thres = 100;
const size_t selector_train_size = 100000;
const size_t selector_group_size = 2000;

const size_t PQ_train_size = 10000;
const size_t nb = 1000000;
const size_t nbatches = 10; //100
const uint32_t batch_size =  nb / nbatches;


const size_t ngt = 100;
const size_t nq = 10000;
const size_t num_search_paras = 10;
const size_t num_recall = 1;

const size_t result_k[num_recall] = {10};
const size_t max_vectors[num_search_paras] = {nb/10, nb/10, nb/10, nb/10, nb/10, nb/10, nb/10, nb/10, nb/10, nb/10};
const size_t reranking_space[num_recall] = {150};
const std::string search_mode = "non parallel";

std::string conf_combination(const uint32_t * ncentroids, const std::string * index_type, 
const size_t layers, const size_t * M_PQ_layer, const size_t * nbits_PQ_layer){
    std::string result = "";
    for (size_t i = 0; i < layers; i++){
        result += "_"; result += index_type[i] == "PQ"? std::to_string(M_PQ_layer[i]) + "_" + std::to_string(nbits_PQ_layer[i]) : std::to_string(ncentroids[i]);
    }
    result += "_" + std::to_string(M_PQ);
    return result;
}

std::string index_combination(const std::string * index_type, const size_t layers, const size_t * LQ_type){
    std::string result = "";
    size_t n_lq = 0;
    for (size_t i = 0; i < layers; i++){
        result += "_"; result += index_type[i]; 
        if (index_type[i] == "VQ" && use_HNSW_VQ) result += "_HNSW";
        if (index_type[i] == "LQ")                {result += std::to_string(LQ_type[n_lq]); n_lq ++;}}
    
    return result;
}

//File paths
const std::string path_learn =     path_folder + "data/" + dataset + "/" + dataset +"_learn.fvecs";
const std::string path_base =      path_folder + "data/" + dataset + "/" + dataset +"_base.fvecs";
const std::string path_gt =        path_folder + "data/" + dataset + "/" + dataset +"_groundtruth.ivecs";
const std::string path_query =     path_folder + "data/" + dataset + "/" + dataset +"_query.fvecs";
