#include <cstdio>
#include <iostream>
#include <string>

/* Parameter setting: */
//Exp parameters
//For index initialization
const size_t layers = 1;    
const size_t VQ_layers = 1; 
const size_t PQ_layers = 1;
const std::string index_type[layers] = {"PQ"};
const uint32_t ncentroids[layers] = {64};

const bool use_reranking = false;
const bool use_HNSW_VQ = false;
const bool use_norm_quantization = false;
const bool use_dynamic_reranking = false;
const bool use_OPQ = false;
const bool use_parallel_indexing = true;
const bool use_train_selector = false;

//For train PQ
const size_t M_PQ = 16;
const size_t M_norm_PQ = 1;
const size_t nbits = 8; //Or 16
const size_t dimension = 128;
//For assigning ID

//For building index
const size_t train_size = 100000; //This is the size of train set
const size_t M_HNSW[VQ_layers] = {100};
const size_t efConstruction [VQ_layers] = {200};
const size_t efSearch[VQ_layers] = {50};

const size_t M_PQ_layer[PQ_layers] = {2};
const size_t nbits_PQ_layer[PQ_layers] = {4};

const size_t OPQ_train_size = 10000;
const size_t selector_train_size = 100000;
const size_t selector_group_size = 2000;

const size_t PQ_train_size = 10000;

const size_t num_train[layers] = {100000};
size_t nb = 1000000;
const uint32_t batch_size = 10000;
const size_t nbatches = nb / batch_size; //100

//For searching
const size_t ngt = 100;
const size_t nq = 1000;
const size_t num_search_paras = 9;
const size_t num_recall = 1;

const size_t result_k[num_recall] = {1};
const size_t max_vectors[num_search_paras] = {1000, 2000, 4000, 6000, 8000, 10000, 12000, 13000, 15000};
const size_t keep_space[layers * num_search_paras] = {10, 15, 20, 25, 30, 35, 40, 45, 50};
const size_t reranking_space[num_recall] = {10};
const std::string search_mode = "non parallel";

bool is_recording = true;

std::string conf_combination(){
    std::string result = "";
    for (size_t i = 0; i < layers; i++){result += "_"; result += index_type[i] == "PQ"? std::to_string(M_PQ_layer[i]) + "_" + std::to_string(nbits_PQ_layer[i]) : std::to_string(ncentroids[i]);}
    result += use_OPQ ? "_OPQ" : "";
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
const std::string base_path = "/home/yujian/Desktop/extra/Similarity Search/similarity_search_datasets/";
const std::string data_path = base_path + "data/";

const std::string dataset = "sift1M";


const std::string folder_model = base_path + model;

//File paths
const std::string path_learn =     data_path + dataset + "/" + dataset +"_learn.fvecs";
const std::string path_base =      data_path + dataset + "/" + dataset +"_base.fvecs";
const std::string path_gt =        data_path + dataset + "/" + dataset +"_groundtruth.ivecs";
const std::string path_query =     data_path + dataset + "/" + dataset +"_query.fvecs";

const std::string path_OPQ =            base_path + model + "/" + dataset + "/opq_matrix_" + std::to_string(M_PQ) + ".opq";
const std::string path_speed_record =   base_path + model + "/" + dataset + "/recording" + ncentroid_conf + "_qps.txt";
const std::string path_record =         base_path + model + "/" + dataset + "/recording" + ncentroid_conf + ".txt";
const std::string path_quantizers =     base_path + model + "/" + dataset + "/quantizer" + ncentroid_conf + ".qt";
const std::string path_ids =            base_path + model + "/" + dataset + "/base_idxs" + ncentroid_conf + ".ivecs";
const std::string path_pq =             base_path + model + "/" + dataset + "/PQ" + std::to_string(M_PQ) + ncentroid_conf + "_" + std::to_string(M_PQ) + "_" + std::to_string(nbits) + ".pq";
const std::string path_pq_norm =        base_path + model + "/" + dataset + "/PQ_NORM" + std::to_string(M_PQ) + ncentroid_conf + ".pq";
const std::string path_index =          base_path + model + "/" + dataset + "/PQ" + std::to_string(M_PQ) + ncentroid_conf + "_" + std::to_string(M_PQ) + "_" + std::to_string(nbits) + ".index";


/**
 **This is the centroids for assigining origin train vectors  size: n_group * dimension
 **/
std::string use_OPQ_label = use_OPQ ? "OPQ" : "";
const std::string path_groups = base_path + model + "/" + dataset + "/selector_centroids_" + std::to_string(selector_group_size) + "_" + use_OPQ_label + ".fvecs";
//This is for recording the labels based on the generated centroids

/**
 ** This is the labels for all assigned vectors, n_group * group_size 
 **/
const std::string path_labels = base_path + model + "/" + dataset + "/selector_ids_" + std::to_string(train_size) + "_" + use_OPQ_label;
