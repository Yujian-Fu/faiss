#include <cstdio>
#include <iostream>
#include <string>
#include "../../parameter_general.h"

/* Parameter setting: */
//Exp parameters
//For index initialization
const size_t layers = 2;
const size_t VQ_layers = 2;
const size_t PQ_layers = 0;
std::string index_type[layers] = {"VQ", "LQ"};
uint32_t ncentroids[layers] = {};

//For assigning ID

//For building index
const size_t M_HNSW[VQ_layers] = {100};
const size_t efConstruction [VQ_layers] = {200};
const size_t efSearch[VQ_layers] = {50};

const size_t M_PQ_layer[PQ_layers] = {};
const size_t nbits_PQ_layer[PQ_layers] = {};


const size_t num_train[layers] = {train_size, train_size};


std::string conf_combination(){
    std::string result = "";
    for (size_t i = 0; i < layers; i++){result += "_"; result += index_type[i] == "PQ"? std::to_string(M_PQ_layer[i]) + "_" + std::to_string(nbits_PQ_layer[i]) : std::to_string(ncentroids[i]);}
    result += use_OPQ ? "_OPQ" : "";
    result += use_HNSW_VQ ? "_HNSW" : "";
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
const std::string data_path = base_path + "data/";
const std::string folder_model = base_path + model;

//File paths
const std::string path_learn =     data_path + dataset + "/" + dataset +"_learn.fvecs";
const std::string path_base =      data_path + dataset + "/" + dataset +"_base.fvecs";
const std::string path_gt =        data_path + dataset + "/" + dataset +"_groundtruth.ivecs";
const std::string path_query =     data_path + dataset + "/" + dataset +"_query.fvecs";

std::string path_record =         base_path + model + "/" + dataset + "/";
const std::string path_OPQ =            base_path + model + "/" + dataset + "/opq_matrix_" + std::to_string(M_PQ) + ".opq";
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
