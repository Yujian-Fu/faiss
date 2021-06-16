#include "./parameters_millions.h"

/* Parameter setting: */
//Exp parameters
//For index initialization
const size_t layers = 2;
const size_t VQ_layers = 2;
const size_t PQ_layers = 0;
const std::string index_type[layers] = {"VQ", "VQ"};
const uint32_t ncentroids[layers] = {200, 10};


//For building index
const size_t M_HNSW[VQ_layers] = {};
const size_t efConstruction [VQ_layers] = {};
const size_t efSearch[VQ_layers] = {};

const size_t M_PQ_layer[PQ_layers] = {};
const size_t nbits_PQ_layer[PQ_layers] = {};
const size_t num_train[layers] = {100000, 100000};

//For searching
const size_t keep_space[layers * num_search_paras] = {10, 2, 10, 4, 10, 6, 20, 2, 20, 4, 30, 6, 40, 2, 40, 4, 40, 6, 
    50, 2};


// Folder path
std::string ncentroid_conf = conf_combination(ncentroids, index_type, layers, M_PQ_layer, nbits_PQ_layer);
std::string model = "models" + index_combination(index_type, layers);


const std::string path_OPQ =        path_folder + model + "/" + dataset + "/opq_matrix_" + std::to_string(M_PQ) + ".opq";
const std::string path_speed_record = path_folder + model + "/" + dataset + "/recording" + ncentroid_conf + "_qps.txt";
const std::string path_record =     path_folder + model + "/" + dataset + "/recording" + ncentroid_conf + ".txt";
const std::string path_quantizers = path_folder + model + "/" + dataset + "/quantizer" + ncentroid_conf + ".qt";
const std::string path_ids =        path_folder + model + "/" + dataset + "/base_idxs" + ncentroid_conf + ".ivecs";
const std::string path_centroid_norm = path_folder + model + "/" + dataset + "/centroid_norm" + ncentroid_conf + ".norm";
const std::string path_pq =  path_folder + model + "/" + dataset + "/PQ" + std::to_string(M_PQ) + ncentroid_conf  + "_" + std::to_string(nbits) + ".pq";
const std::string path_pq_norm =    path_folder + model + "/" + dataset + "/PQ_NORM" + std::to_string(M_PQ) + ncentroid_conf  + ".norm";
const std::string path_base_norm =      path_folder + model + "/" + dataset + "/base_norm" + std::to_string(M_PQ) + ncentroid_conf + "_" + std::to_string(nbits) + ".norm";
const std::string path_index =      path_folder + model + "/" + dataset + "/PQ" + std::to_string(M_PQ) + ncentroid_conf + "_" + std::to_string(nbits) + ".index";

/**
 **This is the centroids for assigining origin train vectors  size: n_group * dimension
 **/
const std::string path_groups = path_folder + model + "/" + dataset + "/selector_centroids_" + std::to_string(selector_group_size) + ".fvecs";
//This is for recording the labels based on the generated centroids

/**
 ** This is the labels for all assigned vectors, n_group * group_size 
 **/
const std::string path_labels = path_folder + model + "/" + dataset + "/selector_ids_" + std::to_string(selector_group_size);