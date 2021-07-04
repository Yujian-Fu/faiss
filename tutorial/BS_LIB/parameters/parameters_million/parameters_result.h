#include "./parameters_VQ_VQ_VQ.h"

// Folder path
std::string ncentroid_conf = conf_combination(ncentroids, index_type, layers, M_PQ_layer, nbits_PQ_layer);
std::string model = "models" + index_combination(index_type, layers, LQ_type);

const std::string path_OPQ =        path_folder + model + "/" + dataset + "/opq_matrix_" + ncentroid_conf + ".opq";
const std::string path_speed_record = path_folder + model + "/" + dataset + "/recording" + ncentroid_conf + "_qps.txt";
const std::string path_record =     path_folder + model + "/" + dataset + "/recording" + ncentroid_conf + ".txt";
const std::string path_quantizers = path_folder + model + "/" + dataset + "/quantizer" + ncentroid_conf + ".qt";
const std::string path_alphas = path_folder + model + "/" + dataset + "/alpha" + ncentroid_conf + ".alpha";
const std::string path_alphas_raw = path_folder + model + "/" + dataset + "/alpha" + ncentroid_conf + ".alpha_raw";
const std::string path_base_alpha_norm = path_folder + model + "/" + dataset + "/base_alpha_norm" + ncentroid_conf + ".norm";
const std::string path_group_HNSW = path_folder + model + "/" + dataset + "/group_HNSWs" + ncentroid_conf + ".gHNSW";
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
