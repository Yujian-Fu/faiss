#include "./parameters_billions.h"

/* Parameter setting: */
//Exp parameters
//For index initialization
const size_t VQ_layers = 1;
const size_t PQ_layers = 1;
const size_t LQ_layers = 0;
const size_t layers = VQ_layers + PQ_layers + LQ_layers;
const size_t LQ_type[LQ_layers] = {};

const std::string index_type[layers] = {"PQ"};
const uint32_t ncentroids[layers-PQ_layers] = {};


//For building index
const size_t M_HNSW[VQ_layers] = {};
const size_t efConstruction [VQ_layers] = {};
const size_t efSearch[VQ_layers] = {};

const size_t M_PQ_layer[PQ_layers] = {2};
const size_t nbits_PQ_layer[PQ_layers] = {14};
const size_t num_train[layers] = {5000000};

//For searching
const size_t keep_space[layers * num_search_paras] = {20, 40, 60, 80, 100, 120, 140, 160, 180, 200};
