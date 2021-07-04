#include "./parameters_millions.h"

/* Parameter setting: */
//Exp parameters
//For index initialization

const size_t VQ_layers = 3;
const size_t PQ_layers = 0;
const size_t LQ_layers = 0;
const size_t layers = VQ_layers + PQ_layers + LQ_layers;
const size_t LQ_type[LQ_layers] = {};

const std::string index_type[layers] = {"VQ", "VQ", "VQ"};
const uint32_t ncentroids[layers-PQ_layers] = {100, 10, 10};


//For building index
const size_t M_HNSW[VQ_layers] = {};
const size_t efConstruction [VQ_layers] = {};
const size_t efSearch[VQ_layers] = {};

const size_t M_PQ_layer[PQ_layers] = {};
const size_t nbits_PQ_layer[PQ_layers] = {};
const size_t num_train[layers] = {100000, 100000, 100000};

//For searching
const size_t keep_space[layers * num_search_paras] = {10, 5, 2, 20, 5, 2, 30, 5, 2, 40, 5, 2, 50, 5, 2, 20, 60, 5, 2 , 40, 5, 5};