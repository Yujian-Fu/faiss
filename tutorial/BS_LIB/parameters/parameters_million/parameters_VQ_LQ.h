#include "./parameters_millions.h"

/* Parameter setting: */
//Exp parameters
//For index initialization

const size_t VQ_layers = 1;
const size_t PQ_layers = 1;
const size_t LQ_layers = 0;
const size_t layers = VQ_layers + PQ_layers + LQ_layers;
const size_t LQ_type[LQ_layers] = {};

const std::string index_type[layers] = {"VQ", "PQ"};
const uint32_t ncentroids[layers-PQ_layers] = {40};


//For building index
bool use_HNSW_VQ[VQ_layers] = {false};
const size_t M_HNSW[VQ_layers] = {4};
//Set efConstruction and efSearch as the same
const size_t efConstruction [VQ_layers] = {10};

const size_t M_HNSW_all[VQ_layers] = {};
const size_t efConstruction_all [VQ_layers] = {};

const size_t M_PQ_layer[PQ_layers] = {2};
const size_t nbits_PQ_layer[PQ_layers] = {8};
const size_t num_train[layers] = {100000, 100000};

//For searching
const size_t keep_space[layers * num_search_paras] = {20, 1000};

