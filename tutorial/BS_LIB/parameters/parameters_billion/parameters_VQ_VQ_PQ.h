#include "./parameters_billions.h"

/* Parameter setting: */
//Exp parameters
//For index initialization
const size_t VQ_layers = 2;
const size_t PQ_layers = 1;
const size_t LQ_layers = 0;
const size_t layers = VQ_layers + PQ_layers + LQ_layers;
const size_t LQ_type[LQ_layers] = {};

const std::string index_type[layers] = {"VQ", "VQ", "PQ"};
const uint32_t ncentroids[layers-PQ_layers] = {100, 100};


//For building index
bool use_HNSW_VQ[VQ_layers] = {false, false};
const size_t M_HNSW[VQ_layers] = {};
const size_t efConstruction [VQ_layers] = {};

const size_t M_HNSW_all[VQ_layers] = {};
const size_t efConstruction_all [VQ_layers] = {};

const size_t M_PQ_layer[PQ_layers] = {2};
const size_t nbits_PQ_layer[PQ_layers] = {8};
const size_t num_train[layers] = {300000, 100000000, 100000000};

//For searching
const size_t keep_space[layers * num_search_paras] = {10, 10, 20, 10, 10, 20, 20, 30, 30, 20, 40, 10, 50, 10, 50, 20, 60, 10, 60, 20};